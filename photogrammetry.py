'''
MIT License

Copyright (c) 2026 Aleksander Berg-Jones

##  Permission is hereby granted, free of charge, to any person obtaining a
##  copy of this software and associated documentation files (the "Software"),
##  to deal in the Software without restriction, including without limitation
##  the rights to use, copy, modify, merge, publish, distribute, sublicense,
##  and/or sell copies of the Software, and to permit persons to whom the
##  Software is furnished to do so, subject to the following conditions:
##
##  The above copyright notice and this permission notice shall be included in
##  all copies or substantial portions of the Software.
##
##  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
##  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
##  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
##  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
##  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
##  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
##  DEALINGS IN THE SOFTWARE.

''' 


"""
Blender 5.x photogrammetry from a turntable-style orbit video

  Stage 1  Compositor (GPU): Movie Clip -> Scale -> RGB to BW -> 16-bit PNG per frame
  Stage 2  CPU: sharpness-based frame selection, SIFT + tracks + incremental SfM + bundle adjustment
  Stage 3  Compositor (GPU, unrolled): plane-sweep stereo; state carried in RGBA of an image
  Stage 4  CPU: multi-view depth consistency fusion -> point cloud
  Stage 5  CPU: Poisson (Open3D, optional) or marching cubes -> Blender mesh object

Run from the Scripting workspace
Edit CFG below first.
"""
import bpy, sys, os, time, glob, shutil, subprocess
import numpy as np
from pathlib import Path
import scipy
import skimage
import cv2
from datetime import datetime
import pickle, json, hashlib

CFG = dict(
	# video=r"E:\projects_3d\ABJ_Shader_Debugger_for_Blender\scenes\compositing_files\photogrammetry\skull_04.MOV",
	# work=r"E:\projects_3d\ABJ_Shader_Debugger_for_Blender\scenes\compositing_files\photogrammetry\skull_01_work",

	video=r"E:\projects_3d\ABJ_Shader_Debugger_for_Blender\scenes\compositing_files\photogrammetry\skull_06.MOV",
	work=r"E:\projects_3d\ABJ_Shader_Debugger_for_Blender\scenes\compositing_files\photogrammetry\skull_06_work",


	redo=[],                # any of "frames", "sfm", "depth" (later stages cascade)
	rotate=0,
	max_frames=220,
	candidates_per_frame=2,
	proc_w=1920,
	dense_long=1280,
	f_guess=0.72,
	match_window=8,
	ratio=0.80,
	n_orbits=3,
	# ---- dense (changing these invalidates the depth cache) ----
	dense_engine="cpu",     # "cpu" (cv2, parallel, trusted) or "compositor" (GPU node graph)
	# dense_engine="compositor",     # "cpu" (cv2, parallel, trusted) or "compositor" (GPU node graph)
	cpu_workers=8,
	cost_best_k=2,          # average the best k of the neighbour costs (handles occlusion); cpu engine
	compare_compositor=False,  # True: also run the GPU graph on one ref and log how well it agrees with CPU
	n_planes=128,
	planes_per_pass=8,
	n_neighbors=4,
	nb_min_deg=3.0,
	nb_max_deg=20.0,
	cost_window=9,
	ref_stride=2, # ####################
	# ref_stride=1,
	obj_radius_frac=0.45,   # (object radius / camera distance) * 1.3 -- NOT 0.65
	# ---- fusion / meshing (instant, use cached depth maps) ----
	crop_radius_frac=0.45,
	cost_thr="auto",        # number, or "auto"
	tsdf_cost_pct=50,       # "auto" threshold = this percentile of each depth map's cost (capped at 0.3)
	min_consistent=1,
	# min_consistent=2,
	cons_tol_planes=3.0,
	fuse_stride=2,
	mesh_method="tsdf",     # "tsdf" or "points_mc" (old behaviour)
	tsdf_res=220,
	tsdf_mu_voxels=3.0,
	# tsdf_min_views=2,
	tsdf_min_views=2,       # a voxel needs this many views to become surface
	# tsdf_min_views=4,       # a voxel needs this many views to become surface
	# tsdf_smooth=0.8,
	tsdf_smooth=0.8,        # 0 disables
	output_radius_m=0.1,
	mc_res=220,
	mc_density_frac=0.15,
	mc_close=2,

	cost_pct=60,            # cost_thr="auto": percentile of in-crop costs used as threshold (capped at 0.3)
	# cost_pct=45,            # cost_thr="auto": percentile of in-crop costs used as threshold (capped at 0.3)
	min_region_px=150,      # drop isolated valid-depth regions smaller than this (kills confetti)
	# min_region_px=400,      # drop isolated valid-depth regions smaller than this (kills confetti)

	mesh_trim_radius=1.0,   # unit = crop radius. Remove mesh beyond this radius, e.g. 0.8 (1.0 = off)
	min_component_frac=0.05,  # drop disconnected mesh pieces smaller than this fraction of the largest

)

def log(*a):
	print("[photo]", *a, flush=True)


# def ensure_packages():
#     import importlib
#     need = []
#     for mod, pipname in (("cv2", "opencv-python"), ("scipy", "scipy"), ("skimage", "scikit-image")):
#         importlib.import_module(mod)



# ----------------------------------------------------------------------------
# Compositor helpers (Blender 5.x first, 4.x fallbacks)
# ----------------------------------------------------------------------------
def new_comp_tree(scene, name):
	if hasattr(scene, "compositing_node_group"):
		nt = bpy.data.node_groups.new(name, "CompositorNodeTree")
		nt.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
		scene.compositing_node_group = nt
	else:
		scene.use_nodes = True
		nt = scene.node_tree
		nt.nodes.clear()
	return nt


def free_comp_tree(scene, nt):
	if hasattr(scene, "compositing_node_group"):
		scene.compositing_node_group = None
		bpy.data.node_groups.remove(nt)


def mk(nt, *types, **props):
	err = None
	n = None
	for t in types:
		try:
			n = nt.nodes.new(t)
			break
		except RuntimeError as e:
			err = e
	if n is None:
		raise RuntimeError(f"No node type available from {types}: {err}")
	for k, v in props.items():
		try:
			setattr(n, k, v)
		except Exception as e:
			log("warn: cannot set", k, "on", n.bl_idname, e)
	return n


def mth(nt, op, a=None, b=None, c=None):
	n = mk(nt, "ShaderNodeMath", "CompositorNodeMath", operation=op)
	for i, v in enumerate((a, b, c)):
		if v is None:
			continue
		if isinstance(v, (int, float)):
			n.inputs[i].default_value = float(v)
		else:
			nt.links.new(v, n.inputs[i])
	return n


def group_output(nt):
	try:
		return mk(nt, "NodeGroupOutput")
	except RuntimeError:
		return mk(nt, "CompositorNodeComposite")


def file_output(nt, directory, fmt, depth, mode, name="out"):
	os.makedirs(directory, exist_ok=True)
	n = mk(nt, "CompositorNodeOutputFile")
	if hasattr(n, "directory"):
		n.directory = directory
		n.file_name = name
		items = n.file_output_items
		if len(items) == 0:
			try:
				items.new("RGBA", "Image")
			except TypeError:
				items.new(socket_type="RGBA", name="Image")
		if hasattr(n, "use_file_extension"):
			n.use_file_extension = True
	else:
		n.base_path = directory
		n.file_slots[0].path = name
	f = n.format
	# Blender 5.x: the node defaults to multilayer EXR, which hides PNG/EXR single-image formats.
	if hasattr(f, "media_type"):
		f.media_type = "IMAGE"
	f.file_format = fmt
	try:
		f.color_depth = depth
	except Exception as e:
		log("warn: color_depth", depth, "not allowed for", fmt, e)
	try:
		f.color_mode = mode
	except Exception as e:
		log("warn: color_mode", mode, "not allowed for", fmt, e)
	if fmt == "OPEN_EXR":
		try:
			f.exr_codec = "ZIP"
		except Exception:
			pass
	return n


def setup_scene(scene, w, h):
	r = scene.render
	r.resolution_x, r.resolution_y, r.resolution_percentage = int(w), int(h), 100
	try:
		r.engine = "BLENDER_WORKBENCH"
	except Exception:
		pass
	r.use_compositing = True
	r.use_sequencer = False
	for obj in (r, scene):
		for attr, val in (("compositor_device", "GPU"), ("compositor_precision", "FULL")):
			if hasattr(obj, attr):
				try:
					setattr(obj, attr, val)
				except Exception as e:
					log("warn: cannot set", attr, e)
	if scene.camera is None:
		cam = bpy.data.objects.new("ps_cam", bpy.data.cameras.new("ps_cam"))
		scene.collection.objects.link(cam)
		scene.camera = cam


def clear_dir(d, pattern="*"):
	os.makedirs(d, exist_ok=True)
	for p in glob.glob(os.path.join(d, pattern)):
		try:
			os.remove(p)
		except OSError:
			pass


# ----------------------------------------------------------------------------
# Stage 1: frame extraction in the compositor
# ----------------------------------------------------------------------------
def extract_frames(cfg, out_dir):
	scene = bpy.context.scene
	clip = bpy.data.movieclips.load(cfg["video"])
	clip.frame_start = 1
	nfr = clip.frame_duration
	cw, ch = clip.size
	pw = cfg["proc_w"]
	ph = int(round(pw * ch / cw))
	log(f"clip {cw}x{ch}, {nfr} frames -> {pw}x{ph}")

	nt = new_comp_tree(scene, "FrameExtract")
	mc = mk(nt, "CompositorNodeMovieClip")
	mc.clip = clip
	last = mc.outputs[0]
	try:
		sc = mk(nt, "CompositorNodeScale", space="RENDER_SIZE", frame_method="STRETCH")
		nt.links.new(last, sc.inputs[0])
		last = sc.outputs[0]
		setup_scene(scene, pw, ph)
	except Exception as e:
		log("warn: Scale node unavailable, writing native size", e)
		setup_scene(scene, cw, ch)
	bw = mk(nt, "CompositorNodeRGBToBW", "ShaderNodeRGBToBW")
	nt.links.new(last, bw.inputs[0])
	clear_dir(out_dir, "*")
	fo = file_output(nt, out_dir, "PNG", "16", "BW", "f_")
	nt.links.new(bw.outputs[0], fo.inputs[0])
	go = group_output(nt)
	nt.links.new(bw.outputs[0], go.inputs[0])

	tmp = os.path.join(cfg["work"], "_tmp_render")
	clear_dir(tmp)
	scene.render.filepath = os.path.join(tmp, "r_")
	scene.render.image_settings.file_format = "PNG"
	scene.render.image_settings.color_mode = "BW"
	# scene.frame_start, scene.frame_end, scene.frame_step = 1, nfr, cfg["frame_step"]

	step = max(1, nfr // (cfg["candidates_per_frame"] * cfg["max_frames"]))
	# scene.frame_start = 1
	# scene.frame_end = nfr
	# scene.frame_step = step

	# log(f"frame_step auto = {step} -> ~{nfr // step} candidate frames, "
	#     f"~{cfg['max_frames']} will be kept (approx {720.0 / cfg['max_frames']:.1f} deg apart for 2 orbits)")


	step = max(1, nfr // (cfg["candidates_per_frame"] * cfg["max_frames"]))
	scene.frame_start, scene.frame_end, scene.frame_step = 1, nfr, step
	log(f"frame_step auto = {step} -> ~{nfr // step} candidate frames, "f"~{cfg['max_frames']} will be kept (approx {720.0 / cfg['max_frames']:.1f} deg apart for 2 orbits)")






	t0 = time.time()
	bpy.ops.render.render(animation=True)
	log(f"extraction render {time.time() - t0:.1f}s")
	free_comp_tree(scene, nt)
	shutil.rmtree(tmp, ignore_errors=True)
	files = sorted(glob.glob(os.path.join(out_dir, "*.png")))
	log("frames written:", len(files))
	return files


def prep_frames(raw_files, cfg, d_sfm, d_dense):
	import cv2
	for d in (d_sfm, d_dense):
		clear_dir(d)
	rot = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
		   270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(cfg["rotate"])

	def load(p):
		im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
		if im.ndim == 3:
			im = im[..., 0]
		g = im.astype(np.float32) / (65535.0 if im.dtype == np.uint16 else 255.0)
		g = np.clip(g, 0, 1) ** (1 / 2.2)  # linear -> display-ish gamma
		if rot is not None:
			g = cv2.rotate(g, rot)
		return g

	sharp = []
	for p in raw_files:
		g = load(p)
		s = 640.0 / g.shape[1]
		small = cv2.resize(g, (640, int(g.shape[0] * s)), interpolation=cv2.INTER_AREA)
		sharp.append(cv2.Laplacian(small, cv2.CV_32F).var())
	sharp = np.array(sharp)
	n = len(raw_files)
	if n > cfg["max_frames"]:
		keep = [b[np.argmax(sharp[b])] for b in np.array_split(np.arange(n), cfg["max_frames"])]
	else:
		keep = list(range(n))
	sfm_paths, dense_paths = [], []
	Wd = Hd = Ws = Hs = None
	for k, i in enumerate(keep):
		g = load(raw_files[i])
		Hs, Ws = g.shape
		p1 = os.path.join(d_sfm, f"{k:04d}.png")
		cv2.imwrite(p1, (g * 255 + 0.5).astype(np.uint8))
		s = cfg["dense_long"] / max(Hs, Ws)
		Wd, Hd = int(round(Ws * s)), int(round(Hs * s))
		d = cv2.resize(g, (Wd, Hd), interpolation=cv2.INTER_AREA)
		mu = cv2.GaussianBlur(d, (0, 0), 6)
		df = d - mu
		sd = np.sqrt(cv2.GaussianBlur(df * df, (0, 0), 6)) + 0.02
		nrm = np.clip(0.5 + 0.25 * df / sd, 0, 1)  # local normalisation, robust to exposure drift
		p2 = os.path.join(d_dense, f"{k:04d}.png")
		cv2.imwrite(p2, (nrm * 255 + 0.5).astype(np.uint8))
		sfm_paths.append(p1)
		dense_paths.append(p2)
	log(f"selected {len(keep)} frames, sfm {Ws}x{Hs}, dense {Wd}x{Hd}")
	return sfm_paths, dense_paths, (Ws, Hs), (Wd, Hd)


# ----------------------------------------------------------------------------
# Stage 2: SfM (OpenCV + SciPy)
# ----------------------------------------------------------------------------
def project_pts(rvec, tvec, P, f, cx, cy):
	theta = np.linalg.norm(rvec, axis=1)[:, None]
	v = np.where(theta > 1e-12, rvec / np.maximum(theta, 1e-12), 0.0)
	c, s = np.cos(theta), np.sin(theta)
	dot = (P * v).sum(1)[:, None]
	Pr = P * c + np.cross(v, P) * s + v * dot * (1 - c)
	Pc = Pr + tvec
	z = np.where(np.abs(Pc[:, 2]) < 1e-9, 1e-9, Pc[:, 2])
	return np.stack([f * Pc[:, 0] / z + cx, f * Pc[:, 1] / z + cy], 1)


def run_sfm(paths, cfg):
	import cv2
	from concurrent.futures import ThreadPoolExecutor
	from scipy.optimize import least_squares
	from scipy.sparse import lil_matrix

	def ranges_str(idx):
		idx = sorted(idx)
		if not idx:
			return "none"
		out, s, e = [], idx[0], idx[0]
		for v in idx[1:]:
			if v == e + 1:
				e = v
			else:
				out.append(f"{s}-{e}" if s != e else f"{s}")
				s = e = v
		out.append(f"{s}-{e}" if s != e else f"{s}")
		return ", ".join(out)

	n = len(paths)
	sift = cv2.SIFT_create(nfeatures=10000, contrastThreshold=0.02)
	pts, des = [], []
	for p in paths:
		im = cv2.imread(p, 0)
		kp, d = sift.detectAndCompute(im, None)
		pts.append(np.array([k.pt for k in kp], np.float64))
		des.append(d)
	H, W = im.shape
	nk = np.array([len(p) for p in pts])
	log(f"{n} frames {W}x{H}; keypoints/frame min {nk.min()} median {int(np.median(nk))} max {nk.max()}")
	if np.median(nk) < 300:
		log("WARNING: very few keypoints. Open the PNGs in <work>/sfm and check they are not black/white/blurred.")
	if min(len(d) if d is not None else 0 for d in des) < 10:
		raise RuntimeError("Some frames have no features at all; check <work>/sfm images.")
	cx, cy = W / 2.0, H / 2.0
	S = dict(f=cfg["f_guess"] * max(W, H))

	def Kmat():
		return np.array([[S["f"], 0, cx], [0, S["f"], cy], [0, 0, 1.0]])

	def match(pair):
		i, j = pair
		fl = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))
		m = fl.knnMatch(des[i], des[j], k=2)
		good = [a for a, b in (x for x in m if len(x) == 2) if a.distance < cfg["ratio"] * b.distance]
		if len(good) < 25:
			return pair, (None, len(good), 0)
		qi = np.array([g.queryIdx for g in good])
		ti = np.array([g.trainIdx for g in good])
		E, mask = cv2.findEssentialMat(pts[i][qi], pts[j][ti], Kmat(), cv2.RANSAC, 0.999, 1.5)
		if E is None or mask is None:
			return pair, (None, len(good), 0)
		mask = mask.ravel().astype(bool)
		if mask.sum() < 25:
			return pair, (None, len(good), int(mask.sum()))
		return pair, ((qi[mask], ti[mask]), len(good), int(mask.sum()))

	w = cfg["match_window"]
	pairs = {(i, j) for i in range(n) for j in range(i + 1, min(n, i + w + 1))}
	if n > 40:
		for i in range(8):
			for j in range(n - 8, n):
				if j > i + w:
					pairs.add((i, j))
		orbits = max(1, int(cfg.get("n_orbits", 1)))
		if orbits > 1:
			per = n / orbits
			for i in range(0, n, 6):
				for k in range(1, orbits):
					c = int(round(i + k * per))
					for j in range(c - 4, c + 5):
						if 0 <= j < n and j > i + w:
							pairs.add((i, j))
	log(f"{len(pairs)} candidate pairs to match")
	t0 = time.time()
	with ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 8) // 2)) as ex:
		raw_res = dict(ex.map(match, sorted(pairs)))
	results = {k: v[0] for k, v in raw_res.items() if v[0] is not None}
	log(f"matched {len(results)}/{len(pairs)} pairs in {time.time() - t0:.1f}s")
	adj = [raw_res[(i, i + 1)] for i in range(n - 1) if (i, i + 1) in raw_res]
	if adj:
		ratio_good = np.array([a[1] for a in adj])
		inl = np.array([a[2] for a in adj])
		log(f"adjacent pairs: ratio-test matches median {int(np.median(ratio_good))}, "
			f"verified inliers median {int(np.median(inl))}, pairs with >=25 inliers: {(inl >= 25).sum()}/{len(adj)}")
		weak = [i for i in range(n - 1) if (i, i + 1) not in raw_res or raw_res[(i, i + 1)][0] is None]
		log(f"adjacent pairs that failed verification (frame i -> i+1): {ranges_str(weak)}")

	# tracks via union-find; conflicting frames are removed from a group instead of discarding the whole group
	parent = {}

	def find(a):
		while parent.setdefault(a, a) != a:
			parent[a] = parent[parent[a]]
			a = parent[a]
		return a

	for (i, j), (qi, ti) in results.items():
		for a, b in zip(qi.tolist(), ti.tolist()):
			ra, rb = find((i, a)), find((j, b))
			if ra != rb:
				parent[ra] = rb
	groups = {}
	for node in list(parent):
		groups.setdefault(find(node), []).append(node)
	tracks = []
	split = 0
	for g in groups.values():
		if len(g) < 2:
			continue
		byf = {}
		for f_, k_ in g:
			byf.setdefault(f_, []).append(k_)
		clean = {f_: ks[0] for f_, ks in byf.items() if len(ks) == 1}
		if len(clean) != len(byf):
			split += 1
		if len(clean) >= 2:
			tracks.append(clean)
	tbf = [[] for _ in range(n)]
	for t, tr in enumerate(tracks):
		for g in tr:
			tbf[g].append(t)
	lens = np.array([len(t) for t in tracks]) if tracks else np.array([0])
	log(f"tracks: {len(tracks)} (mean length {lens.mean():.1f}, max {lens.max()}), "
		f"groups with conflicting frames repaired: {split}/{len(groups)}")

	def kp(g, t):
		return pts[g][tracks[t][g]]

	cams, X = {}, {}

	def center(g):
		R, t = cams[g]
		return -R.T @ t

	def tri_pair(tids, a, b):
		if len(tids) == 0:
			return [], np.zeros((0, 3))
		pa = np.array([kp(a, t) for t in tids])
		pb = np.array([kp(b, t) for t in tids])
		Ra, ta = cams[a]
		Rb, tb = cams[b]
		K = Kmat()
		Pa = K @ np.hstack([Ra, ta[:, None]])
		Pb = K @ np.hstack([Rb, tb[:, None]])
		Xh = cv2.triangulatePoints(Pa, Pb, pa.T, pb.T)
		Xw = (Xh[:3] / Xh[3]).T
		ca, cb = Xw @ Ra.T + ta, Xw @ Rb.T + tb
		qa, qb = ca @ K.T, cb @ K.T
		ea = np.linalg.norm(qa[:, :2] / qa[:, 2:3] - pa, axis=1)
		eb = np.linalg.norm(qb[:, :2] / qb[:, 2:3] - pb, axis=1)
		va, vb = Xw - center(a), Xw - center(b)
		cosang = (va * vb).sum(1) / (np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1) + 1e-12)
		ok = (ca[:, 2] > 0) & (cb[:, 2] > 0) & (ea < 3.0) & (eb < 3.0) & (cosang < np.cos(np.deg2rad(1.0)))
		return [t for t, o in zip(tids, ok) if o], Xw[ok]

	best = None
	maxcommon = 0
	for i in range(0, min(n - 1, 15)):
		for off in range(1, 9):
			j = i + off
			if j >= n:
				break
			common = [t for t in tbf[i] if j in tracks[t]]
			maxcommon = max(maxcommon, len(common))
			if len(common) < 40:
				continue
			pa = np.array([kp(i, t) for t in common])
			pb = np.array([kp(j, t) for t in common])
			E, m = cv2.findEssentialMat(pa, pb, Kmat(), cv2.RANSAC, 0.999, 1.5)
			if E is None:
				continue
			ninl, R, tt, _ = cv2.recoverPose(E[:3], pa, pb, Kmat())
			if ninl < 30:
				continue
			cams.clear()
			cams[i] = (np.eye(3), np.zeros(3))
			cams[j] = (R, tt.ravel())
			tids, Xn = tri_pair(common, i, j)
			if len(tids) < 30:
				continue
			va, vb = Xn - center(i), Xn - center(j)
			ang = np.degrees(np.arccos(np.clip((va * vb).sum(1) / (np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1)), -1, 1)))
			score = len(tids) * min(1.0, np.median(ang) / 3.0)
			if best is None or score > best[0]:
				best = (score, i, j, R.copy(), tt.ravel().copy(), tids, Xn)
	if best is None:
		raise RuntimeError(
			f"SfM init failed (max common tracks between any candidate pair: {maxcommon}). "
			"See the log lines above: if 'adjacent pairs' inliers are low, increase max_frames / lower frame_step, "
			"check <work>/sfm images for blur or wrong rotation, and make sure the scene has texture.")
	_, i0, j0, R0, t0v, tids, Xn = best
	cams.clear()
	cams[i0] = (np.eye(3), np.zeros(3))
	cams[j0] = (R0, t0v)
	for t, x in zip(tids, Xn):
		X[t] = x
	log(f"init pair {i0},{j0}: {len(X)} points")

	def ba(max_nfev=25, free_f=False, cull=6.0):
		fr = sorted(cams)
		fi = {g: i for i, g in enumerate(fr)}
		tl = [t for t in X if sum(1 for g in tracks[t] if g in cams) >= 2]
		ti = {t: i for i, t in enumerate(tl)}
		oc, op, oxy, ot = [], [], [], []
		for t in tl:
			for g, k in tracks[t].items():
				if g in cams:
					oc.append(fi[g]); op.append(ti[t]); oxy.append(pts[g][k]); ot.append((t, g))
		oc, op, oxy = np.array(oc), np.array(op), np.array(oxy)
		rv = np.array([cv2.Rodrigues(cams[g][0])[0].ravel() for g in fr])
		tv = np.array([cams[g][1] for g in fr])
		P = np.array([X[t] for t in tl])
		nc = len(fr)
		ncp = (nc - 1) * 6
		x0 = np.concatenate([np.hstack([rv[1:], tv[1:]]).ravel(), [S["f"]], P.ravel()])
		lb = np.full(len(x0), -np.inf)
		ub = np.full(len(x0), np.inf)
		L = max(W, H)
		if free_f:
			lb[ncp], ub[ncp] = 0.4 * L, 2.0 * L
			x0[ncp] = np.clip(x0[ncp], 0.4 * L + 1e-3, 2.0 * L - 1e-3)
		else:
			lb[ncp], ub[ncp] = S["f"] - 1e-3, S["f"] + 1e-3

		def unpack(x):
			cp = x[:ncp].reshape(-1, 6)
			return (np.vstack([rv[:1], cp[:, :3]]), np.vstack([tv[:1], cp[:, 3:]]),
					x[ncp], x[ncp + 1:].reshape(-1, 3))

		def resid(x):
			R_, T_, f_, P_ = unpack(x)
			return (project_pts(R_[oc], T_[oc], P_[op], f_, cx, cy) - oxy).ravel()

		m = len(oc)
		A = lil_matrix((2 * m, len(x0)), dtype=np.int8)
		idx = np.arange(m)
		cm = oc > 0
		for d in range(2):
			rows = 2 * idx + d
			for c in range(6):
				A[rows[cm], (oc[cm] - 1) * 6 + c] = 1
			A[rows, ncp] = 1
			for c in range(3):
				A[rows, ncp + 1 + 3 * op + c] = 1
		sol = least_squares(resid, x0, jac_sparsity=A, x_scale="jac", loss="huber", f_scale=2.0,
							method="trf", max_nfev=max_nfev, verbose=0, bounds=(lb, ub))
		R_, T_, f_, P_ = unpack(sol.x)
		S["f"] = float(f_)
		for g in fr:
			cams[g] = (cv2.Rodrigues(R_[fi[g]])[0], T_[fi[g]])
		for t in tl:
			X[t] = P_[ti[t]]
		err = np.linalg.norm(resid(sol.x).reshape(-1, 2), axis=1)
		removed = 0
		for e, (t, g) in zip(err, ot):
			if e > cull and g in tracks[t]:
				del tracks[t][g]
				removed += 1
		for t in tl:
			if sum(1 for g in tracks[t] if g in cams) < 2:
				X.pop(t, None)
		log(f"BA: {nc} cams, {len(X)} pts, rms {np.sqrt(np.mean(err ** 2)):.2f}px, "
			f"f={S['f']:.1f}{' (free)' if free_f else ' (fixed)'}, culled obs {removed}")

	ba(15, free_f=False)
	for rnd in range(4):
		before = len(cams)
		tried = set()
		since = 0
		while True:
			cand = [g for g in range(n) if g not in cams and g not in tried]
			if not cand:
				break
			cnt = {g: sum(1 for t in tbf[g] if t in X and g in tracks[t]) for g in cand}
			g = max(cnt, key=cnt.get)
			if cnt[g] < 12:
				break
			tried.add(g)
			tl = [t for t in tbf[g] if t in X and g in tracks[t]]
			obj = np.array([X[t] for t in tl], np.float64)
			img = np.array([kp(g, t) for t in tl], np.float64)
			ok, rv, tv, inl = cv2.solvePnPRansac(obj, img, Kmat(), None, iterationsCount=500,
												 reprojectionError=4.0, confidence=0.999)
			if not ok or inl is None or len(inl) < 12:
				continue
			inl = inl[:, 0]
			rv, tv = cv2.solvePnPRefineLM(obj[inl], img[inl], Kmat(), None, rv, tv)
			cams[g] = (cv2.Rodrigues(rv)[0], tv.ravel())
			by = {}
			for t in tbf[g]:
				if t in X or g not in tracks[t]:
					continue
				part = [h for h in tracks[t] if h != g and h in cams]
				if part:
					h = max(part, key=lambda h_: np.linalg.norm(center(g) - center(h_)))
					by.setdefault(h, []).append(t)
			for h, tl2 in by.items():
				ts, xs = tri_pair(tl2, g, h)
				for t, x in zip(ts, xs):
					X[t] = x
			since += 1
			if len(cams) % 20 == 0:
				log(f"  registered {len(cams)} frames, {len(X)} points (last: frame {g}, "
					f"{cnt[g]} 3D matches, {len(inl)} PnP inliers)")
			if since >= 8:
				ba(15, free_f=len(cams) >= 30)
				since = 0
		if cand:
			best_left = max(cnt.values()) if cnt else 0
			log(f"round {rnd}: stopped with {len(cams)} registered; best remaining frame had {best_left} 3D matches")
		ba(25, free_f=len(cams) >= 30)
		if len(cams) == before:
			break
	ba(40, free_f=True, cull=4.0)
	ba(40, free_f=True, cull=4.0)
	log(f"registered {len(cams)}/{n} frames; registered indices: {ranges_str(cams.keys())}")
	if len(cams) < 0.5 * n:
		log("WARNING: under half of the frames registered; expect a broken reconstruction.")
	K = Kmat()
	return cams, K, (W, H), pts

# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
def orbit_center(cams):
	A = np.zeros((3, 3))
	b = np.zeros(3)
	for g, (R, t) in cams.items():
		C = -R.T @ t
		d = R.T @ np.array([0, 0, 1.0])
		M = np.eye(3) - np.outer(d, d)
		A += M
		b += M @ C
	p = np.linalg.solve(A, b)
	dist = np.median([np.linalg.norm(-R.T @ t - p) for R, t in cams.values()])
	return p, dist


def pick_neighbors(r, regs, cams, p, N, min_deg=3.0, max_deg=30.0):
	Cr = -cams[r][0].T @ cams[r][1] - p
	cand = []
	for g in regs:
		if g == r:
			continue
		Cg = -cams[g][0].T @ cams[g][1] - p
		ang = np.degrees(np.arccos(np.clip(Cr @ Cg / np.linalg.norm(Cr) / np.linalg.norm(Cg), -1, 1)))
		if min_deg <= ang <= max_deg:
			cand.append((ang, g))
	cand.sort()
	return [g for _, g in cand[:N]]


def plane_rows(Kr, Rr, tr, Kn, Rn, tn, d, W, H):
	Rrel = Rn @ Rr.T
	trel = tn - Rrel @ tr
	M = Kn @ (Rrel + np.outer(trel, [0, 0, 1.0]) / d) @ np.linalg.inv(Kr)
	return np.array([M[0] / W, M[2] - M[1] / H, M[2]])  # u-num, v-num, den (u,v in Blender UV space)


# ----------------------------------------------------------------------------
# Stage 3: plane sweep in the compositor (script-unrolled)
# ----------------------------------------------------------------------------
class PlaneSweep:
	def __init__(self, scene, W, H, K, N, window, out_dir):
		self.scene, self.W, self.H, self.K, self.N = scene, W, H, K, N
		self.out_dir = out_dir
		self.cache = {}
		setup_scene(scene, W, H)
		nt = self.nt = new_comp_tree(scene, "PlaneSweep")

		def mkimg(name):
			return bpy.data.images.new(name, W, H, alpha=True, float_buffer=True, is_data=True)

		self.coord_img = mkimg("ps_coords")
		xs = np.arange(W, dtype=np.float32) + 0.5
		ys = H - np.arange(H, dtype=np.float32) - 0.5  # Blender row j (from bottom) -> OpenCV y
		arr = np.zeros((H, W, 4), np.float32)
		arr[..., 0] = xs[None, :]
		arr[..., 1] = ys[:, None]
		arr[..., 3] = 1
		self.coord_img.pixels.foreach_set(arr.ravel())
		self.coord_img.update()
		self.state_img = mkimg("ps_state")

		def img_node(img=None):
			nd = mk(nt, "CompositorNodeImage")
			if img is not None:
				nd.image = img
			return nd

		def sep(src):
			s = mk(nt, "CompositorNodeSeparateColor")
			nt.links.new(src, s.inputs[0])
			return s

		coords = sep(img_node(self.coord_img).outputs[0])
		Xs, Ys = coords.outputs[0], coords.outputs[1]
		self.ref_node = img_node()
		refR = sep(self.ref_node.outputs[0]).outputs[0]
		self.nb_nodes = [img_node() for _ in range(N)]
		st = sep(img_node(self.state_img).outputs[0])
		cost, depth = st.outputs[0], st.outputs[1]
		self.warp_c = []   # [k][n] -> 3 rows x (a, b, c) sockets
		self.depth_c = []  # [k] -> socket

		def lin(Xs_, Ys_):
			t1 = mth(nt, "MULTIPLY_ADD", Xs_, 0.0, 0.0)
			t2 = mth(nt, "MULTIPLY_ADD", Ys_, 0.0, t1.outputs[0])
			return t2.outputs[0], (t1.inputs[1], t2.inputs[1], t1.inputs[2])

		for k in range(self.K):
			costs = []
			wk = []
			for n in range(N):
				nu, ru = lin(Xs, Ys)
				nv, rv_ = lin(Xs, Ys)
				dn, rd = lin(Xs, Ys)
				u = mth(nt, "DIVIDE", nu, dn).outputs[0]
				v = mth(nt, "DIVIDE", nv, dn).outputs[0]
				cb = mk(nt, "CompositorNodeCombineColor")
				nt.links.new(u, cb.inputs[0])
				nt.links.new(v, cb.inputs[1])
				cb.inputs[3].default_value = 1.0
				mu = mk(nt, "CompositorNodeMapUV")
				try:
					mu.filter_type = "ANISOTROPIC"
				except Exception:
					pass
				nt.links.new(self.nb_nodes[n].outputs[0], mu.inputs[0])
				nt.links.new(cb.outputs[0], mu.inputs[1])
				warpR = sep(mu.outputs[0]).outputs[0]
				diff = mth(nt, "ABSOLUTE", mth(nt, "SUBTRACT", refR, warpR).outputs[0])
				ax = mth(nt, "ABSOLUTE", mth(nt, "SUBTRACT", u, 0.5).outputs[0])
				ay = mth(nt, "ABSOLUTE", mth(nt, "SUBTRACT", v, 0.5).outputs[0])
				ins = mth(nt, "LESS_THAN", mth(nt, "MAXIMUM", ax.outputs[0], ay.outputs[0]).outputs[0], 0.5)
				front = mth(nt, "GREATER_THAN", dn, 0.0)
				valid = mth(nt, "MULTIPLY", ins.outputs[0], front.outputs[0])
				cn = mth(nt, "SUBTRACT", mth(nt, "ADD", diff.outputs[0], 1.0).outputs[0], valid.outputs[0])
				costs.append(cn.outputs[0])
				wk.append((ru, rv_, rd))
			self.warp_c.append(wk)
			acc = costs[0]
			for c in costs[1:]:
				acc = mth(nt, "ADD", acc, c).outputs[0]
			mean = mth(nt, "MULTIPLY", acc, 1.0 / N).outputs[0]
			cc = mk(nt, "CompositorNodeCombineColor")
			for i in range(3):
				nt.links.new(mean, cc.inputs[i])
			cc.inputs[3].default_value = 1.0
			bl = mk(nt, "CompositorNodeBlur")
			try:
				bl.filter_type = "BOX"
			except Exception:
				pass
			r = max(1, window // 2)
			if "Size" in bl.inputs:
				try:
					bl.inputs["Size"].default_value = (r, r)
				except Exception:
					try:
						bl.inputs["Size"].default_value = float(r)
					except Exception as e:
						log("warn: blur size", e)
			else:
				try:
					bl.size_x = bl.size_y = r
				except Exception as e:
					log("warn: blur size", e)
			nt.links.new(cc.outputs[0], bl.inputs[0])
			ck = sep(bl.outputs[0]).outputs[0]
			m = mth(nt, "LESS_THAN", ck, cost)
			new_cost = mth(nt, "MINIMUM", ck, cost)
			sub = mth(nt, "SUBTRACT", 0.0, depth)
			new_depth = mth(nt, "MULTIPLY_ADD", sub.outputs[0], m.outputs[0], depth)
			self.depth_c.append(sub.inputs[0])
			cost, depth = new_cost.outputs[0], new_depth.outputs[0]

		fin = mk(nt, "CompositorNodeCombineColor")
		nt.links.new(cost, fin.inputs[0])
		nt.links.new(depth, fin.inputs[1])
		fin.inputs[3].default_value = 1.0
		self.fo = file_output(nt, out_dir, "OPEN_EXR", "32", "RGBA", "state")
		nt.links.new(fin.outputs[0], self.fo.inputs[0])
		go = group_output(nt)
		nt.links.new(fin.outputs[0], go.inputs[0])
		log(f"plane-sweep graph: {len(nt.nodes)} nodes")

	def get_img(self, path):
		if path not in self.cache:
			im = bpy.data.images.load(path, check_existing=True)
			im.colorspace_settings.name = "Non-Color"
			self.cache[path] = im
		return self.cache[path]

	def reset_state(self):
		arr = np.zeros((self.H, self.W, 4), np.float32)
		arr[..., 0] = 10.0
		arr[..., 3] = 1.0
		self.state_img.pixels.foreach_set(arr.ravel())
		self.state_img.update()

	def read_state(self):
		files = glob.glob(os.path.join(self.out_dir, "*.exr"))
		if not files:
			raise RuntimeError("compositor produced no EXR (File Output not executed?)")
		img = bpy.data.images.load(files[0], check_existing=False)
		img.colorspace_settings.name = "Non-Color"
		if tuple(img.size) != (self.W, self.H):
			raise RuntimeError(f"unexpected EXR size {tuple(img.size)}")
		arr = np.empty(self.W * self.H * 4, np.float32)
		img.pixels.foreach_get(arr)
		bpy.data.images.remove(img)
		return arr

	def run_ref(self, ref_path, nb_paths, Kr, Rr, tr, Kn_list, Rn_list, tn_list, depths, per_pass):
		self.ref_node.image = self.get_img(ref_path)
		for nd, p in zip(self.nb_nodes, nb_paths):
			nd.image = self.get_img(p)
		self.reset_state()
		self.scene.frame_set(1)
		arr = None
		for s in range(0, len(depths), per_pass):
			for k in range(per_pass):
				d = depths[s + k]
				self.depth_c[k].default_value = float(d)
				for n in range(self.N):
					rows = plane_rows(Kr, Rr, tr, Kn_list[n], Rn_list[n], tn_list[n], d, self.W, self.H)
					for ri in range(3):
						for ci in range(3):
							self.warp_c[k][n][ri][ci].default_value = float(rows[ri][ci])
			clear_dir(self.out_dir, "*.exr")
			bpy.ops.render.render()
			arr = self.read_state()
			self.state_img.pixels.foreach_set(arr)
			self.state_img.update()
		a = arr.reshape(self.H, self.W, 4)[::-1]  # to OpenCV orientation (top row first)
		return a[..., 1].copy(), a[..., 0].copy()  # depth, cost


# ----------------------------------------------------------------------------
# Stage 4/5: fusion + meshing
# ----------------------------------------------------------------------------


def voxel_down(P, size):
	if len(P) == 0:
		return P
	k = np.floor(P / size).astype(np.int64)
	_, idx = np.unique(k, axis=0, return_index=True)
	return P[idx]

def mesh_marching(P, cfg):
	from scipy import ndimage as ndi
	from skimage.measure import marching_cubes
	res, ext = cfg["mc_res"], 1.15
	cell = 2 * ext / res
	idx = np.floor((P + ext) / cell).astype(np.int64)
	idx = idx[((idx >= 0) & (idx < res)).all(1)]
	flat = np.ravel_multi_index(idx.T, (res, res, res))
	vol = np.bincount(flat, minlength=res ** 3).reshape(res, res, res).astype(np.float32)
	g = ndi.gaussian_filter(vol, 1.0)
	occ = g > cfg["mc_density_frac"] * np.median(g[g > 0])
	occ = ndi.binary_closing(occ, iterations=cfg["mc_close"])
	occ = ndi.binary_fill_holes(occ)
	lab, nl = ndi.label(occ)
	if nl > 1:
		sz = np.bincount(lab.ravel())
		sz[0] = 0
		occ = lab == sz.argmax()
	sm = ndi.gaussian_filter(occ.astype(np.float32), 1.0)
	v, f, _, _ = marching_cubes(sm, 0.5)
	return v * cell - ext, f


def to_blender_mesh(V, F, name, scale):
	me = bpy.data.meshes.new(name)
	me.from_pydata((V * scale).tolist(), [], F.tolist() if F is not None and len(F) else [])
	me.update()
	me.validate()
	ob = bpy.data.objects.new(name, me)
	bpy.context.scene.collection.objects.link(ob)
	for pgn in me.polygons:
		pgn.use_smooth = True
	return ob


# ----------------------------------------------------------------------------

def cfg_hash(cfg, keys, extra=""):
	s = json.dumps({k: cfg[k] for k in keys}, sort_keys=True, default=str) + extra
	return hashlib.md5(s.encode()).hexdigest()[:10]


def load_pickle(path):
	try:
		with open(path, "rb") as f:
			return pickle.load(f)
	except Exception:
		return None


def save_pickle(path, obj):
	with open(path, "wb") as f:
		pickle.dump(obj, f)



def cpu_sweep(ref_img, nb_imgs, Kd, Rr, tr, Rns, tns, depths, window, best_k):
	import cv2
	H, W = ref_img.shape
	Kinv = np.linalg.inv(Kd)
	T = np.array([[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1.0]])
	Ti = np.linalg.inv(T)
	best_c = np.full((H, W), 9.0, np.float32)
	best_d = np.zeros((H, W), np.float32)
	N = len(nb_imgs)
	k = max(1, min(best_k, N))
	for d in depths:
		costs = np.empty((N, H, W), np.float32)
		for i, (nb, Rn, tn) in enumerate(zip(nb_imgs, Rns, tns)):
			Rrel = Rn @ Rr.T
			trel = tn - Rrel @ tr
			M = Kd @ (Rrel + np.outer(trel, [0, 0, 1.0]) / d) @ Kinv   # ref px -> neighbour px (half-pixel convention)
			M = T @ M @ Ti                                              # to OpenCV integer-centre convention
			wp = cv2.warpPerspective(nb, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
									 borderMode=cv2.BORDER_CONSTANT, borderValue=-1.0)
			costs[i] = np.where(wp < 0, 1.0, np.abs(ref_img - wp))
		c = np.partition(costs, k - 1, axis=0)[:k].mean(0) if k < N else costs.mean(0)
		c = cv2.boxFilter(c, -1, (window, window), borderType=cv2.BORDER_REPLICATE)
		m = c < best_c
		best_c = np.where(m, c, best_c)
		best_d = np.where(m, np.float32(d), best_d)
	return best_d, best_c


def filter_depths(depths, costs, ranges, refs, cams, Kd, p, R, cfg, W, H):
	import cv2
	Kinv = np.linalg.inv(Kd)
	ys, xs = np.mgrid[0:H, 0:W]
	base = np.stack([xs + 0.5, ys + 0.5, np.ones((H, W))], -1) @ Kinv.T
	out = {}
	tot = dict(px=0, sphere=0, cost=0, cons=0, region=0)
	hist = np.zeros(6, np.int64)
	thr_log = []
	rad_samples = []
	for r in refs:
		D, C = depths[r], costs[r]
		Rr, tr = cams[r]
		Xw = ((base * D[..., None]) - tr) @ Rr
		rad = np.linalg.norm(Xw - p, axis=-1) / R          # 1.0 = current crop radius
		valid = D > 1e-6
		sph = valid & (rad < 1.0)
		if valid.any():
			lowc = valid & (C < np.percentile(C[valid], 25))
			if lowc.any():
				rad_samples.append(rad[lowc][::20])
		if cfg["cost_thr"] == "auto":
			thr = min(0.3, float(np.percentile(C[sph], cfg["cost_pct"]))) if sph.any() else 0.0
		else:
			thr = float(cfg["cost_thr"])
		thr_log.append(thr)
		okc = sph & (C < thr)
		nbs = pick_neighbors(r, refs, cams, p, 4, min_deg=1.0, max_deg=45.0)
		tol = cfg["cons_tol_planes"] * ranges[r] / cfg["n_planes"]
		good = np.zeros((H, W), np.int32)
		for n in nbs:
			Rn, tn = cams[n]
			Xn = Xw @ Rn.T + tn
			z = Xn[..., 2]
			uv = Xn @ Kd.T
			u = uv[..., 0] / np.maximum(z, 1e-9)
			v = uv[..., 1] / np.maximum(z, 1e-9)
			ui, vi = np.round(u - 0.5).astype(int), np.round(v - 0.5).astype(int)
			inb = (z > 0) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
			dn = depths[n][np.clip(vi, 0, H - 1), np.clip(ui, 0, W - 1)]
			good += (inb & (np.abs(dn - z) < tol)).astype(np.int32)
		hist += np.bincount(np.clip(good[okc], 0, 5), minlength=6)
		ok = okc & (good >= cfg["min_consistent"])
		n_cons = int(ok.sum())
		if cfg["min_region_px"] > 0 and ok.any():
			nl, lab, stats, _ = cv2.connectedComponentsWithStats(ok.astype(np.uint8), connectivity=8)
			keep = np.zeros(nl, bool)
			keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= cfg["min_region_px"]
			ok = keep[lab]
		out[r] = np.where(ok, D, 0.0).astype(np.float32)
		tot["px"] += D.size
		tot["sphere"] += int(sph.sum())
		tot["cost"] += int(okc.sum())
		tot["cons"] += n_cons
		tot["region"] += int(ok.sum())
	log(f"cost_thr setting: {cfg['cost_thr']!r}; thresholds used (min/median/max): "
		f"{min(thr_log):.3f} / {np.median(thr_log):.3f} / {max(thr_log):.3f}")
	log(f"filter funnel: pixels {tot['px']} -> in crop {tot['sphere']} -> cost ok {tot['cost']} "
		f"-> depth-consistent {tot['cons']} -> after region filter {tot['region']}")
	for a, b, name in ((tot["sphere"], tot["px"], "in crop"), (tot["cost"], tot["sphere"], "cost ok"),
					   (tot["cons"], tot["cost"], "consistent"), (tot["region"], tot["cons"], "region filter")):
		log(f"   survival at '{name}': {100.0 * a / max(b, 1):.1f}%")
	if rad_samples:
		rs = np.concatenate(rad_samples)
		pcts = " ".join(f"{np.percentile(rs, q):.2f}" for q in (10, 25, 50, 75, 90))
		log(f"radius (1.0 = current crop) of the best-matching 25% of pixels, pct 10/25/50/75/90: {pcts}")
		log(f"  -> if the 75th/90th percentiles are well above 1.0, your crop is cutting the object; "
			f"if they are well below 1.0 the crop has slack. crop_radius_frac is {cfg['crop_radius_frac']}")
	log(f"agreeing-neighbour histogram among cost-passing pixels (0..5+): {hist.tolist()}")
	return out

def fuse_points(filtered, refs, cams, Kd, stride, W, H):
	Kinv = np.linalg.inv(Kd)
	ys, xs = np.mgrid[0:H:stride, 0:W:stride]
	base = np.stack([xs + 0.5, ys + 0.5, np.ones_like(xs, dtype=np.float64)], -1) @ Kinv.T
	out = []
	for r in refs:
		d = filtered[r][ys, xs]
		ok = d > 0
		Rr, tr = cams[r]
		out.append((((base * d[..., None]) - tr) @ Rr)[ok])
	return np.concatenate(out) if out else np.zeros((0, 3))


def log_point_stats(Pn):
	rad = np.linalg.norm(Pn, axis=1)
	log("points (unit = crop radius): radius pct 10/50/90 = "
		+ " ".join(f"{np.percentile(rad, q):.2f}" for q in (10, 50, 90))
		+ f"; fraction beyond 0.9: {np.mean(rad > 0.9) * 100:.0f}%")
	log("points z (up) pct 5/25/50/75/95 = " + " ".join(f"{np.percentile(Pn[:, 2], q):.2f}" for q in (5, 25, 50, 75, 95)))


def largest_component(v, f):
	from scipy.sparse import coo_matrix
	from scipy.sparse.csgraph import connected_components
	n = len(v)
	e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
	g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
	nc, lab = connected_components(g, directed=False)
	if nc <= 1:
		return v, f
	best = np.bincount(lab).argmax()
	keep_f = lab[f[:, 0]] == best
	f2 = f[keep_f]
	used, inv = np.unique(f2.ravel(), return_inverse=True)
	log(f"mesh components: {nc}; kept largest ({int(keep_f.sum())}/{len(f)} faces)")
	return v[used], inv.reshape(-1, 3)


def mesh_tsdf(filtered, refs, cams, Kd, p, R, ranges, cfg, W, H):
	from scipy import ndimage as ndi
	from skimage.measure import marching_cubes
	res = cfg["tsdf_res"]
	vox = 2.0 * R / (res - 1)
	ax = np.arange(res, dtype=np.float32) * vox - R
	p32 = p.astype(np.float32)
	G = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3) + p32
	inside = np.linalg.norm(G - p32, axis=1) <= R
	Fv = np.zeros(len(G), np.float32)
	Wt = np.zeros(len(G), np.float32)
	fx, fy, cx, cy = Kd[0, 0], Kd[1, 1], Kd[0, 2], Kd[1, 2]
	for r in refs:
		D = filtered[r]
		mu = max(cfg["tsdf_mu_voxels"] * vox, 2.0 * ranges[r] / cfg["n_planes"])
		Rr, tr = cams[r]
		Xc = G @ Rr.T.astype(np.float32) + tr.astype(np.float32)
		z = Xc[:, 2]
		zs = np.maximum(z, 1e-6)
		ui = np.rint(fx * Xc[:, 0] / zs + cx - 0.5).astype(np.int32)
		vi = np.rint(fy * Xc[:, 1] / zs + cy - 0.5).astype(np.int32)
		inb = (z > 1e-6) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
		dm = D[np.clip(vi, 0, H - 1), np.clip(ui, 0, W - 1)]
		sdf = dm - z
		upd = inb & (dm > 0) & (sdf > -mu) & inside
		Fv[upd] += np.clip(sdf[upd] / mu, -1, 1)
		Wt[upd] += 1
	obs = (Wt >= cfg["tsdf_min_views"]).reshape(res, res, res)
	vol = (Fv / np.maximum(Wt, 1)).reshape(res, res, res).astype(np.float32)
	vol[~obs] = 0.0
	if cfg["tsdf_smooth"] > 0:   # smooth only over observed voxels (normalised convolution)
		num = ndi.gaussian_filter(vol * obs, cfg["tsdf_smooth"])
		den = ndi.gaussian_filter(obs.astype(np.float32), cfg["tsdf_smooth"])
		vol = np.where(obs, num / np.maximum(den, 1e-6), 0.0).astype(np.float32)
	log(f"tsdf: observed voxels {int(obs.sum())}, negative (inside) {int(((vol < 0) & obs).sum())}, "
		f"positive (outside) {int(((vol > 0) & obs).sum())}")
	if not ((vol < 0) & obs).any() or not ((vol > 0) & obs).any():
		raise RuntimeError("TSDF has no zero crossing: too few consistent depths. Check filtered_depth_*.png in debug, "
						   "then loosen cost_pct / min_consistent.")
	try:
		v, f, _, _ = marching_cubes(vol, 0.0, mask=obs)
	except TypeError:
		v, f, _, _ = marching_cubes(np.where(obs, vol, 1.0), 0.0)
	v, f = largest_component(v, f)
	return v * vox - R + p, f

def trim_mesh(V, F, radius, min_frac):
	"""V in unit-crop coordinates. Removes faces beyond `radius` and small disconnected pieces."""
	from scipy.sparse import coo_matrix
	from scipy.sparse.csgraph import connected_components
	F = np.asarray(F)
	rad = np.linalg.norm(V, axis=1)
	keep_f = (rad[F] <= radius).all(axis=1)
	F = F[keep_f]
	if len(F) == 0:
		log("trim_mesh: nothing left after radius trim; trim radius too small")
		return V, F
	n = len(V)
	e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
	g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
	nc, lab = connected_components(g, directed=False)
	face_lab = lab[F[:, 0]]
	counts = np.bincount(face_lab, minlength=nc)
	big = counts.max()
	keep_comp = counts >= min_frac * big
	F = F[keep_comp[face_lab]]
	used, inv = np.unique(F.ravel(), return_inverse=True)
	log(f"trim_mesh: radius<={radius:.2f}, components {nc} -> {int(keep_comp.sum())}, faces {len(F)}")
	return V[used], inv.reshape(-1, 3)

def main(cfg=CFG):
	startTime = datetime.now()

	# ensure_packages()
	import cv2
	from concurrent.futures import ThreadPoolExecutor
	scene = bpy.context.scene
	work = cfg["work"]
	cache = os.path.join(work, "cache")
	os.makedirs(cache, exist_ok=True)
	redo = set(cfg.get("redo", []))
	if "frames" in redo:
		redo |= {"sfm", "depth"}
	if "sfm" in redo:
		redo.add("depth")
	t_all = time.time()

	# ---------------- frames ----------------
	raw_dir = os.path.join(work, "raw")
	d_sfm, d_dense = os.path.join(work, "sfm"), os.path.join(work, "dense")
	h_raw = cfg_hash(cfg, ["video", "proc_w", "candidates_per_frame", "max_frames"])
	h_frames = cfg_hash(cfg, ["video", "proc_w", "candidates_per_frame", "max_frames", "rotate", "dense_long"])
	fr = load_pickle(os.path.join(cache, "frames.pkl"))
	if (fr and fr["h"] == h_frames and "frames" not in redo
			and all(os.path.exists(x) for x in fr["sfm_p"] + fr["dense_p"])):
		sfm_p, dense_p, (Ws, Hs), (Wd, Hd) = fr["sfm_p"], fr["dense_p"], fr["ssz"], fr["dsz"]
		log("frames: using cache")
	else:
		raw_json = os.path.join(cache, "raw.json")
		raw = sorted(glob.glob(os.path.join(raw_dir, "*.png")))
		same = False
		if raw and os.path.exists(raw_json) and "frames" not in redo:
			with open(raw_json) as f:
				same = json.load(f).get("h") == h_raw
		if same:
			log("raw frames: using cache")
		else:
			raw = extract_frames(cfg, raw_dir)
			with open(raw_json, "w") as f:
				json.dump({"h": h_raw}, f)
		sfm_p, dense_p, (Ws, Hs), (Wd, Hd) = prep_frames(raw, cfg, d_sfm, d_dense)
		save_pickle(os.path.join(cache, "frames.pkl"),
					dict(h=h_frames, sfm_p=sfm_p, dense_p=dense_p, ssz=(Ws, Hs), dsz=(Wd, Hd)))

	# ---------------- SfM ----------------
	h_sfm = h_frames + cfg_hash(cfg, ["f_guess", "match_window", "ratio", "n_orbits"])
	sf = load_pickle(os.path.join(cache, "sfm.pkl"))
	if sf and sf["h"] == h_sfm and "sfm" not in redo:
		cams, K = sf["cams"], sf["K"]
		log(f"SfM: using cache ({len(cams)} registered frames)")
	else:
		cams, K, _, _ = run_sfm(sfm_p, cfg)
		if len(cams) < max(12, int(0.4 * len(sfm_p))):
			raise RuntimeError(f"SfM only registered {len(cams)}/{len(sfm_p)} frames; fix capture/CFG first.")
		save_pickle(os.path.join(cache, "sfm.pkl"), dict(h=h_sfm, cams=cams, K=K))

	regs = sorted(cams)
	p, dist = orbit_center(cams)
	R_depth = cfg["obj_radius_frac"] * dist
	R_crop = cfg["crop_radius_frac"] * dist
	log(f"orbit center {p}, cam distance {dist:.3f}, depth radius {R_depth:.3f}, crop radius {R_crop:.3f}")
	s = Wd / Ws
	Kd = np.diag([s, s, 1.0]) @ K
	refs = regs[::cfg["ref_stride"]]

	def plan(r):
		nbs = pick_neighbors(r, regs, cams, p, cfg["n_neighbors"],
							 min_deg=cfg["nb_min_deg"], max_deg=cfg["nb_max_deg"])
		while len(nbs) < cfg["n_neighbors"]:
			nbs.append(nbs[-1] if nbs else r)
		Rr, tr = cams[r]
		dc = float((Rr @ p + tr)[2])
		dmin, dmax = max(dc - 1.2 * R_depth, 0.1 * dc), dc + 1.2 * R_depth
		dl = 1.0 / np.linspace(1.0 / dmin, 1.0 / dmax, cfg["n_planes"])
		return nbs, dl, dmax - dmin

	# ---------------- dense depth (per-ref cache, resumable) ----------------
	h_depth = h_sfm + cfg_hash(cfg, ["n_planes", "obj_radius_frac", "ref_stride", "n_neighbors", "nb_min_deg",
									 "nb_max_deg", "cost_window", "dense_engine", "cost_best_k"])
	ddir = os.path.join(cache, "depth_" + h_depth)
	os.makedirs(ddir, exist_ok=True)
	if "depth" in redo:
		clear_dir(ddir, "*.npz")
	assert cfg["n_planes"] % cfg["planes_per_pass"] == 0
	fpath = lambda r: os.path.join(ddir, f"ref_{r:04d}.npz")
	todo = [r for r in refs if not os.path.exists(fpath(r))]
	log(f"depth: {len(refs) - len(todo)} cached, {len(todo)} to compute with engine '{cfg['dense_engine']}'")
	t0 = time.time()
	if cfg["dense_engine"] == "cpu" and todo:
		def cpu_job(r):
			nbs, dl, rng = plan(r)
			ref = cv2.imread(dense_p[r], 0).astype(np.float32) / 255.0
			nb = [cv2.imread(dense_p[n], 0).astype(np.float32) / 255.0 for n in nbs]
			D, C = cpu_sweep(ref, nb, Kd, cams[r][0], cams[r][1], [cams[n][0] for n in nbs],
							 [cams[n][1] for n in nbs], dl, cfg["cost_window"], cfg["cost_best_k"])
			np.savez(fpath(r), D=D, C=C, rng=rng)
			return r
		with ThreadPoolExecutor(max_workers=cfg["cpu_workers"]) as ex:
			for i, r in enumerate(ex.map(cpu_job, todo)):
				log(f"depth ref {r} done ({i + 1}/{len(todo)}, {time.time() - t0:.0f}s)")
	elif todo:
		ps = PlaneSweep(scene, Wd, Hd, cfg["planes_per_pass"], cfg["n_neighbors"], cfg["cost_window"],
						os.path.join(work, "ps_out"))
		for i, r in enumerate(todo):
			nbs, dl, rng = plan(r)
			Rr, tr = cams[r]
			D, C = ps.run_ref(dense_p[r], [dense_p[n] for n in nbs], Kd, Rr, tr, [Kd] * len(nbs),
							  [cams[n][0] for n in nbs], [cams[n][1] for n in nbs], dl, cfg["planes_per_pass"])
			np.savez(fpath(r), D=D, C=C, rng=rng)
			log(f"depth ref {r} done ({i + 1}/{len(todo)}, {time.time() - t0:.0f}s)")
		free_comp_tree(scene, ps.nt)
	depths, costs, ranges = {}, {}, {}
	for r in refs:
		z = np.load(fpath(r))
		depths[r], costs[r], ranges[r] = z["D"], z["C"], float(z["rng"])
	log(f"depth maps ready for {len(refs)} refs")

	dbg = os.path.join(work, "debug")
	os.makedirs(dbg, exist_ok=True)

	def depth_png(D, path):
		lo, hi = np.percentile(D[D > 0], [2, 98]) if (D > 0).any() else (0, 1)
		cv2.imwrite(path, (np.clip((D - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8))

	for r in refs[::max(1, len(refs) // 4)]:
		depth_png(depths[r], os.path.join(dbg, f"depth_{r:04d}.png"))
		cv2.imwrite(os.path.join(dbg, f"cost_{r:04d}.png"), (np.clip(costs[r] / 0.4, 0, 1) * 255).astype(np.uint8))

	# optional: does the GPU compositor graph agree with the CPU reference?
	if cfg.get("compare_compositor") and cfg["dense_engine"] == "cpu":
		r = refs[len(refs) // 2]
		nbs, dl, rng = plan(r)
		ps = PlaneSweep(scene, Wd, Hd, cfg["planes_per_pass"], cfg["n_neighbors"], cfg["cost_window"],
						os.path.join(work, "ps_out"))
		Rr, tr = cams[r]
		D2, C2 = ps.run_ref(dense_p[r], [dense_p[n] for n in nbs], Kd, Rr, tr, [Kd] * len(nbs),
							[cams[n][0] for n in nbs], [cams[n][1] for n in nbs], dl, cfg["planes_per_pass"])
		free_comp_tree(scene, ps.nt)
		step = rng / cfg["n_planes"]
		agree = float(np.mean(np.abs(depths[r] - D2) < 2 * step))
		log(f"COMPARE ref {r}: CPU median cost {np.median(costs[r]):.3f}, GPU median cost {np.median(C2):.3f}, "
			f"depth agreement within 2 planes: {agree * 100:.1f}%")
		depth_png(D2, os.path.join(dbg, f"compare_gpu_depth_{r:04d}.png"))
		depth_png(depths[r], os.path.join(dbg, f"compare_cpu_depth_{r:04d}.png"))

	# ---------------- filter + fuse + mesh (fast, rerun freely) ----------------
	filtered = filter_depths(depths, costs, ranges, refs, cams, Kd, p, R_crop, cfg, Wd, Hd)
	for r in refs[::max(1, len(refs) // 4)]:
		depth_png(filtered[r], os.path.join(dbg, f"filtered_depth_{r:04d}.png"))
	P = fuse_points(filtered, refs, cams, Kd, cfg["fuse_stride"], Wd, Hd)
	log("fused points:", len(P))
	if len(P) < 100:
		raise RuntimeError("Almost nothing survived filtering. See the filter funnel above; loosen cost_pct / "
						   "min_consistent, then rerun (heavy stages are cached).")

	C_ = np.array([-cams[g][0].T @ cams[g][1] for g in regs])
	axis = np.linalg.svd(C_ - C_.mean(0))[2][2]
	upc = np.mean([-cams[g][0][1] for g in regs], 0)
	if axis @ upc < 0:
		axis = -axis
	a = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1.0, 0])
	xv = a - (a @ axis) * axis
	xv /= np.linalg.norm(xv)
	M = np.vstack([xv, np.cross(axis, xv), axis])
	norm = lambda V: (V - p) @ M.T / R_crop

	Pn = voxel_down(norm(P), 0.004)
	log_point_stats(Pn)
	for nme in ("Photogrammetry_Points", "Photogrammetry_Mesh"):
		old = bpy.data.objects.get(nme)
		if old:
			m_ = old.data
			bpy.data.objects.remove(old)
			bpy.data.meshes.remove(m_)
	to_blender_mesh(Pn, None, "Photogrammetry_Points", cfg["output_radius_m"])

	if cfg["mesh_method"] == "tsdf":
		Vw, F = mesh_tsdf(filtered, refs, cams, Kd, p, R_crop, ranges, cfg, Wd, Hd)
		V = norm(Vw)
	else:
		V, F = mesh_marching(Pn, cfg)
		
	V, F = trim_mesh(V, F, cfg["mesh_trim_radius"], cfg["min_component_frac"])
	ob = to_blender_mesh(V, F, "Photogrammetry_Mesh", cfg["output_radius_m"])
	bpy.context.view_layer.objects.active = ob
	ob.select_set(True)
	log(f"DONE in {time.time() - t_all:.0f}s: {len(ob.data.vertices)} verts, {len(ob.data.polygons)} faces")

	totalTime = datetime.now() - startTime
	print(' ')
	bpy.context.scene.frame_set(0)
	print('totalTime = ', totalTime)

	return ob


if __name__ == "__main__":
	main()