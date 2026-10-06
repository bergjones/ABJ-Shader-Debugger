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

import bpy
import bmesh
import math
import mathutils
from mathutils.bvhtree import BVHTree
# bvh = mathutils.bvhtree.BVHTree.FromObject(obj, depsgraph)
from datetime import datetime
import random
import numpy as np
import scipy
import scipy.sparse.linalg as splinalg
from scipy.sparse.linalg import LinearOperator
import importlib
import sys
import copy
import os
import struct
import jax
import jax.numpy as jnp
from jax import jit
from jax.tree_util import Partial

# Force JAX to use 64-bit double precision to maintain mechanical engineering accuracy
jax.config.update("jax_enable_x64", True)

bpy.utils.expose_bundled_modules()
import openvdb as vdb

# ==============================================================================
# 3. POSITION-INDEPENDENT HIGH-STIFFNESS FORCE ENGINE
# ==============================================================================
def analytical_gabor_displace(pos_local, mode_3d=True, scale=5.0, frequency=2.0, anisotropy=1.0, orientation_deg=45.0, blender_displacement_scale=0.5):
	"""
	Refined Analytical Gabor Noise Engine running natively in JAX.
	Matches Blender's procedural parameters precisely across 2D/3D coordinate sets.
	"""
	# 1. Coordinate Vector Space Formatting
	# Scale adjusts the spatial tracking size uniformly
	p = pos_local * scale 

	if not mode_3d:
		# Blender 2D Mode: Project strictly onto XY plane, drop Z tracking
		# This causes the equator artifact, but matches Blender's 2D setting perfectly
		p = p.at[:, 2].set(0.0)

	# 2. Orientation Alignment (Rotation Processing)
	theta = jnp.radians(orientation_deg)

	# Calculate target direction wave vector omega based on rotation angle
	# In 2D/Anisotropic modes, this isolates the alignment of the interleaved bands
	cos_t, sin_t = jnp.cos(theta), jnp.sin(theta)
	omega = jnp.array([cos_t, sin_t, 0.0]) 

	# 3. Kernel Component Splat Evaluation (Gaussian Envelope * Harmonic Wave)
	r_sq = jnp.sum(p**2, axis=-1, keepdims=True)
	r = jnp.sqrt(r_sq + 1e-8)

	# Gaussian Window Function
	gaussian_envelope = jnp.exp(-jnp.pi * r_sq * 0.15)

	# Harmonic Wave Component: Frequency scales perpendicular to noise direction
	# Dot product projects the position vector onto the wave vector omega
	wave_projection = jnp.sum(p * omega, axis=-1, keepdims=True)
	wave_phase = 2.0 * jnp.pi * frequency * wave_projection
	harmonic_signal = jnp.cos(wave_phase)

	# Omnidirectional isotropic component (activated when anisotropy -> 0)
	isotropic_signal = jnp.cos(2.0 * jnp.pi * frequency * r)

	# 4. Anisotropy Blending Loop
	# Blender blends from completely directional (1.0) to dot-like noise (0.0)
	gabor_directional = gaussian_envelope * harmonic_signal
	gabor_omnidirectional = gaussian_envelope * isotropic_signal

	final_noise_val = (anisotropy * gabor_directional) + ((1.0 - anisotropy) * gabor_omnidirectional)

	# 5. Output Direction Mapping
	# Determine the surface normal for the true displacement direction
	normals = pos_local / (jnp.sqrt(jnp.sum(pos_local**2, axis=-1, keepdims=True)) + 1e-8)

	# 0.5 matches your global default vertex displacement strength
	# displacement_strength = 0.5 
	# displacement_strength = 2
	# return normals * final_noise_val * displacement_strength
	return normals * final_noise_val * blender_displacement_scale

def analytical_box_sdf_and_normal(p, center, size, rot_matrix):
	"""
	Computes both the exact Signed Distance Field and the outward-pointing 
	collision normal vector for an arbitrary 3D rotated box primitive.
	"""
	p_local = p - center
	# Rotate query points cleanly into the box's local transformation system
	p_rotated = jnp.matmul(p_local, rot_matrix)
	half_extents = size * 0.5

	d = jnp.abs(p_rotated) - half_extents
	outside_dist = jnp.linalg.norm(jnp.maximum(d, 0.0), axis=-1)
	inside_dist = jnp.minimum(jnp.maximum(d[:, 0], jnp.maximum(d[:, 1], d[:, 2])), 0.0)
	sdf = outside_dist + inside_dist

	# Derive the exact mathematical surface normal vector via localized face evaluation
	sign_vector = jnp.sign(p_rotated)
	max_component_mask = jnp.where(
		(d[:, 0:1] >= d[:, 1:2]) & (d[:, 0:1] >= d[:, 2:3]), jnp.array([1.0, 0.0, 0.0]),
		jnp.where(d[:, 1:2] >= d[:, 2:3], jnp.array([0.0, 1.0, 0.0]), jnp.array([0.0, 0.0, 1.0]))
	)
	local_normal = sign_vector * max_component_mask

	# Rotate the local surface normal back out to global world coordinates
	world_normal = jnp.matmul(local_normal, rot_matrix.T)
	return sdf, world_normal	

def stack_boxes(boxes):
	"""boxes: list of (center, size, rot) -> stacked (B,3), (B,3), (B,3,3)"""
	centers = jnp.stack([b[0] for b in boxes])
	sizes   = jnp.stack([b[1] for b in boxes])
	rots    = jnp.stack([b[2] for b in boxes])
	return centers, sizes, rots

def all_boxes_sdf_and_normal(p, centers, sizes, rots):
	"""Per-particle nearest-box SDF + normal over any number of boxes."""
	sdfs, normals = jax.vmap(analytical_box_sdf_and_normal,
								in_axes=(None, 0, 0, 0))(p, centers, sizes, rots)  # (B,N), (B,N,3)
	idx = jnp.argmin(sdfs, axis=0)
	sdf = jnp.take_along_axis(sdfs, idx[None, :], axis=0)[0]
	normal = jnp.take_along_axis(normals, idx[None, :, None], axis=0)[0]
	return sdf, normal

def mesh_volume_and_gradient(pos, faces):
	"""Signed enclosed volume of a closed triangle mesh and dV/dpos (N,3)."""
	p = pos - jnp.mean(pos, axis=0)
	a, b, c = p[faces[:, 0]], p[faces[:, 1]], p[faces[:, 2]]
	bc = jnp.cross(b, c)
	V = jnp.sum(jnp.sum(a * bc, axis=-1)) / 6.0
	grad = (jnp.zeros_like(pos)
			.at[faces[:, 0]].add(bc / 6.0)
			.at[faces[:, 1]].add(jnp.cross(c, a) / 6.0)
			.at[faces[:, 2]].add(jnp.cross(a, b) / 6.0))
	return V, grad

def build_edges(faces_np, rest_pos):
	"""Returns a tuple (edge_i, edge_j, rest_length) that can be passed straight through jit."""
	e = np.concatenate([faces_np[:, [0, 1]], faces_np[:, [1, 2]], faces_np[:, [2, 0]]])
	e = np.unique(np.sort(e, axis=1), axis=0)
	L0 = np.linalg.norm(rest_pos[e[:, 0]] - rest_pos[e[:, 1]], axis=1)
	return (jnp.array(e[:, 0], dtype=jnp.int32),
			jnp.array(e[:, 1], dtype=jnp.int32),
			jnp.array(L0, dtype=jnp.float32))

def edge_spring_forces(pos, edges, k):
	ei, ej, l0 = edges
	d = pos[ei] - pos[ej]
	L = jnp.sqrt(jnp.sum(d * d, axis=1) + 1e-12)
	f = (-k * (L - l0) / L)[:, None] * d
	return jnp.zeros_like(pos).at[ei].add(f).at[ej].add(-f)

def edge_damping_dv(pos, vel, edges, a):
	"""Damps relative velocity along each edge (a dashpot). Rigid motion is untouched because
	rigid motion doesn't change edge lengths. a = per-substep fraction, clipped for stability."""
	ei, ej, l0 = edges
	d = pos[ei] - pos[ej]
	L = jnp.sqrt(jnp.sum(d * d, axis=1) + 1e-12)
	u = d / L[:, None]
	vrel = jnp.sum((vel[ei] - vel[ej]) * u, axis=1)
	imp = (0.5 * a * vrel)[:, None] * u
	return jnp.zeros_like(vel).at[ei].add(-imp).at[ej].add(imp)

def compute_forces_jax(pos, mu, lam, STATIC_INIT_POS, STATIC_CENTER, STATIC_NORMALS,
					   STATIC_THICKNESS, box_centers, box_sizes, box_rots, edges,
					   collision_radius=0.28, collision_stiffness=1500.0,
					   obstacle_stiffness=35000.0,
					   shape_coherence_stiffness=400.0):

	pos_2d = jnp.reshape(pos, (-1, 3))

	rest_centered = STATIC_INIT_POS - STATIC_CENTER
	R, active_center = compute_best_fit_rotation(pos_2d, rest_centered)

	rotated_local_static = rest_centered @ R.T
	rotated_normals = STATIC_NORMALS @ R.T

	displaced_static_local = rotated_local_static + analytical_gabor_displace(rotated_local_static)
	displaced_static_mesh = displaced_static_local + STATIC_CENTER
	displaced_static_center = jnp.mean(displaced_static_mesh, axis=0)

	local_active = pos_2d - active_center
	displaced_active_mesh = pos_2d + analytical_gabor_displace(local_active)
	displaced_active_center = jnp.mean(displaced_active_mesh, axis=0)

	disp_local = (displaced_active_mesh - displaced_active_center) - \
					(displaced_static_mesh - displaced_static_center)

	F = jnp.eye(3)[None, :, :] + (disp_local[:, :, None] * rotated_normals[:, None, :]) / 1.8

	vmap_det   = jax.vmap(jnp.linalg.det)
	vmap_trace = jax.vmap(jnp.trace)
	vmap_inv_t = jax.vmap(lambda m: jnp.linalg.inv(m).T)

	J = vmap_det(F)
	J_stable = jnp.maximum(J, 0.20)
	I_C = vmap_trace(F)
	F_inv_t = vmap_inv_t(F)

	# alpha = 1.0 + (mu / lam)
	alpha = 1.0 + 0.75 * (mu / lam)    # P(F=I) = 0 -> no built-in outward pressure
	term1 = (mu * (1.0 - 1.0 / (I_C + 1.0)))[:, None, None] * F
	term2 = (lam * (J_stable - alpha))[:, None, None] * F_inv_t
	P = term1 + term2

	continuum_forces = -jnp.matmul(P, rotated_normals[..., None]).squeeze(-1)

	goal_pos = rotated_local_static + active_center
	shape_match_forces = shape_coherence_stiffness * (goal_pos - pos_2d)




	# While touching a surface, stop the springs from holding the top up along the squash axis
	c_sdf, c_nrm = all_boxes_sdf_and_normal(pos_2d, box_centers, box_sizes, box_rots)
	c_w = smooth_contact_weight(c_sdf, band=0.25)
	c_sum = jnp.sum(c_w[:, None] * c_nrm, axis=0)
	c_len = jnp.linalg.norm(c_sum)
	c_axis = jax.lax.stop_gradient(
		jnp.where(c_len > 1e-3, c_sum / jnp.maximum(c_len, 1e-6), jnp.array([0.0, 0.0, 1.0])))
	c_frac = jax.lax.stop_gradient(jnp.clip(jnp.sum(c_w) / (0.02 * pos_2d.shape[0]), 0.0, 1.0))
	axial = (shape_match_forces @ c_axis)[:, None] * c_axis[None, :]
	# AXIAL_RELAX = 0.85        # 0 = old behavior, 1 = no axial spring while touching
	AXIAL_RELAX = 0.85       # 0 = old behavior, 1 = no axial spring while touching
	shape_match_forces = shape_match_forces - AXIAL_RELAX * c_frac * axial


	dist_to_center = jnp.sqrt(jnp.sum(local_active**2, axis=-1) + 1e-8)
	surface_radius = 1.8 + jnp.squeeze(jnp.linalg.norm(analytical_gabor_displace(local_active), axis=-1))
	self_penetration = jnp.maximum(collision_radius - (surface_radius - dist_to_center), 0.0)
	self_collision_forces = STATIC_NORMALS * (self_penetration[:, None] ** 2) * collision_stiffness

	# Obstacle penalty forces over ALL boxes
	sdfs, normals = jax.vmap(analytical_box_sdf_and_normal, in_axes=(None, 0, 0, 0))(
		displaced_active_mesh, box_centers, box_sizes, box_rots)       # (B,N), (B,N,3)
	pen = jnp.maximum(0.05 - sdfs, 0.0)
	box_forces = jnp.sum(normals * ((pen ** 2) * obstacle_stiffness)[..., None], axis=0)

	# CONTINUUM_SCALE = 0.25
	# CONTINUUM_SCALE = 2
	CONTINUUM_SCALE = 1

	# EDGE_K = 1500.0
	EDGE_K = 300.0 ###
	# EDGE_K = 100.0
	edge_forces = edge_spring_forces(pos_2d, edges, EDGE_K)
	total_forces = (continuum_forces * CONTINUUM_SCALE + shape_match_forces + edge_forces + (self_collision_forces * 0.02) + box_forces)

	# total_forces = continuum_forces + shape_match_forces + (self_collision_forces * 0.02) + box_forces
	# total_forces = (continuum_forces * CONTINUUM_SCALE) + shape_match_forces + (self_collision_forces * 0.02) + box_forces
	return jnp.reshape(total_forces, pos.shape)

def loss_function(mu, lam, damping, dt, initial_spike_vel, restitution, object_mass, drag_coefficient, STATIC_INIT_POS, STATIC_CENTER, STATIC_NORMALS, STATIC_THICKNESS, target_frame_idx, target_height):
	"""Loss function tracking separate individual scalars."""
	# trajectory = run_simulation_scan(mu, lam, damping, dt, initial_spike_vel, restitution, object_mass, drag_coefficient, STATIC_INIT_POS, STATIC_CENTER, STATIC_NORMALS, STATIC_THICKNESS, num_frames=300)
	trajectory = run_simulation_scan_substepped(mu, lam, damping, dt, initial_spike_vel, restitution, object_mass, drag_coefficient, STATIC_INIT_POS, STATIC_CENTER, STATIC_NORMALS, STATIC_THICKNESS, num_frames=300)
	frame_positions = trajectory[target_frame_idx]
	
	max_z = jnp.max(frame_positions[:, 2])
	min_z = jnp.min(frame_positions[:, 2])
	bounding_box_center_z = (max_z + min_z) / 2.0
	
	raw_loss = (bounding_box_center_z - target_height) ** 2
	return jnp.log(1.0 + raw_loss)

def quat_mult(q1, q2):
	w1, x1, y1, z1 = q1
	w2, x2, y2, z2 = q2
	return jnp.array([
		w1*w2 - x1*x2 - y1*y2 - z1*z2,
		w1*x2 + x1*w2 + y1*z2 - z1*y2,
		w1*y2 - x1*z2 + y1*w2 + z1*x2,
		w1*z2 + x1*y2 - y1*x2 + z1*w2,
	])

def quat_to_matrix(q):
	w, x, y, z = q
	return jnp.array([
		[1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
		[2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
		[2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
	])

def integrate_quat(q, omega_world, dt):
	omega_quat = jnp.concatenate([jnp.zeros(1), omega_world])
	q_new = q + 0.5 * quat_mult(omega_quat, q) * dt
	return q_new / (jnp.linalg.norm(q_new) + 1e-9)   # renormalize every substep

# def smooth_contact_weight(sdf, band=0.06):
# 	"""
# 	Continuous 0->1 contact weight instead of a hard boolean.
# 	band controls how wide the transition zone is (world units).
# 	This is THE fix for the flapping: no vertex ever jumps discretely
# 	between "in contact" and "free" behavior between substeps.
# 	"""
# 	t = jnp.clip((band - sdf) / band, 0.0, 1.0)
# 	return t * t * (3.0 - 2.0 * t)  # smoothstep

# ---------------------------------------------------------------
# Kabsch / shape-matching best-fit rotation.
# Computes the single rigid rotation R that best aligns the REST shape
# to the CURRENT point cloud (in a least-squares sense), via SVD of the
# 3x3 cross-covariance matrix. This is recomputed fresh from whatever
# `pos` is passed in every time it's called — there is no persistent
# rotation state to drift or desync from the actual mesh. That is the
# core fix: the elastic rest-frame can never disagree with reality.
#
# NOTE (differentiability): SVD gradients can be numerically unstable
# near-degenerate singular values. For a well-spread sphere point cloud
# this is very unlikely to matter, but if you backprop through many
# frames for optimization and see NaN/exploding grads, wrap the R output
# in jax.lax.stop_gradient() at the call site (rotation still governs
# the forward sim correctly; you just stop treating "which way it spun"
# as something to differentiate through).
# ---------------------------------------------------------------
def compute_best_fit_rotation(pos, rest_centered):
	"""
	pos:           (N,3) current world positions
	rest_centered: (N,3) = STATIC_INIT_POS - STATIC_CENTER  (fixed rest, centered)
	returns: R (3,3) rotation with R @ rest_centered_i ≈ pos_i - center,
			 center (3,) = mean(pos)
	"""
	center = jnp.mean(pos, axis=0)
	pos_centered = pos - center

	H = rest_centered.T @ pos_centered  # (3,3) cross-covariance
	U, _, Vt = jnp.linalg.svd(H)

	# Reflection correction so det(R) == +1 (a proper rotation, not a mirror)
	d = jnp.sign(jnp.linalg.det(Vt.T @ U.T))
	correction = jnp.diag(jnp.array([1.0, 1.0, d]))
	R = Vt.T @ correction @ U.T
	return R, center

def smooth_contact_weight(sdf, band=0.06):
	"""Continuous 0->1 contact weight. No vertex ever jumps discretely
	between 'in contact' and 'free' behavior between substeps."""
	t = jnp.clip((band - sdf) / band, 0.0, 1.0)
	return t * t * (3.0 - 2.0 * t)  # smoothstep

def compute_rigid_velocity_field(pos_2d, vel_2d, reg=1e-4):
	"""
	Extracts the best-fit RIGID velocity field (translation + rotation)
	implied by the current particle velocities, using the standard
	angular-momentum / inertia-tensor decomposition:
		omega = I^-1 * L
	where L is angular momentum about the center and I is the instantaneous
	inertia tensor. This is recomputed fresh from `pos_2d`/`vel_2d` every
	call — same self-consistency principle as the SVD rotation fit, so it
	can never desync from the real motion.

	v_rigid_i = v_center + omega x r_i

	Any part of vel_2d NOT explained by this field is, by definition, pure
	deformation (squash, ripple, flapping) — exactly what we want to damp
	without touching rotation/translation.
	"""
	center = jnp.mean(pos_2d, axis=0)
	v_center = jnp.mean(vel_2d, axis=0)
	r = pos_2d - center
	v_rel = vel_2d - v_center

	L = jnp.sum(jnp.cross(r, v_rel), axis=0)
	r_sq = jnp.sum(r * r, axis=1)
	I3 = jnp.eye(3)
	I_tensor = jnp.sum(r_sq[:, None, None] * I3[None, :, :] - r[:, :, None] * r[:, None, :], axis=0)
	I_tensor_reg = I_tensor + reg * jnp.eye(3)  # tiny reg for numerical safety under jit

	omega = jnp.linalg.solve(I_tensor_reg, L)
	v_rigid = v_center[None, :] + jnp.cross(omega[None, :], r)
	return v_rigid

def enforce_min_thickness(pos, vel, rest_centered, normals, min_half=0.08):
	"""Keeps top/bottom layers from crossing or collapsing to zero thickness.
	Works in any orientation (rolling, tumbling) because the squash axis is found from the data."""
	pos_sg = jax.lax.stop_gradient(pos)
	c = jnp.mean(pos_sg, axis=0)
	Xs = pos_sg - c
	cov = Xs.T @ Xs / pos.shape[0]
	_, evecs = jnp.linalg.eigh(cov)
	a = evecs[:, 0]                                   # smallest-variance axis = flatten axis
	R, _ = compute_best_fit_rotation(pos_sg, rest_centered)

	s = (normals @ R.T) @ a                           # how much each vertex faces the axis (-1..1)
	sgn = jnp.sign(s)
	required = min_half * s * s                       # rim vertices (s~0) are barely constrained
	h = sgn * ((pos - c) @ a)                         # signed height above the center plane
	viol = jnp.maximum(required - h, 0.0)

	dirv = sgn[:, None] * a[None, :]
	pos_new = pos + dirv * viol[:, None]
	v_in = jnp.sum(vel * dirv, axis=-1)
	vel_new = vel - dirv * jnp.where(viol > 0.0, jnp.minimum(v_in, 0.0), 0.0)[:, None]
	return pos_new, vel_new

def volume_flow_projection(pos, vel, axis, gain):
	"""Velocity-level volume conservation: lateral strain rate is pushed toward -0.5 * axial strain rate.
	Adds no net momentum (the correction is proportional to the centered lateral offset)."""
	c = jnp.mean(pos, axis=0)
	x = pos - c
	za = x @ axis
	rt = x - za[:, None] * axis[None, :]
	va = vel @ axis
	va_rel = va - jnp.mean(va)
	vt = vel - va[:, None] * axis[None, :]
	vt_rel = vt - jnp.mean(vt, axis=0)

	e_ax = jnp.sum(va_rel * za) / (jnp.sum(za * za) + 1e-6)       # axial strain rate (1/s)
	e_lat = jnp.sum(vt_rel * rt) / (jnp.sum(rt * rt) + 1e-6)      # lateral strain rate (1/s)
	target = -0.5 * e_ax
	return vel + gain * (target - e_lat) * rt

def run_simulation_scan_substepped(mu, lam, damping, dt_frame, initial_spike_vel, restitution,
								   object_mass, drag_coefficient, STATIC_INIT_POS, STATIC_CENTER,
								   STATIC_NORMALS, STATIC_THICKNESS,
								   box_centers, box_sizes, box_rots, edges, num_frames=300):

	SUBSTEPS = 24
	# SUBSTEPS = 96
	# SUBSTEPS = 48
	dt_sub = dt_frame / SUBSTEPS
	init_vel = jnp.zeros_like(STATIC_INIT_POS).at[:, 2].set(initial_spike_vel)
	substep_damping = jnp.exp(jnp.log(damping) * (dt_sub / dt_frame))

	# obstacle_stiffness = 500000
	obstacle_stiffness = 500000
	# CONTACT_BAND = 0.5
	# CONTACT_BAND = 0.8
	# CONTACT_BAND = 0.25
	CONTACT_BAND = 0.1
	# shape_coherence_stiffness = 500
	# shape_coherence_stiffness = 1000
	# shape_coherence_stiffness = 100
	# shape_coherence_stiffness = 15
	# shape_coherence_stiffness = .01
	# LOCAL_DAMPING_RATE = 2

	# EDGE_K_SIM = 300.0
	EDGE_DAMP_RATE = 300.0     # 1/s. The ripple knob: 0 = ringing, higher = calmer, more "gel"
	edge_a = jnp.minimum(1.0 - jnp.exp(-EDGE_DAMP_RATE * dt_sub), 0.3)   # 0.3 cap keeps it stable

	# LOCAL_DAMPING_RATE = 2
	# LOCAL_DAMPING_RATE = 20
	# LOCAL_DAMPING_RATE = 40
	LOCAL_DAMPING_RATE = 8
	# LOCAL_DAMPING_RATE = 4000
	# SPLAT_GAIN = 0.3                 # lateral speed gained, as a fraction of approach speed
	# SPLAT_GAIN = 1                 # lateral speed gained, as a fraction of approach speed
	TANGENTIAL_RESISTANCE_RATE = 30  # 1/s, rate-based so the patch can slide outward

	# MIN_HALF_THICKNESS = 0.5
	# MIN_HALF_THICKNESS = 0.2
	MIN_HALF_THICKNESS = 0.02

	SPLAT_GAIN = 0.5               # was 1.0 (the splat plus bounce was creating energy)
	# RIPPLE_SPEED_CAP = 8.0        # max deformation speed (m/s) relative to the rigid motion
	# RIPPLE_SPEED_CAP = 16.0        # max deformation speed (m/s) relative to the rigid motion
	RIPPLE_SPEED_CAP = 1000.0        # max deformation speed (m/s) relative to the rigid motion
	# RIPPLE_SPEED_CAP = 25.0        # max deformation speed (m/s) relative to the rigid motion
	# RIPPLE_SPEED_CAP = 60.0        # max deformation speed (m/s) relative to the rigid motion
	# RIPPLE_SPEED_CAP = 120.0        # max deformation speed (m/s) relative to the rigid motion

	# shape_coherence_stiffness = 15 # 15 let the rim fly; raise if the body still tears
	# shape_coherence_stiffness = 5 # 15 let the rim fly; raise if the body still tears
	# shape_coherence_stiffness = 1 # 15 let the rim fly; raise if the body still tears
	shape_coherence_stiffness = 10 # 15 let the rim fly; raise if the body still tears
	# shape_coherence_stiffness = 40 # 15 let the rim fly; raise if the body still tears
	# shape_coherence_stiffness = 100 # 15 let the rim fly; raise if the body still tears
	# shape_coherence_stiffness = 500 # 15 let the rim fly; raise if the body still tears

	VOLUME_FLOW_GAIN = .6    # 0 = off, 1 = fully incompressible flow; raise for a flatter steamroll
	# VOLUME_FLOW_GAIN = 1    # 0 = off, 1 = fully incompressible flow; raise for a flatter steamroll


	initial_state_carry = (STATIC_INIT_POS, init_vel)

	def single_physics_substep(state, _):
		pos, vel = state
		gravity = -9.81
		gamma = 2.0 - jnp.sqrt(2.0)
		dt1 = gamma * dt_sub

		c_mid = 1.0 / (gamma * (2.0 - gamma))
		c_cur = ((1.0 - gamma) ** 2) / (gamma * (2.0 - gamma))
		c_f2 = (1.0 - gamma) / (2.0 - gamma)

		v_mags = jnp.linalg.norm(vel, axis=1, keepdims=True)
		drag_forces = -(drag_coefficient / object_mass) * vel * v_mags
		gravity_forces = jnp.zeros_like(vel).at[:, 2].set(gravity)
		environmental_accel = gravity_forces + drag_forces

		f1 = compute_forces_jax(pos, mu, lam, STATIC_INIT_POS, STATIC_CENTER, STATIC_NORMALS,
								STATIC_THICKNESS, box_centers, box_sizes, box_rots, edges,
								.28, 1500, obstacle_stiffness, shape_coherence_stiffness)

		v_mid = vel + (environmental_accel + f1 * substep_damping) * dt1
		pos_est = pos + v_mid * dt1

		f2 = compute_forces_jax(pos_est, mu, lam, STATIC_INIT_POS, STATIC_CENTER, STATIC_NORMALS,
								STATIC_THICKNESS, box_centers, box_sizes, box_rots, edges,
								.28, 1500, obstacle_stiffness, shape_coherence_stiffness)

		new_vel = (c_mid * v_mid) - (c_cur * vel) + (f2 * substep_damping) * dt_sub * c_f2 + \
					environmental_accel * (dt_sub * (1.0 - gamma))
		next_pos = pos + new_vel * dt_sub

		v_rigid_field = compute_rigid_velocity_field(next_pos, new_vel)
		local_damping_decay = jnp.exp(-LOCAL_DAMPING_RATE * dt_sub)
		# new_vel = v_rigid_field + (new_vel - v_rigid_field) * local_damping_decay

		new_vel = new_vel + edge_damping_dv(next_pos, new_vel, edges, edge_a)

		next_pos, new_vel = enforce_min_thickness(next_pos, new_vel, STATIC_INIT_POS - STATIC_CENTER,
												  STATIC_NORMALS, MIN_HALF_THICKNESS)


		# --- Contact against ALL boxes ---
		ROLL_DAMP_RATE = 0.0 # 1/s. Optional: try 2-5 if the wobble persists after the fixes below

		active_sdf, active_normal = all_boxes_sdf_and_normal(next_pos, box_centers, box_sizes, box_rots)
		w_contact = smooth_contact_weight(active_sdf, band=CONTACT_BAND)[:, None]
		w = w_contact.squeeze(-1)

		v_dot_n = jnp.sum(new_vel * active_normal, axis=-1, keepdims=True)
		v_tangent = new_vel - (v_dot_n * active_normal)

		penetration = jnp.maximum(0.02 - active_sdf, 0.0)
		fn_magnitude = (penetration ** 2) * obstacle_stiffness
		ft_magnitude_coulomb = fn_magnitude * 0.45

		v_tangent_mags = jnp.linalg.norm(v_tangent, axis=-1, keepdims=True)
		per_particle_mass = object_mass / len(pos)
		stopping_force_mag = (v_tangent_mags * per_particle_mass) / dt_sub
		friction_mag = jnp.minimum(ft_magnitude_coulomb[:, None], stopping_force_mag)
		safe_dir = v_tangent / jnp.maximum(v_tangent_mags, 1e-5)

		friction_accel = friction_mag / per_particle_mass
		dv_friction = jnp.minimum(friction_accel * dt_sub, v_tangent_mags)
		new_vel_corrected = new_vel - w_contact * safe_dir * dv_friction

		# Push out of actual penetration only
		push_out = active_normal * jnp.maximum(-active_sdf, 0.0)[:, None]
		next_pos_final = next_pos + push_out

		# Reflection for approaching vertices only
		v_dot_n_corr = jnp.sum(new_vel_corrected * active_normal, axis=-1, keepdims=True)
		v_in = jnp.minimum(v_dot_n_corr, 0.0)
		v_reflected = new_vel_corrected - (1.0 + restitution) * v_in * active_normal
		vel_after = new_vel_corrected + w_contact * (v_reflected - new_vel_corrected)

		# Lateral splat: in the contact plane, away from the body center, independent of restitution
		r = next_pos_final - jnp.mean(next_pos_final, axis=0)
		r_t = r - jnp.sum(r * active_normal, axis=-1, keepdims=True) * active_normal
		splat_dir = r_t / jnp.maximum(jnp.linalg.norm(r_t, axis=-1, keepdims=True), 1e-4)
		vel_after = vel_after + w_contact * splat_dir * (-v_in) * SPLAT_GAIN

		# Resistance: NORMAL component only (keeps the pancake squeeze); tangential decays gently
		ROLLING_RESISTANCE = 0.3
		SEPARATING_RESISTANCE_SCALE = 0.0   # normal brake on separating verts: fully off (kills the strings)
		SEP_TANGENTIAL_SCALE = 0.3          # separating verts keep 30% of tangential damping (keeps splat stable)

		vn_a = jnp.sum(vel_after * active_normal, axis=-1, keepdims=True)
		v_n_comp = vn_a * active_normal
		v_t_comp = vel_after - v_n_comp

		sep = (vn_a.squeeze(-1) > 0.0).astype(jnp.float32)
		w_res = w * (1.0 - sep * (1.0 - SEPARATING_RESISTANCE_SCALE))
		w_tan = w * (1.0 - sep * (1.0 - SEP_TANGENTIAL_SCALE))

		factor_n = 1.0 - w_res * (1.0 - ROLLING_RESISTANCE)
		factor_t = jnp.exp(-TANGENTIAL_RESISTANCE_RATE * w_tan * dt_sub)

		next_vel = v_n_comp * factor_n[:, None] + v_t_comp * factor_t[:, None]
		# next_vel = next_vel * jnp.exp(-ROLL_DAMP_RATE * jnp.mean(w) * dt_sub)


		next_vel = next_vel * jnp.exp(-ROLL_DAMP_RATE * jnp.mean(w) * dt_sub)

		# Steamroller: incompressible lateral flow along the contact-plane axis, only while touching
		n_sum = jnp.sum(w_contact * active_normal, axis=0)
		n_len = jnp.linalg.norm(n_sum)
		squash_axis = jax.lax.stop_gradient(
			jnp.where(n_len > 1e-3, n_sum / jnp.maximum(n_len, 1e-6), jnp.array([0.0, 0.0, 1.0])))
		contact_frac = jax.lax.stop_gradient(jnp.clip(jnp.sum(w) / (0.02 * pos.shape[0]), 0.0, 1.0))
		next_vel = volume_flow_projection(next_pos_final, next_vel, squash_axis, VOLUME_FLOW_GAIN * contact_frac)

		# Cap deformation speed (relative to best-fit rigid motion); rigid motion is untouched
		v_rig = compute_rigid_velocity_field(next_pos_final, next_vel)
		dv_def = next_vel - v_rig
		dv_mag = jnp.linalg.norm(dv_def, axis=-1, keepdims=True)
		next_vel = v_rig + dv_def * jnp.minimum(1.0, RIPPLE_SPEED_CAP / jnp.maximum(dv_mag, 1e-6))









		return (next_pos_final, next_vel), next_pos_final

	def advance_single_animation_frame(frame_state, _):
		final_substep_state, _ = jax.lax.scan(single_physics_substep, frame_state, None, length=SUBSTEPS)

		frame_positions, _ = final_substep_state
		return final_substep_state, frame_positions

	_, trajectory = jax.lax.scan(advance_single_animation_frame, (STATIC_INIT_POS, init_vel), None, length=num_frames)
	return trajectory

jit_simulation_engine = jax.jit(run_simulation_scan_substepped, static_argnums=(16,))

# vmap_det = jax.vmap(jnp.linalg.det)
# vmap_trace = jax.vmap(lambda mat: jnp.trace(jnp.dot(mat.T, mat)))
# vmap_inv_t = jax.vmap(lambda mat: jnp.linalg.inv(mat + 1e-5 * jnp.eye(3)).T)

# 1. Define a custom callback class to track progress
class PercentageCallback:
	def __init__(self, tol):
		self.tol = tol
		self.initial_resid = None
		self.iteration = 0

	def __call__(self, x_or_resid):
		# SciPy handles callbacks differently depending on the version.
		# Modern SciPy passes the residual norm directly to the callback.
		# If it's a vector, we calculate its norm.
		if np.isscalar(x_or_resid):
			current_resid = x_or_resid
		else:
			# Fallback if your SciPy version passes the current solution vector 'x'
			# (Note: calculating residual from 'x' requires A and b, 
			# so tracking by residual norm is highly preferred)
			current_resid = np.linalg.norm(x_or_resid) 
		
		# Set the initial residual on the very first iteration
		if self.initial_resid is None or self.initial_resid == 0:
			self.initial_resid = current_resid
			
		self.iteration += 1
		
		# Prevent division by zero if already convergedf
		if self.initial_resid <= self.tol:
			percent = 100.0
		else:
			# Calculate progress on a logarithmic scale since CG converges exponentially
			# Distance remaining in log space divided by total log distance needed
			total_log_dist = np.log10(self.initial_resid) - np.log10(self.tol)
			if total_log_dist > 0:
				current_log_dist = np.log10(self.initial_resid) - np.log10(max(current_resid, self.tol))
				percent = (current_log_dist / total_log_dist) * 100
			else:
				percent = 100.0

		# Clip between 0 and 100 just in case of residual bounces
		percent = clip_percent = max(0.0, min(100.0, percent))
		
		# Print progress overlaying the same line (\r)
		print(f"\rIteration {self.iteration}: Progress ~{percent:.1f}% (Resid: {current_resid:.2e})", end="", flush=True)

@jax.jit
def assemble_global_forces_jax(u_global, v_global, nodes_global, topology, properties, fixed_dofs, dt_scale):
	"""
	XLA JIT-Compiled Global Multi-Phase Force Assembler.
	Computes and gathers internal forces across the entire mesh instantly.
	"""
	# 1. Gather elements into parallel batched tensors
	u_batched = u_global[topology]
	v_batched = v_global[topology]
	coords_batched = nodes_global[topology]

	# 2. Parallel vectorization map across element material property arrays
	mat_ids = jnp.where(properties == 1, 101.0, 202.0)
	Es = jnp.where(properties == 1, 10.0, 1.0)
	nus = jnp.where(properties == 1, 0.45, 0.001)

	# 3. Parallel execution pass using your standalone core kernel
	vmapped_forces = jax.vmap(compute_element_forces_jax, in_axes=(0, 0, 0, 0, 0, 0, None))
	f_batched_tet = vmapped_forces(u_batched, v_batched, coords_batched, mat_ids, Es, nus, dt_scale)

	# 4. Hardware-accelerated High-Order Scatter Pass
	f_int_global = jnp.zeros_like(u_global)
	f_int_global = f_int_global.at[topology.ravel()].add(f_batched_tet.reshape(-1, 3))

	# Flatten and enforce boundary conditions
	f_int_flat = f_int_global.ravel()
	f_int_flat = f_int_flat.at[fixed_dofs].set(0.0)

	return f_int_flat

def compute_element_forces_jax(element_displacements, element_velocities, element_node_coords, mat_id, E, nu, dt1):
	"""Evaluates and integrates total internal forces over the Tet10 element geometry."""
	f_int_element = jnp.zeros((10, 3))
	
	# 4-Point Gauss Quadrature Constants
	a, b = 0.5854101966249685, 0.1381966011250105
	gauss_points = jnp.array([[a,b,b], [b,a,b], [b,b,a], [b,b,b]])
	gauss_weight = 1.0 / 24.0
	
	for gp in gauss_points:
		r, s, t = gp[0], gp[1], gp[2]
		u = 1.0 - r - s - t
		
		# Tet10 Basis derivatives
		dN_dr = jnp.array([-4*u+1, 4*r-1, 0, 0, 4*u-4*r, 4*s, -4*s, -4*t, 4*t, 0])
		dN_ds = jnp.array([-4*u+1, 0, 4*s-1, 0, -4*r, 4*r, 4*u-4*s, -4*t, 0, 4*t])
		dN_dt = jnp.array([-4*u+1, 0, 0, 4*t-1, -4*r, 0, -4*s, 4*u-4*t, 4*r, 4*s])
		
		dN_dxi = jnp.stack([dN_dr, dN_ds, dN_dt], axis=0)
		Jacobian = jnp.dot(dN_dxi, element_node_coords)
		det_J = jnp.linalg.det(Jacobian)
		inv_Jacobian = jnp.linalg.inv(Jacobian)
		
		dN_dx = jnp.dot(inv_Jacobian.T, dN_dxi)
		dV = det_J * gauss_weight
		
		# Extract deformation gradients relative to reference layout
		disp_grad = element_displacements.T @ dN_dx.T
		vel_grad = element_velocities.T @ dN_dx.T
		F = jnp.eye(3, dtype=np.float64) + disp_grad
		
		P_stress = evaluate_p_stress_jax(F, disp_grad, vel_grad, mat_id, E, nu, dt1)
		f_int_element += (P_stress @ dN_dx * dV).T
		
	return f_int_element

vmapped_forces_engine = jax.vmap(compute_element_forces_jax, in_axes=(0, 0, 0, 0, 0, 0, None))

def evaluate_p_stress_jax(F_eval, disp_grad_eval, vel_grad_eval, mat_id, E, nu, dt_scale):
	"""Computes First Piola-Kirchhoff stress tensor universally for any phase using pure JAX mathematical branches."""
	J_vol = jnp.linalg.det(F_eval)

	# ----------------------------------------------------------------------
	# PHASE A: SOLID TISSUE (Stable Neo-Hookean)
	# ----------------------------------------------------------------------
	# mu_solid = E / (2.0 * (1.0 + nu))
	# lambda_solid = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
	# alpha = 1.0 + (mu_solid / lambda_solid)
	# stress_scale = mu_solid * (1.0 - (1.0 / (jnp.trace(F_eval.T @ F_eval) + 1.0)))

	# # Differentiable cofactor formulation using matrix inverses
	# F_cofactor = J_vol * jnp.linalg.inv(F_eval).T
	# P_solid = stress_scale * F_eval + (lambda_solid * (J_vol - alpha)) * F_cofactor

	# ----------------------------------------------------------------------
	# PHASE A: SOLID TISSUE (True 2018 Stable Neo-Hookean Formulation)
	# ----------------------------------------------------------------------
	mu_solid = E / (2.0 * (1.0 + nu))
	lambda_solid = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))

	# Define invariants accurately
	I_C = jnp.trace(F_eval.T @ F_eval)

	# Alpha must incorporate the exact 2018 volume mapping modifier constant
	alpha = 1.0 + (mu_solid / lambda_solid)

	# Compute the precise stable stress scaling component
	stress_scale = mu_solid * (1.0 - (1.0 / (I_C + 1.0)))

	# Compute the native mathematical transpose inverse: F^-T 
	# (This replaces your duplicated J_vol * F_cofactor expansion path)
	F_inv_T = jnp.linalg.inv(F_eval).T

	# The true, non-explosive First Piola-Kirchhoff tensor expression:
	P_solid = stress_scale * F_eval + (lambda_solid * (J_vol - alpha)) * F_inv_T

	# ----------------------------------------------------------------------
	# PHASE B: LIQUID PHASE (Navier-Stokes Continuum)
	# ----------------------------------------------------------------------
	Kf, viscosity_mu = E, nu
	pressure = Kf * (J_vol - 1.0)
	D_tensor = 0.5 * (vel_grad_eval + vel_grad_eval.T)
	div_v = jnp.trace(D_tensor)
	lambda_fluid = -(2.0 / 3.0) * viscosity_mu
	viscous_stress = 2.0 * viscosity_mu * D_tensor + lambda_fluid * div_v * jnp.eye(3)
	total_cauchy = -pressure * jnp.eye(3) + viscous_stress
	P_liquid = J_vol * total_cauchy @ jnp.linalg.inv(F_eval).T
		
	# ----------------------------------------------------------------------
	# PHASE C: AMBIENT AIR BUFFER MATRIX (Compliant Elastic Solid)
	# ----------------------------------------------------------------------
	mu_air, lambda_air = 1e-4, 1e-3
	strain_air = 0.5 * (disp_grad_eval + disp_grad_eval.T)
	P_air = 2.0 * mu_air * strain_air + lambda_air * jnp.trace(strain_air) * jnp.eye(3)

	# ----------------------------------------------------------------------
	# MATHEMATICAL ROUTING BLOCK (Replaces Python if/elif)
	# ----------------------------------------------------------------------
	# We use nested jnp.where statements to choose the right tensor per element.
	# jnp.where takes a boolean array condition and merges the arrays.

	is_solid = (mat_id == 101.0)
	is_liquid = (mat_id == 400.0)

	# If solid, take P_solid. If not, check if liquid. If not liquid, default to air.
	P_stress_final = jnp.where(
		is_solid, 
		P_solid, 
		jnp.where(is_liquid, P_liquid, P_air)
	)

	return P_stress_final
	# return P_air

class Substep1JacobianOperatorJAX(splinalg.LinearOperator):
	def __init__(self, num_dofs_int, dtype, u_global, v_global, nodes_global, topology, properties, fixed_dofs, dt1, M_diag):
		clean_dof = int(num_dofs_int)
		super().__init__(np.dtype(dtype), (clean_dof, clean_dof))
		
		self.u_global = jnp.array(u_global).reshape(-1, 3)
		self.v_global = jnp.array(v_global).reshape(-1, 3)
		self.nodes_global = jnp.array(nodes_global)
		self.topology = jnp.array(topology)
		self.properties = jnp.array(properties)
		self.fixed_dofs = fixed_dofs
		self.dt1 = dt1
		self.M_diag = np.array(M_diag).ravel()

	def _matvec(self, p):
		p_constrained = p.copy().ravel()
		p_constrained[self.fixed_dofs] = 0.0
		p_nodes = jnp.array(p_constrained).reshape(-1, 3)
		
		# Exact Position Chain Rule Multiplier for Trapezoidal Rule
		dx_dv_scale = self.dt1 / 2.0

		u_batched = self.u_global[self.topology]
		v_batched = self.v_global[self.topology]
		coords_batched = self.nodes_global[self.topology]
		p_batched = p_nodes[self.topology]

		mat_ids = jnp.where(self.properties == 1, 101.0, 202.0)
		Es = jnp.where(self.properties == 1, 10.0, 1.0)
		nus = jnp.where(self.properties == 1, 0.45, 0.001)

		def batch_force_vs_disp(u_b):
			return vmapped_forces_engine(u_b, v_batched, coords_batched, mat_ids, Es, nus, self.dt1)
		def batch_force_vs_vel(v_b):
			return vmapped_forces_engine(u_batched, v_b, coords_batched, mat_ids, Es, nus, self.dt1)

		# Evaluate directional derivatives using JAX JVP engine
		_, dF_du = jax.jvp(batch_force_vs_disp, (u_batched,), (p_batched * dx_dv_scale,))
		_, dF_dv = jax.jvp(batch_force_vs_vel, (v_batched,), (p_batched,))
		
		# Chain rule contribution to the force Jacobian
		q_batched_tet = np.array(dF_du + dF_dv)

		q_stiffness_global = np.zeros((len(self.u_global), 3), dtype=np.float64)
		for t_idx, tet in enumerate(np.array(self.topology)):
			q_stiffness_global[tet] += q_batched_tet[t_idx]
			
		q_stiffness_flat = q_stiffness_global.ravel()
		
		res = self.M_diag * p_constrained - (self.dt1 / 2.0) * q_stiffness_flat
		res[self.fixed_dofs] = 0.0
		return res

class Substep2JacobianOperatorJAX(splinalg.LinearOperator):
	def __init__(self, num_dofs_int, dtype, u_global, v_global, nodes_global, topology, properties, fixed_dofs, dt, gamma, M_diag):
		clean_dof = int(num_dofs_int)
		super().__init__(np.dtype(dtype), (clean_dof, clean_dof))
		
		self.u_global = jnp.array(u_global).reshape(-1, 3)
		self.v_global = jnp.array(v_global).reshape(-1, 3)
		self.nodes_global = jnp.array(nodes_global)
		self.topology = jnp.array(topology)
		self.properties = jnp.array(properties)
		self.fixed_dofs = fixed_dofs
		self.dt = dt
		self.gamma = gamma
		self.M_diag = np.array(M_diag).ravel()

	def _matvec(self, p):
		p_constrained = p.copy().ravel()
		p_constrained[self.fixed_dofs] = 0.0
		p_nodes = jnp.array(p_constrained).reshape(-1, 3)
		
		# Calculate matching constants for literature-standard BDF2 step split
		dt1 = self.gamma * self.dt
		dt2 = (1.0 - self.gamma) * self.dt
		alpha_bdf = (2.0 - self.gamma) / (1.0 + self.gamma)
		beta_bdf = (1.0 - self.gamma) / (1.0 + self.gamma)
		
		# Exact position chain rule multiplier: dx/dv = (dt2 * beta_bdf) / alpha_bdf
		dx_dv_scale = (dt2 * beta_bdf) / alpha_bdf
		# dx_dv_scale = self.dt / 2.0
		# dx_dv_scale = (self.dt * (2.0 - self.gamma)) / 2.0

		u_batched = self.u_global[self.topology]
		v_batched = self.v_global[self.topology]
		coords_batched = self.nodes_global[self.topology]
		p_batched = p_nodes[self.topology]

		mat_ids = jnp.where(self.properties == 1, 101.0, 202.0)
		Es = jnp.where(self.properties == 1, 10.0, 1.0)
		nus = jnp.where(self.properties == 1, 0.45, 0.001)

		def batch_force_vs_disp(u_b):
			return vmapped_forces_engine(u_b, v_batched, coords_batched, mat_ids, Es, nus, dt2)
		def batch_force_vs_vel(v_b):
			return vmapped_forces_engine(u_batched, v_b, coords_batched, mat_ids, Es, nus, dt2)

		_, dF_du = jax.jvp(batch_force_vs_disp, (u_batched,), (p_batched * dx_dv_scale,))
		_, dF_dv = jax.jvp(batch_force_vs_vel, (v_batched,), (p_batched,))
		
		q_batched_tet = np.array(dF_du + dF_dv)

		q_stiffness_global = np.zeros((len(self.u_global), 3), dtype=np.float64)
		for t_idx, tet in enumerate(np.array(self.topology)):
			q_stiffness_global[tet] += q_batched_tet[t_idx]
			
		q_stiffness_flat = q_stiffness_global.ravel()
		
		res = alpha_bdf * (self.M_diag * p_constrained) - (dt2 * beta_bdf) * q_stiffness_flat 

		res[self.fixed_dofs] = 0.0
		return res

		# q_stiffness_flat[self.fixed_dofs] = 0.0
		# return self.M_diag * p.ravel() - q_stiffness_flat

class Substep1JacobianOperatorJAX0(splinalg.LinearOperator):
	def __init__(self, num_dofs_int, dtype, u_global, v_global, nodes_global, topology, properties, fixed_dofs, dt1, M_diag):
		clean_dof = int(num_dofs_int)
		super().__init__(np.dtype(dtype), (clean_dof, clean_dof))
		
		self.u_global = jnp.array(u_global).reshape(-1, 3)
		self.v_global = jnp.array(v_global).reshape(-1, 3)
		self.nodes_global = jnp.array(nodes_global)
		self.topology = jnp.array(topology)
		self.properties = jnp.array(properties)
		self.fixed_dofs = fixed_dofs
		self.dt1 = dt1
		self.M_diag = np.array(M_diag).ravel()

	def _matvec(self, p):
		p_constrained = p.copy().ravel()
		p_constrained[self.fixed_dofs] = 0.0
		p_nodes = jnp.array(p_constrained).reshape(-1, 3)
		time_scale = self.dt1 / 2.0

		u_batched = self.u_global[self.topology]
		v_batched = self.v_global[self.topology]
		coords_batched = self.nodes_global[self.topology]
		p_batched = p_nodes[self.topology]

		mat_ids = jnp.where(self.properties == 1, 101.0, 202.0)
		Es = jnp.where(self.properties == 1, 10.0, 1.0)
		nus = jnp.where(self.properties == 1, 0.45, 0.001)

		def batch_force_vs_disp(u_b):
			return vmapped_forces_engine(u_b, v_batched, coords_batched, mat_ids, Es, nus, self.dt1)
		def batch_force_vs_vel(v_b):
			return vmapped_forces_engine(u_batched, v_b, coords_batched, mat_ids, Es, nus, self.dt1)

		_, dF_du = jax.jvp(batch_force_vs_disp, (u_batched,), (p_batched * time_scale,))
		_, dF_dv = jax.jvp(batch_force_vs_vel, (v_batched,), (p_batched,))
		
		q_batched_tet = np.array(dF_du + dF_dv)

		# Assemble elements using a clean NumPy loop to avoid JAX scatter exceptions
		q_stiffness_global = np.zeros((len(self.u_global), 3), dtype=np.float64)
		for t_idx, tet in enumerate(np.array(self.topology)):
			q_stiffness_global[tet] += q_batched_tet[t_idx]
			
		q_stiffness_flat = q_stiffness_global.ravel()
		q_stiffness_flat[self.fixed_dofs] = 0.0
		
		# Subtraction perfectly balances the negative gradient direction of R
		return self.M_diag * p.ravel() - q_stiffness_flat

class Substep2JacobianOperatorJAX0(splinalg.LinearOperator):
	def __init__(self, num_dofs_int, dtype, u_global, v_global, nodes_global, topology, properties, fixed_dofs, dt, gamma, M_diag):
		clean_dof = int(num_dofs_int)
		super().__init__(np.dtype(dtype), (clean_dof, clean_dof))
		
		self.u_global = jnp.array(u_global).reshape(-1, 3)
		self.v_global = jnp.array(v_global).reshape(-1, 3)
		self.nodes_global = jnp.array(nodes_global)
		self.topology = jnp.array(topology)
		self.properties = jnp.array(properties)
		self.fixed_dofs = fixed_dofs
		self.dt = dt
		self.gamma = gamma
		self.M_diag = np.array(M_diag).ravel()

	def _matvec(self, p):
		p_constrained = p.copy().ravel()
		p_constrained[self.fixed_dofs] = 0.0
		p_nodes = jnp.array(p_constrained).reshape(-1, 3)
		time_scale = (self.dt * (2.0 - self.gamma)) / 2.0

		u_batched = self.u_global[self.topology]
		v_batched = self.v_global[self.topology]
		coords_batched = self.nodes_global[self.topology]
		p_batched = p_nodes[self.topology]

		mat_ids = jnp.where(self.properties == 1, 101.0, 202.0)
		Es = jnp.where(self.properties == 1, 10.0, 1.0)
		nus = jnp.where(self.properties == 1, 0.45, 0.001)

		def batch_force_vs_disp(u_b):
			return vmapped_forces_engine(u_b, v_batched, coords_batched, mat_ids, Es, nus, time_scale)
		def batch_force_vs_vel(v_b):
			return vmapped_forces_engine(u_batched, v_b, coords_batched, mat_ids, Es, nus, time_scale)

		_, dF_du = jax.jvp(batch_force_vs_disp, (u_batched,), (p_batched * time_scale,))
		_, dF_dv = jax.jvp(batch_force_vs_vel, (v_batched,), (p_batched,))
		
		q_batched_tet = np.array(dF_du + dF_dv)

		q_stiffness_global = np.zeros((len(self.u_global), 3), dtype=np.float64)
		for t_idx, tet in enumerate(np.array(self.topology)):
			q_stiffness_global[tet] += q_batched_tet[t_idx]
			
		q_stiffness_flat = q_stiffness_global.ravel()
		q_stiffness_flat[self.fixed_dofs] = 0.0
		
		return self.M_diag * p.ravel() - q_stiffness_flat

class myEquation_dFEM:
	def __init__(self):
		super(myEquation_dFEM, self).__init__()

		self.callBack_cg_counter = 0

	def tanh(self, x):
		y = jnp.exp(-2.0 * x)
		return (1.0 - y) / (1.0 + y)

	# 2. IMPLICIT MULTI-PHASE B-REP VIA SDF FUNCTIONS
	def sdf_box(self, p, center, size):
		d = np.abs(p - center) - size
		return np.max(d, axis=-1) + np.min(np.maximum(d, 0.0), axis=-1)

	def sdf_sphere(self, p, center, radius):
		return np.linalg.norm(p - center, axis=-1) - radius

	def join_min(self, sdf1, sdf2):
		"""Sharp Union (Standard Minimum)"""
		return np.minimum(sdf1, sdf2)

	def join_smooth_min(self, sdf1, sdf2, k=0.3):
		"""Smooth Union (Polynomial Smooth Minimum)"""
		# Soft blending operation that smoothly transitions between values
		h = np.clip(0.5 + 0.5 * (sdf2 - sdf1) / k, 0.0, 1.0)
		return np.minimum(sdf1, sdf2) - k * h * (1.0 - h)

	def joined_sdf(self, p):
		"""Union of Sphere and Cube via min operator."""
		# sdf_A = self.sdf_sphere(p, center=np.array([1, 1, 0.0]), radius=.5) ####
		sdf_A = self.sdf_sphere(p, center=np.array([1, 1, 1]), radius=.5)

		# sdf_B = self.sdf_box(p, center=np.array([-1, -1, -1]), size=np.array([1, 1, 1]))
		sdf_B = self.sdf_box(p, center=np.array([0, 0, 0]), size=np.array([1, 1, 1]))

		# Sphere centered at origin, Cube slightly shifted to create a junction
		# s = sphere_sdf(p, radius=1.0, center=(0.0, 0.0, 0.0))
		# c = cube_sdf(p, side=1.2, center=(0.5, 0.5, 0.0))
		# return np.minimum(s, c)
		return np.minimum(sdf_A, sdf_B)

	def subtract_boolean_sdf(self, sdf1, sdf2):
		return np.maximum(sdf1, -sdf2)

	def sdf_vdb_visualizer(self, layer_data):
		grid_list = []

		#########################
		base_dir = bpy.path.abspath("E:/projects_3d/ABJ_Shader_Debugger_for_Blender/scenes/compositing_files/vdb/")

		# THE UNIQUE FILENAME FIX: Scan directory and find the next increment number
		# This forces Blender to register a brand new data block every time you click "Run"
		version = 1
		while os.path.exists(os.path.join(base_dir, f"base_volume_v{version}.vdb")):
			version += 1
		output_path = os.path.join(base_dir, f"base_volume_v{version}.vdb").replace("\\", "/")
		#############################

		for name, array in layer_data:
			resolution = array.shape[0]
			min_bound, max_bound = -2.0, 2.0
			voxel_size2 = (max_bound - min_bound) / (resolution - 1)

			transform_matrix = [
			[0.0, 0.0, voxel_size2, 0.0],
			[0.0, voxel_size2, 0.0, 0.0],
			[voxel_size2, 0.0, 0.0, 0.0],
			[min_bound,  min_bound,  min_bound,1.0]
			]

			g = vdb.FloatGrid()
			g.copyFromArray(np.asfortranarray(array))
			g.name = name
			g.transform = vdb.createLinearTransform(matrix=transform_matrix)
			grid_list.append(g)

		vdb.write(output_path, grids=grid_list)
		###############################################
		# --- CONFIGURATION ---
		vdb_path = output_path
		obj_name = "abj_test_000_Imported_SDF_Volume"

		# grid_name_in_vdb = "core_sdf"
		grid_name_in_vdb = "sdf_joined"
		# grid_name_in_vdb = "muscle_sdf"
		# grid_name_in_vdb = "skin_sdf"

		# 1. IMPORT THE VDB AS A VOLUME OBJECT
		bpy.ops.object.volume_import(filepath=vdb_path, align='WORLD')
		volume_obj = bpy.context.active_object
		volume_obj.name = obj_name

		# 2. CREATE A GEOMETRY NODES MODIFIER
		bpy.ops.object.modifier_add(type='NODES')
		modifier = volume_obj.modifiers[-1]
		modifier.name = "SDF_To_Mesh"

		# 3. SET UP THE GEOMETRY NODE TREE
		node_group = bpy.data.node_groups.new(name="SDF_Triangulation", type="GeometryNodeTree")
		modifier.node_group = node_group
		node_group.nodes.clear()

		# 4. INSTANTIATE THE NECESSARY NODES
		# Group Input
		node_input = node_group.nodes.new(type="NodeGroupInput")
		node_input.location = (-300, 0)

		# NEW: The Get Named Grid Node (Extracts the grid data from the volume geometry)
		node_get_grid = node_group.nodes.new(type="GeometryNodeGetNamedGrid")
		node_get_grid.location = (-50, 0)
		node_get_grid.inputs['Name'].default_value = grid_name_in_vdb

		# The Grid to Mesh Node
		node_grid_to_mesh = node_group.nodes.new(type="GeometryNodeGridToMesh")
		node_grid_to_mesh.location = (200, 0)
		node_grid_to_mesh.inputs['Threshold'].default_value = 0
		node_grid_to_mesh.inputs['Adaptivity'].default_value = 0.0 

		# Group Output
		node_output = node_group.nodes.new(type="NodeGroupOutput")
		node_output.location = (450, 0)

		# 5. INITIALIZE INTERFACE SOCKETS
		node_group.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
		node_group.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')

		# 6. WIRE THE NODES TOGETHER CORRECTLY
		links = node_group.links

		# Link 1: Connect generic Volume Geometry into the "Get Named Grid" node
		links.new(node_input.outputs['Geometry'], node_get_grid.inputs['Volume'])

		# Link 2: Connect the extracted Voxel Grid data into the "Grid to Mesh" node
		links.new(node_get_grid.outputs['Grid'], node_grid_to_mesh.inputs['Grid'])

		# Link 3: Connect generated polygonal Mesh to final Output
		links.new(node_grid_to_mesh.outputs['Mesh'], node_output.inputs['Geometry'])

		print("Successfully linked Volume -> Get Named Grid -> Grid to Mesh!")
		
		myMeshObj = self.bakeVDB_multi(0, vdb_path, volume_obj) ##############################

		return myMeshObj

	def bakeVDB_multi(self, threshold, path, volume_obj):
		# --- CONFIGURATION ---
		# vdb_path = bpy.path.abspath("//compositing_files/sphere_sdf.vdb")
		# vdb_path = 'E:/projects_3d/ABJ_Shader_Debugger_for_Blender/scenes/compositing_files/output_volume.vdb'
		vdb_path = path
		# grid_name = "surface_sdf"
		grid_name = "sdf_joined"

		# 1. Create a temporary hidden volume block to read from disk
		bpy.ops.object.volume_import(filepath=vdb_path, align='WORLD')
		temp_vol_obj = bpy.context.active_object
		temp_vol_obj.name = "TEMP_VOLUME_DATA"
		temp_vol_obj.hide_viewport = True
		temp_vol_obj.hide_render = True

		# 2. Create the true physical Target Mesh Container
		mesh_data = bpy.data.meshes.new("SDF_Polygons")
		mesh_obj = bpy.data.objects.new("SDF_Mesh_Final", mesh_data)
		bpy.context.collection.objects.link(mesh_obj)

		# Ensure the mesh container is active
		bpy.context.view_layer.objects.active = mesh_obj
		mesh_obj.select_set(True)

		# 3. Initialize Geometry Nodes on the MESH container
		modifier = mesh_obj.modifiers.new(name="SDF_Convert", type='NODES')
		node_group = bpy.data.node_groups.new(name="SDF_Tree", type="GeometryNodeTree")
		modifier.node_group = node_group
		node_group.nodes.clear()

		# Initialize interface geometry ports
		node_group.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
		node_group.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')

		# 4. Build the internal node pipeline inside the mesh container
		n_input = node_group.nodes.new(type="NodeGroupInput")
		n_input.location = (-400, 0)

		# Pull geometry from our temporary hidden volume object
		n_obj_info = node_group.nodes.new(type="GeometryNodeObjectInfo")
		n_obj_info.inputs['Object'].default_value = temp_vol_obj
		n_obj_info.location = (-200, 0)

		n_get_grid = node_group.nodes.new(type="GeometryNodeGetNamedGrid")
		n_get_grid.inputs['Name'].default_value = grid_name
		n_get_grid.location = (0, 0)

		n_grid_to_mesh = node_group.nodes.new(type="GeometryNodeGridToMesh")
		n_grid_to_mesh.inputs['Threshold'].default_value = threshold
		n_grid_to_mesh.location = (200, 0)

		n_output = node_group.nodes.new(type="NodeGroupOutput")
		n_output.location = (400, 0)

		# Link everything together
		links = node_group.links
		links.new(n_obj_info.outputs['Geometry'], n_get_grid.inputs['Volume'])
		links.new(n_get_grid.outputs['Grid'], n_grid_to_mesh.inputs['Grid'])
		links.new(n_grid_to_mesh.outputs['Mesh'], n_output.inputs['Geometry'])

		# 5. THE FIX: Apply the modifier on the Mesh Object
		# This forces Blender to calculate the math and collapse it into permanent vertices
		bpy.ops.object.modifier_apply(modifier="SDF_Convert")

		# 6. Housekeeping: Delete the temporary volume file container from the scene
		bpy.data.objects.remove(temp_vol_obj, do_unlink=True)

		bpy.data.objects.remove(volume_obj, do_unlink=True)

		print("Modifier successfully applied! Your object is now a raw, pure polygon mesh.")

		return mesh_obj

	def classify_tet_phase(self, tet_center_coords, core_sdf_grid, muscle_sdf_grid, skin_sdf_grid):
		"""
		Given a list of tet centroid positions (N, 3), query your internal 
		SDF grid arrays to compute the definitive structural material phase.
		"""
		# 1. Sample your continuous array metrics using point coordinates
		# (Assuming simple voxel indexing translation or trilinear sampling)
		
		# Remember our generated math arrays: Negative is OUTSIDE, Positive is INSIDE
		is_inside_core   = core_sdf_grid   >= 0.0
		is_inside_muscle = muscle_sdf_grid >= 0.0
		is_inside_skin   = skin_sdf_grid   >= 0.0
		
		# 2. Sequential Phase masking array initialization (Default to Air/Liquid 0)
		phase_map = np.zeros(len(tet_center_coords), dtype=np.int32)
		
		# Outer layer down to inner layer masking override
		phase_map[is_inside_skin]   = 1  # Element is inside Skin
		phase_map[is_inside_muscle] = 2  # Element is inside Muscle (Overwrites skin)
		phase_map[is_inside_core]   = 3  # Element is inside Core/Bone (Overwrites muscle)
		
		return phase_map

	def material_texture_generation_stride(self, tags):
		pass

		return
	
		"""
		Transforms clean material array data into an execution format structured 
		for 32-bit pixel mapping buffers inside your scripted node generator.
		"""
		packed_pixels = np.zeros((len(tags), 4), dtype=np.float32)
		for idx, tag in enumerate(tags):
			if tag == 101:
				# [Material_ID, Gravity_On, Element_Mass, 0.0] -> Solid responds to global gravity
				packed_pixels[idx] = [101.0, 1.0, 1.0, 0.0]
			else:
				# [Material_ID, Gravity_On, Element_Mass, 0.0] -> Air ignores gravity, tracks pressure
				packed_pixels[idx] = [202.0, 0.0, 0.001, 0.0]
		return packed_pixels

	def voxelChecker0(self):
		#i want to check if a 3d point which is part of a voxel is inside the SDF. How is this possible?

		# To check if a 3D point is inside your Signed Distance Field (SDF) sphere, evaluate the equation at that point and check if the resulting value is less than or equal to zero.In an SDF, a negative value means the point is inside the surface, zero means it is exactly on the surface, and a positive value means it is outside.

		# Define sphere parameters
		center = np.array([0.0, 0.0, 0.0])
		radius = 2.0

		# 1. Checking a single voxel point
		p_single = np.array([1.0, 0.0, 1.0])
		sdf_value = np.linalg.norm(p_single - center) - radius
		is_inside_single = sdf_value <= 0

		print(f"SDF Value: {sdf_value}, Is inside: {is_inside_single}")

		# 2. Checking an array of multiple voxel points at once
		# Assume shape (N, 3) where N is the number of voxel points
		p_voxels = np.array([
			[0.0, 0.0, 0.0],  # Inside (at the center)
			[2.0, 0.0, 0.0],  # On the surface
			[3.0, 3.0, 3.0]   # Outside
		])

		# Vectorized SDF evaluation
		sdf_values = np.linalg.norm(p_voxels - center, axis=-1) - radius

		# Boolean mask: True if inside or on the surface
		is_inside_mask = sdf_values <= 0

		print("SDF Values:", sdf_values)
		print("Is inside mask:", is_inside_mask)


	def visualize_tet_mesh(self, unique_verts, tets, mesh_name="Tet_Debug_Mesh"):
		"""
		Extracts the outer boundary faces from a volumetric tetrahedral mesh
		and creates a Blender 5.2 Mesh Object using high-performance foreach_set.
		
		Parameters:
			unique_verts (np.ndarray): Shape (N, 3), float32 array of vertex coordinates.
			tets (np.ndarray): Shape (M, 4), int32 array of vertex indices forming tetrahedra.
			mesh_name (str): The name assigned to the generated Blender object.
		"""
		# --- Step 1: Extract Boundary Faces from Tetrahedra ---
		# Define the 4 local faces for every tetrahedron
		# Ordered structurally to preserve consistent face normals
		local_faces = np.array([
			[0, 1, 2],
			[0, 2, 3],
			[0, 3, 1],
			[1, 3, 2]
		], dtype=np.int32)
		
		# Map all M tetrahedra across the 4 local faces -> Shape: (M * 4, 3)
		all_faces = tets[:, local_faces].reshape(-1, 3)
		
		# Sort indices per face row-wise so orientation variation won't break matches
		sorted_faces = np.sort(all_faces, axis=1)
		
		# Find unique rows and counts. 
		# Internal faces are shared by exactly 2 tets (count == 2).
		# Boundary faces belong to only 1 tet (count == 1).
		_, indices, counts = np.unique(sorted_faces, axis=0, return_index=True, return_counts=True)
		boundary_face_indices = indices[counts == 1]
		
		# Extract original oriented faces belonging to the boundary
		boundary_faces = all_faces[boundary_face_indices]
		
		# --- Step 2: Initialize Blender 5.2 Mesh Container ---
		# Delete any existing object with the same name to keep the scene clean
		if mesh_name in bpy.data.objects:
			bpy.data.objects.remove(bpy.data.objects[mesh_name], do_unlink=True)
			
		mesh_data = bpy.data.meshes.new(mesh_name + "_Data")
		mesh_obj = bpy.data.objects.new(mesh_name, mesh_data)
		
		# Link the new object to the active collection
		bpy.context.collection.objects.link(mesh_obj)
		
		# --- Step 3: Fast Buffer Copy via foreach_set ---
		num_verts = unique_verts.shape[0]
		num_faces = boundary_faces.shape[0]
		
		# Pre-allocate spaces inside the Blender geometry data block
		mesh_data.vertices.add(num_verts)
		mesh_data.polygons.add(num_faces)
		
		# Blender requires flat 1D contiguous arrays for foreach_set input buffers
		flat_verts = unique_verts.astype(np.float32).ravel()
		
		# Pushing vertex coordinates
		mesh_data.vertices.foreach_set("co", flat_verts)
		
		# Calculate loops: loop_start indicates index offset, loop_total is 3 for triangles
		loop_start = np.arange(0, num_faces * 3, 3, dtype=np.int32)
		loop_total = np.full(num_faces, 3, dtype=np.int32)
		
		# Flatten the boundary faces for the loop total mapping
		flat_faces = boundary_faces.astype(np.int32).ravel()
		
		# Pre-allocate loop structures
		mesh_data.loops.add(num_faces * 3)
		
		# Flush structural topological buffers into the mesh block
		mesh_data.polygons.foreach_set("loop_start", loop_start)
		mesh_data.polygons.foreach_set("loop_total", loop_total)
		mesh_data.loops.foreach_set("vertex_index", flat_faces)
		
		# Update geometry topology and compute boundary normals
		mesh_data.update()
		mesh_data.validate()
		
		return mesh_obj


	def compute_global_lumped_mass(self, nodes_tet10, topology_tet10, element_properties):
		"""
		Computes a physically accurate global Lumped Mass diagonal vector (3 DOFs per node)
		by integrating density across high-order Tet10 shape functions via Gauss Quadrature.
		"""
		num_nodes = len(nodes_tet10)
		# Initialize global mass accumulator per node
		global_mass_per_node = np.zeros(num_nodes, dtype=np.float64)

		# 4-Point Gauss Quadrature Constants
		a, b = 0.5854101966249685, 0.1381966011250105
		gauss_points = np.array([[a,b,b], [b,a,b], [b,b,a], [b,b,b]])
		gauss_weight = 1.0 / 24.0

		for t_idx, tet in enumerate(topology_tet10):
			# 1. Map phase property definitions to find local element density
			# Air (0) -> 0.001 | Solid (1) -> 5.0
			density = 5.0 if element_properties[t_idx] == 1 else 0.001
			
			element_node_coords = nodes_tet10[tet]
			
			# 2. Integrate mass matrix row-sums across Gauss points
			for gp in gauss_points:
				r, s, t = gp[0], gp[1], gp[2]
				u = 1.0 - r - s - t
				
				# Evaluate Tet10 Shape Functions (N) at this Gauss Point
				# Order: 4 Corners [0-3], 6 Mid-side Nodes [4-9]
				N = np.array([
					u * (2.0 * u - 1.0),  # Corner 0
					r * (2.0 * r - 1.0),  # Corner 1
					s * (2.0 * s - 1.0),  # Corner 2
					t * (2.0 * t - 1.0),  # Corner 3
					4.0 * u * r,          # Mid 4 (0-1)
					4.0 * r * s,          # Mid 5 (1-2)
					4.0 * u * s,          # Mid 6 (2-0)
					4.0 * u * t,          # Mid 7 (0-3)
					4.0 * r * t,          # Mid 8 (1-3)
					4.0 * s * t           # Mid 9 (2-3)
				])
				
				# Compute Shape Function Derivatives to find the physical volume element
				dN_dr = np.array([-4*u+1, 4*r-1, 0, 0, 4*u-4*r, 4*s, -4*s, -4*t, 4*t, 0])
				dN_ds = np.array([-4*u+1, 0, 4*s-1, 0, -4*r, 4*r, 4*u-4*s, -4*t, 0, 4*t])
				dN_dt = np.array([-4*u+1, 0, 0, 4*t-1, -4*r, 0, -4*s, 4*u-4*t, 4*r, 4*s])
				dN_dxi = np.stack([dN_dr, dN_ds, dN_dt], axis=0)
				
				Jacobian = np.dot(dN_dxi, element_node_coords)
				det_J = np.linalg.det(Jacobian)
				dV = det_J * gauss_weight
				
				# Physical mass distribution = density * Shape_Function * Volume_Element
				# Row-sum lumping maps N_i into local diagonal slots directly
				local_lumped_mass = density * N * dV
				
				# Scatter back to global node trackers
				global_mass_per_node[tet] += local_lumped_mass

		# 3. Expand the nodal mass scalar array to a 3-DOF vector (X, Y, Z allocation)
		global_lumped_mass_vector = np.zeros(num_nodes * 3, dtype=np.float64)
		for node_idx in range(num_nodes):
			m = global_mass_per_node[node_idx]
			start = node_idx * 3
			global_lumped_mass_vector[start : start + 3] = m
			
		return global_lumped_mass_vector

	def run_tr_bdf2_time_step(self, nodes_tet10, topology_tet10, element_properties, x_t, v_t, F_ext, dt, tol=1e-5, max_newton_iter=5):
		'''
		
		TR-BDF2 References
		https://www.sciencedirect.com/science/article/pii/S0898122121001267
		https://en.wikipedia.org/wiki/Backward_differentiation_formula
		https://en.wikipedia.org/wiki/Trapezoidal_rule

		#Vectorized, JIT-Accelerated TR-BDF2 Step Loop
		Executes one full second-order accurate, energy-preserving TR-BDF2 time step
		for your high-order Tet10 multi-phase system natively on the CPU.
		
		Args:
			x_t: (N, 3) current 64-bit node positions at time t.
			v_t: (N, 3) current 64-bit node velocities at time t.
			F_ext: (3*N,) flat global external force vector (gravity/loads).
			dt: Time increment step (e.g., 0.01 seconds).
			
		Returns:
			x_next, v_next: Updated (N, 3) position and velocity arrays at time t + dt.
		'''

		num_nodes = len(nodes_tet10)
		dof = num_nodes * 3

		nodes_jax = jnp.array(nodes_tet10)
		topology_jax = jnp.array(topology_tet10)
		properties_jax = jnp.array(element_properties)
		f_ext_jax = jnp.array(F_ext).ravel()

		fixed_node_indices = np.where(nodes_tet10[:, 2] <= 0.001)[0]
		fixed_dofs = []
		for node_idx in fixed_node_indices:
			fixed_dofs.extend([node_idx*3, node_idx*3+1, node_idx*3+2])
		fixed_dofs = np.array(fixed_dofs, dtype=np.int32)
		fixed_dofs_jax = jnp.array(fixed_dofs)

		# M_diag = np.ones(dof, dtype=np.float64) * 0.1
		# M_diag[fixed_dofs] = 1.0 
		# M_diag_jax = jnp.array(M_diag)

		# ==========================================================================
		# UPGRADED PROPERTY: TRUE INTEGRATED LUMPED MASS VECTOR
		# ==========================================================================
		# Computes consistent inertia tracking arrays matching your exact mesh shapes
		M_diag = self.compute_global_lumped_mass(nodes_tet10, topology_tet10, element_properties)
		
		# Prevent divide-by-zero or numerical collapse on locked constraint boundary rows
		M_diag[fixed_dofs] = np.maximum(M_diag[fixed_dofs], 1.0)
		
		# Cast to JAX array for internal hardware operations
		M_diag_jax = jnp.array(M_diag, dtype=jnp.float64)

		progress_callback = PercentageCallback(tol=tol)

		gamma = 2.0 - np.sqrt(2.0)
		dt1 = gamma * dt 

		# 2. INITIAL FORCE POINTER: Computed using the new JIT function
		u_t_jax = jnp.array(x_t - nodes_tet10)
		v_t_jax = jnp.array(v_t)
		f_int_t = np.array(assemble_global_forces_jax(u_t_jax, v_t_jax, nodes_jax, topology_jax, properties_jax, fixed_dofs_jax, dt1))

		# ==========================================================================
		# SUBSTEP 1: TRAPEZOIDAL STEP
		# ==========================================================================
		x_gamma = x_t.copy()
		v_gamma = v_t.copy()

		x_gamma0 = x_t.copy()
		v_gamma0 = v_t.copy()

		# return x_gamma0, v_gamma0

		for n_iter in range(max_newton_iter):
			u_gamma_jax = jnp.array(x_gamma - nodes_tet10)
			v_gamma_jax = jnp.array(v_gamma)

			f_int_gamma = np.array(assemble_global_forces_jax(u_gamma_jax, v_gamma_jax, nodes_jax, topology_jax, properties_jax, fixed_dofs_jax, dt1))

			#OLD
			R = M_diag * (v_gamma.ravel() - v_t.ravel()) - (dt1 / 2.0) * (F_ext.ravel() + 2.0 * F_ext.ravel()) - (dt1 / 2.0) * (f_int_t + f_int_gamma) ##old
			R_pos = x_gamma.ravel() - x_t.ravel() - (dt1 / 2.0) * (v_t.ravel() + v_gamma.ravel())
			R_combined = R + M_diag * (R_pos / (dt1 / 2.0))
			# R_combined = R + M_diag * (R_pos / (dt / 2.0))
			R_combined[fixed_dofs] = 0.0

			if np.linalg.norm(R_combined) < tol:
				break

			# ####Pure Momentum Residual Vector (M * dv - dt/2 * sum(F))
			# R_combined = M_diag * (v_gamma.ravel() - v_t.ravel()) - (dt1 / 2.0) * (F_ext.ravel() + F_ext.ravel()) - (dt1 / 2.0) * (f_int_t + f_int_gamma)
			# R_combined[fixed_dofs] = 0.0

			# if np.linalg.norm(R_combined) < tol:
			# 	break

			J_op = Substep1JacobianOperatorJAX(
				num_dofs_int=len(R_combined),
				dtype=np.float64,
				u_global=x_gamma - nodes_tet10,
				v_global=v_gamma,
				nodes_global=nodes_tet10,
				topology=topology_tet10,
				properties=element_properties,
				fixed_dofs=fixed_dofs,
				dt1=dt1,
				# dt1=dt,
				M_diag=M_diag
			)

			######################
			## SOLVERS 1
			#######################
			## CG
			# delta_v_flat, info = splinalg.cg(J_op, -R_combined, x0=np.zeros_like(R_combined), rtol=tol, maxiter=200, callback=progress_callback)

			## GMRES
			# delta_v_flat, info = splinalg.gmres(J_op, -R_combined, restart=30, maxiter=100, callback=progress_callback)

			# BICGSTAB
			M_diag_safe = np.where(M_diag == 0, 1.0, M_diag)
			inv_M = 1.0 / M_diag_safe

			def jacobi_preconditioner(v):
				return inv_M * v
			M_precond = splinalg.LinearOperator(shape=J_op.shape, matvec=jacobi_preconditioner)
			
			delta_v_flat, info = splinalg.bicgstab(
				J_op, 
				-R_combined, 
				x0=np.zeros_like(R_combined), 
				# rtol=tol, 
				rtol=1e-6, 
				maxiter=25, 
				M=M_precond, # Activates the diagonal preconditioning channel
				callback=progress_callback
			)

			if info > 0:
				raise ValueError('Solver convergence stalled on Substep 1.')

			v_gamma += delta_v_flat.reshape(-1, 3)
			x_gamma += (delta_v_flat * (dt1 / 2.0)).reshape(-1, 3)

		# Finalize explicit position calculation for stage 1
		# x_gamma = x_t + (dt1 / 2.0) * (v_t + v_gamma) ########

		# x_gamma = x_gamma0 ########
		# v_gamma = v_gamma0 ########

		# return x_gamma, v_gamma
		# return x_gamma0, v_gamma0

		# ==========================================================================
		# SUBSTEP 2: BDF2 STEP
		# ==========================================================================
		##########
		#new 04 look
		##########

		dt2 = (1.0 - gamma) * dt
		d = dt2 / (dt1 + dt2)

		# alpha_bdf = (2.0 - gamma) / (1.0 + gamma)
		# beta_bdf = (1.0 - gamma) / (1.0 + gamma)

		alpha_bdf = (1.0 + 2.0 * d) / (1.0 + d)
		beta_bdf  = (1.0 - d) / (1.0 + d)  # Note: formulas adapt dynamically based on gamma

		time_scale = (dt * (2.0 - gamma)) / 2.0
		time_scale_bdf = dt2 * beta_bdf
		
		x_next = x_gamma.copy()
		v_next = v_gamma.copy()

		# return x_next, v_next

		x_next0 = x_gamma.copy()
		v_next0 = v_gamma.copy()

		# 2. Compute composite history states for BOTH velocity and position
		v_history_flat = (1.0 / (gamma * (2.0 - gamma))) * v_gamma.ravel() - (((1.0 - gamma)**2) / (gamma * (2.0 - gamma))) * v_t.ravel()
		x_history_flat = (1.0 / (gamma * (2.0 - gamma))) * x_gamma.ravel() - (((1.0 - gamma)**2) / (gamma * (2.0 - gamma))) * x_t.ravel()

		for n_iter in range(max_newton_iter):
			# Kinematically locked position mapping for BDF2 Rule
			x_next_flat = (x_history_flat + dt2 * beta_bdf * v_next.ravel()) / alpha_bdf
			x_next = x_next_flat.reshape(-1, 3)

			# 1. Compute forces dynamically using the jitted assembler
			f_int_next = np.array(
				assemble_global_forces_jax(
					jnp.array(x_next - nodes_tet10), jnp.array(v_next), 
					jnp.array(nodes_tet10), jnp.array(topology_tet10), 
					jnp.array(element_properties), jnp.array(fixed_dofs), dt2
				), 
				dtype=np.float64
			)

			# ## OLD
			R = M_diag * (v_next.ravel() - v_history_flat) - (dt * (2.0 - gamma) / 2.0) * (f_int_next + F_ext)
			R_pos = x_next.ravel() - (((1.0 - gamma)**2) / (gamma * (2.0 - gamma))) * x_t.ravel() 
			R_combined = R + M_diag * (R_pos / dt)
			R_combined[fixed_dofs] = 0.0
			if np.linalg.norm(R_combined) < tol:
				break

			## NEW
			# # Pure BDF2 Momentum Residual (M * (alpha * v - v_hist) - dt * beta * F)
			# R_combined = M_diag * (alpha_bdf * v_next.ravel() - v_history_flat) - dt2 * beta_bdf * F_ext.ravel() - dt2 * beta_bdf * f_int_next
			# R_combined[fixed_dofs] = 0.0

			# if np.linalg.norm(R_combined) < tol:
			# 	break

			J_op2 = Substep2JacobianOperatorJAX(
				num_dofs_int=len(R_combined),
				dtype=np.float64,
				u_global=x_next - nodes_tet10,
				v_global=v_next,
				nodes_global=nodes_tet10,
				topology=topology_tet10,
				properties=element_properties,
				fixed_dofs=fixed_dofs,
				dt=dt,
				# dt=dt1,
				# dt=dt2,
				gamma=gamma,
				M_diag=M_diag
			)
			
			######################
			## SOLVERS 2
			#######################
			##CG
			# delta_v_flat, info = splinalg.cg(J_op2, -R_combined, x0=np.zeros_like(R_combined), rtol=tol, maxiter=200, callback=progress_callback)

			## GMRES
			# delta_v_flat, info = splinalg.gmres(J_op2, -R_combined, restart=30, maxiter=100, callback=progress_callback)

			### BICGSTAB
			M_diag_safe = np.where(M_diag == 0, 1.0, M_diag)
			inv_M = 1.0 / M_diag_safe

			def jacobi_preconditioner(v):
				return inv_M * v

			M_precond2 = splinalg.LinearOperator(shape=J_op2.shape, matvec=jacobi_preconditioner)

			delta_v_flat, info = splinalg.bicgstab(
				J_op2, 
				-R_combined, 
				x0=np.zeros_like(R_combined), 
				rtol=tol, 
				# rtol=1e-6, 
				# maxiter=100, 
				# maxiter=50, 
				maxiter=25, 
				M=M_precond2, # Activates the diagonal preconditioning channel
				callback=progress_callback
			)

			if info > 0:
				raise ValueError('Solver convergence stalled on Substep 2.')

			v_next += delta_v_flat.reshape(-1, 3)
			x_next += (delta_v_flat * time_scale).reshape(-1, 3)

		### Final definitive kinematic synchronization before exporting to Blender frame buffer
		# x_next_flat = (x_history_flat + dt2 * beta_bdf * v_next.ravel()) / alpha_bdf
		# x_next = x_next_flat.reshape(-1, 3)

		# x_next = x_next0
		# x_next = x_history_flat.ravel()
		# v_next = v_next0

		return x_next, v_next
		# return x_next0, v_next
		# return x_next, v_next0

	def visualize_sliced_multiphase_mesh(self, unique_verts, tets, phase_tags, slice_axis=0, slice_val=0.0):
		"""
		Slices the generated multi-material solid lattice in half, pushing the 
		flat buffer arrays directly into Blender 5.2 polygons using foreach_set.
		"""
		# Calculate centroids and mask out one half of the simulation
		centroids = unique_verts[tets].mean(axis=1)
		visible_mask = centroids[:, slice_axis] < slice_val
		sliced_tets = tets[visible_mask]
		sliced_tags = phase_tags[visible_mask]
		
		# Extract unique exposed boundaries and cross-sectional faces
		local_faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int32)
		all_faces = sliced_tets[:, local_faces].reshape(-1, 3)
		
		sorted_faces = np.sort(all_faces, axis=1)
		_, indices, counts = np.unique(sorted_faces, axis=0, return_index=True, return_counts=True)
		
		# Boundary faces + sliced internal elements
		boundary_face_indices = indices[counts == 1]
		boundary_faces = all_faces[boundary_face_indices]
		
		# Back-trace faces to identify their parent element's material assignment
		face_to_tet_idx = boundary_face_indices // 4
		face_phases = sliced_tags[face_to_tet_idx]
		
		# Clean workspace up
		obj_name = "Multiphase_Lattice_Debug"
		if obj_name in bpy.data.objects:
			bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink=True)
			
		mesh_data = bpy.data.meshes.new(obj_name + "_Data")
		mesh_obj = bpy.data.objects.new(obj_name, mesh_data)
		bpy.context.collection.objects.link(mesh_obj)
		
		# Allocate storage blocks inside memory geometry data pools
		mesh_data.vertices.add(len(unique_verts))
		mesh_data.polygons.add(len(boundary_faces))
		mesh_data.loops.add(len(boundary_faces) * 3)
		
		# Rapid serialization and flush via C-backed loops
		mesh_data.vertices.foreach_set("co", unique_verts.astype(np.float32).ravel())
		mesh_data.polygons.foreach_set("loop_start", np.arange(0, len(boundary_faces) * 3, 3, dtype=np.int32))
		mesh_data.polygons.foreach_set("loop_total", np.full(len(boundary_faces), 3, dtype=np.int32))
		mesh_data.loops.foreach_set("vertex_index", boundary_faces.astype(np.int32).ravel())
		
		mesh_data.update()
		mesh_data.validate()
		
		# Define Distinct Materials for Viewport Phase Isolation
		material_colors = {
			0: (0.1, 0.1, 0.1, 1.0), # Default Air/Unassigned (Dark Slate)
			1: (0.85, 0.25, 0.25, 1.0), # Soft Tissue Sphere (Red)
			2: (0.25, 0.45, 0.85, 1.0)  # Rigid Collision Cube (Blue)
		}
		
		for phase_id, color in material_colors.items():
			mat = bpy.data.materials.new(name=f"Phase_{phase_id}_Material")
			mat.use_nodes = False
			mat.diffuse_color = color
			mesh_data.materials.append(mat)
			
		# Map individual polygon elements directly to phase indices
		mesh_data.polygons.foreach_set("material_index", face_phases.astype(np.int32))
		mesh_data.update()
		
		return mesh_obj

	def convert_tet4_lattice_to_tet10(self, nodes, tet4_indices):
		"""
		Upgrades a 4-node linear tetrahedral lattice into a high-precision,
		10-node quadratic element matrix system.
		"""

		# Defensive check: Verify the incoming nodes are truly float64
		assert nodes.dtype == np.float64, "Critical Error: Input nodes must be float64 precision."

		num_nodes = len(nodes)
		tet10_indices = []

		# Track unique edges to prevent creating duplicate midpoint nodes
		edge_to_midpoint_idx = {}
		new_midpoint_nodes = []

		# Local edge configuration mappings for a standard tetrahedron
		# Edge 0-1, 1-2, 2-0, 0-3, 1-3, 2-3
		# local_edges = [(0,1), (1,2), (2,0), (0,3), (1,3), (2,3)]
		local_edges = [(0,1), (1,2), (0,2), (0,3), (1,3), (2,3)]

		current_midpoint_counter = num_nodes

		for tet in tet4_indices:
			tet_10_entry = list(tet) # Start with the original 4 corner indices
			
			for le in local_edges:
				# Sort the edge nodes to ensure unique dictionary tracking hashes
				n_start, n_end = sorted([tet[le[0]], tet[le[1]]])
				edge_key = (n_start, n_end)
				
				if edge_key not in edge_to_midpoint_idx:
					# Calculate the exact geometric 3D midpoint coordinate
					mid_co = (nodes[n_start] + nodes[n_end]) * 0.5
					new_midpoint_nodes.append(mid_co)
					
					# Assign a new unique global index pointer
					edge_to_midpoint_idx[edge_key] = current_midpoint_counter
					current_midpoint_counter += 1
					
				tet_10_entry.append(edge_to_midpoint_idx[edge_key])
				
			tet10_indices.append(tet_10_entry)
			
		# Combine original corner nodes with your high-order midpoint vertices
		all_nodes_extended = np.vstack([nodes, np.array(new_midpoint_nodes)])

		'''
		DOCUMENTATION of returns

		1.. all_nodes_extended (The Nodes
		What it is: A 2D array of 3D geometric coordinates ((x, y, z)).
		Contents: It appends the newly calculated midpoint coordinates to the bottom of your original corner node coordinate list.
		Shape: (Total New Nodes, 3).
		Data Type: Floating-point numbers (float64).

		2. np.array(tet10_indices) (The Elements)
		What it is: A 2D connectivity matrix representing element topology.
		Contents: It does not contain any spatial coordinates. Instead, it contains integer pointers (indices) that map which 10 nodes from all_nodes_extended group together to form each quadratic tetrahedron element.
		Shape: (Number of Tetrahedrons, 10).
		Data Type: Integers (int).

		Direct Comparison

		_Feature_ all_nodes_extended
		_Concept_ Geometry / Locations
		_Data_ Type float64 (e.g., 0.531, 1.240, -0.442)
		_Row Meaning_ A single spatial point in 3D space.

		_Feature_ np.array(tet10_indices)
		_Concept_ Topology / Connectivity
		_Data_ Type int (e.g., 0, 1, 2, 3, 44, 45...)
		_Row Meaning_ A single 10-node tetrahedral element.
		'''

		return all_nodes_extended, np.array(tet10_indices)

	def generate_global_multiphase_mesh(self, resolution, resolution_hi):
		"""
		Meshes the entire simulation bounding box unconditionally, then maps
		tetrahedra to Air, Liquid, Soft Tissue, or Rigid Solid phases.
		"""
		# Define physical properties and separate positions
		# sphere_center = np.array([0.0, 0.0, 1.5], dtype=np.float32)
		# sphere_center = np.array([0.0, 0.0, 1], dtype=np.float64)
		sphere_center = np.array([0.0, 0.0, 1], dtype=np.float64)
		# sphere_radius = 0.8
		sphere_radius = 0.4
		cube_center = np.array([0.0, 0.0, -0.8], dtype=np.float64)
		# cube_size = 1.2
		cube_size = .5

		################
		####### VDB LO
		################
		# Establish simulation domain bounds
		min_bound, max_bound = -2.0, 2.0
		lin_coords = np.linspace(min_bound, max_bound, resolution, dtype=np.float64)
		X, Y, Z = np.meshgrid(lin_coords, lin_coords, lin_coords, indexing='ij')

		# --- Pipeline A: 3D Grids for OpenVDB/Polygonal Conversions (ndim=3) ---
		# Stack along the trailing axis -> Shape: (resolution, resolution, resolution, 3)
		grid_pts_3d = np.stack([X, Y, Z], axis=-1)
		
		# Evaluate fields directly on the 3D tensor
		# These arrays are 3D (ndim=3), perfectly scaled, and centered in world space.
		vdb_sphere_sdf_l = self.sdf_sphere(grid_pts_3d, sphere_center, sphere_radius)
		vdb_box_sdf_l = self.sdf_box(grid_pts_3d, cube_center, cube_size)

		################
		######### VDB HIGH 
		################		
		# Establish simulation domain bounds
		min_bound, max_bound = -2.0, 2.0
		lin_coords_h = np.linspace(min_bound, max_bound, resolution_hi, dtype=np.float64)
		X_h, Y_h, Z_h = np.meshgrid(lin_coords_h, lin_coords_h, lin_coords_h, indexing='ij')

		# --- Pipeline A: 3D Grids for OpenVDB/Polygonal Conversions (ndim=3) ---
		# Stack along the trailing axis -> Shape: (resolution, resolution, resolution, 3)
		grid_pts_3d_h = np.stack([X_h, Y_h, Z_h], axis=-1)
		
		# Evaluate fields directly on the 3D tensor
		# These arrays are 3D (ndim=3), perfectly scaled, and centered in world space.
		vdb_sphere_sdf_h = self.sdf_sphere(grid_pts_3d_h, sphere_center, sphere_radius)
		vdb_box_sdf_h = self.sdf_box(grid_pts_3d_h, cube_center, cube_size)

		# grid_pts_flat = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1, dtype=np.float64)
		grid_pts_flat = grid_pts_3d.reshape(-1, 3)
		
		# Generate the global background structural node lattice
		num_nodes = len(grid_pts_flat)
		node_idx_grid = np.arange(num_nodes, dtype=np.int32).reshape(resolution, resolution, resolution)

		# Updated right-handed Kuhn 5-split local offsets mapping voxels to 5 distinct tets
		# All 5 elements will now yield a consistently positive geometric determinant.
		kuhn_template = np.array([
			[0, 1, 2, 4],  # Corner 1 (det = 1.0)
			[1, 3, 2, 7],  # Corner 2 (det = 1.0)
			[1, 4, 5, 7],  # Corner 3 (Corrected: swapped 5 and 4 -> det = 1.0)
			[4, 2, 6, 7],  # Corner 4 (Corrected: swapped 6 and 2 -> det = 1.0)
			[1, 2, 4, 7]   # Central Core (det = 2.0)
		], dtype=np.int32)

		# Extract the base voxel corner indices across the entire space array
		i, j, k = np.meshgrid(np.arange(resolution-1), np.arange(resolution-1), np.arange(resolution-1), indexing='ij')
		i, j, k = i.ravel(), j.ravel(), k.ravel()
		
		voxel_corners = np.stack([
			node_idx_grid[i,   j,   k],   node_idx_grid[i+1, j,   k],
			node_idx_grid[i,   j+1, k],   node_idx_grid[i+1, j+1, k],
			node_idx_grid[i,   j,   k+1], node_idx_grid[i+1, j,   k+1],
			node_idx_grid[i,   j+1, k+1], node_idx_grid[i+1, j+1, k+1]
		], axis=1)
		
		# Map all voxels out to the complete global tetrahedral matrix array
		tets = voxel_corners[:, kuhn_template].reshape(-1, 4)
		
		# --- Vectorized Multi-Phase Field Sampling ---
		tet_centers = grid_pts_flat[tets].mean(axis=1)
		ds_centers = self.sdf_sphere(tet_centers, sphere_center, sphere_radius)
		db_centers = self.sdf_box(tet_centers, cube_center, cube_size)
		
		# Initialize phase allocation map (Default Phase 0 = Air / Smoke / Gas)
		phase_tags = np.zeros(len(tets), dtype=np.int32)

		phase_tags[ds_centers <= 0] = 1

		''' ALL PHASE TO 0 for debug
		
		# Phase 1: Soft Tissue (Sphere)
		phase_tags[ds_centers <= 0] = 1
		
		# Phase 2: Rigid Structure (Cube)
		phase_tags[db_centers <= 0] = 2

		'''

		# Phase 3: Intervening Liquid Layer
		# Models a physical pool or fluid column sitting between Z=-0.2 and Z=+0.5
		# only where it is not displaced by the solid structures.
		# liquid_mask = (tet_centers[:, 2] > -0.2) & (tet_centers[:, 2] < 0.6) & (ds > 0) & (dc > 0)
		# phase_tags[liquid_mask] = 3

		# p = np.stack([X, Y, Z], axis=-1)
		# ds_toPoly = self.sdf_sphere(p, sphere_center, sphere_radius)
		# db_toPoly = self.sdf_box(p, cube_center, cube_size)
		# ds_toPoly = ds
		# db_toPoly = db

		# return grid_pts, tets, phase_tags, ds_toPoly, db_toPoly
		return grid_pts_flat, tets, phase_tags, vdb_sphere_sdf_l, vdb_sphere_sdf_h, vdb_box_sdf_l, vdb_box_sdf_h

	def visualize_global_multiphase_slice(self, unique_verts, tets, phase_tags, slice_axis=0, slice_val=0.0):
		"""
		Extracts outer and cross-sectional faces from the global simulation 
		continuum and colors them according to their active physical phase.
		"""
		centroids = unique_verts[tets].mean(axis=1)
		visible_mask = centroids[:, slice_axis] < slice_val
		sliced_tets = tets[visible_mask]
		sliced_tags = phase_tags[visible_mask]
		
		local_faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int32)
		all_faces = sliced_tets[:, local_faces].reshape(-1, 3)
		
		sorted_faces = np.sort(all_faces, axis=1)
		_, indices, counts = np.unique(sorted_faces, axis=0, return_index=True, return_counts=True)
		
		boundary_face_indices = indices[counts == 1]
		boundary_faces = all_faces[boundary_face_indices]
		
		face_to_tet_idx = boundary_face_indices // 4
		face_phases = sliced_tags[face_to_tet_idx]
		
		obj_name = "Global_FEM_Continuum_Debug"
		if obj_name in bpy.data.objects:
			bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink=True)
			
		mesh_data = bpy.data.meshes.new(obj_name + "_Data")
		mesh_obj = bpy.data.objects.new(obj_name, mesh_data)
		bpy.context.collection.objects.link(mesh_obj)
		
		mesh_data.vertices.add(len(unique_verts))
		mesh_data.polygons.add(len(boundary_faces))
		mesh_data.loops.add(len(boundary_faces) * 3)
		
		mesh_data.vertices.foreach_set("co", unique_verts.astype(np.float32).ravel())
		mesh_data.polygons.foreach_set("loop_start", np.arange(0, len(boundary_faces) * 3, 3, dtype=np.int32))
		mesh_data.polygons.foreach_set("loop_total", np.full(len(boundary_faces), 3, dtype=np.int32))
		mesh_data.loops.foreach_set("vertex_index", boundary_faces.astype(np.int32).ravel())
		
		mesh_data.update()
		mesh_data.validate()
		
		# 4-Phase Diagnostic Viewport Material Assignments
		material_colors = {
			0: (0.05, 0.05, 0.05, 1.0), # Phase 0: Air/Gas (Dark Gray Background)
			1: (0.85, 0.20, 0.20, 1.0), # Phase 1: Soft Tissue Sphere (Red)
			2: (0.20, 0.35, 0.85, 1.0), # Phase 2: Rigid Base Cube (Blue)
			3: (0.15, 0.75, 0.65, 1.0)  # Phase 3: Liquid Layer (Teal/Cyan)
		}
		
		for phase_id, color in material_colors.items():
			mat = bpy.data.materials.new(name=f"Phase_{phase_id}_Mat")
			mat.use_nodes = False
			mat.diffuse_color = color
			mesh_data.materials.append(mat)
			
		mesh_data.polygons.foreach_set("material_index", face_phases.astype(np.int32))
		mesh_data.update()
		
		return mesh_obj	

	def dFEM(self, abj_sd_b_instance):
		abj_sd_b_instance.deselectAll()
		abj_sd_b_instance.deleteAllObjects()
		abj_sd_b_instance.mega_purge()

		# grad_tanh = jax.grad(self.tanh)
		# print(grad_tanh(1.0))
		# prints 0.4199743

		self.testScene(abj_sd_b_instance)

	def testScene(self, abj_sd_b_instance):
		for volume_block in bpy.data.volumes:
			# Explicitly clear out old active grid trees from system RAM
			for grid in volume_block.grids:
				grid.unload() # Frees voxels from memory, forces file re-read on execution

		self.testVDB_06(abj_sd_b_instance) ####

	def deform_skin_tissue_mesh(self, x_corners_current, topology_tet4, vertex_to_tet_id, vertex_weights):
		"""
		Deforms the high-resolution render skin by multiplying current cage states
		by cached barycentric weights. Output coordinates map to Local Space.
		"""
		active_tets = topology_tet4[vertex_to_tet_id] 
		tet_nodes_x = x_corners_current[active_tets] # Shape: (V, 4, 3)

		w_expanded = vertex_weights[:, :, np.newaxis] # Shape: (V, 4, 1)
		deformed_skin_fl64 = np.sum(tet_nodes_x * w_expanded, axis=1) # Shape: (V, 3)

		return deformed_skin_fl64.astype(np.float32)

	def compile_painted_mesh_to_fem_attributes(self, mesh_obj_name, nodes, tet_indices):
		pass

		return 
	
		"""
		Ingests a pre-generated sliver-free tetrahedral lattice and maps painted 
		Blender vertex color properties to it element-by-element using a BVHTree.
		
		Args:
			mesh_obj_name: String name of your Plasticity STL / Blender input model.
			nodes: (N, 3) array of lattice node positions from generate_sliver_free_lattice.
			tet_indices: (M, 4) array of tetrahedron topologies from generate_sliver_free_lattice.
			
		Returns:
			element_properties: (M, 4) array mapping every individual tetrahedron to its 
								custom [Material_ID, Youngs_Modulus, Poissons_Ratio, Density].
		"""
		context = bpy.context
		obj = context.scene.objects.get(mesh_obj_name)
		if not obj or obj.type != 'MESH':
			raise ValueError(f"Object '{mesh_obj_name}' not found or is a valid mesh.")
			
		# 1. EVALUATE MESH AND BUILD THE ACCELERATED BVH TREE
		depsgraph = context.evaluated_depsgraph_get()
		obj_eval = obj.evaluated_get(depsgraph)
		mesh = obj_eval.to_mesh()
		mesh.transform(obj.matrix_world)

		color_attrs = mesh.color_attributes
		ym_attr = color_attrs.get("YoungsModulus")
		pr_attr = color_attrs.get("PoissonsRatio")

		bvh = BVHTree.FromMesh(mesh)

		# 2. CALCULATE THE EXACT CENTROID OF EVERY INDIVIDUAL GENERATED TETRAHEDRON
		# Instead of manual loops, we use optimized NumPy vector indexing.
		# nodes[tet_indices] creates a shape of (num_tets, 4_nodes, 3_coordinates)
		tet_centers = np.mean(nodes[tet_indices], axis=1)
		num_tets = len(tet_indices)

		# 3. INITIALIZE ALIGNED PHYSICAL PROPERTY MAPS (1-to-1 match with tet_indices)
		element_properties = np.zeros((num_tets, 4), dtype=np.float64)

		# User Engineering Baseline Parameters
		BASE_SOLID_E = 5000.0   # Squishy base tissue
		MAX_SOLID_E  = 50000.0  # Painted tendon stiffness
		BASE_NU      = 0.30     # Compressible boundary
		MAX_NU       = 0.499    # Incompressible volume-preserving boundary

		# 4. LOOP GENERATION INTERPOLATION STEP
		for idx, center in enumerate(tet_centers):
			co = mathutils.Vector(center)
			loc, normal, face_idx, distance = bvh.find_nearest(co)
			
			if loc is not None:
				to_center = co - loc
				
				# Insideness Check: Dot product determines if centroid is inside the STL shell
				if to_center.dot(normal) <= 0.0:
					ym_weight = 0.0
					pr_weight = 0.0
					
					# Fetch local face loop corners for vertex attribute reading
					face = mesh.polygons[face_idx]
					
					if ym_attr or pr_attr:
						loop_yms = []
						loop_prs = []
						for loop_idx in face.loop_indices:
							if ym_attr:
								# Read Red channel value of painted vertex loop attribute
								loop_yms.append(ym_attr.data[loop_idx].color[0])
							if pr_attr:
								loop_prs.append(pr_attr.data[loop_idx].color[0])
						
						if loop_yms: ym_weight = np.mean(loop_yms)
						if loop_prs: pr_weight = np.mean(loop_prs)

					# Convert the 0-1 painted spectrum directly to real engineering scales
					E_value = BASE_SOLID_E + (ym_weight * (MAX_SOLID_E - BASE_SOLID_E))
					nu_value = BASE_NU + (pr_weight * (MAX_NU - BASE_NU))
					
					# Assign: ID=101 (Solid), Young's Modulus, Poisson's Ratio, Mass=1.0
					element_properties[idx] = [101.0, E_value, nu_value, 1.0]
					continue
					
			# Fallback: Elements outside the BVHTree are automatically tagged as multi-phase ambient air
			# Assign: ID=202 (Fluid/Air), E=0, Bulk Modulus = 100.0, Density=0.001
			element_properties[idx] = [202.0, 0.0, 100.0, 0.001]

		obj_eval.to_mesh_clear()
		return element_properties






	def bakeShaderToPts(self, obj):
		# 1. Add the modifier slot and link a clean GeometryNodeTree
		gn_mod = obj.modifiers.new(name="ProceduralDisplacement", type='NODES')
		node_group = bpy.data.node_groups.new(name="GaborDisplaceTree", type='GeometryNodeTree')
		gn_mod.node_group = node_group

		# Create the required structural geometry sockets
		node_group.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
		node_group.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')

		# 2. Create the nodes
		node_in = node_group.nodes.new(type="NodeGroupInput")
		node_out = node_group.nodes.new(type="NodeGroupOutput")

		# CRITICAL ADDITION: Subdivide Mesh node to capture high-frequency noise
		node_subdivide = node_group.nodes.new(type="GeometryNodeSubdivideMesh")
		# Level 4 or 5 gives the Gabor noise enough vertex density to actually resolve
		# node_subdivide.inputs['Level'].default_value = 3
		node_subdivide.inputs['Level'].default_value = 4
		# node_subdivide.inputs['Level'].default_value = 5

		# Set Position & Noise
		node_set_pos = node_group.nodes.new(type="GeometryNodeSetPosition")
		node_noise = node_group.nodes.new(type="ShaderNodeTexGabor") 
		
		# Ensure Gabor settings match your shader (Scale, Frequency, etc.)
		# node_noise.inputs['Scale'].default_value = 5.0 

		# Vector Math nodes for calculating displacement vector
		node_normal = node_group.nodes.new(type="GeometryNodeInputNormal")
		node_multiply = node_group.nodes.new(type="ShaderNodeVectorMath")
		node_multiply.operation = 'MULTIPLY'

		node_scale = node_group.nodes.new(type="ShaderNodeVectorMath")
		node_scale.operation = 'SCALE'
		node_scale.inputs[3].default_value = 0.5  # Displacement Strength

		# 3. Connect the node architecture
		links = node_group.links

		# Link geometry THROUGH the subdivide node first
		links.new(node_in.outputs['Geometry'], node_subdivide.inputs['Mesh'])
		links.new(node_subdivide.outputs['Mesh'], node_set_pos.inputs['Geometry'])
		links.new(node_set_pos.outputs['Geometry'], node_out.inputs['Geometry'])

		# Calculate displacement vector using the newly subdivided geometry normals
		links.new(node_normal.outputs['Normal'], node_multiply.inputs[0])
		links.new(node_noise.outputs['Value'], node_multiply.inputs[1])

		# Scale and apply to offset
		links.new(node_multiply.outputs['Vector'], node_scale.inputs[0])
		links.new(node_scale.outputs['Vector'], node_set_pos.inputs['Offset'])

		# 4. Freeze the Modifier (Bakes the subdivided, high-res mesh to real vertices)
		bpy.ops.object.modifier_apply(modifier=gn_mod.name)

		print(f"Success! Mesh baked with subdivision. {len(obj.data.vertices)} points available.")


	def bakeShaderToPts0(self, obj):
		############################
		#######BAKE GABOR TO PTS
		############################


		# 1. Setup target object 
		# bpy.ops.mesh.primitive_grid_add(x_subdivisions=100, y_subdivisions=100, size=2)
		# ball_obj = bpy.context.active_object

		# 2. Add the modifier slot and link a clean GeometryNodeTree
		gn_mod = obj.modifiers.new(name="ProceduralDisplacement", type='NODES')
		node_group = bpy.data.node_groups.new(name="GaborDisplaceTree", type='GeometryNodeTree')
		gn_mod.node_group = node_group

		# Create the required structural geometry sockets (Blender 5.2 API flat tree style)
		node_group.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
		node_group.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')

		# 3. Create the standard, universally supported nodes
		node_in = node_group.nodes.new(type="NodeGroupInput")
		node_out = node_group.nodes.new(type="NodeGroupOutput")

		# We use 'GeometryNodeSetPosition' which is completely stable and supported
		node_set_pos = node_group.nodes.new(type="GeometryNodeSetPosition")

		# Texture node (e.g., Gabor or Noise)
		node_noise = node_group.nodes.new(type="ShaderNodeTexGabor") 

		# Vector Math nodes to calculate: Normal * Noise Value * Strength
		node_normal = node_group.nodes.new(type="GeometryNodeInputNormal")
		node_multiply = node_group.nodes.new(type="ShaderNodeVectorMath")
		node_multiply.operation = 'MULTIPLY'

		node_scale = node_group.nodes.new(type="ShaderNodeVectorMath")
		node_scale.operation = 'SCALE'
		# node_scale.inputs[3].default_value = 0.3  # This acts as your displacement "Strength"
		node_scale.inputs[3].default_value = 0.5  # This acts as your displacement "Strength"

		# 4. Connect the node architecture
		links = node_group.links

		# Link the core geometry line
		links.new(node_in.outputs['Geometry'], node_set_pos.inputs['Geometry'])
		links.new(node_set_pos.outputs['Geometry'], node_out.inputs['Geometry'])

		# Calculate displacement vector: Normal * Gabor Value
		links.new(node_normal.outputs['Normal'], node_multiply.inputs[0])
		links.new(node_noise.outputs['Value'], node_multiply.inputs[1])

		# Scale the final vector by your strength setting and pipe it into 'Offset'
		links.new(node_multiply.outputs['Vector'], node_scale.inputs[0])
		links.new(node_scale.outputs['Vector'], node_set_pos.inputs['Offset'])

		# 5. Freeze the Modifier (Locks down procedural changes to real vertices)
		bpy.ops.object.modifier_apply(modifier=gn_mod.name)

		print(f"Success! Mesh baked. {len(obj.data.vertices)} points available for your BPY loop.")

	def bounce_diff_03(self, abj_sd_b_instance):
		bpy.context.scene.render.engine = 'CYCLES'
		# bpy.context.scene.render.engine = 'BLENDER_EEVEE'
		bpy.context.scene.cycles.device = 'GPU'

		bpy.context.scene.cycles.samples = 64
		bpy.context.scene.cycles.denoising_use_gpu = True

		# ==============================================================================
		# 1. SCENE CLEANUP & BLENDER ENVIRONMENT LAYOUT SETUP
		# ==============================================================================
		print("Initializing unified JAX Unified Bounce-and-Crush Continuum Engine...")

		for name in ["Rubber_Ball", "Voxel_Collision_Cube"]:
			if name in bpy.data.objects:
				bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)

		SPHERE_RES = 32  # High resolution captures both fluid ripples and flat cushion folds
		# SPHERE_RES = 40  # High resolution captures both fluid ripples and flat cushion folds
		# SPHERE_RES = 48  # High resolution captures both fluid ripples and flat cushion folds
		# SPHERE_RES = 64  # High resolution captures both fluid ripples and flat cushion folds
		# SPHERE_RES = 100  # High resolution captures both fluid ripples and flat cushion folds
		# SPHERE_RES = 128  # High resolution captures both fluid ripples and flat cushion folds
		# SPHERE_RES = 256  # High resolution captures both fluid ripples and flat cushion folds

		# Spawn target sphere at Z = 7.5
		# bpy.ops.mesh.primitive_uv_sphere_add(radius=1.8, location=(0.0, 0.0, 7.5), segments=SPHERE_RES, ring_count=SPHERE_RES)
		bpy.ops.mesh.primitive_uv_sphere_add(radius=1.8, location=(0.0, 0.0, 0), segments=SPHERE_RES, ring_count=SPHERE_RES)
		# bpy.ops.mesh.primitive_cube_add(location=(0.0, 0.0, 7.5), size=6)

		ball_obj = bpy.context.active_object
		ball_obj.name = "Rubber_Ball"

		gaborToPts = 0
		# gaborToPts = 1

		if gaborToPts == 1:
			self.bakeShaderToPts(ball_obj) ######

		# # 3. Switch to Edit Mode to modify the geometry
		# bpy.ops.object.mode_set(mode='EDIT')

		# # 4. Select all geometry (vertices/edges/faces)
		# bpy.ops.mesh.select_all(action='SELECT')

		# # 5. Subdivide the mesh 
		# # Set number_cuts to your desired resolution. Keep smoothness at 0.0 to prevent rounding!
		# # bpy.ops.mesh.subdivide(number_cuts=12, smoothness=0.0)
		# # bpy.ops.mesh.subdivide(number_cuts=16, smoothness=0.0)
		# bpy.ops.mesh.subdivide(number_cuts=32, smoothness=0.0)

		# # 6. Switch back to Object Mode
		# bpy.ops.object.mode_set(mode='OBJECT')





		# return

		# bpy.ops.object.modifier_add(type='SUBSURF')
		# # ball_obj.modifiers["Subdivision"].levels = 1
		# ball_obj.modifiers["Subdivision"].levels = 4
		# bpy.ops.object.modifier_apply(modifier="Subdivision")
		
		# return
		
		# bpy.ops.object.modifier_add(type='SUBSURF')
		# ball_obj.modifiers["Subdivision"].levels = 1
		# # ball_obj.modifiers["Subdivision"].levels = 2
		# ball_obj.modifiers["Subdivision"].use_adaptive_subdivision = True

		# bpy.ops.object.modifier_apply(modifier="Subdivision")

		ball_obj.shape_key_add(name="Basis", from_mix=False)
		bpy.ops.object.shade_smooth()

		mat1 = abj_sd_b_instance.newShader("principled_test_00", "principled", 1, 0, 0)
		bpy.context.active_object.data.materials.clear()
		bpy.context.active_object.data.materials.append(mat1)
		bpy.data.materials["principled_test_00"].node_tree.nodes["Principled BSDF"].inputs[1].default_value = 1
		bpy.data.materials["principled_test_00"].node_tree.nodes["Principled BSDF"].inputs[2].default_value = 0.323263
		bpy.data.materials["principled_test_00"].node_tree.nodes["Principled BSDF"].inputs[20].default_value = 1
		ball_obj.active_material.displacement_method = 'BOTH'


		mat = bpy.data.materials.get("principled_test_00")
		nodes = mat.node_tree.nodes

		nodes = mat.node_tree.nodes

		output_node = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
		if not output_node:
			output_node = nodes.new(type='ShaderNodeOutputMaterial')

		if gaborToPts != 1:
			gabor_node = nodes.new(type='ShaderNodeTexGabor')
			displacement_node = nodes.new(type='ShaderNodeDisplacement')

			mat.node_tree.links.new(gabor_node.outputs['Value'], displacement_node.inputs['Height'])
			mat.node_tree.links.new(displacement_node.outputs['Displacement'], output_node.inputs['Displacement'])
			# displacement_node.inputs[2].default_value = 0.3
			displacement_node.inputs[2].default_value = 0.5

			# bpy.data.materials["principled_test_00"].node_tree.nodes["Gabor Texture"].gabor_type = '3D'
			gabor_node.gabor_type = '3D'


		abj_sd_b_instance.autoArrangeNodes(mat.node_tree)

		# return

		######################################
		# Wide collision floor plane cube (Top face at world Z = 2.0)
		######################################
		bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
		cube_obj = bpy.context.active_object
		cube_obj.name = "Voxel_Collision_Cube"
		cube_obj.scale = (30.0, 30.0, 1.0)
		cube_obj.location = (0.0, 0.0, 1.5) 

		mat1 = abj_sd_b_instance.newShader("principled_test_grd", "principled", 0, 0, 1)

		# checkerNode = nodes.new(type='ShaderNodeTexChecker')
		# principledNode = nodes.new(type='ShaderNodeBsdfPrincipled')

		bpy.context.active_object.data.materials.clear()
		bpy.context.active_object.data.materials.append(mat1)
		bpy.data.materials["principled_test_grd"].node_tree.nodes["Principled BSDF"].inputs[1].default_value = 1
		bpy.data.materials["principled_test_grd"].node_tree.nodes["Principled BSDF"].inputs[2].default_value = 0.323263

		# 1. Get the specific material 
		mat1 = bpy.data.materials.get("principled_test_grd")

		# Ensure use_nodes is enabled so the node tree exists
		mat1.use_nodes = True
		nodes = mat1.node_tree.nodes
		links = mat1.node_tree.links

		# 2. Correctly create the checker node inside mat1's tree
		checkerNode = nodes.new(type='ShaderNodeTexChecker')

		# 3. Find the existing Principled BSDF inside mat1's tree
		# (Using nodes.get avoids errors if it was renamed)
		principledNode = nodes.get("Principled BSDF")

		# 4. Safely create the link using the explicitly targeted tree
		if principledNode:
			links.new(checkerNode.outputs['Color'], principledNode.inputs['Base Color'])

		bpy.data.materials["principled_test_grd"].node_tree.nodes["Checker Texture"].inputs[3].default_value = 10

		mat1.node_tree.links.new(checkerNode.outputs['Color'], bpy.data.materials["principled_test_grd"].node_tree.nodes["Principled BSDF"].inputs[0])

		abj_sd_b_instance.autoArrangeNodes(mat.node_tree)
		abj_sd_b_instance.autoArrangeNodes(mat1.node_tree)

		###########
		# WORLD
		###########

		world = bpy.context.scene.world
		worldtree = world.node_tree
		worldtree.nodes.clear()

		# output_node_world = next((n for n in worldtree.nodes if n.type == 'ShaderNodeOutputWorld'), None)
		# if not output_node:
		# 	output_node_world = worldtree.nodes.new(type='ShaderNodeOutputWorld')

		# output_node = worldtree.nodes.new(type="ShaderNodeOutputWorld")
		# bg_node = worldtree.nodes.new(type="ShaderNodeBackground")

		# output_node_world = worldtree.nodes.new(type="ShaderNodeOutputWorld")
		output_node_world = worldtree.nodes.new('ShaderNodeOutputWorld')

		# node_sky = bpy.ops.node.add_node(use_transform=True, type="ShaderNodeTexSky")
		# node_sky = bpy.ops.node.add_node(use_transform=True, type="ShaderNodeTexSky")
		node_sky = worldtree.nodes.new('ShaderNodeTexSky')
		worldtree.links.new(node_sky.outputs["Color"], output_node_world.inputs["Surface"])

		# bpy.data.worlds["World"].node_tree.nodes["Sky Texture"].sun_size = 0.372541
		# bpy.data.worlds["World"].node_tree.nodes["Sky Texture"].sun_intensity = 21.3
		# bpy.data.worlds["World"].node_tree.nodes["Sky Texture"].sun_rotation = -1.57603
		# node_sky.sun_size = 0.372541
		# node_sky.sun_intensity = 21.3
		node_sky.sun_rotation = 1.65806
		node_sky.sun_elevation = .05

		abj_sd_b_instance.autoArrangeNodes(worldtree)

		bpy.context.view_layer.update()

		abj_sd_b_instance.agxColorSettings_UI()

		# return

		# ==============================================================================
		# 2. EXTRACT SCENE GEOMETRY TO GLOBAL SCOPE ARRAYS
		# ==============================================================================
		mesh = ball_obj.data
		num_verts = len(mesh.vertices)
		orig_coords = np.zeros((num_verts, 3))
		mesh.vertices.foreach_get("co", orig_coords.ravel())

		# Baseline world coordinates tracking matrix
		sphere_tracker = np.copy(orig_coords)
		sphere_tracker[:, 2] += 7.5  

		# Extract reference normal direction vectors
		ref_normals_numpy = np.zeros((num_verts, 3))
		for v in mesh.vertices:
			ref_normals_numpy[v.index] = np.array(v.normal)

		# Precompute thickness profile cushion (14% of initial relative height profile)
		min_init_z = np.min(orig_coords[:, 2])
		vertex_thickness_numpy = (orig_coords[:, 2] - min_init_z) * 0.14  

		# --- GLOBAL SCOPE JAX ARRAYS ---
		# By assigning these to the global module scope, the JAX functions can read them
		# directly via closure. They are never passed through scan, so they CANNOT flatten or swap!

		STATIC_INIT_POS = jnp.array(sphere_tracker)
		STATIC_CENTER = jnp.mean(STATIC_INIT_POS, axis=0) # Pristine Frame 1 rest center
		STATIC_NORMALS = jnp.array(ref_normals_numpy)
		STATIC_THICKNESS = jnp.array(vertex_thickness_numpy)


		# ==============================================================================
		# 4. EXPLICIT AUTOMATED INDIVIDUAL-ARGUMENT BACKPROPAGATION GRADIENT DESCENT LOOP
		# ==============================================================================

		# --- FIXED: WEIGHT PROFILE EXPERIMENT SWITCHBOARD ---
		# Test Case 1: Heavy Lead Brick -> Mass = 50.0, Drag = 0.15 (Slams down hard, bounces high)
		# Test Case 2: Light Feather Cushion -> Mass = 0.8, Drag = 1.85 (Floats down slowly, stays soft)
		# OBJECT_MASS_VAL = 45.0          
		# DRAG_COEFFICIENT_VAL = 0.25   


		# num_frames = 600
		num_frames = 300
		# num_frames = 300

		# ###########
		# #### SPHERE 01
		# ###########
		# MU_VAL = 1450.0
		# LAM_VAL = 4000.0
		# # DAMPING_VAL = 0.995
		# DAMPING_VAL = 0.990
		# # DT_VAL = 0.01
		# # DT_VAL = 0.008
		# DT_VAL = 0.008
		# # INITIAL_SPIKE_VELOCITY = -30
		# INITIAL_SPIKE_VELOCITY = -30
		# RESTITUTION_VAL = .7
		# OBJECT_MASS_VAL = 400
		# DRAG_COEFFICIENT_VAL = .5


	

		##########
		### SPHERE GOOD
		##########
		MU_VAL = 550.0
		LAM_VAL = 3000.0
		# DAMPING_VAL = 0.995
		# DAMPING_VAL = 0.98
		# DAMPING_VAL = 0.98
		DAMPING_VAL = 0.98
		# DAMPING_VAL = 0.94
		# DT_VAL = 0.01
		# DT_VAL = 0.008
		DT_VAL = 0.008


		# DT_VAL = 0.002
		# INITIAL_SPIKE_VELOCITY = -30 * 4
		# INITIAL_SPIKE_VELOCITY = -30
		INITIAL_SPIKE_VELOCITY = -60.0
		# INITIAL_SPIKE_VELOCITY = -90


		# INITIAL_SPIKE_VELOCITY = -50
		# RESTITUTION_VAL = .87
		RESTITUTION_VAL = .7
		# RESTITUTION_VAL = .5
		OBJECT_MASS_VAL = 300.0
		DRAG_COEFFICIENT_VAL = .05



		# ########
		# # CUBE
		# ########
		# MU_VAL = 1590.0
		# LAM_VAL = 3000.0
		# # DAMPING_VAL = 0.995
		# DAMPING_VAL = 0.99
		# # DT_VAL = 0.01
		# # DT_VAL = 0.008
		# DT_VAL = 0.008
		# INITIAL_SPIKE_VELOCITY = -30
		# # INITIAL_SPIKE_VELOCITY = -10
		# RESTITUTION_VAL = .9
		# OBJECT_MASS_VAL = 400
		# DRAG_COEFFICIENT_VAL = .03






		TARGET_FRAME = 100
		TARGET_HEIGHT = 12.5


		loss_fn = jax.jit(lambda m, l, d, t, v, r, ma, dr: loss_function(m, l, d, t, v, r, ma, dr, STATIC_INIT_POS, STATIC_CENTER, STATIC_NORMALS, STATIC_THICKNESS, TARGET_FRAME, TARGET_HEIGHT))

		# Target index slots securely: 0=mu, 1=lam, 2=damping, 4=velocity, 6=mass
		grad_fn = jax.jit(jax.grad(
			lambda m, l, d, t, v, r, ma, dr: loss_function(m, l, d, t, v, r, ma, dr, STATIC_INIT_POS, STATIC_CENTER, STATIC_NORMALS, STATIC_THICKNESS, TARGET_FRAME, TARGET_HEIGHT),
			argnums=(0, 1, 2, 4, 6)
		))


		print(f"\n[JAX Optimizer] Launching local manual-argument gradient search trajectory tracking...")

		lr_stiffness = 2.5    
		# lr_damping = 4e-4 ##
		# lr_damping = 2
		lr_damping = .1
		# lr_velocity = 4e-1    
		lr_velocity = 10 
		# lr_mass = 0.5         # Gradient search rate for weight scale optimization
		lr_mass = .5         # Gradient search rate for weight scale optimization
		# num_steps = 20
		# num_steps = 10
		# num_steps = 5
		num_steps = 2


		for iteration in range(num_steps):
			continue

			current_loss = loss_fn(MU_VAL, LAM_VAL, DAMPING_VAL, DT_VAL, INITIAL_SPIKE_VELOCITY, RESTITUTION_VAL, OBJECT_MASS_VAL, DRAG_COEFFICIENT_VAL)
			raw_grads = grad_fn(MU_VAL, LAM_VAL, DAMPING_VAL, DT_VAL, INITIAL_SPIKE_VELOCITY, RESTITUTION_VAL, OBJECT_MASS_VAL, DRAG_COEFFICIENT_VAL)
			
			grad_mu, grad_lam, grad_damp, grad_vel, grad_mass = raw_grads
			
			# FIXED: True Sign Fallback. If gradients hit a zero plateau, we use a constant 
			# directional sign push to dynamically kick the parameter out of the flat zone
			step_mu   = jnp.sign(grad_mu) * lr_stiffness if jnp.abs(grad_mu) > 1e-5 else jnp.sign(MU_VAL - 450.0) * lr_stiffness
			step_lam  = jnp.sign(grad_lam) * lr_stiffness if jnp.abs(grad_lam) > 1e-5 else jnp.sign(LAM_VAL - 3500.0) * lr_stiffness
			step_damp = grad_damp * lr_damping if jnp.abs(grad_damp) > 1e-5 else (DAMPING_VAL - 0.94) * lr_damping
			
			# Track velocity derivatives out of dead zones smoothly
			step_vel  = jnp.sign(grad_vel) * lr_velocity if jnp.abs(grad_vel) > 1e-5 else -1.5 * lr_velocity
			step_mass = jnp.sign(grad_mass) * lr_mass      if jnp.abs(grad_mass) > 1e-5 else grad_mass * 2
			
			# Apply individual unrolled updates smoothly
			MU_VAL                 = float(jnp.clip(MU_VAL + step_mu, 200.0, 5000.0))
			LAM_VAL                = float(jnp.clip(LAM_VAL + step_lam, 1000.0, 15000.0))
			DAMPING_VAL            = float(jnp.clip(DAMPING_VAL - step_damp, 0.98, 0.994))
			OBJECT_MASS_VAL        = float(jnp.clip(OBJECT_MASS_VAL + step_mass, 0.1, 200.0)) # Clip weight to valid ranges
			
			# FIXED: Expanded the clipping boundary ceiling completely to support -120.0 m/s
			# INITIAL_SPIKE_VELOCITY = float(jnp.clip(INITIAL_SPIKE_VELOCITY - step_vel, -120.0, -10.0))
			INITIAL_SPIKE_VELOCITY = float(jnp.clip(INITIAL_SPIKE_VELOCITY + step_vel, -120.0, -10.0))
			
			print(f"  Step {iteration+1:02d} -> Loss: {current_loss:.4f} | Mass Weight: {OBJECT_MASS_VAL:.2f} kg | Spike Vel: {INITIAL_SPIKE_VELOCITY:.2f} m/s | Mu: {MU_VAL:.1f} | Lam: {LAM_VAL:.1f} | Damping: {DAMPING_VAL:.4f} Dt: {DT_VAL:.4f}")
			
		print("\n[JAX Engine] Optimization target achieved! Pulling verified trajectory memory buffer...")

		# num_frames = 600
		
		
		final_trajectory_matrix = jit_simulation_engine(
			MU_VAL, LAM_VAL, DAMPING_VAL, DT_VAL, INITIAL_SPIKE_VELOCITY, RESTITUTION_VAL, OBJECT_MASS_VAL, DRAG_COEFFICIENT_VAL, STATIC_INIT_POS, STATIC_CENTER, STATIC_NORMALS, STATIC_THICKNESS, num_frames
		)

		baked_frames_positions = np.array(final_trajectory_matrix)
		depsgraph = bpy.context.evaluated_depsgraph_get()

		print("[Blender Pipeline] Writing exact JAX position memory to timeline Shape Keys...")

		for idx, frame in enumerate(range(1, num_frames + 1)):
			print('frame_idx = ', idx)
			bpy.context.scene.frame_set(frame)
			
			frame_coords = baked_frames_positions[frame - 1]
			baked_local_coords = np.copy(frame_coords)
			baked_local_coords[:, 2] -= 7.5
			
			skey = ball_obj.shape_key_add(name=f"Frame_{frame}", from_mix=False)
			skey.data.foreach_set("co", baked_local_coords.ravel())
			
			skey.value = 1.0
			skey.keyframe_insert(data_path="value", frame=frame)
			if frame > 1:
				skey.value = 0.0
				skey.keyframe_insert(data_path="value", frame=frame - 1)
				prev_key = ball_obj.data.shape_keys.key_blocks[f"Frame_{frame-1}"]
				prev_key.value = 0.0
				prev_key.keyframe_insert(data_path="value", frame=frame)
				
			depsgraph.update()

		bpy.context.scene.frame_set(1)
		print("\n[Bake Finished] Unified engine execution completed successfully! Press Spacebar.")

		# bpy.context.scene.render.fps = 240
		bpy.context.scene.render.fps = 120
		# bpy.context.scene.render.fps = 120 * 4
		bpy.context.scene.frame_end = num_frames

	def bounce_diff_04(self, abj_sd_b_instance):
		bpy.context.scene.render.engine = 'CYCLES'
		# bpy.context.scene.render.engine = 'BLENDER_EEVEE'
		bpy.context.scene.cycles.device = 'GPU'

		bpy.context.scene.cycles.samples = 64
		bpy.context.scene.cycles.denoising_use_gpu = True

		# ==============================================================================
		# 1. SCENE CLEANUP & BLENDER ENVIRONMENT LAYOUT SETUP
		# ==============================================================================
		print("Initializing unified JAX Unified Bounce-and-Crush Continuum Engine...")

		for name in ["Rubber_Ball", "Voxel_Collision_Cube"]:
			if name in bpy.data.objects:
				bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)

		# SPHERE_RES = 8  # High resolution captures both fluid ripples and flat cushion folds
		# SPHERE_RES = 16  # High resolution captures both fluid ripples and flat cushion folds
		SPHERE_RES = 32  # High resolution captures both fluid ripples and flat cushion folds ### !!!!!!!
		# SPHERE_RES = 40  # High resolution captures both fluid ripples and flat cushion folds
		# SPHERE_RES = 48  # High resolution captures both fluid ripples and flat cushion folds
		# SPHERE_RES = 64  # High resolution captures both fluid ripples and flat cushion folds
		# SPHERE_RES = 100  # High resolution captures both fluid ripples and flat cushion folds
		# SPHERE_RES = 128  # High resolution captures both fluid ripples and flat cushion folds
		# SPHERE_RES = 256  # High resolution captures both fluid ripples and flat cushion folds

		# Spawn target sphere at Z = 7.5
		# bpy.ops.mesh.primitive_uv_sphere_add(radius=1.8, location=(0.0, 0.0, 7.5), segments=SPHERE_RES, ring_count=SPHERE_RES)
		bpy.ops.mesh.primitive_ico_sphere_add(radius=1.8, location=(0.0, 0.0, 7.5), subdivisions=4)
		# bpy.ops.mesh.primitive_ico_sphere_add(radius=1.8, location=(0.0, 0.0, 0), subdivisions=4)
		# bpy.ops.mesh.primitive_cube_add(location=(0.0, 0.0, 7.5), size=6)

		ball_obj = bpy.context.active_object
		ball_obj.name = "Rubber_Ball"

		ball_obj.shape_key_add(name="Basis", from_mix=False)
		bpy.ops.object.shade_smooth()

		ball_obj = bpy.context.active_object

		# return

		gaborToPts = 0
		# gaborToPts = 1

		# if gaborToPts == 1:
		# 	self.bakeShaderToPts(ball_obj) ######

		mat1 = abj_sd_b_instance.newShader("principled_test_00", "principled", 1, 0, 0)
		bpy.context.active_object.data.materials.clear()
		bpy.context.active_object.data.materials.append(mat1)
		bpy.data.materials["principled_test_00"].node_tree.nodes["Principled BSDF"].inputs[1].default_value = 1
		bpy.data.materials["principled_test_00"].node_tree.nodes["Principled BSDF"].inputs[2].default_value = 0.323263
		bpy.data.materials["principled_test_00"].node_tree.nodes["Principled BSDF"].inputs[20].default_value = 1
		ball_obj.active_material.displacement_method = 'BOTH'

		mat = bpy.data.materials.get("principled_test_00")
		nodes = mat.node_tree.nodes

		nodes = mat.node_tree.nodes

		output_node = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
		if not output_node:
			output_node = nodes.new(type='ShaderNodeOutputMaterial')

		if gaborToPts != 1:
			gabor_node = nodes.new(type='ShaderNodeTexGabor')
			displacement_node = nodes.new(type='ShaderNodeDisplacement')

			mat.node_tree.links.new(gabor_node.outputs['Value'], displacement_node.inputs['Height'])
			mat.node_tree.links.new(displacement_node.outputs['Displacement'], output_node.inputs['Displacement'])
			# displacement_node.inputs[2].default_value = 0.3
			displacement_node.inputs[2].default_value = 0.5

			# bpy.data.materials["principled_test_00"].node_tree.nodes["Gabor Texture"].gabor_type = '3D'
			gabor_node.gabor_type = '3D'

			# checkerNode = nodes.new(type='ShaderNodeTexChecker')
			# principledNode = nodes.get("Principled BSDF")
			# mat.node_tree.links.new(checkerNode.outputs['Color'], principledNode.inputs['Base Color'])
			# checkerNode.inputs[2].default_value = (0, 0, 0, 1)


		abj_sd_b_instance.autoArrangeNodes(mat.node_tree)

		######## CUBE FLOOR ###########
		######## CUBE FLOOR ###########
		######## CUBE FLOOR ###########

		# Spawn Cube Ground Floor Object
		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 1.5))
		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -20))
		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -15))
		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -30))


		# bpy.ops.mesh.primitive_ico_sphere_add(radius=6, location=(0.0, 0.0, 7.5), subdivisions=4)
		# bpy.ops.mesh.primitive_ico_sphere_add(radius=48, location=(0.0, 0.0, -75.5), subdivisions=4)

		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -40))
		bpy.ops.mesh.primitive_cube_add(size=1.0, location=(15.0, 0.0, -40))
		cube_floor = bpy.context.active_object
		cube_floor.name = "Voxel_Collision_Cube"
		cube_floor.scale = (1200.0, 1200.0, 1.0)
		cube_floor.rotation_euler = (0.0, 0.45, 0.0)

		# cube_floor = bpy.context.active_object

		mat1 = abj_sd_b_instance.newShader("principled_test_00_grd", "principled", .2, 0, 0)
		cube_floor.data.materials.clear()
		cube_floor.data.materials.append(mat1)
		bpy.data.materials["principled_test_00_grd"].node_tree.nodes["Principled BSDF"].inputs[1].default_value = 1
		bpy.data.materials["principled_test_00_grd"].node_tree.nodes["Principled BSDF"].inputs[2].default_value = 0.323263
		bpy.data.materials["principled_test_00_grd"].node_tree.nodes["Principled BSDF"].inputs[20].default_value = 1
		cube_floor.active_material.displacement_method = 'BOTH'

		mat = bpy.data.materials.get("principled_test_00_grd")
		nodes = mat.node_tree.nodes

		nodes = mat.node_tree.nodes





		checkerNode = nodes.new(type='ShaderNodeTexChecker')
		principledNode = nodes.get("Principled BSDF")
		mat.node_tree.links.new(checkerNode.outputs['Color'], principledNode.inputs['Base Color'])
		checkerNode.inputs[2].default_value = (0, 0, 0, 1)
		# bpy.data.materials["principled_test_grd"].node_tree.nodes["Checker Texture"].inputs[3].default_value = 10
		checkerNode.inputs[3].default_value = 100






		output_node = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
		if not output_node:
			output_node = nodes.new(type='ShaderNodeOutputMaterial')

		abj_sd_b_instance.autoArrangeNodes(mat.node_tree)

		##########  MID OBSTACLE ##############
		##########  MID OBSTACLE ##############
		##########  MID OBSTACLE ##############

		# Spawn Second Mid-Air Intercepting Cube Obstacle
		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.5, 0.0, 4.5))
		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.5, 0.0, 2.75))
		bpy.ops.mesh.primitive_cube_add(size=1.0, location=(10, 0.0, -60)) #####s
		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.5, 0.0, -100))  
		mid_obstacle = bpy.context.active_object
		mid_obstacle.name = "Mid_Air_Obstacle"
		# mid_obstacle.rotation_euler = (0.45, 0.25, 0.78)
		# mid_obstacle.rotation_euler = (0.65, -0.45, 0) #####
		mid_obstacle.rotation_euler = (0.65, -0.45, -45)
		# mid_obstacle.scale = (2.0, 2.0, 1.5)
		# mid_obstacle.scale = (.5, .5, .5)
		# mid_obstacle.scale = (1, 1, 1)
		# mid_obstacle.scale = (.5, 2, .5)
		# mid_obstacle.scale = (2, .5, .5)
		# mid_obstacle.scale = (30, 30, 1)
		mid_obstacle.scale = (600, 600, 1)

		#mid_obstacle
		
		# mat1 = abj_sd_b_instance.newShader("principled_test_00", "principled", .5, .5, .5)
		mat1 = abj_sd_b_instance.newShader("principled_test_00_mid", "principled", 0, 0, 1)
		mid_obstacle.data.materials.clear()
		mid_obstacle.data.materials.append(mat1)
		bpy.data.materials["principled_test_00_mid"].node_tree.nodes["Principled BSDF"].inputs[1].default_value = 1
		bpy.data.materials["principled_test_00_mid"].node_tree.nodes["Principled BSDF"].inputs[2].default_value = 0.323263
		bpy.data.materials["principled_test_00_mid"].node_tree.nodes["Principled BSDF"].inputs[20].default_value = 1
		mid_obstacle.active_material.displacement_method = 'BOTH'

		mat = bpy.data.materials.get("principled_test_00_mid")
		nodes = mat.node_tree.nodes

		nodes = mat.node_tree.nodes

		checkerNode = nodes.new(type='ShaderNodeTexChecker')
		principledNode = nodes.get("Principled BSDF")
		mat.node_tree.links.new(checkerNode.outputs['Color'], principledNode.inputs['Base Color'])
		checkerNode.inputs[2].default_value = (0, 0, 0, 1)
		# bpy.data.materials["principled_test_grd"].node_tree.nodes["Checker Texture"].inputs[3].default_value = 10
		checkerNode.inputs[3].default_value = 100

		output_node = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
		if not output_node:
			output_node = nodes.new(type='ShaderNodeOutputMaterial')

		# if gaborToPts != 1:
		# 	gabor_node = nodes.new(type='ShaderNodeTexGabor')
		# 	displacement_node = nodes.new(type='ShaderNodeDisplacement')

		# 	mat.node_tree.links.new(gabor_node.outputs['Value'], displacement_node.inputs['Height'])
		# 	mat.node_tree.links.new(displacement_node.outputs['Displacement'], output_node.inputs['Displacement'])
		# 	# displacement_node.inputs[2].default_value = 0.3
		# 	displacement_node.inputs[2].default_value = 0.5

		# 	# bpy.data.materials["principled_test_00"].node_tree.nodes["Gabor Texture"].gabor_type = '3D'
		# 	gabor_node.gabor_type = '3D'

		abj_sd_b_instance.autoArrangeNodes(mat.node_tree)

		# return

		##########  MID OBSTACLE 2 ##############
		##########  MID OBSTACLE 2 ##############
		##########  MID OBSTACLE 2 ##############

		# Spawn Second Mid-Air Intercepting Cube Obstacle
		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.5, 0.0, 4.5))
		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.5, 0.0, 2.75))
		bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.5, 0.0, -10)) #####s
		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.5, 0.0, -100))  
		mid_obstacle2 = bpy.context.active_object
		mid_obstacle2.name = "Mid_Air_Obstacle_2"
		# mid_obstacle.rotation_euler = (0.45, 0.25, 0.78)
		mid_obstacle2.rotation_euler = (0.65, 0.45, -90)
		# mid_obstacle.scale = (2.0, 2.0, 1.5)
		# mid_obstacle.scale = (.5, .5, .5)
		# mid_obstacle.scale = (1, 1, 1)
		# mid_obstacle.scale = (.5, 2, .5)
		# mid_obstacle.scale = (2, .5, .5)
		# mid_obstacle.scale = (30, 30, 1)
		mid_obstacle2.scale = (10, 10, 1)

		# return

		#mid_obstacle 2
		
		# mat1 = abj_sd_b_instance.newShader("principled_test_00", "principled", .5, .5, .5)
		mat1 = abj_sd_b_instance.newShader("principled_test_00_mid2", "principled", 0, 0, 1)
		mid_obstacle.data.materials.clear()
		mid_obstacle.data.materials.append(mat1)
		bpy.data.materials["principled_test_00_mid2"].node_tree.nodes["Principled BSDF"].inputs[1].default_value = 1
		bpy.data.materials["principled_test_00_mid2"].node_tree.nodes["Principled BSDF"].inputs[2].default_value = 0.323263
		bpy.data.materials["principled_test_00_mid2"].node_tree.nodes["Principled BSDF"].inputs[20].default_value = 1
		mid_obstacle.active_material.displacement_method = 'BOTH'

		mat = bpy.data.materials.get("principled_test_00_mid2")
		nodes = mat.node_tree.nodes

		nodes = mat.node_tree.nodes

		checkerNode = nodes.new(type='ShaderNodeTexChecker')
		principledNode = nodes.get("Principled BSDF")
		mat.node_tree.links.new(checkerNode.outputs['Color'], principledNode.inputs['Base Color'])
		checkerNode.inputs[2].default_value = (0, 0, 0, 1)
		# bpy.data.materials["principled_test_grd"].node_tree.nodes["Checker Texture"].inputs[3].default_value = 10
		checkerNode.inputs[3].default_value = 100

		output_node = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
		if not output_node:
			output_node = nodes.new(type='ShaderNodeOutputMaterial')

		# if gaborToPts != 1:
		# 	gabor_node = nodes.new(type='ShaderNodeTexGabor')
		# 	displacement_node = nodes.new(type='ShaderNodeDisplacement')

		# 	mat.node_tree.links.new(gabor_node.outputs['Value'], displacement_node.inputs['Height'])
		# 	mat.node_tree.links.new(displacement_node.outputs['Displacement'], output_node.inputs['Displacement'])
		# 	# displacement_node.inputs[2].default_value = 0.3
		# 	displacement_node.inputs[2].default_value = 0.5

		# 	# bpy.data.materials["principled_test_00"].node_tree.nodes["Gabor Texture"].gabor_type = '3D'
		# 	gabor_node.gabor_type = '3D'

		abj_sd_b_instance.autoArrangeNodes(mat.node_tree)

		# return
		

		##########  MID OBSTACLE 3 ##############
		##########  MID OBSTACLE 3 ##############
		##########  MID OBSTACLE 3 ##############

		# Spawn Second Mid-Air Intercepting Cube Obstacle
		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.5, 0.0, 4.5))
		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.5, 0.0, 2.75))
		bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.5, 0.0, -20)) #####s
		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.5, 0.0, -100))  
		mid_obstacle3 = bpy.context.active_object
		mid_obstacle3.name = "Mid_Air_Obstacle_3"
		# mid_obstacle.rotation_euler = (0.45, 0.25, 0.78)
		# mid_obstacle3.rotation_euler = (0.65, 0.25, -0.28)
		# mid_obstacle.scale = (2.0, 2.0, 1.5)
		# mid_obstacle.scale = (.5, .5, .5)
		# mid_obstacle.scale = (1, 1, 1)
		# mid_obstacle.scale = (.5, 2, .5)
		# mid_obstacle.scale = (2, .5, .5)
		# mid_obstacle.scale = (30, 30, 1)
		mid_obstacle3.scale = (1000, 1000, 1)

		#mid_obstacle
		
		# mat1 = abj_sd_b_instance.newShader("principled_test_00", "principled", .5, .5, .5)
		mat1 = abj_sd_b_instance.newShader("principled_test_00_mid3", "principled", 0, 0, 1)
		mid_obstacle3.data.materials.clear()
		mid_obstacle3.data.materials.append(mat1)
		bpy.data.materials["principled_test_00_mid3"].node_tree.nodes["Principled BSDF"].inputs[1].default_value = 1
		bpy.data.materials["principled_test_00_mid3"].node_tree.nodes["Principled BSDF"].inputs[2].default_value = 0.323263
		bpy.data.materials["principled_test_00_mid3"].node_tree.nodes["Principled BSDF"].inputs[20].default_value = 1
		mid_obstacle3.active_material.displacement_method = 'BOTH'

		mat = bpy.data.materials.get("principled_test_00_mid3")
		nodes = mat.node_tree.nodes

		nodes = mat.node_tree.nodes

		checkerNode = nodes.new(type='ShaderNodeTexChecker')
		principledNode = nodes.get("Principled BSDF")
		mat.node_tree.links.new(checkerNode.outputs['Color'], principledNode.inputs['Base Color'])
		checkerNode.inputs[2].default_value = (0, 0, 0, 1)
		# bpy.data.materials["principled_test_grd"].node_tree.nodes["Checker Texture"].inputs[3].default_value = 10
		checkerNode.inputs[3].default_value = 100

		output_node = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
		if not output_node:
			output_node = nodes.new(type='ShaderNodeOutputMaterial')

		# if gaborToPts != 1:
		# 	gabor_node = nodes.new(type='ShaderNodeTexGabor')
		# 	displacement_node = nodes.new(type='ShaderNodeDisplacement')

		# 	mat.node_tree.links.new(gabor_node.outputs['Value'], displacement_node.inputs['Height'])
		# 	mat.node_tree.links.new(displacement_node.outputs['Displacement'], output_node.inputs['Displacement'])
		# 	# displacement_node.inputs[2].default_value = 0.3
		# 	displacement_node.inputs[2].default_value = 0.5

		# 	# bpy.data.materials["principled_test_00"].node_tree.nodes["Gabor Texture"].gabor_type = '3D'
		# 	gabor_node.gabor_type = '3D'

		abj_sd_b_instance.autoArrangeNodes(mat.node_tree)

		# return

		# return
		
		###########
		# WORLD
		###########

		world = bpy.context.scene.world
		worldtree = world.node_tree
		worldtree.nodes.clear()

		# output_node_world = next((n for n in worldtree.nodes if n.type == 'ShaderNodeOutputWorld'), None)
		# if not output_node:
		# 	output_node_world = worldtree.nodes.new(type='ShaderNodeOutputWorld')

		# output_node = worldtree.nodes.new(type="ShaderNodeOutputWorld")
		# bg_node = worldtree.nodes.new(type="ShaderNodeBackground")

		# output_node_world = worldtree.nodes.new(type="ShaderNodeOutputWorld")
		output_node_world = worldtree.nodes.new('ShaderNodeOutputWorld')

		# node_sky = bpy.ops.node.add_node(use_transform=True, type="ShaderNodeTexSky")
		# node_sky = bpy.ops.node.add_node(use_transform=True, type="ShaderNodeTexSky")
		node_sky = worldtree.nodes.new('ShaderNodeTexSky')
		worldtree.links.new(node_sky.outputs["Color"], output_node_world.inputs["Surface"])

		# bpy.data.worlds["World"].node_tree.nodes["Sky Texture"].sun_size = 0.372541
		# bpy.data.worlds["World"].node_tree.nodes["Sky Texture"].sun_intensity = 21.3
		# bpy.data.worlds["World"].node_tree.nodes["Sky Texture"].sun_rotation = -1.57603
		# node_sky.sun_size = 0.372541
		# node_sky.sun_intensity = 21.3
		node_sky.sun_rotation = 1.65806
		node_sky.sun_elevation = .05

		abj_sd_b_instance.autoArrangeNodes(worldtree)

		bpy.context.view_layer.update()

		abj_sd_b_instance.agxColorSettings_UI()

		# return

		bpy.context.evaluated_depsgraph_get().update()

		def get_jax_transforms(obj):
			mw = obj.matrix_world
			center = jnp.array(mw.to_translation())
			size = jnp.array(obj.dimensions)
			r3 = mw.to_3x3().normalized()
			rot_matrix = jnp.array([[r3[0][0], r3[0][1], r3[0][2]],
									[r3[1][0], r3[1][1], r3[1][2]],
									[r3[2][0], r3[2][1], r3[2][2]]])
			return center, size, rot_matrix

		b1_center, b1_size, b1_rot = get_jax_transforms(cube_floor)
		b2_center, b2_size, b2_rot = get_jax_transforms(mid_obstacle)
		b3_center, b3_size, b3_rot = get_jax_transforms(mid_obstacle2)
		b4_center, b4_size, b4_rot = get_jax_transforms(mid_obstacle3)

		# return

		# ==============================================================================
		# 2. EXTRACT SCENE GEOMETRY TO GLOBAL SCOPE ARRAYS
		# ==============================================================================
		mesh = ball_obj.data
		num_verts = len(mesh.vertices)
		orig_coords = np.zeros((num_verts, 3))
		mesh.vertices.foreach_get("co", orig_coords.ravel())

		# Baseline world coordinates tracking matrix
		sphere_tracker = np.copy(orig_coords)
		sphere_tracker[:, 2] += 7.5  

		# Extract reference normal direction vectors
		ref_normals_numpy = np.zeros((num_verts, 3))
		for v in mesh.vertices:
			ref_normals_numpy[v.index] = np.array(v.normal)

		# Precompute thickness profile cushion (14% of initial relative height profile)
		min_init_z = np.min(orig_coords[:, 2])
		vertex_thickness_numpy = (orig_coords[:, 2] - min_init_z) * 0.14  

		# --- GLOBAL SCOPE JAX ARRAYS ---
		# By assigning these to the global module scope, the JAX functions can read them
		# directly via closure. They are never passed through scan, so they CANNOT flatten or swap!

		STATIC_INIT_POS = jnp.array(sphere_tracker)
		STATIC_CENTER = jnp.mean(STATIC_INIT_POS, axis=0) # Pristine Frame 1 rest center
		STATIC_NORMALS = jnp.array(ref_normals_numpy)
		STATIC_THICKNESS = jnp.array(vertex_thickness_numpy)


		# ==============================================================================
		# 4. EXPLICIT AUTOMATED INDIVIDUAL-ARGUMENT BACKPROPAGATION GRADIENT DESCENT LOOP
		# ==============================================================================

		# --- FIXED: WEIGHT PROFILE EXPERIMENT SWITCHBOARD ---
		# Test Case 1: Heavy Lead Brick -> Mass = 50.0, Drag = 0.15 (Slams down hard, bounces high)
		# Test Case 2: Light Feather Cushion -> Mass = 0.8, Drag = 1.85 (Floats down slowly, stays soft)
		# OBJECT_MASS_VAL = 45.0          
		# DRAG_COEFFICIENT_VAL = 0.25   


		# num_frames = 600
		# num_frames = 1200
		num_frames = 1600
		# num_frames = 2000
		# num_frames = 300
		# num_frames = 300

		# ###########
		# #### SPHERE 01
		# ###########
		# MU_VAL = 1450.0
		# LAM_VAL = 4000.0
		# # DAMPING_VAL = 0.995
		# DAMPING_VAL = 0.990
		# # DT_VAL = 0.01
		# # DT_VAL = 0.008
		# DT_VAL = 0.008
		# # INITIAL_SPIKE_VELOCITY = -30
		# INITIAL_SPIKE_VELOCITY = -30
		# RESTITUTION_VAL = .7
		# OBJECT_MASS_VAL = 400
		# DRAG_COEFFICIENT_VAL = .5


	

		##########
		### SPHERE GOOD
		##########
		MU_VAL = 550.0
		LAM_VAL = 3000.0
		DAMPING_VAL = 0.995
		# DAMPING_VAL = 0.99
		# DAMPING_VAL = 0.98
		# DAMPING_VAL = 0.97 ####
		# DAMPING_VAL = 0.96
		# DAMPING_VAL = 0.95
		# DAMPING_VAL = 0.94
		# DT_VAL = 0.01
		# DT_VAL = 0.008
		DT_VAL = 0.008


		# INITIAL_SPIKE_VELOCITY = -15
		INITIAL_SPIKE_VELOCITY = -30
		# INITIAL_SPIKE_VELOCITY = -60.0
		# INITIAL_SPIKE_VELOCITY = -90


		# INITIAL_SPIKE_VELOCITY = -50
		# RESTITUTION_VAL = .95
		RESTITUTION_VAL = .87
		# RESTITUTION_VAL = .7
		# RESTITUTION_VAL = .5
		OBJECT_MASS_VAL = 300.0
		DRAG_COEFFICIENT_VAL = .05



		# ########
		# # CUBE
		# ########
		# MU_VAL = 1590.0
		# LAM_VAL = 3000.0
		# # DAMPING_VAL = 0.995
		# DAMPING_VAL = 0.99
		# # DT_VAL = 0.01
		# # DT_VAL = 0.008
		# DT_VAL = 0.008
		# INITIAL_SPIKE_VELOCITY = -30
		# # INITIAL_SPIKE_VELOCITY = -10
		# RESTITUTION_VAL = .9
		# OBJECT_MASS_VAL = 400
		# DRAG_COEFFICIENT_VAL = .03






		TARGET_FRAME = 100
		TARGET_HEIGHT = 12.5


		loss_fn = jax.jit(lambda m, l, d, t, v, r, ma, dr: loss_function(m, l, d, t, v, r, ma, dr, STATIC_INIT_POS, STATIC_CENTER, STATIC_NORMALS, STATIC_THICKNESS, TARGET_FRAME, TARGET_HEIGHT))

		# Target index slots securely: 0=mu, 1=lam, 2=damping, 4=velocity, 6=mass
		grad_fn = jax.jit(jax.grad(
			lambda m, l, d, t, v, r, ma, dr: loss_function(m, l, d, t, v, r, ma, dr, STATIC_INIT_POS, STATIC_CENTER, STATIC_NORMALS, STATIC_THICKNESS, TARGET_FRAME, TARGET_HEIGHT),
			argnums=(0, 1, 2, 4, 6)
		))


		print(f"\n[JAX Optimizer] Launching local manual-argument gradient search trajectory tracking...")

		lr_stiffness = 2.5    
		# lr_damping = 4e-4 ##
		# lr_damping = 2
		lr_damping = .1
		# lr_velocity = 4e-1    
		lr_velocity = 10 
		# lr_mass = 0.5         # Gradient search rate for weight scale optimization
		lr_mass = .5         # Gradient search rate for weight scale optimization
		# num_steps = 20
		# num_steps = 10
		# num_steps = 5
		num_steps = 2


		for iteration in range(num_steps):
			continue

			current_loss = loss_fn(MU_VAL, LAM_VAL, DAMPING_VAL, DT_VAL, INITIAL_SPIKE_VELOCITY, RESTITUTION_VAL, OBJECT_MASS_VAL, DRAG_COEFFICIENT_VAL)
			raw_grads = grad_fn(MU_VAL, LAM_VAL, DAMPING_VAL, DT_VAL, INITIAL_SPIKE_VELOCITY, RESTITUTION_VAL, OBJECT_MASS_VAL, DRAG_COEFFICIENT_VAL)
			
			grad_mu, grad_lam, grad_damp, grad_vel, grad_mass = raw_grads
			
			# FIXED: True Sign Fallback. If gradients hit a zero plateau, we use a constant 
			# directional sign push to dynamically kick the parameter out of the flat zone
			step_mu   = jnp.sign(grad_mu) * lr_stiffness if jnp.abs(grad_mu) > 1e-5 else jnp.sign(MU_VAL - 450.0) * lr_stiffness
			step_lam  = jnp.sign(grad_lam) * lr_stiffness if jnp.abs(grad_lam) > 1e-5 else jnp.sign(LAM_VAL - 3500.0) * lr_stiffness
			step_damp = grad_damp * lr_damping if jnp.abs(grad_damp) > 1e-5 else (DAMPING_VAL - 0.94) * lr_damping
			
			# Track velocity derivatives out of dead zones smoothly
			step_vel  = jnp.sign(grad_vel) * lr_velocity if jnp.abs(grad_vel) > 1e-5 else -1.5 * lr_velocity
			step_mass = jnp.sign(grad_mass) * lr_mass      if jnp.abs(grad_mass) > 1e-5 else grad_mass * 2
			
			# Apply individual unrolled updates smoothly
			MU_VAL                 = float(jnp.clip(MU_VAL + step_mu, 200.0, 5000.0))
			LAM_VAL                = float(jnp.clip(LAM_VAL + step_lam, 1000.0, 15000.0))
			DAMPING_VAL            = float(jnp.clip(DAMPING_VAL - step_damp, 0.98, 0.994))
			OBJECT_MASS_VAL        = float(jnp.clip(OBJECT_MASS_VAL + step_mass, 0.1, 200.0)) # Clip weight to valid ranges
			
			# FIXED: Expanded the clipping boundary ceiling completely to support -120.0 m/s
			# INITIAL_SPIKE_VELOCITY = float(jnp.clip(INITIAL_SPIKE_VELOCITY - step_vel, -120.0, -10.0))
			INITIAL_SPIKE_VELOCITY = float(jnp.clip(INITIAL_SPIKE_VELOCITY + step_vel, -120.0, -10.0))
			
			print(f"  Step {iteration+1:02d} -> Loss: {current_loss:.4f} | Mass Weight: {OBJECT_MASS_VAL:.2f} kg | Spike Vel: {INITIAL_SPIKE_VELOCITY:.2f} m/s | Mu: {MU_VAL:.1f} | Lam: {LAM_VAL:.1f} | Damping: {DAMPING_VAL:.4f} Dt: {DT_VAL:.4f}")
			
		print("\n[JAX Engine] Optimization target achieved! Pulling verified trajectory memory buffer...")

		# num_frames = 600
		
		
		# final_trajectory_matrix = jit_simulation_engine(
		# 	MU_VAL, LAM_VAL, DAMPING_VAL, DT_VAL, INITIAL_SPIKE_VELOCITY, RESTITUTION_VAL, OBJECT_MASS_VAL, DRAG_COEFFICIENT_VAL, STATIC_INIT_POS, STATIC_CENTER, STATIC_NORMALS, STATIC_THICKNESS, num_frames
		# )

		# final_trajectory_matrix = jit_simulation_engine(MU_VAL, LAM_VAL, DAMPING_VAL, DT_VAL, INITIAL_SPIKE_VELOCITY, RESTITUTION_VAL,OBJECT_MASS_VAL, DRAG_COEFFICIENT_VAL, STATIC_INIT_POS, STATIC_CENTER,STATIC_NORMALS, STATIC_THICKNESS, b1_center, b1_size, b1_rot, b2_center, b2_size, b2_rot, num_frames)
		# final_trajectory_matrix = jit_simulation_engine(MU_VAL, LAM_VAL, DAMPING_VAL, DT_VAL, INITIAL_SPIKE_VELOCITY, RESTITUTION_VAL, OBJECT_MASS_VAL, DRAG_COEFFICIENT_VAL, STATIC_INIT_POS, STATIC_CENTER, STATIC_NORMALS, STATIC_THICKNESS, b1_center, b1_size, b1_rot, b2_center, b2_size, b2_rot, b3_center, b3_size, b3_rot, num_frames)
		final_trajectory_matrix = jit_simulation_engine(MU_VAL, LAM_VAL, DAMPING_VAL, DT_VAL, INITIAL_SPIKE_VELOCITY, RESTITUTION_VAL, OBJECT_MASS_VAL, DRAG_COEFFICIENT_VAL, STATIC_INIT_POS, STATIC_CENTER, STATIC_NORMALS, STATIC_THICKNESS, b1_center, b1_size, b1_rot, b2_center, b2_size, b2_rot, b3_center, b3_size, b3_rot, b4_center, b4_size, b4_rot, num_frames)

		baked_frames_positions = np.array(final_trajectory_matrix)
		depsgraph = bpy.context.evaluated_depsgraph_get()

		print("[Blender Pipeline] Writing exact JAX position memory to timeline Shape Keys...")

		for idx, frame in enumerate(range(1, num_frames + 1)):
			print('frame_idx = ', idx)
			bpy.context.scene.frame_set(frame)
			
			frame_coords = baked_frames_positions[frame - 1]
			baked_local_coords = np.copy(frame_coords)
			baked_local_coords[:, 2] -= 7.5
			
			skey = ball_obj.shape_key_add(name=f"Frame_{frame}", from_mix=False)
			skey.data.foreach_set("co", baked_local_coords.ravel())
			
			skey.value = 1.0
			skey.keyframe_insert(data_path="value", frame=frame)
			if frame > 1:
				skey.value = 0.0
				skey.keyframe_insert(data_path="value", frame=frame - 1)
				prev_key = ball_obj.data.shape_keys.key_blocks[f"Frame_{frame-1}"]
				prev_key.value = 0.0
				prev_key.keyframe_insert(data_path="value", frame=frame)
				
			depsgraph.update()

		bpy.context.scene.frame_set(1)
		print("\n[Bake Finished] Unified engine execution completed successfully! Press Spacebar.")

		# bpy.context.scene.render.fps = 240
		bpy.context.scene.render.fps = 120
		# bpy.context.scene.render.fps = 120 * 4
		bpy.context.scene.frame_end = num_frames


	def bounce_diff_05(self, abj_sd_b_instance):
		# bpy.context.scene.render.engine = 'CYCLES'
		bpy.context.scene.render.engine = 'BLENDER_EEVEE'
		
		bpy.context.scene.cycles.device = 'GPU'

		bpy.context.scene.cycles.samples = 64
		bpy.context.scene.cycles.denoising_use_gpu = True

		# ==============================================================================
		# 1. SCENE CLEANUP & BLENDER ENVIRONMENT LAYOUT SETUP
		# ==============================================================================
		print("Initializing unified JAX Unified Bounce-and-Crush Continuum Engine...")

		for name in ["Rubber_Ball", "Voxel_Collision_Cube"]:
			if name in bpy.data.objects:
				bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)

		# SPHERE_RES = 8  # High resolution captures both fluid ripples and flat cushion folds
		# SPHERE_RES = 16  # High resolution captures both fluid ripples and flat cushion folds
		SPHERE_RES = 32  # High resolution captures both fluid ripples and flat cushion folds ### !!!!!!!
		# SPHERE_RES = 40  # High resolution captures both fluid ripples and flat cushion folds
		# SPHERE_RES = 48  # High resolution captures both fluid ripples and flat cushion folds
		# SPHERE_RES = 64  # High resolution captures both fluid ripples and flat cushion folds
		# SPHERE_RES = 100  # High resolution captures both fluid ripples and flat cushion folds
		# SPHERE_RES = 128  # High resolution captures both fluid ripples and flat cushion folds
		# SPHERE_RES = 256  # High resolution captures both fluid ripples and flat cushion folds

		# Spawn target sphere at Z = 7.5
		# bpy.ops.mesh.primitive_uv_sphere_add(radius=1.8, location=(0.0, 0.0, 7.5), segments=SPHERE_RES, ring_count=SPHERE_RES)
		bpy.ops.mesh.primitive_ico_sphere_add(radius=1.8, location=(0.0, 0.0, 7.5), subdivisions=4)
		# bpy.ops.mesh.primitive_ico_sphere_add(radius=1.8, location=(0.0, 0.0, 0), subdivisions=4)
		# bpy.ops.mesh.primitive_cube_add(location=(0.0, 0.0, 7.5), size=6)

		ball_obj = bpy.context.active_object
		ball_obj.name = "Rubber_Ball"

		ball_obj.shape_key_add(name="Basis", from_mix=False)
		bpy.ops.object.shade_smooth()

		ball_obj = bpy.context.active_object

		# return

		gaborToPts = 0
		# gaborToPts = 1

		# if gaborToPts == 1:
		# 	self.bakeShaderToPts(ball_obj) ######

		mat1 = abj_sd_b_instance.newShader("principled_test_00", "principled", 1, 0, 0)
		bpy.context.active_object.data.materials.clear()
		bpy.context.active_object.data.materials.append(mat1)
		bpy.data.materials["principled_test_00"].node_tree.nodes["Principled BSDF"].inputs[1].default_value = 1
		bpy.data.materials["principled_test_00"].node_tree.nodes["Principled BSDF"].inputs[2].default_value = 0.323263
		bpy.data.materials["principled_test_00"].node_tree.nodes["Principled BSDF"].inputs[20].default_value = 1
		ball_obj.active_material.displacement_method = 'BOTH'

		mat = bpy.data.materials.get("principled_test_00")
		nodes = mat.node_tree.nodes

		nodes = mat.node_tree.nodes

		output_node = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
		if not output_node:
			output_node = nodes.new(type='ShaderNodeOutputMaterial')

		if gaborToPts != 1:
			gabor_node = nodes.new(type='ShaderNodeTexGabor')
			displacement_node = nodes.new(type='ShaderNodeDisplacement')

			mat.node_tree.links.new(gabor_node.outputs['Value'], displacement_node.inputs['Height'])
			mat.node_tree.links.new(displacement_node.outputs['Displacement'], output_node.inputs['Displacement'])
			# displacement_node.inputs[2].default_value = 0.3
			displacement_node.inputs[2].default_value = 0.5

			# bpy.data.materials["principled_test_00"].node_tree.nodes["Gabor Texture"].gabor_type = '3D'
			gabor_node.gabor_type = '3D'

			# checkerNode = nodes.new(type='ShaderNodeTexChecker')
			# principledNode = nodes.get("Principled BSDF")
			# mat.node_tree.links.new(checkerNode.outputs['Color'], principledNode.inputs['Base Color'])
			# checkerNode.inputs[2].default_value = (0, 0, 0, 1)


		abj_sd_b_instance.autoArrangeNodes(mat.node_tree)

		######## CUBE FLOOR ###########
		######## CUBE FLOOR ###########
		######## CUBE FLOOR ###########

		# Spawn Cube Ground Floor Object
		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 1.5))
		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -20))
		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -15))
		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -30))


		# bpy.ops.mesh.primitive_ico_sphere_add(radius=6, location=(0.0, 0.0, 7.5), subdivisions=4)
		# bpy.ops.mesh.primitive_ico_sphere_add(radius=48, location=(0.0, 0.0, -75.5), subdivisions=4)

		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -40))
		bpy.ops.mesh.primitive_cube_add(size=1.0, location=(15.0, 0.0, -40))
		cube_floor = bpy.context.active_object
		cube_floor.name = "Voxel_Collision_Cube"
		cube_floor.scale = (1200.0, 1200.0, 1.0)
		cube_floor.rotation_euler = (0.0, 0.45, 0.0)

		# cube_floor = bpy.context.active_object

		mat1 = abj_sd_b_instance.newShader("principled_test_00_grd", "principled", .2, 0, 0)
		cube_floor.data.materials.clear()
		cube_floor.data.materials.append(mat1)
		bpy.data.materials["principled_test_00_grd"].node_tree.nodes["Principled BSDF"].inputs[1].default_value = 1
		bpy.data.materials["principled_test_00_grd"].node_tree.nodes["Principled BSDF"].inputs[2].default_value = 0.323263
		bpy.data.materials["principled_test_00_grd"].node_tree.nodes["Principled BSDF"].inputs[20].default_value = 1
		cube_floor.active_material.displacement_method = 'BOTH'

		mat = bpy.data.materials.get("principled_test_00_grd")
		nodes = mat.node_tree.nodes

		nodes = mat.node_tree.nodes





		checkerNode = nodes.new(type='ShaderNodeTexChecker')
		principledNode = nodes.get("Principled BSDF")
		mat.node_tree.links.new(checkerNode.outputs['Color'], principledNode.inputs['Base Color'])
		checkerNode.inputs[2].default_value = (0, 0, 0, 1)
		# bpy.data.materials["principled_test_grd"].node_tree.nodes["Checker Texture"].inputs[3].default_value = 10
		checkerNode.inputs[3].default_value = 100






		output_node = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
		if not output_node:
			output_node = nodes.new(type='ShaderNodeOutputMaterial')

		abj_sd_b_instance.autoArrangeNodes(mat.node_tree)

		##########  MID OBSTACLE ##############
		##########  MID OBSTACLE ##############
		##########  MID OBSTACLE ##############

		# Spawn Second Mid-Air Intercepting Cube Obstacle
		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.5, 0.0, 4.5))
		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.5, 0.0, 2.75))
		bpy.ops.mesh.primitive_cube_add(size=1.0, location=(10, 0.0, -60)) #####s
		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.5, 0.0, -100))  
		mid_obstacle = bpy.context.active_object
		mid_obstacle.name = "Mid_Air_Obstacle"
		# mid_obstacle.rotation_euler = (0.45, 0.25, 0.78)
		# mid_obstacle.rotation_euler = (0.65, -0.45, 0) #####
		mid_obstacle.rotation_euler = (0.65, -0.45, -45)
		# mid_obstacle.scale = (2.0, 2.0, 1.5)
		# mid_obstacle.scale = (.5, .5, .5)
		# mid_obstacle.scale = (1, 1, 1)
		# mid_obstacle.scale = (.5, 2, .5)
		# mid_obstacle.scale = (2, .5, .5)
		# mid_obstacle.scale = (30, 30, 1)
		mid_obstacle.scale = (600, 600, 1)

		#mid_obstacle
		
		# mat1 = abj_sd_b_instance.newShader("principled_test_00", "principled", .5, .5, .5)
		mat1 = abj_sd_b_instance.newShader("principled_test_00_mid", "principled", 0, 0, 1)
		mid_obstacle.data.materials.clear()
		mid_obstacle.data.materials.append(mat1)
		bpy.data.materials["principled_test_00_mid"].node_tree.nodes["Principled BSDF"].inputs[1].default_value = 1
		bpy.data.materials["principled_test_00_mid"].node_tree.nodes["Principled BSDF"].inputs[2].default_value = 0.323263
		bpy.data.materials["principled_test_00_mid"].node_tree.nodes["Principled BSDF"].inputs[20].default_value = 1
		mid_obstacle.active_material.displacement_method = 'BOTH'

		mat = bpy.data.materials.get("principled_test_00_mid")
		nodes = mat.node_tree.nodes

		nodes = mat.node_tree.nodes

		checkerNode = nodes.new(type='ShaderNodeTexChecker')
		principledNode = nodes.get("Principled BSDF")
		mat.node_tree.links.new(checkerNode.outputs['Color'], principledNode.inputs['Base Color'])
		checkerNode.inputs[2].default_value = (0, 0, 0, 1)
		# bpy.data.materials["principled_test_grd"].node_tree.nodes["Checker Texture"].inputs[3].default_value = 10
		checkerNode.inputs[3].default_value = 100

		output_node = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
		if not output_node:
			output_node = nodes.new(type='ShaderNodeOutputMaterial')

		# if gaborToPts != 1:
		# 	gabor_node = nodes.new(type='ShaderNodeTexGabor')
		# 	displacement_node = nodes.new(type='ShaderNodeDisplacement')

		# 	mat.node_tree.links.new(gabor_node.outputs['Value'], displacement_node.inputs['Height'])
		# 	mat.node_tree.links.new(displacement_node.outputs['Displacement'], output_node.inputs['Displacement'])
		# 	# displacement_node.inputs[2].default_value = 0.3
		# 	displacement_node.inputs[2].default_value = 0.5

		# 	# bpy.data.materials["principled_test_00"].node_tree.nodes["Gabor Texture"].gabor_type = '3D'
		# 	gabor_node.gabor_type = '3D'

		abj_sd_b_instance.autoArrangeNodes(mat.node_tree)

		# return

		##########  MID OBSTACLE 2 ##############
		##########  MID OBSTACLE 2 ##############
		##########  MID OBSTACLE 2 ##############

		# Spawn Second Mid-Air Intercepting Cube Obstacle
		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.5, 0.0, 4.5))
		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.5, 0.0, 2.75))
		bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.5, 0.0, -10)) #####s
		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.5, 0.0, -100))  
		mid_obstacle2 = bpy.context.active_object
		mid_obstacle2.name = "Mid_Air_Obstacle_2"
		# mid_obstacle.rotation_euler = (0.45, 0.25, 0.78)
		mid_obstacle2.rotation_euler = (0.65, 0.45, -90)
		# mid_obstacle.scale = (2.0, 2.0, 1.5)
		# mid_obstacle.scale = (.5, .5, .5)
		# mid_obstacle.scale = (1, 1, 1)
		# mid_obstacle.scale = (.5, 2, .5)
		# mid_obstacle.scale = (2, .5, .5)
		# mid_obstacle.scale = (30, 30, 1)
		mid_obstacle2.scale = (10, 10, 1)

		# return

		#mid_obstacle 2
		
		# mat1 = abj_sd_b_instance.newShader("principled_test_00", "principled", .5, .5, .5)
		mat1 = abj_sd_b_instance.newShader("principled_test_00_mid2", "principled", 0, 0, 1)
		mid_obstacle.data.materials.clear()
		mid_obstacle.data.materials.append(mat1)
		bpy.data.materials["principled_test_00_mid2"].node_tree.nodes["Principled BSDF"].inputs[1].default_value = 1
		bpy.data.materials["principled_test_00_mid2"].node_tree.nodes["Principled BSDF"].inputs[2].default_value = 0.323263
		bpy.data.materials["principled_test_00_mid2"].node_tree.nodes["Principled BSDF"].inputs[20].default_value = 1
		mid_obstacle.active_material.displacement_method = 'BOTH'

		mat = bpy.data.materials.get("principled_test_00_mid2")
		nodes = mat.node_tree.nodes

		nodes = mat.node_tree.nodes

		checkerNode = nodes.new(type='ShaderNodeTexChecker')
		principledNode = nodes.get("Principled BSDF")
		mat.node_tree.links.new(checkerNode.outputs['Color'], principledNode.inputs['Base Color'])
		checkerNode.inputs[2].default_value = (0, 0, 0, 1)
		# bpy.data.materials["principled_test_grd"].node_tree.nodes["Checker Texture"].inputs[3].default_value = 10
		checkerNode.inputs[3].default_value = 100

		output_node = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
		if not output_node:
			output_node = nodes.new(type='ShaderNodeOutputMaterial')

		# if gaborToPts != 1:
		# 	gabor_node = nodes.new(type='ShaderNodeTexGabor')
		# 	displacement_node = nodes.new(type='ShaderNodeDisplacement')

		# 	mat.node_tree.links.new(gabor_node.outputs['Value'], displacement_node.inputs['Height'])
		# 	mat.node_tree.links.new(displacement_node.outputs['Displacement'], output_node.inputs['Displacement'])
		# 	# displacement_node.inputs[2].default_value = 0.3
		# 	displacement_node.inputs[2].default_value = 0.5

		# 	# bpy.data.materials["principled_test_00"].node_tree.nodes["Gabor Texture"].gabor_type = '3D'
		# 	gabor_node.gabor_type = '3D'

		abj_sd_b_instance.autoArrangeNodes(mat.node_tree)

		# return
		

		##########  MID OBSTACLE 3 ##############
		##########  MID OBSTACLE 3 ##############
		##########  MID OBSTACLE 3 ##############

		# Spawn Second Mid-Air Intercepting Cube Obstacle
		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.5, 0.0, 4.5))
		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.5, 0.0, 2.75))
		bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.5, 0.0, -20)) #####s
		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.5, 0.0, -100))  
		mid_obstacle3 = bpy.context.active_object
		mid_obstacle3.name = "Mid_Air_Obstacle_3"
		# mid_obstacle.rotation_euler = (0.45, 0.25, 0.78)
		# mid_obstacle3.rotation_euler = (0.65, 0.25, -0.28)
		# mid_obstacle.scale = (2.0, 2.0, 1.5)
		# mid_obstacle.scale = (.5, .5, .5)
		# mid_obstacle.scale = (1, 1, 1)
		# mid_obstacle.scale = (.5, 2, .5)
		# mid_obstacle.scale = (2, .5, .5)
		# mid_obstacle.scale = (30, 30, 1)
		mid_obstacle3.scale = (1000, 1000, 1)

		#mid_obstacle
		
		# mat1 = abj_sd_b_instance.newShader("principled_test_00", "principled", .5, .5, .5)
		mat1 = abj_sd_b_instance.newShader("principled_test_00_mid3", "principled", 0, 0, 1)
		mid_obstacle3.data.materials.clear()
		mid_obstacle3.data.materials.append(mat1)
		bpy.data.materials["principled_test_00_mid3"].node_tree.nodes["Principled BSDF"].inputs[1].default_value = 1
		bpy.data.materials["principled_test_00_mid3"].node_tree.nodes["Principled BSDF"].inputs[2].default_value = 0.323263
		bpy.data.materials["principled_test_00_mid3"].node_tree.nodes["Principled BSDF"].inputs[20].default_value = 1
		mid_obstacle3.active_material.displacement_method = 'BOTH'

		mat = bpy.data.materials.get("principled_test_00_mid3")
		nodes = mat.node_tree.nodes

		nodes = mat.node_tree.nodes

		checkerNode = nodes.new(type='ShaderNodeTexChecker')
		principledNode = nodes.get("Principled BSDF")
		mat.node_tree.links.new(checkerNode.outputs['Color'], principledNode.inputs['Base Color'])
		checkerNode.inputs[2].default_value = (0, 0, 0, 1)
		# bpy.data.materials["principled_test_grd"].node_tree.nodes["Checker Texture"].inputs[3].default_value = 10
		checkerNode.inputs[3].default_value = 100

		output_node = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
		if not output_node:
			output_node = nodes.new(type='ShaderNodeOutputMaterial')

		# if gaborToPts != 1:
		# 	gabor_node = nodes.new(type='ShaderNodeTexGabor')
		# 	displacement_node = nodes.new(type='ShaderNodeDisplacement')

		# 	mat.node_tree.links.new(gabor_node.outputs['Value'], displacement_node.inputs['Height'])
		# 	mat.node_tree.links.new(displacement_node.outputs['Displacement'], output_node.inputs['Displacement'])
		# 	# displacement_node.inputs[2].default_value = 0.3
		# 	displacement_node.inputs[2].default_value = 0.5

		# 	# bpy.data.materials["principled_test_00"].node_tree.nodes["Gabor Texture"].gabor_type = '3D'
		# 	gabor_node.gabor_type = '3D'

		abj_sd_b_instance.autoArrangeNodes(mat.node_tree)

		# return

		# return
		
		###########
		# WORLD
		###########

		world = bpy.context.scene.world
		worldtree = world.node_tree
		worldtree.nodes.clear()

		# output_node_world = next((n for n in worldtree.nodes if n.type == 'ShaderNodeOutputWorld'), None)
		# if not output_node:
		# 	output_node_world = worldtree.nodes.new(type='ShaderNodeOutputWorld')

		# output_node = worldtree.nodes.new(type="ShaderNodeOutputWorld")
		# bg_node = worldtree.nodes.new(type="ShaderNodeBackground")

		# output_node_world = worldtree.nodes.new(type="ShaderNodeOutputWorld")
		output_node_world = worldtree.nodes.new('ShaderNodeOutputWorld')

		# node_sky = bpy.ops.node.add_node(use_transform=True, type="ShaderNodeTexSky")
		# node_sky = bpy.ops.node.add_node(use_transform=True, type="ShaderNodeTexSky")
		node_sky = worldtree.nodes.new('ShaderNodeTexSky')
		worldtree.links.new(node_sky.outputs["Color"], output_node_world.inputs["Surface"])

		# bpy.data.worlds["World"].node_tree.nodes["Sky Texture"].sun_size = 0.372541
		# bpy.data.worlds["World"].node_tree.nodes["Sky Texture"].sun_intensity = 21.3
		# bpy.data.worlds["World"].node_tree.nodes["Sky Texture"].sun_rotation = -1.57603
		# node_sky.sun_size = 0.372541
		# node_sky.sun_intensity = 21.3
		node_sky.sun_rotation = 1.65806
		node_sky.sun_elevation = .05

		abj_sd_b_instance.autoArrangeNodes(worldtree)

		bpy.context.view_layer.update()

		abj_sd_b_instance.agxColorSettings_UI()

		# return

		bpy.context.evaluated_depsgraph_get().update()

		def get_jax_transforms(obj):
			mw = obj.matrix_world
			center = jnp.array(mw.to_translation())
			size = jnp.array(obj.dimensions)
			r3 = mw.to_3x3().normalized()
			rot_matrix = jnp.array([[r3[0][0], r3[0][1], r3[0][2]],
									[r3[1][0], r3[1][1], r3[1][2]],
									[r3[2][0], r3[2][1], r3[2][2]]])
			return center, size, rot_matrix


		box_centers, box_sizes, box_rots = stack_boxes([
			get_jax_transforms(cube_floor),
			get_jax_transforms(mid_obstacle),
			get_jax_transforms(mid_obstacle2),
			get_jax_transforms(mid_obstacle3)])

		# ==============================================================================
		# 2. EXTRACT SCENE GEOMETRY TO GLOBAL SCOPE ARRAYS
		# ==============================================================================
		mesh = ball_obj.data
		num_verts = len(mesh.vertices)
		orig_coords = np.zeros((num_verts, 3))
		mesh.vertices.foreach_get("co", orig_coords.ravel())

		# Triangle indices for the volume-preservation term (ico sphere = all triangles)
		faces_numpy = np.array([list(p.vertices) for p in mesh.polygons], dtype=np.int32)
		STATIC_FACES = jnp.array(faces_numpy)

		sphere_tracker = np.copy(orig_coords)
		sphere_tracker[:, 2] += 7.5

		ref_normals_numpy = np.zeros((num_verts, 3))
		for v in mesh.vertices:
			ref_normals_numpy[v.index] = np.array(v.normal)

		min_init_z = np.min(orig_coords[:, 2])
		vertex_thickness_numpy = (orig_coords[:, 2] - min_init_z) * 0.14

		STATIC_INIT_POS = jnp.array(sphere_tracker)
		STATIC_CENTER = jnp.mean(STATIC_INIT_POS, axis=0)
		STATIC_NORMALS = jnp.array(ref_normals_numpy)
		STATIC_THICKNESS = jnp.array(vertex_thickness_numpy)

		num_frames = 1600

		MU_VAL = 550.0
		LAM_VAL = 3000.0
		DAMPING_VAL = 0.995
		DT_VAL = 0.008
		INITIAL_SPIKE_VELOCITY = -30
		# RESTITUTION_VAL = .87
		RESTITUTION_VAL = .5
		OBJECT_MASS_VAL = 300.0
		DRAG_COEFFICIENT_VAL = .05
		TARGET_FRAME = 100
		TARGET_HEIGHT = 12.5

		final_trajectory_matrix = jit_simulation_engine(
			MU_VAL, LAM_VAL, DAMPING_VAL, DT_VAL, INITIAL_SPIKE_VELOCITY, RESTITUTION_VAL,
			OBJECT_MASS_VAL, DRAG_COEFFICIENT_VAL, STATIC_INIT_POS, STATIC_CENTER,
			STATIC_NORMALS, STATIC_THICKNESS, STATIC_FACES,
			box_centers, box_sizes, box_rots, num_frames)

		'''

		b1_center, b1_size, b1_rot = get_jax_transforms(cube_floor)
		b2_center, b2_size, b2_rot = get_jax_transforms(mid_obstacle)
		b3_center, b3_size, b3_rot = get_jax_transforms(mid_obstacle2)
		b4_center, b4_size, b4_rot = get_jax_transforms(mid_obstacle3)

		# ==============================================================================
		# 2. EXTRACT SCENE GEOMETRY TO GLOBAL SCOPE ARRAYS
		# ==============================================================================
		mesh = ball_obj.data
		num_verts = len(mesh.vertices)
		orig_coords = np.zeros((num_verts, 3))
		mesh.vertices.foreach_get("co", orig_coords.ravel())

		# Baseline world coordinates tracking matrix
		sphere_tracker = np.copy(orig_coords)
		sphere_tracker[:, 2] += 7.5  

		# Extract reference normal direction vectors
		ref_normals_numpy = np.zeros((num_verts, 3))
		for v in mesh.vertices:
			ref_normals_numpy[v.index] = np.array(v.normal)

		# Precompute thickness profile cushion (14% of initial relative height profile)
		min_init_z = np.min(orig_coords[:, 2])
		vertex_thickness_numpy = (orig_coords[:, 2] - min_init_z) * 0.14  

		# --- GLOBAL SCOPE JAX ARRAYS ---
		# By assigning these to the global module scope, the JAX functions can read them
		# directly via closure. They are never passed through scan, so they CANNOT flatten or swap!

		STATIC_INIT_POS = jnp.array(sphere_tracker)
		STATIC_CENTER = jnp.mean(STATIC_INIT_POS, axis=0) # Pristine Frame 1 rest center
		STATIC_NORMALS = jnp.array(ref_normals_numpy)
		STATIC_THICKNESS = jnp.array(vertex_thickness_numpy)


		# ==============================================================================
		# 4. EXPLICIT AUTOMATED INDIVIDUAL-ARGUMENT BACKPROPAGATION GRADIENT DESCENT LOOP
		# ==============================================================================

		# --- FIXED: WEIGHT PROFILE EXPERIMENT SWITCHBOARD ---
		# Test Case 1: Heavy Lead Brick -> Mass = 50.0, Drag = 0.15 (Slams down hard, bounces high)
		# Test Case 2: Light Feather Cushion -> Mass = 0.8, Drag = 1.85 (Floats down slowly, stays soft)
		# OBJECT_MASS_VAL = 45.0          
		# DRAG_COEFFICIENT_VAL = 0.25   


		# num_frames = 600
		# num_frames = 1200
		num_frames = 1600
		# num_frames = 2000
		# num_frames = 300
		# num_frames = 300



	

		##########
		### SPHERE GOOD
		##########
		MU_VAL = 550.0
		LAM_VAL = 3000.0
		DAMPING_VAL = 0.995
		DT_VAL = 0.008
		INITIAL_SPIKE_VELOCITY = -30
		RESTITUTION_VAL = .87
		OBJECT_MASS_VAL = 300.0
		DRAG_COEFFICIENT_VAL = .05

		TARGET_FRAME = 100
		TARGET_HEIGHT = 12.5
		'''

		'''
		loss_fn = jax.jit(lambda m, l, d, t, v, r, ma, dr: loss_function(m, l, d, t, v, r, ma, dr, STATIC_INIT_POS, STATIC_CENTER, STATIC_NORMALS, STATIC_THICKNESS, TARGET_FRAME, TARGET_HEIGHT))

		# Target index slots securely: 0=mu, 1=lam, 2=damping, 4=velocity, 6=mass
		grad_fn = jax.jit(jax.grad(
			lambda m, l, d, t, v, r, ma, dr: loss_function(m, l, d, t, v, r, ma, dr, STATIC_INIT_POS, STATIC_CENTER, STATIC_NORMALS, STATIC_THICKNESS, TARGET_FRAME, TARGET_HEIGHT),
			argnums=(0, 1, 2, 4, 6)
		))


		print(f"\n[JAX Optimizer] Launching local manual-argument gradient search trajectory tracking...")

		lr_stiffness = 2.5    
		# lr_damping = 4e-4 ##
		# lr_damping = 2
		lr_damping = .1
		# lr_velocity = 4e-1    
		lr_velocity = 10 
		# lr_mass = 0.5         # Gradient search rate for weight scale optimization
		lr_mass = .5         # Gradient search rate for weight scale optimization
		# num_steps = 20
		# num_steps = 10
		# num_steps = 5
		num_steps = 2


		for iteration in range(num_steps):
			continue

			current_loss = loss_fn(MU_VAL, LAM_VAL, DAMPING_VAL, DT_VAL, INITIAL_SPIKE_VELOCITY, RESTITUTION_VAL, OBJECT_MASS_VAL, DRAG_COEFFICIENT_VAL)
			raw_grads = grad_fn(MU_VAL, LAM_VAL, DAMPING_VAL, DT_VAL, INITIAL_SPIKE_VELOCITY, RESTITUTION_VAL, OBJECT_MASS_VAL, DRAG_COEFFICIENT_VAL)
			
			grad_mu, grad_lam, grad_damp, grad_vel, grad_mass = raw_grads
			
			# FIXED: True Sign Fallback. If gradients hit a zero plateau, we use a constant 
			# directional sign push to dynamically kick the parameter out of the flat zone
			step_mu   = jnp.sign(grad_mu) * lr_stiffness if jnp.abs(grad_mu) > 1e-5 else jnp.sign(MU_VAL - 450.0) * lr_stiffness
			step_lam  = jnp.sign(grad_lam) * lr_stiffness if jnp.abs(grad_lam) > 1e-5 else jnp.sign(LAM_VAL - 3500.0) * lr_stiffness
			step_damp = grad_damp * lr_damping if jnp.abs(grad_damp) > 1e-5 else (DAMPING_VAL - 0.94) * lr_damping
			
			# Track velocity derivatives out of dead zones smoothly
			step_vel  = jnp.sign(grad_vel) * lr_velocity if jnp.abs(grad_vel) > 1e-5 else -1.5 * lr_velocity
			step_mass = jnp.sign(grad_mass) * lr_mass      if jnp.abs(grad_mass) > 1e-5 else grad_mass * 2
			
			# Apply individual unrolled updates smoothly
			MU_VAL                 = float(jnp.clip(MU_VAL + step_mu, 200.0, 5000.0))
			LAM_VAL                = float(jnp.clip(LAM_VAL + step_lam, 1000.0, 15000.0))
			DAMPING_VAL            = float(jnp.clip(DAMPING_VAL - step_damp, 0.98, 0.994))
			OBJECT_MASS_VAL        = float(jnp.clip(OBJECT_MASS_VAL + step_mass, 0.1, 200.0)) # Clip weight to valid ranges
			
			# FIXED: Expanded the clipping boundary ceiling completely to support -120.0 m/s
			# INITIAL_SPIKE_VELOCITY = float(jnp.clip(INITIAL_SPIKE_VELOCITY - step_vel, -120.0, -10.0))
			INITIAL_SPIKE_VELOCITY = float(jnp.clip(INITIAL_SPIKE_VELOCITY + step_vel, -120.0, -10.0))
			
			print(f"  Step {iteration+1:02d} -> Loss: {current_loss:.4f} | Mass Weight: {OBJECT_MASS_VAL:.2f} kg | Spike Vel: {INITIAL_SPIKE_VELOCITY:.2f} m/s | Mu: {MU_VAL:.1f} | Lam: {LAM_VAL:.1f} | Damping: {DAMPING_VAL:.4f} Dt: {DT_VAL:.4f}")
			
		print("\n[JAX Engine] Optimization target achieved! Pulling verified trajectory memory buffer...")

		'''

		# num_frames = 600
		
		
		# final_trajectory_matrix = jit_simulation_engine(
		# 	MU_VAL, LAM_VAL, DAMPING_VAL, DT_VAL, INITIAL_SPIKE_VELOCITY, RESTITUTION_VAL, OBJECT_MASS_VAL, DRAG_COEFFICIENT_VAL, STATIC_INIT_POS, STATIC_CENTER, STATIC_NORMALS, STATIC_THICKNESS, num_frames
		# )

		# final_trajectory_matrix = jit_simulation_engine(MU_VAL, LAM_VAL, DAMPING_VAL, DT_VAL, INITIAL_SPIKE_VELOCITY, RESTITUTION_VAL,OBJECT_MASS_VAL, DRAG_COEFFICIENT_VAL, STATIC_INIT_POS, STATIC_CENTER,STATIC_NORMALS, STATIC_THICKNESS, b1_center, b1_size, b1_rot, b2_center, b2_size, b2_rot, num_frames)
		# final_trajectory_matrix = jit_simulation_engine(MU_VAL, LAM_VAL, DAMPING_VAL, DT_VAL, INITIAL_SPIKE_VELOCITY, RESTITUTION_VAL, OBJECT_MASS_VAL, DRAG_COEFFICIENT_VAL, STATIC_INIT_POS, STATIC_CENTER, STATIC_NORMALS, STATIC_THICKNESS, b1_center, b1_size, b1_rot, b2_center, b2_size, b2_rot, b3_center, b3_size, b3_rot, num_frames)
		
		
		# final_trajectory_matrix = jit_simulation_engine(MU_VAL, LAM_VAL, DAMPING_VAL, DT_VAL, INITIAL_SPIKE_VELOCITY, RESTITUTION_VAL, OBJECT_MASS_VAL, DRAG_COEFFICIENT_VAL, STATIC_INIT_POS, STATIC_CENTER, STATIC_NORMALS, STATIC_THICKNESS, b1_center, b1_size, b1_rot, b2_center, b2_size, b2_rot, b3_center, b3_size, b3_rot, b4_center, b4_size, b4_rot, num_frames) #OLD 24 (0 base)

		baked_frames_positions = np.array(final_trajectory_matrix)
		depsgraph = bpy.context.evaluated_depsgraph_get()

		print("[Blender Pipeline] Writing exact JAX position memory to timeline Shape Keys...")

		for idx, frame in enumerate(range(1, num_frames + 1)):
			print('frame_idx = ', idx)
			bpy.context.scene.frame_set(frame)
			
			frame_coords = baked_frames_positions[frame - 1]
			baked_local_coords = np.copy(frame_coords)
			baked_local_coords[:, 2] -= 7.5
			
			skey = ball_obj.shape_key_add(name=f"Frame_{frame}", from_mix=False)
			skey.data.foreach_set("co", baked_local_coords.ravel())
			
			skey.value = 1.0
			skey.keyframe_insert(data_path="value", frame=frame)
			if frame > 1:
				skey.value = 0.0
				skey.keyframe_insert(data_path="value", frame=frame - 1)
				prev_key = ball_obj.data.shape_keys.key_blocks[f"Frame_{frame-1}"]
				prev_key.value = 0.0
				prev_key.keyframe_insert(data_path="value", frame=frame)
				
			depsgraph.update()

		bpy.context.scene.frame_set(1)
		print("\n[Bake Finished] Unified engine execution completed successfully! Press Spacebar.")

		# bpy.context.scene.render.fps = 240
		bpy.context.scene.render.fps = 120
		# bpy.context.scene.render.fps = 120 * 4
		bpy.context.scene.frame_end = num_frames

		

		for area in bpy.context.screen.areas: 
			if area.type == 'VIEW_3D':
				for space in area.spaces: 
					if space.type == 'VIEW_3D':
						# space.shading.type = 'MATERIAL'
						space.shading.type = 'RENDERED'

						# bpy.context.space_data.shading.type = 'RENDERED'


	def bounce_diff_06(self, abj_sd_b_instance):
		# bpy.context.scene.render.engine = 'CYCLES'
		bpy.context.scene.render.engine = 'BLENDER_EEVEE'
		
		bpy.context.scene.cycles.device = 'GPU'

		bpy.context.scene.cycles.samples = 64
		bpy.context.scene.cycles.denoising_use_gpu = True

		# ==============================================================================
		# 1. SCENE CLEANUP & BLENDER ENVIRONMENT LAYOUT SETUP
		# ==============================================================================
		print("Initializing unified JAX Unified Bounce-and-Crush Continuum Engine...")

		for name in ["Rubber_Ball", "Voxel_Collision_Cube"]:
			if name in bpy.data.objects:
				bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)

		# SPHERE_RES = 8  # High resolution captures both fluid ripples and flat cushion folds
		# SPHERE_RES = 16  # High resolution captures both fluid ripples and flat cushion folds
		SPHERE_RES = 32  # High resolution captures both fluid ripples and flat cushion folds ### !!!!!!!
		# SPHERE_RES = 40  # High resolution captures both fluid ripples and flat cushion folds
		# SPHERE_RES = 48  # High resolution captures both fluid ripples and flat cushion folds
		# SPHERE_RES = 64  # High resolution captures both fluid ripples and flat cushion folds
		# SPHERE_RES = 100  # High resolution captures both fluid ripples and flat cushion folds
		# SPHERE_RES = 128  # High resolution captures both fluid ripples and flat cushion folds
		# SPHERE_RES = 256  # High resolution captures both fluid ripples and flat cushion folds

		# Spawn target sphere at Z = 7.5
		# bpy.ops.mesh.primitive_uv_sphere_add(radius=1.8, location=(0.0, 0.0, 7.5), segments=SPHERE_RES, ring_count=SPHERE_RES)
		bpy.ops.mesh.primitive_ico_sphere_add(radius=1.8, location=(0.0, 0.0, 7.5), subdivisions=4)
		# bpy.ops.mesh.primitive_ico_sphere_add(radius=1.8, location=(0.0, 0.0, 0), subdivisions=4)
		# bpy.ops.mesh.primitive_cube_add(location=(0.0, 0.0, 7.5), size=6)

		ball_obj = bpy.context.active_object
		ball_obj.name = "Rubber_Ball"

		ball_obj.shape_key_add(name="Basis", from_mix=False)
		bpy.ops.object.shade_smooth()

		ball_obj = bpy.context.active_object

		# return

		gaborToPts = 0
		# gaborToPts = 1

		# if gaborToPts == 1:
		# 	self.bakeShaderToPts(ball_obj) ######

		mat1 = abj_sd_b_instance.newShader("principled_test_00", "principled", 1, 0, 0)
		bpy.context.active_object.data.materials.clear()
		bpy.context.active_object.data.materials.append(mat1)
		bpy.data.materials["principled_test_00"].node_tree.nodes["Principled BSDF"].inputs[1].default_value = 1
		bpy.data.materials["principled_test_00"].node_tree.nodes["Principled BSDF"].inputs[2].default_value = 0.323263
		bpy.data.materials["principled_test_00"].node_tree.nodes["Principled BSDF"].inputs[20].default_value = 1
		ball_obj.active_material.displacement_method = 'BOTH'

		mat = bpy.data.materials.get("principled_test_00")
		nodes = mat.node_tree.nodes

		nodes = mat.node_tree.nodes

		output_node = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
		if not output_node:
			output_node = nodes.new(type='ShaderNodeOutputMaterial')

		if gaborToPts != 1:
			gabor_node = nodes.new(type='ShaderNodeTexGabor')
			displacement_node = nodes.new(type='ShaderNodeDisplacement')

			mat.node_tree.links.new(gabor_node.outputs['Value'], displacement_node.inputs['Height'])
			mat.node_tree.links.new(displacement_node.outputs['Displacement'], output_node.inputs['Displacement'])
			# displacement_node.inputs[2].default_value = 0.3
			displacement_node.inputs[2].default_value = 0.5

			# bpy.data.materials["principled_test_00"].node_tree.nodes["Gabor Texture"].gabor_type = '3D'
			gabor_node.gabor_type = '3D'

			# checkerNode = nodes.new(type='ShaderNodeTexChecker')
			# principledNode = nodes.get("Principled BSDF")
			# mat.node_tree.links.new(checkerNode.outputs['Color'], principledNode.inputs['Base Color'])
			# checkerNode.inputs[2].default_value = (0, 0, 0, 1)


		abj_sd_b_instance.autoArrangeNodes(mat.node_tree)

		######## CUBE FLOOR ###########
		######## CUBE FLOOR ###########
		######## CUBE FLOOR ###########

		# Spawn Cube Ground Floor Object
		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 1.5))
		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -20))
		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -15))
		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -30))


		# bpy.ops.mesh.primitive_ico_sphere_add(radius=6, location=(0.0, 0.0, 7.5), subdivisions=4)
		# bpy.ops.mesh.primitive_ico_sphere_add(radius=48, location=(0.0, 0.0, -75.5), subdivisions=4)

		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -40))
		bpy.ops.mesh.primitive_cube_add(size=1.0, location=(15.0, 0.0, -40))
		cube_floor = bpy.context.active_object
		cube_floor.name = "Voxel_Collision_Cube"
		cube_floor.scale = (1200.0, 1200.0, 1.0)
		cube_floor.rotation_euler = (0.0, 0.45, 0.0)

		# cube_floor = bpy.context.active_object

		mat1 = abj_sd_b_instance.newShader("principled_test_00_grd", "principled", .2, 0, 0)
		cube_floor.data.materials.clear()
		cube_floor.data.materials.append(mat1)
		bpy.data.materials["principled_test_00_grd"].node_tree.nodes["Principled BSDF"].inputs[1].default_value = 1
		bpy.data.materials["principled_test_00_grd"].node_tree.nodes["Principled BSDF"].inputs[2].default_value = 0.323263
		bpy.data.materials["principled_test_00_grd"].node_tree.nodes["Principled BSDF"].inputs[20].default_value = 1
		cube_floor.active_material.displacement_method = 'BOTH'

		mat = bpy.data.materials.get("principled_test_00_grd")
		nodes = mat.node_tree.nodes

		nodes = mat.node_tree.nodes





		checkerNode = nodes.new(type='ShaderNodeTexChecker')
		principledNode = nodes.get("Principled BSDF")
		mat.node_tree.links.new(checkerNode.outputs['Color'], principledNode.inputs['Base Color'])
		checkerNode.inputs[2].default_value = (0, 0, 0, 1)
		# bpy.data.materials["principled_test_grd"].node_tree.nodes["Checker Texture"].inputs[3].default_value = 10
		checkerNode.inputs[3].default_value = 100






		output_node = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
		if not output_node:
			output_node = nodes.new(type='ShaderNodeOutputMaterial')

		abj_sd_b_instance.autoArrangeNodes(mat.node_tree)

		##########  MID OBSTACLE ##############
		##########  MID OBSTACLE ##############
		##########  MID OBSTACLE ##############

		# Spawn Second Mid-Air Intercepting Cube Obstacle
		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.5, 0.0, 4.5))
		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.5, 0.0, 2.75))
		bpy.ops.mesh.primitive_cube_add(size=1.0, location=(10, 0.0, -60)) #####s
		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.5, 0.0, -100))  
		mid_obstacle = bpy.context.active_object
		mid_obstacle.name = "Mid_Air_Obstacle"
		# mid_obstacle.rotation_euler = (0.45, 0.25, 0.78)
		# mid_obstacle.rotation_euler = (0.65, -0.45, 0) #####
		mid_obstacle.rotation_euler = (0.65, -0.45, -45)
		# mid_obstacle.scale = (2.0, 2.0, 1.5)
		# mid_obstacle.scale = (.5, .5, .5)
		# mid_obstacle.scale = (1, 1, 1)
		# mid_obstacle.scale = (.5, 2, .5)
		# mid_obstacle.scale = (2, .5, .5)
		# mid_obstacle.scale = (30, 30, 1)
		mid_obstacle.scale = (600, 600, 1)

		#mid_obstacle
		
		# mat1 = abj_sd_b_instance.newShader("principled_test_00", "principled", .5, .5, .5)
		mat1 = abj_sd_b_instance.newShader("principled_test_00_mid", "principled", 0, 0, 1)
		mid_obstacle.data.materials.clear()
		mid_obstacle.data.materials.append(mat1)
		bpy.data.materials["principled_test_00_mid"].node_tree.nodes["Principled BSDF"].inputs[1].default_value = 1
		bpy.data.materials["principled_test_00_mid"].node_tree.nodes["Principled BSDF"].inputs[2].default_value = 0.323263
		bpy.data.materials["principled_test_00_mid"].node_tree.nodes["Principled BSDF"].inputs[20].default_value = 1
		mid_obstacle.active_material.displacement_method = 'BOTH'

		mat = bpy.data.materials.get("principled_test_00_mid")
		nodes = mat.node_tree.nodes

		nodes = mat.node_tree.nodes

		checkerNode = nodes.new(type='ShaderNodeTexChecker')
		principledNode = nodes.get("Principled BSDF")
		mat.node_tree.links.new(checkerNode.outputs['Color'], principledNode.inputs['Base Color'])
		checkerNode.inputs[2].default_value = (0, 0, 0, 1)
		# bpy.data.materials["principled_test_grd"].node_tree.nodes["Checker Texture"].inputs[3].default_value = 10
		checkerNode.inputs[3].default_value = 100

		output_node = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
		if not output_node:
			output_node = nodes.new(type='ShaderNodeOutputMaterial')

		# if gaborToPts != 1:
		# 	gabor_node = nodes.new(type='ShaderNodeTexGabor')
		# 	displacement_node = nodes.new(type='ShaderNodeDisplacement')

		# 	mat.node_tree.links.new(gabor_node.outputs['Value'], displacement_node.inputs['Height'])
		# 	mat.node_tree.links.new(displacement_node.outputs['Displacement'], output_node.inputs['Displacement'])
		# 	# displacement_node.inputs[2].default_value = 0.3
		# 	displacement_node.inputs[2].default_value = 0.5

		# 	# bpy.data.materials["principled_test_00"].node_tree.nodes["Gabor Texture"].gabor_type = '3D'
		# 	gabor_node.gabor_type = '3D'

		abj_sd_b_instance.autoArrangeNodes(mat.node_tree)

		# return

		##########  MID OBSTACLE 2 ##############
		##########  MID OBSTACLE 2 - ACTUAL MID COLLIDER ##############
		##########  MID OBSTACLE 2 ##############

		# Spawn Second Mid-Air Intercepting Cube Obstacle
		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.5, 0.0, 4.5))
		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.5, 0.0, 2.75))
		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.5, 0.0, -10)) #####
		bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.5, 0.0, -13.5))
		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.5, 0.0, -100))  
		mid_obstacle2 = bpy.context.active_object
		mid_obstacle2.name = "Mid_Air_Obstacle_2"
		# mid_obstacle.rotation_euler = (0.45, 0.25, 0.78)



		mid_obstacle2.rotation_euler = (0.65, 0.45, -90) ### !!!!!!!! ####



		# mid_obstacle.scale = (2.0, 2.0, 1.5)
		# mid_obstacle.scale = (.5, .5, .5)
		# mid_obstacle.scale = (1, 1, 1)
		# mid_obstacle.scale = (.5, 2, .5)
		# mid_obstacle.scale = (2, .5, .5)
		# mid_obstacle.scale = (30, 30, 1)
		mid_obstacle2.scale = (10, 10, 1) #####################
		# mid_obstacle2.scale = (20, 20, 1)
		# mid_obstacle2.scale = (40, 40, 8) ### !!!!!
		# mid_obstacle2.scale = (40, 40, 1)

		# return

		#mid_obstacle 2
		
		# mat1 = abj_sd_b_instance.newShader("principled_test_00", "principled", .5, .5, .5)
		mat1 = abj_sd_b_instance.newShader("principled_test_00_mid2", "principled", 0, 0, 1)
		mid_obstacle.data.materials.clear()
		mid_obstacle.data.materials.append(mat1)
		bpy.data.materials["principled_test_00_mid2"].node_tree.nodes["Principled BSDF"].inputs[1].default_value = 1
		bpy.data.materials["principled_test_00_mid2"].node_tree.nodes["Principled BSDF"].inputs[2].default_value = 0.323263
		bpy.data.materials["principled_test_00_mid2"].node_tree.nodes["Principled BSDF"].inputs[20].default_value = 1
		mid_obstacle.active_material.displacement_method = 'BOTH'

		mat = bpy.data.materials.get("principled_test_00_mid2")
		nodes = mat.node_tree.nodes

		nodes = mat.node_tree.nodes

		checkerNode = nodes.new(type='ShaderNodeTexChecker')
		principledNode = nodes.get("Principled BSDF")
		mat.node_tree.links.new(checkerNode.outputs['Color'], principledNode.inputs['Base Color'])
		checkerNode.inputs[2].default_value = (0, 0, 0, 1)
		# bpy.data.materials["principled_test_grd"].node_tree.nodes["Checker Texture"].inputs[3].default_value = 10
		checkerNode.inputs[3].default_value = 100

		output_node = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
		if not output_node:
			output_node = nodes.new(type='ShaderNodeOutputMaterial')

		# if gaborToPts != 1:
		# 	gabor_node = nodes.new(type='ShaderNodeTexGabor')
		# 	displacement_node = nodes.new(type='ShaderNodeDisplacement')

		# 	mat.node_tree.links.new(gabor_node.outputs['Value'], displacement_node.inputs['Height'])
		# 	mat.node_tree.links.new(displacement_node.outputs['Displacement'], output_node.inputs['Displacement'])
		# 	# displacement_node.inputs[2].default_value = 0.3
		# 	displacement_node.inputs[2].default_value = 0.5

		# 	# bpy.data.materials["principled_test_00"].node_tree.nodes["Gabor Texture"].gabor_type = '3D'
		# 	gabor_node.gabor_type = '3D'

		abj_sd_b_instance.autoArrangeNodes(mat.node_tree)

		# return
		

		##########  MID OBSTACLE 3 ##############
		##########  MID OBSTACLE 3 ##############
		##########  MID OBSTACLE 3 ##############

		# Spawn Second Mid-Air Intercepting Cube Obstacle
		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.5, 0.0, 4.5))
		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.5, 0.0, 2.75))
		bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.5, 0.0, -20)) #####s
		# bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.5, 0.0, -100))  
		mid_obstacle3 = bpy.context.active_object
		mid_obstacle3.name = "Mid_Air_Obstacle_3"
		# mid_obstacle.rotation_euler = (0.45, 0.25, 0.78)
		# mid_obstacle3.rotation_euler = (0.65, 0.25, -0.28)
		# mid_obstacle.scale = (2.0, 2.0, 1.5)
		# mid_obstacle.scale = (.5, .5, .5)
		# mid_obstacle.scale = (1, 1, 1)
		# mid_obstacle.scale = (.5, 2, .5)
		# mid_obstacle.scale = (2, .5, .5)
		# mid_obstacle.scale = (30, 30, 1)
		mid_obstacle3.scale = (1000, 1000, 1)

		#mid_obstacle
		
		# mat1 = abj_sd_b_instance.newShader("principled_test_00", "principled", .5, .5, .5)
		mat1 = abj_sd_b_instance.newShader("principled_test_00_mid3", "principled", 0, 0, 1)
		mid_obstacle3.data.materials.clear()
		mid_obstacle3.data.materials.append(mat1)
		bpy.data.materials["principled_test_00_mid3"].node_tree.nodes["Principled BSDF"].inputs[1].default_value = 1
		bpy.data.materials["principled_test_00_mid3"].node_tree.nodes["Principled BSDF"].inputs[2].default_value = 0.323263
		bpy.data.materials["principled_test_00_mid3"].node_tree.nodes["Principled BSDF"].inputs[20].default_value = 1
		mid_obstacle3.active_material.displacement_method = 'BOTH'

		mat = bpy.data.materials.get("principled_test_00_mid3")
		nodes = mat.node_tree.nodes

		nodes = mat.node_tree.nodes

		checkerNode = nodes.new(type='ShaderNodeTexChecker')
		principledNode = nodes.get("Principled BSDF")
		mat.node_tree.links.new(checkerNode.outputs['Color'], principledNode.inputs['Base Color'])
		checkerNode.inputs[2].default_value = (0, 0, 0, 1)
		# bpy.data.materials["principled_test_grd"].node_tree.nodes["Checker Texture"].inputs[3].default_value = 10
		checkerNode.inputs[3].default_value = 100

		output_node = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
		if not output_node:
			output_node = nodes.new(type='ShaderNodeOutputMaterial')

		# if gaborToPts != 1:
		# 	gabor_node = nodes.new(type='ShaderNodeTexGabor')
		# 	displacement_node = nodes.new(type='ShaderNodeDisplacement')

		# 	mat.node_tree.links.new(gabor_node.outputs['Value'], displacement_node.inputs['Height'])
		# 	mat.node_tree.links.new(displacement_node.outputs['Displacement'], output_node.inputs['Displacement'])
		# 	# displacement_node.inputs[2].default_value = 0.3
		# 	displacement_node.inputs[2].default_value = 0.5

		# 	# bpy.data.materials["principled_test_00"].node_tree.nodes["Gabor Texture"].gabor_type = '3D'
		# 	gabor_node.gabor_type = '3D'

		abj_sd_b_instance.autoArrangeNodes(mat.node_tree)

		# return

		# return
		
		###########
		# WORLD
		###########

		world = bpy.context.scene.world
		worldtree = world.node_tree
		worldtree.nodes.clear()

		# output_node_world = next((n for n in worldtree.nodes if n.type == 'ShaderNodeOutputWorld'), None)
		# if not output_node:
		# 	output_node_world = worldtree.nodes.new(type='ShaderNodeOutputWorld')

		# output_node = worldtree.nodes.new(type="ShaderNodeOutputWorld")
		# bg_node = worldtree.nodes.new(type="ShaderNodeBackground")

		# output_node_world = worldtree.nodes.new(type="ShaderNodeOutputWorld")
		output_node_world = worldtree.nodes.new('ShaderNodeOutputWorld')

		# node_sky = bpy.ops.node.add_node(use_transform=True, type="ShaderNodeTexSky")
		# node_sky = bpy.ops.node.add_node(use_transform=True, type="ShaderNodeTexSky")
		node_sky = worldtree.nodes.new('ShaderNodeTexSky')
		worldtree.links.new(node_sky.outputs["Color"], output_node_world.inputs["Surface"])

		# bpy.data.worlds["World"].node_tree.nodes["Sky Texture"].sun_size = 0.372541
		# bpy.data.worlds["World"].node_tree.nodes["Sky Texture"].sun_intensity = 21.3
		# bpy.data.worlds["World"].node_tree.nodes["Sky Texture"].sun_rotation = -1.57603
		# node_sky.sun_size = 0.372541
		# node_sky.sun_intensity = 21.3
		node_sky.sun_rotation = 1.65806
		node_sky.sun_elevation = .05

		abj_sd_b_instance.autoArrangeNodes(worldtree)

		bpy.context.view_layer.update()

		abj_sd_b_instance.agxColorSettings_UI()

		# return

		bpy.context.evaluated_depsgraph_get().update()

		def get_jax_transforms(obj):
			mw = obj.matrix_world
			center = jnp.array(mw.to_translation())
			size = jnp.array(obj.dimensions)
			r3 = mw.to_3x3().normalized()
			rot_matrix = jnp.array([[r3[0][0], r3[0][1], r3[0][2]],
									[r3[1][0], r3[1][1], r3[1][2]],
									[r3[2][0], r3[2][1], r3[2][2]]])
			return center, size, rot_matrix


		box_centers, box_sizes, box_rots = stack_boxes([
			get_jax_transforms(cube_floor),
			get_jax_transforms(mid_obstacle),
			get_jax_transforms(mid_obstacle2),
			get_jax_transforms(mid_obstacle3)])

		# ==============================================================================
		# 2. EXTRACT SCENE GEOMETRY TO GLOBAL SCOPE ARRAYS
		# ==============================================================================
		mesh = ball_obj.data
		num_verts = len(mesh.vertices)
		orig_coords = np.zeros((num_verts, 3))
		mesh.vertices.foreach_get("co", orig_coords.ravel())

		# Triangle indices for the volume-preservation term (ico sphere = all triangles)
		faces_numpy = np.array([list(p.vertices) for p in mesh.polygons], dtype=np.int32)
		STATIC_FACES = jnp.array(faces_numpy)

		sphere_tracker = np.copy(orig_coords)
		sphere_tracker[:, 2] += 7.5

		ref_normals_numpy = np.zeros((num_verts, 3))
		for v in mesh.vertices:
			ref_normals_numpy[v.index] = np.array(v.normal)

		min_init_z = np.min(orig_coords[:, 2])
		vertex_thickness_numpy = (orig_coords[:, 2] - min_init_z) * 0.14

		STATIC_INIT_POS = jnp.array(sphere_tracker)
		STATIC_CENTER = jnp.mean(STATIC_INIT_POS, axis=0)
		STATIC_NORMALS = jnp.array(ref_normals_numpy)
		STATIC_THICKNESS = jnp.array(vertex_thickness_numpy)

		# num_frames = 1600
		# num_frames = 2200
		num_frames = 300
		# num_frames = 200
		# num_frames = 4000

		MU_VAL = 550.0
		LAM_VAL = 3000.0
		DAMPING_VAL = 0.995
		# DAMPING_VAL = 0.99
		# DAMPING_VAL = 0.97
		# DAMPING_VAL = 0.95
		DT_VAL = 0.008
		INITIAL_SPIKE_VELOCITY = -30
		# INITIAL_SPIKE_VELOCITY = -60
		RESTITUTION_VAL = .9
		# RESTITUTION_VAL = .87
		# RESTITUTION_VAL = .7
		# RESTITUTION_VAL = .5
		# RESTITUTION_VAL = .1
		# RESTITUTION_VAL = .95
		OBJECT_MASS_VAL = 300.0
		DRAG_COEFFICIENT_VAL = .05
		TARGET_FRAME = 100
		TARGET_HEIGHT = 12.5

		# return
		faces_np = np.array([list(p.vertices) for p in ball_obj.data.polygons])
		edges = build_edges(faces_np, np.array(STATIC_INIT_POS))

		final_trajectory_matrix = jit_simulation_engine(
			MU_VAL, LAM_VAL, DAMPING_VAL, DT_VAL, INITIAL_SPIKE_VELOCITY, RESTITUTION_VAL,
			OBJECT_MASS_VAL, DRAG_COEFFICIENT_VAL, STATIC_INIT_POS, STATIC_CENTER,
			STATIC_NORMALS, STATIC_THICKNESS,
			box_centers, box_sizes, box_rots, edges, num_frames)

		


		# return

		##check inverted faces
		# print('CHECK INVERTED FACES')
		# mesh = ball_obj.data
		# F = np.array([list(p.vertices) for p in mesh.polygons])
		# traj = np.array(final_trajectory_matrix)
		# bad = []

		# for f in range(0, traj.shape[0], 10):
		# 	P = traj[f]; a, b, c = P[F[:,0]], P[F[:,1]], P[F[:,2]]
		# 	nrm = np.cross(b - a, c - a); ctr = (a + b + c) / 3 - P.mean(axis=0)
		# 	bad.append(float((np.sum(nrm * ctr, axis=1) < 0).mean()))

		# print("max fraction of inverted faces:", max(bad))

		# ##check flatness ratio
		# print('FLATNESS RATIO')
		# fl = np.array(final_trajectory_matrix)
		# # for f in range(2000, fl.shape[0], 100):
		# # for f in range(0, fl.shape[0], 100):
		# for f in range(0, 100, 4):
		# 	s, _ = all_boxes_sdf_and_normal(jnp.array(fl[f]), box_centers, box_sizes, box_rots)
		# 	print(f, "min_sdf=%.3f" % float(s.min()), "com=", np.round(fl[f].mean(axis=0), 2))


		# print('FOOTPRINT')
		# fl = np.array(final_trajectory_matrix)
		# # for f in range(30, 70, 2):
		# for f in range(25, 135, 2):
		# 	P = fl[f]
		# 	s, _ = all_boxes_sdf_and_normal(jnp.array(P), box_centers, box_sizes, box_rots)
		# 	touch = np.array(s) < 0.1
		# 	rad = np.linalg.norm(P[touch][:, :2] - P[:, :2].mean(0), axis=1).max() if touch.any() else 0.0
		# 	print(f, "footprint_radius=%.2f" % rad, "height=%.2f" % (P[:, 2].max() - P[:, 2].min()))


		# print('stuck verts')
		# fl = np.array(final_trajectory_matrix)
		# for f in range(44, 80, 2):
		# 	P = fl[f]
		# 	s, _ = all_boxes_sdf_and_normal(jnp.array(P), box_centers, box_sizes, box_rots)
		# 	touching = int((np.array(s) < 0.1).sum())
		# 	print(f, "touching=%d" % touching, "com_height=%.2f" % (P[:, 2].mean() - (-9.5)),
		# 		"max_dist_from_com=%.2f" % np.linalg.norm(P - P.mean(0), axis=1).max())






		# print('steamroll TOP ABOVE FLOOR check 0000000000')
		# fl = np.array(final_trajectory_matrix)
		# for f in range(28, 60, 2):
		# 	P = fl[f]
		# 	s, _ = all_boxes_sdf_and_normal(jnp.array(P), box_centers, box_sizes, box_rots)
		# 	touch = np.array(s) < 0.1
		# 	rad = np.linalg.norm(P[touch][:, :2] - P[:, :2].mean(0), axis=1).max() if touch.any() else 0.0
		# 	print(f, "footprint=%.2f" % rad, "height=%.2f" % (P[:, 2].max() - P[:, 2].min()),
		# 		"top_above_floor=%.2f" % (P[:, 2].max() + 9.5))


		# print('variable velocity check 1111111111 !!!!')
		# # num_frames = 100

		# faces_np = np.array([list(p.vertices) for p in ball_obj.data.polygons])
		# rest_c = STATIC_INIT_POS - STATIC_CENTER

		# def shape_err(P):
		# 	R, c = compute_best_fit_rotation(jnp.array(P), rest_c)
		# 	return float(jnp.sqrt(jnp.mean(jnp.sum((jnp.array(P) - c - rest_c @ R.T) ** 2, axis=1))))

		# def inv_frac(P):
		# 	a, b, c = P[faces_np[:, 0]], P[faces_np[:, 1]], P[faces_np[:, 2]]
		# 	nrm = np.cross(b - a, c - a)
		# 	ctr = (a + b + c) / 3 - P.mean(axis=0)
		# 	return float((np.sum(nrm * ctr, axis=1) < 0).mean())


		# for v0 in [-20, -30, -45, -60]:
		# 	tr = np.array(jit_simulation_engine(MU_VAL, LAM_VAL, DAMPING_VAL, DT_VAL, v0, RESTITUTION_VAL,
		# 			OBJECT_MASS_VAL, DRAG_COEFFICIENT_VAL, STATIC_INIT_POS, STATIC_CENTER,
		# 			STATIC_NORMALS, STATIC_THICKNESS, box_centers, box_sizes, box_rots, edges, num_frames))
		# 	errs = [shape_err(tr[f]) for f in range(0, num_frames, 5)]
		# 	inv = max(inv_frac(tr[f]) for f in range(0, num_frames, 5))
		# 	i_peak = int(np.argmax(errs))
		# 	print("v0=%d peak_err=%.2f (frame %d) err_after_30f=%.2f err_after_60f=%.2f max_inverted=%.3f max_span=%.1f" % (
		# 		v0, errs[i_peak], i_peak * 5,
		# 		errs[min(i_peak + 6, len(errs) - 1)], errs[min(i_peak + 12, len(errs) - 1)],
		# 		inv, max(np.linalg.norm(tr[f] - tr[f].mean(0), axis=1).max() for f in range(0, num_frames, 5))))

		# far = box_centers + jnp.array([0.0, 0.0, -1000.0])
		# f_rest = compute_forces_jax(STATIC_INIT_POS, MU_VAL, LAM_VAL, STATIC_INIT_POS, STATIC_CENTER,
		# 							STATIC_NORMALS, STATIC_THICKNESS, far, box_sizes, box_rots, edges, num_frames)
		# print("mean rest-pose force:", float(jnp.linalg.norm(f_rest, axis=1).mean()))


		'''
		def summarize(v0):
			# tr = np.array(jit_simulation_engine(MU_VAL, LAM_VAL, DAMPING_VAL, DT_VAL, v0, RESTITUTION_VAL,
					# OBJECT_MASS_VAL, DRAG_COEFFICIENT_VAL, STATIC_INIT_POS, STATIC_CENTER,
					# STATIC_NORMALS, STATIC_THICKNESS, box_centers, box_sizes, box_rots, edges, 200))
		
			tr = np.array(jit_simulation_engine(MU_VAL, LAM_VAL, DAMPING_VAL, DT_VAL, -45, RESTITUTION_VAL,
					OBJECT_MASS_VAL, DRAG_COEFFICIENT_VAL, STATIC_INIT_POS, STATIC_CENTER,
					STATIC_NORMALS, STATIC_THICKNESS, box_centers, box_sizes, box_rots, edges, 200))
			print([round(float(tr[f][:, 2].max() - tr[f][:, 2].min()), 2) for f in range(30, 130, 3)])


			hs, rs = [], []
			for f in range(0, 200, 2):
				P = tr[f]
				s, _ = all_boxes_sdf_and_normal(jnp.array(P), box_centers, box_sizes, box_rots)
				t = np.array(s) < 0.1
				hs.append(P[:, 2].max() - P[:, 2].min())
				rs.append(np.linalg.norm(P[t][:, :2] - P[:, :2].mean(0), axis=1).max() if t.any() else 0.0)
			print("v0=%d min_height=%.2f max_footprint=%.2f" % (v0, min(hs), max(rs)))

		for v0 in [-20, -30, -45, -60]:
			summarize(v0)
		'''




		'''


		'''




		# print('free fall')
		# far = box_centers + jnp.array([0.0, 0.0, -1000.0])
		# fl = np.array(jit_simulation_engine(MU_VAL, LAM_VAL, DAMPING_VAL, DT_VAL, INITIAL_SPIKE_VELOCITY,
		# 		RESTITUTION_VAL, OBJECT_MASS_VAL, DRAG_COEFFICIENT_VAL, STATIC_INIT_POS, STATIC_CENTER,
		# 		STATIC_NORMALS, STATIC_THICKNESS, far, box_sizes, box_rots, 100))
		# for f in range(0, 100, 5):
		# 	P = fl[f]
		# 	print(f, "height=%.2f width=%.2f" % (P[:, 2].max() - P[:, 2].min(), P[:, 0].max() - P[:, 0].min()))



		# FLATNESS RATIO
		# 2000 min_sdf=0.007 com= [ -7.58   2.76 -17.99]
		# 2100 min_sdf=0.007 com= [ -7.59   2.78 -17.99]

		# return

		'''

		b1_center, b1_size, b1_rot = get_jax_transforms(cube_floor)
		b2_center, b2_size, b2_rot = get_jax_transforms(mid_obstacle)
		b3_center, b3_size, b3_rot = get_jax_transforms(mid_obstacle2)
		b4_center, b4_size, b4_rot = get_jax_transforms(mid_obstacle3)

		# ==============================================================================
		# 2. EXTRACT SCENE GEOMETRY TO GLOBAL SCOPE ARRAYS
		# ==============================================================================
		mesh = ball_obj.data
		num_verts = len(mesh.vertices)
		orig_coords = np.zeros((num_verts, 3))
		mesh.vertices.foreach_get("co", orig_coords.ravel())

		# Baseline world coordinates tracking matrix
		sphere_tracker = np.copy(orig_coords)
		sphere_tracker[:, 2] += 7.5  

		# Extract reference normal direction vectors
		ref_normals_numpy = np.zeros((num_verts, 3))
		for v in mesh.vertices:
			ref_normals_numpy[v.index] = np.array(v.normal)

		# Precompute thickness profile cushion (14% of initial relative height profile)
		min_init_z = np.min(orig_coords[:, 2])
		vertex_thickness_numpy = (orig_coords[:, 2] - min_init_z) * 0.14  

		# --- GLOBAL SCOPE JAX ARRAYS ---
		# By assigning these to the global module scope, the JAX functions can read them
		# directly via closure. They are never passed through scan, so they CANNOT flatten or swap!

		STATIC_INIT_POS = jnp.array(sphere_tracker)
		STATIC_CENTER = jnp.mean(STATIC_INIT_POS, axis=0) # Pristine Frame 1 rest center
		STATIC_NORMALS = jnp.array(ref_normals_numpy)
		STATIC_THICKNESS = jnp.array(vertex_thickness_numpy)


		# ==============================================================================
		# 4. EXPLICIT AUTOMATED INDIVIDUAL-ARGUMENT BACKPROPAGATION GRADIENT DESCENT LOOP
		# ==============================================================================

		# --- FIXED: WEIGHT PROFILE EXPERIMENT SWITCHBOARD ---
		# Test Case 1: Heavy Lead Brick -> Mass = 50.0, Drag = 0.15 (Slams down hard, bounces high)
		# Test Case 2: Light Feather Cushion -> Mass = 0.8, Drag = 1.85 (Floats down slowly, stays soft)
		# OBJECT_MASS_VAL = 45.0          
		# DRAG_COEFFICIENT_VAL = 0.25   


		# num_frames = 600
		# num_frames = 1200
		num_frames = 1600
		# num_frames = 2000
		# num_frames = 300
		# num_frames = 300



	

		##########
		### SPHERE GOOD
		##########
		MU_VAL = 550.0
		LAM_VAL = 3000.0
		DAMPING_VAL = 0.995
		DT_VAL = 0.008
		INITIAL_SPIKE_VELOCITY = -30
		RESTITUTION_VAL = .87
		OBJECT_MASS_VAL = 300.0
		DRAG_COEFFICIENT_VAL = .05

		TARGET_FRAME = 100
		TARGET_HEIGHT = 12.5
		'''

		'''
		loss_fn = jax.jit(lambda m, l, d, t, v, r, ma, dr: loss_function(m, l, d, t, v, r, ma, dr, STATIC_INIT_POS, STATIC_CENTER, STATIC_NORMALS, STATIC_THICKNESS, TARGET_FRAME, TARGET_HEIGHT))

		# Target index slots securely: 0=mu, 1=lam, 2=damping, 4=velocity, 6=mass
		grad_fn = jax.jit(jax.grad(
			lambda m, l, d, t, v, r, ma, dr: loss_function(m, l, d, t, v, r, ma, dr, STATIC_INIT_POS, STATIC_CENTER, STATIC_NORMALS, STATIC_THICKNESS, TARGET_FRAME, TARGET_HEIGHT),
			argnums=(0, 1, 2, 4, 6)
		))


		print(f"\n[JAX Optimizer] Launching local manual-argument gradient search trajectory tracking...")

		lr_stiffness = 2.5    
		# lr_damping = 4e-4 ##
		# lr_damping = 2
		lr_damping = .1
		# lr_velocity = 4e-1    
		lr_velocity = 10 
		# lr_mass = 0.5         # Gradient search rate for weight scale optimization
		lr_mass = .5         # Gradient search rate for weight scale optimization
		# num_steps = 20
		# num_steps = 10
		# num_steps = 5
		num_steps = 2


		for iteration in range(num_steps):
			continue

			current_loss = loss_fn(MU_VAL, LAM_VAL, DAMPING_VAL, DT_VAL, INITIAL_SPIKE_VELOCITY, RESTITUTION_VAL, OBJECT_MASS_VAL, DRAG_COEFFICIENT_VAL)
			raw_grads = grad_fn(MU_VAL, LAM_VAL, DAMPING_VAL, DT_VAL, INITIAL_SPIKE_VELOCITY, RESTITUTION_VAL, OBJECT_MASS_VAL, DRAG_COEFFICIENT_VAL)
			
			grad_mu, grad_lam, grad_damp, grad_vel, grad_mass = raw_grads
			
			# FIXED: True Sign Fallback. If gradients hit a zero plateau, we use a constant 
			# directional sign push to dynamically kick the parameter out of the flat zone
			step_mu   = jnp.sign(grad_mu) * lr_stiffness if jnp.abs(grad_mu) > 1e-5 else jnp.sign(MU_VAL - 450.0) * lr_stiffness
			step_lam  = jnp.sign(grad_lam) * lr_stiffness if jnp.abs(grad_lam) > 1e-5 else jnp.sign(LAM_VAL - 3500.0) * lr_stiffness
			step_damp = grad_damp * lr_damping if jnp.abs(grad_damp) > 1e-5 else (DAMPING_VAL - 0.94) * lr_damping
			
			# Track velocity derivatives out of dead zones smoothly
			step_vel  = jnp.sign(grad_vel) * lr_velocity if jnp.abs(grad_vel) > 1e-5 else -1.5 * lr_velocity
			step_mass = jnp.sign(grad_mass) * lr_mass      if jnp.abs(grad_mass) > 1e-5 else grad_mass * 2
			
			# Apply individual unrolled updates smoothly
			MU_VAL                 = float(jnp.clip(MU_VAL + step_mu, 200.0, 5000.0))
			LAM_VAL                = float(jnp.clip(LAM_VAL + step_lam, 1000.0, 15000.0))
			DAMPING_VAL            = float(jnp.clip(DAMPING_VAL - step_damp, 0.98, 0.994))
			OBJECT_MASS_VAL        = float(jnp.clip(OBJECT_MASS_VAL + step_mass, 0.1, 200.0)) # Clip weight to valid ranges
			
			# FIXED: Expanded the clipping boundary ceiling completely to support -120.0 m/s
			# INITIAL_SPIKE_VELOCITY = float(jnp.clip(INITIAL_SPIKE_VELOCITY - step_vel, -120.0, -10.0))
			INITIAL_SPIKE_VELOCITY = float(jnp.clip(INITIAL_SPIKE_VELOCITY + step_vel, -120.0, -10.0))
			
			print(f"  Step {iteration+1:02d} -> Loss: {current_loss:.4f} | Mass Weight: {OBJECT_MASS_VAL:.2f} kg | Spike Vel: {INITIAL_SPIKE_VELOCITY:.2f} m/s | Mu: {MU_VAL:.1f} | Lam: {LAM_VAL:.1f} | Damping: {DAMPING_VAL:.4f} Dt: {DT_VAL:.4f}")
			
		print("\n[JAX Engine] Optimization target achieved! Pulling verified trajectory memory buffer...")

		'''

		# num_frames = 600
		
		
		# final_trajectory_matrix = jit_simulation_engine(
		# 	MU_VAL, LAM_VAL, DAMPING_VAL, DT_VAL, INITIAL_SPIKE_VELOCITY, RESTITUTION_VAL, OBJECT_MASS_VAL, DRAG_COEFFICIENT_VAL, STATIC_INIT_POS, STATIC_CENTER, STATIC_NORMALS, STATIC_THICKNESS, num_frames
		# )

		# final_trajectory_matrix = jit_simulation_engine(MU_VAL, LAM_VAL, DAMPING_VAL, DT_VAL, INITIAL_SPIKE_VELOCITY, RESTITUTION_VAL,OBJECT_MASS_VAL, DRAG_COEFFICIENT_VAL, STATIC_INIT_POS, STATIC_CENTER,STATIC_NORMALS, STATIC_THICKNESS, b1_center, b1_size, b1_rot, b2_center, b2_size, b2_rot, num_frames)
		# final_trajectory_matrix = jit_simulation_engine(MU_VAL, LAM_VAL, DAMPING_VAL, DT_VAL, INITIAL_SPIKE_VELOCITY, RESTITUTION_VAL, OBJECT_MASS_VAL, DRAG_COEFFICIENT_VAL, STATIC_INIT_POS, STATIC_CENTER, STATIC_NORMALS, STATIC_THICKNESS, b1_center, b1_size, b1_rot, b2_center, b2_size, b2_rot, b3_center, b3_size, b3_rot, num_frames)
		
		
		# final_trajectory_matrix = jit_simulation_engine(MU_VAL, LAM_VAL, DAMPING_VAL, DT_VAL, INITIAL_SPIKE_VELOCITY, RESTITUTION_VAL, OBJECT_MASS_VAL, DRAG_COEFFICIENT_VAL, STATIC_INIT_POS, STATIC_CENTER, STATIC_NORMALS, STATIC_THICKNESS, b1_center, b1_size, b1_rot, b2_center, b2_size, b2_rot, b3_center, b3_size, b3_rot, b4_center, b4_size, b4_rot, num_frames) #OLD 24 (0 base)

		baked_frames_positions = np.array(final_trajectory_matrix)
		depsgraph = bpy.context.evaluated_depsgraph_get()

		print("[Blender Pipeline] Writing exact JAX position memory to timeline Shape Keys...")








		local_frames = baked_frames_positions.copy()
		local_frames[:, :, 2] -= 7.5
		# pc2_path = os.path.join(tempfile.gettempdir(), "rubber_ball.pc2")

		pc2_path = bpy.path.abspath("E:/projects_3d/ABJ_Shader_Debugger_for_Blender/scenes/compositing_files/rubber_ball.pc2")

		self.write_pc2(pc2_path, local_frames)

		mod = ball_obj.modifiers.new("BallCache", 'MESH_CACHE')
		mod.cache_format = 'PC2'
		mod.filepath = pc2_path
		mod.frame_start = 1





		'''
		for idx, frame in enumerate(range(1, num_frames + 1)):
			print('frame_idx = ', idx)
			bpy.context.scene.frame_set(frame)
			
			frame_coords = baked_frames_positions[frame - 1]
			baked_local_coords = np.copy(frame_coords)
			baked_local_coords[:, 2] -= 7.5
			
			skey = ball_obj.shape_key_add(name=f"Frame_{frame}", from_mix=False)
			skey.data.foreach_set("co", baked_local_coords.ravel())
			
			skey.value = 1.0
			skey.keyframe_insert(data_path="value", frame=frame)
			if frame > 1:
				skey.value = 0.0
				skey.keyframe_insert(data_path="value", frame=frame - 1)
				prev_key = ball_obj.data.shape_keys.key_blocks[f"Frame_{frame-1}"]
				prev_key.value = 0.0
				prev_key.keyframe_insert(data_path="value", frame=frame)
				
			depsgraph.update()
		'''

		bpy.context.scene.frame_set(1)
		print("\n[Bake Finished] Unified engine execution completed successfully! Press Spacebar.")

		# bpy.context.scene.render.fps = 240
		bpy.context.scene.render.fps = 120
		# bpy.context.scene.render.fps = 120 * 4
		bpy.context.scene.frame_end = num_frames

		for area in bpy.context.screen.areas: 
			if area.type == 'VIEW_3D':
				for space in area.spaces: 
					if space.type == 'VIEW_3D':
						space.shading.type = 'RENDERED'

	def write_pc2(self, path, frames_local):
		"""frames_local: (num_frames, num_verts, 3) float array in the object's LOCAL space."""
		n_frames, n_verts, _ = frames_local.shape
		with open(path, "wb") as f:
			f.write(struct.pack("<12siiffi", b"POINTCACHE2\0", 1, n_verts, 1.0, 1.0, n_frames))
			f.write(frames_local.astype(np.float32).tobytes())

	def testSDF_01(self):
		pass

	def testVDB_06(self, abj_sd_b_instance):
		startTime = datetime.now()

		# self.bounce_diff_03(abj_sd_b_instance)
		# self.bounce_diff_04(abj_sd_b_instance) #arb collision with pancake
		# self.bounce_diff_05(abj_sd_b_instance) #new
		self.bounce_diff_06(abj_sd_b_instance) #arb collision with pancake arbitrary obj

		totalTime = datetime.now() - startTime
		print(' ')
		bpy.context.scene.frame_set(0)
		print('totalTime = ', totalTime)

		return

		#look @ execute_production_fem_bake_with_skin

		# nodes_tet4, topology_tet4, tags, vdb_sphere_sdf_l, vdb_sphere_sdf_h, vdb_box_sdf_l, vdb_box_sdf_h = self.generate_global_multiphase_mesh(24, 96)
		# nodes_tet4, topology_tet4, tags, vdb_sphere_sdf_l, vdb_sphere_sdf_h, vdb_box_sdf_l, vdb_box_sdf_h = self.generate_global_multiphase_mesh(16, 16)
		# nodes_tet4, topology_tet4, tags, vdb_sphere_sdf_l, vdb_sphere_sdf_h, vdb_box_sdf_l, vdb_box_sdf_h = self.generate_global_multiphase_mesh(32, 16) #######
		nodes_tet4, topology_tet4, tags, vdb_sphere_sdf_l, vdb_sphere_sdf_h, vdb_box_sdf_l, vdb_box_sdf_h = self.generate_global_multiphase_mesh(8, 16)

		# return

		myTetLattice = self.visualize_global_multiphase_slice(nodes_tet4, topology_tet4, tags, slice_axis=0, slice_val=0.0) ###########

		# return
		# self.visualize_global_multiphase_slice(nodes_tet4, topology_tet4, tags, slice_axis=0, slice_val=1.0) ###########

		# return

		# layer_data_s_l = [("sdf_joined", vdb_sphere_sdf_l)]
		# myBox_l = self.sdf_vdb_visualizer(layer_data_s_l)

		# layer_data_b_l = [("sdf_joined", vdb_box_sdf_l)]
		# mySphere_l = self.sdf_vdb_visualizer(layer_data_b_l)

		layer_data_s_h = [("sdf_joined", vdb_sphere_sdf_h)]
		mySphere_h = self.sdf_vdb_visualizer(layer_data_s_h)

		vertex_tet_ids, vertex_bary_weights = self.precompute_skin_barycentric_weights(
			mySphere_h, nodes_tet4, topology_tet4, tags
		)

		nodes_tet10, topology_tet10 = self.convert_tet4_lattice_to_tet10(nodes_tet4, topology_tet4)

		###############################
		#### BAKE
		##############################
		# obj = bpy.context.scene.objects.get(mySphere_h)
		# obj = bpy.context.scene.objects.get(mySphere_h.name)
		obj = mySphere_h
		mesh = obj.data

		if not mesh.shape_keys:
			obj.shape_key_add(name="Basis")

		# 1. FIX: Explicitly toggle absolute mode on the underlying mesh block
		obj.data.shape_keys.use_relative = False

		# return

		# total_frames = 2
		# total_frames = 3
		total_frames = 10
		# total_frames = 1
		# frame_dt=0.1 ########sss
		# frame_dt=0.01 ########sss
		# frame_dt=0.0001 ######## new
		# frame_dt=0.0000001 ######## new2
		frame_dt=.000001 ######## new2 current
		# frame_dt=float(1 / 24) ######## new2 current
		# frame_dt=.5
		# frame_dt=1

		# Initialize simulation states
		x_current = nodes_tet10.copy()
		v_current = np.zeros((len(nodes_tet10), 3), dtype=np.float64)

		# F_ext = np.zeros(len(nodes_tet10)*3, dtype=np.float64)
		# F_ext = np.array([0, -9.81, 0])
		# gravity = np.array([0, -9.81, 0]) ##########
		# gravity = np.array([0, 0, 0])
		gravity = np.array([0, 9.8, 0])
		# gravity = np.array([-9.81, -9.81, -9.81])

		num_nodes = len(nodes_tet10) # Assuming v_t is shape (num_nodes, 3)

		# Tile the [0, -9.81, 0] gravity force across all nodes and flatten it
		# If F_ext is already a per-node force density (like force per unit mass), multiply by mass:
		F_ext = np.tile(gravity, num_nodes) 

		# --- PHASE 2: THE TR-BDF2 TIME STRIDE LOOP ---
		# for frame in range(0, total_frames + 1):
		# for frame in range(2, total_frames + 1):
		for frame in range(1, total_frames + 1): ######
			print('~~~~~~~~~~~~~~~~~~~~~~~ FRAME = ', frame)
			print('~~~~~~~~~~~~~~~~~~~~~~~ FRAME = ', frame)
			print('~~~~~~~~~~~~~~~~~~~~~~~ FRAME = ', frame)

			bpy.context.scene.frame_set(frame)
						
			x_next, v_next = self.run_tr_bdf2_time_step(nodes_tet10, topology_tet10, tags, 
				# x_current, v_current, F_ext, frame_dt, 1e-5, 1)
				x_current, v_current, F_ext, frame_dt, 1e-5, 5)

			x_current, v_current = x_next.copy(), v_next.copy()

			# 2. Extract active frame cage corner states
			num_corners = len(nodes_tet4)
			# x_corners_current = x_current[0:num_corners]

			# # Fix 2: EXTRACT CAGE NODES CORRECTLY USING 2D INDICES (No flat-slice corruption)
			num_corners = len(nodes_tet4)
			x_corners_current = x_current[0:num_corners, :] # Extract full X, Y, Z columns for support nodes

			# 3. STREAMING SKIN DEFORMATION PASS
			# Evaluates the high-res vertex tracking vectors seamlessly
			# deformed_skin_coords_32 = self.deform_skin_tissue_mesh(
			# 	x_corners_current, topology_tet4, vertex_tet_ids, vertex_bary_weights
			# )

			# 3. STREAMING SKIN DEFORMATION PASS (Now natively in Local Object Space)
			local_skin_coords = self.deform_skin_tissue_mesh(
				x_corners_current, topology_tet4, vertex_tet_ids, vertex_bary_weights
			)

			# 4. BAKE TO NATIVE BLENDER ANIMATION timetracks
			# sk = obj.shape_key_add(name=f"FEM_Frame_{frame:04d}")
			sk = obj.shape_key_add(name=f"FEM_Frame_{frame:04d}", from_mix=False)
			sk.data.foreach_set("co", local_skin_coords.ravel())

			#Insert evaluation timeline driving metrics
			sk.value = 0.0
			sk.keyframe_insert(data_path="value", frame=frame - 1)

			sk.value = 1.0
			sk.keyframe_insert(data_path="value", frame=frame)

			if frame != total_frames:
				sk.value = 0.0
				sk.keyframe_insert(data_path="value", frame=frame + 1)

		# 2. Keyframe the absolute evaluation time track linearly across the timeline
		# In Absolute mode, 'eval_time' tracks from 0.0 to C (where C = 10.0 per shape key)
		# The evaluation indices map directly as: Basis=0, Frame1=10, Frame2=20, Frame3=30...
		# for frame in range(1, total_frames + 1):
		# 	obj.data.shape_keys.eval_time = frame * 10.0
		# 	obj.data.shape_keys.keyframe_insert(data_path="eval_time", frame=frame)

		# 	# Clean curve interpolation handles to be strictly linear (avoids time bending)
		# 	anim_data = obj.data.shape_keys.animation_data
		# 	if anim_data and anim_data.action and anim_data.action_slot:
		# 		action = anim_data.action
		# 		slot = anim_data.action_slot
				
		# 		# In Blender 5.x, animation data lives in layers and strips
		# 		if action.layers:
		# 			# Safely fetch the channelbag assigned to this specific object slot
		# 			channelbag = action.layers[0].strips[0].channelbag(slot)
					
		# 			# Iterate through the fcurves stored inside the channelbag
		# 			for fcurve in channelbag.fcurves:
		# 				if fcurve.data_path == "eval_time":
		# 					for kp in fcurve.keyframe_points:
		# 						kp.interpolation = 'LINEAR'

		totalTime = datetime.now() - startTime
		print(' ')
		bpy.context.scene.frame_set(0)
		print('totalTime = ', totalTime)

	def precompute_skin_barycentric_weights(self, render_mesh_obj, nodes_tet4, topology_tet4, phase_tags):
		"""
		Finds the containing tetrahedron for every vertex in the high-res render mesh,
		forcing bindings ONLY to elements assigned to Phase 1 (Soft Tissue Sphere).
		"""
		mesh = render_mesh_obj.data
		num_verts = len(mesh.vertices)
		
		# Keep coordinates in LOCAL space to match your background nodes grid
		render_coords = np.zeros((num_verts, 3), dtype=np.float64)
		mesh.vertices.foreach_get("co", render_coords.ravel())

		vertex_to_tet_id = np.full(num_verts, -1, dtype=np.int32)
		vertex_weights = np.zeros((num_verts, 4), dtype=np.float64)

		# ==========================================================================
		# STRATEGY UPDATE: Extract indices of elements matching phase 1
		# ==========================================================================
		target_tet_global_indices = np.where(phase_tags == 1)[0]
		filtered_topology = topology_tet4[target_tet_global_indices]

		print(f"Pre-computing barycentric weights for {num_verts} skin vertices...")
		print(f"Restricting search grid strictly to {len(filtered_topology)} elements marked as Phase 1.")

		# Loop ONLY over the valid soft-tissue elements
		for local_idx, global_t_idx in enumerate(target_tet_global_indices):
			tet = filtered_topology[local_idx]
			v0, v1, v2, v3 = nodes_tet4[tet]
			
			# Build parametric transformation space matrix
			T = np.column_stack([v0 - v3, v1 - v3, v2 - v3])
			try:
				T_inv = np.linalg.inv(T)
			except np.linalg.inv.LinAlgError:
				continue 

			# Projection calculation mapping cleanly to row vector structures
			diffs = render_coords - v3
			w012 = diffs @ T_inv
			
			w0, w1, w2 = w012[:, 0], w012[:, 1], w012[:, 2]
			w3 = 1.0 - (w0 + w1 + w2)

			# Inversion threshold limit test
			inside_mask = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4) & (w3 >= -1e-4)
			
			valid_indices = np.where(inside_mask & (vertex_to_tet_id == -1))[0]
			if len(valid_indices) > 0:
				# Save the GLOBAL element array tracking index to prevent index mapping mismatch later
				vertex_to_tet_id[valid_indices] = global_t_idx
				vertex_weights[valid_indices] = np.column_stack([
					w0[valid_indices], w1[valid_indices], w2[valid_indices], w3[valid_indices]
				])

		# Handle outer edge boundary vertices by evaluating distance only to Phase 1 centroids
		unassigned_count = np.sum(vertex_to_tet_id == -1)
		if unassigned_count > 0:
			print(f"Warning: {unassigned_count} skin vertices fell outside Phase 1 cells. Snapping to closest tissue element.")
			
			phase_1_centers = nodes_tet4[filtered_topology].mean(axis=1)
			
			for idx in np.where(vertex_to_tet_id == -1)[0]:
				v_coord = render_coords[idx]
				# Identify closest element among Phase 1 elements exclusively
				closest_local_idx = np.argmin(np.linalg.norm(phase_1_centers - v_coord, axis=1))
				global_t_idx = target_tet_global_indices[closest_local_idx]
				
				# Reconstruct exact weights for the boundary target block
				v0, v1, v2, v3 = nodes_tet4[topology_tet4[global_t_idx]]
				T_inv = np.linalg.inv(np.column_stack([v0 - v3, v1 - v3, v2 - v3]))
				w012_fallback = (v_coord - v3) @ T_inv
				w0_f, w1_f, w2_f = w012_fallback[0], w012_fallback[1], w012_fallback[2]
				
				vertex_to_tet_id[idx] = global_t_idx
				vertex_weights[idx] = [w0_f, w1_f, w2_f, 1.0 - (w0_f + w1_f + w2_f)]

		return vertex_to_tet_id, vertex_weights

	def precompute_skin_barycentric_weights1(self, render_mesh_obj, nodes_tet4, topology_tet4):
		"""
		Finds the containing tetrahedron for every vertex in the high-res render mesh
		and computes its 4 corresponding barycentric coordinate weighting factors.
		Executed in matching Local Coordinates space.
		"""
		mesh = render_mesh_obj.data
		num_verts = len(mesh.vertices)
		
		# 1. CORRECT: Keep coordinates in LOCAL space to match nodes_tet4 exactly
		render_coords = np.zeros((num_verts, 3), dtype=np.float64)
		mesh.vertices.foreach_get("co", render_coords.ravel())

		vertex_to_tet_id = np.full(num_verts, -1, dtype=np.int32)
		vertex_weights = np.zeros((num_verts, 4), dtype=np.float64)

		print(f"Pre-computing barycentric weights for {num_verts} skin vertices...")

		# 2. CORRECTED GEOMETRIC SEARCH PASS
		for t_idx, tet in enumerate(topology_tet4):
			v0, v1, v2, v3 = nodes_tet4[tet]
			
			# Build parametric transformation space matrix T = [v0-v3, v1-v3, v2-v3]
			T = np.column_stack([v0 - v3, v1 - v3, v2 - v3])
			try:
				T_inv = np.linalg.inv(T)
			except np.linalg.inv.LinAlgError:
				continue 

			# 3. FIX: Row-Vector projection math maps to T_inv, NOT T_inv.T
			diffs = render_coords - v3
			w012 = diffs @ T_inv  # Correct conversion equivalent to T_inv @ vector
			
			w0, w1, w2 = w012[:, 0], w012[:, 1], w012[:, 2]
			w3 = 1.0 - (w0 + w1 + w2)

			# Inclusion test with a small numerical epsilon safety cushion
			inside_mask = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4) & (w3 >= -1e-4)
			
			valid_indices = np.where(inside_mask & (vertex_to_tet_id == -1))[0]
			if len(valid_indices) > 0:
				vertex_to_tet_id[valid_indices] = t_idx
				vertex_weights[valid_indices] = np.column_stack([w0[valid_indices], w1[valid_indices], w2[valid_indices], w3[valid_indices]])

		# Catch remaining outer skin boundary vertices gracefully
		unassigned_count = np.sum(vertex_to_tet_id == -1)
		if unassigned_count > 0:
			print(f"Warning: {unassigned_count} vertices fell outside the background grid. Finding nearest element...")
			# Clean fallback: map unassigned vertices to the geometrically closest element center
			tet_centers = nodes_tet4[topology_tet4].mean(axis=1)
			for idx in np.where(vertex_to_tet_id == -1)[0]:
				v_coord = render_coords[idx]
				closest_tet = np.argmin(np.linalg.norm(tet_centers - v_coord, axis=1))
				
				# Recalculate weights for the closest element cell explicitly
				v0, v1, v2, v3 = nodes_tet4[topology_tet4[closest_tet]]
				T_inv = np.linalg.inv(np.column_stack([v0 - v3, v1 - v3, v2 - v3]))
				w012_fallback = (v_coord - v3) @ T_inv
				w0_f, w1_f, w2_f = w012_fallback[0], w012_fallback[1], w012_fallback[2]
				
				vertex_to_tet_id[idx] = closest_tet
				vertex_weights[idx] = [w0_f, w1_f, w2_f, 1.0 - (w0_f + w1_f + w2_f)]

		return vertex_to_tet_id, vertex_weights

	def precompute_skin_barycentric_weights0(self, render_mesh_obj, nodes_tet4, topology_tet4):
		"""
		Finds the containing tetrahedron for every vertex in the high-res render mesh
		and computes its 4 corresponding barycentric coordinate weighting factors.
		Executed efficiently using vector transformations.
		
		Returns:
			vertex_to_tet_id: (V,) int32 array tracking containing tet index per vertex.
			vertex_weights: (V, 4) float64 array tracking [w0, w1, w2, w3] weights per vertex.
		"""
		# 1. EXTRACT RAW HIGH-RESOLUTION VERTEX WORLD COORDIANTES
		mesh = render_mesh_obj.data
		num_verts = len(mesh.vertices)
		
		# Pre-allocate flat numpy arrays for ultra-fast vector execution
		render_coords = np.zeros((num_verts, 3), dtype=np.float64)
		mesh.vertices.foreach_get("co", render_coords.ravel())
		
		# Transform arrays to global world space configuration
		world_matrix = np.array(render_mesh_obj.matrix_world, dtype=np.float64)[:3, :4]
		render_coords = (render_coords @ world_matrix[:, :3].T) + world_matrix[:, 3]

		vertex_to_tet_id = np.full(num_verts, -1, dtype=np.int32)
		vertex_weights = np.zeros((num_verts, 4), dtype=np.float64)

		print(f"Pre-computing barycentric weights for {num_verts} skin vertices...")

		# 2. VECTORIZED GEOMETRIC SEARCH PASS (Executed once at initialization)
		# Loop over your low-resolution sliver-free background elements
		for t_idx, tet in enumerate(topology_tet4):
			# Isolate the 4 reference corner node positions
			v0, v1, v2, v3 = nodes_tet4[tet]
			
			# Build parametric transformation space matrix T = [v0-v3, v1-v3, v2-v3]
			T = np.column_stack([v0 - v3, v1 - v3, v2 - v3])
			try:
				T_inv = np.linalg.inv(T)
			except np.linalg.inv.LinAlgError:
				continue # Protect loop constraints against unaligned/degenerate slices

			# Vectorized check: sample remaining unassigned render vertices
			# Project world space coordinates down to localized parameter weights
			diffs = render_coords - v3
			w012 = diffs @ T_inv.T  # Shape: (V, 3)
			
			w0, w1, w2 = w012[:, 0], w012[:, 1], w012[:, 2]
			w3 = 1.0 - (w0 + w1 + w2)

			# Enforce analytical insideness constraints (inclusion threshold allowance)
			inside_mask = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5) & (w3 >= -1e-5)
			
			# Overwrite slice arrays for vertices that fall inside this specific element volume
			valid_indices = np.where(inside_mask & (vertex_to_tet_id == -1))[0]
			if len(valid_indices) > 0:
				vertex_to_tet_id[valid_indices] = t_idx
				vertex_weights[valid_indices] = np.column_stack([w0[valid_indices], w1[valid_indices], w2[valid_indices], w3[valid_indices]])

		# Error handling for loose vertices hanging outside your background simulation box
		unassigned_count = np.sum(vertex_to_tet_id == -1)
		if unassigned_count > 0:
			print(f"Warning: {unassigned_count} skin vertices fell outside the background solver box. Snapping to fallback defaults.")
			# Fallback tracking routine: force map to closest valid element cell
			vertex_to_tet_id[vertex_to_tet_id == -1] = 0
			vertex_weights[vertex_to_tet_id == 0] = [0.25, 0.25, 0.25, 0.25]

		return vertex_to_tet_id, vertex_weights