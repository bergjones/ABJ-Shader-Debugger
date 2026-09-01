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
from scipy.spatial import Delaunay
import importlib
import sys
import copy
import os
import jax
import jax.numpy as jnp
from jax import jit

# Force JAX to use 64-bit double precision to maintain mechanical engineering accuracy
jax.config.update("jax_enable_x64", True)

bpy.utils.expose_bundled_modules()
import openvdb as vdb

class myEquation_dFEM:
	def __init__(self):
		super(myEquation_dFEM, self).__init__()

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

	def subtract_boolean_sdf(self, sdf1, sdf2):
		return np.maximum(sdf1, -sdf2)

	def testVDB_04(self, threshold, grid_resolution, radius):
		# Loop through all volume data blocks loaded in memory
		for volume_block in bpy.data.volumes:
			# Explicitly clear out old active grid trees from system RAM
			for grid in volume_block.grids:
				grid.unload() # Frees voxels from memory, forces file re-read on execution

		# --- 1. SET UP FEM COORDINATE PARAMETERS ---
		# --- 1. USER CONFIGURATION (METERS) ---
		core_radius   = 1.0  # Innermost Phase (Bone)
		muscle_radius = 2.2  # Mid-layer Phase
		skin_radius   = 2.5  # Outermost structural layer

		# Add explicit margin padding to prevent the outer voxel grid from clipping into a box shape
		padding = 1.2
		R_bound = radius * padding  # Bounds run from -2.4 to +2.4

		# Calculate precise cell scale transformations
		total_width = R_bound * 2.0
		voxel_size = total_width / grid_resolution

		# --- 2. EVALUATE CENTERED NUMPY SPACE ---
		# Generate spaces symmetrically spanning across the true origin
		x, y, z = np.ogrid[-R_bound:R_bound:1j*grid_resolution, 
						-R_bound:R_bound:1j*grid_resolution, 
						-R_bound:R_bound:1j*grid_resolution]

		# Create meshgrid of shape (resolution, resolution, resolution, 3)
		X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
		p = np.stack([X, Y, Z], axis=-1)

		# Sphere positioned at (-0.4, 0, 0)
		# sdf_A = self.sdf_sphere(p, center=np.array([-0.4, 0.0, 0.0]), radius=1.0)
		# sdf_A = self.sdf_sphere(p, center=np.array([1, 1, 0.0]), radius=.2) ######
		sdf_A = self.sdf_sphere(p, center=np.array([1, 1, 0.0]), radius=.5)

		# Box positioned at (0.4, 0, 0)
		# sdf_B = self.sdf_box(p, center=np.array([0, 0, 0]), size=np.array([1, 1, 1])) #######
		sdf_B = self.sdf_box(p, center=np.array([0, 0, 0]), size=np.array([1, 1, 1]))

		JOIN_MODE = 'SMOOTH'
		# JOIN_MODE = 'SHARP'
		sdf_combined0 = None
		sdf_combined = None

		# if JOIN_MODE == 'SHARP':
		# 	sdf_combined = self.join_min(sdf_A, sdf_B)
		# else:
		# 	# sdf_combined = self.join_smooth_min(sdf_A, sdf_B, k=0.5)
		# 	sdf_combined = self.subtract_boolean_sdf(sdf_A, sdf_B)

		if JOIN_MODE == 'SHARP':
			sdf_combined0 = self.join_min(sdf_A, sdf_B)
		else:
			sdf_combined0 = self.join_smooth_min(sdf_A, sdf_B, k=0.5)
			# sdf_combined0 = self.subtract_boolean_sdf(sdf_A, sdf_B)

		# sdf_C = self.sdf_box(p, center=np.array([0.4, 0.5, 3]), size=np.array([0.7, 0.7, 0.7]))
		# sdf_C = self.sdf_box(p, center=np.array([0.4, -2, 2]), size=np.array([0.7, 0.7, 0.7]))
		# sdf_C = self.sdf_box(p, center=np.array([0.1, -2, 2]), size=np.array([3, 3, 3]))
		# sdf_C = self.sdf_box(p, center=np.array([-2, -.9, -.8]), size=np.array([3, 3, 3]))
		# sdf_C = self.sdf_box(p, center=np.array([-2, -.9, -.8]), size=np.array([.5, .5, .5]))

		# sdf_combined = self.subtract_boolean_sdf(sdf_combined0, sdf_C)
		sdf_combined = sdf_combined0

		# sdf_combined = sdf_combined0
			
		# Inverted solid sphere equation 
		# numpy_sdf = -(np.sqrt(x**2 + y**2 + z**2) - radius)
		dist_from_center = np.sqrt(x**2 + y**2 + z**2)

		# Calculate individual primitive SDF arrays 
		# Inverted (-) so Blender's internal Grid-to-Mesh reads them as solid
		sdf_core   = -(dist_from_center - core_radius)
		sdf_muscle = -(dist_from_center - muscle_radius)
		sdf_skin   = -(dist_from_center - skin_radius)
		sdf_joined = sdf_combined

		###############################################
		base_dir = bpy.path.abspath("E:/projects_3d/ABJ_Shader_Debugger_for_Blender/scenes/compositing_files/vdb/")

		# THE UNIQUE FILENAME FIX: Scan directory and find the next increment number
		# This forces Blender to register a brand new data block every time you click "Run"
		version = 1
		while os.path.exists(os.path.join(base_dir, f"base_volume_v{version}.vdb")):
			version += 1
		output_path = os.path.join(base_dir, f"base_volume_v{version}.vdb").replace("\\", "/")

		# Initialize OpenVDB grids with matching voxel transforms
		grid_list = []
		# layer_data = [("core_sdf", sdf_core), ("muscle_sdf", sdf_muscle), ("skin_sdf", sdf_skin)]
		layer_data = [("core_sdf", sdf_core), ("muscle_sdf", sdf_muscle), ("skin_sdf", sdf_skin), ("sdf_joined", sdf_joined)]
		# layer_data = [("sdf_joined", sdf_joined)]

		# box_extent = 3.0
		box_extent = 10.0

		for name, array in layer_data:
		# 	g = vdb.FloatGrid(background=-10.0)
			g = vdb.FloatGrid(background=box_extent)
			g.copyFromArray(np.asfortranarray(array, dtype=np.float32))
			g.name = name
			g.transform = vdb.createLinearTransform(voxelSize=voxel_size)
			grid_list.append(g)

		vdb.write(output_path, grids=grid_list)
		###############################################
		# --- CONFIGURATION ---
		vdb_path = output_path
		# obj_name = "Imported_SDF_Volume"
		obj_name = "abj_test_000_Imported_SDF_Volume"
		# This must match the exact grid name you defined when exporting the VDB
		# grid_name_in_vdb = "surface_sdf"

		# grid_name_in_vdb = "core_sdf"
		grid_name_in_vdb = "sdf_joined"
		# grid_name_in_vdb = "muscle_sdf"
		# grid_name_in_vdb = "skin_sdf"

		# layer_data = [("core_sdf", sdf_core), ("muscle_sdf", sdf_muscle), ("skin_sdf", sdf_skin)]

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
		# node_grid_to_mesh.inputs['Threshold'].default_value = 0.0
		node_grid_to_mesh.inputs['Threshold'].default_value = threshold
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

		# volume_obj.location = (-grid_resolution / 2, -grid_resolution / 2, 0)
		# volume_obj.location = (20, -20, 0)

		self.bakeVDB_multi(threshold, vdb_path, grid_resolution, voxel_size)

	def bakeVDB_multi(self, threshold, path, grid_resolution, voxel_size):
		# --- CONFIGURATION ---
		# vdb_path = bpy.path.abspath("//compositing_files/sphere_sdf.vdb")
		# vdb_path = 'E:/projects_3d/ABJ_Shader_Debugger_for_Blender/scenes/compositing_files/output_volume.vdb'
		vdb_path = path
		# grid_name = "surface_sdf"
		grid_name = "sdf_joined"

		half_grid_offset = (grid_resolution / 2.0) * voxel_size
		inverse_translation = -half_grid_offset

		# 1. Create a temporary hidden volume block to read from disk
		bpy.ops.object.volume_import(filepath=vdb_path, align='WORLD')
		temp_vol_obj = bpy.context.active_object
		temp_vol_obj.name = "TEMP_VOLUME_DATA"
		temp_vol_obj.hide_viewport = True
		temp_vol_obj.hide_render = True
		temp_vol_obj.location = (inverse_translation, inverse_translation, inverse_translation)

		# 2. Create the true physical Target Mesh Container
		mesh_data = bpy.data.meshes.new("SDF_Polygons")
		mesh_obj = bpy.data.objects.new("SDF_Mesh_Final", mesh_data)
		# mesh_obj.location = (-grid_resolution / 2, -grid_resolution / 2, 0)
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

		print("Modifier successfully applied! Your object is now a raw, pure polygon mesh.")

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

	def convert_blender_mesh_to_sdf_grid(self, obj_name, nx=16, ny=16, nz=24, grid_scale=0.2):
		pass

		return

		"""
		Converts a native Blender polygonal mesh into an implicit Signed Distance Field (SDF) 
		and returns multi-phase material properties for a continuous tetrahedral lattice.
		
		Returns:
			nodes: (N, 3) float array of tetrahedral lattice node coordinates.
			material_tags: (M,) int array mapping each element to 101 (Solid) or 202 (Air).
		"""
		# 1. FETCH AND INITIALIZE THE BLENDER MESH DATA
		context = bpy.context
		obj = context.scene.objects.get(obj_name)
		if not obj or obj.type != 'MESH':
			raise ValueError(f"Object '{obj_name}' not found or is not a valid polygonal mesh.")
			
		# Ensure all modifiers are evaluated and fetch world coordinates
		depsgraph = context.evaluated_depsgraph_get()
		obj_eval = obj.evaluated_get(depsgraph)
		mesh = obj_eval.to_mesh()
		mesh.transform(obj.matrix_world) # Transform vertex arrays to global world space

		# 2. GENERATE THE NATIVE BLENDER BVH TREE FOR ULTRA-FAST CALCULATIONS
		# This C-accelerated tree allows us to calculate distances without Python loops
		bvh = BVHTree.FromMesh(mesh)

		# 3. CONSTRUCT THE UNIFORM BACKGROUND LATTICE BOUNDS
		# We build a bounding region that encapsulates the object and the nearby ambient air
		x = np.linspace(-nx * grid_scale / 2, nx * grid_scale / 2, nx + 1)
		y = np.linspace(-ny * grid_scale / 2, ny * grid_scale / 2, ny + 1)
		z = np.linspace(0, nz * grid_scale, nz + 1) # Floor aligned
		
		X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
		nodes = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

		# 4. LOOP GENERATION LOGIC FOR CONVERTING TO IMPLICIT CELL SPACE
		# To keep your automated compositor compiler fast, we query element center points
		# For a 5-tet voxel split grid, let's create structural element center positions:
		# (Assuming simple voxel tracking for demonstration; maps seamlessly to your BCC lattice)
		voxel_centers = []
		voxel_connectivity = []
		
		# Simple voxel traversal matrix loop to compute element centers
		for i in range(nx):
			for j in range(ny):
				for k in range(nz):
					cx = (x[i] + x[i+1]) * 0.5
					cy = (y[j] + y[j+1]) * 0.5
					cz = (z[k] + z[k+1]) * 0.5
					voxel_centers.append([cx, cy, cz])
					
		voxel_centers = np.array(voxel_centers)
		num_elements = len(voxel_centers)
		
		# Initialize material array: Default state is 202 (Ambient Multi-phase Fluid/Air)
		material_tags = np.full(num_elements, 202, dtype=np.int32)

		# 5. C-ACCELERATED SDF EVALUATION STEP
		# We query the distance from every element center to the surface geometry
		for idx, center in enumerate(voxel_centers):
			co = mathutils.Vector(center)
			
			# Find closest point on mesh surface
			# Returns: (Vector location, Vector normal, int face_index, float distance)
			loc, normal, face_idx, distance = bvh.find_nearest(co)
			
			if loc is not None:
				# Mathematical Insideness Test using the dot product of the surface normal
				# Vector pointing from the mesh surface out toward our element center coordinate
				to_center = co - loc
				
				# If the vector points opposite to the surface face normal, it is INSIDE the solid!
				if to_center.dot(normal) <= 0.0:
					# Assign 101: Continuous Solid Mechanical Phase (e.g. Robot arm tissue)
					material_tags[idx] = 101

		# Clean mesh allocation memory out of Blender environment cache
		obj_eval.to_mesh_clear()
		
		print(f"Successfully compiled polygon model '{obj_name}' into continuous multi-phase grid.")
		print(f"Solid Elements (101): {np.sum(material_tags == 101)} | Air Elements (202): {np.sum(material_tags == 202)}")
		
		return nodes, self.material_texture_generation_stride(material_tags)

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
		#blender 5.2 i have a sdf sphere equation of (np.linalg.norm(p - center, axis=-1) - radius) now i want to check if a 3d point which is part of a voxel is inside the SDF. How is this possible?

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

	# def generate_active_narrow_band_lattice_0(self, mesh_objects_list, nx=60, ny=60, nz=60, grid_scale=0.1):
	def generate_active_narrow_band_lattice_0(self):
		"""
		Generates a high-resolution, sliver-free multi-phase tetrahedral mesh 
		that ONLY allocates elements where geometry actually exists, saving 128GB RAM.
		"""
		# 1. COMPILE ALL INPUT MESHES INTO A UNIFIED SPATIAL SEARCH ENGINE
		# (Combines bones, muscles, and outer skin layers into one temporary validation mesh)
		# For testing, assume a unified BVHTree of the whole character
		# unified_bvh = build_unified_bvh_from_collection(mesh_objects_list)

		nx=60
		ny=60
		nz=60
		grid_scale=0.1
		
		x = np.linspace(-nx*grid_scale/2, nx*grid_scale/2, nx+1, dtype=np.float64)
		y = np.linspace(-ny*grid_scale/2, ny*grid_scale/2, ny+1, dtype=np.float64)
		z = np.linspace(0, nz*grid_scale, nz+1, dtype=np.float64)
		
		active_voxels = []
		node_registry = {}
		current_node_counter = 0

		# def sdf_sphere(self, p, center, radius):
		# 	return np.linalg.norm(p - center, axis=-1) - radius

		print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
		print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
		print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
		
		# 2. THE SPATIAL PRUNING PASS
		# Loop through the massive high-res grid conceptually, but filter aggressively
		for i in range(nx):
			for j in range(ny):
				for k in range(nz):
					# Calculate the exact center point of this specific voxel cell
					cx = (x[i] + x[i+1]) * 0.5
					cy = (y[j] + y[j+1]) * 0.5
					cz = (z[k] + z[k+1]) * 0.5
					co = mathutils.Vector([cx, cy, cz])
					
					# Check distance to closest tissue surface
					# loc, normal, face_idx, dist = unified_bvh.find_nearest(co)

					center = np.array([0.0, 0.0, 0.0])
					radius = 2.0

					p_single = np.array([co.x, co.y, co.z])
					sdf_value = np.linalg.norm(p_single - center) - radius
					is_inside = sdf_value <= 0

					# if is_inside == True:
					# 	print(f"SDF Value: {sdf_value}, Is inside: {is_inside}")

					# return
					
					# NARROW BAND CRITERIA: Only keep the voxel if it is inside the body, 
					# or within a tight 1-voxel padding envelope of the outer skin shell.
					# is_inside = False
					# if loc is not None:
					# 	if (co - loc).dot(normal) <= 0.0 or dist <= (grid_scale * 1.2):
					# 		is_inside = True
							
					# if is_inside:
					# 	# Save the voxel indices
					# 	active_voxels.append((i, j, k))
						
					# 	# Register only the 8 corner nodes of this active cell into memory
					# 	for di in:
					# 		for dj in:
					# 			for dk in:
					# 				coord_key = (i+di, j+dj, k+dk)
					# 				if coord_key not in node_registry:
					# 					node_registry[coord_key] = current_node_counter
					# 					current_node_counter += 1

		print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
		print('~~~~~~~~~~~~~~~ DONE ~~~~~~~~~~~~~~')
		print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

		return

		# # 3. RECONSTRUCT CONTINUOUS NODE CORNER COORDINATE ARRAYS
		# nodes_fl64 = np.zeros((current_node_counter, 3), dtype=np.float64)
		# for coord_key, flat_id in node_registry.items():
		# 	nodes_fl64[flat_id] = [x[coord_key[0]], y[coord_key[1]], z[coord_key[2]]]

		# # 4. ALTERNATING 5-TET TOPOLOGY GENERATION (Only processing filled space)
		# tet_indices = []
		# for (i, j, k) in active_voxels:
		# 	# Map 3D coordinate keys directly to your compressed flat node tracker IDs
		# 	n0 = node_registry[(i,   j,   k)]
		# 	n1 = node_registry[(i+1, j,   k)]
		# 	n2 = node_registry[(i+1, j+1, k)]
		# 	n3 = node_registry[(i,   j+1, k)]
		# 	n4 = node_registry[(i,   j,   k+1)]
		# 	n5 = node_registry[(i+1, j,   k+1)]
		# 	n6 = node_registry[(i+1, j+1, k+1)]
		# 	n7 = node_registry[(i,   j+1, k+1)]
			
		# 	# Apply your exact same sliver-free alternating topology configs smoothly
		# 	if (i + j + k) % 2 == 0:
		# 		tet_indices.append([n0, n1, n3, n4])
		# 		tet_indices.append([n1, n2, n3, n6])
		# 		tet_indices.append([n1, n4, n5, n6])
		# 		tet_indices.append([n3, n4, n6, n7])
		# 		tet_indices.append([n1, n3, n4, n6])
		# 	else:
		# 		tet_indices.append([n0, n1, n2, n5])
		# 		tet_indices.append([n0, n2, n3, n7])
		# 		tet_indices.append([n0, n4, n5, n7])
		# 		tet_indices.append([n2, n5, n6, n7])
		# 		tet_indices.append([n0, n2, n5, n7])

		# return nodes_fl64, np.array(tet_indices, dtype=np.int32)

	def generate_sliver_free_lattice(self, nx, ny, nz, scale=1.0):
		"""
		Generates a 100% stable, uniform tetrahedral grid using a 5-tet voxel split.
		"""
		x = np.linspace(0, nx * scale, nx + 1, dtype=np.float64)
		y = np.linspace(0, ny * scale, ny + 1, dtype=np.float64)
		z = np.linspace(0, nz * scale, nz + 1, dtype=np.float64)
		X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
		nodes = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

		# Helper function to map a 3D grid coordinate to its flat 1D node index
		def get_node_idx(i, j, k):
			return i * (ny + 1) * (nz + 1) + j * (nz + 1) + k

		tet_indices = []

		# 2. Iterate through cells and apply alternating 5-tet cuts
		for i in range(nx):
			for j in range(ny):
				for k in range(nz):
					# Fetch the 8 corner indices of the current voxel cell
					n0 = get_node_idx(i,   j,   k)
					n1 = get_node_idx(i+1, j,   k)
					n2 = get_node_idx(i+1, j+1, k)
					n3 = get_node_idx(i,   j+1, k)
					n4 = get_node_idx(i,   j,   k+1)
					n5 = get_node_idx(i+1, j,   k+1)
					n6 = get_node_idx(i+1, j+1, k+1)
					n7 = get_node_idx(i,   j+1, k+1)

					# Alternating topology configuration ensures a perfectly conformal mesh
					if (i + j + k) % 2 == 0:
						tet_indices.append([n0, n1, n3, n4])
						tet_indices.append([n1, n2, n3, n6])
						tet_indices.append([n1, n4, n5, n6])
						tet_indices.append([n3, n4, n6, n7])
						tet_indices.append([n1, n3, n4, n6])  # Center tet
					else:
						tet_indices.append([n0, n1, n2, n5])
						tet_indices.append([n0, n2, n3, n7])
						tet_indices.append([n0, n4, n5, n7])
						tet_indices.append([n2, n5, n6, n7])
						tet_indices.append([n0, n2, n5, n7])  # Center tet

		return nodes, np.array(tet_indices, dtype=np.int32)

	def dFEM(self, abj_sd_b_instance):
		abj_sd_b_instance.deselectAll()
		abj_sd_b_instance.deleteAllObjects()
		abj_sd_b_instance.mega_purge()

		grad_tanh = jax.grad(self.tanh)
		print(grad_tanh(1.0))
		# prints 0.4199743

		self.testScene()

		# self.testVDB_02(0, 8, 2)
		# self.testVDB_03(0, 8, 2) #########
		# self.bakeVDB(0)
		# self.bakeVDB(0, 64)

	def testScene(self):
		#threshold / density / radius

		# self.testVDB_04(0, 8, 2)
		# self.testVDB_04(0, 8, 2)
		# self.testVDB_04(0, 100, 6)
		self.testVDB_04(0, 128, 2)

		# self.generate_active_narrow_band_lattice_0()

		return
