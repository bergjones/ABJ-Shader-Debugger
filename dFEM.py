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

		
	def run_tr_bdf2_time_step(self, myEquation_dFEM, nodes_tet10, topology_tet10, element_properties, x_t, v_t, F_ext, dt, tol=1e-5, max_newton_iter=5):
		'''
		TR-BDF2 References
		https://www.sciencedirect.com/science/article/pii/S0898122121001267
		https://en.wikipedia.org/wiki/Backward_differentiation_formula
		https://en.wikipedia.org/wiki/Trapezoidal_rule

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
		
		# Establish boundary conditions tracking (nodes locked to floor)
		fixed_node_indices = np.where(nodes_tet10[:, 2] <= 0.001)[0]
		fixed_dofs = []
		for node_idx in fixed_node_indices:
			fixed_dofs.extend([node_idx*3, node_idx*3+1, node_idx*3+2])
		fixed_dofs = np.array(fixed_dofs, dtype=np.int32)
		
		# ... [Establish fixed_dofs and M_diag arrays arrays] ...
		
		# Calculate mass vector per node based on element property distributions
		# Represented here as a lumped mass diagonal vector for performance efficiency
		M_diag = np.ones(dof, dtype=np.float64) * 0.1 
		M_diag[fixed_dofs] = 1.0 # Protect fixed boundary math diagonals

		gamma = 2.0 - np.sqrt(2.0)
		
		# Initialize your reference state tracking operator at the historical position
		op_t = MatrixFreeTet10Operator(nodes_tet10, topology_tet10, element_properties, x_t - nodes_tet10, fixed_dofs, myEquation_dFEM)
		
		# Exact Force Gathering Check: Read the true force directly from the initial operator state
		f_int_t = op_t.compute_forces_and_action(p_vector=None)

		# ==========================================================================
		# SUBSTEP 1: TRAPEZOIDAL RULE STEP (From t to t + gamma*dt)
		# ==========================================================================
		dt1 = gamma * dt
		x_gamma = x_t.copy()
		v_gamma = v_t.copy()

		for n_iter in range(max_newton_iter):
			# Re-instantiate the operator at the current trial position coordinates
			op_gamma = MatrixFreeTet10Operator(nodes_tet10, topology_tet10, element_properties, x_gamma - nodes_tet10, fixed_dofs, myEquation_dFEM)
			
			# Calculate internal forces for this Newton iteration pass pass
			f_int_gamma = op_gamma.compute_forces_and_action(p_vector=None)
			
			# Calculate your step 1 residual vector mapping mapping
			R = M_diag * (v_gamma.ravel() - v_t.ravel()) - (dt1 / 2.0) * (f_int_t + f_int_gamma + 2.0 * F_ext)
			R_pos = x_gamma.ravel() - x_t.ravel() - (dt1 / 2.0) * (v_t.ravel() + v_gamma.ravel())
			R_combined = R + M_diag * (R_pos / dt1)
			R_combined[fixed_dofs] = 0.0
			
			if np.linalg.norm(R_combined) < tol:
				break

			# Define the Jacobian operator mapping for the Conjugate Gradient solver
			class Substep1JacobianOperator(splinalg.LinearOperator):
				def __init__(self, shape, dtype):
					self.shape, self.dtype = shape, dtype
				def _matvec(self, p):
					# We leverage op_gamma's fast tracking channel to compute Ke * p
					_, y_action = op_gamma.compute_forces_and_action(p)
					return M_diag * p - (dt1 / 2.0) * y_action

			J_op = Substep1JacobianOperator((len(R_combined), len(R_combined)), np.float64)
			
			delta_v_flat, _ = splinalg.cg(J_op, -R_combined, tol=1e-6)
			v_gamma += delta_v_flat.reshape(-1, 3)
			x_gamma += (delta_v_flat * (dt1 / 2.0)).reshape(-1, 3)

		# ==========================================================================
		# SUBSTEP 2: BDF2 STEP (From t + gamma*dt to t + dt)
		# ==========================================================================
		dt2 = (1.0 - gamma) * dt
		d = dt2 / (dt1 + dt2)

		# Coefficients born directly from BDF2 polynomial tracking formulas
		alpha_bdf = (1.0 + 2.0 * d) / (1.0 + d)
		beta_bdf  = (1.0 + d) / (1.0 + d)  # Note: formulas adapt dynamically based on gamma

		x_next = x_gamma.copy()
		v_next = v_gamma.copy()

		# Newton-Raphson Loop for Substep 2
		for n_iter in range(max_newton_iter):
			op_next = MatrixFreeTet10Operator(nodes_tet10, topology_tet10, element_properties, x_next - nodes_tet10, fixed_dofs, myEquation_dFEM)
			
			f_int_next = op_next.compute_forces_and_action(p_vector=x_next)
			
			# Calculate step 2 residual mapping
			# R = M * (alpha_bdf * v_next - combined_past_history) - dt2 * (F_int(next) + F_ext)
			# For TR-BDF2 standard configurations, the composite formula evaluates cleanly as:
			v_history_flat = (1.0 / (gamma * (2.0 - gamma))) * v_gamma.ravel() - (((1.0 - gamma)**2) / (gamma * (2.0 - gamma))) * v_t.ravel()
			R = M_diag * (v_next.ravel() - v_history_flat) - (dt * (2.0 - gamma) / 2.0) * (f_int_next + F_ext)
			
			R_pos = x_next.ravel() - (((1.0 - gamma)**2) / (gamma * (2.0 - gamma))) * x_t.ravel() # Historical positions bounds
			
			R_combined = R + M_diag * (R_pos / dt)
			R_combined[fixed_dofs] = 0.0
			
			if np.linalg.norm(R_combined) < tol:
				break
				
			class Substep2JacobianOperator(splinalg.LinearOperator):
				def __init__(self, shape, dtype):
					self.shape, self.dtype = shape, dtype
				def _matvec(self, p):
					return M_diag * p - (dt * (2.0 - gamma) / 2.0) * op_next._matvec(p)
					
			J_op2 = Substep2JacobianOperator((dof, dof), np.float64)

			delta_v_flat, _ = splinalg.cg(J_op2, -R_combined, tol=1e-6)

			v_next += delta_v_flat.reshape(-1, 3)
			x_next += (delta_v_flat * (dt * (2.0 - gamma) / 2.0)).reshape(-1, 3)
			
		return x_next, v_next
	
	def compute_tet10_multiphase_dual_kernel(element_node_coords, element_displacements, element_velocities, p_element_trial, E, nu, mat_id):
		"""
		Production Multi-Phase Engine: Evaluates Solid, Air, and Navier-Stokes Liquid 
		phases simultaneously inside a single high-order 4-point Gauss Quadrature loop.

		# ==============================================================================
		# CONTINUUM MECHANICS MATRIX-FREE KERNEL
		# ==============================================================================
		# This kernel evaluates the action of the Tangent Stiffness Operator (Ke * p)
		# for a High-Order Quadratic Tetrahedron (Tet10) under a Stable Neo-Hookean
		# energy potential model. 
		#
		# MATHEMATICAL & ENGINEERING REFERENCES:
		# 1. Finite Element Framework: https://en.wikipedia.org/wiki/Finite_element_method
		# 2. Quadrature Volume Integration: https://en.wikipedia.org/wiki/Gaussian_quadrature
		# 3. Kinematic Kinematics (Tensor F): https://en.wikipedia.org/wiki/Finite_strain_theory
		# 4. Material Constitutive Law: https://en.wikipedia.org/wiki/Neo-Hookean_solid
		#5. Stable Neo-Hookean Flesh Simulation : Smith, De Goes, Kim : https://research.pixar.com/docs/2018.SiggraphPapers.SGK.b.pdf
		
		Args:
			element_velocities: (10, 3) float64 array of current frame node velocities.
		"""
		# Initialize output vectors
		f_int_element = np.zeros((10, 3), dtype=np.float64)
		q_flat = np.zeros(30, dtype=np.float64)
		p_flat = p_element_trial.ravel()

		# 4-Point Gauss Quadrature Constants
		a = 0.5854101966249685
		b = 0.1381966011250105
		# gauss_points = np.array([[a,b,b], [b,a,b], [b,b,a], [b,b,b]], dtype=np.float64)
		# # gauss_points = np.array([[a,b,b,b], [b,a,b,b], [b,b,a,b], [b,b,b,b]], dtype=np.float64)
		
		gauss_points = np.array([
		[a, b, b], 
		[b, a, b], 
		[b, b, a], 
		[b, b, b]
		], dtype=np.float64)
		
		gauss_weight = 1.0 / 24.0  

		# r = gauss_points[:, 0]  # Array: [a, b, b, b]
		# s = gauss_points[:, 1]  # Array: [b, a, b, b]
		# t = gauss_points[:, 2]  # Array: [b, b, a, b]

		for gp in gauss_points:
			r, s, t = gp[0], gp[1], gp[2]
			# r, s, t = gp[:, 0], gp[:, 1], gp[:, 2]
			u = 1.0 - r - s - t

			dN_dr = np.array([4*r - 1, 0, 0, -4*u + 1,  4*s,  -4*s, 0, 4*t, 0, -4*t])
			dN_ds = np.array([0, 4*s - 1, 0, -4*u + 1,  4*r,   0, -4*t, 0, 4*t, -4*r])
			dN_dt = np.array([0, 0, 4*t - 1, -4*u + 1,   0,  -4*r,  4*s, 0, 4*r, -4*s])

			dN_dlocal = np.stack([dN_dr, dN_ds, dN_dt], axis=0)

			# Map to global world space coordinates
			# Jacobian = dN_dlocal @ element_node_coords

			print('r = ', r)
			print('s = ', s)
			print('t = ', t)

			# r =  0.5854101966249685
			# s =  0.1381966011250105
			# t =  0.1381966011250105

			# Fix the Einstein Summation string for 2D inputs:
			# i = 3 local axes, n = 10 nodes, j = 3 global axes (X, Y, Z)
			# This outputs a single square 3x3 Jacobian matrix for this specific loop iteration
			Jacobian = np.einsum('in,nj->ij', dN_dlocal, element_node_coords) # Shape: (3, 3)
    
			# Jacobian = np.einsum('ing,nj->gij', dN_dlocal, element_node_coords) # Outputs: (4, 3, 3)
			# Jacobian = np.einsum('ing,nj->ijg', dN_dlocal, element_node_coords) # $$$$$$$$$$$$$$
			# Jacobian = np.einsum('ijg,jk->igk', dN_dlocal, element_node_coords) ########
			# Jacobian = np.einsum('ink,nj->ijk', dN_dlocal, element_node_coords)
			# Jacobian = np.einsum('bqin,bnj->bqij', dN_dlocal, element_node_coords)
			# Jacobian = np.einsum('ijg,jk->gik', dN_dlocal, element_node_coords)

			# Jacobian = np.moveaxis(Jacobian_raw, 2, 0)

			# then when i try to multiply the dn_dlocal stack by np.vstack "element_node_coords" with this command (Jacobian = dN_dlocal @ element_node_coords) I get valueerror: matmul: input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?)(size 10 is different from 3)

			# print('$$$$$$$$$$$$$$$$$$')
			# print('dN_dlocal shape = ', dN_dlocal.shape)
			# print('dN_dlocal shape = ', Jacobian.shape)
			# print('$$$$$$$$$$$$$$$$$$')


			#input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?, k), (k, m?)->(n?,m?)(size 3 is different from 10)

			det_J = np.linalg.det(Jacobian)

			# if det_J <= 0.0:
			# 	raise ValueError("Critical Element Inversion Safeguard Triggered: Mesh geometry crushed.")

			# inv_Jacobian = None

			# Add a tiny epsilon guard if the matrix is singular or collapsed
			# if np.abs(det_J.any()) < 1e-9:
			# 	# Option A: Use pseudo-inverse to handle degenerate elements gracefully
			# 	inv_Jacobian = np.linalg.pinv(Jacobian)
			# else:
			# 	inv_Jacobian = np.linalg.inv(Jacobian)

			inv_Jacobian = np.linalg.inv(Jacobian)
			# dN_dglobal = inv_Jacobian @ dN_dlocal
			# dN_dglobal = np.einsum('gij,jng->gin', inv_Jacobian, dN_dlocal)

			# Calculate dN_dglobal for this iteration
			# i = 3 global spatial axes, j = 3 local derivative axes, n = 10 nodes
			# (3, 3) @ (3, 10) -> Outputs a (3, 10) matrix natively using the @ operator
			dN_dglobal = inv_Jacobian @ dN_dlocal # Shape: (3, 10)


			# disp_gradient = np.einsum('ni,gjn->gij', element_displacements, dN_dglobal)
			disp_gradient = dN_dglobal @ element_displacements # Shape: (3, 3)
			# disp_gradient = dN_dglobal @ element_displacements # Shape: (3, 3)

			# F = np.eye(3, dtype=np.float64)[np.newaxis, :, :] + disp_gradient

			# F = np.eye(3, dtype=np.float64) + (element_displacements.T @ dN_dglobal.T)
			F = np.eye(3, dtype=np.float64) + disp_gradient

			print('!!!!!! disp_gradient = ', disp_gradient)


			print('!!!!!! dN_dglobal.shape = ', dN_dglobal.shape)

			# Compute Kinematics
			# F = np.eye(3, dtype=np.float64) + (element_displacements.T @ dN_dglobal.T) ###
			# F = np.eye(3, dtype=np.float64) + (np.einsum('gij,jng->gin', element_displacements, dN_dglobal.T))
			J_vol = np.linalg.det(F)

			# ======================================================================
			# PHASE-SPECIFIC MATERAl CONSTITUTIVE SWAPPING
			# ======================================================================
			if mat_id == 101.0:
				# --- PHASE A: SOLID TISSUE (Stable Neo-Hookean) ---
				mu = E / (2.0 * (1.0 + nu))
				lambda_param = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
				alpha = 1.0 + (mu / lambda_param)
				
				stress_scale = mu * (1.0 - (1.0 / (np.trace(F.T @ F) + 1.0)))
				F_cofactor = np.zeros((3,3), dtype=np.float64)
				for i in range(3): F_cofactor[:, i] = np.cross(F[:, (i+1)%3], F[:, (i+2)%3])
				P_stress = stress_scale * F + (lambda_param * (J_vol - alpha)) * F_cofactor
				
				# Stiffness multipliers for Solid
				mu_stiff = mu
				lambda_stiff = lambda_param

			elif mat_id == 400.0:
				# --- PHASE B: LIQUID PHASE (Navier-Stokes Continuum) ---
				# E behaves as Bulk Modulus (Kf), nu behaves as Dynamic Viscosity (Visco)
				Kf = E   # e.g., 50000.0 for snappy water
				viscosity = nu # e.g., 50.0 for water, 5000.0 for thick syrup/honey
				
				# 1. Compute Fluid Pressure via Equation of State (EOS)
				pressure = Kf * (J_vol - 1.0)
				
				# 2. Compute Spatial Velocity Gradient L = grad(v)
				L = element_velocities.T @ dN_dglobal.T
				
				# 3. Rate-of-Strain Tensor D = 0.5 * (L + L^T)
				D_tensor = 0.5 * (L + L.T)
				
				# 4. Total Stress Tensor P = -p*I + 2*viscosity*D
				P_stress = -pressure * np.eye(3, dtype=np.float64) + 2.0 * viscosity * D_tensor
				
				# Tangent dampening coefficients mapping to the stiffness action loops
				mu_stiff = viscosity
				lambda_stiff = Kf

			else:
				# --- PHASE C: AMBIENT AIR BUFFER MATRIX ---
				pressure = -100.0 * (J_vol - 1.0)
				P_stress = pressure * np.eye(3, dtype=np.float64)
				# fluid_pressure = -100.0 * (J_vol - 1.0)
				# P_stress = fluid_pressure * np.eye(3, dtype=np.float64)
				mu_stiff = 10.0
				lambda_stiff = 100.0

			# ======================================================================
			# CORE ACCUMULATION PASS
			# ======================================================================
			# Force integration mapping: f_int = P * dN_dglobal
			f_int_element += (P_stress @ dN_dglobal * det_J * gauss_weight).T

			# Matrix-free stiffness action map: q = Ke * p
			for i in range(10):
				dNi = dN_dglobal[:, i]
				row_idx = i * 3
				q_node_block = np.zeros(3, dtype=np.float64)
				for j in range(10):
					dNj = dN_dglobal[:, j]
					p_node_j = p_flat[j*3 : j*3 + 3]
					
					mat_term = dNi * np.dot(dNj, p_node_j) * lambda_stiff
					geom_term = p_node_j * (np.dot(dNi, dNj) * mu_stiff)
					q_node_block += mat_term + geom_term
					
				q_flat[row_idx : row_idx + 3] += q_node_block * det_J * gauss_weight

			print('DONE ~~~~~~~~~~~~~~')
			print('DONE ~~~~~~~~~~~~~~')
			print('DONE ~~~~~~~~~~~~~~')
			print('DONE ~~~~~~~~~~~~~~')
			print('DONE ~~~~~~~~~~~~~~')
			print('DONE ~~~~~~~~~~~~~~')

		return f_int_element, q_flat.reshape(10, 3)

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
		local_edges = [(0,1), (1,2), (2,0), (0,3), (1,3), (2,3)]

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
		
		# Define Kuhn 5-split local offsets mapping voxels to 5 distinct tets
		kuhn_template = np.array([
			[0, 1, 2, 4],  # Corner 1
			[1, 3, 2, 7],  # Corner 2
			[1, 5, 4, 7],  # Corner 3
			[4, 6, 2, 7],  # Corner 4
			[1, 2, 4, 7]   # Central Core
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
		
		# Phase 1: Soft Tissue (Sphere)
		phase_tags[ds_centers <= 0] = 1
		
		# Phase 2: Rigid Structure (Cube)
		phase_tags[db_centers <= 0] = 2
		
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

		#threshold / density / radius



		a = 0.5854101966249685
		b = 0.1381966011250105

		gauss_points = np.array([
		[a, b, b], 
		[b, a, b], 
		[b, b, a], 
		[b, b, b]
		], dtype=np.float64)
		


		for idx, gp in enumerate(gauss_points):
			r, s, t = gp[0], gp[1], gp[2]

			print('idx = ', idx)
			print('r = ', r)
			print('s = ', s)
			print('t = ', t)

		'''
		idx =  0
		r =  0.5854101966249685
		s =  0.1381966011250105
		t =  0.1381966011250105
		idx =  1
		r =  0.1381966011250105
		s =  0.5854101966249685
		t =  0.1381966011250105
		idx =  2
		r =  0.1381966011250105
		s =  0.1381966011250105
		t =  0.5854101966249685
		idx =  3
		r =  0.1381966011250105
		s =  0.1381966011250105
		t =  0.1381966011250105
		'''










		return





		self.testVDB_06(abj_sd_b_instance) ####

		return

	def deform_skin_tissue_mesh(self, x_corners_current, topology_tet4, vertex_to_tet_id, vertex_weights):
		"""
		Deforms the high-resolution render skin by multiplying current cage states
		by cached barycentric weights. Completely matrix-free.
		
		Args:
			x_corners_current: (N, 3) float64 array of current frame node positions from solver.
			topology_tet4: (M, 4) int32 array tracking element connectivity.
			vertex_to_tet_id: (V,) int32 pre-computed containment map.
			vertex_weights: (V, 4) float64 pre-computed barycentric mapping arrays.
			
		Returns:
			deformed_skin_coords: (V, 3) float32 array ready for native Blender casting.
		"""
		# 1. FETCH UPDATED CAGE NODE COORDINATES FOR EVERY VERTEX'S TARGET TET
		# Gather element corner point pointers matching vertex mappings
		active_tets = topology_tet4[vertex_to_tet_id] # Shape: (V, 4)
		
		# Extract absolute 3D position matrices for the 4 corners of all tets simultaneously
		# Produces a high-dimensional vector array: (V, 4, 3)
		tet_nodes_x = x_corners_current[active_tets]
		
		# 2. RUN BARYCENTRIC RECONSTRUCTION LOP
		# New_Pos = w0*v0 + w1*v1 + w2*v2 + w3*v3
		# We expand vertex_weights dimension footprint to multiply cleanly across the 3D columns
		w_expanded = vertex_weights[:, :, np.newaxis] # Shape: (V, 4, 1)
		
		# Multiply elements element-by-element and sum along the tet corner axis
		deformed_skin_fl64 = np.sum(tet_nodes_x * w_expanded, axis=1) # Shape: (V, 3)
		
		# 3. DOWNCAST TO SINGLE PRECISION ONLY AT GRAPHICS HANDOFF BOUNDARY
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

	def testVDB_06(self, abj_sd_b_instance):
		#look @ execute_production_fem_bake_with_skin

		nodes_tet4, topology_tet4, tags, vdb_sphere_sdf_l, vdb_sphere_sdf_h, vdb_box_sdf_l, vdb_box_sdf_h = self.generate_global_multiphase_mesh(24, 96)
		# nodes_tet4, topology_tet4, tags, ds_toPoly, db_toPoly = self.generate_global_multiphase_mesh(resolution=96)

		self.visualize_global_multiphase_slice(nodes_tet4, topology_tet4, tags, slice_axis=0, slice_val=0.0) ###########

		# layer_data_s_l = [("sdf_joined", vdb_sphere_sdf_l)]
		# myBox_l = self.sdf_vdb_visualizer(layer_data_s_l)

		# layer_data_b_l = [("sdf_joined", vdb_box_sdf_l)]
		# mySphere_l = self.sdf_vdb_visualizer(layer_data_b_l)


		layer_data_s_h = [("sdf_joined", vdb_sphere_sdf_h)]
		mySphere_h = self.sdf_vdb_visualizer(layer_data_s_h)

		# layer_data_b_h = [("sdf_joined", vdb_box_sdf_h)]
		# myBox_h = self.sdf_vdb_visualizer(layer_data_b_h)

		vertex_tet_ids, vertex_bary_weights = self.precompute_skin_barycentric_weights(mySphere_h, nodes_tet4, topology_tet4)

		nodes_tet10, topology_tet10 = self.convert_tet4_lattice_to_tet10(nodes_tet4, topology_tet4)

		###############################
		#### BAKE
		##############################
		# obj = bpy.context.scene.objects.get(high_res_mesh_name)
		obj = mySphere_h
		mesh = obj.data

		total_frames = 10
		frame_dt=0.01

		# Initialize simulation states
		x_current = nodes_tet10.copy()
		v_current = np.zeros((len(nodes_tet10), 3), dtype=np.float64)

		# F_ext = np.zeros(len(nodes_tet10)*3, dtype=np.float64)
		F_ext = np.array([0, -9.81, 0])

		if not mesh.shape_keys:
			obj.shape_key_add(name="Basis")

		# --- PHASE 2: THE TR-BDF2 TIME STRIDE LOOP ---
		for frame in range(1, total_frames + 1):
			bpy.context.scene.frame_set(frame)
			x_next, v_next = self.run_tr_bdf2_time_step(myEquation_dFEM,
				nodes_tet10, topology_tet10, tags, 
				x_current, v_current, F_ext, frame_dt)
			
			x_current, v_current = x_next, v_next

			# 2. Extract active frame cage corner states
			num_corners = len(nodes_tet4)
			x_corners_current = x_current[0:num_corners]

			# 3. STREAMING SKIN DEFORMATION PASS
			# Evaluates the high-res vertex tracking vectors seamlessly
			deformed_skin_coords_32 = self.deform_skin_tissue_mesh(
				x_corners_current, topology_tet4, vertex_tet_ids, vertex_bary_weights
			)

			# 4. BAKE TO NATIVE BLENDER ANIMATION timetracks
			sk = obj.shape_key_add(name=f"FEM_Frame_{frame:04d}")
			
			# Shift local coordinates back to local object space before caching inside data-block
			# Reverses world matrix transformations to prevent double-transform artifacts during joint actions
			inv_world_matrix = np.array(obj.matrix_world.inverted(), dtype=np.float32)[:3, :4]
			local_skin_coords = (deformed_skin_coords_32 @ inv_world_matrix[:, :3].T) + inv_world_matrix[:, 3]

			# Push raw array memory block straight into Blender's C-arrays instantaneously
			sk.data.foreach_set("co", local_skin_coords.ravel())
			
			# Insert evaluation timeline driving metrics
			sk.value = 0.0
			sk.keyframe_insert(data_path="value", frame=frame - 1)
			sk.value = 1.0
			sk.keyframe_insert(data_path="value", frame=frame)
			sk.value = 0.0
			sk.keyframe_insert(data_path="value", frame=frame + 1)

	def precompute_skin_barycentric_weights(self, render_mesh_obj, nodes_tet4, topology_tet4):
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

class MatrixFreeTet10Operator(splinalg.LinearOperator):
	def __init__(self, nodes_tet10, topology_tet10, element_properties, current_displacements, fixed_dofs, myEquation_dFEM):
		self.nodes = nodes_tet10
		self.topology = topology_tet10
		self.properties = element_properties
		self.current_U = current_displacements
		self.fixed_dofs = fixed_dofs
		self.dof = len(nodes_tet10) * 3
		self.shape = (self.dof, self.dof)
		self.dtype = np.float64
		self.myEquation_dFEM_usable = myEquation_dFEM

	def compute_forces_and_action(self, p_vector=None):
		"""
		The production entry point. If p_vector is None, it returns ONLY the 
		global internal force vector. If p_vector is provided, it returns BOTH.
		"""
		f_int_global = np.zeros(self.dof, dtype=np.float64)
		y_action_global = np.zeros(self.dof, dtype=np.float64)

		p_nodes = np.zeros((len(self.nodes), 3), dtype=np.float64)

		p_constrained = np.zeros((len(self.nodes), 3), dtype=np.float64)

		if p_vector is not None:
			p_constrained = p_vector.copy()
			p_constrained[self.fixed_dofs] = 0.0
			p_nodes = p_constrained.reshape(-1, 3)

		# p_velocity = np.zeros((len(self.nodes), 3), dtype=np.float64)
		p_velocity = p_constrained.reshape(-1, 3)	

		# ONE UNIFIED LOOP FOR ALL CONTINUUM PHYSICS
		#self.compile_painted_mesh_to_fem_attributes()
		for t_idx, tet in enumerate(self.topology):
			# mat_id, E, nu, density = self.properties[t_idx]
			mat_id = None
			E = None
			nu = None
			density = None

			if self.properties[t_idx] == 0: #air
				mat_id = 202.0
				E = 0
				nu = .001
				density = 0

			elif self.properties[t_idx] == 1: #solid SPHERE
				mat_id = 101.0
				E = 5000
				nu = .499 # .3 - .499
				density = 1

			elif self.properties[t_idx] == 2: #solid CUBE
				mat_id = 101.0
				E = 5000
				nu = .499 # .3 - .499
				density = 0
			
			f_local, q_local = self.myEquation_dFEM_usable.compute_tet10_multiphase_dual_kernel(self.nodes[tet], self.current_U[tet], p_velocity[tet], p_nodes[tet], E, nu, mat_id)
			
			# SCATTER PASS
			for local_idx, global_node_idx in enumerate(tet):
				start = global_node_idx * 3
				f_int_global[start : start + 3] += f_local[local_idx]

				if p_vector is not None:
					y_action_global[start : start + 3] += q_local[local_idx]

		if p_vector is not None:
			y_action_global[self.fixed_dofs] = p_vector[self.fixed_dofs] * 1.0
			return f_int_global, y_action_global
			
		return f_int_global

	def _matvec(self, p):
		# Mandatory SciPy callback. Evaluates strictly the action product channels
		_, y_action = self.compute_forces_and_action(p)
		return y_action