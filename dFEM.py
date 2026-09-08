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
import jax
import jax.numpy as jnp
from jax import jit

# Force JAX to use 64-bit double precision to maintain mechanical engineering accuracy
jax.config.update("jax_enable_x64", True)

bpy.utils.expose_bundled_modules()
import openvdb as vdb

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
		F = jnp.eye(3) + disp_grad
		
		P_stress = evaluate_p_stress_jax(F, disp_grad, vel_grad, mat_id, E, nu, dt1)
		f_int_element += (P_stress @ dN_dx * dV).T
		
	return f_int_element	


import jax.numpy as jnp

def evaluate_p_stress_jax(F_eval, disp_grad_eval, vel_grad_eval, mat_id, E, nu, dt_scale):
	"""Computes First Piola-Kirchhoff stress tensor universally for any phase using pure JAX mathematical branches."""
	J_vol = jnp.linalg.det(F_eval)

	# ----------------------------------------------------------------------
	# PHASE A: SOLID TISSUE (Stable Neo-Hookean)
	# ----------------------------------------------------------------------
	mu_solid = E / (2.0 * (1.0 + nu))
	lambda_solid = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
	alpha = 1.0 + (mu_solid / lambda_solid)
	stress_scale = mu_solid * (1.0 - (1.0 / (jnp.trace(F_eval.T @ F_eval) + 1.0)))

	# Differentiable cofactor formulation using matrix inverses
	F_cofactor = J_vol * jnp.linalg.inv(F_eval).T
	P_solid = stress_scale * F_eval + (lambda_solid * (J_vol - alpha)) * F_cofactor
		
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

class Substep1JacobianOperatorJAX_debug(splinalg.LinearOperator):
# class Substep1JacobianOperatorJAX_debug(LinearOperator):
	# def __init__(self, num_dofs_int, dtype, u_global, v_global, nodes_global, topology, properties, fixed_dofs, dt1, M_diag):
	def __init__(self, dtype, num_dofs_int):
		clean_dof = int(num_dofs_int)
		# super().__init__(dtype=np.dtype(dtype), shape=(clean_dof, clean_dof))
		# super().__init__(dtype=np.dtype(dtype), shape=(clean_dof, clean_dof))

		super().__init__(np.dtype(dtype), (clean_dof, clean_dof))


		
		# self.shape = (int(dof_size), int(dof_size))
		# self.dtype = np.dtype(dtype)

		# super().__init__(dtype=dtype, shape=(dof_size, dof_size))
	
		def _matvec(self, p):
			"""Computes J_op @ p globally in a SINGLE hardware pass via jax.vmap and jax.jvp."""
			
			# return p.ravel()
			return np.asarray(p, dtype=self.dtype).ravel()

# class Substep1JacobianOperatorJAX0(splinalg.LinearOperator):
class Substep1JacobianOperatorJAX(LinearOperator):
	def __init__(self, dtype, num_dofs_int, u_global, v_global, nodes_global, topology, properties, fixed_dofs, dt1, M_diag):
		# dof_size = int(shape[0])
		clean_dof = int(num_dofs_int)
		# super().__init__(dtype=dtype, shape=shape)
		super().__init__(dtype=dtype, shape=(clean_dof, clean_dof))

		# # Keep references to the entire global mesh data structure
		# self.u_global = np.array(u_global).reshape(-1, 3)
		# self.v_global = np.array(v_global).reshape(-1, 3)
		# self.nodes_global = np.array(nodes_global)
		# self.topology = topology
		# self.properties = properties
		# self.fixed_dofs = fixed_dofs
		# self.dt1 = dt1
		# self.M_diag = np.array(M_diag).ravel()
		# # self.num_dofs = shape[0]
		# self.num_dofs = clean_dof

		# 1. Cast the ENTIRE global structure into JAX arrays once during initialization
		self.u_global = jnp.array(u_global).reshape(-1, 3)
		self.v_global = jnp.array(v_global).reshape(-1, 3)
		self.nodes_global = jnp.array(nodes_global)
		self.topology = jnp.array(topology)         # Shape: (Num_Elements, 10)
		self.properties = jnp.array(properties)     # Shape: (Num_Elements,)
		self.fixed_dofs = fixed_dofs
		self.M_diag = jnp.array(M_diag).ravel()
		# self.num_dofs = shape[0]
		self.num_dofs = clean_dof
		self.dt1 = dt1


		# 2. Pre-compile the multi-element batch wrapper using vmap
		# This transforms compute_element_forces_jax from operating on 1 element to N elements

		self.vmapped_forces = jax.vmap(
			compute_element_forces_jax, 
			in_axes=(0, 0, 0, 0, 0, 0, None) # Batch along axis 0 for local arrays, dt1 is shared
		)				

	# '''

	def _matvec(self, p):
		"""Computes J_op @ p globally in a SINGLE hardware pass via jax.vmap and jax.jvp."""
		p_constrained = p.copy().ravel()
		p_constrained[self.fixed_dofs] = 0.0
		p_nodes = jnp.array(p_constrained).reshape(-1, 3)

		time_scale = self.dt1 / 2.0

		# Gather all element local data slices across the mesh into batched 3D tensors instantly
		u_batched = self.u_global[self.topology]      # Shape: (Num_Elements, 10, 3)
		v_batched = self.v_global[self.topology]      # Shape: (Num_Elements, 10, 3)
		coords_batched = self.nodes_global[self.topology] # Shape: (Num_Elements, 10, 3)
		p_batched = p_nodes[self.topology]            # Shape: (Num_Elements, 10, 3)

		# Map property tags directly to continuous material tracking arrays for the batch
		# Air (0) -> mat_id=202, E=1, nu=0.001 | Solid (1) -> mat_id=101, E=10, nu=0.45
		mat_ids = jnp.where(self.properties == 1, 101.0, 202.0)
		Es = jnp.where(self.properties == 1, 10.0, 1.0)
		nus = jnp.where(self.properties == 1, 0.45, 0.001)

		# return 0

		# Define batch closures for JAX automatic differentiation
		# def batch_force_vs_disp(u_b):
		# 	return self.vmapped_forces(u_b, v_batched, coords_batched, mat_ids, list(Es), list(nus), self.dt1)
			
		# def batch_force_vs_vel(v_b):
		# 	return self.vmapped_forces(u_batched, v_b, coords_batched, mat_ids, list(Es), list(nus), self.dt1)

		# Remove list() wraps. Es and nus are already batched arrays!
		def batch_force_vs_disp(u_b):
			return self.vmapped_forces(u_b, v_batched, coords_batched, mat_ids, Es, nus, self.dt1)
			
		def batch_force_vs_vel(v_b):
			return self.vmapped_forces(u_batched, v_b, coords_batched, mat_ids, Es, nus, self.dt1)

		# Call JAX JVP ON THE ENTIRE MESH AT ONCE! No Python loops.
		_, dF_du_batched = jax.jvp(batch_force_vs_disp, (u_batched,), (p_batched * time_scale,))
		_, dF_dv_batched = jax.jvp(batch_force_vs_vel, (v_batched,), (p_batched,))
		
		q_batched_tet = dF_du_batched + dF_dv_batched # Shape: (Num_Elements, 10, 3)

		# Scatter-Add the local 10x3 calculations back into a global vector of size 11742
		# JAX's .at[index].add() optimizes this scatter pass seamlessly on hardware
		q_stiffness_global = jnp.zeros((len(self.u_global), 3))
		
		# Expand indices to match the batched flattened structures
		flat_topology = self.topology.ravel() # Shape: (Num_Elements * 10,)
		flat_q = q_batched_tet.reshape(-1, 3)  # Shape: (Num_Elements * 10, 3)
		
		q_stiffness_global = q_stiffness_global.at[flat_topology].add(flat_q)
		q_stiffness_flat = np.array(q_stiffness_global.ravel())

		# Enforce boundary conditions and pair with diagonal system inertia
		q_stiffness_flat[self.fixed_dofs] = 0.0
		return np.array(self.M_diag * p.ravel() + q_stiffness_flat)

		# ######################
		# #RECURSIVE debug 0
		# ######################

		# q_stiffness_global = jnp.zeros((len(self.u_global), 3))
		# q_stiffness_global = q_stiffness_global.at[self.topology.ravel()].add(q_batched_tet.reshape(-1, 3))

		# # --- THE RECURSION PROOF CONVERSION ---
		# # 1. Compute the final matrix multiplication mapping completely in pure JAX
		# result_jax = self.M_diag * jnp.array(p.ravel()) + q_stiffness_global.ravel()

		# # 2. Convert JAX array directly to a standard Python list, then build a raw float64 NumPy array
		# # This completely strips out any hidden JAX tracer attributes, preventing SciPy type loop recursion
		# result_np = np.array(result_jax.tolist(), dtype=np.float64)

		# # 3. Enforce boundary conditions cleanly onto the final real array vector
		# result_np[self.fixed_dofs] = 0.0

		# # Return a perfectly flat, clean standard NumPy vector of size 11742
		# return result_np.ravel()

		# '''

	# OLD
	# def _matvec(self, p):
	# 	"""Computes J_op @ p globally by looping over elements and applying JAX locally."""
	# 	# 1. Enforce boundary conditions and reshape the incoming search direction p
	# 	p_constrained = p.copy().ravel()
	# 	p_constrained[self.fixed_dofs] = 0.0
	# 	p_nodes = p_constrained.reshape(-1, 3)
		
	# 	# Initialize the global stiffness action vector (same length as R_combined)
	# 	q_stiffness_global = np.zeros(self.num_dofs, dtype=np.float64)
		
	# 	# Substep 1 Time Integration Multiplier
	# 	time_scale = self.dt1 / 2.0

	# 	# 2. THE GLOBAL TOPOLOGY LOOP
	# 	for t_idx, tet in enumerate(self.topology):
	# 		# Extract material properties for this specific element
	# 		if self.properties[t_idx] == 0:    # Air
	# 			mat_id, E, nu = 202.0, 1.0, 0.001
	# 		elif self.properties[t_idx] == 1:  # Solid Sphere
	# 			mat_id, E, nu = 101.0, 10.0, 0.45
	# 		else:
	# 			continue

	# 		# Isolate the exact 10 nodes for this tet element from global tracking fields
	# 		u_local = jnp.array(self.u_global[tet])
	# 		v_local = jnp.array(self.v_global[tet])
	# 		coords_local = jnp.array(self.nodes_global[tet])
	# 		p_local = jnp.array(p_nodes[tet]) # Local slice of solver perturbation direction

	# 		# Define localized inline functions for this specific tet's forces
	# 		def local_force_vs_disp(u_state):
	# 			return self.compute_element_forces_jax(u_state, v_local, coords_local, mat_id, E, nu, time_scale)
				
	# 		def local_force_vs_vel(v_state):
	# 			return self.compute_element_forces_jax(u_local, v_state, coords_local, mat_id, E, nu, time_scale)

	# 		# Compute exact local analytical actions via JAX
	# 		_, dF_du_local = jax.jvp(local_force_vs_disp, (u_local,), (p_local * time_scale,))
	# 		_, dF_dv_local = jax.jvp(local_force_vs_vel, (v_local,), (p_local,))
			
	# 		# Combine the localized displacement and velocity stiffness updates
	# 		q_local_tet = np.array(dF_du_local + dF_dv_local).ravel() # Length 30 flat array

	# 		# Gather (scatter-add) the local 30 DOFs back into the global tracking vector
	# 		# We map local element node dimensions back to global degrees of freedom
	# 		for local_node_idx, global_node_idx in enumerate(tet):
	# 			g_dof = global_node_idx * 3
	# 			q_stiffness_global[g_dof:g_dof+3] += q_local_tet[local_node_idx*3 : local_node_idx*3+3]

	# 	# 3. Apply Boundary Conditions to the final action matrix output
	# 	q_stiffness_global[self.fixed_dofs] = 0.0
		
	# 	# Return physical structural matrix action mapping: M_diag * p + K * p
	# 	return self.M_diag * p.ravel() + q_stiffness_global


######## VMAP 2

class Substep2JacobianOperatorJAX(LinearOperator):
	def __init__(self, shape, dtype, u_global, v_global, nodes_global, topology, properties, fixed_dofs, dt, gamma, M_diag):
		super().__init__(dtype=dtype, shape=shape)
		self.u_global = jnp.array(u_global).reshape(-1, 3)
		self.v_global = jnp.array(v_global).reshape(-1, 3)
		self.nodes_global = jnp.array(nodes_global)
		self.topology = jnp.array(topology)
		self.properties = jnp.array(properties)
		self.fixed_dofs = fixed_dofs
		self.M_diag = jnp.array(M_diag).ravel()
		self.dt = dt
		self.gamma = gamma

		time_scale = (self.dt * (2.0 - self.gamma)) / 2.0
		self.vmapped_forces = jax.vmap(compute_element_forces_jax, in_axes=(0, 0, 0, 0, 0, 0, None))

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
			return self.vmapped_forces(u_b, v_batched, coords_batched, mat_ids, Es, nus, time_scale)

		def batch_force_vs_vel(v_b):
			return self.vmapped_forces(u_batched, v_b, coords_batched, mat_ids, Es, nus, time_scale)

		_, dF_du_batched = jax.jvp(batch_force_vs_disp, (u_batched,), (p_batched * time_scale,))
		_, dF_dv_batched = jax.jvp(batch_force_vs_vel, (v_batched,), (p_batched,))
		
		q_batched_tet = dF_du_batched + dF_dv_batched

		q_stiffness_global = jnp.zeros((len(self.u_global), 3))
		q_stiffness_global = q_stiffness_global.at[self.topology.ravel()].add(q_batched_tet.reshape(-1, 3))
		q_stiffness_flat = np.array(q_stiffness_global.ravel())

		q_stiffness_flat[self.fixed_dofs] = 0.0
		return np.array(self.M_diag * p.ravel() + q_stiffness_flat)



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

	# Create a small callback to monitor convergence health
	def cg_callback(self, xk):
		# Simply print a dot or iteration notice to watch progress in the console
		print(".", end="", flush=True)



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

		print('!!!!!!!!!! LEN DOF : ', len(fixed_dofs))

		# ... [Establish fixed_dofs and M_diag arrays arrays] ...
		
		# Calculate mass vector per node based on element property distributions
		# Represented here as a lumped mass diagonal vector for performance efficiency
		M_diag = np.ones(dof, dtype=np.float64) * 0.1 
		M_diag[fixed_dofs] = 1.0 # Protect fixed boundary math diagonals

		gamma = 2.0 - np.sqrt(2.0)
		dt1 = gamma * dt
		
		# Initialize your reference state tracking operator at the historical position
		op_t = MatrixFreeTet10Operator(nodes_tet10, topology_tet10, element_properties, x_t - nodes_tet10, v_t, fixed_dofs, dt1, myEquation_dFEM)

		# Exact Force Gathering Check: Read the true force directly from the initial operator state
		f_int_t = op_t.compute_forces_and_action()

		# return 0, 0 # hang debug

		# ==========================================================================
		# SUBSTEP 1: TRAPEZOIDAL RULE STEP (From t to t + gamma*dt)
		# ==========================================================================
		# dt1 = gamma * dt
		###STOCK
		x_gamma = x_t.copy()
		v_gamma = v_t.copy()

		# x_gamma = v_t.copy()
		# v_gamma = x_t.copy()

		# return v_gamma, x_gamma
		# return x_gamma, v_gamma

		# return 0, 0 # hang debug
		# return 0, 0 # hang debug

		print('topology_tet10.shape[1] = ', topology_tet10.shape[1])

		for n_iter in range(max_newton_iter):
			# Re-instantiate the operator at the current trial position coordinates

			op_gamma = MatrixFreeTet10Operator(nodes_tet10, topology_tet10, element_properties, x_gamma - nodes_tet10, v_gamma, fixed_dofs, dt1, myEquation_dFEM)
			
			# Calculate internal forces for this Newton iteration pass
			f_int_gamma = op_gamma.compute_forces_and_action()

			# Calculate your step 1 residual vector mapping mapping ### OLD
			R = M_diag * (v_gamma.ravel() - v_t.ravel()) - (dt1 / 2.0) * (f_int_t + f_int_gamma + 2.0 * F_ext)
			R_pos = x_gamma.ravel() - x_t.ravel() - (dt1 / 2.0) * (v_t.ravel() + v_gamma.ravel())
			R_combined = R + M_diag * (R_pos / dt1)
			R_combined[fixed_dofs] = 0.0

			# CORRECTED SUBSTEP 1 RESIDUAL MECHANICS
			'''
			# 1. Internal forces should add to the inertia to balance external loads correctly
			R = M_diag * (v_gamma.ravel() - v_t.ravel()) - (dt1 / 2.0) * (F_ext + 2.0 * F_ext) + (dt1 / 2.0) * (f_int_t + f_int_gamma)
			# 2. Position continuity constraint mapping
			R_pos = x_gamma.ravel() - x_t.ravel() - (dt1 / 2.0) * (v_t.ravel() + v_gamma.ravel())

			# 3. Combine them ensuring the positional error penalty drives convergence back to origin
			R_combined = R + M_diag * (R_pos / (dt1 / 2.0))

			R_combined[fixed_dofs] = 0.0
			'''

			print('~~~~~~~~~~~~~~ DEBUG START TR ~~~~~~~~~~~~')
			print('R_combined = ', R_combined)
			print('len(R_combined) = ', len(R_combined))
			print('fixed_dofs = ', fixed_dofs)
			print('len(fixed_dofs) = ', len(fixed_dofs))
			print('dof = ', dof)
			print('~~~~~~~~~~~~~~ DEBUG END TR ~~~~~~~~~~~~')

			# return 0, 0

			if np.linalg.norm(R_combined) < tol:
				break

			# Define the Jacobian operator mapping for the Conjugate Gradient solver


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
					F = jnp.eye(3) + disp_grad
					
					# P_stress = self.evaluate_p_stress_jax(F, disp_grad, vel_grad, mat_id, E, nu, dt1)
					P_stress = evaluate_p_stress_jax(F, disp_grad, vel_grad, mat_id, E, nu, dt1)
					f_int_element += (P_stress @ dN_dx * dV).T
					
				return f_int_element	

			def evaluate_p_stress_jax(F_eval, disp_grad_eval, vel_grad_eval, mat_id, E, nu, dt_scale):
				"""Computes First Piola-Kirchhoff stress tensor universally for any phase using JAX."""
				# JAX matrix determinants are natively differentiable
				J_vol = jnp.linalg.det(F_eval)
				
				# --- PHASE A: SOLID TISSUE (Stable Neo-Hookean) ---
				if mat_id == 101.0:
					mu = E / (2.0 * (1.0 + nu))
					lambda_param = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
					alpha = 1.0 + (mu / lambda_param)
					
					# trace and matrix operations map directly to JAX primitives
					stress_scale = mu * (1.0 - (1.0 / (jnp.trace(F_eval.T @ F_eval) + 1.0)))
					
					# Differentiable cofactor formulation using matrix inverses
					# J_vol * F^-T is mathematically identical to your cross-product loop
					F_cofactor = J_vol * jnp.linalg.inv(F_eval).T
					
					return stress_scale * F_eval + (lambda_param * (J_vol - alpha)) * F_cofactor
					
				# --- PHASE B: LIQUID PHASE (Navier-Stokes Continuum) ---
				elif mat_id == 400.0:
					# Map incoming variables to physics parameters
					# E behaves as Bulk Modulus (Kf), nu behaves as Dynamic Viscosity (mu)
					Kf = E
					viscosity_mu = nu
					dt_scale = dt1  # Dynamically receives time_scale from the JAX operator
					
					# 1. Compute Fluid Pressure via Equation of State (EOS)
					pressure = Kf * (J_vol - 1.0)
					
					# 2. Compute Rate-of-Strain Tensor (D) from velocity gradient
					# vel_grad_eval represents L = grad(v)
					D_tensor = 0.5 * (vel_grad_eval + vel_grad_eval.T)
					
					# 3. Compute Divergence of Velocity (Trace of Rate-of-Strain)
					# div_v represents the local volumetric expansion rate of the fluid
					div_v = jnp.trace(D_tensor)
					
					# 4. Apply True Compressible Navier-Stokes Viscous Stress Equation
					# Total Viscous Stress = 2*mu*D + lambda_fluid*trace(D)*Identity
					# According to the Stokes hypothesis, bulk viscosity defaults to: lambda_fluid = -2/3 * mu
					lambda_fluid = -(2.0 / 3.0) * viscosity_mu
					viscous_stress = 2.0 * viscosity_mu * D_tensor + lambda_fluid * div_v * jnp.eye(3)
					
					# 5. Integrate dynamic tangent scaling parameters for JAX AD tracking channels
					# These variables tell JAX how the operator scales when differentiating forces 
					# with respect to velocity/displacement updates across the active sub-interval width.
					mu_stiff = viscosity_mu * dt_scale
					lambda_stiff = Kf * dt_scale
					
					# 6. Return the true First Piola-Kirchhoff Fluid Stress Tensor
					# In a spatial reference frame, P_fluid = J * \sigma * F^-T
					# For a standard Eulerian/weakly-compressible fluid mapping:
					total_cauchy_stress = -pressure * jnp.eye(3) + viscous_stress

					return J_vol * total_cauchy_stress @ jnp.linalg.inv(F_eval).T

				# --- PHASE C: AMBIENT AIR BUFFER MATRIX (Compliant Elastic Solid) ---
				elif mat_id == 202.0:
					mu_stiff = 1e-4
					lambda_stiff = 1e-3
					strain_air = 0.5 * (disp_grad_eval + disp_grad_eval.T)
					return 2.0 * mu_stiff * strain_air + lambda_stiff * jnp.trace(strain_air) * jnp.eye(3)
					
				return jnp.zeros((3, 3))

			# return 0, 0

			########### DEBUG !!!!!!!!!!!!!!
			########### DEBUG !!!!!!!!!!!!!!
			########### DEBUG !!!!!!!!!!!!!!
			# class Substep1JacobianOperatorJAX(splinalg.LinearOperator):
			# 	def __init__(self, shape, dtype, u_global, v_global, nodes_global, topology, properties, fixed_dofs, dt1, M_diag):
			# 		dof_size = int(shape[0])
			# 		# super().__init__(dtype=dtype, shape=shape)
			# 		super().__init__(dtype=dtype, shape=(dof_size, dof_size))
				
			# 		def _matvec(self, p):
			# 			"""Computes J_op @ p globally in a SINGLE hardware pass via jax.vmap and jax.jvp."""
						
			# 			return p.ravel() 


			# J_op = Substep1JacobianOperator((len(R_combined), len(R_combined)), np.float64)
			# J_op = Substep1JacobianOperator((len(dof), len(dof)), np.float64, x_t.ravel(), v_t.ravel(), nodes_tet10)
			# J_op = Substep1JacobianOperator((len(dof), len(dof)), np.float64, x_t.ravel(), v_t.ravel(), nodes_tet10)
			# J_op = Substep1JacobianOperator((len(dof), len(dof)), np.float64, x_next - nodes_tet10, v_next - nodes_tet10, element_properties)

			print('debug trbdf2 step -1')

			num_dofs_int = len(R_combined)
			dof_size = int(R_combined[0])

			print('num_dofs_int = ', num_dofs_int)
			print('dof_size = ', dof_size)

			# if dof_size <= 0:
			# 	raise ValueError(f"Object dof error : <= 0.")
			

			# J_op = Substep1JacobianOperatorJAX
			# J_op = Substep1JacobianOperatorJAX_debug(
			# np.float64, num_dofs_int,
			# # (num_dofs_int, num_dofs_int),

			# )


			J_op = Substep1JacobianOperatorJAX(
			np.float64, #dtype
			num_dofs_int, #shape
			u_global=x_gamma - nodes_tet10,  # Current trial displacement (U = X - X_reference) #uglobal
			v_global=v_gamma,                 # Current trial velocity
			nodes_global=nodes_tet10, 
			topology=topology_tet10, 
			properties=element_properties, 
			fixed_dofs=fixed_dofs, 
			dt1=dt1, 
			M_diag=M_diag
			)



			# J_op = Substep1JacobianOperatorJAX(
			# # shape=len(R_combined),
			# # shape=(len(R_combined), len(R_combined)),
			# # shape=(num_dofs_int, num_dofs_int),
			# dtype=np.float64,
			# shape=num_dofs_int, 
			# u_global=x_gamma - nodes_tet10,  # Current trial displacement (U = X - X_reference)
			# v_global=v_gamma,                 # Current trial velocity
			# nodes_global=nodes_tet10, 
			# topology=topology_tet10, 
			# properties=element_properties, 
			# fixed_dofs=fixed_dofs, 
			# dt1=dt1, 
			# M_diag=M_diag
			# )



			# J_op = Substep1JacobianOperatorJAX
			# # J_op = Substep1JacobianOperatorJAX_debug(
			# # shape=len(R_combined),
			# # shape=(len(R_combined), len(R_combined)),
			# shape=(num_dofs_int, num_dofs_int),
			# # shape = (dof, dof),
			# dtype=np.float64,
			# u_global=x_gamma - nodes_tet10,  # Current trial displacement (U = X - X_reference)
			# v_global=v_gamma,                 # Current trial velocity
			# nodes_global=nodes_tet10, 
			# topology=topology_tet10, 
			# properties=element_properties, 
			# fixed_dofs=fixed_dofs, 
			# dt1=dt1, 
			# M_diag=M_diag
			# )

			# return 0, 0

			print('debug trbdf2 step 0')


			myRtol = 1e-6
			progress_callback = PercentageCallback(tol=tol)

			print('debug trbdf2 step 1')
		
			#####
			## DEBUG
			#####

			# Create a Diagonal (Jacobi) Preconditioner to fix bad row conditioning
			# The system matrix diagonal is roughly M_diag (since Ke is small or scaled)
			M_diag_safe = np.where(M_diag == 0, 1.0, M_diag)
			inv_M = 1.0 / M_diag_safe

			print('debug trbdf2 step 2')


			#ValueError: cannot reshape array of size 1 into shape (11742,)

			# delta_v_flat, info = splinalg.cg(J_op, -R_combined, x0=np.zeros_like(R_combined), rtol=myRtol, maxiter=100, callback=progress_callback) ###
			delta_v_flat, info = splinalg.cg(J_op, -R_combined, x0=np.zeros_like(R_combined), rtol=myRtol, maxiter=100) ###

			print('debug trbdf2 step 3')

			def jacobi_preconditioner(v):
				return inv_M * v

			# Wrap the preconditioner function for SciPy
			# M_precond = splinalg.LinearOperator(shape=J_op.shape, matvec=jacobi_preconditioner)
			
			# CHECK
			# delta_v_flat, info = splinalg.bicgstab(
			# 	J_op, 
			# 	-R_combined, 
			# 	x0=np.zeros_like(R_combined), 
			# 	rtol=myRtol, 
			# 	maxiter=100, 
			# 	M=M_precond, # Activates the diagonal preconditioning channel
			# 	callback=progress_callback
			# )

			# delta_v_flat, info = splinalg.gmres(J_op, -R_combined, restart=30, maxiter=100, callback=progress_callback)

			#time to solve splinalg.bicgstab = 20 sec
			#time to solve splinalg.cg =  21 sec

			print(f"\nCG finished with exit code: {info}")

			if info > 0:
				print("Warning: CG convergence stalled. Matrix may not be perfectly SPD.")

			# continue

			v_gamma += delta_v_flat.reshape(-1, 3)
			x_gamma += (delta_v_flat * (dt1 / 2.0)).reshape(-1, 3)

		return 0, 0

		# return v_gamma, x_gamma ######stock
		# return x_gamma, v_gamma
		# return 0, 0

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
			op_next = MatrixFreeTet10Operator(nodes_tet10, topology_tet10, element_properties, x_next - nodes_tet10, v_next, fixed_dofs, dt1, myEquation_dFEM)
			
			f_int_next = op_next.compute_forces_and_action()
			
			# Calculate step 2 residual mapping
			# R = M * (alpha_bdf * v_next - combined_past_history) - dt2 * (F_int(next) + F_ext)
			# For TR-BDF2 standard configurations, the composite formula evaluates cleanly as:
			v_history_flat = (1.0 / (gamma * (2.0 - gamma))) * v_gamma.ravel() - (((1.0 - gamma)**2) / (gamma * (2.0 - gamma))) * v_t.ravel()
			R = M_diag * (v_next.ravel() - v_history_flat) - (dt * (2.0 - gamma) / 2.0) * (f_int_next + F_ext)
			
			R_pos = x_next.ravel() - (((1.0 - gamma)**2) / (gamma * (2.0 - gamma))) * x_t.ravel() # Historical positions bounds
			
			# R_combined = R + M_diag * (R_pos / dt) ######
			R_combined = R + M_diag * (R_pos / dt2)

			R_combined = R_combined.ravel()
			# R_combined = R_combined[0].ravel() 

			print('~~~~~~~~~~~~~~ DEBUG START BDF ~~~~~~~~~~~~')
			print('R_combined = ', R_combined)
			print('len(R_combined) = ', len(R_combined))
			print('fixed_dofs = ', fixed_dofs)
			print('len(fixed_dofs) = ', len(fixed_dofs))
			print('dof = ', dof)
			print('~~~~~~~~~~~~~~ DEBUG END BDF ~~~~~~~~~~~~')

			R_combined[fixed_dofs] = 0.0
			
			if np.linalg.norm(R_combined) < tol:
				break	

			# def _matvec(self, p):
			# 	p_jax = jnp.array(p).reshape(-1, 3)
			# 	time_scale = (self.dt * (2.0 - self.gamma)) / 2.0  # BDF2 time factor

			# 	# Pass the BDF2 time_scale down so fluids scale to Substep 2 parameters
			# 	def local_force_vs_disp(u_state):
			# 		return compute_element_forces_jax(u_state, v_local, coords_local, mat_id, E, nu, time_scale)
					
			# 	def local_force_vs_vel(v_state):
			# 		return compute_element_forces_jax(u_local, v_state, coords_local, mat_id, E, nu, time_scale)

			# 	# Evaluate exact Substep 2 derivatives using the BDF2 scaling factor
			# 	_, dF_du_local = jax.jvp(local_force_vs_disp, (u_local,), (p_local * time_scale,))
			# 	_, dF_dv_local = jax.jvp(local_force_vs_vel, (v_local,), (p_local,))
				
			# 	q_local_tet = np.array(dF_du_local + dF_dv_local).ravel()

			# '''
			
			# '''
        
			# J_op2 = Substep2JacobianOperator((dof, dof), np.float64)

			J_op2 = Substep2JacobianOperatorJAX(
			# shape=len(R_combined),
			shape=(len(R_combined), len(R_combined)),
			# shape = (dof, dof),
			dtype=np.float64, 
			u_global=x_next - nodes_tet10,   # Trial displacement for end-of-frame
			v_global=v_next,                  # Trial velocity for end-of-frame
			nodes_global=nodes_tet10, 
			topology=topology_tet10, 
			properties=element_properties, 
			fixed_dofs=fixed_dofs, 
			dt=dt, 
			gamma=gamma, 
			M_diag=M_diag
			)

			# delta_v_flat, _ = splinalg.cg(J_op2, -R_combined, rtol=1e-6, callback=progress_callback)
			delta_v_flat, info = splinalg.cg(J_op2, -R_combined, rtol=1e-6, callback=progress_callback)

			# delta_v_flat, info = splinalg.cg(J_op2, -R_combined, x0=np.zeros_like(R_combined), rtol=myRtol, maxiter=100, callback=progress_callback) ###

			# M_precond2 = splinalg.LinearOperator(shape=J_op2.shape, matvec=jacobi_preconditioner)

			# delta_v_flat, info = splinalg.bicgstab(
			# 	J_op2, 
			# 	-R_combined, 
			# 	x0=np.zeros_like(R_combined), 
			# 	rtol=myRtol, 
			# 	maxiter=100, 
			# 	M=M_precond2, # Activates the diagonal preconditioning channel
			# 	callback=progress_callback
			# )

			# delta_v_flat, info = splinalg.gmres(J_op2, -R_combined, restart=30, maxiter=100, callback=progress_callback)

			print(f"\nCG 2 finished with exit code: {info}")

			v_next += delta_v_flat.reshape(-1, 3)
			x_next += (delta_v_flat * (dt * (2.0 - gamma) / 2.0)).reshape(-1, 3)

		# return 0, 0 # hang debug
		return x_next, v_next ###########
		# return v_next, x_next

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

		#start debug here 9/5 !!!!!!!!!

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
		startTime = datetime.now()

		#look @ execute_production_fem_bake_with_skin

		# nodes_tet4, topology_tet4, tags, vdb_sphere_sdf_l, vdb_sphere_sdf_h, vdb_box_sdf_l, vdb_box_sdf_h = self.generate_global_multiphase_mesh(24, 96)
		# nodes_tet4, topology_tet4, tags, vdb_sphere_sdf_l, vdb_sphere_sdf_h, vdb_box_sdf_l, vdb_box_sdf_h = self.generate_global_multiphase_mesh(16, 16)
		nodes_tet4, topology_tet4, tags, vdb_sphere_sdf_l, vdb_sphere_sdf_h, vdb_box_sdf_l, vdb_box_sdf_h = self.generate_global_multiphase_mesh(8, 16)

		self.visualize_global_multiphase_slice(nodes_tet4, topology_tet4, tags, slice_axis=0, slice_val=0.0) ###########
		# self.visualize_global_multiphase_slice(nodes_tet4, topology_tet4, tags, slice_axis=0, slice_val=1.0) ###########

		# return

		# layer_data_s_l = [("sdf_joined", vdb_sphere_sdf_l)]
		# myBox_l = self.sdf_vdb_visualizer(layer_data_s_l)

		# layer_data_b_l = [("sdf_joined", vdb_box_sdf_l)]
		# mySphere_l = self.sdf_vdb_visualizer(layer_data_b_l)


		layer_data_s_h = [("sdf_joined", vdb_sphere_sdf_h)]
		mySphere_h = self.sdf_vdb_visualizer(layer_data_s_h)

		# return

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

		# total_frames = 3
		total_frames = 1
		frame_dt=0.01 ########
		# frame_dt=.5
		# frame_dt=1

		# Initialize simulation states
		x_current = nodes_tet10.copy()
		v_current = np.zeros((len(nodes_tet10), 3), dtype=np.float64)

		# F_ext = np.zeros(len(nodes_tet10)*3, dtype=np.float64)
		# F_ext = np.array([0, -9.81, 0])
		# gravity = np.array([0, -9.81, 0]) ##########
		gravity = np.array([0, 0, 0])

		num_nodes = len(nodes_tet10) # Assuming v_t is shape (num_nodes, 3)

		# Tile the [0, -9.81, 0] gravity force across all nodes and flatten it
		# If F_ext is already a per-node force density (like force per unit mass), multiply by mass:
		F_ext = np.tile(gravity, num_nodes) 

		if not mesh.shape_keys:
			obj.shape_key_add(name="Basis")

		# --- PHASE 2: THE TR-BDF2 TIME STRIDE LOOP ---
		# for frame in range(0, total_frames + 1):
		# for frame in range(2, total_frames + 1):
		for frame in range(1, total_frames + 1): ######
			print('~~~~~~~~~~~~~~~~~~~~~~~ FRAME = ', frame)
			bpy.context.scene.frame_set(frame)
			x_next, v_next = self.run_tr_bdf2_time_step(myEquation_dFEM,
				nodes_tet10, topology_tet10, tags, 
				x_current, v_current, F_ext, frame_dt)

			# print('frame = ', frame)
			# print('x_next = ', x_next)
			# print('v_next = ', v_next)

			continue
			
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
			# sk.value = 0.0
			sk.keyframe_insert(data_path="value", frame=frame)
			sk.value = 0.0
			sk.keyframe_insert(data_path="value", frame=frame + 1)

		totalTime = datetime.now() - startTime
		print('totalTime = ', totalTime)

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
	# def __init__(self, nodes_tet10, topology_tet10, element_properties, current_displacements, fixed_dofs, dt1, myEquation_dFEM):
	def __init__(self, nodes_tet10, topology_tet10, element_properties, current_displacements, current_veolocity, fixed_dofs, dt1, myEquation_dFEM):
		self.nodes = nodes_tet10
		self.topology = topology_tet10
		self.properties = element_properties
		self.current_U = current_displacements
		self.current_V = current_veolocity
		self.fixed_dofs = fixed_dofs
		self.dof = len(nodes_tet10) * 3
		self.shape = (self.dof, self.dof)
		self.dtype = np.float64
		self.dt1 = dt1
		self.myEquation_dFEM_usable = myEquation_dFEM

	def compute_element_forces_jax(self, element_displacements, element_velocities, element_node_coords, mat_id, E, nu, dt1):
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
			F = jnp.eye(3) + disp_grad
			
			P_stress = self.evaluate_p_stress_jax(F, disp_grad, vel_grad, mat_id, E, nu, dt1)
			f_int_element += (P_stress @ dN_dx * dV).T
			
		return f_int_element		

	def evaluate_p_stress_jax(self, F_eval, disp_grad_eval, vel_grad_eval, mat_id, E, nu, dt_scale):
		"""Computes First Piola-Kirchhoff stress tensor universally for any phase using JAX."""
		# JAX matrix determinants are natively differentiable
		J_vol = jnp.linalg.det(F_eval)
		
		# --- PHASE A: SOLID TISSUE (Stable Neo-Hookean) ---
		if mat_id == 101.0:
			mu = E / (2.0 * (1.0 + nu))
			lambda_param = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
			alpha = 1.0 + (mu / lambda_param)
			
			# trace and matrix operations map directly to JAX primitives
			stress_scale = mu * (1.0 - (1.0 / (jnp.trace(F_eval.T @ F_eval) + 1.0)))
			
			# Differentiable cofactor formulation using matrix inverses
			# J_vol * F^-T is mathematically identical to your cross-product loop
			F_cofactor = J_vol * jnp.linalg.inv(F_eval).T
			
			return stress_scale * F_eval + (lambda_param * (J_vol - alpha)) * F_cofactor
			
		# --- PHASE B: LIQUID PHASE (Navier-Stokes Continuum) ---
		elif mat_id == 400.0:
			# Map incoming variables to physics parameters
			# E behaves as Bulk Modulus (Kf), nu behaves as Dynamic Viscosity (mu)
			Kf = E
			viscosity_mu = nu
			dt_scale = dt1  # Dynamically receives time_scale from the JAX operator
			
			# 1. Compute Fluid Pressure via Equation of State (EOS)
			pressure = Kf * (J_vol - 1.0)
			
			# 2. Compute Rate-of-Strain Tensor (D) from velocity gradient
			# vel_grad_eval represents L = grad(v)
			D_tensor = 0.5 * (vel_grad_eval + vel_grad_eval.T)
			
			# 3. Compute Divergence of Velocity (Trace of Rate-of-Strain)
			# div_v represents the local volumetric expansion rate of the fluid
			div_v = jnp.trace(D_tensor)
			
			# 4. Apply True Compressible Navier-Stokes Viscous Stress Equation
			# Total Viscous Stress = 2*mu*D + lambda_fluid*trace(D)*Identity
			# According to the Stokes hypothesis, bulk viscosity defaults to: lambda_fluid = -2/3 * mu
			lambda_fluid = -(2.0 / 3.0) * viscosity_mu
			viscous_stress = 2.0 * viscosity_mu * D_tensor + lambda_fluid * div_v * jnp.eye(3)
			
			# 5. Integrate dynamic tangent scaling parameters for JAX AD tracking channels
			# These variables tell JAX how the operator scales when differentiating forces 
			# with respect to velocity/displacement updates across the active sub-interval width.
			mu_stiff = viscosity_mu * dt_scale
			lambda_stiff = Kf * dt_scale
			
			# 6. Return the true First Piola-Kirchhoff Fluid Stress Tensor
			# In a spatial reference frame, P_fluid = J * \sigma * F^-T
			# For a standard Eulerian/weakly-compressible fluid mapping:
			total_cauchy_stress = -pressure * jnp.eye(3) + viscous_stress

			return J_vol * total_cauchy_stress @ jnp.linalg.inv(F_eval).T

		# elif mat_id == 400.0:
		# 	Kf = E
		# 	viscosity = nu
		# 	pressure = Kf * (J_vol - 1.0)
		# 	D_tensor = 0.5 * (vel_grad_eval + vel_grad_eval.T)

		# 	mu_stiff = viscosity * dt1
		# 	lambda_stiff = Kf * dt1

		# 	# return -pressure * jnp.eye(3) + 2.0 * viscosity * D_tensor
		# 	return -pressure * jnp.eye(3) + 2.0 * viscosity * D_tensor
			
		# --- PHASE C: AMBIENT AIR BUFFER MATRIX (Compliant Elastic Solid) ---
		elif mat_id == 202.0:
			mu_stiff = 1e-4
			lambda_stiff = 1e-3
			strain_air = 0.5 * (disp_grad_eval + disp_grad_eval.T)
			return 2.0 * mu_stiff * strain_air + lambda_stiff * jnp.trace(strain_air) * jnp.eye(3)
			
		return jnp.zeros((3, 3))

	def compute_forces_and_action(self):
		"""
		Computes and assembles the global internal force vector.
		The 'p_vector' parameter is retained only for backwards compatibility 
		with your existing pipeline calls, but it is no longer used.
		"""
		# Initialize the global 1D tracking array (e.g., size 11742)
		f_int_global = np.zeros(self.dof, dtype=np.float64)
		
		# ONE UNIFIED LOOP FOR ALL CONTINUUM PHYSICS (Base Force Assembly Only)
		for t_idx, tet in enumerate(self.topology):
			if self.properties[t_idx] == 0:    # Air
				mat_id, E, nu = 202.0, 1.0, 0.001
			elif self.properties[t_idx] == 1:  # Solid Sphere
				mat_id, E, nu = 101.0, 10.0, 0.45
			else:
				continue

			# Call the new JAX-native force calculator 
			# It only returns one item: f_local (the 10x3 internal forces)
			f_local = self.compute_element_forces_jax(
				element_displacements=jnp.array(self.current_U[tet]),
				element_velocities=jnp.array(self.current_V[tet]), # Assuming tracked on self
				element_node_coords=jnp.array(self.nodes[tet]),
				mat_id=mat_id,
				E=E,
				nu=nu,
				dt1=self.dt1
			)


			# f_local = compute_element_forces_jax(
			# 	element_displacements=jnp.array(self.current_U[tet]),
			# 	element_velocities=jnp.array(self.p_velocity_nodes[tet]), # Assuming tracked on self
			# 	element_node_coords=jnp.array(self.nodes[tet]),
			# 	mat_id=mat_id,
			# 	E=E,
			# 	nu=nu,
			# 	dt1=self.dt1
			# )
			
			# Convert JAX array back to NumPy for standard scatter processing
			f_local_np = np.array(f_local)

			# --- HIGH-ORDER SCATTER PASS (STILL ESSENTIAL FOR FORCES) ---
			for local_idx in range(10):
				global_node_idx = tet[local_idx]
				start = global_node_idx * 3
				
				# Assembly of structural elements into the unified global state
				f_int_global[start : start + 3] += f_local_np[local_idx]

		# Enforce fixed boundary conditions on the force residual mapping
		f_int_global[self.fixed_dofs] = 0.0

		return f_int_global.ravel()

	def _matvec(self, p):
		# Mandatory SciPy callback. Evaluates strictly the action product channels
		y_action = self.compute_forces_and_action()
		return y_action