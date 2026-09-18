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
from jax.tree_util import Partial

# Force JAX to use 64-bit double precision to maintain mechanical engineering accuracy
jax.config.update("jax_enable_x64", True)

bpy.utils.expose_bundled_modules()
import openvdb as vdb


# ==============================================================================
# 3. POSITION-INDEPENDENT HIGH-STIFFNESS FORCE ENGINE
# ==============================================================================
# def compute_forces_jax(pos, mu, lam, STATIC_INIT_POS, STATIC_CENTER, STATIC_NORMALS, STATIC_THICKNESS, collision_radius=0.28, collision_stiffness=1500.0, floor_z=2.0, floor_stiffness=35000.0):
# def compute_forces_jax(pos, mu, lam, STATIC_INIT_POS, STATIC_CENTER, STATIC_NORMALS, STATIC_THICKNESS, collision_radius=0.28, collision_stiffness=1500.0, floor_z=2.0, floor_stiffness=35000.0):
def compute_forces_jax(pos, mu, lam, STATIC_INIT_POS, STATIC_CENTER, STATIC_NORMALS, STATIC_THICKNESS, collision_radius=0.28, collision_stiffness=1500.0, floor_z=2.05, floor_stiffness=35000.0):
	pos_2d = jnp.reshape(pos, (-1, 3))
	num_p = pos_2d.shape

	# FIXED: True Relative Transformation Framing completely removes translation drift.
	# By isolating deformation purely relative to the ACTIVE center of mass, 
	# uniform freefall acceleration generates EXACTLY zero strain forces.
	active_center = jnp.mean(pos_2d, axis=0)
	disp_local = (pos_2d - active_center) - (STATIC_INIT_POS - STATIC_CENTER)

	#Smith stable 2018
	# Vectorized batch outer product broadcasting handles local strain gradient mapping
	F = jnp.eye(3)[None, :, :] + (disp_local[:, :, None] * STATIC_NORMALS[:, None, :]) / 1.8

	J = vmap_det(F)
	J_stable = jnp.maximum(J, 0.20)
	I_C = vmap_trace(F)
	F_inv_t = vmap_inv_t(F)

	alpha = 1.0 + (mu / lam)
	term1 = (mu * (1.0 - 1.0 / (I_C + 1.0)))[:, None, None] * F
	term2 = (lam * (J_stable - alpha))[:, None, None] * F_inv_t
	P = term1 + term2

	continuum_forces = -jnp.matmul(P, STATIC_NORMALS[..., None]).squeeze(-1)

	# Pairwise self-collision avoidance bubbles
	diff = pos_2d[:, None, :] - pos_2d[None, :, :]  
	dists = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8) 
	overlap = jnp.maximum(collision_radius - dists, 0.0)
	col_normals = diff / dists[..., None]

	repulsion_mag = (overlap ** 2) * collision_stiffness
	# repulsion_mag = repulsion_mag * (jnp.eye(num_p) == 0)
	repulsion_mag = repulsion_mag * (jnp.eye(num_p[0]) == 0)
	
	self_collision_forces = jnp.sum(col_normals * repulsion_mag[..., None], axis=1) * 0.02

	# Smooth potential floor check against the dynamic lowest extreme bounding node
	lowest_vertex_z = jnp.min(pos_2d[:, 2])
	floor_penetration = jnp.maximum((floor_z + 0.02) - lowest_vertex_z, 0.0)
	floor_push_z = (floor_penetration ** 2) * floor_stiffness

	# Apply floor forces cleanly to the active bottom hemisphere layer vertices
	floor_forces_mask = jnp.where(pos_2d[:, 2] < active_center[2], floor_push_z, 0.0)
	floor_forces = jnp.zeros_like(pos_2d).at[:, 2].set(floor_forces_mask)

	total_forces = continuum_forces + self_collision_forces + floor_forces
	return jnp.reshape(total_forces, pos.shape)

def jax_physics_step(state, step_idx, mu, lam, damping, dt, restitution, object_mass, drag_coefficient, STATIC_INIT_POS, STATIC_CENTER, STATIC_NORMALS, STATIC_THICKNESS):
	pos, vel = state
	gravity = -9.81
	gamma = 2.0 - jnp.sqrt(2.0)
	dt1 = gamma * dt

	# --- FIXED: VECTORIZED AERODYNAMIC DRAG ACCELERATION ---
	# Velocity-dependent drag vector directly maps drag as deceleration: a_drag = - (b/m) * v * |v|
	v_mags = jnp.linalg.norm(vel, axis=1, keepdims=True)
	drag_forces = - (drag_coefficient / object_mass) * vel * v_mags

	gravity_forces1 = jnp.zeros_like(vel).at[:, 2].set(gravity)
	total_accel1 = gravity_forces1 + drag_forces
	vel_est1 = vel + total_accel1 * dt1

	f1 = compute_forces_jax(pos, mu, lam, STATIC_INIT_POS, STATIC_CENTER, STATIC_NORMALS, STATIC_THICKNESS)
	v_mid = vel_est1 + f1 * dt1

	pos_est = pos + v_mid * dt1
	f2 = compute_forces_jax(pos_est, mu, lam, STATIC_INIT_POS, STATIC_CENTER, STATIC_NORMALS, STATIC_THICKNESS)

	c_mid = 1.0 / (gamma * (2.0 - gamma))
	c_cur = ((1.0 - gamma) ** 2) / (gamma * (2.0 - gamma))

	# Re-inject gravity and aerodynamic drag into Stage 2 TR-BDF2 recovery step
	gravity_forces2 = jnp.zeros_like(vel).at[:, 2].set(gravity)
	total_accel2 = gravity_forces2 + drag_forces

	new_vel = (c_mid * v_mid) - (c_cur * vel) + (f2 * dt * (1.0 - gamma) / (2.0 - gamma)) + total_accel2 * dt
	new_vel *= damping
	next_pos = pos + new_vel * dt

	min_allowed_z = 2.0 + STATIC_THICKNESS
	below_floor = next_pos[:, 2] <= min_allowed_z
	clamped_z = jnp.where(below_floor, min_allowed_z, next_pos[:, 2])
	next_pos = next_pos.at[:, 2].set(clamped_z)

	v_z_reflected = jnp.where(below_floor & (new_vel[:, 2] < 0), -new_vel[:, 2] * restitution, new_vel[:, 2])
	lost_momentum_magnitude = jnp.where(below_floor & (new_vel[:, 2] < 0), jnp.abs(new_vel[:, 2]) * (1.0 - restitution), 0.0)

	center_xy = jnp.mean(next_pos[:, :2], axis=0)
	dir_xy = next_pos[:, :2] - center_xy
	out_dir = dir_xy / jnp.maximum(jnp.linalg.norm(dir_xy, axis=1, keepdims=True), 1e-4)

	new_vel_x = new_vel[:, 0] + out_dir[:, 0] * lost_momentum_magnitude * 0.85
	new_vel_y = new_vel[:, 1] + out_dir[:, 1] * lost_momentum_magnitude * 0.85

	next_vel = new_vel.at[:, 0].set(new_vel_x)
	next_vel = next_vel.at[:, 1].set(new_vel_y)
	next_vel = next_vel.at[:, 2].set(v_z_reflected)

	return (next_pos, next_vel), next_pos

def loss_function(mu, lam, damping, dt, initial_spike_vel, restitution, object_mass, drag_coefficient, STATIC_INIT_POS, STATIC_CENTER, STATIC_NORMALS, STATIC_THICKNESS, target_frame_idx, target_height):
	"""Loss function tracking separate individual scalars."""
	trajectory = run_simulation_scan(mu, lam, damping, dt, initial_spike_vel, restitution, object_mass, drag_coefficient, STATIC_INIT_POS, STATIC_CENTER, STATIC_NORMALS, STATIC_THICKNESS, num_frames=300)
	frame_positions = trajectory[target_frame_idx]
	
	max_z = jnp.max(frame_positions[:, 2])
	min_z = jnp.min(frame_positions[:, 2])
	bounding_box_center_z = (max_z + min_z) / 2.0
	
	raw_loss = (bounding_box_center_z - target_height) ** 2
	return jnp.log(1.0 + raw_loss)



def run_simulation_scan(mu, lam, damping, dt, initial_spike_vel, restitution, object_mass, drag_coefficient, STATIC_INIT_POS, STATIC_CENTER, STATIC_NORMALS, STATIC_THICKNESS, num_frames=300):
	"""FIXED: Parameters are passed as separate scalar arguments to completely prevent index bleeding."""
	init_vel = jnp.zeros_like(STATIC_INIT_POS).at[:, 2].set(initial_spike_vel)

	def step_fn(state, x):
		return jax_physics_step(state, x, mu, lam, damping, dt, restitution, object_mass, drag_coefficient, STATIC_INIT_POS, STATIC_CENTER, STATIC_NORMALS, STATIC_THICKNESS)
		
	_, trajectory = jax.lax.scan(step_fn, (STATIC_INIT_POS, init_vel), None, length=num_frames)
	return trajectory


# Compiled engine tracks separate inputs seamlessly
jit_simulation_engine = jax.jit(run_simulation_scan, static_argnums=(12,))

vmap_det = jax.vmap(jnp.linalg.det)
vmap_trace = jax.vmap(lambda mat: jnp.trace(jnp.dot(mat.T, mat)))
vmap_inv_t = jax.vmap(lambda mat: jnp.linalg.inv(mat + 1e-5 * jnp.eye(3)).T)


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

		# Spawn target sphere at Z = 7.5
		bpy.ops.mesh.primitive_uv_sphere_add(radius=1.8, location=(0.0, 0.0, 7.5), segments=SPHERE_RES, ring_count=SPHERE_RES)
		# bpy.ops.mesh.primitive_cube_add(location=(0.0, 0.0, 7.5), size=6)

		ball_obj = bpy.context.active_object
		ball_obj.name = "Rubber_Ball"



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



		'''
		############################
		#######BAKE GABOR TO PTS
		############################


		# 1. Setup target object 
		# bpy.ops.mesh.primitive_grid_add(x_subdivisions=100, y_subdivisions=100, size=2)
		# ball_obj = bpy.context.active_object

		# 2. Add the modifier slot and link a clean GeometryNodeTree
		gn_mod = ball_obj.modifiers.new(name="ProceduralDisplacement", type='NODES')
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

		print(f"Success! Mesh baked. {len(ball_obj.data.vertices)} points available for your BPY loop.")


		'''


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

		gabor_node = nodes.new(type='ShaderNodeTexGabor')
		displacement_node = nodes.new(type='ShaderNodeDisplacement')

		mat.node_tree.links.new(gabor_node.outputs['Value'], displacement_node.inputs['Height'])
		mat.node_tree.links.new(displacement_node.outputs['Displacement'], output_node.inputs['Displacement'])
		# displacement_node.inputs[2].default_value = 0.3
		displacement_node.inputs[2].default_value = 0.5

		abj_sd_b_instance.autoArrangeNodes(mat.node_tree)

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


		num_frames = 600
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
		DAMPING_VAL = 0.98
		# DT_VAL = 0.01
		# DT_VAL = 0.008
		DT_VAL = 0.008
		INITIAL_SPIKE_VELOCITY = -30
		# INITIAL_SPIKE_VELOCITY = -50
		# RESTITUTION_VAL = .87
		RESTITUTION_VAL = .7
		OBJECT_MASS_VAL = 300
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
		num_steps = 5


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

		bpy.context.scene.render.fps = 240
		bpy.context.scene.frame_end = num_frames

	def testVDB_06(self, abj_sd_b_instance):
		startTime = datetime.now()

		self.bounce_diff_03(abj_sd_b_instance)

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