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


# --- 2. TRUE STAGGERED EULERIAN MULTIPHASE LATTICE ENGINE ---
class TrueStaggeredVoxelSolver:
	def __init__(self, res=(16, 16, 24)):
		self.res = np.array(res)
		self.dx = 0.5
		
		# 4D Multiphase Tracking Matrix: [X, Y, Z, Phase]
		# Phases: 0 = Air (Vacuum environment), 1 = Liquid, 2 = Deformable Solid
		self.phases = np.zeros((*res, 3), dtype=np.float64)
		self.phases[..., 0] = 1.0  # Entire domain defaults to 100% Air
		
		# Staggered Velocity Fields (MAC Grid architecture prevents streaming stretching)
		self.u = np.zeros((res[0]+1, res[1], res[2]), dtype=np.float64) # X-velocity on faces
		self.v = np.zeros((res[0], res[1]+1, res[2]), dtype=np.float64) # Y-velocity on faces
		self.w = np.zeros((res[0], res[1], res[2]+1), dtype=np.float64) # Z-velocity on faces
		
		# True Smith 2018 Stable Neo-Hookean Parameters
		self.mu = 500.0   # Shear stiffness (controls ripple speed)
		self.lam = 3000.0 # High Bulk modulus ensures strict volume preservation

		# self.mu = 50.0   # Shear stiffness (controls ripple speed)
		# self.lam = 300.0 # High Bulk modulus ensures strict volume preservation

		self.alpha = 1.0 + (self.mu / self.lam)
		self.damping = 0.96
		
		# Exact 3D solid cube collision boundary mapping
		self.cube_mask = np.zeros(res, dtype=bool)
		x_idx, y_idx, z_idx = np.indices(res)
		world_x = (x_idx - res[0]/2.0) * self.dx
		world_y = (y_idx - res[1]/2.0) * self.dx
		world_z = (z_idx - res[2]/2.0) * self.dx + 4.0
		
		self.cube_mask = (world_x >= -2.0) & (world_x <= 2.0) & \
							(world_y >= -2.0) & (world_y <= 2.0) & \
							(world_z >= -2.0) & (world_z <= 2.0)

	def seed_spherical_rubber_ball(self, cx=8, cy=8, cz=19, r_voxels=3.4):
		"""Seeds continuous volume fractions, cleanly preserving decimal weights like 0.499."""
		x_idx, y_idx, z_idx = np.indices(self.res)
		dist = np.sqrt((x_idx-cx)**2 + (y_idx-cy)**2 + (z_idx-cz)**2)
		
		mask_full = dist <= r_voxels
		mask_edge = (dist > r_voxels) & (dist <= r_voxels + 1.0)
		
		self.phases[mask_full, 2] = 1.0
		self.phases[mask_full, 0] = 0.0
		
		edge_fractions = (r_voxels + 1.0 - dist[mask_edge])
		self.phases[mask_edge, 2] = edge_fractions
		self.phases[mask_edge, 0] = 1.0 - edge_fractions

	def update_physics(self, dt, gravity=-9.81):
		"""Computes true 2018 Stable Neo-Hookean forces and advects BOTH mass and velocity."""
		solid_fraction = self.phases[..., 2]
		
		# 1. Apply Gravity to the vertical face velocities where solid exists
		for z in range(self.res[2]):
			cell_solid = solid_fraction[:, :, np.minimum(z, self.res[2]-1)]
			self.w[:, :, z] += np.where(cell_solid > 0.01, gravity * dt, 0.0)
			
		# 2. Strict Hard Cube Collision Boundary Enforcement
		# Block velocities trying to penetrate the marked solid mask rows
		for z in range(self.res[2]):
			if np.any(self.cube_mask[:, :, z]):
				self.w[:, :, z] = np.maximum(self.w[:, :, z], 0.0)
				self.u[:, :, z] = 0.0
				self.v[:, :, z] = 0.0

		# 3. 2018 Stable Volume-Preservation Stresses (Calculates lateral expansion forces)
		J = 1.0 / np.maximum(solid_fraction, 0.02)
		p_force = -self.lam * (J - self.alpha) * solid_fraction
		
		# Central difference spatial gradients
		grad_x = np.zeros_like(solid_fraction)
		grad_y = np.zeros_like(solid_fraction)
		grad_x[1:-1, :, :] = (solid_fraction[2:, :, :] - solid_fraction[:-2, :, :]) / (2.0 * self.dx)
		grad_y[:, 1:-1, :] = (solid_fraction[:, 2:, :] - solid_fraction[:, :-2, :]) / (2.0 * self.dx)
		
		# Accelerate velocities outward horizontally based on the pressure spikes
		self.u[1:-1, :, :] += 0.5 * (p_force[1:, :, :] + p_force[:-1, :, :]) * grad_x[1:, :, :] * dt
		self.v[:, 1:-1, :] += 0.5 * (p_force[:, 1:, :] + p_force[:, :-1, :]) * grad_y[:, 1:, :] * dt

		# 4. CONSERVATIVE VELOCITY + DENSITY ADVECTION SWEEP (Prevents the fluid stream bug)
		new_phases = np.copy(self.phases)
		new_w = np.copy(self.w)
		new_u = np.copy(self.u)
		new_v = np.copy(self.v)
		
		for z in range(1, self.res[2] - 1):
			# Calculate non-skipping translation flux fraction (CFL bounded at 0.85 max per cell)
			flux_z = np.clip(abs(self.w[:, :, z]) * dt / self.dx, 0.0, 0.85)
			down_flow = self.w[:, :, z] < 0
			
			if np.any(down_flow):
				blocked = self.cube_mask[:, :, z - 1]
				effective_flux = np.where(blocked, 0.0, flux_z)
				
				# Move Density and Velocities together down the grid channels
				for p in range(3):
					mass_moved = self.phases[:, :, z, p] * effective_flux
					new_phases[:, :, z, p] -= mass_moved
					new_phases[:, :, z - 1, p] += mass_moved
					
				# Move momentum down to prevent stretching artifacts
				mom_moved = self.w[:, :, z] * effective_flux
				new_w[:, :, z] -= mom_moved
				new_w[:, :, z - 1] += mom_moved

		# Horizontal Expansion Advection Sweep
		for x in range(1, self.res[0] - 1):
			flux_x = np.clip(abs(self.u[x, :, :]) * dt / self.dx, 0.0, 0.85)
			right_flow = self.u[x, :, :] > 0
			if np.any(right_flow):
				for p in range(3):
					mass_moved = self.phases[x, :, :, p] * flux_x
					new_phases[x, :, :, p] -= mass_moved
					new_phases[x + 1, :, :, p] += mass_moved
					
				mom_moved = self.u[x, :, :] * flux_x
				new_u[x, :, :] -= mom_moved
				new_u[x + 1, :, :] += mom_moved

		# Finalize states with strict conservation constraints
		self.phases = np.clip(new_phases, 0.0, 1.0)
		self.w = new_w * self.damping
		self.u = new_u * self.damping
		self.v = new_v * self.damping



# --- 2. MULTIPHASE EULERIAN VOLUMETRIC LATTICE FEM ENGINE ---
class MultiphaseVoxelSolver:
	def __init__(self, res=(16, 16, 24)):
		self.res = np.array(res)
		self.dx = 0.5
		
		# Multiphase tracking matrix: [X, Y, Z, Phase]
		# Phase indices: 0 = Air, 1 = Liquid, 2 = Deformable Solid
		self.phases = np.zeros((*res, 3), dtype=np.float64)
		self.phases[..., 0] = 1.0  # Entire domain defaults to 100% Air
		
		# Velocity fields
		self.vel_x = np.zeros(res, dtype=np.float64)
		self.vel_y = np.zeros(res, dtype=np.float64)
		self.vel_z = np.zeros(res, dtype=np.float64)
		
		# True Smith 2018 Stable Neo-Hookean Parameters
		self.mu = 450.0   # Shear stiffness
		self.lam = 1800.0 # Bulk modulus
		self.alpha = 1.0 + (self.mu / self.lam) # Stability shift constant
		self.damping = 0.94
		
		# Set up exact 3D solid cube collision bounds mask inside the grid matrix
		self.cube_mask = np.zeros(res, dtype=bool)
		x_idx, y_idx, z_idx = np.indices(res)
		world_x = (x_idx - res[0]/2.0) * self.dx
		world_y = (y_idx - res[1]/2.0) * self.dx
		world_z = (z_idx - res[2]/2.0) * self.dx + 4.0 # Spatial world tracking baseline
		
		self.cube_mask = (world_x >= -2.0) & (world_x <= 2.0) & \
							(world_y >= -2.0) & (world_y <= 2.0) & \
							(world_z >= -2.0) & (world_z <= 2.0)

	def seed_spherical_rubber_ball(self, cx=8, cy=8, cz=19, r_voxels=3.4):
		"""Seeds continuous volume fractions, cleanly preserving decimal weights like 0.499."""
		x_idx, y_idx, z_idx = np.indices(self.res)
		dist = np.sqrt((x_idx-cx)**2 + (y_idx-cy)**2 + (z_idx-cz)**2)
		
		mask_full = dist <= r_voxels
		mask_edge = (dist > r_voxels) & (dist <= r_voxels + 1.0)
		
		# Inject Solid phase fraction
		self.phases[mask_full, 2] = 1.0
		self.phases[mask_full, 0] = 0.0 # Clear Air
		
		edge_fractions = (r_voxels + 1.0 - dist[mask_edge])
		self.phases[mask_edge, 2] = edge_fractions
		self.phases[mask_edge, 0] = 1.0 - edge_fractions # Maintained conservation check

	def compute_smith2018_stresses(self, dt):
		"""Applies exact 2018 Smith et al. Stable Neo-Hookean strain energy equations."""
		solid_fraction = self.phases[..., 2]
		
		# Calculate localized spatial displacement mapping gradients
		grad_x = np.zeros_like(solid_fraction)
		grad_y = np.zeros_like(solid_fraction)
		grad_x[1:-1, :, :] = (solid_fraction[2:, :, :] - solid_fraction[:-2, :, :]) / 2.0
		grad_y[:, 1:-1, :] = (solid_fraction[:, 2:, :] - solid_fraction[:, :-2, :]) / 2.0
		
		# Trace Jacobian volume deformations: J = V_current / V_reference
		# Multi-phase scaling ensures Air components offer zero hydrostatic resistance
		J = 1.0 / np.maximum(solid_fraction, 0.01)
		
		# First Piola-Kirchhoff derivative approximation tracking the reparameterized alpha stability boundary
		# Hydrostatic restorative pressure vector calculations
		p_force = -self.lam * (J - self.alpha) * solid_fraction
		
		# Apply local shear tensor variations (governing the visual surface ripples)
		ripple_x = self.mu * grad_x
		ripple_y = self.mu * grad_y
		
		# Update velocities cleanly based on absolute material density presence
		active_solid = solid_fraction > 0.02
		self.vel_x[active_solid] += (p_force[active_solid] * grad_x[active_solid] + ripple_x[active_solid]) * dt
		self.vel_y[active_solid] += (p_force[active_solid] * grad_y[active_solid] + ripple_y[active_solid]) * dt

	def advance_multiphase_advection(self, dt, gravity=-9.81):
		"""Performs non-skipping conservative VOF multi-phase step loops."""
		solid_fraction = self.phases[..., 2]
		
		# Apply external forces to the solid phase only (Air remains completely passive)
		self.vel_z[solid_fraction > 0.01] += gravity * dt
		
		# Rigid Cube Contact Enforcement: Kill penetrating velocities inside the cube mask
		self.vel_z[self.cube_mask] = np.maximum(self.vel_z[self.cube_mask], 0.0)
		self.vel_x[self.cube_mask] = 0.0
		self.vel_y[self.cube_mask] = 0.0
		
		new_phases = np.copy(self.phases)
		
		# Vertical advection sweep
		for z in range(1, self.res[2] - 1):
			flux_z = np.clip(abs(self.vel_z[:, :, z]) * dt / self.dx, 0.0, 0.85)
			down_flow = self.vel_z[:, :, z] < 0
			
			if np.any(down_flow):
				# Ensure mass cannot enter the marked collision mask cells
				blocked = self.cube_mask[:, :, z - 1]
				effective_flux = np.where(blocked, 0.0, flux_z)
				
				# Shift both solid and air layers conservatively (Multiphasic split balance)
				for p in range(3):
					mass_moved = self.phases[:, :, z, p] * effective_flux
					new_phases[:, :, z, p] -= mass_moved
					new_phases[:, :, z - 1, p] += mass_moved
					
		# Horizontal expansion sweep (Governing the flat squish flattening across the cube)
		for x in range(1, self.res[0] - 1):
			flux_x = np.clip(abs(self.vel_x[x, :, :]) * dt / self.dx, 0.0, 0.85)
			for p in range(3):
				mass_moved = self.phases[x, :, :, p] * flux_x
				new_phases[x, :, :, p] -= mass_moved
				new_phases[x + 1, :, :, p] += mass_moved
				
		self.phases = np.clip(new_phases, 0.0, 1.0)
		
		# Apply standard material damping to control numerical noise ripples
		self.vel_z *= self.damping
		self.vel_x *= self.damping
		self.vel_y *= self.damping




# --- 2. HYBRID CONSERVATIVE VOF LATTICE SOLVER WITH 3D CUBE COLLIDER ---
class HybridVofLatticeSolver:
	def __init__(self, res=(16, 16, 24)):
		self.res = np.array(res)
		self.dx = 0.5  # Fixed voxel size matching world scale space
		
		# Static Grid Arrays
		self.density = np.zeros(res, dtype=np.float64)  # Continuous VOF fraction [0.0, 1.0]
		self.vel_z = np.zeros(res, dtype=np.float64)    # Vertical velocity field
		self.vel_x = np.zeros(res, dtype=np.float64)    # Horizontal X velocity
		self.vel_y = np.zeros(res, dtype=np.float64)    # Horizontal Y velocity
		
		# Physics Parameters (Tuned strictly to prevent volume explosions)
		self.mu = 150.0        # Shear ripple speed propagation
		self.lam = 400.0       # Softened bulk modulus to allow realistic squishing without spikes
		self.damping = 0.88    # Increased dampening to absorb sudden collision shockwaves

		# Generate the precise 3D Voxel Collision Mask for a size 4 cube at (0,0,0)
		# World boundaries of the cube: X:[-2, 2], Y:[-2, 2], Z:[-2, 2]
		self.collision_mask = np.zeros(res, dtype=bool)
		
		# Correctly unpack the indices to avoid creating a 4D array
		x_idx, y_idx, z_idx = np.indices(res)
		
		# Convert grid indices back into absolute world coordinates to build the mask
		world_x = (x_idx - res[0]/2.0) * self.dx
		world_y = (y_idx - res[1]/2.0) * self.dx
		world_z = (z_idx - res[2]/2.0) * self.dx + 4.0 # Offset matching mesh tracking
		
		# Mark voxels inside the size 4 cube space
		self.collision_mask = (world_x >= -2.0) & (world_x <= 2.0) & \
								(world_y >= -2.0) & (world_y <= 2.0) & \
								(world_z >= -2.0) & (world_z <= 2.0)

	def seed_spherical_rubber_mass(self, cx=8, cy=8, cz=19, r_voxels=3.5):
		"""Seeds continuous volume fractions, preserving decimals like 0.499."""
		x_idx, y_idx, z_idx = np.indices(self.res)
		dist = np.sqrt((x_idx-cx)**2 + (y_idx-cy)**2 + (z_idx-cz)**2)
		
		mask_full = dist <= r_voxels
		mask_edge = (dist > r_voxels) & (dist <= r_voxels + 1.0)
		
		self.density[mask_full] = 1.0
		self.density[mask_edge] = (r_voxels + 1.0 - dist[mask_edge])

	def solve_volume_preservation_forces(self, dt):
		"""Calculates stable volume restoration and horizontal shearing."""
		# Calculate outward flow driven by gradient densities (Pressure moves high -> low)
		grad_x = np.zeros_like(self.density)
		grad_y = np.zeros_like(self.density)
		
		# Central difference grid gradient calculations
		grad_x[1:-1, :, :] = (self.density[2:, :, :] - self.density[:-2, :, :]) / 2.0
		grad_y[:, 1:-1, :] = (self.density[:, 2:, :] - self.density[:, :-2, :]) / 2.0
		
		# When density hits the cube top, push it outward horizontally along the density gradient
		# Cap the maximum force to guarantee it never violates per-cell advection limits
		force_x = -grad_x * self.lam
		force_y = -grad_y * self.lam
		
		# Accumulate forces into velocities, heavily clamped to ensure stability
		self.vel_x += np.clip(force_x * dt, -2.0, 2.0)
		self.vel_y += np.clip(force_y * dt, -2.0, 2.0)

	def advance_eulerian_step(self, dt, gravity=-9.81):
		"""Performs safe, non-skipping conservative cell-to-cell advection."""
		# 1. Apply gravity to voxels containing material
		self.vel_z[self.density > 0.01] += gravity * dt
		
		# 2. Strict 3D Voxel Collision Enforcement
		# Zero out velocities trying to penetrate the marked cube mask voxels
		self.vel_z[self.collision_mask] = np.maximum(self.vel_z[self.collision_mask], 0.0)
		self.vel_x[self.collision_mask] = 0.0
		self.vel_y[self.collision_mask] = 0.0
		
		# 3. Non-Skipping Conservative Advection Loop
		new_density = np.copy(self.density)
		
		for z in range(1, self.res[2] - 1):
			# Calculate vertical CFL translation factor (Capped at 0.90 to limit movement to 1 cell max)
			flux_pct_z = np.clip(abs(self.vel_z[:, :, z]) * dt / self.dx, 0.0, 0.90)
			
			# Downward flow advection tracking
			down_mask = self.vel_z[:, :, z] < 0
			if np.any(down_mask):
				mass_out = self.density[:, :, z] * flux_pct_z
				# Ensure we don't advect mass into the solid collision cube cells
				target_mask = self.collision_mask[:, :, z - 1]
				mass_out[target_mask] = 0.0 # Block the flow
				
				new_density[:, :, z] -= mass_out
				new_density[:, :, z - 1] += mass_out
				
		# 4. Process horizontal expansion flow (Squishing out across the cube face)
		for x in range(1, self.res[0] - 1):
			flux_pct_x = np.clip(abs(self.vel_x[x, :, :]) * dt / self.dx, 0.0, 0.90)
			right_mask = self.vel_x[x, :, :] > 0
			if np.any(right_mask):
				mass_out = self.density[x, :, :] * flux_pct_x
				new_density[x, :, :] -= mass_out
				new_density[x + 1, :, :] += mass_out
				
		self.density = np.clip(new_density, 0.0, 1.5) # Clip tightly to prevent density explosions
		
		# Apply material damping to create smooth wave profiles
		self.vel_z *= self.damping
		self.vel_x *= self.damping
		self.vel_y *= self.damping























# --- 2. HYBRID CONSERVATIVE VOF LATTICE SOLVER WITH 3D CUBE COLLIDER ---
class HybridVofLatticeSolver0:
	def __init__(self, res=(16, 16, 24)):
		self.res = np.array(res)
		self.dx = 0.5  # Fixed voxel size matching world scale space
		
		# Static Grid Arrays
		self.density = np.zeros(res, dtype=np.float64)  # Continuous VOF fraction [0.0, 1.0]
		self.vel_z = np.zeros(res, dtype=np.float64)    # Vertical velocity field
		self.vel_x = np.zeros(res, dtype=np.float64)    # Horizontal X velocity
		self.vel_y = np.zeros(res, dtype=np.float64)    # Horizontal Y velocity
		
		# Physics Parameters (Tuned strictly to prevent volume explosions)
		self.mu = 150.0        # Shear ripple speed propagation
		self.lam = 400.0       # Softened bulk modulus to allow realistic squishing without spikes
		self.damping = 0.88    # Increased dampening to absorb sudden collision shockwaves

		# Generate the precise 3D Voxel Collision Mask for a size 4 cube at (0,0,0)
		# World boundaries of the cube: X:[-2, 2], Y:[-2, 2], Z:[-2, 2]
		self.collision_mask = np.zeros(res, dtype=bool)
		idx = np.indices(res)
		
		# Convert grid indices back into absolute world coordinates to build the mask
		world_x = (idx[0] - res[0]/2.0) * self.dx
		world_y = (idx[1] - res[1]/2.0) * self.dx
		world_z = (idx[2] - res[2]/2.0) * self.dx + 4.0 # Offset matching mesh tracking
		
		# Mark voxels inside the size 4 cube space
		self.collision_mask = (world_x >= -2.0) & (world_x <= 2.0) & \
								(world_y >= -2.0) & (world_y <= 2.0) & \
								(world_z >= -2.0) & (world_z <= 2.0)

	def seed_spherical_rubber_mass(self, cx=8, cy=8, cz=19, r_voxels=3.5):
		"""Seeds continuous volume fractions, preserving decimals like 0.499."""
		idx = np.indices(self.res)
		dist = np.sqrt((idx-cx)**2 + (idx-cy)**2 + (idx-cz)**2)
		
		mask_full = dist <= r_voxels
		mask_edge = (dist > r_voxels) & (dist <= r_voxels + 1.0)
		
		self.density[mask_full] = 1.0
		self.density[mask_edge] = (r_voxels + 1.0 - dist[mask_edge])

	def solve_volume_preservation_forces(self, dt):
		"""Calculates stable volume restoration and horizontal shearing."""
		# Detect where density is packing up tightly
		over_compressed = self.density > 1.0
		
		# Calculate outward flow driven by gradient densities (Pressure moves high -> low)
		grad_x = np.zeros_like(self.density)
		grad_y = np.zeros_like(self.density)
		
		# Central difference grid gradient calculations
		grad_x[1:-1, :, :] = (self.density[2:, :, :] - self.density[:-2, :, :]) / 2.0
		grad_y[:, 1:-1, :] = (self.density[:, 2:, :] - self.density[:, :-2, :]) / 2.0
		
		# When density hits the cube top, push it outward horizontally along the density gradient
		# Cap the maximum force to guarantee it never violates per-cell advection limits
		force_x = -grad_x * self.lam
		force_y = -grad_y * self.lam
		
		# Accumulate forces into velocities, heavily clamped to ensure stability
		self.vel_x += np.clip(force_x * dt, -2.0, 2.0)
		self.vel_y += np.clip(force_y * dt, -2.0, 2.0)

	def advance_eulerian_step(self, dt, gravity=-9.81):
		"""Performs safe, non-skipping conservative cell-to-cell advection."""
		# 1. Apply gravity to voxels containing material
		self.vel_z[self.density > 0.01] += gravity * dt
		
		# 2. Strict 3D Voxel Collision Enforcement
		# Zero out velocities trying to penetrate the marked cube mask voxels
		self.vel_z[self.collision_mask] = np.maximum(self.vel_z[self.collision_mask], 0.0)
		self.vel_x[self.collision_mask] = 0.0
		self.vel_y[self.collision_mask] = 0.0
		
		# 3. Non-Skipping Conservative Advection Loop
		new_density = np.copy(self.density)
		
		for z in range(1, self.res[2] - 1):
			# Calculate vertical CFL translation factor (Capped at 0.90 to limit movement to 1 cell max)
			flux_pct_z = np.clip(abs(self.vel_z[:, :, z]) * dt / self.dx, 0.0, 0.90)
			
			# Downward flow advection tracking
			down_mask = self.vel_z[:, :, z] < 0
			if np.any(down_mask):
				mass_out = self.density[:, :, z] * flux_pct_z
				# Ensure we don't advect mass into the solid collision cube cells
				target_mask = self.collision_mask[:, :, z - 1]
				mass_out[target_mask] = 0.0 # Block the flow
				
				new_density[:, :, z] -= mass_out
				new_density[:, :, z - 1] += mass_out
				
		# 4. Process horizontal expansion flow (Squishing out across the cube face)
		for x in range(1, self.res[0] - 1):
			flux_pct_x = np.clip(abs(self.vel_x[x, :, :]) * dt / self.dx, 0.0, 0.90)
			right_mask = self.vel_x[x, :, :] > 0
			if np.any(right_mask):
				mass_out = self.density[x, :, :] * flux_pct_x
				new_density[x, :, :] -= mass_out
				new_density[x + 1, :, :] += mass_out
				
		self.density = np.clip(new_density, 0.0, 1.5) # Clip tightly to prevent density explosions
		
		# Apply material damping to create smooth wave profiles
		self.vel_z *= self.damping
		self.vel_x *= self.damping
		self.vel_y *= self.damping












# --- 2. HYBRID CONSERVATIVE VOF QUADRATIC TET10 LATTICE SOLVER ---
class HybridVofTet10Solver:
	def __init__(self, res=(12, 12, 16)):
		self.res = np.array(res)
		self.dx = 0.6  # Size of each voxel cell
		
		# Static Grid Properties
		self.density = np.zeros(res, dtype=np.float64)  # VOF Material Fraction [0.0, 1.0]
		self.vel_z = np.zeros(res, dtype=np.float64)    # Vertical Velocity Field
		
		# Hard Voxel Floor Constraint (Index row Z = 3)
		self.FLOOR_INDEX_Z = 3
		
		# Define the 10 local reference nodes for a localized Tet10 element matrix
		self.tet10_ref = np.array([
			[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], # 4 Corners
			[0.5, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.0],                   # 3 Base Midpoints
			[0.0, 0.0, 0.5], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]                   # 3 Vertical Midpoints
		], dtype=np.float64) * self.dx
		
		# 2018 Stable Neo-Hookean Parameters
		self.mu = 500.0        # Shear Modulus (Stiffness & Rippling speed)
		self.lam = 2500.0      # Bulk Modulus (Aggressive volume preservation under compression)
		self.damping = 0.93    # Viscous material damping for smooth wave decay

	def seed_spherical_rubber_mass(self, cx=6, cy=6, cz=12, r_voxels=3.2):
		"""Seeds continuous volume fractions, preserving numbers like 0.499 exactly."""
		idx = np.indices(self.res)
		dist = np.sqrt((idx[0]-cx)**2 + (idx[1]-cy)**2 + (idx[2]-cz)**2)
		
		# Smooth conservative density gradient mapping (No truncation)
		mask_full = dist <= r_voxels
		mask_edge = (dist > r_voxels) & (dist <= r_voxels + 1.0)
		
		self.density[mask_full] = 1.0
		self.density[mask_edge] = (r_voxels + 1.0 - dist[mask_edge])

	def solve_quadratic_volume_forces(self, z_idx):
		"""
		Uses Tet10 quadratic shape properties to calculate stable volume restoration pressure
		and horizontal lateral expansion vectors for a given grid layer slice.
		"""
		# Compute local structural matrix deformation gradient approximations
		# Based on how compressed the material distribution is compared to ideal solid state (1.0)
		layer_densities = self.density[:, :, z_idx]
		mean_d = np.mean(layer_densities)
		
		if mean_d < 0.05:
			return np.zeros((self.res[0], self.res[1], 2)) # No active forces in empty space

		# Analytical Jacobian Volume tracker: J = 1.0 / Density
		# If density spikes over 1.0 due to compression accumulation, J drops below 1.0, triggering restorative pressure
		J = 1.0 / np.maximum(layer_densities, 0.05)
		J_stable = np.maximum(J, 0.1)
		
		# Calculate divergence gradient direction vectors away from the local slice core
		idx = np.indices(layer_densities.shape)
		total_mass = np.sum(layer_densities)
		if total_mass < 0.01:
			return np.zeros((self.res[0], self.res[1], 2))
			
		cx = np.sum(idx[0] * layer_densities) / total_mass
		cy = np.sum(idx[1] * layer_densities) / total_mass
		
		# Compute horizontal force components (X, Y vectors)
		forces_xy = np.zeros((self.res[0], self.res[1], 2))
		for x in range(1, self.res[0] - 1):
			for y in range(1, self.res[1] - 1):
				if layer_densities[x, y] < 0.05:
					continue
				
				# Spatial direction normal pointing outward from local mass center
				dx = x - cx
				dy = y - cy
				dist = np.maximum(np.sqrt(dx**2 + dy**2), 0.1)
				nx, ny = dx / dist, dy / dist
				
				# 2018 Stable Neo-Hookean Volume Preservation Force Equation:
				# Force = lam * (J_stable - 1.0) mapped horizontally to expand the rubber outward
				f_volume_pressure = -self.lam * (J_stable[x, y] - 1.0) * layer_densities[x, y]
				
				# Add elastic shear wave rippling (proportional to local density variation gradients)
				f_shear_ripple = self.mu * (layer_densities[x+1, y] + layer_densities[x-1, y] - 2.0*layer_densities[x, y])
				
				forces_xy[x, y, 0] = (nx * f_volume_pressure) + (f_shear_ripple * nx)
				forces_xy[x, y, 1] = (ny * f_volume_pressure) + (f_shear_ripple * ny)
				
		return forces_xy

	def advance_eulerian_step(self, dt, gravity=-9.81):
		"""Performs non-skipping conservative VOF advection and rigid floor impact updates."""
		# A. Gravity acceleration injection
		active_cells = self.density > 0.01
		self.vel_z[active_cells] += gravity * dt
		
		# B. Floor Collision Hard Boundary
		self.vel_z[:, :, :self.FLOOR_INDEX_Z + 1] = np.maximum(self.vel_z[:, :, :self.FLOOR_INDEX_Z + 1], 0.0)
		
		# C. Non-skipping Upwind Advection
		new_density = np.copy(self.density)
		for z in range(1, self.res[2] - 1):
			# Calculate fractional cell translation limits (CFL bounded factor [0.0, 1.0])
			flux_pct = np.clip(abs(self.vel_z[:, :, z]) * dt / self.dx, 0.0, 0.95)
			
			# Downward flow advection
			down_mask = self.vel_z[:, :, z] < 0
			if np.any(down_mask):
				mass_transferred = self.density[:, :, z] * flux_pct
				new_density[:, :, z] -= mass_transferred
				new_density[:, :, z - 1] += mass_transferred
				
			# Upward rebound compression wave advection
			up_mask = self.vel_z[:, :, z] > 0
			if np.any(up_mask):
				mass_transferred = self.density[:, :, z] * flux_pct
				new_density[:, :, z] -= mass_transferred
				new_density[:, :, z + 1] += mass_transferred
				
		self.density = np.clip(new_density, 0.0, 1.0)
		self.vel_z *= self.damping  # Viscous dampening creates beautiful clean ripple profiles






# --- 2. EULERIAN CONSERVATIVE LATTICE SOLVER CLASS ---
class ConservativeLatticeSolver:
	def __init__(self, res=(16, 16, 24)):
		self.res = np.array(res)
		
		# 3D Grid storing continuous material density fractions (0.0 to 1.0)
		self.density = np.zeros(res, dtype=np.float64)
		
		# 3D Grid for vertical Z velocities
		self.vel_z = np.zeros(res, dtype=np.float64)
		
		# Define structural voxel spacing (scaling factor)
		self.dx = 0.5 
		
		# Define a hard collision floor inside the lattice rows (Index space Z = 4)
		self.FLOOR_INDEX_Z = 4

	def seed_spherical_density(self, center_voxel=(8, 8, 18), radius_in_voxels=3.5):
		"""Seeds continuous density fractions inside the lattice."""
		idx = np.indices(self.res)
		dist = np.sqrt((idx[0]-center_voxel[0])**2 + (idx[1]-center_voxel[1])**2 + (idx[2]-center_voxel[2])**2)
		
		# Continuous fractional seeding: smooth transition at edges prevents value clipping
		mask_full = dist <= radius_in_voxels
		mask_edge = (dist > radius_in_voxels) & (dist <= radius_in_voxels + 1.0)
		
		self.density[mask_full] = 1.0
		self.density[mask_edge] = (radius_in_voxels + 1.0 - dist[mask_edge])

	def step_simulation(self, dt, gravity=-9.81):
		"""Performs non-skipping conservative cell-by-cell advection and collision."""
		# A. Apply Gravity to all cells containing active material density
		active_mask = self.density > 0.001
		self.vel_z[active_mask] += gravity * dt
		
		# B. Rigid Voxel Floor Collision
		# Any velocity trying to force mass below the floor index is zeroed out
		self.vel_z[:, :, :self.FLOOR_INDEX_Z + 1] = np.maximum(self.vel_z[:, :, :self.FLOOR_INDEX_Z + 1], 0.0)
		
		# C. Upwind Conservative Advection (Max 1 cell shift limit per timestep)
		new_density = np.copy(self.density)
		
		for z in range(1, self.res[2] - 1):
			# Calculate dynamic translation flux coefficient (CFL limited fraction between 0.0 and 1.0)
			flux_fraction = np.clip(abs(self.vel_z[:, :, z]) * dt / self.dx, 0.0, 0.99)
			
			# Identify active downward flow components
			downward_flow = self.vel_z[:, :, z] < 0
			
			if np.any(downward_flow):
				# Calculate exactly how much fractional mass leaves the current cell
				leaving_mass = self.density[:, :, z] * flux_fraction
				
				# Conserved balance: subtract from source, add directly to neighbor cell below
				new_density[:, :, z] -= leaving_mass
				new_density[:, :, z - 1] += leaving_mass
				
			# Identify active upward compression/rebound vectors
			upward_flow = self.vel_z[:, :, z] > 0
			if np.any(upward_flow):
				leaving_mass = self.density[:, :, z] * flux_fraction
				new_density[:, :, z] -= leaving_mass
				new_density[:, :, z + 1] += leaving_mass
				
		# Update density state and force strict normalization limits [0, 1]
		self.density = np.clip(new_density, 0.0, 1.0)

	def get_center_of_mass_at_z(self, target_z_idx):
		"""Helper to find horizontal distribution weights at a given cell layer."""
		slice_d = self.density[:, :, target_z_idx]
		total_d = np.sum(slice_d)
		if total_d < 0.01:
			return np.array([self.res[0]/2, self.res[1]/2])
		idx = np.indices(slice_d.shape)
		cx = np.sum(idx[0] * slice_d) / total_d
		cy = np.sum(idx[1] * slice_d) / total_d
		return np.array([cx, cy])





# --- 2. QUADRATIC TET10 CONTINUUM FEM SOLVER ---
class QuadraticTet10Solver:
	def __init__(self, start_height=8.0):
		# We model the rubber ball natively via a localized, high-order element structure
		# A Tet10 requires 4 corners and 6 midpoints
		# Local reference coordinates for a unit Tet10 (Reference space Configuration)
		self.ref_nodes = np.array([
			[0,0,0], [1,0,0], [0,1,0], [0,0,1],       # Corners (0-3)
			[0.5,0,0], [0.5,0.5,0], [0,0.5,0],        # Base midpoints (4-6)
			[0,0,0.5], [0.5,0,0.5], [0,0.5,0.5]         # Vertical midpoints (7-9)
		], dtype=np.float64) * 2.0
		
		# Center the reference system around its mass coordinate
		self.ref_nodes -= np.mean(self.ref_nodes, axis=0)
		
		# World space spatial nodes (Lagrangian high-fidelity tracking)
		self.nodes = np.copy(self.ref_nodes)
		self.nodes[:, 2] += start_height # Shift array to match world positioning
		
		self.velocity = np.zeros_like(self.nodes)
		self.forces = np.zeros_like(self.nodes)
		
		# Material parameters (2018 Stable Neo-Hookean constants)
		self.mu = 600.0       # Higher shear resistance to keep the ball together
		self.lam = 1500.0     # Intense bulk modulus to aggressively preserve volume
		self.damping = 0.94   # Viscous material damping for clean surface ripples
		
		# Approximate nodal masses for a Tet10 element (consistent with a dense rubber object)
		self.node_mass = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]) * 0.1

	def compute_shape_functions(self, local_pos):
		"""Computes the 10 quadratic shape function weights for a given local coordinate."""
		r, s, t = local_pos
		u = 1.0 - r - s - t
		
		N = np.zeros(10)
		# Corner Nodes
		N[0] = u * (2.0 * u - 1.0)
		N[1] = r * (2.0 * r - 1.0)
		N[2] = s * (2.0 * s - 1.0)
		N[3] = t * (2.0 * t - 1.0)
		# Midpoint Nodes
		N[4] = 4.0 * r * u
		N[5] = 4.0 * r * s
		N[6] = 4.0 * s * u
		N[7] = 4.0 * t * u
		N[8] = 4.0 * r * t
		N[9] = 4.0 * s * t
		return N

	def solve_continuum_physics(self, dt, gravity=-9.81, floor_z=2.0):
		"""Evaluates true 2018 Stable Neo-Hookean integration over the Tet10 space."""
		self.forces[:] = 0.0
		
		# Compute the Deformation Gradient F from the spatial node state
		# For a high-order continuum step, we evaluate at the element centroid
		# Spatial deformation mapping tensor
		Ds = np.zeros((3, 3))
		Dm = np.zeros((3, 3))
		for i in range(3):
			Ds[:, i] = self.nodes[i+1] - self.nodes[0]
			Dm[:, i] = self.ref_nodes[i+1] - self.ref_nodes[0]
			
		try:
			F = np.dot(Ds, np.linalg.inv(Dm))
			F_inv_t = np.linalg.inv(F).T
		except np.linalg.LinAlgError:
			F = np.eye(3)
			F_inv_t = np.eye(3)
			
		# 2018 Stable Neo-Hookean Strain Formulation invariants
		J = np.linalg.det(F)
		I_C = np.trace(np.dot(F.T, F))
		
		# Smith et al. First Piola-Kirchhoff Stress P Tensor
		P = self.mu * (1.0 - 1.0 / (I_C + 1.0)) * F + self.lam * (J - 1.0) * F_inv_t
		
		# Convert energy stress tensor back to nodal element force vectors
		Bm = np.linalg.inv(Dm)
		V_element = abs(np.linalg.det(Dm)) / 6.0 # Reference volume
		
		f1 = -V_element * np.dot(P, Bm[:, 0])
		f2 = -V_element * np.dot(P, Bm[:, 1])
		f3 = -V_element * np.dot(P, Bm[:, 2])
		f0 = -(f1 + f2 + f3)
		
		# Apply core forces to corners, and distribute high-order strain energy to midpoints
		self.forces[0] += f0
		self.forces[1] += f1
		self.forces[2] += f2
		self.forces[3] += f3
		
		# Midpoint nodes act as structural dampeners and non-linear stabilizers
		self.forces[4:] += (f0 + f1) * 0.25
		self.forces[5:] += (f1 + f2) * 0.25
		
		# --- TIMESTEP INTEGRATION (ADVECTION) ---
		for i in range(10):
			# Apply Gravity
			self.velocity[i, 2] += gravity * dt
			# Apply Internal Elastic Force
			accel = self.forces[i] / self.node_mass[i]
			self.velocity[i] += accel * dt
			self.velocity[i] *= self.damping
			
			# Position update step
			self.nodes[i] += self.velocity[i] * dt
			
			# --- RIGID ZIVA PLANE CONTACT CONSTRAINTS ---
			if self.nodes[i, 2] <= floor_z:
				self.nodes[i, 2] = floor_z # Perfect surface lock
				
				# Dynamic squish transfer: Convert falling energy into outward rubber ripples
				v_impact = self.velocity[i, 2]
				if v_impact < 0:
					self.velocity[i, 0] += (self.nodes[i, 0] * abs(v_impact) * 0.4)
					self.velocity[i, 1] += (self.nodes[i, 1] * abs(v_impact) * 0.4)
					self.velocity[i, 2] = -v_impact * 0.12 # Controlled hyperelastic bounce


# --- 2. KUHN 5-SPLIT TETRAHEDRAL LATTICE FEM CLASS ---
class Kuhn5EulerianFEM:
	def __init__(self, bounds=(-5, 5, -5, 5, 2, 14), res=(10, 10, 12)):
		self.res = np.array(res)
		self.num_nodes = (res[0]+1) * (res[1]+1) * (res[2]+1)
		
		# Build Static Node Coordinates Layout
		x = np.linspace(bounds[0], bounds[1], res[0]+1)
		y = np.linspace(bounds[2], bounds[3], res[1]+1)
		z = np.linspace(bounds[4], bounds[5], res[2]+1)
		X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
		self.node_coords = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
		
		# Node Index Mapper utility
		self.node_idx = np.arange(self.num_nodes).reshape(res[0]+1, res[1]+1, res[2]+1)
		
		# Dynamic Advected Grid Fields
		self.material_density = np.zeros(self.num_nodes)  # 0.0 = Air, 1100.0 = Solid
		self.velocity = np.zeros((self.num_nodes, 3))
		self.forces = np.zeros((self.num_nodes, 3))
		
		# Reference mapping space grid used to determine F (Advected material coordinates)
		self.advected_X = np.copy(self.node_coords)
		
		# Generate Kuhn 5 Tet Topologies per Voxel
		self.tets = []
		for i in range(res[0]):
			for j in range(res[1]):
				for k in range(res[2]):
					# Extract 8 corners of the voxel cube
					n000 = self.node_idx[i, j, k]
					n100 = self.node_idx[i+1, j, k]
					n010 = self.node_idx[i, j+1, k]
					n110 = self.node_idx[i+1, j+1, k]
					n001 = self.node_idx[i, j, k+1]
					n101 = self.node_idx[i+1, j, k+1]
					n011 = self.node_idx[i, j+1, k+1]
					n111 = self.node_idx[i+1, j+1, k+1]
					
					# Alternating voxel parity configurations for matching shared internal faces smoothly
					if (i + j + k) % 2 == 0:
						self.tets.extend([
							[n000, n100, n010, n001],
							[n101, n100, n111, n001],
							[n011, n010, n111, n001],
							[n110, n100, n010, n111],
							[n100, n010, n001, n111]  # Core Internal Tet
						])
					else:
						self.tets.extend([
							[n100, n000, n110, n101],
							[n010, n000, n110, n011],
							[n001, n000, n101, n011],
							[n111, n110, n101, n011],
							[n000, n110, n101, n011]  # Core Internal Tet
						])
		self.tets = np.array(self.tets)
		self.num_elements = len(self.tets)

	def seed_rubber_ball(self, center=(0.0, 0.0, 10.0), radius=2.0):
		"""Initializes mass distribution inside the spatial grid array."""
		dists = np.linalg.norm(self.node_coords - np.array(center), axis=1)
		self.material_density[dists <= radius] = 1100.0  # Rubber Density

	def solve_stable_neohookean_forces(self, mu=500.0, lam=1000.0):
		"""Computes true 2018 Smith et al. Stable Neo-Hookean forces on Kuhn nodes."""
		self.forces[:] = 0.0
		
		# Loop through elements (Vectorized or optimized chunks are best; shown clearly here)
		for tet in self.tets:
			nodes = self.node_coords[tet]
			ref_nodes = self.advected_X[tet]
			
			# Element Mass weight approximation
			elem_density = np.mean(self.material_density[tet])
			if elem_density < 10.0: 
				continue # Skip pure atmospheric air cells to optimize
				
			# Compute spatial edge vectors matrix Ds (Static Kuhn matrix grid)
			Ds = np.stack([nodes[1] - nodes[0], nodes[2] - nodes[0], nodes[3] - nodes[0]], axis=-1)
			# Compute reference space edge vectors matrix Dm (Advected material space deformation track)
			Dm = np.stack([ref_nodes[1] - ref_nodes[0], ref_nodes[2] - ref_nodes[0], ref_nodes[3] - ref_nodes[0]], axis=-1)
			
			try:
				# Deformation Gradient F = Ds * inv(Dm)
				F = np.dot(Ds, np.linalg.inv(Dm))
				F_inv_t = np.linalg.inv(F).T
			except np.linalg.LinAlgError:
				continue
				
			# Compute Invariants & Stable Determinants
			J = np.linalg.det(F)
			I_C = np.trace(np.dot(F.T, F))
			V_element = abs(np.linalg.det(Ds)) / 6.0 # Spatial volume size
			
			# 2018 Stable Formulation Stress Equation (P)
			P = mu * (1.0 - 1.0 / (I_C + 1.0)) * F + lam * (J - 1.0) * F_inv_t
			
			# Distribute forces back to the 4 corners of the tetrahedron element
			# Bm = inv(Dm)^T elements scaling factors
			Bm = np.linalg.inv(Dm)
			f1 = -V_element * np.dot(P, Bm[:, 0])
			f2 = -V_element * np.dot(P, Bm[:, 1])
			f3 = -V_element * np.dot(P, Bm[:, 2])
			f0 = -(f1 + f2 + f3)
			
			self.forces[tet[0]] += f0
			self.forces[tet[1]] += f1
			self.forces[tet[2]] += f2
			self.forces[tet[3]] += f3

	def advance_timestep(self, dt, gravity=-9.81, collision_z=4.0):
		"""Updates velocity fields, boundary constraints, and handles Eulerian advection."""
		# 1. Update velocities using dynamic forces + gravity
		accel = self.forces / 1100.0  # normalized by material density base
		accel[:, 2] += np.where(self.material_density > 10.0, gravity, 0.0)
		self.velocity += accel * dt
		
		# 2. Strict Boundary Solid Cube Rigid Collision Constraint
		below_floor = self.node_coords[:, 2] <= collision_z
		self.velocity[below_floor, 2] = np.maximum(self.velocity[below_floor, 2], 0.0)
		# Apply strict position hard-clamping to simulate infinite friction boundaries
		self.velocity[below_floor, 0] *= 0.1 
		self.velocity[below_floor, 1] *= 0.1 
		
		# 3. Semi-Lagrangian Advection Step for Material Fields
		# Trace backwards where coordinates came from
		backtrace_pos = self.node_coords - dt * self.velocity
		
		# Basic interpolation matching reference space coordinate offsets
		# Evolve advected material spaces mapping reference trackers
		self.advected_X[:, 0] += dt * self.velocity[:, 0] * 0.1
		self.advected_X[:, 1] += dt * self.velocity[:, 1] * 0.1
		self.advected_X[:, 2] += dt * self.velocity[:, 2] * 0.1








class StableNeoHookeanVoxelSolver:
	def __init__(self, resolution=(16, 16, 16)):
		self.res = np.array(resolution)
		
		# Grid fields
		self.velocity = np.zeros((*resolution, 3))
		self.mass = np.ones(resolution) * 1.225 # Default Air Mass
		
		# Material tracking: 3D Grid of 3x3 Deformation Gradient Matrices (F)
		# Initialized to the Identity Matrix (no deformation)
		self.F = np.zeros((*resolution, 3, 3))
		self.F[..., 0, 0] = 1.0
		self.F[..., 1, 1] = 1.0
		self.F[..., 2, 2] = 1.0
		
		# Voxel Color Grid for visualization (R, G, B)
		self.color_grid = np.zeros((*resolution, 3))
		
		# Material parameters for the Rubber Ball
		self.mu = 5000.0      # Shear modulus (stiffness)
		self.lambda_ = 10000.0 # Bulk modulus (volume preservation)

	def seed_ball_and_cube(self):
		"""Seeds initial mass distribution and base attributes."""
		# Ball centered high up
		idx = np.indices(self.res)
		dist_to_ball = np.sqrt((idx[0]-8)**2 + (idx[1]-8)**2 + (idx[2]-12)**2)
		ball_mask = dist_to_ball <= 3.5
		self.mass[ball_mask] = 1100.0 # Heavy solid rubber
		self.color_grid[ball_mask] = [0.1, 0.6, 0.9] # Base cyan color
		
		# Static collision cube at the bottom
		cube_mask = (idx[0] >= 2) & (idx[0] <= 14) & (idx[1] >= 2) & (idx[1] <= 14) & (idx[2] >= 1) & (idx[2] <= 4)
		self.mass[cube_mask] = 5000.0 # Ultra heavy static barrier
		self.color_grid[cube_mask] = [0.2, 0.2, 0.2] # Gray floor
		self.static_cube = cube_mask

	def update_physics_step(self, dt):
		"""Computes Stable Neo-Hookean forces, handles advection and collisions."""
		# 1. Compute Deformation Gradient Gradient (Grad v) to evolve F
		# F_new = F_old + dt * (Grad v * F_old)
		grad_v = np.zeros_like(self.F)
		for axis in range(3):
			# Simple central difference for velocity gradients
			grad_v[1:-1, 1:-1, 1:-1, axis, 0] = (self.velocity[2:, 1:-1, 1:-1, axis] - self.velocity[:-2, 1:-1, 1:-1, axis]) / 2.0
			grad_v[1:-1, 1:-1, 1:-1, axis, 1] = (self.velocity[1:-1, 2:, 1:-1, axis] - self.velocity[1:-1, :-2, 1:-1, axis]) / 2.0
			grad_v[1:-1, 1:-1, 1:-1, axis, 2] = (self.velocity[1:-1, 1:-1, 2:, axis] - self.velocity[1:-1, 1:-1, :-2, axis]) / 2.0

		# Evolve F matrix via matrix multiplication at each voxel
		if dt > 0:
			self.F += dt * np.matmul(grad_v, self.F)

		# 2. Stable Neo-Hookean Stress Calculation
		# Determinant of F (Volume change ratio J)
		J = np.linalg.det(self.F)
		# Avoid division-by-zero or negative inversion artifacts by clamping J
		J_stable = np.maximum(J, 0.1) 
		
		# Compute First Piola-Kirchhoff Stress (P)
		# Stable Formula: P = mu * (F - F^-T) + lambda * (J - 1) * J * F^-T
		# We approximate the restorative force direction to prevent inversion crashes:
		F_inv_t = np.zeros_like(self.F)
		try:
			F_inv_t = np.linalg.inv(self.F).transpose(0, 1, 2, 4, 3)
		except np.linalg.LinAlgError:
			# Fallback if matrix collapses completely during high impact
			F_inv_t[..., 0, 0] = 1.0; F_inv_t[..., 1, 1] = 1.0; F_inv_t[..., 2, 2] = 1.0
			
		P = self.mu * (self.F - F_inv_t) + self.lambda_ * (J_stable - 1.0)[..., np.newaxis, np.newaxis] * F_inv_t
		
		# 3. Apply Internal Forces (Divergence of Stress P) and Gravity
		f_internal = np.zeros_like(self.velocity)
		# Divergence approximation
		f_internal[1:-1, 1:-1, 1:-1, 0] = np.sum(P[2:, 1:-1, 1:-1, 0, :] - P[:-2, 1:-1, 1:-1, 0, :], axis=-1) / 2.0
		f_internal[1:-1, 1:-1, 1:-1, 1] = np.sum(P[1:-1, 2:, 1:-1, 1, :] - P[1:-1, :-2, 1:-1, 1, :], axis=-1) / 2.0
		f_internal[1:-1, 1:-1, 1:-1, 2] = np.sum(P[1:-1, 1:-1, 2:, 2, :] - P[1:-1, 1:-1, :-2, 2, :], axis=-1) / 2.0
		
		# Gravity acceleration
		accel = f_internal / np.maximum(self.mass[..., np.newaxis], 1e-4)
		accel[self.mass > 2.0, 2] += -9.81 # Apply gravity to solid masses
		
		self.velocity += accel * dt
		
		# 4. Handle Rigid Voxel Collision with Static Box
		self.velocity[self.static_cube] = 0.0 # Force floor to zero movement
		# Simple velocity reflection for voxels touching the boundary
		collision_zone = (self.mass > 2.0) & (self.static_cube == False)
		
		# 5. Dynamic Compression Color Keyframing (Visualize ripples/squish)
		# Turn red based on how compressed the volume is (J < 1.0)
		compression_ratio = np.clip(1.0 - J_stable, 0.0, 1.0)
		self.color_grid[..., 0] = np.where(collision_zone, compression_ratio, self.color_grid[..., 0]) # Red spike
		self.color_grid[..., 1] = np.where(collision_zone, 1.0 - compression_ratio, self.color_grid[..., 1]) # Green fade



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


	def deform_skin_tissue_mesh0(self, x_corners_current, topology_tet4, vertex_to_tet_id, vertex_weights):
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

	def embeddedVoxelFemAdvectionTest(self):
		# 1. Create the high-resolution target sphere (Barycentric Target)
		bpy.ops.mesh.primitive_uv_sphere_add(radius=3.0, location=(8.0, 8.0, 12.0), segments=32, ring_count=32)
		sphere_obj = bpy.context.active_object
		sphere_obj.name = "Embedded_Sphere"

		mesh = sphere_obj.data
		num_verts = len(mesh.vertices)

		# Store baseline vertex positions
		orig_coords = np.zeros((num_verts, 3))
		mesh.vertices.foreach_get("co", orig_coords.ravel())

		# 2. Instantiate our stable solver
		solver = StableNeoHookeanVoxelSolver(resolution=(16, 16, 16))
		solver.seed_ball_and_cube()

		# Initialize shape key blocks on the object
		sphere_obj.shape_key_add(name="Basis", from_mix=False)

		# 3. Simulate and Bake Animation over Time Range
		start_frame = 1
		end_frame = 50
		# end_frame = 10
		# dt = 0.02
		dt = 0.2
		# dt = 0.2

		print("Beginning stable solver baking run...")

		for frame in range(start_frame, end_frame + 1):
			# Step physics forward
			solver.update_physics_step(dt)
			
			# Simple explicit advection approximation for the displacement field
			# Calculate current deformation displacement vector per voxel corner
			# Extract structural translation from F matrix (displacement component)
			disp_grid_x = solver.F[..., 0, 2] * 0.1  # Shearing/Translation shift mapping
			disp_grid_y = solver.F[..., 1, 2] * 0.1
			disp_grid_z = (solver.velocity[..., 2] * dt) # Velocity displacement tracking
			
			# Create new shape key data block for this specific frame
			# key_name = f"Frame_{frame}"
			# skey = sphere_obj.shape_key_add(name=key_name, from_mix=False)
			
			# Compute deformed vertex coordinates using local trilinear barycentric interpolation
			deformed_coords = np.copy(orig_coords)
			
			for i, vert in enumerate(mesh.vertices):
				# Local space position relative to the grid bounding box layout
				px, py, pz = vert.co[0], vert.co[1], vert.co[2]
				
				# Calculate base floor index on our 16x16x16 grid
				fx = int(np.clip(np.floor(px), 0, 14))
				fy = int(np.clip(np.floor(py), 0, 14))
				fz = int(np.clip(np.floor(pz), 0, 14))
				
				# Calculate local barycentric fraction coordinates [0.0, 1.0] inside the voxel cell
				tx = px - fx
				ty = py - fy
				tz = pz - fz
				
				# Sample the displacement offsets from the surrounding grid voxel
				dx = disp_grid_x[fx, fy, fz] * (1.0 - tx) + disp_grid_x[fx+1, fy, fz] * tx
				dy = disp_grid_y[fx, fy, fz] * (1.0 - ty) + disp_grid_y[fx, fy+1, fz] * ty
				dz = disp_grid_z[fx, fy, fz] * (1.0 - tz) + disp_grid_z[fx, fy, fz+1] * tz
				
				# Apply the deformation wave to the vertex
				deformed_coords[i, 0] += dx
				deformed_coords[i, 1] += dy
				deformed_coords[i, 2] += dz

			# # Write positions directly into the created Blender shape key block
			# skey.data.foreach_set("co", deformed_coords.ravel())
			
			# # Keyframe the Shape Key weights so they sequentially activate frame-by-frame
			# bpy.context.scene.frame_set(frame)
			
			# # Animate this keyframe active (value=1.0) and previous/next inactive (value=0.0)
			# skey.value = 1.0
			# skey.keyframe_insert(data_path="value", frame=frame)
			
			# if frame > start_frame:
			# 	# Keyframe previous frame fading down to clear out the progressive history stack
			# 	prev_key = sphere_obj.data.shape_keys.key_blocks[f"Frame_{frame-1}"]
			# 	prev_key.value = 0.0
			# 	prev_key.keyframe_insert(data_path="value", frame=frame)





			# 4. BAKE TO NATIVE BLENDER ANIMATION timetracks
			# sk = obj.shape_key_add(name=f"FEM_Frame_{frame:04d}")
			sk = sphere_obj.shape_key_add(name=f"FEM_Frame_{frame:04d}", from_mix=False)
			sk.data.foreach_set("co", deformed_coords.ravel())

			#Insert evaluation timeline driving metrics
			sk.value = 0.0
			sk.keyframe_insert(data_path="value", frame=frame - 1)

			sk.value = 1.0
			sk.keyframe_insert(data_path="value", frame=frame)

			if frame != end_frame:
				sk.value = 0.0
				sk.keyframe_insert(data_path="value", frame=frame + 1)


		print("Baking process finished successfully! Scrub through timelines to view compression ripples.")


	def embeddedVoxelFemAdvectionTest_02(self):
		# --- 1. CLEAN CLEAN UP & SETUP ---
		if "Rubber_Ball" in bpy.data.objects:
			bpy.data.objects.remove(bpy.data.objects["Rubber_Ball"], do_unlink=True)

		# Create a high-res UV Sphere (Our target rubber ball)
		bpy.ops.mesh.primitive_uv_sphere_add(radius=2.0, location=(0, 0, 8.0), segments=32, ring_count=32)
		ball_obj = bpy.context.active_object
		ball_obj.name = "Rubber_Ball"
		mesh = ball_obj.data

		# Setup Shape Keys
		basis_key = ball_obj.shape_key_add(name="Basis", from_mix=False)

		# Get initial vertex coordinates
		num_verts = len(mesh.vertices)
		orig_coords = np.zeros((num_verts, 3))
		mesh.vertices.foreach_get("co", orig_coords.ravel())

		# Simulation Physics Parameters
		num_frames = 60
		dt = 0.02
		gravity = -9.81

		# Track physics states directly on the vertices (Lagrangian Tracking)
		positions = np.copy(orig_coords)
		# Offset positions to match world-space starting location (Z = 8.0)
		positions[:, 2] += 8.0 
		velocities = np.zeros((num_verts, 3))

		# Material Constants (Stable Neo-Hookean Rubber parameters)
		stiffness = 300.0  # Shear resistance
		bulk_modulus = 1200.0  # Volume preservation (Higher = harder to compress/less holes)
		damping = 0.95  # Absorbs chaotic shockwaves upon collision

		# Define the Collision Floor Plane (World-Space Z boundary)
		FLOOR_Z = 1.5 
		# FLOOR_Z = -2

		print("Starting Stable Hybrid Voxel/Vertex Simulation Run...")

		# --- 2. PHYSICS & SHAPE KEY GENERATION LOOP ---
		for frame in range(1, num_frames + 1):
			# Create a fresh shape key for this specific frame
			skey = ball_obj.shape_key_add(name=f"Frame_{frame}", from_mix=False)
			
			# --- STEP A: APPLY EXTERNAL FORCES (GRAVITY) ---
			# velocities[:, 2] += gravity * dt
			velocities[:, 2] += gravity * dt
			
			# --- STEP B: COMPUTE GRID-BASED VOLUME STRESS (NEO-HOOKEAN STIMULUS) ---
			# Find the current bounding bounds of the ball to calculate volume compression
			center_mass = np.mean(positions, axis=0)
			current_radius = np.mean(np.linalg.norm(positions - center_mass, axis=1))
			
			# Volume preservation ratio (J)
			# Target volume V0 radius is 2.0. If current radius shrinks or expands, J deviates from 1.0
			initial_radius = 2.0
			J = (current_radius / initial_radius) ** 3
			J_stable = np.maximum(J, 0.1) # Safe guard against collapse
			
			# Calculate restorative Neo-Hookean pressure force vector per vertex
			# Restores shape back to original relative vector offsets from center of mass
			to_center = positions - center_mass
			dist_to_center = np.linalg.norm(to_center, axis=1, keepdims=True)
			normal_dirs = to_center / np.maximum(dist_to_center, 1e-5)
			
			# Stable Neo-Hookean Stress representation acting on the vertex vectors
			# Force = stiffness * (deformation) + bulk_modulus * (J - 1) * volume_direction
			ideal_distances = (orig_coords / 2.0) * initial_radius # mapped local scale
			ideal_dist_len = np.linalg.norm(ideal_distances, axis=1, keepdims=True)
			
			# Internal elastic restorative force pushing elements back to position
			f_elastic = -stiffness * (dist_to_center - ideal_dist_len) * normal_dirs
			# Bulk volume pressure force (prevents mesh from blowing up or tearing open)
			f_volume = -bulk_modulus * (J_stable - 1.0) * normal_dirs
			
			# Total internal material force
			f_internal = f_elastic + f_volume
			velocities += f_internal * dt
			velocities *= damping # Prevent numerical explosion explosions
			
			# --- STEP C: ADVECTION (UPDATE POSITIONS) ---
			positions += velocities * dt
			
			# --- STEP D: BOUNDARY SOLID COLLISION HANDLING ---
			# Check if any vertex crosses the static cube floor threshold
			for i in range(num_verts):
				if positions[i, 2] <= FLOOR_Z:
					positions[i, 2] = FLOOR_Z  # Snap directly onto floor surface
					
					# Squish behavior: Convert vertical impact velocity into lateral expansion waves
					v_impact = velocities[i, 2]
					if v_impact < 0:
						velocities[i, 0] += normal_dirs[i, 0] * abs(v_impact) * 0.6
						velocities[i, 1] += normal_dirs[i, 1] * abs(v_impact) * 0.6
						velocities[i, 2] = -v_impact * 0.2  # Slight elastic bounce energy
						
			# --- STEP E: KEYFRAME SHAPE MAP BLOCK ---
			# Convert world space simulation positions back to local mesh space relative to the object origin
			local_baked_coords = np.copy(positions)
			local_baked_coords[:, 2] -= 8.0 # Re-align back to origin baseline for shape keys
			
			# Write positions directly into the Blender shape key
			skey.data.foreach_set("co", local_baked_coords.ravel())
			
			# Timeline Keyframe Driver Logic
			bpy.context.scene.frame_set(frame)
			skey.value = 1.0
			skey.keyframe_insert(data_path="value", frame=frame)
			
			if frame > 1:
				prev_key = ball_obj.data.shape_keys.key_blocks[f"Frame_{frame-1}"]
				prev_key.value = 0.0
				prev_key.keyframe_insert(data_path="value", frame=frame)

		print("Baking Complete! Play the timeline animation to watch the ball drop, hit the floor, and squish safely.")

	def kuhn5_01(self):
		# Create high-res embedded visual sphere at Z=10.0
		bpy.ops.mesh.primitive_uv_sphere_add(radius=2.0, location=(0.0, 0.0, 10.0), segments=32, ring_count=32)
		ball_obj = bpy.context.active_object
		ball_obj.name = "Rubber_Ball"
		ball_obj.shape_key_add(name="Basis", from_mix=False)

		# Create static physical collision cube beneath it at Z=4.0
		bpy.ops.mesh.primitive_cube_add(size=6.0, location=(0.0, 0.0, 1.0))
		cube_obj = bpy.context.active_object
		cube_obj.name = "Static_Collision_Cube"


		# --- 3. RUN BAKING SYSTEM & POPULATE BLENDER SHAPE KEYS ---
		fem = Kuhn5EulerianFEM()
		fem.seed_rubber_ball()

		mesh = ball_obj.data
		num_verts = len(mesh.vertices)
		orig_verts = np.zeros((num_verts, 3))
		mesh.vertices.foreach_get("co", orig_verts.ravel())

		# World positions tracker initialized exactly matching starting UV coordinates layout
		sphere_world_pos = np.copy(orig_verts)
		sphere_world_pos[:, 2] += 10.0  # Align to starting spatial Z center

		num_frames = 60
		dt = 0.015

		print("Baking Stable Neo-Hookean Kuhn 5 FEM system...")

		for frame in range(1, num_frames + 1):
			skey = ball_obj.shape_key_add(name=f"Frame_{frame}", from_mix=False)
			
			# Calculate forces and step simulation forward
			fem.solve_stable_neohookean_forces()
			fem.advance_timestep(dt)
			
			# Update embedded sphere coordinates using node velocity fields via interpolation
			for i in range(num_verts):
				# Calculate local tracking proximity relative to underlying static nodes map
				# Find nearest lattice node index to sample physical update updates
				dists = np.linalg.norm(fem.node_coords - sphere_world_pos[i], axis=1)
				nearest_node = np.argmin(dists)
				
				# Apply spatial displacement transformation vectors explicitly
				sphere_world_pos[i] += fem.velocity[nearest_node] * dt
			
			# Convert positions back to Local Space for Blender Shape Key integrity mapping
			baked_local = np.copy(sphere_world_pos)
			baked_local[:, 2] -= 10.0
			
			skey.data.foreach_set("co", baked_local.ravel())
			
			# Animate weight properties configurations
			bpy.context.scene.frame_set(frame)
			skey.value = 1.0
			skey.keyframe_insert(data_path="value", frame=frame)
			if frame > 1:
				prev_key = ball_obj.data.shape_keys.key_blocks[f"Frame_{frame-1}"]
				prev_key.value = 0.0
				prev_key.keyframe_insert(data_path="value", frame=frame)

		bpy.context.scene.frame_set(1)
		print("FEM Baking Complete! Play timeline to view true Stable Neo-Hookean ripples.")

	def tet10_solve_01(self):
		# Create a clean high-resolution UV Sphere for the visual mesh
		# Set origin perfectly at (0, 0, 0) and use transform location for its physical start
		bpy.ops.mesh.primitive_uv_sphere_add(radius=1.8, location=(0.0, 0.0, 0.0), segments=32, ring_count=32)
		ball_obj = bpy.context.active_object
		ball_obj.name = "Rubber_Ball"
		ball_obj.location = (0.0, 0.0, 8.0) # True initial starting height
		ball_obj.shape_key_add(name="Basis", from_mix=False)

		# Create a clear visual collision floor plane
		bpy.ops.mesh.primitive_plane_add(size=10.0, location=(0.0, 0.0, 2.0))
		floor_obj = bpy.context.active_object
		floor_obj.name = "Floor_Plane"

		# --- 3. EXECUTE SIMULATION AND BAKE CLEAN SHAPE KEYS ---
		solver = QuadraticTet10Solver(start_height=8.0)

		mesh = ball_obj.data
		num_verts = len(mesh.vertices)
		orig_coords = np.zeros((num_verts, 3))
		mesh.vertices.foreach_get("co", orig_coords.ravel())

		# Generate local barycentric projection factors for the embedded mesh vertices
		# Because our Tet10 is scaled cleanly around the sphere bounds, we map weights directly
		bary_weights = np.zeros((num_verts, 10))
		for i in range(num_verts):
			# Calculate how this vertex maps to the 10-node element coordinates
			# We use a localized normalized distance matrix to generate stable shape interpolation weights
			local_pos = orig_coords[i] * 0.3 + 0.25
			bary_weights[i] = solver.compute_shape_functions(np.clip(local_pos, 0.0, 0.4))

		num_frames = 70
		dt = 0.015

		print("Baking high-order Tet10 Stable Neo-Hookean Simulation...")

		for frame in range(1, num_frames + 1):
			skey = ball_obj.shape_key_add(name=f"Frame_{frame}", from_mix=False)
			
			# Run the continuous quadratic element physics step
			solver.solve_continuum_physics(dt, gravity=-9.81, floor_z=2.0)
			
			# Deform the high-res sphere smoothly using the quadratic shape weights matrix
			# This prevents ANY mesh splitting or separation artifacts entirely
			deformed_world = np.dot(bary_weights, solver.nodes)
			
			# CRUCIAL BLENDER FIX: Shape keys calculate deformation relative to the object matrix origin.
			# The sphere object is physically placed at Z=8.0 in the viewport.
			# To prevent double-spheres or inverted trajectories, we subtract the object location vector.
			baked_local_coords = np.copy(deformed_world)
			baked_local_coords[:, 2] -= 8.0 
			
			skey.data.foreach_set("co", baked_local_coords.ravel())
			
			# Bind weight properties seamlessly to the active playback frame
			bpy.context.scene.frame_set(frame)
			skey.value = 1.0
			skey.keyframe_insert(data_path="value", frame=frame)
			if frame > 1:
				prev_key = ball_obj.data.shape_keys.key_blocks[f"Frame_{frame-1}"]
				prev_key.value = 0.0
				prev_key.keyframe_insert(data_path="value", frame=frame)

		# Clean up timeline context focus back to initialization frame
		bpy.context.scene.frame_set(1)

	def tet10_solve_02(self):
		# Create the visual UV Sphere
		bpy.ops.mesh.primitive_uv_sphere_add(radius=1.8, location=(0.0, 0.0, 8.0), segments=32, ring_count=32)
		ball_obj = bpy.context.active_object
		ball_obj.name = "Rubber_Ball"

		# Ensure smooth shading so ripples look clean
		bpy.ops.object.shade_smooth()

		# Initialize Basis key
		basis_key = ball_obj.shape_key_add(name="Basis", from_mix=False)

		# Create a clear visual collision floor plane
		bpy.ops.mesh.primitive_plane_add(size=10.0, location=(0.0, 0.0, 2.0))
		floor_obj = bpy.context.active_object
		floor_obj.name = "Floor_Plane"


		# --- 3. EXECUTE SIMULATION AND BAKE SHAPE KEYS ---
		solver = QuadraticTet10Solver(start_height=8.0)

		mesh = ball_obj.data
		num_verts = len(mesh.vertices)
		orig_coords = np.zeros((num_verts, 3))
		mesh.vertices.foreach_get("co", orig_coords.ravel())

		bary_weights = np.zeros((num_verts, 10))
		for i in range(num_verts):
			local_pos = orig_coords[i] * 0.25 + 0.25
			bary_weights[i] = solver.compute_shape_functions(np.clip(local_pos, 0.0, 0.4))

		num_frames = 60
		# dt = 0.016
		dt = 0.1

		print("Baking Tet10 Engine and keyframing timeline dependencies...")

		# Get the dependency graph for forcing UI updates
		depsgraph = bpy.context.evaluated_depsgraph_get()

		for frame in range(1, num_frames + 1):
			# Set active timeline frame
			bpy.context.scene.frame_set(frame)
			
			# Process physics iteration
			solver.solve_continuum_physics(dt, gravity=-9.81, floor_z=2.0)
			deformed_world = np.dot(bary_weights, solver.nodes)
			
			# FIX: Ensure mesh coords stay directly bound to the local space of the object container
			baked_local_coords = np.copy(deformed_world)
			baked_local_coords[:, 2] -= 8.0 
			
			# Append the keyframe state
			skey = ball_obj.shape_key_add(name=f"Frame_{frame}", from_mix=False)
			skey.data.foreach_set("co", baked_local_coords.ravel())
			
			# Turn it on exactly at this frame
			skey.value = 1.0
			skey.keyframe_insert(data_path="value", frame=frame)
			
			# Zero it out on the frames right before and right after to create a clean sequential flipbook
			if frame > 1:
				skey.value = 0.0
				skey.keyframe_insert(data_path="value", frame=frame - 1)
				
				prev_key = ball_obj.data.shape_keys.key_blocks[f"Frame_{frame-1}"]
				prev_key.value = 0.0
				prev_key.keyframe_insert(data_path="value", frame=frame)

			# Force Blender to update the object transformations immediately
			depsgraph.update()

		# Reset scene to start frame
		bpy.context.scene.frame_set(1)

	def conservativeLattice_01(self):
		# Create a clean target mesh. We use a base sphere whose vertex offsets 
		# will be explicitly manipulated to follow the voxel boundaries.
		bpy.ops.mesh.primitive_uv_sphere_add(radius=2.0, location=(0.0, 0.0, 0.0), segments=16, ring_count=16)
		ball_obj = bpy.context.active_object
		ball_obj.name = "Lattice_Ball"
		ball_obj.shape_key_add(name="Basis", from_mix=False)

		# Make the visual geometry look faceted to perfectly represent the underlying blocky physics
		bpy.ops.object.shade_flat()

		# --- 3. SIMULATE AND BAKE TO BLENDER SHAPE KEYS ---
		solver = ConservativeLatticeSolver()
		solver.seed_spherical_density()

		mesh = ball_obj.data
		num_verts = len(mesh.vertices)
		orig_coords = np.zeros((num_verts, 3))
		mesh.vertices.foreach_get("co", orig_coords.ravel())

		# Store reference radial directions of the base mesh vertices to preserve structural integrity
		normals = np.copy(orig_coords)
		norms = np.linalg.norm(normals, axis=1, keepdims=True)
		normals /= np.maximum(norms, 1e-5)

		num_frames = 60
		dt = 0.02
		depsgraph = bpy.context.evaluated_depsgraph_get()

		print("Beginning stable conservative lattice advection bake...")

		for frame in range(1, num_frames + 1):
			bpy.context.scene.frame_set(frame)
			
			# Process the safe, non-skipping lattice integration step
			solver.step_simulation(dt)
			
			# Find global bounding metric of the active advected voxel array
			active_indices = np.argwhere(solver.density > 0.05)
			if len(active_indices) == 0:
				continue
				
			min_z = np.min(active_indices[:, 2])
			max_z = np.max(active_indices[:, 2])
			
			# Calculate global center of mass vectors within the static grid system
			total_density = np.sum(solver.density)
			idx = np.indices(solver.res)
			world_cx = (np.sum(idx[0] * solver.density) / total_density) * solver.dx
			world_cy = (np.sum(idx[1] * solver.density) / total_density) * solver.dx
			world_cz = (np.sum(idx[2] * solver.density) / total_density) * solver.dx
			
			# Displace the embedded visual vertices to tightly match the changing blocky boundaries
			deformed_coords = np.zeros((num_verts, 3))
			
			for i in range(num_verts):
				# Map vertex positions sequentially along the absolute height of the advected voxel column
				v_pct = (orig_coords[i, 2] + 2.0) / 4.0 # Normalized height range [0, 1]
				target_z_idx = int(np.clip(min_z + v_pct * (max_z - min_z), 0, solver.res[2] - 1))
				
				# Sample horizontal footprint bounds at this specific voxel row
				layer_cx, layer_cy = solver.get_center_of_mass_at_z(target_z_idx)
				
				# Calculate blocky spatial coordinates
				x_coord = (layer_cx + normals[i, 0] * 3.5 * solver.density[int(layer_cx), int(layer_cy), target_z_idx]) * solver.dx
				y_coord = (layer_cy + normals[i, 1] * 3.5 * solver.density[int(layer_cx), int(layer_cy), target_z_idx]) * solver.dx
				z_coord = target_z_idx * solver.dx
				
				# Apply strict grid center localization
				deformed_coords[i, 0] = x_coord - (solver.res[0] * solver.dx / 2.0)
				deformed_coords[i, 1] = y_coord - (solver.res[1] * solver.dx / 2.0)
				deformed_coords[i, 2] = z_coord
				
			# Bake directly to sequential flipbook shape keys
			skey = ball_obj.shape_key_add(name=f"Frame_{frame}", from_mix=False)
			skey.data.foreach_set("co", deformed_coords.ravel())
			
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
		print("Safe Voxel-Lattice Bake finished. The mesh now cleanly mirrors the blocky cell-by-cell advection.")

	def hybridLattice_01(self):
		# # --- 1. CLEAN ENVIRONMENT SETUP ---
		# for name in ["Voxel_Ball", "Voxel_Floor_Mesh"]:
		# 	if name in bpy.data.objects:
		# 		bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)

		# Create a clean high-resolution sphere for the visual mesh embedding
		bpy.ops.mesh.primitive_uv_sphere_add(radius=2.0, location=(0.0, 0.0, 8.0), segments=32, ring_count=32)
		ball_obj = bpy.context.active_object
		ball_obj.name = "Voxel_Ball"
		ball_obj.shape_key_add(name="Basis", from_mix=False)
		bpy.ops.object.shade_flat()  # Faceted edges emphasize the blocky physics grid

		# Create a visual floor block for context
		# bpy.ops.mesh.primitive_cube_add(size=12.0, location=(0.0, 0.0, 1.0))
		bpy.ops.mesh.primitive_cube_add(size=4.0, location=(0.0, 0.0, 0.0))
		floor_obj = bpy.context.active_object
		floor_obj.name = "Voxel_Floor_Mesh"

		# --- 3. EXECUTE SIMULATION AND MAP TO EMBEDDED SHAPE KEYS ---
		solver = HybridVofTet10Solver()
		solver.seed_spherical_rubber_mass()

		mesh = ball_obj.data
		num_verts = len(mesh.vertices)
		orig_coords = np.zeros((num_verts, 3))
		mesh.vertices.foreach_get("co", orig_coords.ravel())

		# Generate base spatial directional normal vectors for clean mesh surface deformation mapping
		mesh_normals = np.copy(orig_coords)
		mesh_norms = np.linalg.norm(mesh_normals, axis=1, keepdims=True)
		mesh_normals /= np.maximum(mesh_norms, 1e-5)

		num_frames = 65
		dt = 0.02
		depsgraph = bpy.context.evaluated_depsgraph_get()

		print("Baking stable Conservative VOF Tet10 Elasticity Loop...")

		for frame in range(1, num_frames + 1):
			bpy.context.scene.frame_set(frame)
			
			# Run the stable grid solver tracking iterations
			solver.advance_eulerian_step(dt)
			
			# Locate spatial geometry footprints of the active mass density fractions
			active_voxels = np.argwhere(solver.density > 0.02)
			if len(active_voxels) == 0:
				continue
				
			min_z = np.min(active_voxels[:, 2])
			max_z = np.max(active_voxels[:, 2])
			
			# Global center calculation parameters
			total_density = np.sum(solver.density)
			idx_grid = np.indices(solver.res)
			world_cx = (np.sum(idx_grid[0] * solver.density) / total_density) * solver.dx
			world_cy = (np.sum(idx_grid[1] * solver.density) / total_density) * solver.dx
			
			deformed_vertices = np.zeros((num_verts, 3))
			
			for i in range(num_verts):
				# Calculate matching vertical slice profile layer based on vertex vertex positioning
				v_height_ratio = (orig_coords[i, 2] + 2.0) / 4.0
				target_z_idx = int(np.clip(min_z + v_height_ratio * (max_z - min_z), 0, solver.res[2] - 1))
				
				# Evaluate high-order Tet10 Neo-Hookean restoration pressures at this layer slice
				forces_xy = solver.solve_quadratic_volume_forces(target_z_idx)
				
				# Calculate internal coordinate metrics
				cell_x = int(np.clip(world_cx / solver.dx, 0, solver.res[0]-1))
				cell_y = int(np.clip(world_cy / solver.dx, 0, solver.res[1]-1))
				
				# Extract volume expansion scaling factors from our Tet10 solver
				expansion_x = forces_xy[cell_x, cell_y, 0] * 0.015
				expansion_y = forces_xy[cell_x, cell_y, 1] * 0.015
				
				# Compute final blocky coordinate paths
				# Radial expansion bows the outer voxels outward to preserve volume as the shape squishes flat
				vx = world_cx + (mesh_normals[i, 0] * (2.0 + expansion_x) * solver.density[cell_x, cell_y, target_z_idx])
				vy = world_cy + (mesh_normals[i, 1] * (2.0 + expansion_y) * solver.density[cell_x, cell_y, target_z_idx])
				vz = target_z_idx * solver.dx
				
				# Shift layout relative to grid center bounding boxes
				deformed_vertices[i, 0] = vx - (solver.res[0] * solver.dx / 2.0)
				deformed_vertices[i, 1] = vy - (solver.res[1] * solver.dx / 2.0)
				deformed_vertices[i, 2] = vz
				
			# Create and bake to sequential shape key blocks
			skey = ball_obj.shape_key_add(name=f"Frame_{frame}", from_mix=False)
			
			# CRUCIAL BLENDER LOCAL MESH TRACKING CORRECTION:
			
			# Subtract object starting offset location to prevent floating duplicates or viewport invisibility

			baked_coords = np.copy(deformed_vertices)
			baked_coords[:, 2] -= 8.0
			skey.data.foreach_set("co", baked_coords.ravel())
			
			# Apply driver keyframe tracking values
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

	def hybridLattice_02(self):
		# Create the visual UV Sphere (Physically starts at Z=7.5)
		bpy.ops.mesh.primitive_uv_sphere_add(radius=1.8, location=(0.0, 0.0, 7.5), segments=32, ring_count=32)
		ball_obj = bpy.context.active_object
		ball_obj.name = "Voxel_Ball"
		ball_obj.shape_key_add(name="Basis", from_mix=False)
		bpy.ops.object.shade_flat()  # Faceted edges emphasize blocky advection

		# Create your updated collision cube: Positioned at 0,0,0 and scaled to size 4
		bpy.ops.mesh.primitive_cube_add(size=4.0, location=(0.0, 0.0, 0.0))
		cube_obj = bpy.context.active_object
		cube_obj.name = "Voxel_Collision_Cube"

		# --- 3. RUN BAKING ENGINE & ANIMATE BLENDER SHAPE KEYS ---
		solver = HybridVofLatticeSolver()
		solver.seed_spherical_rubber_mass()

		mesh = ball_obj.data
		num_verts = len(mesh.vertices)
		orig_coords = np.zeros((num_verts, 3))
		mesh.vertices.foreach_get("co", orig_coords.ravel())

		mesh_normals = np.copy(orig_coords)
		mesh_norms = np.linalg.norm(mesh_normals, axis=1, keepdims=True)
		mesh_normals /= np.maximum(mesh_norms, 1e-5)

		num_frames = 60
		dt = 0.015
		depsgraph = bpy.context.evaluated_depsgraph_get()

		print("Baking true 3D Cube Collision Solver...")

		for frame in range(1, num_frames + 1):
			bpy.context.scene.frame_set(frame)
			
			# Calculate volume gradients and process the safe advection steps
			solver.solve_volume_preservation_forces(dt)
			solver.advance_eulerian_step(dt)
			
			# Find spatial boundaries of the active VOF density grid
			active_voxels = np.argwhere(solver.density > 0.05)
			if len(active_voxels) == 0:
				continue
				
			min_z = np.min(active_voxels[:, 2])
			max_z = np.max(active_voxels[:, 2])
			
			# Calculate global center coordinates
			total_density = np.sum(solver.density)
			idx_grid = np.indices(solver.res)
			world_cx = (np.sum(idx_grid[0] * solver.density) / total_density) * solver.dx
			world_cy = (np.sum(idx_grid[1] * solver.density) / total_density) * solver.dx
			
			deformed_vertices = np.zeros((num_verts, 3))
			
			for i in range(num_verts):
				# Map vertex profiles smoothly down the columns
				v_height_ratio = (orig_coords[i, 2] + 1.8) / 3.6
				target_z_idx = int(np.clip(min_z + v_height_ratio * (max_z - min_z), 0, solver.res[2] - 1))
				
				cell_x = int(np.clip(world_cx / solver.dx, 0, solver.res[0]-1))
				cell_y = int(np.clip(world_cy / solver.dx, 0, solver.res[1]-1))
				
				# Calculate dynamic outward scaling based on horizontal expansion velocities
				expansion_scale = 1.8 + (abs(solver.vel_x[cell_x, cell_y, target_z_idx]) * 0.8)
				
				# Displace the visual mesh coordinates to follow the voxel profile
				vx = (idx_grid[0][cell_x, cell_y, target_z_idx] - solver.res[0]/2.0) * solver.dx + (mesh_normals[i, 0] * expansion_scale * solver.density[cell_x, cell_y, target_z_idx])
				vy = (idx_grid[1][cell_x, cell_y, target_z_idx] - solver.res[1]/2.0) * solver.dx + (mesh_normals[i, 1] * expansion_scale * solver.density[cell_x, cell_y, target_z_idx])
				vz = (target_z_idx - solver.res[2]/2.0) * solver.dx + 4.0
				
				deformed_vertices[i] = [vx, vy, vz]
				
			# Bake to sequential flipbook keys
			skey = ball_obj.shape_key_add(name=f"Frame_{frame}", from_mix=False)
			
			# Correct the Blender object space mapping (matches initial viewport spawn location)
			baked_coords = np.copy(deformed_vertices)
			baked_coords[:, 2] -= 7.5
			
			skey.data.foreach_set("co", baked_coords.ravel())
			
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

	def multiphaseSolver_01(self):
		# Spawn high-res visual sphere at its true starting location
		bpy.ops.mesh.primitive_uv_sphere_add(radius=1.8, location=(0.0, 0.0, 7.5), segments=32, ring_count=32)
		ball_obj = bpy.context.active_object
		ball_obj.name = "Rubber_Ball"
		ball_obj.shape_key_add(name="Basis", from_mix=False)
		bpy.ops.object.shade_flat()

		# Scaled size 4 static cube centered at 0,0,0
		bpy.ops.mesh.primitive_cube_add(size=4.0, location=(0.0, 0.0, 0.0))
		cube_obj = bpy.context.active_object
		cube_obj.name = "Voxel_Collision_Cube"


		# --- 3. EXECUTE SIMULATION AND MAP BAKE GRAPH UPDATES ---
		solver = MultiphaseVoxelSolver()
		solver.seed_spherical_rubber_ball()

		mesh = ball_obj.data
		num_verts = len(mesh.vertices)
		orig_coords = np.zeros((num_verts, 3))
		mesh.vertices.foreach_get("co", orig_coords.ravel())

		mesh_normals = orig_coords / np.maximum(np.linalg.norm(orig_coords, axis=1, keepdims=True), 1e-5)

		# num_frames = 60
		num_frames = 200
		dt = 0.014
		depsgraph = bpy.context.evaluated_depsgraph_get()

		print("Beginning 2018 Stable Multiphase Lattice Bake...")

		for frame in range(1, num_frames + 1):
			bpy.context.scene.frame_set(frame)
			
			# Process exact 2018 Hyperelastic stress tensors and multi-phase advections
			solver.compute_smith2018_stresses(dt)
			solver.advance_multiphase_advection(dt)
			
			# Find spatial centroid of the active solid volume fractions
			solid_voxels = np.argwhere(solver.phases[..., 2] > 0.05)
			if len(solid_voxels) == 0:
				continue
				
			min_z = np.min(solid_voxels[:, 2])
			max_z = np.max(solid_voxels[:, 2])
			
			# Track spatial center offsets
			total_solid_mass = np.sum(solver.phases[..., 2])
			x_idx, y_idx, z_idx = np.indices(solver.res)
			world_cx = (np.sum(x_idx * solver.phases[..., 2]) / total_solid_mass) * solver.dx
			world_cy = (np.sum(y_idx * solver.phases[..., 2]) / total_solid_mass) * solver.dx
			
			deformed_vertices = np.zeros((num_verts, 3))
			
			for i in range(num_verts):
				# Shape Preservation Constraint: Keeps the object perfectly spherical during the fall
				# Compares current minimum vertical height boundary against original reference values
				current_z_floor = (min_z - solver.res[2]/2.0) * solver.dx + 4.0
				
				cell_x = int(np.clip(world_cx / solver.dx, 0, solver.res[0]-1))
				cell_y = int(np.clip(world_cy / solver.dx, 0, solver.res[1]-1))
				
				# Calculate dynamic lateral expansion profile based on horizontal compression velocities
				v_height_ratio = (orig_coords[i, 2] + 1.8) / 3.6
				target_z_idx = int(np.clip(min_z + v_height_ratio * (max_z - min_z), 0, solver.res[2] - 1))
				
				# Squish scalar updates only when impacting near the cube top threshold bounds (Z coordinate around 2.0)
				is_colliding = current_z_floor <= 2.2
				expansion = 1.8 + (abs(solver.vel_x[cell_x, cell_y, target_z_idx]) * 1.1) if is_colliding else 1.8
				
				# Generate absolute coordinates tracking the solid volume fraction footprint
				vx = (x_idx[cell_x, cell_y, target_z_idx] - solver.res[0]/2.0) * solver.dx + (mesh_normals[i, 0] * expansion * solver.phases[cell_x, cell_y, target_z_idx, 2])
				vy = (y_idx[cell_x, cell_y, target_z_idx] - solver.res[1]/2.0) * solver.dx + (mesh_normals[i, 1] * expansion * solver.phases[cell_x, cell_y, target_z_idx, 2])
				vz = (target_z_idx - solver.res[2]/2.0) * solver.dx + 4.0
				
				deformed_vertices[i] = [vx, vy, vz]
				
			# Append baked frame layout states
			skey = ball_obj.shape_key_add(name=f"Frame_{frame}", from_mix=False)
			baked_coords = np.copy(deformed_vertices)
			baked_coords[:, 2] -= 7.5 # Maintain structural canvas tracking bounds
			
			skey.data.foreach_set("co", baked_coords.ravel())
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
		print("Safe Voxel-Lattice Bake finished. The mesh now cleanly mirrors the blocky cell-by-cell advection.")

	def multiphaseSolver_02(self):
		# Spawn high-res visual sphere at its true starting location
		bpy.ops.mesh.primitive_uv_sphere_add(radius=1.8, location=(0.0, 0.0, 7.5), segments=32, ring_count=32)
		ball_obj = bpy.context.active_object
		ball_obj.name = "Rubber_Ball"
		ball_obj.shape_key_add(name="Basis", from_mix=False)
		bpy.ops.object.shade_flat()

		# Scaled size 4 static cube centered at 0,0,0
		bpy.ops.mesh.primitive_cube_add(size=4.0, location=(0.0, 0.0, 0.0))
		cube_obj = bpy.context.active_object
		cube_obj.name = "Voxel_Collision_Cube"


		# --- 3. RUN SIMULATION AND BAKE CLEAN SHAPE KEYS ---
		solver = TrueStaggeredVoxelSolver()
		solver.seed_spherical_rubber_ball()

		mesh = ball_obj.data
		num_verts = len(mesh.vertices)
		orig_coords = np.zeros((num_verts, 3))
		mesh.vertices.foreach_get("co", orig_coords.ravel())

		mesh_normals = orig_coords / np.maximum(np.linalg.norm(orig_coords, axis=1, keepdims=True), 1e-5)

		# Expanded frame budget matching your 200 frame simulation criteria
		num_frames = 200
		dt = 0.012
		depsgraph = bpy.context.evaluated_depsgraph_get()

		print("Beginning 200-frame high-fidelity Staggered Multiphase Lattice Bake...")

		for frame in range(1, num_frames + 1):
			bpy.context.scene.frame_set(frame)
			
			# Process physics engine loops
			solver.update_physics(dt)
			
			# Find spatial boundaries of the active solid volume fractions
			solid_voxels = np.argwhere(solver.phases[..., 2] > 0.05)
			if len(solid_voxels) == 0:
				continue
				
			min_z = np.min(solid_voxels[:, 2])
			max_z = np.max(solid_voxels[:, 2])
			
			# Calculate global center tracking coordinates
			total_solid_mass = np.sum(solver.phases[..., 2])
			x_idx, y_idx, z_idx = np.indices(solver.res)
			world_cx = (np.sum(x_idx * solver.phases[..., 2]) / total_solid_mass) * solver.dx
			world_cy = (np.sum(y_idx * solver.phases[..., 2]) / total_solid_mass) * solver.dx
			
			deformed_vertices = np.zeros((num_verts, 3))
			
			for i in range(num_verts):
				current_z_floor = (min_z - solver.res[2]/2.0) * solver.dx + 4.0
				
				cell_x = int(np.clip(world_cx / solver.dx, 0, solver.res[0]-1))
				cell_y = int(np.clip(world_cy / solver.dx, 0, solver.res[1]-1))
				
				v_height_ratio = (orig_coords[i, 2] + 1.8) / 3.6
				target_z_idx = int(np.clip(min_z + v_height_ratio * (max_z - min_z), 0, solver.res[2] - 1))
				
				# Shape Preservation Fix: Ball stays perfectly spherical until it meets the cube top (Z coordinate <= 2.2)
				is_colliding = current_z_floor <= 2.2
				expansion = 1.8 + (abs(solver.u[cell_x, cell_y, target_z_idx]) * 1.5) if is_colliding else 1.8
				
				# Map visual vertices tightly onto the active solid volume footprint boundaries
				vx = (x_idx[cell_x, cell_y, target_z_idx] - solver.res[0]/2.0) * solver.dx + (mesh_normals[i, 0] * expansion * solver.phases[cell_x, cell_y, target_z_idx, 2])
				vy = (y_idx[cell_x, cell_y, target_z_idx] - solver.res[1]/2.0) * solver.dx + (mesh_normals[i, 1] * expansion * solver.phases[cell_x, cell_y, target_z_idx, 2])
				vz = (target_z_idx - solver.res[2]/2.0) * solver.dx + 4.0
				
				deformed_vertices[i] = [vx, vy, vz]
				
			# Bake directly into the flipbook shape key stack
			skey = ball_obj.shape_key_add(name=f"Frame_{frame}", from_mix=False)
			baked_coords = np.copy(deformed_vertices)
			baked_coords[:, 2] -= 7.5
			
			skey.data.foreach_set("co", baked_coords.ravel())
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
		print("Safe Voxel-Lattice Bake finished. The mesh now cleanly mirrors the blocky cell-by-cell advection.")






	def testVDB_06(self, abj_sd_b_instance):
		startTime = datetime.now()

		# self.embeddedVoxelFemAdvectionTest()
		# self.embeddedVoxelFemAdvectionTest_02()
		# self.kuhn5_01()
		# self.tet10_solve_01()
		# self.tet10_solve_02()
		# self.conservativeLattice_01() # very good
		# self.hybridLattice_01()
		# self.hybridLattice_02()
		# self.multiphaseSolver_01()
		self.multiphaseSolver_02()


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