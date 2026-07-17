'''
Copyright (C) 2025 Aleksander Berg-Jones

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 3
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, see <https://www.gnu.org/licenses>.
''' 

import bpy
import bmesh
import math
import mathutils
from datetime import datetime
import random
import numpy as np
import importlib
import sys
import copy

from . import simple_spec_abj
from . import GGX_hable_abj
from . import spectral
from . import spectral3_glsl

if "bpy" in locals():
	prefix = __package__ + '.'
	for name, module in sys.modules.copy().items():
		if name.startswith(prefix):
			basename = name.removeprefix(prefix)
			globals()[basename] = importlib.reload(module)

import bpy

from .GGX_hable_abj import myEquation_GGX
from .simple_spec_abj import myEquation_simple_spec

myEquation_GGX_class = myEquation_GGX()
myEquation_simple_spec_class = myEquation_simple_spec()

class ABJ_Shader_Debugger():
	def __init__(self):
		self.renderPasses_simple = False
		self.renderPasses_GGX = False

		self.useRestoredRxyzValues = True
		self.breakEarlyForRandomLightAndRxyz = False

		self.chosen_diffuse_equation = 'oren'
		# self.chosen_specular_equation = 'simple'
		self.chosen_specular_equation = 'GGX'

		self.colorspace_18_hue_list = []
		self.distanceFromCam_all_list = []
		self.distanceFromCam_raycastRenderable_list = []
		self.colorspace_18_continued_j = 0
		self.colorWheelGradient_18_created = False

		self.compositor_setup = False
		self.chosen_text_rgb_precision = '0'
		self.text_radius_0_stored = 0.005
		self.text_radius_1_stored = 0.6

		self.text_rotate_x_stored = 0.0
		self.text_rotate_y_stored = 0.0
		self.text_rotate_z_stored = 0.0

		self.text_gradient_rotate_x_stored = 0.0
		self.text_gradient_rotate_y_stored = 0.0
		self.text_gradient_rotate_z_stored = 0.0

		
		self.val_gradient_circle_override = 0
		self.val_gradient_circle_override_side = None


		self.myLoc_arrow_01 = None

		#spectral compositor
		self.node_spectral_gamma = None
		self.node_spectral_epsilon = None
		self.node_spectral_size = None





		self.nodeOut = None
		self.nodeViewer = None

		self.cam1_debug = None
		self.cam2_debug = None
		self.vector_toEuler_debug = None

		self.changedSpecularEquation_variables = False
		self.changedDiffuseEquation_variables = False
		self.skip_showing_visibility_raycast_check = True

		self.Ci_render_temp_list = []
		self.selectedFaceMat_temp_list = []

		self.myPixel = None
		self.fov = None
		self.aspect = None
		self.zNear = None
		self.zFar = None
		self.zFarScale = 1.53

		self.runOnce_part2_preProcess = False
		self.debugV_223 = None
		self.myDebugFaces = []
		self.allNamesToToggleDuringRaycast = []
		self.arrow_dynamic_instance_M_all_list_matrixOnly = []
		self.arrow_dynamic_instance_M_all_list_matrixOnly_debug = []
		self.shadingDict_global = {}
		self.shadingList_perFace = []
		self.shadingStages_all = []
		self.shadingStages_perFace_stepList = []
		self.shadingStages_selectedFaces = []
		self.textRef_all = []
		self.objectsToToggleOnOffLater = []
		self.debugStageIterPlusMinus = False
		self.recently_cleared_selFaces = False

		self.aov_stored = None
		self.rdotvpow_stored = None
		self.spectralmix_stored = None
		self.spectralmix2_stored = None
		self.oren_roughness_stored = None
		self.ggx_roughness_stored = None
		self.ggx_fresnel_stored = None

		self.is_metallic_stored = None
		self.aniso_specular_stored = None
		self.aniso_rotation_stored = None
		self.aniso_roughnessX_stored = None
		self.aniso_roughnessY_stored = None
		self.use_18_hue_colorspace_prop_stored = None



		self.breakpointsOverrideToggle = False
		self.skip_refresh_override_aov = False
		self.skip_refresh_override_RdotVpow = False
		self.skip_refresh_override_spectral = False
		self.skip_refresh_override_spectral2 = False
		self.skip_refresh_override_oren_roughness = False
		self.skip_refresh_override_GGX_roughness = False
		self.skip_refresh_override_GGX_fresnel = False

		self.skip_refresh_override_is_metallic = False
		self.skip_refresh_override_aniso_specular = False
		self.skip_refresh_override_aniso_rotation = False
		self.skip_refresh_override_aniso_roughnessX = False
		self.skip_refresh_override_aniso_roughnessY = False
		self.skip_refresh_override_use_18_hue_colorspace = False


		self.shadingPlane_sel_r = 0.0
		self.shadingPlane_sel_g = 0.0
		self.shadingPlane_sel_b = 1.0

		self.circular_gradient_text_counter = 0

		self.world_mat_cam_stored_np = None
		self.shadingPlane = None
		self.len_cam_arrow = 1
		self.len_light_arrow = 4
		self.len_arrow_frontExtend = 1.25
		self.scaleCamArrows = 0.5
		self.arrowWidth = 0.025
		self.len_reg_arrowExtendPos = -18
		self.scaleRegArrows = 0.25
		self.arrow_wings = .75
		self.myOrigin = mathutils.Vector((0, 0, 0))

		self.diffuse_or_emission_og_shading = 'diffuse'
		# self.diffuse_or_emission_og_shading = 'emission'
		self.adjustedColors = False

		#instance
		##########
		## NEW
		##########
		self.myCubeV_halfFov_LB_og = None
		self.myCubeV_halfFov_LB_og_Matrix = None
		self.myCubeV_halfFov_LB_og_Matrix_np = None
		self.myCubeV_halfFov_LB_og_dupe = None

		self.myCubeV_halfFov_LT_og = None
		self.myCubeV_halfFov_LT_og_Matrix = None
		self.myCubeV_halfFov_LT_og_Matrix_np = None
		self.myCubeV_halfFov_LT_og_dupe = None

		self.myCubeV_halfFov_RT_og = None
		self.myCubeV_halfFov_RT_og_Matrix = None
		self.myCubeV_halfFov_RT_og_Matrix_np = None
		self.myCubeV_halfFov_RT_og_dupe = None

		self.myCubeV_halfFov_RB_og = None
		self.myCubeV_halfFov_RB_og_Matrix = None
		self.myCubeV_halfFov_RB_og_Matrix_np = None
		self.myCubeV_halfFov_RB_og_dupe = None


		########
		## OLD
		########
		self.myCubeV_halfFov_L_og = None
		self.myCubeV_halfFov_L_og_Matrix = None
		self.myCubeV_halfFov_L_og_Matrix_np = None
		self.myCubeV_halfFov_L_og_dupe = None

		self.myCubeV_halfFov_R_og = None
		self.myCubeV_halfFov_R_og_Matrix = None
		self.myCubeV_halfFov_R_og_Matrix_np = None
		self.myCubeV_halfFov_R_og_dupe = None

		self.dynamicM_halfFov_LB = None
		self.dynamicM_halfFov_LT = None
		self.dynamicM_halfFov_RT = None
		self.dynamicM_halfFov_RB = None

		self.dynamicM_halfFov_LB_vector = None
		self.dynamicM_halfFov_LT_vector = None
		self.dynamicM_halfFov_RT_vector = None
		self.dynamicM_halfFov_RB_vector = None

		self.myCubeLight_og = None
		self.myCubeLight_og_Matrix = None
		self.myCubeLight_og_Matrix_np = None
		self.myCubeLight_dupe = None

		self.myCubeH_og = None
		self.myCubeH_og_Matrix = None
		self.myCubeH_og_Matrix_np = None
		self.myCubeH_dupe = None

		self.myCubeN_og = None
		self.myCubeN_og_Matrix = None
		self.myCubeN_og_Matrix_np = None
		self.myCubeN_dupe = None
		
		self.myCubeR_og = None
		self.myCubeR_og_Matrix = None
		self.myCubeR_dupe = None
		
		self.myCubeCam = None
		self.myCubeCam_og_Matrix = None
		self.myCubeCam_og_Matrix_np = None
		self.myCubeCam_dupe = None

		self.mySun = None
		self.myV = None

		self.specTester_instanceGrp = None
		self.startTime_stage1 = None
		self.startTime_stage2 = None

		#############
		##### PROFILE
		#############
		self.profileCode_part1 = False
		
		self.profile_stage1_00_a = None
		self.profile_stage1_00_b = None
		self.profile_stage1_00_final = None

		self.profile_stage1_01_a = None
		self.profile_stage1_01_b = None
		self.profile_stage1_01_final = None

		self.profile_stage1_02_a = None
		self.profile_stage1_02_b = None
		self.profile_stage1_02_final = None

		self.profile_stage1_03_a = None
		self.profile_stage1_03_b = None
		self.profile_stage1_03_final = None

		self.profile_stage1_04_a = None
		self.profile_stage1_04_b = None
		self.profile_stage1_04_final = None

		self.profile_stage1_05_a = None
		self.profile_stage1_05_b = None
		self.profile_stage1_05_final = None

		self.profile_stage1_06_a = None
		self.profile_stage1_06_b = None
		self.profile_stage1_06_final = None

		self.profile_stage1_07_a = None
		self.profile_stage1_07_b = None
		self.profile_stage1_07_final = None

		self.profile_stage1_08_a = None
		self.profile_stage1_08_b = None
		self.profile_stage1_08_final = None

		self.profile_stage1_09_a = None
		self.profile_stage1_09_b = None
		self.profile_stage1_09_final = None

		self.profile_stage1_10_a = None
		self.profile_stage1_10_b = None
		self.profile_stage1_10_final = None

		##########
		self.profileCode_part2 = False

		self.profile_stage2_00_a = None
		self.profile_stage2_00_b = None
		self.profile_stage2_00_final = None

		self.profile_stage2_01_a = None
		self.profile_stage2_01_b = None
		self.profile_stage2_01_final = None

		self.profile_stage2_02_a = None
		self.profile_stage2_02_b = None
		self.profile_stage2_02_final = None

		self.profile_stage2_03_a = None
		self.profile_stage2_03_b = None
		self.profile_stage2_03_final = None

		self.profile_stage2_04_a = None
		self.profile_stage2_04_b = None
		self.profile_stage2_04_final = None

		self.profile_stage2_05_a = None
		self.profile_stage2_05_b = None
		self.profile_stage2_05_final = None

		self.profile_stage2_06_a = None
		self.profile_stage2_06_b = None
		self.profile_stage2_06_final = None

		self.profile_stage2_07_a = None
		self.profile_stage2_07_b = None
		self.profile_stage2_07_final = None

		self.profile_stage2_08_a = None
		self.profile_stage2_08_b = None
		self.profile_stage2_08_final = None

		self.profile_stage2_09_a = None
		self.profile_stage2_09_b = None
		self.profile_stage2_09_final = None

		self.profile_stage2_10_a = None
		self.profile_stage2_10_b = None
		self.profile_stage2_10_final = None

		self.text_extrude_amt = 0.05

		self.printDetailedInfo = True

		self.pos_camera_global = (5, 5, 5)
		self.pos_camera_global_v = mathutils.Vector((self.pos_camera_global[0], self.pos_camera_global[1], self.pos_camera_global[2]))
		# self.pos_light_global =  (0.766, 0.836, 0.427)
		self.pos_light_global = (0.184085, 1.99077, 0.427)
		self.pos_light_global_toRestore = (0.184085, 1.99077, 0.427) 

		# self.pos_light_global =  (-0.08, 0.675, 0.327) # debug
		# self.pos_light_global = (0, 0, 5) ############################## overhead
		# self.pos_light_global = (0, 10, 0) ####
		# self.pos_light_global = (0, 0, 10) ####
		# self.pos_light_global = (10, 10, 0)
		# self.pos_light_global = (-10, -10, -10)
		self.pos_light_global_v = mathutils.Vector((self.pos_light_global[0], self.pos_light_global[1], self.pos_light_global[2]))

		self.RandomRotationAxis = 'X'
		self.RandomRotationDegree = 0

		self.myBreakpointList = []

		self.myBreakpointList.append('breakpoint_000_enum_prop')
		self.myBreakpointList.append('breakpoint_001_enum_prop')
		self.myBreakpointList.append('breakpoint_002_enum_prop')
		self.myBreakpointList.append('breakpoint_003_enum_prop')
		self.myBreakpointList.append('breakpoint_004_enum_prop')
		self.myBreakpointList.append('breakpoint_005_enum_prop')
		self.myBreakpointList.append('breakpoint_006_enum_prop')
		self.myBreakpointList.append('breakpoint_007_enum_prop')
		self.myBreakpointList.append('breakpoint_008_enum_prop')
		self.myBreakpointList.append('breakpoint_009_enum_prop')
		self.myBreakpointList.append('breakpoint_010_enum_prop')
		self.myBreakpointList.append('breakpoint_011_enum_prop')
		self.myBreakpointList.append('breakpoint_012_enum_prop')
		self.myBreakpointList.append('breakpoint_013_enum_prop')
		self.myBreakpointList.append('breakpoint_014_enum_prop')
		self.myBreakpointList.append('breakpoint_015_enum_prop')
		self.myBreakpointList.append('breakpoint_016_enum_prop')
		self.myBreakpointList.append('breakpoint_017_enum_prop')
		self.myBreakpointList.append('breakpoint_018_enum_prop')
		self.myBreakpointList.append('breakpoint_019_enum_prop')
		self.myBreakpointList.append('breakpoint_020_enum_prop')
		self.myBreakpointList.append('breakpoint_021_enum_prop')
		self.myBreakpointList.append('breakpoint_022_enum_prop')
		self.myBreakpointList.append('breakpoint_023_enum_prop')
		self.myBreakpointList.append('breakpoint_024_enum_prop')
		self.myBreakpointList.append('breakpoint_025_enum_prop')

	def look_at(self, obj_camera, point):
		loc_camera = obj_camera.matrix_world.to_translation()

		direction = point - loc_camera

		# point the cameras '-Z' and use its 'Y' as up
		rot_quat = direction.to_track_quat('-Z', 'Y')

		# assume we're using euler rotation
		obj_camera.rotation_euler = rot_quat.to_euler()

	def look_at2(self, obj_camera, point):
		loc_camera = obj_camera.matrix_world.to_translation()

		direction = point - loc_camera
		direction.normalize()

		rot_quat = direction.to_track_quat('X', 'Z')

		# assume we're using euler rotation
		obj_camera.rotation_euler = rot_quat.to_euler()

	def refreshPart2_UI(self):
		self.doIt_part2_render()
		bpy.ops.object.mode_set(mode="OBJECT")

	##STAGES
	def clamp(self, value, minimum, maximum):
		"""Clamps the value between the minimum and maximum values."""
		return max(minimum, min(value, maximum))

	def stages_selectfaces_UI(self):
		for i in bpy.context.selected_objects:
			for j in self.shadingStages_perFace_stepList:
				if j["shadingPlane"] == i.name:
					self.shadingStages_selectedFaces.append(j['idx'])

	def batchIdentifyBreakpointValue(self, idx, step, passedIdx):
		breakpointLimit = 25

		storedValueUsable_prev = None

		toUse = self.myBreakpointList[idx]
		combo = 'bpy.context.scene.' + toUse

		items = bpy.context.scene.bl_rna.properties[toUse].enum_items
		items_id = items[eval(combo)].identifier

		usableNextStage = None

		for j in self.shadingStages_perFace_stepList:

			if j['idx'] == passedIdx:
				maxRange_usable = None

				if self.chosen_specular_equation == 'GGX':
					maxRange_usable = 17

				if self.chosen_specular_equation == 'simple':

					if self.renderPasses_simple == True:
						maxRange_usable = 4

					if self.skip_showing_visibility_raycast_check == True:
						maxRange_usable = 2
					else:
						maxRange_usable = 7

				######################################################
				#get the breakpoint
				#get the enum value chosen for that breakpoint
				#use that to get the current stage

				j["breakpoint_idx"] = self.clamp(j["breakpoint_idx"] + step, 0, breakpointLimit + 1)
				
				nextStage = self.myBreakpointList[j['breakpoint_idx']]

				combo_nextStage = 'bpy.context.scene.' + nextStage
				items_nextStage = bpy.context.scene.bl_rna.properties[nextStage].enum_items
				items_id_nextStage = items_nextStage[eval(combo_nextStage)].identifier

				usableNextStage = int(items_id_nextStage)
				usableNextStageClamped = self.clamp(usableNextStage, 0, maxRange_usable)
				j["stage"] = usableNextStageClamped
				break

		return usableNextStage

	def stageIdx_plusMinus_UI(self, step):
		usableBreakpointOverride_items = bpy.context.scene.bl_rna.properties['breakpoint_override_enum_prop'].enum_items
		usableBreakpointOverride_id = usableBreakpointOverride_items[bpy.context.scene.breakpoint_override_enum_prop].identifier

		if usableBreakpointOverride_id == 'regular':

			for i in self.shadingStages_selectedFaces:
				for j in self.shadingStages_perFace_stepList:	
					if j["idx"] == i: #mySplitFaceIndexUsable:
						myNextStage = self.batchIdentifyBreakpointValue(j["breakpoint_idx"], step, i)

		elif usableBreakpointOverride_id == 'override':
			pass
			# iterate by 1 without having to change order

		if self.debugStageIterPlusMinus == False:
			self.refreshPart2_UI()

	def stageIdx_zero_UI(self):
		for i in bpy.context.selected_objects:
			for j in self.shadingStages_perFace_stepList:
				if j["shadingPlane"] == i.name:
					j["stage"] = 0

					print('j[stage] = ', j['stage'])

		self.refreshPart2_UI()

	def stage_resetAll_UI(self):
		for i in self.shadingStages_selectedFaces:
			for j in self.shadingStages_perFace_stepList:
				if j['idx'] == i:
					for k in bpy.context.scene.objects:
						if k.name == j["shadingPlane"]:
							bpy.context.view_layer.objects.active = k
							bpy.context.active_object.data.materials.clear()
					j["breakpoint_idx"] = 0
					j["stage"] = 0

		self.shadingStages_selectedFaces.clear()

		self.recently_cleared_selFaces = True

		self.refreshPart2_UI()


	def remap_range(self, value, old_min, old_max, new_min, new_max):
		return new_min + (value - old_min) * (new_max - new_min) / (old_max - old_min)

		# value = 5
		# old_min = 0
		# old_max = 10
		# new_min = 0
		# new_max = 100

		# new_value = remap_range(value, old_min, old_max, new_min, new_max)
		# print(new_value) # 50.0

	def written_manual_cross(self, vector_a, vector_b):
		# Define two 3D vectors as lists
		# vector_a = [1, 2, 3]
		# vector_b = [4, 5, 6]

		# vector_a = mathutils.Vector((.8, .2, .3))
		# vector_b = mathutils.Vector((.5, .5, .5))

		# Calculate the components of the cross product manually
		cross_product_x = (vector_a[1] * vector_b[2]) - (vector_a[2] * vector_b[1])
		cross_product_y = (vector_a[2] * vector_b[0]) - (vector_a[0] * vector_b[2])
		cross_product_z = (vector_a[0] * vector_b[1]) - (vector_a[1] * vector_b[0])

		# Store the result in a new list
		# cross_product_vector = [cross_product_x, cross_product_y, cross_product_z]

		cross_product_vector = mathutils.Vector((cross_product_x, cross_product_y, cross_product_z))

		return cross_product_vector

	def written_manual_dotProduct(self, vec1, vec2):
		# vector_a = [1, 2, 3]
		# vector_b = [4, 5, 6]

		written_dot_product_result = 0
		for i in range(len(vec1)):
			written_dot_product_result += vec1[i] * vec2[i]
		return written_dot_product_result

		myBPY_result_vec0 = mathutils.Vector((.8, .2, .3))
		myBPY_result_vec1 = mathutils.Vector((.5, .5, .5))

		myResult = mathutils.Vector.dot(myBPY_result_vec0, myBPY_result_vec1)
		print('my bpy dot product result = ', myResult)
		
		#manual
		len0 = .8 * .5
		len1 = .2 * .5
		len2 = .3 * .5

		comboDot = len0 + len1 + len2
		print('my manual dot product result = ', comboDot)

	def written_reflect(self, incident_vector, normal_vector):
		"""
		Calculates the reflection of an incident vector across a normal vector.

		Args:
			incident_vector (np.array): The vector representing the incoming ray.
			normal_vector (np.array): The normal vector of the surface at the point of reflection.
									This vector should be normalized (unit length).

		Returns:
			np.array: The reflected vector.
		"""
		# Ensure normal_vector is normalized
		# normal_vector = normal_vector / np.linalg.norm(normal_vector)

		normal_vector = self.abjNormalize_written(normal_vector)

		# The reflection formula: R = I - 2 * (I . N) * N
		# where I is the incident vector, N is the normal vector, and . is the dot product.
		# dot_product = np.dot(incident_vector, normal_vector)
		dot_product = self.written_manual_dotProduct(incident_vector, normal_vector)
		reflected_vector = incident_vector - 2 * dot_product * normal_vector
		return reflected_vector
	
	#This method calls a function from spectral3_glsl.py, under MIT license (see file)
	def abj_spectral_multiblend(self):
		for area in bpy.data.screens["Layout"].areas:
		# for area in bpy.data.screens[bpy.context.window.screen].areas:
			if area.type == 'VIEW_3D':
				for space in area.spaces:
					if space.type == 'VIEW_3D':

						space.shading.type = 'MATERIAL'

						usableToggle = None
						if space.overlay.show_floor == True:
							usableToggle = False
						else:
							usableToggle = True

						usableToggle = False

						space.overlay.show_axis_x = usableToggle
						space.overlay.show_axis_y = usableToggle
						space.overlay.show_axis_z = usableToggle
						space.overlay.show_cursor = usableToggle
						space.overlay.show_floor = usableToggle

		bpy.context.scene.view_settings.view_transform = 'Standard'
		# bpy.context.scene.view_settings.look = 'AgX - Punchy'
		bpy.context.scene.view_settings.look = 'None'
		bpy.context.scene.render.use_multiview = False
	
		val_spectral_multi_0_Blend_prop = bpy.context.scene.spectral_multi_0_Blend_prop
		val_spectral_multi_0_Tint_prop = bpy.context.scene.spectral_multi_0_Tint_prop
		val_spectral_multi_0_Factor_prop = bpy.context.scene.spectral_multi_0_Factor_prop

		val_spectral_multi_1_Blend_prop = bpy.context.scene.spectral_multi_1_Blend_prop
		val_spectral_multi_1_Tint_prop = bpy.context.scene.spectral_multi_1_Tint_prop
		val_spectral_multi_1_Factor_prop = bpy.context.scene.spectral_multi_1_Factor_prop

		val_spectral_multi_2_Blend_prop = bpy.context.scene.spectral_multi_2_Blend_prop
		val_spectral_multi_2_Tint_prop = bpy.context.scene.spectral_multi_2_Tint_prop
		val_spectral_multi_2_Factor_prop = bpy.context.scene.spectral_multi_2_Factor_prop

		val_spectral_multi_3_Blend_prop = bpy.context.scene.spectral_multi_3_Blend_prop
		val_spectral_multi_3_Tint_prop = bpy.context.scene.spectral_multi_3_Tint_prop
		val_spectral_multi_3_Factor_prop = bpy.context.scene.spectral_multi_3_Factor_prop

		#tri color blending
		allOutputRatios = []




		usableSpectralMultiblendEquation_items = bpy.context.scene.bl_rna.properties['spectral_multiblend_equation_enum_prop'].enum_items
		usableSpectralMultiblendEquation_id = usableSpectralMultiblendEquation_items[bpy.context.scene.spectral_multiblend_equation_enum_prop].identifier

		# if usableSpectralMultiblendEquation_id == 2:
		# 	print('2 number')
		# elif usableSpectralMultiblendEquation_id == '2':
		# 	print('2 string')

		# return

		if usableSpectralMultiblendEquation_id == '2':
			outputColorTriBlend = spectral3_glsl.spectral_mix2(val_spectral_multi_0_Blend_prop, val_spectral_multi_0_Tint_prop, val_spectral_multi_0_Factor_prop, val_spectral_multi_1_Blend_prop, val_spectral_multi_1_Tint_prop, val_spectral_multi_1_Factor_prop)

		elif usableSpectralMultiblendEquation_id == '3':
			outputColorTriBlend = spectral3_glsl.spectral_mix3(val_spectral_multi_0_Blend_prop, val_spectral_multi_0_Tint_prop, val_spectral_multi_0_Factor_prop, val_spectral_multi_1_Blend_prop, val_spectral_multi_1_Tint_prop, val_spectral_multi_1_Factor_prop, val_spectral_multi_2_Blend_prop, val_spectral_multi_2_Tint_prop, val_spectral_multi_2_Factor_prop)

		elif usableSpectralMultiblendEquation_id == '4':
			outputColorTriBlend = spectral3_glsl.spectral_mix4(val_spectral_multi_0_Blend_prop, val_spectral_multi_0_Tint_prop, val_spectral_multi_0_Factor_prop, val_spectral_multi_1_Blend_prop, val_spectral_multi_1_Tint_prop, val_spectral_multi_1_Factor_prop, val_spectral_multi_2_Blend_prop, val_spectral_multi_2_Tint_prop, val_spectral_multi_2_Factor_prop, val_spectral_multi_3_Blend_prop, val_spectral_multi_3_Tint_prop, val_spectral_multi_3_Factor_prop)

		#gamma correct
		val_gamma_correct_gradient_color_prop = bpy.context.scene.gamma_correct_gradient_color_prop

		if val_gamma_correct_gradient_color_prop == True:
			gammaCorrect = mathutils.Vector((2.2, 2.2, 2.2))
			gammaCorrect_r = pow(outputColorTriBlend.x, gammaCorrect.x)
			gammaCorrect_g = pow(outputColorTriBlend.y, gammaCorrect.y)
			gammaCorrect_b = pow(outputColorTriBlend.z, gammaCorrect.z)

			outputColorTriBlend = mathutils.Vector((gammaCorrect_r, gammaCorrect_g, gammaCorrect_b))

		allOutputRatios.append(outputColorTriBlend)

		self.makeGradientGrid_color_tri(allOutputRatios)

		# print('allOutputRatios = ', allOutputRatios)

	def written_render(self):
		self.objectsToToggleOnOffLater.clear()

		self.deselectAll()
		self.deleteAllObjects()
		self.mega_purge()

		for area in bpy.data.screens["Layout"].areas:
		# for area in bpy.data.screens[bpy.context.window.screen].areas:
			if area.type == 'VIEW_3D':
				for space in area.spaces:
					if space.type == 'VIEW_3D':

						# space.shading.type = 'MATERIAL'

						# space.overlay.grid_scale = 2

						usableToggle = None
						if space.overlay.show_floor == True:
							usableToggle = False
						else:
							usableToggle = True

						usableToggle = False

						# space.overlay.show_floor = usableToggle
						space.overlay.show_axis_x = usableToggle
						space.overlay.show_axis_y = usableToggle
						space.overlay.show_axis_z = usableToggle
						space.overlay.show_cursor = usableToggle

		startTime = datetime.now()

		###########
		#DEFAULT CAMERA
		#############
		cam1_data = bpy.data.cameras.new('Camera')
		cam = bpy.data.objects.new('Camera', cam1_data)
		bpy.context.collection.objects.link(cam)

		###################################
		###### SET CAMERA POS / LOOK AT
		###################################
		self.myCam = bpy.data.objects["Camera"]

		# self.myCam.location = self.pos_camera_global
		self.myCam.location = mathutils.Vector((20, 0, 0))
		# self.myCam.location = mathutils.Vector((20, 0, 10))
		# self.myCam.location = mathutils.Vector((20, 0, -20))

		self.myCam.location = mathutils.Vector((-2, 25, 13)) #GOOD
		# self.myCam.location = mathutils.Vector((-2, 110, 58))
		# self.myCam.location = mathutils.Vector((-2, 40, 58))
		#  
		# bpy.context.object.data.type = 'ORTHO'
		self.myCam.data.type = 'PERSP'
		self.myCam.data.lens = 35

		#near 8.5x11 printable ratio
		# bpy.context.scene.render.resolution_x = 3900
		# bpy.context.scene.render.resolution_y = 3000

		# bpy.context.scene.render.resolution_x = 2550
		# bpy.context.scene.render.resolution_y = 1970

		bpy.context.scene.render.resolution_x = 1000
		bpy.context.scene.render.resolution_y = 1000

		self.myCam.data.clip_start = 1
		# self.myCam.data.clip_start = .1
		# self.myCam.data.clip_start = .5
		# self.myCam.data.clip_end = 100
		self.myCam.data.clip_end = 500

		self.myCam.data.lens_unit = 'FOV'
		self.myCam.data.angle = 1.0472

		self.updateScene() # need

		bpy.context.scene.camera = bpy.data.objects["Camera"]

		# PM * VM * MM (T * R * S)

		########
		#SET WRITTEN UI SETTINGS
		########
		# usableAspect_items = bpy.context.scene.bl_rna.properties['written_aspect_prop'].enum_items
		# usableAspectType_id = usableAspect_items[bpy.context.scene.written_aspect_prop].identifier
		# self.aspect = usableAspectType_id

		val_aspect_prop = bpy.context.scene.written_aspect_prop
		self.aspect = val_aspect_prop

		val_fov_prop = bpy.context.scene.written_fov_prop
		self.fov = val_fov_prop

		val_zNear_prop = bpy.context.scene.written_znear_prop
		self.zNear = val_zNear_prop

		val_zFar_prop = bpy.context.scene.written_zfar_prop
		self.zFar = val_zFar_prop

		# self.oren_roughness_stored = val_oren_roughness_prop

		# usableFOV_items = bpy.context.scene.bl_rna.properties['written_fov_prop'].enum_items
		# usableFovType_id = usableFOV_items[bpy.context.scene.written_fov_prop].identifier
		# self.aspect = usableFovType_id

		# usableZNear_items = bpy.context.scene.bl_rna.properties['written_znear_prop'].enum_items
		# usableZNearType_id = usableZNear_items[bpy.context.scene.written_znear_prop].identifier
		# self.zNear = usableZNearType_id

		# usableZFar_items = bpy.context.scene.bl_rna.properties['written_zfar_prop'].enum_items
		# usableZFarType_id = usableZFar_items[bpy.context.scene.written_zfar_prop].identifier
		# self.zFar = usableZFarType_id

		########
		#PM
		########
		myPM = self.writtenMatrix_createPM()

		###########
		# VM
		##########
		myVM = self.writtenMatrix_createVM(self.myCam.location)

		self.createArrowsForHalfFOV() #########

		##########################
		center = self.myOrigin
		f = self.abjNormalize_written(center - self.myCam.location)

		self.myV = -f

		zNear_set = 1
		zFar_set = 250

		pt0 = f * (zNear_set / 2) ##
		rad = (zFar_set / 4)

		# mult = 1
		# mult = 90
		mult = 150
		# mult = 250

		pt_00 = mathutils.Vector((pt0.x - 30, pt0.y - mult, pt0.z - 15, 1)) ### START
		pt_01 = mathutils.Vector((pt0.x - 30, pt0.y - mult, pt0.z + 15, 1)) ### START
		pt_02 = mathutils.Vector((pt0.x + 30, pt0.y - mult, pt0.z + 15, 1)) ### START
		pt_03 = mathutils.Vector((pt0.x + 30, pt0.y - mult, pt0.z - 15, 1)) ### START

		pt_04 = mathutils.Vector((pt0.x - 30, pt0.y - mult, pt0.z - 15, 1)) ### START
		pt_05 = mathutils.Vector((pt0.x - 30, pt0.y - mult, pt0.z + 15, 1)) ### START
		pt_06 = mathutils.Vector((pt0.x + 30, pt0.y - mult, pt0.z + 15, 1)) ### START
		pt_07 = mathutils.Vector((pt0.x + 30, pt0.y - mult, pt0.z - 15, 1)) ### START

		pt_08 = mathutils.Vector((pt0.x - 90, pt0.y - mult, pt0.z - 65, 1)) ### START
		pt_09 = mathutils.Vector((pt0.x - 60, pt0.y - mult, pt0.z + 45, 1)) ### START
		pt_10 = mathutils.Vector((pt0.x + 60, pt0.y - mult, pt0.z + 45, 1)) ### START
		pt_11 = mathutils.Vector((pt0.x + 90, pt0.y - mult, pt0.z - 65, 1)) ### START

		allObjLocs = None

		allObjLocs = []
		allObjLocs.append(pt_00)
		# allObjLocs.append(pt_01)
		# allObjLocs.append(pt_02)
		# allObjLocs.append(pt_03)
		# allObjLocs.append(pt_04)

		# allObjLocs.append(pt_05)
		# allObjLocs.append(pt_06)
		# allObjLocs.append(pt_07)
		# allObjLocs.append(pt_08)
		# allObjLocs.append(pt_09)
		# allObjLocs.append(pt_10)
		# allObjLocs.append(pt_11)

		#TO DO 9/9
		#match the following to calc:
		#
		#set up triangle face indices here because the presets wont be there in other formats.
		#do a world space raycast from camera position to triangle position
		#if the triangle is visible, get NDC points of those triangles points. Draw and label the points and triangles. 

		bpy.ops.object.mode_set(mode="OBJECT")

		myMVP = myPM @ myVM

		scene = bpy.context.scene
		camera = scene.camera

		self.written_set_up_ortho_render()

		useFragments = True

		if useFragments == True : 
			for idx, i in enumerate(allObjLocs):
				# bpy.ops.mesh.primitive_uv_sphere_add(radius=rad / 4)
				# bpy.ops.mesh.primitive_uv_sphere_add(radius=rad / 4, segments=8, ring_count=8)

				bpy.ops.mesh.primitive_monkey_add()

				myInputMesh = bpy.context.active_object
				myInputMesh.select_set(1)

				bpy.ops.object.modifier_add(type='SUBSURF')
				myObj = bpy.context.active_object
				myObj.modifiers["Subdivision"].levels = 1
				bpy.ops.object.modifier_apply(modifier="Subdivision")

				#TRIANGULATE
				bpy.ops.object.modifier_add(type='TRIANGULATE')
				bpy.ops.object.modifier_apply(modifier="Triangulate")

				myInputMesh.location = i.xyz

				bpy.ops.object.transform_apply(location=1, rotation=1, scale=1)

				# Get the active object, which must be a mesh
				obj = bpy.context.view_layer.objects.active
				world_space_verts = None
				world_space_verts = []

				bpy.ops.object.mode_set(mode='OBJECT')

				mesh = obj.data
				world_matrix = obj.matrix_world

				# for face_idx, poly in enumerate(mesh.polygons):
				for f in enumerate(mesh.polygons):
					# print('f = ', f)
					# print('f[0] = ', f[0])
					# continue 

					usableFaceColor = mathutils.Vector((0, 0, 0))

					if f[0] % 2 == 0:
						usableFaceColor = mathutils.Vector((1, 0, 0))
					# if poly % 2 == 0:

					# usableFaceColor = mathutils.Vector((0, 1, 0))

					# mat1 = self.newShader("ShaderVisualizer_gradient_" + str(j), "emission", Ci.x, Ci.y, Ci.z)
					mat1 = self.newShader("ShaderVisualizer_gradient_" + str(f), "emission", usableFaceColor.x, usableFaceColor.y, usableFaceColor.z)
					# mat1 = self.newShader("ShaderVisualizer_gradient_" + str(poly), "emission", usableFaceColor.x, usableFaceColor.y, usableFaceColor.z)
					# bpy.context.active_object.data.materials.clear()
					bpy.context.active_object.data.materials.append(mat1)





					# 3. Add the new material to the object's material slots
					# obj.data.materials.append(new_mat)
					mat_index = len(obj.data.materials) - 1

					# 4. Set a specific face (e.g., polygon index 0) to use the new material
					# face_index = face_idx
					if f[0] < len(obj.data.polygons):
						obj.data.polygons[f[0]].material_index = mat_index


					# for vert_idx in poly.vertices:
					# 	local_vert = mesh.vertices[vert_idx].co

					# 	world_vert = world_matrix @ local_vert

				# for vertex in obj.data.vertices:
				# 	# world_coord = obj.matrix_world @ vertex.co
				# 	# world_space_verts.append(world_coord)

				# 	face_vert_indices = [list(poly.vertices) for poly in obj.data.polygons]

				# 	for face_idx, vert_indices in enumerate(face_vert_indices):
				# 		print(f"Face {face_idx}: Vertices {vert_indices}")
				# 		print()

						# usableFaceColor = mathutils.Vector((0, 0, 0))

						# if face_idx % 2 == 0:
						# 	usableFaceColor = mathutils.Vector((1, 0, 0))
						# # if poly % 2 == 0:

						# usableFaceColor = mathutils.Vector((0, 1, 0))

						# # mat1 = self.newShader("ShaderVisualizer_gradient_" + str(j), "emission", Ci.x, Ci.y, Ci.z)
						# mat1 = self.newShader("ShaderVisualizer_gradient_" + str(face_idx), "emission", usableFaceColor.x, usableFaceColor.y, usableFaceColor.z)
						# # mat1 = self.newShader("ShaderVisualizer_gradient_" + str(poly), "emission", usableFaceColor.x, usableFaceColor.y, usableFaceColor.z)
						# bpy.context.active_object.data.materials.clear()
						# bpy.context.active_object.data.materials.append(mat1)

				# myInputMesh.hide_set(1)
				myInputMesh.hide_render = True

				continue

				# for idx2, j in enumerate(world_space_verts):
				# 	bpy.context.view_layer.objects.active = self.myPixel
				# 	myDupeGradient = self.copyObject()
				# 	myDupeGradient.name = 'dupeGradient_' + str(idx) + '_' + str(idx2)

				# 	myDupeGradient.rotation_euler = mathutils.Vector((0, math.radians(90), 0))

				# 	Ci = mathutils.Vector((0, 0, 0))	

				# 	bpy.context.view_layer.objects.active = myDupeGradient
				# 	mat1 = self.newShader("ShaderVisualizer_gradient_" + str(j), "emission", Ci.x, Ci.y, Ci.z)
				# 	bpy.context.active_object.data.materials.clear()
				# 	bpy.context.active_object.data.materials.append(mat1)

				# 	gradientScale = 0.1

				# 	myDupeGradient.scale = mathutils.Vector((gradientScale, gradientScale, gradientScale))
					
				# 	xMin = -5
				# 	xMax = 5
				# 	yMin = -5
				# 	yMax = 5

				# 	j4 = mathutils.Vector((j.x, j.y, j.z, 1))

				# 	myNDC = self.NDC_get(j4, myMVP)

				# 	if myNDC.x > 1 or myNDC.x < -1 or myNDC.y > 1 or myNDC.y < -1 or myNDC.z > 1 or myNDC.z < -1:
				# 		myDupeGradient.hide_set(1)
				# 		myDupeGradient.hide_render = True
				# 		continue

				# 	gradient_startPos = mathutils.Vector((-.9, myNDC.x * 5, myNDC.y * 5))

				# 	myDupeGradient.location = gradient_startPos

		else:
			for idx, i in enumerate(allObjLocs):
				# bpy.ops.mesh.primitive_uv_sphere_add(radius=rad / 4)
				bpy.ops.mesh.primitive_uv_sphere_add(radius=rad / 4, segments=8, ring_count=8)
				myInputMesh = bpy.context.active_object
				myInputMesh.select_set(1)
				myInputMesh.location = i.xyz

				bpy.ops.object.transform_apply(location=1, rotation=1, scale=1)

				# Get the active object, which must be a mesh
				obj = bpy.context.view_layer.objects.active
				world_space_verts = None
				world_space_verts = []

				for vertex in obj.data.vertices:
					world_coord = obj.matrix_world @ vertex.co
					world_space_verts.append(world_coord)

				# myInputMesh.hide_set(1)
				myInputMesh.hide_render = True

				for idx2, j in enumerate(world_space_verts):
					bpy.context.view_layer.objects.active = self.myPixel
					myDupeGradient = self.copyObject()
					myDupeGradient.name = 'dupeGradient_' + str(idx) + '_' + str(idx2)

					myDupeGradient.rotation_euler = mathutils.Vector((0, math.radians(90), 0))

					Ci = mathutils.Vector((0, 0, 0))	

					bpy.context.view_layer.objects.active = myDupeGradient
					mat1 = self.newShader("ShaderVisualizer_gradient_" + str(j), "emission", Ci.x, Ci.y, Ci.z)
					bpy.context.active_object.data.materials.clear()
					bpy.context.active_object.data.materials.append(mat1)

					gradientScale = 0.1

					myDupeGradient.scale = mathutils.Vector((gradientScale, gradientScale, gradientScale))
					
					xMin = -5
					xMax = 5
					yMin = -5
					yMax = 5

					j4 = mathutils.Vector((j.x, j.y, j.z, 1))

					myNDC = self.NDC_get(j4, myMVP)

					if myNDC.x > 1 or myNDC.x < -1 or myNDC.y > 1 or myNDC.y < -1 or myNDC.z > 1 or myNDC.z < -1:
						myDupeGradient.hide_set(1)
						myDupeGradient.hide_render = True
						continue

					gradient_startPos = mathutils.Vector((-.9, myNDC.x * 5, myNDC.y * 5))

					myDupeGradient.location = gradient_startPos

		self.defaultColorSettings_UI()

		totalTime = datetime.now() - startTime
		print('totalTime = ', totalTime)

	def createArrowsForHalfFOV(self):
		#############################################################################################
		################################# (INSTANCE ORIGINALS) ###############################
		#############################################################################################

		###############################
		######### CUBE HALF FOV LB ############
		###############################
		# self.myCubeV_halfFov_LB_og = self.createArrowFullProcess('myCubeV_halfFov_LB_og', 'back', False, self.myOrigin, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
		self.myCubeV_halfFov_LB_og = self.createArrowFullProcess('myCubeV_halfFov_LB_og', 'back', False, self.myOrigin, 1.0, 0.0, 0.0, 0.5, 0.5, 0.5)

		###############################
		######### CUBE HALF FOV LT ############
		###############################
		# self.myCubeV_halfFov_LT_og = self.createArrowFullProcess('myCubeV_halfFov_LT_og', 'back', False, self.myOrigin, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
		self.myCubeV_halfFov_LT_og = self.createArrowFullProcess('myCubeV_halfFov_LT_og', 'back', False, self.myOrigin, 1.0, 0.0, 0.0, 0.5, 0.5, 0.5)

		###############################
		######### CUBE HALF FOV RT ############
		###############################
		# self.myCubeV_halfFov_RT_og = self.createArrowFullProcess('myCubeV_halfFov_RT_og', 'back', False, self.myOrigin, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
		self.myCubeV_halfFov_RT_og = self.createArrowFullProcess('myCubeV_halfFov_RT_og', 'back', False, self.myOrigin, 1.0, 0.0, 0.0, 0.5, 0.5, 0.5)

		###############################
		######### CUBE HALF FOV RB ############
		###############################
		# self.myCubeV_halfFov_RB_og = self.createArrowFullProcess('myCubeV_halfFov_RB_og', 'back', False, self.myOrigin, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
		self.myCubeV_halfFov_RB_og = self.createArrowFullProcess('myCubeV_halfFov_RB_og', 'back', False, self.myOrigin, 1.0, 0.0, 0.0, 0.5, 0.5, 0.5)

		#############
		### CLEAN UP
		############
		self.deselectAll()

		bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)

		#apply emission shaders to arrows
		bpy.context.view_layer.objects.active = self.myCubeV_halfFov_LB_og
		mat1 = self.newShader("HalfFOV_arrow_LB", "emission", 1, 0, 0)
		bpy.context.active_object.data.materials.clear()
		bpy.context.active_object.data.materials.append(mat1)

		bpy.context.view_layer.objects.active = self.myCubeV_halfFov_LT_og
		mat1 = self.newShader("HalfFOV_arrow_LT", "emission", 1, 0, 0)
		bpy.context.active_object.data.materials.clear()
		bpy.context.active_object.data.materials.append(mat1)

		bpy.context.view_layer.objects.active = self.myCubeV_halfFov_RT_og
		mat1 = self.newShader("HalfFOV_arrow_RT", "emission", 1, 0, 0)
		bpy.context.active_object.data.materials.clear()
		bpy.context.active_object.data.materials.append(mat1)

		bpy.context.view_layer.objects.active = self.myCubeV_halfFov_RB_og
		mat1 = self.newShader("HalfFOV_arrow_RB", "emission", 1, 0, 0)
		bpy.context.active_object.data.materials.clear()
		bpy.context.active_object.data.materials.append(mat1)

		bpy.data.materials["HalfFOV_arrow_LB"].node_tree.nodes["Emission"].inputs[1].default_value = 10
		bpy.data.materials["HalfFOV_arrow_RB"].node_tree.nodes["Emission"].inputs[1].default_value = 10
		bpy.data.materials["HalfFOV_arrow_RT"].node_tree.nodes["Emission"].inputs[1].default_value = 10
		bpy.data.materials["HalfFOV_arrow_LT"].node_tree.nodes["Emission"].inputs[1].default_value = 10


		bpy.context.view_layer.objects.active = self.myCubeV_halfFov_LB_og
		self.myCubeV_halfFov_LB_og_dupe = self.copyObject()
		self.myCubeV_halfFov_LB_og_dupe.name = 'myCubeV_halfFov_LB_og_dupe'

		bpy.context.view_layer.objects.active = self.myCubeV_halfFov_LT_og
		self.myCubeV_halfFov_LT_og_dupe = self.copyObject()
		self.myCubeV_halfFov_LT_og_dupe.name = 'myCubeV_halfFov_LT_og_dupe'

		bpy.context.view_layer.objects.active = self.myCubeV_halfFov_RT_og
		self.myCubeV_halfFov_RT_og_dupe = self.copyObject()
		self.myCubeV_halfFov_RT_og_dupe.name = 'myCubeV_halfFov_RT_og_dupe'

		bpy.context.view_layer.objects.active = self.myCubeV_halfFov_RB_og
		self.myCubeV_halfFov_RB_og_dupe = self.copyObject()
		self.myCubeV_halfFov_RB_og_dupe.name = 'myCubeV_halfFov_RB_og_dupe'

		#############
		### RESET USABLE DUPES
		#############
		self.myCubeV_halfFov_LB_og_dupe.matrix_world = self.myCubeV_halfFov_LB_og_Matrix
		self.myCubeV_halfFov_LT_og_dupe.matrix_world = self.myCubeV_halfFov_LT_og_Matrix
		self.myCubeV_halfFov_RT_og_dupe.matrix_world = self.myCubeV_halfFov_RT_og_Matrix
		self.myCubeV_halfFov_RB_og_dupe.matrix_world = self.myCubeV_halfFov_RB_og_Matrix

		bpy.context.view_layer.objects.active = self.myCubeV_halfFov_LB_og

		self.dynamicM_halfFov_LB = None
		self.dynamicM_halfFov_LT = None
		self.dynamicM_halfFov_RT = None
		self.dynamicM_halfFov_RB = None

		self.equation_dynamic_cubeV_halfFov_all_creation()
		HalfFov_LB_M = self.dynamicM_halfFov_LB
		HalfFov_LT_M = self.dynamicM_halfFov_LT
		HalfFov_RT_M = self.dynamicM_halfFov_RT
		HalfFov_RB_M = self.dynamicM_halfFov_RB

		self.updateScene()

		HalfFov_LB_M_np = np.array(HalfFov_LB_M)
		HalfFov_LT_M_np = np.array(HalfFov_LT_M)
		HalfFov_RT_M_np = np.array(HalfFov_RT_M)
		HalfFov_RB_M_np = np.array(HalfFov_RB_M)

		mySplitFaceIndexUsable = '0'

		arrow_dynamic_instance_M_dict = {
			'mySplitFaceIndexUsable' : mySplitFaceIndexUsable,
			# 'N_M_np' : N_M_np,
			# 'L_M_np' : L_M_np,
			# 'V_M_np' : V_M_np,
			# 'R_M_np' : R_M_np,
			# 'H_M_np' : H_M_np,
			'HalfFov_LB_M_np' : HalfFov_LB_M_np,
			'HalfFov_LT_M_np' : HalfFov_LT_M_np,
			'HalfFov_RT_M_np' : HalfFov_RT_M_np,
			'HalfFov_RB_M_np' : HalfFov_RB_M_np,
		}

		self.arrow_dynamic_instance_M_all_list_matrixOnly_debug.append(arrow_dynamic_instance_M_dict) ##########

		myArrow_LB = self.show_arrow_halfFOV_LB(mySplitFaceIndexUsable)
		myArrow_LT = self.show_arrow_halfFOV_LT(mySplitFaceIndexUsable)
		myArrow_RT = self.show_arrow_halfFOV_RT(mySplitFaceIndexUsable)
		myArrow_RB = self.show_arrow_halfFOV_RB(mySplitFaceIndexUsable)

		self.myCubeV_halfFov_LB_og.hide_set(1)
		self.myCubeV_halfFov_LB_og.hide_render = True

		self.myCubeV_halfFov_LB_og_dupe.hide_set(1)
		self.myCubeV_halfFov_LB_og_dupe.hide_render = True

		self.myCubeV_halfFov_LT_og.hide_set(1)
		self.myCubeV_halfFov_LT_og.hide_render = True

		self.myCubeV_halfFov_LT_og_dupe.hide_set(1)
		self.myCubeV_halfFov_LT_og_dupe.hide_render = True

		self.myCubeV_halfFov_RT_og.hide_set(1)
		self.myCubeV_halfFov_RT_og.hide_render = True

		self.myCubeV_halfFov_RT_og_dupe.hide_set(1)
		self.myCubeV_halfFov_RT_og_dupe.hide_render = True

		self.myCubeV_halfFov_RB_og.hide_set(1)
		self.myCubeV_halfFov_RB_og.hide_render = True

		self.myCubeV_halfFov_RB_og_dupe.hide_set(1)
		self.myCubeV_halfFov_RB_og_dupe.hide_render = True

	def equation_dynamic_cubeV_halfFov_all_creation(self):
		cam_forward = self.myCam.matrix_world.to_quaternion() @ mathutils.Vector((0, 0, -1))
		cam_forward = self.myV

		upW = mathutils.Vector((0, 0, 1))

		center = mathutils.Vector((0, 0, 0))
		camP = self.myCam.location
		f = self.abjNormalize_written(center - camP)
		u = self.abjNormalize_written(upW)
		mySideV = self.abjNormalize_written(self.written_manual_cross(f, u))
		objUp = self.written_manual_cross(mySideV, f)

		my_vector = f
		rotation_axis = objUp

		##########################
		######### LB #############
		##########################
		rotation_angle = math.radians(0.5 * self.fov)
		# rotation_matrix = mathutils.Matrix.Rotation(rotation_angle, 4, rotation_axis)
		rotation_matrix = mathutils.Matrix.Rotation(rotation_angle, 4, objUp)
		rotated_vector = rotation_matrix @ my_vector

		rotation_angle_upDown = -rotation_angle / self.aspect
		rotation_matrix_upDown = mathutils.Matrix.Rotation(rotation_angle_upDown, 4, mySideV)
		rotated_vector = rotation_matrix_upDown @ rotated_vector

		rotated_vector = rotated_vector.normalized()

		self.dynamicM_halfFov_LB_vector = rotated_vector

		self.myCubeV_halfFov_LB_og_dupe.matrix_world = self.myCubeV_halfFov_LB_og_Matrix
		
		bpy.context.view_layer.objects.active = self.myCubeV_halfFov_LB_og_dupe
		self.myCubeV_halfFov_LB_og_dupe.location = self.myCam.location + (rotated_vector * self.zNear)

		self.updateScene()

		#look at direct
		rot_quat = rotated_vector.to_track_quat('X', 'Z')
		# rot_quat = rotated_vector

		# assume we're using euler rotation
		self.myCubeV_halfFov_LB_og_dupe.rotation_euler = rot_quat.to_euler()
		# self.myCubeV_halfFov_LB_og_dupe.rotation_euler = rot_quat
		# self.myCubeV_halfFov_LB_og_dupe.rotation_euler = rotated_vector

		# # #####################
		bpy.ops.object.mode_set(mode="OBJECT")
		self.deselectAll()
		self.myCubeV_halfFov_LB_og_dupe.select_set(1)

		myCubeV_halfFov_LB_og_Matrix_np = np.array(self.myCubeV_halfFov_LB_og_dupe.matrix_world)

		#################
		scale_factor = self.dynamicM_halfFov_LB_vector * self.zFar * self.zFarScale

		#scale to camera position
		self.objScaling_toMatchPosition_localSolve(self.myCubeV_halfFov_LB_og_dupe, self.myCubeV_halfFov_LB_og.name, scale_factor, 1, 0, myCubeV_halfFov_LB_og_Matrix_np)

		self.updateScene()

		self.dynamicM_halfFov_LB = self.myCubeV_halfFov_LB_og_dupe.matrix_world

		##########################
		######### LT #############
		##########################
		rotation_angle = math.radians(0.5 * self.fov)
		rotation_matrix = mathutils.Matrix.Rotation(rotation_angle, 4, objUp)
		rotated_vector = rotation_matrix @ my_vector

		rotation_angle_upDown = rotation_angle / self.aspect
		rotation_matrix_upDown = mathutils.Matrix.Rotation(rotation_angle_upDown, 4, mySideV)
		rotated_vector = rotation_matrix_upDown @ rotated_vector

		rotated_vector = rotated_vector.normalized()

		self.dynamicM_halfFov_LT_vector = rotated_vector

		self.myCubeV_halfFov_LT_og_dupe.matrix_world = self.myCubeV_halfFov_LT_og_Matrix
		
		bpy.context.view_layer.objects.active = self.myCubeV_halfFov_LT_og_dupe
		# self.myCubeV_halfFov_LT_og_dupe.location = self.myCam.location
		self.myCubeV_halfFov_LT_og_dupe.location = self.myCam.location + (rotated_vector * self.zNear)

		self.updateScene()

		#look at direct
		rot_quat = rotated_vector.to_track_quat('X', 'Z')
		# rot_quat = rotated_vector

		# assume we're using euler rotation
		self.myCubeV_halfFov_LT_og_dupe.rotation_euler = rot_quat.to_euler()
		# self.myCubeV_halfFov_LT_og_dupe.rotation_euler = rot_quat
		# self.myCubeV_halfFov_LT_og_dupe.rotation_euler = rotated_vector

		# # #####################
		bpy.ops.object.mode_set(mode="OBJECT")
		self.deselectAll()
		self.myCubeV_halfFov_LT_og_dupe.select_set(1)

		myCubeV_halfFov_LT_og_Matrix_np = np.array(self.myCubeV_halfFov_LT_og_dupe.matrix_world)

		#################
		scale_factor = self.dynamicM_halfFov_LT_vector * self.zFar * self.zFarScale

		#scale to camera position
		self.objScaling_toMatchPosition_localSolve(self.myCubeV_halfFov_LT_og_dupe, self.myCubeV_halfFov_LT_og_dupe.name, scale_factor, 1, 0, myCubeV_halfFov_LT_og_Matrix_np)

		self.updateScene()

		self.dynamicM_halfFov_LT = self.myCubeV_halfFov_LT_og_dupe.matrix_world

		##########################
		######### RT #############
		##########################
		rotation_angle = math.radians(-0.5 * self.fov)
		# rotation_matrix = mathutils.Matrix.Rotation(rotation_angle, 4, rotation_axis)
		rotation_matrix = mathutils.Matrix.Rotation(rotation_angle, 4, objUp)
		rotated_vector = rotation_matrix @ my_vector

		rotation_angle_upDown = rotation_angle / self.aspect
		rotation_matrix_upDown = mathutils.Matrix.Rotation(rotation_angle_upDown, 4, mySideV)
		rotated_vector = rotation_matrix_upDown @ rotated_vector

		rotated_vector = rotated_vector.normalized()

		self.dynamicM_halfFov_RT_vector = rotated_vector

		self.myCubeV_halfFov_RT_og_dupe.matrix_world = self.myCubeV_halfFov_RT_og_Matrix
		
		bpy.context.view_layer.objects.active = self.myCubeV_halfFov_RT_og_dupe
		# self.myCubeV_halfFov_RT_og_dupe.location = self.myCam.location
		self.myCubeV_halfFov_RT_og_dupe.location = self.myCam.location + (rotated_vector * self.zNear)

		self.updateScene()

		#look at direct
		rot_quat = rotated_vector.to_track_quat('X', 'Z')
		# rot_quat = rotated_vector

		# assume we're using euler rotation
		self.myCubeV_halfFov_RT_og_dupe.rotation_euler = rot_quat.to_euler()
		# self.myCubeV_halfFov_RT_og_dupe.rotation_euler = rot_quat
		# self.myCubeV_halfFov_RT_og_dupe.rotation_euler = rotated_vector

		# # #####################
		bpy.ops.object.mode_set(mode="OBJECT")
		self.deselectAll()
		self.myCubeV_halfFov_RT_og_dupe.select_set(1)

		myCubeV_halfFov_RT_og_Matrix = np.array(self.myCubeV_halfFov_RT_og_dupe.matrix_world)

		#################
		scale_factor = self.dynamicM_halfFov_RT_vector * self.zFar * self.zFarScale

		#scale to camera position
		self.objScaling_toMatchPosition_localSolve(self.myCubeV_halfFov_RT_og_dupe, self.myCubeV_halfFov_RT_og_dupe.name, scale_factor, 1, 0, myCubeV_halfFov_RT_og_Matrix)

		self.updateScene()

		self.dynamicM_halfFov_RT = self.myCubeV_halfFov_RT_og_dupe.matrix_world

		##########################
		######### RB #############
		##########################
		rotation_angle = math.radians(-0.5 * self.fov)
		# rotation_matrix = mathutils.Matrix.Rotation(rotation_angle, 4, rotation_axis)
		rotation_matrix = mathutils.Matrix.Rotation(rotation_angle, 4, objUp)
		rotated_vector = rotation_matrix @ my_vector

		rotation_angle_upDown = -rotation_angle / self.aspect
		rotation_matrix_upDown = mathutils.Matrix.Rotation(rotation_angle_upDown, 4, mySideV)
		rotated_vector = rotation_matrix_upDown @ rotated_vector

		rotated_vector = rotated_vector.normalized()

		self.dynamicM_halfFov_RB_vector = rotated_vector

		self.myCubeV_halfFov_RB_og_dupe.matrix_world = self.myCubeV_halfFov_RB_og_Matrix
		
		bpy.context.view_layer.objects.active = self.myCubeV_halfFov_RB_og_dupe
		# self.myCubeV_halfFov_RB_og_dupe.location = self.myCam.location
		self.myCubeV_halfFov_RB_og_dupe.location = self.myCam.location + (rotated_vector * self.zNear)

		self.updateScene()

		#look at direct
		rot_quat = rotated_vector.to_track_quat('X', 'Z')
		# rot_quat = rotated_vector

		# assume we're using euler rotation
		self.myCubeV_halfFov_RB_og_dupe.rotation_euler = rot_quat.to_euler()
		# self.myCubeV_halfFov_RB_og_dupe.rotation_euler = rot_quat
		# self.myCubeV_halfFov_RB_og_dupe.rotation_euler = rotated_vector

		# # #####################
		bpy.ops.object.mode_set(mode="OBJECT")
		self.deselectAll()
		self.myCubeV_halfFov_RB_og_dupe.select_set(1)

		myCubeV_halfFov_RB_og_Matrix = np.array(self.myCubeV_halfFov_RB_og_dupe.matrix_world)

		#################
		scale_factor = self.dynamicM_halfFov_RB_vector * self.zFar * self.zFarScale

		#scale to camera position
		self.objScaling_toMatchPosition_localSolve(self.myCubeV_halfFov_RB_og_dupe, self.myCubeV_halfFov_RB_og_dupe.name, scale_factor, 1, 0, myCubeV_halfFov_RB_og_Matrix)

		self.updateScene()

		self.dynamicM_halfFov_RB = self.myCubeV_halfFov_RB_og_dupe.matrix_world

	def writtenMatrix_createPM(self):
		#GL PM transposed

		deg2rad = self.fov * (math.pi / 180)
		tangent = math.tan(deg2rad / 2)
	
		myPM = mathutils.Matrix.Identity(4)
		myPM.zero()
		myPM[0][0] = 1 / (self.aspect * tangent)
		myPM[1][1] = 1 / (tangent)
		myPM[2][2] = -(self.zFar + self.zNear) / (self.zFar - self.zNear)
		myPM[2][3] = -(2 * self.zFar * self.zNear) / (self.zFar - self.zNear)
		myPM[3][2] = -1
		myPM[3][3] = 0

		return myPM

	def writtenMatrix_createVM(self, camP):
		center = self.myOrigin
		upW = mathutils.Vector((0, 0, 1))

		# scene = bpy.context.scene
		# camera = scene.camera

		f = self.abjNormalize_written(center - camP)
		u = self.abjNormalize_written(upW)
		s = self.abjNormalize_written(self.written_manual_cross(f, u))
		u = self.written_manual_cross(s, f)

		myVM = mathutils.Matrix.Identity(4)
		myVM[0][0] = s.x
		myVM[0][1] = s.y
		myVM[0][2] = s.z
		myVM[0][3] = -mathutils.Vector.dot(s, camP)

		myVM[1][0] = u.x
		myVM[1][1] = u.y
		myVM[1][2] = u.z
		myVM[1][3] = -mathutils.Vector.dot(u, camP) #13
		
		myVM[2][0] = -f.x
		myVM[2][1] = -f.y
		myVM[2][2] = -f.z
		myVM[2][3] = mathutils.Vector.dot(f, camP) #14

		myVM[3][0] = 0
		myVM[3][1] = 0
		myVM[3][2] = 0
		myVM[3][3] = 1

		return myVM

	def NDC_get(self, pt, myMVP):
		vert = myMVP @ pt
		vert_NDC = mathutils.Vector((vert.xyz / vert.w))

		return vert_NDC

	def written_set_up_ortho_render(self):

		# '''
		###################################
		###### SET CAMERA POS / LOOK AT
		###################################
		self.myCam = bpy.data.objects["Camera"]

		# self.myCam.location = self.pos_camera_global
		self.myCam.location = mathutils.Vector((20, 0, 0))
		# self.myCam.location = mathutils.Vector((-2, 25, 13)) #OLD

		self.myCam.rotation_euler = mathutils.Vector((math.radians(90), 0, math.radians(90)))

		# bpy.context.object.data.type = 'ORTHO'
		self.myCam.data.type = 'ORTHO'
		self.myCam.data.ortho_scale = 10

		bpy.context.scene.render.resolution_x = 1000
		bpy.context.scene.render.resolution_y = 1000
		self.updateScene() # need
		# '''

		#####################
		### input mesh
		#####################

		usablePrimitiveType_gradient_id = 'grid'
		if usablePrimitiveType_gradient_id == 'grid':
			bpy.ops.mesh.primitive_grid_add()

		myInputMesh = bpy.context.active_object
		myInputMesh.select_set(1)
		myInputMesh.hide_set(1)
		# myInputMesh.hide_render = True

		#####################
		### grey background
		#####################
		bpy.context.view_layer.objects.active = myInputMesh
		myDupeGradient_bg = self.copyObject()
		myDupeGradient_bg.name = 'dupeGradient_background'
		myDupeGradient_bg.scale = mathutils.Vector((5, 5, 5))
		myDupeGradient_bg.location = mathutils.Vector((-1, 0, 0))
		myDupeGradient_bg.rotation_euler = mathutils.Vector((0, math.radians(90), 0))

		bpy.context.view_layer.objects.active = myDupeGradient_bg

		greyBG = 0.5
		greyBG = pow(greyBG, 2.2)

		mat1 = self.newShader("ShaderVisualizer_gradientBG", "emission", greyBG, greyBG, greyBG)

		# mat1 = self.newShader("ShaderVisualizer_gradientBG", "emission", 0.5, 0.5, 0.5)
		bpy.context.active_object.data.materials.clear()
		bpy.context.active_object.data.materials.append(mat1)

		self.myPixel = myInputMesh

	def abjNormalize_written(self, inVec):
		outVec0 = inVec * (1 / math.sqrt(mathutils.Vector.dot(inVec, inVec)))

		return outVec0
	
	def stageIdx_print_UI(self):



		for area in bpy.data.screens["Layout"].areas:
		# for area in bpy.data.screens[bpy.context.window.screen].areas:
			if area.type == 'VIEW_3D':
				for space in area.spaces:
					if space.type == 'VIEW_3D':

						# space.overlay.grid_scale = 2

						usableToggle = None
						if space.overlay.show_floor == True:
							usableToggle = False
						else:
							usableToggle = True

						usableToggle = False

						# space.overlay.show_floor = usableToggle
						space.overlay.show_axis_x = usableToggle
						space.overlay.show_axis_y = usableToggle
						space.overlay.show_axis_z = usableToggle
						space.overlay.show_cursor = usableToggle


		return
	
		shadingDict_simple_specular_visualization = {
			'description' : 'Simple Specular Visualization',
			'stage_000' : 'N....show N arrow (cubeN)',
			'stage_001' : 'V....show V arrow (myCubeCam)',
			'stage_002' : 'N_dot_V......show both myCubeN and myCubeCam',
			'stage_003' : 'N_dot_V over ortho compensate trick, so continue...raycast from faceCenter to V',
			'stage_004' : 'faceCenter_to_V_rayCast was TRUE so continue...raycast from faceCenter to L',
			'stage_005' : 'faceCenter_to_L_rayCast was TRUE so continue......show arrows N and L',
			'stage_009' : 'R.....show R arrow (cubeR) along with N and L',
			'stage_010' : 'final shade',
		}

		for key, value in shadingDict_simple_specular_visualization.items():
			print(f"{key}: {value}")
		print(' ')

	def toggleExtras_UI(self):
		# for screen in bpy.data.screens:
		# 	print(screen.name) 

			# Animation
			# Compositing
			# Geometry Nodes
			# Layout
			# Modeling
			# Rendering
			# Scripting
			# Sculpting
			# Shading
			# Texture Paint
			# UV Editing

		# return

		# for area in bpy.data.screens["Scripting"].areas:
		for area in bpy.data.screens["Layout"].areas:
		# for area in bpy.data.screens[bpy.context.window.screen].areas:
			if area.type == 'VIEW_3D':
				for space in area.spaces:
					if space.type == 'VIEW_3D':

						# space.overlay.grid_scale = 2

						usableToggle = None
						if space.overlay.show_floor == True:
							usableToggle = False
						else:
							usableToggle = True

						space.overlay.show_floor = usableToggle
						space.overlay.show_axis_x = usableToggle
						space.overlay.show_axis_y = usableToggle
						space.overlay.show_axis_z = usableToggle
						space.overlay.show_cursor = usableToggle

						break

	def stereo_retinal_rivalry_fix(self, choice):
		# choice = 'cubeCam'
		# choice = 'cubeN_instance'
		# choice = 'cubeR_instance'
		# choice = 'light_instance'

		if choice == 'cubeCam':
			usableColor = mathutils.Vector((0, 0, 0))

			if self.adjustedColors == True:
				usableColor = mathutils.Vector((0.0, 1.0, 0.9)) #stereo
			else:
				usableColor = mathutils.Vector((0.0, 1.0, 0.9)) #agx

			for i in bpy.context.scene.objects:
				if i.name == self.myCubeCam.name:
					bpy.context.view_layer.objects.active = i
					mat1 = self.newShader(choice + '_niceColors', self.diffuse_or_emission_og_shading, usableColor[0], usableColor[1], usableColor[2])
					bpy.context.active_object.data.materials.clear()
					bpy.context.active_object.data.materials.append(mat1)

		else:
			if self.objectsToToggleOnOffLater:
				for i in self.objectsToToggleOnOffLater:
					usableColor = mathutils.Vector((0, 0, 0))

					if choice == 'cubeN_instance':
						if self.adjustedColors == True:
							usableColor = mathutils.Vector((0.0, 1.0, 0.25)) #stereo
						else:
							usableColor = mathutils.Vector((1.0, 0.0, 0.0)) #agx

					elif choice == 'cubeR_instance':
						if self.adjustedColors == True:
							usableColor = mathutils.Vector((0.2, 0.87, 1.0)) #stereo
						else:
							usableColor = mathutils.Vector((0.5, 0.0, 1.0)) #agx

					elif choice == 'light_instance':
						if self.adjustedColors == True:
							usableColor = mathutils.Vector((1.0, 1.0, 0.0)) #stereo
						else:
							usableColor = mathutils.Vector((1.0, 1.0, 0.0)) #agx

					if choice in i.name:
						bpy.context.view_layer.objects.active = i
						mat1 = self.newShader(choice + '_niceColors', self.diffuse_or_emission_og_shading, usableColor[0], usableColor[1], usableColor[2])
						bpy.context.active_object.data.materials.clear()
						bpy.context.active_object.data.materials.append(mat1)

	def agxColorSettings_UI(self):
		bpy.context.scene.view_settings.view_transform = 'AgX'
		# bpy.context.scene.view_settings.look = 'AgX - Punchy'
		bpy.context.scene.view_settings.look = 'None'
		bpy.context.scene.render.use_multiview = False

		self.adjustedColors = False
		try:
			self.stereo_retinal_rivalry_fix('cubeCam')
			self.stereo_retinal_rivalry_fix('cubeN_instance')
			self.stereo_retinal_rivalry_fix('cubeR_instance')
			self.stereo_retinal_rivalry_fix('light_instance')
		except:
			pass

		for area in bpy.context.screen.areas: 
			if area.type == 'VIEW_3D':
				for space in area.spaces: 
					if space.type == 'VIEW_3D':
						space.shading.type = 'MATERIAL'

	def textColorSettings_UI(self):
		self.agxColorSettings_UI() #must switch before

		for area in bpy.context.screen.areas: 
			if area.type == 'VIEW_3D':
				for space in area.spaces: 
					if space.type == 'VIEW_3D':
						space.shading.type = 'SOLID'

						space.shading.background_type = 'VIEWPORT'
						space.shading.background_color = (1, 1, 1)

						space.shading.color_type = 'SINGLE'
						space.shading.single_color = (1, 1, 1)

						space.shading.light = 'FLAT'

						usableToggle = False
						space.overlay.show_floor = usableToggle
						space.overlay.show_axis_x = usableToggle
						space.overlay.show_axis_y = usableToggle
						space.overlay.show_axis_z = usableToggle
						space.overlay.show_cursor = usableToggle


		for i in self.textRef_all:
			for j in bpy.context.scene.objects:
				if j.name == i:
					j.hide_set(0)

		self.adjustedColors = False
		try:
			self.stereo_retinal_rivalry_fix('cubeCam')
			self.stereo_retinal_rivalry_fix('cubeN_instance')
			self.stereo_retinal_rivalry_fix('cubeR_instance')
			self.stereo_retinal_rivalry_fix('light_instance')
		except:
			pass

	def stereoscopicColorSettings_UI(self):
		bpy.context.scene.view_settings.view_transform = 'Standard'
		bpy.context.scene.view_settings.look = 'None'
		bpy.context.scene.render.use_multiview = True

		for area in bpy.context.screen.areas: 
			if area.type == 'VIEW_3D':
				for space in area.spaces: 
					if space.type == 'VIEW_3D':
						space.shading.type = 'MATERIAL'

		self.adjustedColors = True
		self.stereo_retinal_rivalry_fix('cubeCam')
		self.stereo_retinal_rivalry_fix('cubeN_instance')
		self.stereo_retinal_rivalry_fix('cubeR_instance')
		self.stereo_retinal_rivalry_fix('light_instance')

	def defaultColorSettings_UI(self):
		bpy.context.scene.view_settings.view_transform = 'Standard'
		# bpy.context.scene.view_settings.look = 'AgX - Punchy'
		bpy.context.scene.view_settings.look = 'None'
		bpy.context.scene.render.use_multiview = False

		self.adjustedColors = False
		try:
			self.stereo_retinal_rivalry_fix('cubeCam')
			self.stereo_retinal_rivalry_fix('cubeN_instance')
			self.stereo_retinal_rivalry_fix('cubeR_instance')
			self.stereo_retinal_rivalry_fix('light_instance')
		except:
			pass

		for area in bpy.context.screen.areas: 
			if area.type == 'VIEW_3D':
				for space in area.spaces: 
					if space.type == 'VIEW_3D':
						space.shading.type = 'MATERIAL'		

	################
	def restoreCameraView_UI(self):
		restoredCamMatrix = mathutils.Matrix(self.world_mat_cam_stored_np.tolist())
		self.myCam.matrix_world = restoredCamMatrix

	def showhideCubeCam_UI(self):
		if self.myCubeCam.hide_get() == 1:
			self.myCubeCam.hide_set(0)
		else:
			self.myCubeCam.hide_set(1)

	def showhideText_UI(self):
		for i in self.textRef_all:
			for j in bpy.context.scene.objects:
				if j.name == i:
					if j.hide_get() == 1:
						j.hide_set(0)

					else:
						j.hide_set(1)

	def showhideArrows_UI(self):
		if self.objectsToToggleOnOffLater:
			print('self.objectsToToggleOnOffLater = ', self.objectsToToggleOnOffLater)

			for i in self.objectsToToggleOnOffLater:
				if i.hide_get() == 1:
					i.hide_set(0)

				else:
					i.hide_set(1)

	def genRandomVertexColor(self):
		randR_0 = random.randint(0, 9)
		randR_1 = random.randint(0, 9)
		randR_2 = random.randint(0, 9)

		randomCombo = '0.' + str(randR_0) + str(randR_1) + str(randR_2)

		return randomCombo

	def updateText_UI(self):
		print('some text')

	def restoreLight_UI(self):
		self.pos_light_global = self.pos_light_global_toRestore

		self.pos_light_global_v = mathutils.Vector((self.pos_light_global[0], self.pos_light_global[1], self.pos_light_global[2]))

		# self.DoIt_part1_preprocess()

	def restoreRxyz_UI(self):
		self.useRestoredRxyzValues = True

		self.DoIt_part1_preprocess()

	def randomLight_UI(self):
		posNegX_0 = float(self.genRandomVertexColor())
		posNegX_1 = None
		if posNegX_0 < 0.5:
			posNegX_1 = -posNegX_0
		else:
			posNegX_1 = posNegX_0

		posNegX_1 = posNegX_1 * random.randint(0, 10)

		posNegY_0 = float(self.genRandomVertexColor())
		posNegY_1 = None
		if posNegY_0 < 0.5:
			posNegY_1 = -posNegY_0
		else:
			posNegY_1 = posNegY_0

		posNegY_1 = posNegY_1 * random.randint(0, 10)

		posNegZ_0 = float(self.genRandomVertexColor())
		posNegZ_1 = None
		if posNegZ_0 < 0.5:
			posNegZ_1 = -posNegZ_0
		else:
			posNegZ_1 = posNegZ_0

		posNegZ_1 = posNegZ_1 * random.randint(0, 10)

		self.pos_light_global = (posNegX_1, posNegY_1, posNegZ_1)
		self.pos_light_global_v = mathutils.Vector((self.pos_light_global[0], self.pos_light_global[1], self.pos_light_global[2]))
		
		# print('self.pos_light_global NEW = ', self.pos_light_global)
		self.DoIt_part1_preprocess()

		self.random_withMinimumShading('light')

	def randomRotation_UI(self):
		rand_initial = float(self.genRandomVertexColor())

		axisToUse = None
		if rand_initial < 0.333:
			axisToUse = 'X'

		if rand_initial > 0.333 and rand_initial < 0.666:
			axisToUse = 'Y'

		if rand_initial > 0.666:
			axisToUse = 'Z'

		degrees = None

		randR_0 = random.randint(0, 9)
		randR_1 = random.randint(0, 9)
		randR_2 = random.randint(0, 9)

		randomCombo = str(randR_0) + str(randR_1) + str(randR_2)
		degrees = float(randomCombo)

		self.RandomRotationAxis = axisToUse
		self.RandomRotationDegree = degrees

		print('randomly rotate the input mesh')

		self.useRestoredRxyzValues = False
		self.DoIt_part1_preprocess()

		self.random_withMinimumShading('rotation')

	def random_withMinimumShading(self, type):
		val_min_shaded_prop = bpy.context.scene.min_shaded_prop

		if val_min_shaded_prop > 0:
			self.breakEarlyForRandomLightAndRxyz = True
			self.doIt_part2_render()
			#break early after spec

			min_shaded_counter_list = []

			for i in self.shadingList_perFace:
				mySplitFaceIndexUsable = i['mySplitFaceIndexUsable']
				spec = i['spec']

				if spec > 0:
					min_shaded_counter_list.append(mySplitFaceIndexUsable)

			if len(min_shaded_counter_list) < val_min_shaded_prop:
				print('random with min shaded running again, only ', len(min_shaded_counter_list), ' / ', val_min_shaded_prop)
				
				if type == 'light':
					self.randomLight_UI()
				elif type == 'rotation':
					self.randomRotation_UI()

			else:
				print('random with min shaded success for ', len(min_shaded_counter_list), ' / ', val_min_shaded_prop)
			
			bpy.ops.object.mode_set(mode="OBJECT")

			self.breakEarlyForRandomLightAndRxyz = False

	def static_debugOnly_Stage1_UI(self):
		#check and create 18 hue colorspace if using
		val_use_18_hue_colorspace_prop = bpy.context.scene.use_18_hue_colorspace_prop

		if val_use_18_hue_colorspace_prop == True:
			if self.colorWheelGradient_18_created == False:
				self.colorGradient_circular_preset18_0()

		self.setupCompositor()

		self.DoIt_part1_preprocess()

	def updateScene(self):
		bpy.context.view_layer.update() ####### !

	def deselectAll_editMode(self):
		bpy.ops.mesh.select_all(action='DESELECT')

	def deselectAll(self):
		try:
			bpy.ops.object.mode_set(mode="OBJECT")
			bpy.ops.object.select_all(action='DESELECT')
		except:
			pass

	def deleteAllObjects(self):
		for i in bpy.context.scene.objects:
			obj = bpy.data.objects.get(i.name)

			if obj:
				# Remove from all collections
				for col in obj.users_collection:
					col.objects.unlink(obj)

				# Delete the object
				bpy.data.objects.remove(obj, do_unlink=True)

	def deleteAllText18HueLabelObjects(self):
		for i in self.textRef_all:
			for j in bpy.context.scene.objects:
				if j.name == i:
					obj = bpy.data.objects.get(i)

					if obj:
						# Remove from all collections
						for col in obj.users_collection:
							col.objects.unlink(obj)

						# Delete the object
						bpy.data.objects.remove(obj, do_unlink=True)

	def deleteSpecificObject(self, objToDelete):
		for i in bpy.context.scene.objects:
			obj = bpy.data.objects.get(objToDelete)

			if obj:
				# Remove from all collections
				for col in obj.users_collection:
					col.objects.unlink(obj)

				# Delete the object
				bpy.data.objects.remove(obj, do_unlink=True)

	def mega_purge(self):
		orphan_ob = [o for o in bpy.data.objects if not o.users]
		while orphan_ob:
			bpy.data.objects.remove(orphan_ob.pop())
			
		orphan_mesh = [m for m in bpy.data.meshes if not m.users]
		while orphan_mesh:
			bpy.data.meshes.remove(orphan_mesh.pop())
			
		orphan_mat = [m for m in bpy.data.materials if not m.users]
		while orphan_mat:
			bpy.data.materials.remove(orphan_mat.pop())

		def purge_node_groups():   
			orphan_node_group = [g for g in bpy.data.node_groups if not g.users]
			while orphan_node_group:
				bpy.data.node_groups.remove(orphan_node_group.pop())
			if [g for g in bpy.data.node_groups if not g.users]: purge_node_groups()
		purge_node_groups()
			
		orphan_texture = [t for t in bpy.data.textures if not t.users]
		while orphan_texture:
			bpy.data.textures.remove(orphan_texture.pop())

		orphan_images = [i for i in bpy.data.images if not i.users]
		while orphan_images:
			bpy.data.images.remove(orphan_images.pop())

		orphan_cameras = [c for c in bpy.data.cameras if not c.users]
		while orphan_cameras :
			bpy.data.cameras.remove(orphan_cameras.pop())

	def copyObject(self):
		src_obj = bpy.context.active_object
		new_obj = src_obj.copy()
		new_obj.data = src_obj.data.copy()
		new_obj.animation_data_clear()
		bpy.context.collection.objects.link(new_obj)

		return new_obj

	def newMaterial(self, id):
		mat = bpy.data.materials.get(id)

		if mat is None:
			mat = bpy.data.materials.new(name=id)

		mat.use_nodes = True

		# if mat.node_tree:
		mat.node_tree.links.clear()
		mat.node_tree.nodes.clear()

		return mat

	def newShader(self, id, type, r, g, b):

		mat = self.newMaterial(id)

		nodes = mat.node_tree.nodes
		links = mat.node_tree.links
		output = nodes.new(type='ShaderNodeOutputMaterial')

		if type == "diffuse":
			shader = nodes.new(type='ShaderNodeBsdfDiffuse')
			nodes["Diffuse BSDF"].inputs[0].default_value = (r, g, b, 1)

		elif type == "emission":
			shader = nodes.new(type='ShaderNodeEmission')

			#set _rgb_ to 1 and use _strength_ only to go above 1 with glare bloom but greyscale only
			# nodes["Emission"].inputs[0].default_value = (1, 1, 1, 1)
			# nodes["Emission"].inputs[1].default_value = r

			#regular rgb 0-1
			nodes["Emission"].inputs[0].default_value = (r, g, b, 1)
			nodes["Emission"].inputs[1].default_value = 1

		elif type == "glossy":
			shader = nodes.new(type='ShaderNodeBsdfGlossy')
			nodes["Glossy BSDF"].inputs[0].default_value = (r, g, b, 1)
			nodes["Glossy BSDF"].inputs[1].default_value = 0

		elif type == 'principled':
			shader = nodes.new(type='ShaderNodeBsdfPrincipled')
			nodes["Principled BSDF"].inputs[0].default_value = (r, g, b, 1)

		links.new(shader.outputs[0], output.inputs[0])

		return mat

	def resizeTester(self):
		self.deselectAll()

		bpy.ops.mesh.primitive_cube_add()
		myDc1 = bpy.context.active_object
		myDc1.name = "dc1"
		myDc1.location = (5,5,5)

		mat1 = self.newShader("Shader1", "emission", 1, 0, 1)
		bpy.context.active_object.data.materials.clear()
		bpy.context.active_object.data.materials.append(mat1)
		# return

		bpy.context.view_layer.objects.active = myDc1
		myDc2 = self.copyObject()
		myDc2.name = 'dc2'
		myDc2.location = (-5, -5, -5)
		bpy.context.active_object.data.materials.clear()
		mat2 = self.newShader("Shader2", "emission", 1, 0, 0)
		bpy.context.active_object.data.materials.append(mat2)

		self.deselectAll()
		bpy.ops.object.mode_set(mode="OBJECT")
		myDc2.select_set(1)
		# bpy.ops.transform.resize(value=(2, 4, 2), orient_type='GLOBAL')
		myDc2.scale = mathutils.Vector((2, 4, 2))

	def createArrowFullProcess(self, name, front_or_back_arrow_origin, doLookAt, lookAtPos, r_agx, g_agx, b_agx, r_stereo, g_stereo, b_stereo):
		# self.myCubeCam = self.createArrowFullProcess('myCubeCam', 'back', True, self.myOrigin, 0.0, 1.0, 0.9, 0.0, 1.0, 0.9)

		# self.myCubeLight_og = self.createArrowFullProcess('myCubeLight_og', 'front', False, self.myOrigin, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0)
		
		# self.myCubeN_og = self.createArrowFullProcess('myCubeN_og', 'back', False, self.myOrigin, 1.0, 0.0, 0.0, 0.0, 1.0, 0.25)

		self.deselectAll()

		bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)

		bpy.ops.mesh.primitive_cube_add()
		outputArrow = bpy.context.active_object
		outputArrow.name = name

		bpy.ops.object.mode_set(mode="EDIT")
		bpy.ops.mesh.subdivide(number_cuts=2)

		bpy.ops.object.mode_set(mode="OBJECT")

		self.deselectAll()
		outputArrow.select_set(1)
		outputArrow.scale = mathutils.Vector((self.len_cam_arrow, self.arrowWidth, self.arrowWidth))		
		
		bpy.ops.object.transform_apply(location=1, rotation=1, scale=1)
		bpy.ops.object.mode_set(mode="EDIT")
		self.deselectAll_editMode()

		bpy.ops.object.mode_set(mode="OBJECT")

		self.makeArrowFromCube(outputArrow, self.len_reg_arrowExtendPos)
		bpy.ops.object.mode_set(mode="OBJECT") ###################
		self.deselectAll()
		outputArrow.select_set(1)
		outputArrow.scale = mathutils.Vector((self.scaleRegArrows, self.scaleRegArrows, self.scaleRegArrows))

		bpy.ops.object.transform_apply(location=1, rotation=1, scale=1) ##########

		if front_or_back_arrow_origin == 'back':
			bpy.context.scene.cursor.location = (self.len_reg_arrowExtendPos * self.scaleRegArrows, 0.0, 0.0)
		else:
			bpy.context.scene.cursor.location = (self.len_arrow_frontExtend * self.scaleRegArrows, 0.0, 0.0)

		bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')

		if name == 'myCubeCam':
			self.myCubeCam_og_Matrix = outputArrow.matrix_world
			self.myCubeCam_og_Matrix_np = np.array(outputArrow.matrix_world)
			self.world_mat_cam_stored_np = np.array(self.myCam.matrix_world)

		elif name == 'myCubeLight_og':
			self.myCubeLight_og_Matrix = outputArrow.matrix_world
			self.myCubeLight_og_Matrix_np = np.array(outputArrow.matrix_world)
		
		elif name == 'myCubeH_og':
			self.myCubeH_og_Matrix = outputArrow.matrix_world
			self.myCubeH_og_Matrix_np = np.array(outputArrow.matrix_world)

		elif name == 'myCubeN_og':
			self.myCubeN_og_Matrix = outputArrow.matrix_world
			self.myCubeN_og_Matrix_np = np.array(outputArrow.matrix_world)

		elif name == 'myCubeV_halfFov_LB_og':
			self.myCubeV_halfFov_LB_og_Matrix = outputArrow.matrix_world
			self.myCubeV_halfFov_LB_og_Matrix_np = np.array(outputArrow.matrix_world)


		elif name == 'myCubeV_halfFov_LT_og':
			self.myCubeV_halfFov_LT_og_Matrix = outputArrow.matrix_world
			self.myCubeV_halfFov_LT_og_Matrix_np = np.array(outputArrow.matrix_world)


		elif name == 'myCubeV_halfFov_RB_og':
			self.myCubeV_halfFov_RB_og_Matrix = outputArrow.matrix_world
			self.myCubeV_halfFov_RB_og_Matrix_np = np.array(outputArrow.matrix_world)


		elif name == 'myCubeV_halfFov_RT_og':
			self.myCubeV_halfFov_RT_og_Matrix = outputArrow.matrix_world
			self.myCubeV_halfFov_RT_og_Matrix_np = np.array(outputArrow.matrix_world)


		elif name == 'myCubeV_halfFov_L_og':
			self.myCubeV_halfFov_L_og_Matrix = outputArrow.matrix_world
			self.myCubeV_halfFov_L_og_Matrix_np = np.array(outputArrow.matrix_world)

		elif name == 'myCubeV_halfFov_R_og':
			self.myCubeV_halfFov_R_og_Matrix = outputArrow.matrix_world
			self.myCubeV_halfFov_R_og_Matrix_np = np.array(outputArrow.matrix_world)

		if doLookAt == True:
			outputArrow.location = self.myCam.location
			bpy.ops.object.transform_apply(location=1, rotation=1, scale=1)

			bpy.context.scene.cursor.location = self.myCam.location
			bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')

			self.look_at2(outputArrow, lookAtPos)

			self.deselectAll()
			outputArrow.select_set(1)

			mWorld_temp = np.array(outputArrow.matrix_world)

			self.objScaling_toMatchPosition_localSolve(outputArrow, outputArrow.name, lookAtPos, 1, 0, mWorld_temp)


		####
		bpy.context.view_layer.objects.active = outputArrow

		mat1 = None
		if self.adjustedColors == False:
			mat1 = self.newShader(name, self.diffuse_or_emission_og_shading, r_agx, g_agx, b_agx) ####
		elif self.adjustedColors == True:	
			mat1 = self.newShader(name, self.diffuse_or_emission_og_shading, r_stereo, g_stereo, b_stereo)

		bpy.context.active_object.data.materials.clear()
		bpy.context.active_object.data.materials.append(mat1)

		return outputArrow

	def setupCompositor(self):
		bpy.context.scene.use_nodes = True

		# nodetree = bpy.context.scene.node_tree

		nodetree = bpy.data.node_groups.new("My new comp", "CompositorNodeTree")
		bpy.context.scene.compositing_node_group = nodetree

		# clear default nodes
		for node in nodetree.nodes:
			nodetree.nodes.remove(node)

		# add the glare node
		node0 = nodetree.nodes.new("CompositorNodeRLayers")
		node0.location = (0,0)

		node1 = nodetree.nodes.new("CompositorNodeGlare")
		node1.location = (400,0)
		node1.inputs[1].default_value = 'Bloom'
		# bpy.data.node_groups["My new comp"].nodes["Glare"].inputs[7].default_value = 500
		node1.inputs[7].default_value = 50

		n_img = nodetree.nodes.new("CompositorNodeImage")
		node2 = nodetree.nodes.new("NodeGroupOutput")

		node2.location = (800,0)

		nodetree.interface.new_socket(name="Output", in_out='OUTPUT',socket_type='NodeSocketColor')

		# connect the nodes
		nodetree.links.new(node0.outputs["Image"],node1.inputs[0])
		nodetree.links.new(node1.outputs["Image"],node2.inputs[0])

		self.compositor_setup = True

	def spectral_compositor_debugging_exit_visualizer(self, nodetree, nodeToView, readPixelX, readPixelY):
		
		############################
		#the user must have saved a new, default scene to a file and have a folder called 'compositing_files' in that directory
		############################

		###########
		#DEFAULT CAMERA
		#############
		# self.pos_camera_global = (10, 10, 10) #spectral
		# self.pos_camera_global = (20, -30, 15) #spectral
		# self.pos_camera_global = (5, -20, 10) #spectral ####
		# self.pos_camera_global = (2, -15, 10) #spectral
		self.pos_camera_global = (2, -13.5, 8) #spectral

		cam1_data = bpy.ops.object.camera_add(
			location=(self.pos_camera_global),  # x, y, z coordinates
			rotation=(0.0, 0.0, 0.0)   # x, y, z rotation in radians
		)

		self.myCam = bpy.data.objects["Camera"]
		self.myCam.data.clip_start = 1
		# # self.myCam.data.clip_start = .1
		# # self.myCam.data.clip_start = .5
		self.myCam.data.clip_end = 100
		# self.myCam.data.clip_end = 500

		bpy.context.scene.camera = self.myCam

		self.updateScene() # need
		# self.look_at(self.myCam, myInputMesh3.location)

		bpy.context.scene.render.engine = 'CYCLES'
		bpy.context.scene.cycles.device = 'GPU'

		# bpy.context.scene.render.resolution_x = 3840
		# bpy.context.scene.render.resolution_y = 2160

		bpy.context.scene.cycles.samples = 1024

		bpy.context.scene.cycles.denoising_use_gpu = True

		# bpy.context.scene.view_settings.view_transform = 'AgX'
		# bpy.context.scene.view_settings.look = 'AgX - Punchy'


		##########
		## DEBUG_SPECTRAL_COMPOSITOR
		#########
		if "Cube" in bpy.data.objects:
			cube_obj = bpy.data.objects["Cube"]
			# Unlink and remove the object completely
			bpy.data.objects.remove(cube_obj, do_unlink=True)

		########
		##### 00
		########
		bpy.ops.mesh.primitive_monkey_add()
		myInputMesh = bpy.context.active_object
		myInputMesh.select_set(1)
		myInputMesh.location = mathutils.Vector((0, 0, 5))
		bpy.ops.object.shade_smooth()

		mat1 = self.newShader("principled_test_00", "principled", 1, 0, 0)
		bpy.context.active_object.data.materials.clear()
		bpy.context.active_object.data.materials.append(mat1)
		bpy.data.materials["principled_test_00"].node_tree.nodes["Principled BSDF"].inputs[1].default_value = 1
		bpy.data.materials["principled_test_00"].node_tree.nodes["Principled BSDF"].inputs[2].default_value = 0.323263
		bpy.data.materials["principled_test_00"].node_tree.nodes["Principled BSDF"].inputs[20].default_value = 1

		bpy.ops.object.modifier_add(type='SUBSURF')
		myObj = bpy.context.active_object
		myObj.modifiers["Subdivision"].levels = 1
		myObj.modifiers["Subdivision"].use_adaptive_subdivision = True
		myObj.active_material.displacement_method = 'BOTH'

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
		displacement_node.inputs[2].default_value = 0.3

		self.autoArrangeNodes(mat.node_tree)

		########
		##### 01
		########
		bpy.ops.mesh.primitive_monkey_add()
		myInputMesh2 = bpy.context.active_object
		myInputMesh2.select_set(1)
		myInputMesh2.location = mathutils.Vector((1, 8, 5))
		bpy.ops.object.shade_smooth()

		mat1 = self.newShader("principled_test_01", "principled", 1, 0, 0)
		bpy.context.active_object.data.materials.clear()
		bpy.context.active_object.data.materials.append(mat1)
		bpy.data.materials["principled_test_01"].node_tree.nodes["Principled BSDF"].inputs[1].default_value = 1
		bpy.data.materials["principled_test_01"].node_tree.nodes["Principled BSDF"].inputs[2].default_value = 0.1
		bpy.data.materials["principled_test_01"].node_tree.nodes["Principled BSDF"].inputs[20].default_value = 1

		bpy.ops.object.modifier_add(type='SUBSURF')
		myObj = bpy.context.active_object
		myObj.modifiers["Subdivision"].levels = 1
		myObj.modifiers["Subdivision"].use_adaptive_subdivision = True
		myObj.active_material.displacement_method = 'BOTH'

		mat = bpy.data.materials.get("principled_test_01")
		nodes = mat.node_tree.nodes

		nodes = mat.node_tree.nodes

		output_node = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
		if not output_node:
			output_node = nodes.new(type='ShaderNodeOutputMaterial')

		gabor_node = nodes.new(type='ShaderNodeTexGabor')
		displacement_node = nodes.new(type='ShaderNodeDisplacement')

		mat.node_tree.links.new(gabor_node.outputs['Value'], displacement_node.inputs['Height'])
		mat.node_tree.links.new(displacement_node.outputs['Displacement'], output_node.inputs['Displacement'])
		displacement_node.inputs[2].default_value = 0.3

		self.autoArrangeNodes(mat.node_tree)

		# mat1 = self.newShader("greenM", "emission", 0, 1, 0)
		# mat1 = self.newShader("greenM", "diffuse", 0, 1, 0)
		# bpy.context.active_object.data.materials.clear()
		# bpy.context.active_object.data.materials.append(mat1)

		########
		##### 02
		########
		bpy.ops.mesh.primitive_monkey_add()
		myInputMesh = bpy.context.active_object
		myInputMesh.select_set(1)
		myInputMesh.location = mathutils.Vector((3, 13, 5))
		bpy.ops.object.shade_smooth()

		mat1 = self.newShader("principled_test_02", "principled", 1, 0, 0)
		bpy.context.active_object.data.materials.clear()
		bpy.context.active_object.data.materials.append(mat1)
		bpy.data.materials["principled_test_02"].node_tree.nodes["Principled BSDF"].inputs[1].default_value = 1
		bpy.data.materials["principled_test_02"].node_tree.nodes["Principled BSDF"].inputs[2].default_value = 0.1
		bpy.data.materials["principled_test_02"].node_tree.nodes["Principled BSDF"].inputs[20].default_value = 1


		bpy.ops.object.modifier_add(type='SUBSURF')
		myObj = bpy.context.active_object
		myObj.modifiers["Subdivision"].levels = 1
		myObj.modifiers["Subdivision"].use_adaptive_subdivision = True
		myObj.active_material.displacement_method = 'BOTH'

		mat = bpy.data.materials.get("principled_test_02")
		nodes = mat.node_tree.nodes

		nodes = mat.node_tree.nodes

		output_node = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
		if not output_node:
			output_node = nodes.new(type='ShaderNodeOutputMaterial')

		gabor_node = nodes.new(type='ShaderNodeTexGabor')
		displacement_node = nodes.new(type='ShaderNodeDisplacement')

		mat.node_tree.links.new(gabor_node.outputs['Value'], displacement_node.inputs['Height'])
		mat.node_tree.links.new(displacement_node.outputs['Displacement'], output_node.inputs['Displacement'])
		displacement_node.inputs[2].default_value = 0.3

		self.autoArrangeNodes(mat.node_tree)

		########
		##### 03
		########
		bpy.ops.mesh.primitive_monkey_add()
		myInputMesh = bpy.context.active_object
		myInputMesh.select_set(1)
		myInputMesh.location = mathutils.Vector((5, 16, 5))
		bpy.ops.object.shade_smooth()

		mat1 = self.newShader("principled_test_03", "principled", 1, 0, 0)
		bpy.context.active_object.data.materials.clear()
		bpy.context.active_object.data.materials.append(mat1)
		bpy.data.materials["principled_test_03"].node_tree.nodes["Principled BSDF"].inputs[1].default_value = 1
		bpy.data.materials["principled_test_03"].node_tree.nodes["Principled BSDF"].inputs[2].default_value = 0.1
		bpy.data.materials["principled_test_03"].node_tree.nodes["Principled BSDF"].inputs[20].default_value = 1

		bpy.ops.object.modifier_add(type='SUBSURF')
		myObj = bpy.context.active_object
		myObj.modifiers["Subdivision"].levels = 1
		myObj.modifiers["Subdivision"].use_adaptive_subdivision = True
		myObj.active_material.displacement_method = 'BOTH'

		mat = bpy.data.materials.get("principled_test_03")
		nodes = mat.node_tree.nodes

		nodes = mat.node_tree.nodes

		output_node = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
		if not output_node:
			output_node = nodes.new(type='ShaderNodeOutputMaterial')

		gabor_node = nodes.new(type='ShaderNodeTexGabor')
		displacement_node = nodes.new(type='ShaderNodeDisplacement')

		mat.node_tree.links.new(gabor_node.outputs['Value'], displacement_node.inputs['Height'])
		mat.node_tree.links.new(displacement_node.outputs['Displacement'], output_node.inputs['Displacement'])
		displacement_node.inputs[2].default_value = 0.3

		self.autoArrangeNodes(mat.node_tree)

		####
		# CUBE GROUND
		####
		bpy.ops.mesh.primitive_cube_add()
		myInputMesh3 = bpy.context.active_object
		myInputMesh3.select_set(1)
		myInputMesh3.location = mathutils.Vector((0, 0, -5))
		myInputMesh3.scale = mathutils.Vector((20, 20, 2))

		# mat1 = self.newShader("blueM", "emission", 0, 0, 1)
		# mat1 = self.newShader("blueM", "diffuse", 0, 0, 1)
		# bpy.context.active_object.data.materials.clear()
		# bpy.context.active_object.data.materials.append(mat1)

		mat1 = self.newShader("principled_test_grd", "principled", 0, 0, 1)
		bpy.context.active_object.data.materials.clear()
		bpy.context.active_object.data.materials.append(mat1)
		bpy.data.materials["principled_test_grd"].node_tree.nodes["Principled BSDF"].inputs[1].default_value = 1
		bpy.data.materials["principled_test_grd"].node_tree.nodes["Principled BSDF"].inputs[2].default_value = 0.323263

		self.autoArrangeNodes(mat.node_tree)

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
		output_node_world = worldtree.nodes.new(type="ShaderNodeOutputWorld")

		# node_sky = bpy.ops.node.add_node(use_transform=True, type="ShaderNodeTexSky")
		# node_sky = bpy.ops.node.add_node(use_transform=True, type="ShaderNodeTexSky")
		node_sky = worldtree.nodes.new('ShaderNodeTexSky')
		worldtree.links.new(node_sky.outputs["Color"], output_node_world.inputs["Surface"])

		# bpy.data.worlds["World"].node_tree.nodes["Sky Texture"].sun_size = 0.372541
		# bpy.data.worlds["World"].node_tree.nodes["Sky Texture"].sun_intensity = 21.3
		# bpy.data.worlds["World"].node_tree.nodes["Sky Texture"].sun_rotation = -1.57603

		# return

		# node_sky.sun_size = 0.372541
		# node_sky.sun_intensity = 21.3
		node_sky.sun_rotation = 1.65806

		self.look_at(self.myCam, myInputMesh2.location)

		nodetree.links.new(nodeToView.outputs[0], self.nodeOut.inputs[0]) ###### !!!!!!!!!!!
		nodetree.links.new(nodeToView.outputs[0], self.nodeViewer.inputs[0]) ###### !!!!!!!!!!!

		self.autoArrangeNodes(nodetree)
		self.autoArrangeNodes(worldtree)

		self.compositor_setup = True

		########################################################
		#write node to disc and read pixel
		########################################################
		# bpy.context.scene.view_settings.view_transform = 'Standard'
		# bpy.context.scene.view_settings.look = 'AgX - Punchy'
		# bpy.context.scene.view_settings.look = 'None'

		bpy.context.scene.view_settings.view_transform = 'AgX'
		bpy.context.scene.view_settings.look = 'AgX - Punchy'

		bpy.context.scene.render.use_multiview = False

		# temp_filepath = bpy.path.abspath("//compositor_pixel_temp.png")
		temp_filepath = "//compositing_files/readSingleOutputPixel.png"
		original_filepath = bpy.context.scene.render.filepath

		bpy.context.scene.render.filepath = temp_filepath

		# 2. Render to write the compositor result out to disk
		bpy.ops.render.render(write_still=True)

		# Restore original render filepath
		bpy.context.scene.render.filepath = original_filepath

		# 3. Load the saved image back via bpy.data.images to inspect pixel data safely
		img = bpy.data.images.load(temp_filepath, check_existing=False)

		# bpy.data.images[temp_filepath].reload()
		# "E:/projects_3d/ABJ_Shader_Debugger_for_Blender/scenes/compositing_files/readSingleOutputPixel.png"
		# bpy.data.images["E:/projects_3d/ABJ_Shader_Debugger_for_Blender/scenes/compositing_files/readSingleOutputPixel.png"].reload()

		# bpy.ops.image.reload()

		self.updateScene()

		for img in bpy.data.images:
			if img.filepath.endswith("readSingleOutputPixel.png"):
				img.reload()

		# bpy.data.images["readSingleOutputPixel.png"].reload()

		# 4. Choose your target single pixel coordinates (x, y)
		width = img.size[0]
		height = img.size[1]

		# Check bounds
		if 0 <= readPixelX < width and 0 <= readPixelY < height:
			# Calculate 1D index for RGBA array (4 channels per pixel)
			pixel_index = (readPixelY * width + readPixelX) * 4
			
			r = img.pixels[pixel_index]
			g = img.pixels[pixel_index + 1]
			b = img.pixels[pixel_index + 2]
			a = img.pixels[pixel_index + 3]
			
			# print(f"Pixel at ({readPixelX}, {readPixelY}) -> R: {r:.4f}, G: {g:.4f}, B: {b:.4f}, A: {a:.4f}")
			print(f"Pixel at ({readPixelX}, {readPixelY}) -> R: {r}, G: {g}, B: {b}, A: {a}")
		else:
			print("Coordinates are out of image bounds.")

		# 5. Clean up temporary image data block from Blender memory
		bpy.data.images.remove(img)

	def spectral_compositor_stock(self):
		self.deselectAll()
		self.deleteAllObjects()
		self.mega_purge()

		bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True

		compGroupName = 'spectralCompositor'

		nodetree = bpy.data.node_groups.new(compGroupName, "CompositorNodeTree")
		bpy.context.scene.compositing_node_group = nodetree

		for node in nodetree.nodes:
			nodetree.nodes.remove(node)

		node0 = nodetree.nodes.new("CompositorNodeRLayers")

		nodetree.interface.new_socket(name="Output", in_out='OUTPUT', socket_type='NodeSocketColor')
		self.nodeOut = nodetree.nodes.new("NodeGroupOutput")

		self.nodeViewer = nodetree.nodes.new("CompositorNodeViewer")

		self.spectral_compositor_debugging_exit_visualizer(nodetree, node0, 932, 633)
		return

	#This function is based on spectral3_glsl.py, under MIT license by Ronald van Wijnen (see file)
	def spectral_compositor(self):
		self.deselectAll()
		self.deleteAllObjects()
		self.mega_purge()

		bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True

		compGroupName = 'spectralCompositor'

		nodetree = bpy.data.node_groups.new(compGroupName, "CompositorNodeTree")
		bpy.context.scene.compositing_node_group = nodetree

		for node in nodetree.nodes:
			nodetree.nodes.remove(node)

		node0 = nodetree.nodes.new("CompositorNodeRLayers")

		nodetree.interface.new_socket(name="Output", in_out='OUTPUT', socket_type='NodeSocketColor')
		self.nodeOut = nodetree.nodes.new("NodeGroupOutput")

		self.nodeViewer = nodetree.nodes.new("CompositorNodeViewer")

		# self.spectral_compositor_debugging_exit_visualizer(nodetree, node0, 932, 633)
		# return

		###################
		# CONSTANTS
		###################
		# SPECTRAL_GAMMA
		self.node_spectral_gamma = nodetree.nodes.new("ShaderNodeValue")
		self.node_spectral_gamma.outputs[0].default_value = 2.4
		# self.node_spectral_gamma.outputs[0].default_value = 2.2

		# SPECTRAL_EPSILON
		self.node_spectral_epsilon = nodetree.nodes.new("ShaderNodeValue")
		self.node_spectral_epsilon.outputs[0].default_value = 0.0000000000000001

		#SPECTRAL_SIZE
		self.node_spectral_size = nodetree.nodes.new("FunctionNodeInputInt")
		self.node_spectral_size.integer = 38

		SPECTRAL_SIZE = 38

		##### 1
		# # SPECTRAL_TINTINGSTRENGTH_1 (AS DEPTH)
		
		###########
		#DEPTH
		###########
		# add map range node
		node_mapRange = nodetree.nodes.new("ShaderNodeMapRange")
		node_mapRange.location = (0,0)
		node_mapRange.label = 'depth_adjustable'
		node_mapRange.data_type = 'FLOAT'
		node_mapRange.clamp = True
		node_mapRange.inputs[1].default_value = 0.1
		node_mapRange.inputs[2].default_value = 100
		# node_mapRange.inputs[3].default_value = -0.5
		node_mapRange.inputs[3].default_value = -0.25
		node_mapRange.inputs[4].default_value = 1

		node_mapRange.use_custom_color = True
		node_mapRange.color = (1, 0, 0)
		nodetree.links.new(node0.outputs["Depth"], node_mapRange.inputs[0])

		node_depthMultiply = nodetree.nodes.new("ShaderNodeMath")
		node_depthMultiply.operation = 'MULTIPLY'
		node_depthMultiply.label = 'depth_multiply'
		# node_depthMultiply.inputs[0].default_value = 100
		# node_depthMultiply.inputs[0].default_value = 50
		# node_depthMultiply.inputs[0].default_value = 4
		# node_depthMultiply.inputs[0].default_value = 20
		node_depthMultiply.inputs[0].default_value = -12
		node_depthMultiply.use_custom_color = True
		node_depthMultiply.color = (1, 0, 0)
		nodetree.links.new(node_mapRange.outputs[0], node_depthMultiply.inputs[1])

		# add depth + val -> Spectral Color 0 TINT
		node_spectral_tintingStrength1 = nodetree.nodes.new("ShaderNodeMath")
		node_spectral_tintingStrength1.operation = 'ADD'
		node_spectral_tintingStrength1.label = 'tint1'
		# node_spectral_tintingStrength1.inputs[0].default_value = 1
		# node_spectral_tintingStrength1.inputs[0].default_value = .5
		node_spectral_tintingStrength1.inputs[0].default_value = 1.3
		# node_spectral_tintingStrength1.inputs[0].default_value = 2.4
		nodetree.links.new(node_depthMultiply.outputs[0], node_spectral_tintingStrength1.inputs[1])
		node_spectral_tintingStrength1.use_custom_color = True
		node_spectral_tintingStrength1.color = (1, 0, 0)

		# SPECTRAL_FACTOR_1
		node_spectral_factor1 = nodetree.nodes.new("ShaderNodeValue")
		node_spectral_factor1.outputs[0].default_value = 1
		# node_spectral_factor1.outputs[0].default_value = 15
		node_spectral_factor1.label = 'factor 1'
		node_spectral_factor1.use_custom_color = True
		node_spectral_factor1.color = (1, 0, 0)

		##### 2
		# SPECTRAL_TINTINGSTRENGTH_2
		node_spectral_tintingStrength2 = nodetree.nodes.new("ShaderNodeValue")
		node_spectral_tintingStrength2.outputs[0].default_value = 1
		node_spectral_tintingStrength2.label = 'tint 2'
		node_spectral_tintingStrength2.use_custom_color = True
		node_spectral_tintingStrength2.color = (1, 0, 0)

		# SPECTRAL_FACTOR_2
		node_spectral_factor2 = nodetree.nodes.new("ShaderNodeValue")
		node_spectral_factor2.outputs[0].default_value = 1
		node_spectral_factor2.label = 'factor 2'
		node_spectral_factor2.use_custom_color = True
		node_spectral_factor2.color = (1, 0, 0)

		###################
		#color 2 split
		###################
		node_color2 = nodetree.nodes.new("CompositorNodeRGB")
		node_color2.outputs[0].default_value = (0.5, 0.5, 0.5, 1) #linear

		##############
		## lrgb1
		##############
		# node_lrgb1 = nodetree.nodes.new("ShaderNodeCombineXYZ")
		node_lrgb1_combo = nodetree.nodes.new("ShaderNodeCombineXYZ")

		lrgb1_r = self.spectral_compositor_uncompand(nodetree, node0, 0)
		lrgb1_g = self.spectral_compositor_uncompand(nodetree, node0, 1)
		lrgb1_b = self.spectral_compositor_uncompand(nodetree, node0, 2)

		nodetree.links.new(lrgb1_r.outputs[0], node_lrgb1_combo.inputs[0])
		nodetree.links.new(lrgb1_g.outputs[0], node_lrgb1_combo.inputs[1])
		nodetree.links.new(lrgb1_b.outputs[0], node_lrgb1_combo.inputs[2])

		# print('READPIXEL : node_lrgb1_combo')
		# self.spectral_compositor_debugging_exit_visualizer(nodetree, node_lrgb1_combo, 932, 633)
		# return

		##############
		## lrgb2
		##############
		node_lrgb2_combo = nodetree.nodes.new("ShaderNodeCombineXYZ")

		lrgb2_r = self.spectral_compositor_uncompand(nodetree, node_color2, 0)
		lrgb2_g = self.spectral_compositor_uncompand(nodetree, node_color2, 1)
		lrgb2_b = self.spectral_compositor_uncompand(nodetree, node_color2, 2)

		nodetree.links.new(lrgb2_r.outputs[0], node_lrgb2_combo.inputs[0])
		nodetree.links.new(lrgb2_g.outputs[0], node_lrgb2_combo.inputs[1])
		nodetree.links.new(lrgb2_b.outputs[0], node_lrgb2_combo.inputs[2])

		##############
		## R1 and R2
		##############
		R1 = self.spectral_compositor_linear_to_reflectance(nodetree, node_lrgb1_combo)
		# return
		R2 = self.spectral_compositor_linear_to_reflectance(nodetree, node_lrgb2_combo)


		##############
		## luminance
		##############
		luminance1 = self.spectral_compositor_reflectance_to_xyz_p0(nodetree, R1)
		luminance1.label = 'luminance1'
		luminance1_separated = nodetree.nodes.new("ShaderNodeSeparateXYZ")
		luminance1_separated.label = 'luminance1_sep'
		nodetree.links.new(luminance1.outputs[0], luminance1_separated.inputs[0])

		luminance2 = self.spectral_compositor_reflectance_to_xyz_p0(nodetree, R2)
		luminance2.label = 'luminance2'
		luminance2_separated = nodetree.nodes.new("ShaderNodeSeparateXYZ")
		luminance2_separated.label = 'luminance2_sep'
		nodetree.links.new(luminance2.outputs[0], luminance2_separated.inputs[0])

		##############
		## R
		##############
		concentration1 = self.spectral_compositor_concentration(nodetree, node_spectral_factor1, node_spectral_tintingStrength1, luminance1_separated)
		concentration2 = self.spectral_compositor_concentration(nodetree, node_spectral_factor2, node_spectral_tintingStrength2, luminance2_separated)

		totalConcentration = nodetree.nodes.new("ShaderNodeMath")
		totalConcentration.operation = 'ADD'
		totalConcentration.label = 'totalConcentration'
		nodetree.links.new(concentration1.outputs[0], totalConcentration.inputs[0])
		nodetree.links.new(concentration2.outputs[0], totalConcentration.inputs[1])

		R = [0] * SPECTRAL_SIZE

		for i in range(SPECTRAL_SIZE):
			ksMix = nodetree.nodes.new("ShaderNodeValue")
			ksMix.outputs[0].default_value = 0

			KS_R1 = self.spectral_compositor_KS(nodetree, R1[i])
			KS_R2 = self.spectral_compositor_KS(nodetree, R2[i])

			ksMix_mult_R1 = nodetree.nodes.new("ShaderNodeMath")
			ksMix_mult_R1.operation = 'MULTIPLY'
			ksMix_mult_R1.label = 'ksMix_mult_R1'
			nodetree.links.new(KS_R1.outputs[0], ksMix_mult_R1.inputs[0])
			nodetree.links.new(concentration1.outputs[0], ksMix_mult_R1.inputs[1])

			ksMix_mult_R2 = nodetree.nodes.new("ShaderNodeMath")
			ksMix_mult_R2.operation = 'MULTIPLY'
			ksMix_mult_R2.label = 'ksMix_mult_R2'
			nodetree.links.new(KS_R2.outputs[0], ksMix_mult_R2.inputs[0])
			nodetree.links.new(concentration2.outputs[0], ksMix_mult_R2.inputs[1])
			
			ksMix_R1_and_R2 = nodetree.nodes.new("ShaderNodeMath")
			ksMix_R1_and_R2.operation = 'ADD'
			ksMix_R1_and_R2.label = 'ksMix_R1_and_R2'
			nodetree.links.new(ksMix_mult_R1.outputs[0], ksMix_R1_and_R2.inputs[0])
			nodetree.links.new(ksMix_mult_R2.outputs[0], ksMix_R1_and_R2.inputs[1])

			###########
			spectralR_divide = nodetree.nodes.new("ShaderNodeMath")
			spectralR_divide.operation = 'DIVIDE'
			ksMix_mult_R2.label = 'spectralR_divide'
			nodetree.links.new(ksMix_R1_and_R2.outputs[0], spectralR_divide.inputs[0])
			nodetree.links.new(totalConcentration.outputs[0], spectralR_divide.inputs[1])

			R[i] = self.spectral_compositor_KM(nodetree, spectralR_divide)

		reflectanceEnd = self.spectral_compositor_reflectance_to_xyz_p0(nodetree, R)
		reflectanceEnd.label = 'reflectanceEnd'
		node_mapRange.use_custom_color = True
		node_mapRange.color = (1, 0, 0)

		# print('READPIXEL : reflectanceEnd')
		# self.spectral_compositor_debugging_exit_visualizer(nodetree, reflectanceEnd, 932, 633)
		# return

		spectral_xyz_to_srgb_end = self.spectral_compositor_xyz_to_srgb(nodetree, reflectanceEnd)

		print('READPIXEL : spectral_xyz_to_srgb_end')
		self.spectral_compositor_debugging_exit_visualizer(nodetree, spectral_xyz_to_srgb_end, 932, 633)
		return

	#This function is based on spectral3_glsl.py, under MIT license by Ronald van Wijnen (see file)
	def spectral_compositor_KS(self, nodetree, R):
		node_subtract = nodetree.nodes.new("ShaderNodeMath")
		node_subtract.operation = 'SUBTRACT'
		node_subtract.inputs[0].default_value = 1
		nodetree.links.new(R.outputs[0], node_subtract.inputs[1])

		node_pow = nodetree.nodes.new("ShaderNodeMath")
		node_pow.operation = 'POWER'
		node_pow.label = 'node_pow'
		node_pow.inputs[1].default_value = 2
		nodetree.links.new(node_subtract.outputs[0], node_pow.inputs[0])

		node_multiplyR = nodetree.nodes.new("ShaderNodeMath")
		node_multiplyR.operation = 'MULTIPLY'
		node_multiplyR.outputs[0].default_value = 2
		nodetree.links.new(R.outputs[0], node_multiplyR.inputs[1])

		node_divide = nodetree.nodes.new("ShaderNodeMath")
		node_divide.operation = 'DIVIDE'
		nodetree.links.new(node_pow.outputs[0], node_divide.inputs[0])
		nodetree.links.new(node_multiplyR.outputs[0], node_divide.inputs[1])

		return node_divide
	
	#This function is based on spectral3_glsl.py, under MIT license by Ronald van Wijnen (see file)
	def spectral_compositor_xyz_to_srgb(self, nodetree, xyz_combo):

		xyz = nodetree.nodes.new("ShaderNodeSeparateXYZ")
		xyz.label = 'xyz'
		nodetree.links.new(xyz_combo.outputs[0], xyz.inputs[0])
	
		################
		## WRITTEN CUSTOM DOT PRODUCT
		###############

		######### R
		node_dotR_0 = nodetree.nodes.new("ShaderNodeMath")
		node_dotR_0.operation = 'MULTIPLY'
		node_dotR_0.inputs[0].default_value = 3.2409699419045200
		nodetree.links.new(xyz.outputs[0], node_dotR_0.inputs[1])

		node_dotR_1 = nodetree.nodes.new("ShaderNodeMath")
		node_dotR_1.operation = 'MULTIPLY'
		node_dotR_1.inputs[0].default_value = -1.537383177570090
		nodetree.links.new(xyz.outputs[1], node_dotR_1.inputs[1])

		node_dotR_2 = nodetree.nodes.new("ShaderNodeMath")
		node_dotR_2.operation = 'MULTIPLY'
		node_dotR_2.inputs[0].default_value = -0.4986107602930030
		nodetree.links.new(xyz.outputs[2], node_dotR_2.inputs[1])

		node_add_R_0 = nodetree.nodes.new("ShaderNodeMath")
		node_add_R_0.operation = 'ADD'
		nodetree.links.new(node_dotR_0.outputs[0], node_add_R_0.inputs[0])
		nodetree.links.new(node_dotR_1.outputs[0], node_add_R_0.inputs[1])

		node_dotProd_R = nodetree.nodes.new("ShaderNodeMath")
		node_dotProd_R.operation = 'ADD'
		nodetree.links.new(node_add_R_0.outputs[0], node_dotProd_R.inputs[0])
		nodetree.links.new(node_dotR_2.outputs[0], node_dotProd_R.inputs[1])

		# print('READPIXEL : node_dotProd_R')
		# self.spectral_compositor_debugging_exit_visualizer(nodetree, node_dotProd_R, 932, 633)
		# return

		######### G
		node_dotG_0 = nodetree.nodes.new("ShaderNodeMath")
		node_dotG_0.operation = 'MULTIPLY'
		node_dotG_0.inputs[0].default_value = -0.9692436362808790
		nodetree.links.new(xyz.outputs[0], node_dotG_0.inputs[1])

		node_dotG_1 = nodetree.nodes.new("ShaderNodeMath")
		node_dotG_1.operation = 'MULTIPLY'
		node_dotG_1.inputs[0].default_value = 1.875967501507720
		nodetree.links.new(xyz.outputs[1], node_dotG_1.inputs[1])

		node_dotG_2 = nodetree.nodes.new("ShaderNodeMath")
		node_dotG_2.operation = 'MULTIPLY'
		node_dotG_2.inputs[0].default_value = 0.0415550574071756
		nodetree.links.new(xyz.outputs[2], node_dotG_2.inputs[1])

		node_add_G_0 = nodetree.nodes.new("ShaderNodeMath")
		node_add_G_0.operation = 'ADD'
		nodetree.links.new(node_dotG_0.outputs[0], node_add_G_0.inputs[0])
		nodetree.links.new(node_dotG_1.outputs[0], node_add_G_0.inputs[1])

		node_dotProd_G = nodetree.nodes.new("ShaderNodeMath")
		node_dotProd_G.operation = 'ADD'
		nodetree.links.new(node_add_G_0.outputs[0], node_dotProd_G.inputs[0])
		nodetree.links.new(node_dotG_2.outputs[0], node_dotProd_G.inputs[1])

		########### B
		node_dotB_0 = nodetree.nodes.new("ShaderNodeMath")
		node_dotB_0.operation = 'MULTIPLY'
		node_dotB_0.inputs[0].default_value = 0.0556300796969936
		nodetree.links.new(xyz.outputs[0], node_dotB_0.inputs[1])

		node_dotB_1 = nodetree.nodes.new("ShaderNodeMath")
		node_dotB_1.operation = 'MULTIPLY'
		node_dotB_1.inputs[0].default_value = -0.203976958888976
		nodetree.links.new(xyz.outputs[1], node_dotB_1.inputs[1])

		node_dotB_2 = nodetree.nodes.new("ShaderNodeMath")
		node_dotB_2.operation = 'MULTIPLY'
		node_dotB_2.inputs[0].default_value = 1.0569715142428700
		nodetree.links.new(xyz.outputs[2], node_dotB_2.inputs[1])

		node_add_B_0 = nodetree.nodes.new("ShaderNodeMath")
		node_add_B_0.operation = 'ADD'
		nodetree.links.new(node_dotB_0.outputs[0], node_add_B_0.inputs[0])
		nodetree.links.new(node_dotB_1.outputs[0], node_add_B_0.inputs[1])

		node_dotProd_B = nodetree.nodes.new("ShaderNodeMath")
		node_dotProd_B.operation = 'ADD'
		nodetree.links.new(node_add_B_0.outputs[0], node_dotProd_B.inputs[0])
		nodetree.links.new(node_dotB_2.outputs[0], node_dotProd_B.inputs[1])

		#####

		dotProdCombo_custom = nodetree.nodes.new("ShaderNodeCombineXYZ")
		nodetree.links.new(node_dotProd_R.outputs[0], dotProdCombo_custom.inputs[0]) 
		nodetree.links.new(node_dotProd_G.outputs[0], dotProdCombo_custom.inputs[1]) 
		nodetree.links.new(node_dotProd_B.outputs[0], dotProdCombo_custom.inputs[2]) 

		return self.spectral_compositor_linear_to_srgb(nodetree, dotProdCombo_custom)

	def spectral_compositor_clamp_01(self, nodetree, inputNode):
		node_min = nodetree.nodes.new("ShaderNodeMath")
		node_min.operation = 'MINIMUM'
		node_min.inputs[1].default_value = 1
		nodetree.links.new(inputNode.outputs[0], node_min.inputs[0])
		# nodetree.links.new(inputNode.outputs['Result'], node_min.inputs[0])

		node_max = nodetree.nodes.new("ShaderNodeMath")
		node_max.operation = 'MAXIMUM'
		node_max.inputs[0].default_value = 0
		nodetree.links.new(node_min.outputs[0], node_max.inputs[1])

		return node_max
	
	#This function is based on spectral3_glsl.py, under MIT license by Ronald van Wijnen (see file)
	def spectral_compositor_linear_to_srgb(self, nodetree, dotProdCombo):

		dotProdCombo_sep = nodetree.nodes.new("ShaderNodeSeparateXYZ")
		dotProdCombo_sep.label = 'dotProdCombo_sep'
		nodetree.links.new(dotProdCombo.outputs[0], dotProdCombo_sep.inputs[0])

		compandR = self.spectral_compositor_compand(nodetree, dotProdCombo_sep, 0)
		compandG = self.spectral_compositor_compand(nodetree, dotProdCombo_sep, 1)
		compandB = self.spectral_compositor_compand(nodetree, dotProdCombo_sep, 2)

		compandR_clamped = self.spectral_compositor_clamp_01(nodetree, compandR)
		compandG_clamped = self.spectral_compositor_clamp_01(nodetree, compandG)
		compandB_clamped = self.spectral_compositor_clamp_01(nodetree, compandB)

		finalColor = nodetree.nodes.new("ShaderNodeCombineXYZ")
		# finalColor = nodetree.nodes.new("CompositorNodeCombineColor")

		nodetree.links.new(compandR_clamped.outputs[0], finalColor.inputs[0])
		nodetree.links.new(compandG_clamped.outputs[0], finalColor.inputs[1])
		nodetree.links.new(compandB_clamped.outputs[0], finalColor.inputs[2])

		return finalColor

	#This function is based on spectral3_glsl.py, under MIT license by Ronald van Wijnen (see file)
	def spectral_compositor_KM(self, nodetree, KS):
		km_power = nodetree.nodes.new("ShaderNodeMath")
		km_power.operation = 'POWER'
		km_power.label = 'km_power'
		km_power.inputs[1].default_value = 2
		nodetree.links.new(KS.outputs[0], km_power.inputs[0])

		km_mult = nodetree.nodes.new("ShaderNodeMath")
		km_mult.operation = 'MULTIPLY'
		km_mult.inputs[0].default_value = 2
		nodetree.links.new(KS.outputs[0], km_mult.inputs[1])

		km_add0 = nodetree.nodes.new("ShaderNodeMath")
		km_add0.operation = 'ADD'
		nodetree.links.new(km_power.outputs[0], km_add0.inputs[0])
		nodetree.links.new(km_mult.outputs[0], km_add0.inputs[1])

		km_sqrt = nodetree.nodes.new("ShaderNodeMath")
		km_sqrt.operation = 'SQRT'
		nodetree.links.new(km_add0.outputs[0], km_sqrt.inputs[0])

		####
		km_add1 = nodetree.nodes.new("ShaderNodeMath")
		km_add1.operation = 'ADD'
		km_add1.inputs[0].default_value = 1
		nodetree.links.new(KS.outputs[0], km_add1.inputs[1])

		km_subtract = nodetree.nodes.new("ShaderNodeMath")
		km_subtract.operation = 'SUBTRACT'
		nodetree.links.new(km_add1.outputs[0], km_subtract.inputs[0])
		nodetree.links.new(km_sqrt.outputs[0], km_subtract.inputs[1])

		return km_subtract

	#This function is based on spectral3_glsl.py, under MIT license by Ronald van Wijnen (see file)
	def spectral_compositor_concentration(self, nodetree, factor, tintingStrength, luminanceSep):
		concentration_factor_pow = nodetree.nodes.new("ShaderNodeMath")
		concentration_factor_pow.operation = 'POWER'
		concentration_factor_pow.label = 'concentration_factor_pow'
		concentration_factor_pow.inputs[1].default_value = 2
		nodetree.links.new(factor.outputs[0], concentration_factor_pow.inputs[0])

		concentration_tint_pow = nodetree.nodes.new("ShaderNodeMath")
		concentration_tint_pow.operation = 'POWER'
		concentration_tint_pow.label = 'concentration_tint_pow'
		concentration_tint_pow.inputs[1].default_value = 2
		nodetree.links.new(tintingStrength.outputs[0], concentration_tint_pow.inputs[0])

		node_concentration_multiply_0 = nodetree.nodes.new("ShaderNodeMath")
		node_concentration_multiply_0.operation = 'MULTIPLY'
		nodetree.links.new(concentration_factor_pow.outputs[0], node_concentration_multiply_0.inputs[0])
		nodetree.links.new(concentration_tint_pow.outputs[0], node_concentration_multiply_0.inputs[1])

		node_concentration_multiply_1 = nodetree.nodes.new("ShaderNodeMath")
		node_concentration_multiply_1.operation = 'MULTIPLY'
		nodetree.links.new(node_concentration_multiply_0.outputs[0], node_concentration_multiply_1.inputs[0])
		nodetree.links.new(luminanceSep.outputs[1], node_concentration_multiply_1.inputs[1])

		return node_concentration_multiply_1

	#This function is based on spectral3_glsl.py, under MIT license by Ronald van Wijnen (see file)
	def spectral_compositor_uncompand(self, nodetree, inputCombo, idx):
		# input_split = nodetree.nodes.new("CompositorNodeSeparateColor")
		input_split = nodetree.nodes.new("ShaderNodeSeparateXYZ")
		input_split.label = 'uncompand input split'
		# input_split.use_custom_color = True
		# input_split.color = (1, 0, 0)

		# nodetree.links.new(inputCombo.outputs["Image"], input_split.inputs[0])
		nodetree.links.new(inputCombo.outputs[0], input_split.inputs[0])

		node_math1 = nodetree.nodes.new("ShaderNodeMath")
		node_math1.operation = 'LESS_THAN'
		node_math1.label = 'math1 uncompand'
		node_math1.inputs[1].default_value = 0.04045
		nodetree.links.new(input_split.outputs[idx], node_math1.inputs[0])

		node_mix1 = nodetree.nodes.new("ShaderNodeMix")
		nodetree.links.new(node_math1.outputs[0], node_mix1.inputs[0])

		node_math2 = nodetree.nodes.new("ShaderNodeMath")
		node_math2.operation = 'DIVIDE'
		node_math1.label = 'math2 uncompand A'
		node_math2.inputs[1].default_value = 12.92
		nodetree.links.new(input_split.outputs[idx], node_math2.inputs[0])

		nodetree.links.new(node_math2.outputs[0], node_mix1.inputs['A'])

		node_math3 = nodetree.nodes.new("ShaderNodeMath")
		node_math3.operation = 'ADD'
		node_math1.label = 'math3 uncompand'
		node_math3.inputs[1].default_value = 0.055
		nodetree.links.new(input_split.outputs[idx], node_math3.inputs[0])

		node_math4 = nodetree.nodes.new("ShaderNodeMath")
		node_math4.operation = 'DIVIDE'
		node_math4.inputs[1].default_value = 1.055
		nodetree.links.new(node_math3.outputs[0], node_math4.inputs[0])

		node_math5 = nodetree.nodes.new("ShaderNodeMath")
		node_math5.operation = 'POWER'
		node_math5.label = 'node_math5 pow B'
		nodetree.links.new(node_math4.outputs[0], node_math5.inputs[0])
		nodetree.links.new(self.node_spectral_gamma.outputs[0], node_math5.inputs[1])

		nodetree.links.new(node_math5.outputs[0], node_mix1.inputs['B'])

		return node_mix1
	
	#This function is based on spectral3_glsl.py, under MIT license by Ronald van Wijnen (see file)
	def spectral_compositor_uncompand(self, nodetree, inputColorSplitNode):
		node_math1 = nodetree.nodes.new("ShaderNodeMath")
		node_math1.operation = 'LESS_THAN'
		node_math1.label = 'math1 uncompand'
		node_math1.inputs[1].default_value = 0.04045
		nodetree.links.new(inputColorSplitNode.outputs[0], node_math1.inputs[0])

		node_mix1 = nodetree.nodes.new("ShaderNodeMix")
		nodetree.links.new(node_math1.outputs[0], node_mix1.inputs[0])

		node_math2 = nodetree.nodes.new("ShaderNodeMath")
		node_math2.operation = 'DIVIDE'
		node_math1.label = 'math2 uncompand'
		node_math2.inputs[1].default_value = 12.92
		nodetree.links.new(inputColorSplitNode.outputs[0], node_math2.inputs[0])

		nodetree.links.new(node_math2.outputs[0], node_mix1.inputs['A'])

		node_math3 = nodetree.nodes.new("ShaderNodeMath")
		node_math3.operation = 'ADD'
		node_math1.label = 'math3 uncompand'
		node_math3.inputs[1].default_value = 0.055
		nodetree.links.new(inputColorSplitNode.outputs[0], node_math3.inputs[0])

		node_math4 = nodetree.nodes.new("ShaderNodeMath")
		node_math4.operation = 'DIVIDE'
		node_math4.inputs[1].default_value = 1.055
		nodetree.links.new(node_math3.outputs[0], node_math4.inputs[0])

		node_math5 = nodetree.nodes.new("ShaderNodeMath")
		node_math5.operation = 'POWER'
		node_math5.label = 'node_math5 pow'
		nodetree.links.new(node_math4.outputs[0], node_math5.inputs[0])
		nodetree.links.new(self.node_spectral_gamma.outputs[0], node_math5.inputs[1])

		nodetree.links.new(node_math5.outputs[0], node_mix1.inputs['B'])

		return node_mix1
	
	#This function is based on spectral3_glsl.py, under MIT license by Ronald van Wijnen (see file)
	def spectral_compositor_compand(self, nodetree, inputNode, idx):
		node_math1 = nodetree.nodes.new("ShaderNodeMath")
		node_math1.operation = 'LESS_THAN'
		node_math1.label = 'compand_nodeMath1'
		node_math1.inputs[1].default_value = 0.0031308 
		nodetree.links.new(inputNode.outputs[idx], node_math1.inputs[0])

		node_mix1 = nodetree.nodes.new("ShaderNodeMix")
		nodetree.links.new(node_math1.outputs[0], node_mix1.inputs[0])

		node_math2 = nodetree.nodes.new("ShaderNodeMath")
		node_math2.operation = 'MULTIPLY'
		node_math2.inputs[1].default_value = 12.92
		node_math2.label = 'compand_nodeMath2'
		nodetree.links.new(inputNode.outputs[idx], node_math2.inputs[0])

		nodetree.links.new(node_math2.outputs[0], node_mix1.inputs['A'])

		#################
		node_math3 = nodetree.nodes.new("ShaderNodeMath")
		node_math3.operation = 'DIVIDE'
		node_math3.inputs[0].default_value = 1.0
		nodetree.links.new(self.node_spectral_gamma.outputs[0], node_math3.inputs[1])

		node_math4 = nodetree.nodes.new("ShaderNodeMath")
		node_math4.operation = 'POWER'
		node_math4.label = 'node_math4 pow'
		nodetree.links.new(inputNode.outputs[idx], node_math4.inputs[0])
		nodetree.links.new(node_math3.outputs[0], node_math4.inputs[1])

		node_math5 = nodetree.nodes.new("ShaderNodeMath")
		node_math5.operation = 'MULTIPLY'
		node_math5.inputs[0].default_value = 1.055
		nodetree.links.new(node_math4.outputs[0], node_math5.inputs[1])

		node_math6 = nodetree.nodes.new("ShaderNodeMath")
		node_math6.operation = 'SUBTRACT'
		node_math6.inputs[1].default_value = .055
		nodetree.links.new(node_math5.outputs[0], node_math6.inputs[0])

		nodetree.links.new(node_math6.outputs[0], node_mix1.inputs['B'])
		
		return node_mix1
		
	#This function is based on spectral3_glsl.py, under MIT license by Ronald van Wijnen (see file)
	def spectral_compositor_linear_to_reflectance(self, nodetree, lrgbCombo):
		
		lrgbCombo_sep = nodetree.nodes.new("ShaderNodeSeparateXYZ")
		lrgbCombo_sep.label = 'lrgbCombo_sep'
		nodetree.links.new(lrgbCombo.outputs[0], lrgbCombo_sep.inputs[0])

		node_w = nodetree.nodes.new("ShaderNodeMath")
		node_w.operation = 'MINIMUM'
		node_w.label = 'L2R_w'
		nodetree.links.new(lrgbCombo_sep.outputs[0], node_w.inputs[0])

		node_w_min_1 = nodetree.nodes.new("ShaderNodeMath")
		node_w_min_1.operation = 'MINIMUM'
		nodetree.links.new(lrgbCombo_sep.outputs[1], node_w_min_1.inputs[0])
		nodetree.links.new(lrgbCombo_sep.outputs[2], node_w_min_1.inputs[1])
		nodetree.links.new(node_w_min_1.outputs[0], node_w.inputs[1])

		node_w3 = nodetree.nodes.new("ShaderNodeCombineXYZ")
		nodetree.links.new(node_w.outputs[0], node_w3.inputs[0])
		nodetree.links.new(node_w.outputs[0], node_w3.inputs[1])
		nodetree.links.new(node_w.outputs[0], node_w3.inputs[2])

		node_lrgb_minus_w3 = nodetree.nodes.new("ShaderNodeVectorMath")
		node_lrgb_minus_w3.operation = 'SUBTRACT'
		node_lrgb_minus_w3.label = 'lrgb - w3'
		nodetree.links.new(lrgbCombo.outputs[0], node_lrgb_minus_w3.inputs[0])
		nodetree.links.new(node_w3.outputs[0], node_lrgb_minus_w3.inputs[1])

		node_lrgb_split = nodetree.nodes.new("ShaderNodeSeparateXYZ")
		nodetree.links.new(node_lrgb_minus_w3.outputs[0], node_lrgb_split.inputs[0])

		# CMY
		node_c = nodetree.nodes.new("ShaderNodeMath")
		node_c.operation = 'MINIMUM'
		node_c.label = "L2R C"
		nodetree.links.new(node_lrgb_split.outputs[1], node_c.inputs[0])
		nodetree.links.new(node_lrgb_split.outputs[2], node_c.inputs[1])

		node_m = nodetree.nodes.new("ShaderNodeMath")
		node_m.operation = 'MINIMUM'
		node_m.label = "L2R M"
		nodetree.links.new(node_lrgb_split.outputs[0], node_m.inputs[0])
		nodetree.links.new(node_lrgb_split.outputs[2], node_m.inputs[1])

		node_y = nodetree.nodes.new("ShaderNodeMath")
		node_y.operation = 'MINIMUM'
		node_y.label = "L2R Y"
		nodetree.links.new(node_lrgb_split.outputs[0], node_y.inputs[0])
		nodetree.links.new(node_lrgb_split.outputs[1], node_y.inputs[1])

		########## RGB
		#### R #####
		node_r = nodetree.nodes.new("ShaderNodeMath")
		node_r.operation = 'MINIMUM'
		node_r.label = 'L2R R'

		node_r_max0 = nodetree.nodes.new("ShaderNodeMath")
		node_r_max0.operation = 'MAXIMUM'
		node_r_max0.inputs[0].default_value = 0

		node_r_max0_subtract = nodetree.nodes.new("ShaderNodeMath")
		node_r_max0_subtract.operation = 'SUBTRACT'
		nodetree.links.new(node_lrgb_split.outputs[0], node_r_max0_subtract.inputs[0])
		nodetree.links.new(node_lrgb_split.outputs[2], node_r_max0_subtract.inputs[1])
		nodetree.links.new(node_r_max0_subtract.outputs[0], node_r_max0.inputs[1])

		node_r_max1 = nodetree.nodes.new("ShaderNodeMath")
		node_r_max1.operation = 'MAXIMUM'
		node_r_max1.inputs[0].default_value = 0

		node_r_max1_subtract = nodetree.nodes.new("ShaderNodeMath")
		node_r_max1_subtract.operation = 'SUBTRACT'
		nodetree.links.new(node_lrgb_split.outputs[0], node_r_max1_subtract.inputs[0])
		nodetree.links.new(node_lrgb_split.outputs[1], node_r_max1_subtract.inputs[1])
		nodetree.links.new(node_r_max1_subtract.outputs[0], node_r_max1.inputs[1])

		nodetree.links.new(node_r_max0.outputs[0], node_r.inputs[0])
		nodetree.links.new(node_r_max1.outputs[0], node_r.inputs[1])

		#### G #####
		node_g = nodetree.nodes.new("ShaderNodeMath")
		node_g.operation = 'MINIMUM'
		node_g.label = 'L2R G'

		node_g_max0 = nodetree.nodes.new("ShaderNodeMath")
		node_g_max0.operation = 'MAXIMUM'
		node_g_max0.inputs[0].default_value = 0

		node_g_max0_subtract = nodetree.nodes.new("ShaderNodeMath")
		node_g_max0_subtract.operation = 'SUBTRACT'
		nodetree.links.new(node_lrgb_split.outputs[1], node_g_max0_subtract.inputs[0])
		nodetree.links.new(node_lrgb_split.outputs[2], node_g_max0_subtract.inputs[1])
		nodetree.links.new(node_g_max0_subtract.outputs[0], node_g_max0.inputs[1])

		node_g_max1 = nodetree.nodes.new("ShaderNodeMath")
		node_g_max1.operation = 'MAXIMUM'
		node_g_max1.inputs[0].default_value = 0

		node_g_max1_subtract = nodetree.nodes.new("ShaderNodeMath")
		node_g_max1_subtract.operation = 'SUBTRACT'
		nodetree.links.new(node_lrgb_split.outputs[1], node_g_max1_subtract.inputs[0])
		nodetree.links.new(node_lrgb_split.outputs[0], node_g_max1_subtract.inputs[1])
		nodetree.links.new(node_g_max1_subtract.outputs[0], node_g_max1.inputs[1])

		nodetree.links.new(node_g_max0.outputs[0], node_g.inputs[0])
		nodetree.links.new(node_g_max1.outputs[0], node_g.inputs[1])

		#### B #####
		node_b = nodetree.nodes.new("ShaderNodeMath")
		node_b.operation = 'MINIMUM'
		node_b.label = 'L2R B'

		node_b_max0 = nodetree.nodes.new("ShaderNodeMath")
		node_b_max0.operation = 'MAXIMUM'
		node_b_max0.inputs[0].default_value = 0

		node_b_max0_subtract = nodetree.nodes.new("ShaderNodeMath")
		node_b_max0_subtract.operation = 'SUBTRACT'
		nodetree.links.new(node_lrgb_split.outputs[2], node_b_max0_subtract.inputs[0])
		nodetree.links.new(node_lrgb_split.outputs[1], node_b_max0_subtract.inputs[1])
		nodetree.links.new(node_b_max0_subtract.outputs[0], node_b_max0.inputs[1])

		node_b_max1 = nodetree.nodes.new("ShaderNodeMath")
		node_b_max1.operation = 'MAXIMUM'
		node_b_max1.inputs[0].default_value = 0

		node_b_max1_subtract = nodetree.nodes.new("ShaderNodeMath")
		node_b_max1_subtract.operation = 'SUBTRACT'
		nodetree.links.new(node_lrgb_split.outputs[2], node_b_max1_subtract.inputs[0])
		nodetree.links.new(node_lrgb_split.outputs[0], node_b_max1_subtract.inputs[1])
		nodetree.links.new(node_b_max1_subtract.outputs[0], node_b_max1.inputs[1])

		nodetree.links.new(node_b_max0.outputs[0], node_b.inputs[0])
		nodetree.links.new(node_b_max1.outputs[0], node_b.inputs[1])

		# print('READPIXEL : node_lrgb1_combo')
		# self.spectral_compositor_debugging_exit_visualizer(nodetree, node_cmy_debug_combo, 932, 633)
		# # self.spectral_compositor_debugging_exit_visualizer(nodetree, node_rgb_debug_combo, 932, 633)
		# return

		############
		### OUTPUT R
		############
		SPECTRAL_EPSILON = 0.0000000000000001
		SPECTRAL_SIZE = 38

		R = [0] * SPECTRAL_SIZE

		R[ 0] = self.spectral_compositor_R(nodetree, node_w, 1.0011607271876400, node_c, 0.9705850013229620, node_m, 0.9906735573199880, node_y, 0.0210523371789306, node_r, 0.0315605737777207, node_g, 0.0095560747554212, node_b, 0.9794047525020140)
		R[ 1] = self.spectral_compositor_R(nodetree, node_w, 1.0011606515972800, node_c, 0.9705924981434250, node_m, 0.9906715249619790, node_y, 0.0210564627517414, node_r, 0.0315520718330149, node_g, 0.0095581580120851, node_b, 0.9794007068431300)
		R[ 2] = self.spectral_compositor_R(nodetree, node_w, 1.0011603192274700, node_c, 0.9706253487298910, node_m, 0.9906625823534210, node_y, 0.0210746178695038, node_r, 0.0315148215513658, node_g, 0.0095673245444588, node_b, 0.9793829034702610)
		R[ 3] = self.spectral_compositor_R(nodetree, node_w, 1.0011586727078900, node_c, 0.9707868061190170, node_m, 0.9906181076447950, node_y, 0.0211649058448753, node_r, 0.0313318044982702, node_g, 0.0096129126297349, node_b, 0.9792943649455940)
		R[ 4] = self.spectral_compositor_R(nodetree, node_w, 1.0011525984455200, node_c, 0.9713686732282480, node_m, 0.9904514808787100, node_y, 0.0215027957272504, node_r, 0.0306729857725527, node_g, 0.0097837090401843, node_b, 0.9789630146085700)
		R[ 5] = self.spectral_compositor_R(nodetree, node_w, 1.0011325252899800, node_c, 0.9731632306212520, node_m, 0.9898710814002040, node_y, 0.0226738799041561, node_r, 0.0286480476989607, node_g, 0.0103786227058710, node_b, 0.9778144666940430)
		R[ 6] = self.spectral_compositor_R(nodetree, node_w, 1.0010850066332700, node_c, 0.9767402231587650, node_m, 0.9882866087596400, node_y, 0.0258235649693629, node_r, 0.0246450407045709, node_g, 0.0120026452378567, node_b, 0.9747243211338360)
		R[ 7] = self.spectral_compositor_R(nodetree, node_w, 1.0009968788945300, node_c, 0.9815876054913770, node_m, 0.9842906927975040, node_y, 0.0334879385639851, node_r, 0.0192960753663651, node_g, 0.0160977721473922, node_b, 0.9671984823439730)
		R[ 8] = self.spectral_compositor_R(nodetree, node_w, 1.0008652515227400, node_c, 0.9862802656529490, node_m, 0.9739349056253060, node_y, 0.0519069663740307, node_r, 0.0142066612220556, node_g, 0.0267061902231680, node_b, 0.9490796575305750)
		R[ 9] = self.spectral_compositor_R(nodetree, node_w, 1.0006962900094000, node_c, 0.9899491476891340, node_m, 0.9418178384601450, node_y, 0.1007490148334730, node_r, 0.0102942608878609, node_g, 0.0595555440185881, node_b, 0.9008501289409770)
		R[10] = self.spectral_compositor_R(nodetree, node_w, 1.0005049611488800, node_c, 0.9924927015384200, node_m, 0.8173903261951560, node_y, 0.2391298997068470, node_r, 0.0076191460521811, node_g, 0.1860398265328260, node_b, 0.7631504454622400)
		R[11] = self.spectral_compositor_R(nodetree, node_w, 1.0003080818799200, node_c, 0.9941456804052560, node_m, 0.4324728050657290, node_y, 0.5348043122727480, node_r, 0.0058980410835420, node_g, 0.5705798201161590, node_b, 0.4659221716493190)
		R[12] = self.spectral_compositor_R(nodetree, node_w, 1.0001196660201300, node_c, 0.9951839750332120, node_m, 0.1384539782588700, node_y, 0.7978075786430300, node_r, 0.0048233247781713, node_g, 0.8614677684002920, node_b, 0.2012632804510050)
		R[13] = self.spectral_compositor_R(nodetree, node_w, 0.9999527659684070, node_c, 0.9957567501108180, node_m, 0.0537347216940033, node_y, 0.9114498940673840, node_r, 0.0042298748350633, node_g, 0.9458790897676580, node_b, 0.0877524413419623)
		R[14] = self.spectral_compositor_R(nodetree, node_w, 0.9998218368992970, node_c, 0.9959128182867100, node_m, 0.0292174996673231, node_y, 0.9537979630045070, node_r, 0.0040599171299341, node_g, 0.9704654864743050, node_b, 0.0457176793291679)
		R[15] = self.spectral_compositor_R(nodetree, node_w, 0.9997386095575930, node_c, 0.9956061578345280, node_m, 0.0213136517508590, node_y, 0.9712416154654290, node_r, 0.0043533695594676, node_g, 0.9784136302844500, node_b, 0.0284706050521843)
		R[16] = self.spectral_compositor_R(nodetree, node_w, 0.9997095516396120, node_c, 0.9945976009618540, node_m, 0.0201349530181136, node_y, 0.9793031238075880, node_r, 0.0053434425970201, node_g, 0.9795890314112240, node_b, 0.0205271767569850)
		R[17] = self.spectral_compositor_R(nodetree, node_w, 0.9997319302106270, node_c, 0.9922157154923700, node_m, 0.0241323096280662, node_y, 0.9833801195075750, node_r, 0.0076917201010463, node_g, 0.9755335369086320, node_b, 0.0165302792310211)
		R[18] = self.spectral_compositor_R(nodetree, node_w, 0.9997994363461950, node_c, 0.9862364527832490, node_m, 0.0372236145223627, node_y, 0.9854612465677550, node_r, 0.0135969795736536, node_g, 0.9622887553978130, node_b, 0.0145135107212858)
		R[19] = self.spectral_compositor_R(nodetree, node_w, 0.9999003303166710, node_c, 0.9679433372645410, node_m, 0.0760506552706601, node_y, 0.9864350469766050, node_r, 0.0316975442661115, node_g, 0.9231215745131200, node_b, 0.0136003508637687)
		R[20] = self.spectral_compositor_R(nodetree, node_w, 1.0000204065261100, node_c, 0.8912850042449430, node_m, 0.2053754719423990, node_y, 0.9867382506701410, node_r, 0.1078611963552490, node_g, 0.7934340189431110, node_b, 0.0133604258769571)
		R[21] = self.spectral_compositor_R(nodetree, node_w, 1.0001447879365800, node_c, 0.5362024778620530, node_m, 0.5412689034604390, node_y, 0.9866178824450320, node_r, 0.4638126031687040, node_g, 0.4592701359024290, node_b, 0.0135488943145680)
		R[22] = self.spectral_compositor_R(nodetree, node_w, 1.0002599790341200, node_c, 0.1541081190018780, node_m, 0.8158416850864860, node_y, 0.9862777767586430, node_r, 0.8470554052720110, node_g, 0.1855741036663030, node_b, 0.0139594356366992)
		R[23] = self.spectral_compositor_R(nodetree, node_w, 1.0003557969708900, node_c, 0.0574575093228929, node_m, 0.9128177041239760, node_y, 0.9858605924440560, node_r, 0.9431854093939180, node_g, 0.0881774959955372, node_b, 0.0144434255753570)
		R[24] = self.spectral_compositor_R(nodetree, node_w, 1.0004275378026900, node_c, 0.0315349873107007, node_m, 0.9463398301669620, node_y, 0.9854749276762100, node_r, 0.9688621506965580, node_g, 0.0543630228766700, node_b, 0.0148854440621406)
		R[25] = self.spectral_compositor_R(nodetree, node_w, 1.0004762334488800, node_c, 0.0222633920086335, node_m, 0.9599276963319910, node_y, 0.9851769347655580, node_r, 0.9780306674736030, node_g, 0.0406288447060719, node_b, 0.0152254296999746)
		R[26] = self.spectral_compositor_R(nodetree, node_w, 1.0005072096750800, node_c, 0.0182022841492439, node_m, 0.9662605952303120, node_y, 0.9849715740141810, node_r, 0.9820436438543060, node_g, 0.0342215204316970, node_b, 0.0154592848180209)
		R[27] = self.spectral_compositor_R(nodetree, node_w, 1.0005251915637300, node_c, 0.0162990559732640, node_m, 0.9693259700584240, node_y, 0.9848463034157120, node_r, 0.9839236237187070, node_g, 0.0311185790956966, node_b, 0.0156018026485961)
		R[28] = self.spectral_compositor_R(nodetree, node_w, 1.0005350960689600, node_c, 0.0153656239334613, node_m, 0.9708545367213990, node_y, 0.9847753518111990, node_r, 0.9848454841543820, node_g, 0.0295708898336134, node_b, 0.0156824871281936)
		R[29] = self.spectral_compositor_R(nodetree, node_w, 1.0005402209748200, node_c, 0.0149111568733976, node_m, 0.9716050665281280, node_y, 0.9847380666252650, node_r, 0.9852942758145960, node_g, 0.0288108739348928, node_b, 0.0157248764360615)
		R[30] = self.spectral_compositor_R(nodetree, node_w, 1.0005427281678400, node_c, 0.0146954339898235, node_m, 0.9719627697573920, node_y, 0.9847196483117650, node_r, 0.9855072952198250, node_g, 0.0284486271324597, node_b, 0.0157458108784121)
		R[31] = self.spectral_compositor_R(nodetree, node_w, 1.0005438956908700, node_c, 0.0145964146717719, node_m, 0.9721272722745090, node_y, 0.9847110233919390, node_r, 0.9856050715398370, node_g, 0.0282820301724731, node_b, 0.0157556123350225)
		R[32] = self.spectral_compositor_R(nodetree, node_w, 1.0005444821215100, node_c, 0.0145470156699655, node_m, 0.9722094177458120, node_y, 0.9847066833006760, node_r, 0.9856538499335780, node_g, 0.0281988376490237, node_b, 0.0157605443964911)
		R[33] = self.spectral_compositor_R(nodetree, node_w, 1.0005447695999200, node_c, 0.0145228771899495, node_m, 0.9722495776784240, node_y, 0.9847045543930910, node_r, 0.9856776850338830, node_g, 0.0281581655342037, node_b, 0.0157629637515278)
		R[34] = self.spectral_compositor_R(nodetree, node_w, 1.0005448988776200, node_c, 0.0145120341118965, node_m, 0.9722676219987420, node_y, 0.9847035963093700, node_r, 0.9856883918061220, node_g, 0.0281398910216386, node_b, 0.0157640525629106)
		R[35] = self.spectral_compositor_R(nodetree, node_w, 1.0005449625468900, node_c, 0.0145066940939832, node_m, 0.9722765094621500, node_y, 0.9847031240775520, node_r, 0.9856936646900310, node_g, 0.0281308901665811, node_b, 0.0157645892329510)
		R[36] = self.spectral_compositor_R(nodetree, node_w, 1.0005449892705800, node_c, 0.0145044507314479, node_m, 0.9722802433068740, node_y, 0.9847029256150900, node_r, 0.9856958798482050, node_g, 0.0281271086805816, node_b, 0.0157648147772649)
		R[37] = self.spectral_compositor_R(nodetree, node_w, 1.0005449969930000, node_c, 0.0145038009464639, node_m, 0.9722813248265600, node_y, 0.9847028681227950, node_r, 0.9856965214637620, node_g, 0.0281260133612096, node_b, 0.0157648801149616)

		return R
	
	#This function is based on spectral3_glsl.py, under MIT license by Ronald van Wijnen (see file)
	def spectral_compositor_reflectance_to_xyz_p0(self, nodetree, R):
		xyz_start = nodetree.nodes.new("FunctionNodeInputVector")
		xyz_start.vector[0] = 0.0
		xyz_start.vector[1] = 0.0
		xyz_start.vector[2] = 0.0

		l0 = self.spectral_compositor_reflectance_to_xyz_p1(nodetree, R[0], 0.0000646919989576, 0.0000018442894440, 0.0003050171476380, xyz_start)
		l1 = self.spectral_compositor_reflectance_to_xyz_p1(nodetree, R[1], 0.0002194098998132, 0.0000062053235865, 0.0010368066663574, l0)
		l2 = self.spectral_compositor_reflectance_to_xyz_p1(nodetree, R[2], 0.0011205743509343, 0.0000310096046799, 0.0053131363323992, l1)
		l3 = self.spectral_compositor_reflectance_to_xyz_p1(nodetree, R[3], 0.0037666134117111, 0.0001047483849269, 0.0179543925899536, l2)
		l4 = self.spectral_compositor_reflectance_to_xyz_p1(nodetree, R[4], 0.0118805536037990, 0.0003536405299538, 0.0570775815345485, l3)
		l5 = self.spectral_compositor_reflectance_to_xyz_p1(nodetree, R[5], 0.0232864424191771, 0.0009514714056444, 0.1136516189362870, l4)
		l6 = self.spectral_compositor_reflectance_to_xyz_p1(nodetree, R[6], 0.0345594181969747, 0.0022822631748318, 0.1733587261835500, l5)
		l7 = self.spectral_compositor_reflectance_to_xyz_p1(nodetree, R[7], 0.0372237901162006, 0.0042073290434730, 0.1962065755586570, l6)
		l8 = self.spectral_compositor_reflectance_to_xyz_p1(nodetree, R[8], 0.0324183761091486, 0.0066887983719014, 0.1860823707062960, l7)
		l9 = self.spectral_compositor_reflectance_to_xyz_p1(nodetree, R[9], 0.0212332056093810, 0.0098883960193565, 0.1399504753832070, l8)
		l10 = self.spectral_compositor_reflectance_to_xyz_p1(nodetree, R[10], 0.0104909907685421, 0.0152494514496311, 0.0891745294268649, l9)
		l11 = self.spectral_compositor_reflectance_to_xyz_p1(nodetree, R[11], 0.0032958375797931, 0.0214183109449723, 0.0478962113517075, l10)
		l12 = self.spectral_compositor_reflectance_to_xyz_p1(nodetree, R[12], 0.0005070351633801, 0.0334229301575068, 0.0281456253957952, l11)
		l13 = self.spectral_compositor_reflectance_to_xyz_p1(nodetree, R[13], 0.0009486742057141, 0.0513100134918512, 0.0161376622950514, l12)
		l14 = self.spectral_compositor_reflectance_to_xyz_p1(nodetree, R[14], 0.0062737180998318, 0.0704020839399490, 0.0077591019215214, l13)
		l15 = self.spectral_compositor_reflectance_to_xyz_p1(nodetree, R[15], 0.0168646241897775, 0.0878387072603517, 0.0042961483736618, l14)
		l16 = self.spectral_compositor_reflectance_to_xyz_p1(nodetree, R[16], 0.0286896490259810, 0.0942490536184085, 0.0020055092122156, l15)
		l17 = self.spectral_compositor_reflectance_to_xyz_p1(nodetree, R[17], 0.0426748124691731, 0.0979566702718931, 0.0008614711098802, l16)
		l18 = self.spectral_compositor_reflectance_to_xyz_p1(nodetree, R[18], 0.0562547481311377, 0.0941521856862608, 0.0003690387177652, l17)
		l19 = self.spectral_compositor_reflectance_to_xyz_p1(nodetree, R[19], 0.0694703972677158, 0.0867810237486753, 0.0001914287288574, l18)
		l20 = self.spectral_compositor_reflectance_to_xyz_p1(nodetree, R[20], 0.0830531516998291, 0.0788565338632013, 0.0001495555858975, l19)
		l21 = self.spectral_compositor_reflectance_to_xyz_p1(nodetree, R[21], 0.0861260963002257, 0.0635267026203555, 0.0000923109285104, l20)
		l22 = self.spectral_compositor_reflectance_to_xyz_p1(nodetree, R[22], 0.0904661376847769, 0.0537414167568200, 0.0000681349182337, l21)
		l23 = self.spectral_compositor_reflectance_to_xyz_p1(nodetree, R[23], 0.0850038650591277, 0.0426460643574120, 0.0000288263655696, l22)
		l24 = self.spectral_compositor_reflectance_to_xyz_p1(nodetree, R[24], 0.0709066691074488, 0.0316173492792708, 0.0000157671820553, l23)
		l25 = self.spectral_compositor_reflectance_to_xyz_p1(nodetree, R[25], 0.0506288916373645, 0.0208852059213910, 0.0000039406041027, l24)
		l26 = self.spectral_compositor_reflectance_to_xyz_p1(nodetree, R[26], 0.0354739618852640, 0.0138601101360152, 0.0000015840125870, l25)
		l27 = self.spectral_compositor_reflectance_to_xyz_p1(nodetree, R[27], 0.0214682102597065, 0.0081026402038399, 0.0000000000000000, l26)
		l28 = self.spectral_compositor_reflectance_to_xyz_p1(nodetree, R[28], 0.0125164567619117, 0.0046301022588030, 0.0000000000000000, l27)
		l29 = self.spectral_compositor_reflectance_to_xyz_p1(nodetree, R[29], 0.0068045816390165, 0.0024913800051319, 0.0000000000000000, l28)
		l30 = self.spectral_compositor_reflectance_to_xyz_p1(nodetree, R[30], 0.0034645657946526, 0.0012593033677378, 0.0000000000000000, l29)
		l31 = self.spectral_compositor_reflectance_to_xyz_p1(nodetree, R[31], 0.0014976097506959, 0.0005416465221680, 0.0000000000000000, l30)
		l32 = self.spectral_compositor_reflectance_to_xyz_p1(nodetree, R[32], 0.0007697004809280, 0.0002779528920067, 0.0000000000000000, l31)
		l33 = self.spectral_compositor_reflectance_to_xyz_p1(nodetree, R[33], 0.0004073680581315, 0.0001471080673854, 0.0000000000000000, l32)
		l34 = self.spectral_compositor_reflectance_to_xyz_p1(nodetree, R[34], 0.0001690104031614, 0.0000610327472927, 0.0000000000000000, l33)
		l35 = self.spectral_compositor_reflectance_to_xyz_p1(nodetree, R[35], 0.0000952245150365, 0.0000343873229523, 0.0000000000000000, l34)
		l36 = self.spectral_compositor_reflectance_to_xyz_p1(nodetree, R[36], 0.0000490309872958, 0.0000177059860053, 0.0000000000000000, l35)
		l37 = self.spectral_compositor_reflectance_to_xyz_p1(nodetree, R[37], 0.0000199961492222, 0.0000072209749130, 0.0000000000000000, l36)

		return l37
	
	#This function is based on spectral3_glsl.py, under MIT license by Ronald van Wijnen (see file)
	def spectral_compositor_reflectance_to_xyz_p1(self, nodetree, R1_or_R2, v0, v1, v2, inputAdd):
		#0
		R1_or_R2_mult = nodetree.nodes.new("ShaderNodeVectorMath")
		R1_or_R2_mult.operation = 'MULTIPLY'
		R1_or_R2_mult.inputs[1].default_value[0] = v0
		R1_or_R2_mult.inputs[1].default_value[1] = v1
		R1_or_R2_mult.inputs[1].default_value[2] = v2
		nodetree.links.new(R1_or_R2.outputs[0], R1_or_R2_mult.inputs[0])

		node_add = nodetree.nodes.new("ShaderNodeVectorMath")
		node_add.operation = 'ADD'
		nodetree.links.new(inputAdd.outputs[0], node_add.inputs[0])
		nodetree.links.new(R1_or_R2_mult.outputs[0], node_add.inputs[1])

		return node_add

	#This function is based on spectral3_glsl.py, under MIT license by Ronald van Wijnen (see file)
	def spectral_compositor_R(self, nodetree, node_w, valW, node_c, valC, node_m, valM, node_y, valY, node_r, valR, node_g, valG, node_b, valB):
		SPECTRAL_EPSILON = 0.0000000000000001

		#multiply pairs and then add all

		#W
		w = nodetree.nodes.new("ShaderNodeMath")
		w.operation = 'MULTIPLY'

		node_valW = nodetree.nodes.new("ShaderNodeValue")
		node_valW.outputs[0].default_value = valW
		nodetree.links.new(node_w.outputs[0], w.inputs[0])
		nodetree.links.new(node_valW.outputs[0], w.inputs[1])

		#C
		c = nodetree.nodes.new("ShaderNodeMath")
		c.operation = 'MULTIPLY'

		node_valC = nodetree.nodes.new("ShaderNodeValue")
		node_valC.outputs[0].default_value = valC
		nodetree.links.new(node_c.outputs[0], c.inputs[0])
		nodetree.links.new(node_valC.outputs[0], c.inputs[1])

		# M
		m = nodetree.nodes.new("ShaderNodeMath")
		m.operation = 'MULTIPLY'

		node_valM = nodetree.nodes.new("ShaderNodeValue")
		node_valM.outputs[0].default_value = valM
		nodetree.links.new(node_m.outputs[0], m.inputs[0])
		nodetree.links.new(node_valM.outputs[0], m.inputs[1])

		# Y
		y = nodetree.nodes.new("ShaderNodeMath")
		y.operation = 'MULTIPLY'

		node_valY = nodetree.nodes.new("ShaderNodeValue")
		node_valY.outputs[0].default_value = valY
		nodetree.links.new(node_y.outputs[0], y.inputs[0])
		nodetree.links.new(node_valY.outputs[0], y.inputs[1])

		# R
		r = nodetree.nodes.new("ShaderNodeMath")
		r.operation = 'MULTIPLY'

		node_valR = nodetree.nodes.new("ShaderNodeValue")
		node_valR.outputs[0].default_value = valR
		nodetree.links.new(node_r.outputs[0], r.inputs[0])
		nodetree.links.new(node_valR.outputs[0], r.inputs[1])

		# G
		g = nodetree.nodes.new("ShaderNodeMath")
		g.operation = 'MULTIPLY'

		node_valG = nodetree.nodes.new("ShaderNodeValue")
		node_valG.outputs[0].default_value = valG
		nodetree.links.new(node_g.outputs[0], g.inputs[0])
		nodetree.links.new(node_valG.outputs[0], g.inputs[1])

		# B
		b = nodetree.nodes.new("ShaderNodeMath")
		b.operation = 'MULTIPLY'

		node_valB = nodetree.nodes.new("ShaderNodeValue")
		node_valB.outputs[0].default_value = valB
		nodetree.links.new(node_b.outputs[0], b.inputs[0])
		nodetree.links.new(node_valB.outputs[0], b.inputs[1])


		######## ADD 0
		add_cm = nodetree.nodes.new("ShaderNodeMath")
		add_cm.operation = 'ADD'
		nodetree.links.new(c.outputs[0], add_cm.inputs[0])
		nodetree.links.new(m.outputs[0], add_cm.inputs[1])

		add_yr = nodetree.nodes.new("ShaderNodeMath")
		add_yr.operation = 'ADD'
		nodetree.links.new(y.outputs[0], add_yr.inputs[0])
		nodetree.links.new(r.outputs[0], add_yr.inputs[1])

		add_gb = nodetree.nodes.new("ShaderNodeMath")
		add_gb.operation = 'ADD'
		nodetree.links.new(g.outputs[0], add_gb.inputs[0])
		nodetree.links.new(b.outputs[0], add_gb.inputs[1])

		######## ADD 1
		add_cmyr = nodetree.nodes.new("ShaderNodeMath")
		add_cm.operation = 'ADD'
		nodetree.links.new(add_cm.outputs[0], add_cmyr.inputs[0])
		nodetree.links.new(add_yr.outputs[0], add_cmyr.inputs[1])

		add_gbw = nodetree.nodes.new("ShaderNodeMath")
		add_cm.operation = 'ADD'
		nodetree.links.new(add_gb.outputs[0], add_gbw.inputs[0])
		nodetree.links.new(w.outputs[0], add_gbw.inputs[1])

		######## ADD FINAL
		add_final = nodetree.nodes.new("ShaderNodeMath")
		add_cm.operation = 'ADD'
		nodetree.links.new(add_cmyr.outputs[0], add_final.inputs[0])
		nodetree.links.new(add_gbw.outputs[0], add_final.inputs[1])

		R_out = nodetree.nodes.new("ShaderNodeMath")
		R_out.operation = 'MAXIMUM'
		R_out.inputs[0].default_value = SPECTRAL_EPSILON
		nodetree.links.new(add_final.outputs[0], R_out.inputs[1])

		return R_out

	def autoArrangeNodes(self, nodetree, x_spacing=250, y_spacing=250):
		# x_spacing=250
		# y_spacing=250
		"""
		Auto-arranges nodes in the active compositor graph based on flow dependencies.
		"""
		# 1. Separate nodes into roots (no inputs) and dependent nodes
		roots = []
		dependents = []
		for node in nodetree.nodes:
			has_inputs = False
			for input_socket in node.inputs:
				if input_socket.is_linked:
					has_inputs = True
					break
			if has_inputs:
				dependents.append(node)
			else:
				roots.append(node)

		# 2. Traverse and assign columns based on maximum distance from a root
		node_columns = {}

		def traverse_forward(node, column_index):
			if node.name not in node_columns or column_index > node_columns[node.name]:
				node_columns[node.name] = column_index
				for output in node.outputs:
					for link in output.links:
						traverse_forward(link.to_node, column_index + 1)

		for root in roots:
			traverse_forward(root, 0)

		# Catch-all for disconnected/cyclical nodes
		for dep in dependents:
			if dep.name not in node_columns:
				node_columns[dep.name] = 1

		# 3. Group nodes by column and sort them by current Y to preserve relative order
		columns = {}
		for node_name, col_idx in node_columns.items():
			if col_idx not in columns:
				columns[col_idx] = []
			columns[col_idx].append(nodetree.nodes[node_name])
			
		for col_idx in columns:
			columns[col_idx].sort(key=lambda n: n.location.y, reverse=True)

		# 4. Position nodes with calculated offsets
		current_x = 0
		for col_idx in sorted(columns.keys()):
			nodes_in_col = columns[col_idx]
			current_y = 0
			max_height = 0
			
			for node in nodes_in_col:
				node.location.x = current_x
				node.location.y = current_y
				
				# Update step for the next node in the same column
				node_height = node.dimensions.y
				current_y -= (node_height + y_spacing)
				max_height = max(max_height, node_height)
				
			# Offset for the next column
			max_width = max([n.dimensions.x for n in nodes_in_col]) if nodes_in_col else 0
			current_x += (max_width + x_spacing)

	def samplePoints(self):
		#TO DO

		return

		usableToggle = False
		
		for area in bpy.data.screens["Layout"].areas:
			if area.type == 'VIEW_3D':
				for space in area.spaces:
					if space.type == 'VIEW_3D':
						space.overlay.show_wireframes = True
						space.overlay.show_floor = usableToggle
						space.overlay.show_axis_x = usableToggle
						space.overlay.show_axis_y = usableToggle
						space.overlay.show_axis_z = usableToggle

		######################
		### MAKE THE OG PLANE
		######################
		bpy.ops.mesh.primitive_plane_add(size=10, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))

		myInputMesh = bpy.context.active_object
		bpy.context.view_layer.objects.active = myInputMesh

		bpy.ops.object.modifier_add(type='SUBSURF')

		myInputMesh.modifiers["Subdivision"].levels = 1
		myInputMesh.modifiers["Subdivision"].subdivision_type = 'SIMPLE'
		myInputMesh.modifiers["Subdivision"].show_only_control_edges = False
		# myInputMesh.modifiers["Subdivision"].levels = 2
		myInputMesh.modifiers["Subdivision"].levels = 3

		bpy.ops.object.modifier_apply(modifier="Subdivision")

		######################
		### COPY THE OG PLANE ONCE SO IT IS USABLE
		######################
		bpy.context.view_layer.objects.active = myInputMesh
		myUsableOGMesh = self.copyObject()
		myUsableOGMesh.name = 'myUsableOGMesh'

		######################
		### MAKE THE SAMPLE PLANE (WHICH NEEDS TO BE DYNAMICALLY SUBDIVIDED)
		######################
		bpy.context.view_layer.objects.active = myInputMesh
		mySampleMesh = self.copyObject()
		mySampleMesh.name = 'mySampleMesh'

		# bpy.context.view_layer.objects.active = mySampleMesh
		# mySampleMesh.select_set(1)

		# bpy.ops.object.modifier_add(type='SUBSURF')

		# mySampleMesh.modifiers["Subdivision"].levels = 1
		# mySampleMesh.modifiers["Subdivision"].subdivision_type = 'SIMPLE'
		# mySampleMesh.modifiers["Subdivision"].show_only_control_edges = False
		# mySampleMesh.modifiers["Subdivision"].levels = 2
		# # mySampleMesh.modifiers["Subdivision"].levels = 3


		######################
		### SPLIT THE USABLE OG MESH
		######################

		#SPLIT MESH INTO FACE OBJECTS
		self.deselectAll()

		# usableSplitMesh = self.splitObjectIntoFacesFunc0(myInputMesh)
		usableSplitMesh = self.splitObjectIntoFacesFunc0(myUsableOGMesh)
		self.shadingList_perFace = []

		growVertList = []
		selList_0 = []
		selList_1 = []

		######################
		### GROW THE SELECTION IN A LOOP
		######################
		#take the first center vert location and add to growVertList
		#take the first face and take its verts and add to growVertList
		#loop through the split faces on 
		#if any of those faces include that vert location, select them and add to selList_0

		#these selected are the base area...join them?

		#take the location of the other (here 3) verts on each selected face
		#gather the locations of all the verts on each selected face  and add to growVertList
		#loop through the split faces on 
		#if any of those faces include that vert location, select them and add to selList_1

		#establish the neighboring sample points by random sampling if the angle between OG faces is less than X

		#if any of the verts in the selected faces equal the verts on a established sample point

		self.deselectAll()

		# bpy.ops.object.modifier_apply(modifier="Subdivision")

	def align_vectors_bpy(self, v1, v2):
		"""
		Calculates a rotation matrix to align vector v1 with vector v2.
		Assumes v1 and v2 are 3D numpy arrays.
		"""
		v1 = self.abjNormalize_written(v1) # Normalize v1
		v2 = self.abjNormalize_written(v2) # Normalize v2

		# Calculate rotation axis
		axis = self.written_manual_cross(v1, v2)
		axis_norm = self.abjNormalize_written(axis)

		if axis_norm == 0: # Vectors are already aligned or opposite
			if self.written_manual_dotProduct(v1, v2) > 0: # Already aligned
				return mathutils.Matrix.identity(4)
		else: # Opposite direction, rotate 180 degrees around an arbitrary axis
			myR = mathutils.Matrix.Identity(4)
			myR[0][0] = -1
			myR[0][1] = 0
			myR[0][2] = 0
			myR[0][3] = 0

			myR[1][0] = 0
			myR[1][1] = -1
			myR[1][2] = 0
			myR[1][3] = 0

			myR[2][0] = 0
			myR[2][1] = 0
			myR[2][2] = 1
			myR[2][3] = 0

			myR[3][0] = 0
			myR[3][1] = 0
			myR[3][2] = 0
			myR[3][3] = 1

			return myR

			# return np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]) # Example: rotate around Z

		axis = axis / axis_norm

		# Calculate rotation angle
		angle = math.acos(self.written_manual_dotProduct(v1, v2))

		# Construct rotation matrix using Rodrigues' rotation formula
		K = np.array([[0, -axis[2], axis[1]],
		[axis[2], 0, -axis[0]],
		[-axis[1], axis[0], 0]])

		R = np.identity(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)



		myR = mathutils.Matrix.Identity(4)
		myR[0][0] = -1
		myR[0][1] = 0
		myR[0][2] = 0
		myR[0][3] = 0

		myR[1][0] = 0
		myR[1][1] = -1
		myR[1][2] = 0
		myR[1][3] = 0

		myR[2][0] = 0
		myR[2][1] = 0
		myR[2][2] = 1
		myR[2][3] = 0

		myR[3][0] = 0
		myR[3][1] = 0
		myR[3][2] = 0
		myR[3][3] = 1

		return myR

		
		return R	
	
	def align_vectors_numpy(v1, v2):
		"""
		Calculates a rotation matrix to align vector v1 with vector v2.
		Assumes v1 and v2 are 3D numpy arrays.
		"""
		v1 = v1 / np.linalg.norm(v1) # Normalize v1
		v2 = v2 / np.linalg.norm(v2) # Normalize v2

		# Calculate rotation axis
		axis = np.cross(v1, v2)
		axis_norm = np.linalg.norm(axis)

		if axis_norm == 0: # Vectors are already aligned or opposite
			if np.dot(v1, v2) > 0: # Already aligned
				return np.identity(3)
		else: # Opposite direction, rotate 180 degrees around an arbitrary axis
			return np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]) # Example: rotate around Z

		axis = axis / axis_norm

		# Calculate rotation angle
		angle = np.arccos(np.dot(v1, v2))

		# Construct rotation matrix using Rodrigues' rotation formula
		K = np.array([[0, -axis[2], axis[1]],
		[axis[2], 0, -axis[0]],
		[-axis[1], axis[0], 0]])
		R = np.identity(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
		
		return R
	
	#https://blender.stackexchange.com/questions/316333/context-is-incorrect-error-from-project-from-view
	def uvFromCam(self):
		camera_name = "Camera"

		if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
			bpy.ops.object.mode_set(mode='OBJECT')

		cam = bpy.data.objects[camera_name]
		bpy.context.scene.camera = cam
		# myInputMesh.select_set(1)

		bpy.ops.object.mode_set(mode='EDIT')
		bpy.ops.mesh.select_mode(type='FACE')

		bpy.ops.mesh.select_all(action="SELECT")
		for oWindow in bpy.context.window_manager.windows:
			oScreen = oWindow.screen
			for oArea in oScreen.areas:
				if oArea.type == 'VIEW_3D':  
					for oRegion in oArea.regions:
						if oRegion.type == 'WINDOW':
							context_override = bpy.context.copy()  # EDITED
							override = {
								'window': oWindow, 
								'screen': oScreen, 
								'area': oArea, 
								'region': oRegion, 
								'scene': bpy.context.scene, 
								'edit_object': bpy.context.edit_object, 
								'active_object': bpy.context.active_object, 
								'selected_objects': bpy.context.selected_objects
							}
							for k, v in override.items():  # EDITED
								context_override[k] = v
							with bpy.context.temp_override(**context_override):  # EDITED
								bpy.ops.uv.project_from_view(camera_bounds=True, correct_aspect=True, scale_to_bounds=False)

								# bpy.ops.uv.project_from_view(camera_bounds=True, correct_aspect=True, scale_to_bounds=True)

		bpy.ops.object.mode_set(mode='OBJECT')


	def batch_angle_based_uv_unwrap(self):
		selected_objects = bpy.context.selected_objects

		if not selected_objects:
			print("No objects selected for unwrapping.")
			return

		bpy.ops.object.select_all(action='DESELECT')

		for obj in selected_objects:
			if obj.type == 'MESH':
				bpy.context.view_layer.objects.active = obj
				obj.select_set(True)

				bpy.ops.object.mode_set(mode='EDIT')
				
				bpy.ops.mesh.select_all(action='SELECT')
				
				bpy.ops.uv.unwrap(method='ANGLE_BASED', fill_holes=True, correct_aspect=True, use_subsurf_data=False, margin=0.001, no_flip=False, iterations=10, use_weights=False, weight_group="uv_importance", weight_factor=1)
				
				bpy.ops.object.mode_set(mode='OBJECT')
				
				obj.select_set(False)

	def DoIt_part1_preprocess(self):
		self.startTime_stage1 = datetime.now() ############

		self.profile_stage1_02_final = self.startTime_stage1 - self.startTime_stage1
		self.profile_stage1_03_final = self.startTime_stage1 - self.startTime_stage1
		self.profile_stage1_04_final = self.startTime_stage1 - self.startTime_stage1
		self.profile_stage1_05_final = self.startTime_stage1 - self.startTime_stage1
		self.profile_stage1_06_final = self.startTime_stage1 - self.startTime_stage1
		self.profile_stage1_07_final = self.startTime_stage1 - self.startTime_stage1
		# self.profile_stage1_08_final = self.startTime_stage1 - self.startTime_stage1
		# self.profile_stage1_09_final = self.startTime_stage1 - self.startTime_stage1
		# self.profile_stage1_10_final = self.startTime_stage1 - self.startTime_stage1
		# print('should be zero... ', self.profile_stage1_02_final)

		self.profile_stage1_06_a = datetime.now() ################

		bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)
		bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')

		self.deselectAll()
		self.deleteAllObjects()
		self.mega_purge()

		self.shadingList_perFace.clear()
		self.shadingStages_perFace_stepList.clear()
		self.shadingStages_selectedFaces.clear()
		self.arrow_dynamic_instance_M_all_list_matrixOnly.clear()
		self.objectsToToggleOnOffLater.clear()
		self.textRef_all.clear()
		self.myDebugFaces.clear()
		self.distanceFromCam_all_list.clear()
		self.distanceFromCam_raycastRenderable_list.clear()

		self.renderPasses_simple = False
		self.renderPasses_GGX = False

		self.runOnce_part2_preProcess = False

		aov_items = bpy.context.scene.bl_rna.properties['aov_enum_prop'].enum_items
		aov_id = aov_items[bpy.context.scene.aov_enum_prop].identifier
		self.aov_stored = aov_id

		rdotvpow_items = bpy.context.scene.bl_rna.properties['r_dot_v_pow_enum_prop'].enum_items
		rdotvpow_id = rdotvpow_items[bpy.context.scene.r_dot_v_pow_enum_prop].identifier
		self.rdotvpow_stored = rdotvpow_id

		val_spectral_mix_multiplier_prop = bpy.context.scene.spectral_mix_multiplier_prop
		self.spectralmix_stored = val_spectral_mix_multiplier_prop

		val_spectral_mix_multiplier2_prop = bpy.context.scene.spectral_mix_multiplier2_prop
		self.spectralmix2_stored = val_spectral_mix_multiplier2_prop

		val_oren_roughness_prop = bpy.context.scene.oren_roughness_prop
		self.oren_roughness_stored = val_oren_roughness_prop

		val_ggx_roughness_prop = bpy.context.scene.ggx_roughness_prop
		self.ggx_roughness_stored = val_ggx_roughness_prop

		val_ggx_fresnel_prop = bpy.context.scene.ggx_fresnel_prop
		self.ggx_fresnel_stored = val_ggx_fresnel_prop

		val_is_metallic_prop = bpy.context.scene.is_metallic_prop
		self.is_metallic_stored = val_is_metallic_prop

		val_aniso_specular_prop = bpy.context.scene.aniso_specular_prop
		self.aniso_specular_stored = val_aniso_specular_prop

		val_aniso_rotation_prop = bpy.context.scene.aniso_rotation_prop
		self.aniso_rotation_stored = val_aniso_rotation_prop

		val_aniso_roughnessX_prop = bpy.context.scene.aniso_roughnessX_prop
		self.aniso_roughnessX_stored = val_aniso_roughnessX_prop
		
		val_aniso_roughnessY_prop = bpy.context.scene.aniso_roughnessY_prop
		self.aniso_roughnessY_stored = val_aniso_roughnessY_prop


		val_text_radius_0_prop = bpy.context.scene.text_radius_0_prop
		self.text_radius_0_stored = val_text_radius_0_prop

		val_text_radius_1_prop = bpy.context.scene.text_radius_1_prop
		self.text_radius_1_stored = val_text_radius_1_prop

		val_use_18_hue_colorspace_prop = bpy.context.scene.use_18_hue_colorspace_prop
		self.use_18_hue_colorspace_prop_stored = val_use_18_hue_colorspace_prop


		val_text_rotate_x_prop = bpy.context.scene.text_rotate_x_prop
		val_text_rotate_y_prop = bpy.context.scene.text_rotate_y_prop
		val_text_rotate_z_prop = bpy.context.scene.text_rotate_z_prop

		self.text_rotate_x_stored = val_text_rotate_x_prop
		self.text_rotate_y_stored = val_text_rotate_y_prop
		self.text_rotate_z_stored = val_text_rotate_z_prop

		val_text_gradient_rotate_x_prop = bpy.context.scene.text_gradient_rotate_x_prop
		val_text_gradient_rotate_y_prop = bpy.context.scene.text_gradient_rotate_y_prop
		val_text_gradient_rotate_z_prop = bpy.context.scene.text_gradient_rotate_z_prop

		self.text_gradient_rotate_x_stored = val_text_gradient_rotate_x_prop
		self.text_gradient_rotate_y_stored = val_text_gradient_rotate_y_prop
		self.text_gradient_rotate_z_stored = val_text_gradient_rotate_z_prop

		if self.debugStageIterPlusMinus == True:
			self.shadingStages_selectedFaces.clear()
			self.shadingStages_selectedFaces.append('242')

			test_stagesDict_perFace0 = {
				# 'idx' : mySplitFaceIndexUsable,
				'idx' : '242',
				# 'shadingPlane' : self.shadingPlane.name,
				'shadingPlane' :'suzanne_242',
				# 'stage' : usableBreakpoint000_items_id,
				'stage' : 0,
				'breakpoint_idx' : 0,
			}
			self.shadingStages_perFace_stepList.append(test_stagesDict_perFace0)

			self.stageIdx_plusMinus_UI(1)

			# for j in self.shadingStages_perFace_stepList:
			# 	print('j from failing = ', j)

			return

		###########
		#DEFAULT CAMERA
		#############
		cam1_data = bpy.ops.object.camera_add(
			location=(self.pos_camera_global),  # x, y, z coordinates
			rotation=(0.0, 0.0, 0.0)   # x, y, z rotation in radians
		)

		###################################
		###### SET CAMERA POS / LOOK AT
		###################################
		self.myCam = bpy.data.objects["Camera"]

		self.myCam.data.clip_start = 1
		# self.myCam.data.clip_start = .1
		# self.myCam.data.clip_start = .5
		# self.myCam.data.clip_end = 100
		self.myCam.data.clip_end = 500

		self.myCam.location = self.pos_camera_global
		self.updateScene() # need

		self.look_at(self.myCam, self.myOrigin)

		# self.myV = self.myCam.matrix_world.to_translation()
		# self.myV.normalize()

		f = self.abjNormalize_written(self.myOrigin - self.myCam.location)
		self.myV = -f

		# self.myCam.rotation_euler = mathutils.Vector((math.radians(self.myV.xyz), 0, math.radians(90)))
		# self.myCam.rotation_euler = mathutils.Vector((math.radians(-self.myV.x), math.radians(-self.myV.y), math.radians(-self.myV.z)))
		# self.myCam.rotation_euler = mathutils.Vector((math.radians(self.myV.x), math.radians(self.myV.y), math.radians(self.myV.z)))

		###################################
		###### INPUT MESH / XFORM ###########
		###################################
		usablePrimitiveType_items = bpy.context.scene.bl_rna.properties['primitive_enum_prop'].enum_items
		usablePrimitiveType_id = usablePrimitiveType_items[bpy.context.scene.primitive_enum_prop].identifier

		multipleObj = True
		# multipleObj = False
		myInputMesh = None
		myInputMeshDebug0 = None
		myInputMeshDebug1 = None
		myInputMeshDebug2 = None

		self.deselectAll()

		if usablePrimitiveType_id == 'cube':
			bpy.ops.mesh.primitive_cube_add()

			myInputMesh = bpy.context.active_object
			myInputMesh.select_set(1)

			if self.useRestoredRxyzValues == True:
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='X', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='Y', orient_type='GLOBAL')

			else:
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='X', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='Y', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(self.RandomRotationDegree), orient_axis=self.RandomRotationAxis, orient_type='GLOBAL')

			bpy.ops.object.transform_apply(location=1, rotation=1, scale=1)

			if multipleObj == True:
				bpy.ops.mesh.primitive_cube_add()
				myInputMeshDebug0 = bpy.context.active_object
				myInputMeshDebug0.location = mathutils.Vector((-5, -4.5, -2.2))

				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='X', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='Y', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(self.RandomRotationDegree), orient_axis=self.RandomRotationAxis, orient_type='GLOBAL')

				bpy.ops.mesh.primitive_cube_add()
				myInputMeshDebug1 = bpy.context.active_object
				myInputMeshDebug1.location = mathutils.Vector((-8.1, -6.6, -2.9))

				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='X', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='Y', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(self.RandomRotationDegree), orient_axis=self.RandomRotationAxis, orient_type='GLOBAL')

				bpy.ops.mesh.primitive_cube_add()
				myInputMeshDebug2 = bpy.context.active_object
				myInputMeshDebug2.location = mathutils.Vector((-11.1, -8.1, -3.6))

				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='X', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='Y', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(self.RandomRotationDegree), orient_axis=self.RandomRotationAxis, orient_type='GLOBAL')

				bpy.ops.object.transform_apply(location=1, rotation=1, scale=1)

				self.deselectAll()

		elif usablePrimitiveType_id == 'uv_sphere':
			bpy.ops.mesh.primitive_uv_sphere_add()

			myInputMesh = bpy.context.active_object
			myInputMesh.select_set(1)

			if self.useRestoredRxyzValues == True:
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='X', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='Y', orient_type='GLOBAL')

			else:
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='X', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='Y', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(self.RandomRotationDegree), orient_axis=self.RandomRotationAxis, orient_type='GLOBAL')

			bpy.ops.object.transform_apply(location=1, rotation=1, scale=1)

			if multipleObj == True:
				bpy.ops.mesh.primitive_uv_sphere_add()
				myInputMeshDebug0 = bpy.context.active_object
				myInputMeshDebug0.location = mathutils.Vector((-5, -4.5, -2.2))

				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='X', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='Y', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(self.RandomRotationDegree), orient_axis=self.RandomRotationAxis, orient_type='GLOBAL')

				bpy.ops.mesh.primitive_uv_sphere_add()
				myInputMeshDebug1 = bpy.context.active_object
				myInputMeshDebug1.location = mathutils.Vector((-8.1, -6.6, -2.9))

				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='X', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='Y', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(self.RandomRotationDegree), orient_axis=self.RandomRotationAxis, orient_type='GLOBAL')

				bpy.ops.mesh.primitive_uv_sphere_add()
				myInputMeshDebug2 = bpy.context.active_object
				myInputMeshDebug2.location = mathutils.Vector((-11.1, -8.1, -3.6))

				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='X', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='Y', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(self.RandomRotationDegree), orient_axis=self.RandomRotationAxis, orient_type='GLOBAL')

				bpy.ops.object.transform_apply(location=1, rotation=1, scale=1)

				self.deselectAll()

		elif usablePrimitiveType_id == 'ico_sphere':
			bpy.ops.mesh.primitive_ico_sphere_add()

			myInputMesh = bpy.context.active_object
			myInputMesh.select_set(1)

			if self.useRestoredRxyzValues == True:
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='X', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='Y', orient_type='GLOBAL')

			else:
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='X', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='Y', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(self.RandomRotationDegree), orient_axis=self.RandomRotationAxis, orient_type='GLOBAL')

			bpy.ops.object.transform_apply(location=1, rotation=1, scale=1)

			if multipleObj == True:
				bpy.ops.mesh.primitive_ico_sphere_add()
				myInputMeshDebug0 = bpy.context.active_object
				myInputMeshDebug0.location = mathutils.Vector((-5, -4.5, -2.2))

				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='X', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='Y', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(self.RandomRotationDegree), orient_axis=self.RandomRotationAxis, orient_type='GLOBAL')

				bpy.ops.mesh.primitive_ico_sphere_add()
				myInputMeshDebug1 = bpy.context.active_object
				myInputMeshDebug1.location = mathutils.Vector((-8.1, -6.6, -2.9))

				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='X', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='Y', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(self.RandomRotationDegree), orient_axis=self.RandomRotationAxis, orient_type='GLOBAL')

				bpy.ops.mesh.primitive_ico_sphere_add()
				myInputMeshDebug2 = bpy.context.active_object
				myInputMeshDebug2.location = mathutils.Vector((-11.1, -8.1, -3.6))

				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='X', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='Y', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(self.RandomRotationDegree), orient_axis=self.RandomRotationAxis, orient_type='GLOBAL')

				bpy.ops.object.transform_apply(location=1, rotation=1, scale=1)

				self.deselectAll()

		elif usablePrimitiveType_id == 'cylinder':
			bpy.ops.mesh.primitive_cylinder_add()

			myInputMesh = bpy.context.active_object
			myInputMesh.select_set(1)

			if self.useRestoredRxyzValues == True:
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='X', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='Y', orient_type='GLOBAL')

			else:
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='X', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='Y', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(self.RandomRotationDegree), orient_axis=self.RandomRotationAxis, orient_type='GLOBAL')

			bpy.ops.object.transform_apply(location=1, rotation=1, scale=1)

			if multipleObj == True:
				bpy.ops.mesh.primitive_cylinder_add()
				myInputMeshDebug0 = bpy.context.active_object
				myInputMeshDebug0.location = mathutils.Vector((-5, -4.5, -2.2))

				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='X', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='Y', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(self.RandomRotationDegree), orient_axis=self.RandomRotationAxis, orient_type='GLOBAL')

				bpy.ops.mesh.primitive_cylinder_add()
				myInputMeshDebug1 = bpy.context.active_object
				myInputMeshDebug1.location = mathutils.Vector((-8.1, -6.6, -2.9))

				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='X', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='Y', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(self.RandomRotationDegree), orient_axis=self.RandomRotationAxis, orient_type='GLOBAL')

				bpy.ops.mesh.primitive_cylinder_add()
				myInputMeshDebug2 = bpy.context.active_object
				myInputMeshDebug2.location = mathutils.Vector((-11.1, -8.1, -3.6))

				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='X', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='Y', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(self.RandomRotationDegree), orient_axis=self.RandomRotationAxis, orient_type='GLOBAL')

				bpy.ops.object.transform_apply(location=1, rotation=1, scale=1)

				self.deselectAll()

		elif usablePrimitiveType_id == 'cone':
			bpy.ops.mesh.primitive_cone_add()

			myInputMesh = bpy.context.active_object
			myInputMesh.select_set(1)

			if self.useRestoredRxyzValues == True:
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='X', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='Y', orient_type='GLOBAL')

			else:
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='X', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='Y', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(self.RandomRotationDegree), orient_axis=self.RandomRotationAxis, orient_type='GLOBAL')

			bpy.ops.object.transform_apply(location=1, rotation=1, scale=1)

			if multipleObj == True:
				bpy.ops.mesh.primitive_cone_add()
				myInputMeshDebug0 = bpy.context.active_object
				myInputMeshDebug0.location = mathutils.Vector((-5, -4.5, -2.2))

				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='X', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='Y', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(self.RandomRotationDegree), orient_axis=self.RandomRotationAxis, orient_type='GLOBAL')

				bpy.ops.mesh.primitive_cone_add()
				myInputMeshDebug1 = bpy.context.active_object
				myInputMeshDebug1.location = mathutils.Vector((-8.1, -6.6, -2.9))

				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='X', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='Y', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(self.RandomRotationDegree), orient_axis=self.RandomRotationAxis, orient_type='GLOBAL')

				bpy.ops.mesh.primitive_cone_add()
				myInputMeshDebug2 = bpy.context.active_object
				myInputMeshDebug2.location = mathutils.Vector((-11.1, -8.1, -3.6))

				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='X', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='Y', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(self.RandomRotationDegree), orient_axis=self.RandomRotationAxis, orient_type='GLOBAL')

				bpy.ops.object.transform_apply(location=1, rotation=1, scale=1)

				self.deselectAll()

		elif usablePrimitiveType_id == 'torus':
			bpy.ops.mesh.primitive_torus_add()

			myInputMesh = bpy.context.active_object
			myInputMesh.select_set(1)

			if self.useRestoredRxyzValues == True:
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='X', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='Y', orient_type='GLOBAL')

			else:
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='X', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='Y', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(self.RandomRotationDegree), orient_axis=self.RandomRotationAxis, orient_type='GLOBAL')

			bpy.ops.object.transform_apply(location=1, rotation=1, scale=1)

			if multipleObj == True:
				bpy.ops.mesh.primitive_torus_add()
				myInputMeshDebug0 = bpy.context.active_object
				myInputMeshDebug0.location = mathutils.Vector((-5, -4.5, -2.2))

				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='X', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='Y', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(self.RandomRotationDegree), orient_axis=self.RandomRotationAxis, orient_type='GLOBAL')

				bpy.ops.mesh.primitive_torus_add()
				myInputMeshDebug1 = bpy.context.active_object
				myInputMeshDebug1.location = mathutils.Vector((-8.1, -6.6, -2.9))

				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='X', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='Y', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(self.RandomRotationDegree), orient_axis=self.RandomRotationAxis, orient_type='GLOBAL')

				bpy.ops.mesh.primitive_torus_add()
				myInputMeshDebug2 = bpy.context.active_object
				myInputMeshDebug2.location = mathutils.Vector((-11.1, -8.1, -3.6))

				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='X', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='Y', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(self.RandomRotationDegree), orient_axis=self.RandomRotationAxis, orient_type='GLOBAL')

				bpy.ops.object.transform_apply(location=1, rotation=1, scale=1)

				self.deselectAll()

		elif usablePrimitiveType_id == 'monkey':
			bpy.ops.mesh.primitive_monkey_add()

			myInputMesh = bpy.context.active_object
			myInputMesh.select_set(1)

			if self.useRestoredRxyzValues == True:
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='X', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='Y', orient_type='GLOBAL')

			else:
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='X', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='Y', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(self.RandomRotationDegree), orient_axis=self.RandomRotationAxis, orient_type='GLOBAL')

			bpy.ops.object.transform_apply(location=1, rotation=1, scale=1)

			if multipleObj == True:
				bpy.ops.mesh.primitive_monkey_add()
				myInputMeshDebug0 = bpy.context.active_object
				myInputMeshDebug0.location = mathutils.Vector((-5, -4.5, -2.2))

				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='X', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='Y', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(self.RandomRotationDegree), orient_axis=self.RandomRotationAxis, orient_type='GLOBAL')

				bpy.ops.mesh.primitive_monkey_add()
				myInputMeshDebug1 = bpy.context.active_object
				myInputMeshDebug1.location = mathutils.Vector((-8.1, -6.6, -2.9))

				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='X', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='Y', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(self.RandomRotationDegree), orient_axis=self.RandomRotationAxis, orient_type='GLOBAL')

				bpy.ops.mesh.primitive_monkey_add()
				myInputMeshDebug2 = bpy.context.active_object
				myInputMeshDebug2.location = mathutils.Vector((-11.1, -8.1, -3.6))

				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='X', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(180), orient_axis='Y', orient_type='GLOBAL')
				bpy.ops.transform.rotate(value=math.radians(self.RandomRotationDegree), orient_axis=self.RandomRotationAxis, orient_type='GLOBAL')

				bpy.ops.object.transform_apply(location=1, rotation=1, scale=1)

				self.deselectAll()

		# myInputMesh = bpy.context.active_object
		# myInputMesh.select_set(1)

		#TRIANGULATE
		# bpy.ops.object.modifier_add(type='TRIANGULATE')
		# bpy.ops.object.modifier_apply(modifier="Triangulate")

		####SUBDIVIDE
		usableSubDToggle_items = bpy.context.scene.bl_rna.properties['subdivision_toggle_enum_prop'].enum_items
		usablePrimitiveType_id = usableSubDToggle_items[bpy.context.scene.subdivision_toggle_enum_prop].identifier

		myInputMesh.select_set(1)

		if multipleObj == True:
			myInputMeshDebug0.select_set(1)
			myInputMeshDebug1.select_set(1)
			myInputMeshDebug2.select_set(1)
			bpy.ops.object.join()

		myInputMesh = bpy.context.active_object

		if usablePrimitiveType_id == 'subd_1':
			bpy.ops.object.modifier_add(type='SUBSURF')
			myObj = bpy.context.active_object
			myObj.modifiers["Subdivision"].levels = 1
			
			bpy.ops.object.modifier_apply(modifier="Subdivision")

		if multipleObj == False:
			if usablePrimitiveType_id == 'subd_2':
				bpy.ops.object.modifier_add(type='SUBSURF')
				myObj = bpy.context.active_object
				# myObj.modifiers["Subdivision"].levels = 1
				myObj.modifiers["Subdivision"].levels = 2
				
				bpy.ops.object.modifier_apply(modifier="Subdivision")

		#TRIANGULATE
		bpy.ops.object.modifier_add(type='TRIANGULATE')
		bpy.ops.object.modifier_apply(modifier="Triangulate")

		self.profile_stage1_06_b = str(datetime.now() - self.profile_stage1_06_a)
		if self.profileCode_part1 == True:
			print('~~~~~~~~~ self.profile_stage1_06_b = ', self.profile_stage1_06_b)

		self.uvFromCam()

		########
		#Dupe for raycasting
		########

		bpy.context.view_layer.objects.active = myInputMesh
		myDupeForMaterialCheck = self.copyObject()
		myDupeForMaterialCheck.name = 'dupeForMaterialCheck'
		myDupeForMaterialCheck.hide_set(1)
		myDupeForMaterialCheck.hide_render = True

		self.profile_stage1_00_a = datetime.now() ################

		#SPLIT MESH INTO FACE OBJECTS
		self.deselectAll()

		usableSplitMesh = self.splitObjectIntoFacesFunc0(myInputMesh)
		self.shadingList_perFace = []

		self.profile_stage1_00_b = str(datetime.now() - self.profile_stage1_00_a)
		if self.profileCode_part1 == True:
			print('~~~~~~~~~ self.profile_stage1_00_b = ', self.profile_stage1_00_b)

		#############################################################################################
		######################## CREATE ORIGINALS #######################################
		#############################################################################################
		self.profile_stage1_01_a = datetime.now() ################

		###############################
		######### CUBE CAM ############
		###############################
		self.myCubeCam = self.createArrowFullProcess('myCubeCam', 'back', True, self.myOrigin, 0.0, 1.0, 0.9, 0.0, 1.0, 0.9)

		#####################
		## SUN LIGHT FOR TEST
		#####################
		bpy.ops.object.light_add(type='SUN', radius=1, align='WORLD', location=(self.pos_light_global_v), scale=(1, 1, 1))
		self.mySun = bpy.context.active_object
		self.mySun.name = "mySun"
		# self.mySun.hide_set(1)

		bpy.ops.object.transform_apply(location=1, rotation=1, scale=1) ##### thursday debug

		#############################################################################################
		################################# (INSTANCE ORIGINALS) ###############################
		#############################################################################################

		###############################
		######### CUBE LIGHT (POINT) ############
		###############################
		self.myCubeLight_og = self.createArrowFullProcess('myCubeLight_og', 'front', False, self.myOrigin, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0)

		###############################
		######### CUBE H ############
		###############################
		self.myCubeH_og = self.createArrowFullProcess('myCubeH_og', 'front', False, self.myOrigin, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

		###############################
		######### CUBE N ############
		###############################
		self.myCubeN_og = self.createArrowFullProcess('myCubeN_og', 'back', False, self.myOrigin, 1.0, 0.0, 0.0, 0.0, 1.0, 0.25)

		###############################
		######### CUBE R ############
		###############################
		bpy.context.view_layer.objects.active = self.myCubeN_og
		self.myCubeR_og = self.copyObject()
		self.myCubeR_og.name = 'myCubeR_og'

		self.myCubeR_og_Matrix = self.myCubeR_og.matrix_world

		####
		bpy.context.view_layer.objects.active = self.myCubeR_og

		mat2 = None
		if self.adjustedColors == True:
			mat2 = self.newShader("cubeR_og", self.diffuse_or_emission_og_shading, 0.2, 0.87, 1.0)
		elif self.adjustedColors == False:	
			mat2 = self.newShader("cubeR_og", self.diffuse_or_emission_og_shading, 0.5, 0.0, 1.0) ###
		bpy.context.active_object.data.materials.clear()
		bpy.context.active_object.data.materials.append(mat2)

		# print('CREATED CUBE 2')

		#############
		### CLEAN UP
		############
		self.deselectAll()

		bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)

		bpy.context.view_layer.objects.active = self.myCubeLight_og
		self.myCubeLight_dupe = self.copyObject()
		self.myCubeLight_dupe.name = 'myCubeLight_dupe'

		bpy.context.view_layer.objects.active = self.myCubeH_og
		self.myCubeH_dupe = self.copyObject()
		self.myCubeH_dupe.name = 'myCubeH_dupe'

		bpy.context.view_layer.objects.active = self.myCubeN_og
		self.myCubeN_dupe = self.copyObject()
		self.myCubeN_dupe.name = 'myCubeN_dupe'

		bpy.context.view_layer.objects.active = self.myCubeR_og
		self.myCubeR_dupe = self.copyObject()
		self.myCubeR_dupe.name = 'myCubeR_dupe'

		bpy.context.view_layer.objects.active = self.myCubeCam
		self.myCubeCam_dupe = self.copyObject()
		self.myCubeCam_dupe.name = 'myCubeCam_dupe'

		self.myCubeCam_dupe.hide_set(1)

		#########
		# equation

		self.shadingDict_global = {
			'V' : self.myV,
		}

		self.profile_stage1_01_b = str(datetime.now() - self.profile_stage1_01_a)
		if self.profileCode_part1 == True:
			print('~~~~~~~~~ self.profile_stage1_01_b = ', self.profile_stage1_01_b)

		#############################################################################################
		######################################## FACE DEPENDANT #####################################
		#############################################################################################
		self.deselectAll()

		for i in usableSplitMesh:
			self.profile_stage1_04_a = datetime.now() ################

			self.shadingPlane = None

			for j in bpy.context.scene.objects:
				if j.name == i:
					self.shadingPlane = j
					self.shadingPlane.select_set(1)
					bpy.context.view_layer.objects.active = j
					self.shadingPlane = bpy.context.active_object
			
			if self.shadingPlane == None:
				print('ERROR: Could not find shading plane for : ',  i)
				return
				
			# bpy.ops.object.mode_set(mode="OBJECT")

			self.profile_stage1_04_b = datetime.now() - self.profile_stage1_04_a
			self.profile_stage1_04_final += self.profile_stage1_04_b

			self.profile_stage1_07_a = datetime.now() ################

			# self.deselectAll()

			#############
			### DEBUG ONLY SELECTED IDX
			#############
			mySplitFaceIndexUsable = i.split('_', -1)[1]

			# mySplitFaceIndexUsable = mySplitFaceIndexUsable_debug # ******* DONT ACTIVATE FOR FULL PREPROCESS
			# print('mySplitFaceIndexUsable = ', mySplitFaceIndexUsable)

			#############
			### RESET USABLE DUPES
			#############
			self.myCubeLight_dupe.matrix_world = self.myCubeLight_og_Matrix
			self.myCubeH_dupe.matrix_world = self.myCubeH_og_Matrix

			self.myCubeN_dupe.matrix_world = self.myCubeN_og_Matrix
			self.myCubeR_dupe.matrix_world = self.myCubeR_og_Matrix

			self.profile_stage1_07_b = datetime.now() - self.profile_stage1_07_a
			self.profile_stage1_07_final += self.profile_stage1_07_b

			###############################
			######### INFO (PER FACE) ############
			###############################
			self.profile_stage1_02_a = datetime.now() ################

			##################################################################################
			###################################### STORE SHADE PARAMS #####################################
			##################################################################################
			# myEquation_simple_spec_class.equation_part1_preProcess_00(myABJ_SD_B, mySplitFaceIndexUsable)
			self.equation_part1_preProcess_00(mySplitFaceIndexUsable)

		self.distanceFromCam_all_list.sort()

		if self.profileCode_part1 == True:
			print('~~~~~~~~~ self.profile_stage1_02_final = ', self.profile_stage1_02_final)
			print('~~~~~~~~~ self.profile_stage1_03_final = ', self.profile_stage1_03_final)
			print('~~~~~~~~~ self.profile_stage1_04_final = ', self.profile_stage1_04_final)
			print('~~~~~~~~~ self.profile_stage1_07_final = ', self.profile_stage1_07_final)

		#gotcha
		# for i in self.shadingStages_perFace_stepList:	
		# 	for key, value in i.items():
		# 		print(f"{key}: {value}")

		# self.profile_stage1_05_a = datetime.now() ################

		self.updateScene()

		bpy.ops.object.mode_set(mode="OBJECT")

		self.myCubeLight_og.hide_set(1)
		self.myCubeH_og.hide_set(1)
		self.myCubeN_og.hide_set(1)
		self.myCubeR_og.hide_set(1)

		self.myCubeLight_dupe.hide_set(1)
		self.myCubeH_dupe.hide_set(1)
		self.myCubeN_dupe.hide_set(1)
		self.myCubeR_dupe.hide_set(1)

		# self.profile_stage1_05_b = datetime.now() - self.profile_stage1_05_a
		# if self.profileCode_part1 == True:
		# 	print('~~~~~~~~~ self.profile_stage1_05_b = ', self.profile_stage1_05_b)

		# self.profile_stage1_04_final += self.profile_stage1_05_b

		print('TIME TO COMPLETE stage 1 (preprocess) = ' + str(datetime.now() - self.startTime_stage1))
		print(' ')

		self.updateScene()

		self.myCubeCam.hide_set(1)
		bpy.ops.object.mode_set(mode="OBJECT")

		self.deselectAll()

	def equation_part1_preProcess_00(self, mySplitFaceIndexUsable):
		faceCenter = self.shadingPlane.location
		pos = self.shadingPlane.location

		myL = mathutils.Vector((self.pos_light_global_v - pos)).normalized()

		##################################################################################
		###################################### STORE SHADE PARAMS #####################################
		##################################################################################
		bpy.context.view_layer.objects.active = self.shadingPlane

		normalDir = self.getFaceNormal() ################

		myN = normalDir.normalized()
	
		########################
		########## DIFFUSE ########
		########################
		N_dot_L = max(np.dot(myN, myL), 0.0)

		########################
		########## SPEC ########
		########################
		# myR = -myL.reflect(myN)

		#remember, L and N must be normalized
		subtractVector = mathutils.Vector((2, 2, 2))
		reflectManual = -(myL - subtractVector) * (myL.dot(myN)) * myN
		myR = reflectManual

		myH = (self.myV + myL).normalized()
		R_dot_V_control = max(self.myV.dot(myR), 0.0)
		N_dot_V = max(myN.dot(self.myV), 0.0)

		distance = (self.pos_light_global_v - pos).length
		attenuation = 1.0 / (distance * distance)

		# distance_from_cam = (self.myV - faceCenter).length
		distance_from_cam = (self.myCam.location - faceCenter).length

		#Tangents
		mesh = self.shadingPlane.data
		mesh.calc_tangents()
		# mesh.calc_tangents(uvmap='UVMap')
		myTangent = None
		for poly in mesh.polygons:
			for loop_index in poly.loop_indices:
				loop = mesh.loops[loop_index]
				# myTangent = loop.tangent
				myTangent = loop.tangent.normalized()
				bitangent_sign = loop.bitangent_sign

		shadingDict_perFace = {
			'mySplitFaceIndexUsable' : mySplitFaceIndexUsable,
			'shadingPlane' : self.shadingPlane.name,
			'faceCenter' : faceCenter,
			'N_dot_L' : N_dot_L,
			'N_dot_V' : N_dot_V,
			'R_dot_V' : R_dot_V_control,
			'attenuation' : attenuation,
			'L' : myL,
			'N' : myN,
			'R' : myR,
			'H' : myH,
			'T' : myTangent,
			'B_sign' : bitangent_sign,
			'spec' : 0,
			'faceCenter_to_V_rayCast' : False,
			'faceCenter_to_L_rayCast' : False,
		}

		test_stagesDict_perFace0 = {
			'idx' : mySplitFaceIndexUsable,
			'shadingPlane' : self.shadingPlane.name,
			# 'stage' : usableBreakpoint000_items_id,
			'stage' : 0,
			'breakpoint_idx' : 0,
		}

		self.myDebugFaces.append(mySplitFaceIndexUsable)

		self.shadingList_perFace.append(shadingDict_perFace)

		self.shadingStages_perFace_stepList.append(test_stagesDict_perFace0)

		self.distanceFromCam_all_list.append(distance_from_cam)

	def doIt_part2_render(self):
		startTime = datetime.now()

		# self.samplePoints()

		usableTextRGBPrecision_items = bpy.context.scene.bl_rna.properties['text_rgb_precision_enum_prop'].enum_items
		usableTextRGBPrecision_id = usableTextRGBPrecision_items[bpy.context.scene.text_rgb_precision_enum_prop].identifier

		if usableTextRGBPrecision_id != '-1':
			if self.compositor_setup == False:
				self.setupCompositor()
			self.DoIt_part1_preprocess()

		self.profile_stage2_00_final = startTime - startTime
		self.profile_stage2_01_final = startTime - startTime
		self.profile_stage2_02_final = startTime - startTime
		self.profile_stage2_03_final = startTime - startTime
		self.profile_stage2_04_final = startTime - startTime
		self.profile_stage2_05_final = startTime - startTime
		self.profile_stage2_06_final = startTime - startTime
		self.profile_stage2_07_final = startTime - startTime
		self.profile_stage2_08_final = startTime - startTime
		self.profile_stage2_09_final = startTime - startTime
		self.profile_stage2_10_final = startTime - startTime

		self.profile_stage2_00_a = datetime.now() ################

		for i in self.objectsToToggleOnOffLater:
			i.hide_set(1)

		# self.objectsToToggleOnOffLater.clear()

		# V = self.shadingDict_global['V']

		aov_items = bpy.context.scene.bl_rna.properties['aov_enum_prop'].enum_items
		aov_id = aov_items[bpy.context.scene.aov_enum_prop].identifier

		if self.aov_stored != aov_id:
			self.skip_refresh_override_aov = True
			self.aov_stored = aov_id

		rdotvpow_items = bpy.context.scene.bl_rna.properties['r_dot_v_pow_enum_prop'].enum_items
		rdotvpow_id = rdotvpow_items[bpy.context.scene.r_dot_v_pow_enum_prop].identifier

		############
		#### SPECULAR EQUATION
		############
		val_ggx_roughness_prop = bpy.context.scene.ggx_roughness_prop
		val_ggx_fresnel_prop = bpy.context.scene.ggx_fresnel_prop

		val_is_metallic_prop = bpy.context.scene.is_metallic_prop
		val_aniso_specular_prop = bpy.context.scene.aniso_specular_prop
		val_aniso_rotation_prop = bpy.context.scene.aniso_rotation_prop
		val_aniso_roughnessX_prop = bpy.context.scene.aniso_roughnessX_prop
		val_aniso_roughnessY_prop = bpy.context.scene.aniso_roughnessY_prop

		val_spectral_mix_multiplier_prop = bpy.context.scene.spectral_mix_multiplier_prop
		val_spectral_mix_multiplier2_prop = bpy.context.scene.spectral_mix_multiplier2_prop

		self.changedSpecularEquation_variables = False
		self.changedDiffuseEquation_variables = False

		if self.spectralmix_stored != val_spectral_mix_multiplier_prop:
			self.skip_refresh_override_spectral = True
			self.spectralmix_stored = val_spectral_mix_multiplier_prop
			self.changedSpecularEquation_variables = True

		if self.spectralmix2_stored != val_spectral_mix_multiplier2_prop:
			self.skip_refresh_override_spectral2 = True
			self.spectralmix2_stored = val_spectral_mix_multiplier2_prop
			self.changedSpecularEquation_variables = True

		if self.rdotvpow_stored != rdotvpow_id:
			self.skip_refresh_override_RdotVpow = True
			self.rdotvpow_stored = rdotvpow_id
			self.changedSpecularEquation_variables = True

		if self.ggx_roughness_stored != val_ggx_roughness_prop:
			self.skip_refresh_override_GGX_roughness = True
			self.ggx_roughness_stored = val_ggx_roughness_prop
			self.changedSpecularEquation_variables = True

		if self.ggx_fresnel_stored != val_ggx_fresnel_prop:
			self.skip_refresh_override_GGX_fresnel = True
			self.ggx_fresnel_stored = val_ggx_fresnel_prop
			self.changedSpecularEquation_variables = True

		if self.is_metallic_stored != val_is_metallic_prop:
			self.skip_refresh_override_is_metallic = True
			self.is_metallic_stored = val_is_metallic_prop
			self.changedSpecularEquation_variables = True

		if self.aniso_specular_stored != val_aniso_specular_prop:
			self.skip_refresh_override_aniso_specular = True
			self.aniso_specular_stored = val_aniso_specular_prop
			self.changedSpecularEquation_variables = True

		if self.aniso_rotation_stored != val_aniso_rotation_prop:
			self.skip_refresh_override_aniso_rotation = True
			self.aniso_rotation_stored = val_aniso_rotation_prop
			self.changedSpecularEquation_variables = True

		if self.aniso_roughnessX_stored != val_aniso_roughnessX_prop:
			self.skip_refresh_override_aniso_roughnessX = True
			self.aniso_roughnessX_stored = val_aniso_roughnessX_prop
			self.changedSpecularEquation_variables = True

		if self.aniso_roughnessY_stored != val_aniso_roughnessY_prop:
			self.skip_refresh_override_aniso_roughnessY = True
			self.aniso_roughnessY_stored = val_aniso_roughnessY_prop
			self.changedSpecularEquation_variables = True



		usableSpecularEquationType_items = bpy.context.scene.bl_rna.properties['specular_equation_enum_prop'].enum_items
		usableSpecularEquationType_id = usableSpecularEquationType_items[bpy.context.scene.specular_equation_enum_prop].identifier

		if self.renderPasses_simple == True:
			# usableSpecularEquationType_id = 'simple'
			usableSpecularEquationType_id = 'GGX'
		elif self.renderPasses_GGX == True:
			usableSpecularEquationType_id = 'GGX'

		if usableSpecularEquationType_id != self.chosen_specular_equation:
			self.changedSpecularEquation_variables = True
			self.chosen_specular_equation = usableSpecularEquationType_id

		############
		#### DIFFUSE EQUATION
		############
		val_oren_roughness_prop = bpy.context.scene.oren_roughness_prop

		usableDiffuseEquationType_items = bpy.context.scene.bl_rna.properties['diffuse_equation_enum_prop'].enum_items
		usableDiffuseEquationType_id = usableDiffuseEquationType_items[bpy.context.scene.diffuse_equation_enum_prop].identifier

		if usableDiffuseEquationType_id != self.chosen_diffuse_equation:
			self.changedDiffuseEquation_variables = True
			self.chosen_diffuse_equation = usableDiffuseEquationType_id

		if self.oren_roughness_stored != val_oren_roughness_prop:
			self.skip_refresh_override_oren_roughness = True
			self.oren_roughness_stored = val_oren_roughness_prop
			self.changedSpecularEquation_variables = True

		
		############
		### 18 HUE COLORSPACE
		############
		val_use_18_hue_colorspace_prop = bpy.context.scene.use_18_hue_colorspace_prop

		if self.use_18_hue_colorspace_prop_stored != val_use_18_hue_colorspace_prop:
			self.skip_refresh_override_use_18_hue_colorspace = True
			self.use_18_hue_colorspace_prop_stored = val_use_18_hue_colorspace_prop

		############
		#### TEXT CHECKER
		############

		usableTextRGBPrecision_items = bpy.context.scene.bl_rna.properties['text_rgb_precision_enum_prop'].enum_items
		usableTextRGBPrecision_id = usableTextRGBPrecision_items[bpy.context.scene.text_rgb_precision_enum_prop].identifier

		if usableTextRGBPrecision_id != self.chosen_text_rgb_precision:
			self.changedSpecularEquation_variables = True
			self.chosen_text_rgb_precision = usableTextRGBPrecision_id

		val_text_radius_0_prop = bpy.context.scene.text_radius_0_prop
		val_text_radius_1_prop = bpy.context.scene.text_radius_1_prop

		if self.text_radius_0_stored != val_text_radius_0_prop:
			self.text_radius_0_stored = val_text_radius_0_prop
			self.changedSpecularEquation_variables = True

		if self.text_radius_1_stored != val_text_radius_1_prop:
			self.text_radius_1_stored = val_text_radius_1_prop
			self.changedSpecularEquation_variables = True

		val_text_rotate_x_prop = bpy.context.scene.text_rotate_x_prop
		val_text_rotate_y_prop = bpy.context.scene.text_rotate_y_prop
		val_text_rotate_z_prop = bpy.context.scene.text_rotate_z_prop
		
		if self.text_rotate_x_stored != val_text_rotate_x_prop:
			self.text_rotate_x_stored = val_text_rotate_x_prop
			self.changedSpecularEquation_variables = True

		if self.text_rotate_y_stored != val_text_rotate_y_prop:
			self.text_rotate_y_stored = val_text_rotate_y_prop
			self.changedSpecularEquation_variables = True

		if self.text_rotate_z_stored != val_text_rotate_z_prop:
			self.text_rotate_z_stored = val_text_rotate_z_prop
			self.changedSpecularEquation_variables = True

		#########################

		if self.runOnce_part2_preProcess == False or self.changedSpecularEquation_variables == True:
			if self.chosen_specular_equation == 'simple':
				myEquation_simple_spec_class.equation_part2_preProcess(myABJ_SD_B)
			elif self.chosen_specular_equation == 'GGX':
				myEquation_GGX_class.equation_part2_preProcess(myABJ_SD_B)

			self.runOnce_part2_preProcess = True

		if self.breakEarlyForRandomLightAndRxyz == True:
			#reset refresh override skips
			self.skip_refresh_override_RdotVpow = False
			self.skip_refresh_override_spectral = False
			self.skip_refresh_override_oren_roughness = False
			self.skip_refresh_override_GGX_roughness = False
			self.skip_refresh_override_GGX_fresnel = False

			self.skip_refresh_override_is_metallic = False
			self.skip_refresh_override_aniso_specular = False
			self.skip_refresh_override_aniso_rotation = False
			self.skip_refresh_override_aniso_roughnessX = False
			self.skip_refresh_override_aniso_roughnessY = False
			self.skip_refresh_override_use_18_hue_colorspace = False

			return
		
		self.profile_stage2_01_a = datetime.now() 

		if self.chosen_specular_equation == 'simple':
			myEquation_simple_spec_class.equation_part2_preProcess_arrow_matricies(myABJ_SD_B)
		elif self.chosen_specular_equation == 'GGX':
			myEquation_GGX_class.equation_part2_preProcess_arrow_matricies(myABJ_SD_B)

		self.profile_stage2_01_b = str(datetime.now() - self.profile_stage2_01_a)
		if self.profileCode_part2 == True:
			print('~~~~~~~~~ self.profile_stage2_01_b (Matrix setup) = ', self.profile_stage2_01_b)


		self.Ci_render_temp_list.clear()
		self.selectedFaceMat_temp_list.clear()

		if self.chosen_specular_equation == 'simple':
			myEquation_simple_spec_class.equation_part3_switch_stages(myABJ_SD_B)
		elif self.chosen_specular_equation == 'GGX':
			myEquation_GGX_class.equation_part3_switch_stages(myABJ_SD_B)

		#############################
		### FINAL RENDER
		#############################
		self.deleteAllText18HueLabelObjects()
		self.textRef_all.clear()

		#experimental depth sort
		for i in self.shadingList_perFace:
			mySplitFaceIndexUsable = i['mySplitFaceIndexUsable']
			faceCenter = i['faceCenter']
			faceCenter_to_L_rayCast = i['faceCenter_to_L_rayCast']
			faceCenter_to_V_rayCast = i['faceCenter_to_V_rayCast']

			# if mySplitFaceIndexUsable in self.Ci_render_temp_list:
			# 	# if faceCenter_to_L_rayCast == True:
			# 	# if faceCenter_to_V_rayCast == True:
			# 	# distance_from_cam = (self.myV - faceCenter).length
			# 	distance_from_cam = (self.myCam.location - faceCenter).length
			# 	# distance_from_cam = (self.myCam.location - faceCenter).length
			# 	# distance_from_cam = self.myCam.location.length - faceCenter.length
			# 	# distance_from_cam = (self.myCam.location - faceCenter)
			# 	self.distanceFromCam_raycastRenderable_list.append(distance_from_cam)




			distance_from_cam = (self.myCam.location - faceCenter).length
			# distance_from_cam = (self.myCam.location - faceCenter).length
			# distance_from_cam = self.myCam.location.length - faceCenter.length
			# distance_from_cam = (self.myCam.location - faceCenter)
			self.distanceFromCam_raycastRenderable_list.append(distance_from_cam)
			
		
		self.distanceFromCam_raycastRenderable_list.sort()

		# distanceFromCam_raycastRenderable_list_newRange = []
		# oldMin = self.distanceFromCam_raycastRenderable_list[0]
		# oldMax = self.distanceFromCam_raycastRenderable_list[-1]

		# for i in self.distanceFromCam_raycastRenderable_list:
		# 	myRemap = self.remap_range(i, oldMin, oldMax, 0, 1)
		# 	distanceFromCam_raycastRenderable_list_newRange.append(myRemap)

		# distanceFromCam_raycastRenderable_list_newRange.sort()

		# self.distanceFromCam_raycastRenderable_list.clear()
		# self.distanceFromCam_raycastRenderable_list = distanceFromCam_raycastRenderable_list_newRange

		for i in self.shadingList_perFace:
			mySplitFaceIndexUsable = i['mySplitFaceIndexUsable']

			if self.skip_refresh_determine(mySplitFaceIndexUsable) == True:
				continue

			mySplitFaceIndexUsable = i['mySplitFaceIndexUsable']
			shadingPlane = i['shadingPlane']
			N_dot_L = i['N_dot_L']
			spec = i['spec']
			attenuation = i['attenuation']

			N_dot_V = i['N_dot_V']
			L = i['L']
			N = i['N']
			T = i['T']
			V = self.myV
			faceCenter = i['faceCenter']

			faceCenter_to_V_rayCast = i['faceCenter_to_V_rayCast']
			faceCenter_to_L_rayCast = i['faceCenter_to_L_rayCast']

			# if mySplitFaceIndexUsable in self.Ci_render_temp_list:
			finalDiffuse = 1
			pos_L2_global_v = mathutils.Vector((0, 0, -40))

			debugVal = .05

			if faceCenter_to_V_rayCast == False:
				finalDiffuse = 0
				finalDiffuse = N_dot_L * debugVal

				finalDiffuseL2 = 0
				finalDiffuseL2 =  N_dot_L * debugVal
				# finalDiffuseL2 = N_dot_L2

			if faceCenter_to_L_rayCast == False:
				finalDiffuse = 0
				finalDiffuse =  N_dot_L * debugVal

				finalDiffuseL2 = 0
				finalDiffuseL2 =  N_dot_L * debugVal

			if faceCenter_to_L_rayCast == True:
				if usableDiffuseEquationType_id == 'oren':
					finalDiffuse = self.oren(N_dot_L, V, L, N, N_dot_V, self.oren_roughness_stored)

				elif usableDiffuseEquationType_id == 'simple':
					finalDiffuse = N_dot_L

				myL_L2 = mathutils.Vector((pos_L2_global_v - faceCenter)).normalized()

				N_dot_L2 = max(np.dot(N, myL_L2), 0.0)
				finalDiffuseL2 = N_dot_L2

			#################
			#GI or multiple lights
			#################
			diff_Cs_V = mathutils.Vector((1.0, 0.0, 0.0))

			# ### ADDITIONAL DIFFUSE GI APPROX
			# # pos_L2_global_v = mathutils.Vector((0, 0, -20))
			# pos_L2_global_v = mathutils.Vector((0, 0, -40))
			# myL_L2 = mathutils.Vector((pos_L2_global_v - faceCenter)).normalized()

			# distanceL1 = (self.pos_light_global_v - faceCenter).length
			distanceL1 = (self.myV - faceCenter).length

			distanceL2 = (pos_L2_global_v - faceCenter).length
			attenuationL2 = 1.0 / (distanceL2 * distanceL2)

			# N_dot_L2 = max(np.dot(N, myL_L2), 0.0)
			# finalDiffuseL2 = N_dot_L2

			self.final_Ci_output(aov_id, shadingPlane, mySplitFaceIndexUsable, finalDiffuse, finalDiffuseL2, spec, distanceL1, distanceL2, attenuation, attenuationL2, faceCenter_to_V_rayCast, faceCenter_to_L_rayCast, faceCenter)

			if mySplitFaceIndexUsable in self.selectedFaceMat_temp_list:
				if self.renderPasses_simple == False and self.renderPasses_GGX == False:
					self.setActiveStageMaterial(shadingPlane, mySplitFaceIndexUsable, self.shadingPlane_sel_r, self.shadingPlane_sel_g, self.shadingPlane_sel_b)
						


			# if self.renderPasses_simple == False and self.renderPasses_GGX == False:
				# self.setActiveStageMaterial(shadingPlane, mySplitFaceIndexUsable, self.shadingPlane_sel_r, self.shadingPlane_sel_g, self.shadingPlane_sel_b)
			# self.setActiveStageMaterial(shadingPlane, mySplitFaceIndexUsable, self.shadingPlane_sel_r, self.shadingPlane_sel_g, self.shadingPlane_sel_b)
						



			'''
			if mySplitFaceIndexUsable in self.Ci_render_temp_list:
				finalDiffuse = 1

				if faceCenter_to_L_rayCast == False:
					finalDiffuse = 0

				if faceCenter_to_L_rayCast == True:
					if usableDiffuseEquationType_id == 'oren':
						finalDiffuse = self.oren(N_dot_L, V, L, N, N_dot_V, self.oren_roughness_stored)

					elif usableDiffuseEquationType_id == 'simple':
						finalDiffuse = N_dot_L

					# distance_from_cam = (self.myV - faceCenter).length
					# self.distanceFromCam_raycastRenderable_list.append(distance_from_cam)
					# self.distanceFromCam_raycastRenderable_list.sort()

				# finalDiffuse = self.oren(N_dot_L, V, L, N, N_dot_V, self.oren_roughness_stored)

				# distance_from_cam = (self.myV - faceCenter).length
				# self.distanceFromCam_raycastRenderable_list.append(distance_from_cam)
				# self.distanceFromCam_raycastRenderable_list.sort()

				#################
				#GI or multiple lights
				#################

				diff_Cs_V = mathutils.Vector((1.0, 0.0, 0.0))

				### ADDITIONAL DIFFUSE GI APPROX
				# pos_L2_global_v = mathutils.Vector((0, 0, -20))
				pos_L2_global_v = mathutils.Vector((0, 0, -40))
				myL_L2 = mathutils.Vector((pos_L2_global_v - faceCenter)).normalized()

				# distanceL1 = (self.pos_light_global_v - faceCenter).length
				distanceL1 = (self.myV - faceCenter).length

				distanceL2 = (pos_L2_global_v - faceCenter).length
				attenuationL2 = 1.0 / (distanceL2 * distanceL2)

				N_dot_L2 = max(np.dot(N, myL_L2), 0.0)
				# diff_Cs_V_L2 = mathutils.Vector((0.0, 1.0, 0.0))


				# combo = finalDiffuse + N_dot_L2

				# finalDiffuseL2 = (finalDiffuse * diff_Cs_V * attenuation) + (N_dot_L2 * diff_Cs_V_L2 * attenuationL2)
				finalDiffuseL2 = N_dot_L2


				# self.final_Ci_output(aov_id, shadingPlane, mySplitFaceIndexUsable, finalDiffuse, spec, attenuation, faceCenter_to_V_rayCast, faceCenter_to_L_rayCast)
				# self.final_Ci_output(aov_id, shadingPlane, mySplitFaceIndexUsable, finalDiffuse, spec, attenuation, faceCenter_to_V_rayCast, faceCenter_to_L_rayCast)
				self.final_Ci_output(aov_id, shadingPlane, mySplitFaceIndexUsable, finalDiffuse, finalDiffuseL2, spec, distanceL1, distanceL2, attenuation, attenuationL2, faceCenter_to_V_rayCast, faceCenter_to_L_rayCast, faceCenter)

			elif mySplitFaceIndexUsable in self.selectedFaceMat_temp_list:
				if self.renderPasses_simple == False and self.renderPasses_GGX == False:
					self.setActiveStageMaterial(shadingPlane, mySplitFaceIndexUsable, self.shadingPlane_sel_r, self.shadingPlane_sel_g, self.shadingPlane_sel_b)

				'''

		# bpy.ops.object.mode_set(mode="OBJECT")

		#reset refresh override skips
		self.skip_refresh_override_RdotVpow = False
		self.skip_refresh_override_spectral = False
		self.skip_refresh_override_oren_roughness = False
		self.skip_refresh_override_GGX_roughness = False
		self.skip_refresh_override_GGX_fresnel = False

		self.skip_refresh_override_is_metallic = False
		self.skip_refresh_override_aniso_specular = False
		self.skip_refresh_override_aniso_rotation = False
		self.skip_refresh_override_aniso_roughnessX = False
		self.skip_refresh_override_aniso_roughnessY = False

		#HIDE IN RENDER ALSO
		self.myCubeCam.hide_render = True
		self.myCubeCam_dupe.hide_render = True
		self.myCubeH_og.hide_render = True
		self.myCubeH_dupe.hide_render = True

		self.myCubeLight_dupe.hide_render = True
		self.myCubeLight_og.hide_render = True

		self.myCubeN_dupe.hide_render = True
		self.myCubeN_og.hide_render = True

		self.myCubeR_dupe.hide_render = True
		self.myCubeR_og.hide_render = True

		#default hide text
		self.showhideText_UI()
	
		print('TIME TO COMPLETE (render) = ' + str(datetime.now() - startTime))
		print(' ')

		# self.updateScene()

	def test_gradientRows(self, startIdx, rangeLength):
		gradientArrayTest = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]

		sortedRows = []
		usableCurrRow = 0

		for idx, i in enumerate(gradientArrayTest):
			myRange = range(startIdx, startIdx + rangeLength)
			
			usable_Z_Row = None

			if idx > myRange[-1]:
				startIdx = startIdx + rangeLength
				myRange = range(startIdx, startIdx + rangeLength)
				usableCurrRow += 1
			
			if myRange[0] <= idx <= myRange[-1]:
				usable_Z_Row = usableCurrRow

			sortedRows.append(usable_Z_Row)

		print('gradientArrayTest = ', gradientArrayTest)
		print('sortedRows = ', sortedRows)

	#This method calls a function from spectral.py, under MIT license (see file)
	def printColorGradient(self):

		usableTextRGBPrecision_items = bpy.context.scene.bl_rna.properties['text_rgb_precision_enum_prop'].enum_items
		usableTextRGBPrecision_id = usableTextRGBPrecision_items[bpy.context.scene.text_rgb_precision_enum_prop].identifier

		precisionVal = int(usableTextRGBPrecision_id)
		usableSteps = None

		val_gradient_method0_step_prop = bpy.context.scene.gradient_method0_step_prop

		steps = val_gradient_method0_step_prop
		# steps = 10

		val_gradient_color0_prop = bpy.context.scene.gradient_color0_prop
		val_gradient_color1_prop = bpy.context.scene.gradient_color1_prop

		startColor = val_gradient_color0_prop
		endColor = val_gradient_color1_prop

		outputRatio_x = None
		outputRatio_y = None
		outputRatio_z = None
		allOutputRatios = []

		usableTextRGBPrecision_items = bpy.context.scene.bl_rna.properties['text_rgb_precision_enum_prop'].enum_items
		usableTextRGBPrecision_id = usableTextRGBPrecision_items[bpy.context.scene.text_rgb_precision_enum_prop].identifier

		precisionVal = int(usableTextRGBPrecision_id)

		if precisionVal == -1:
			# precisionVal = 5
			precisionVal = 3 # temp


		usableAdditiveOrSubtractiveColorBlending_items = bpy.context.scene.bl_rna.properties['additive_or_subtractive_color_blending_enum_prop'].enum_items
		usableAdditiveOrSubtractiveColorBlending_id = usableAdditiveOrSubtractiveColorBlending_items[bpy.context.scene.additive_or_subtractive_color_blending_enum_prop].identifier

		if usableAdditiveOrSubtractiveColorBlending_id == 'additive':
			pass

		elif usableAdditiveOrSubtractiveColorBlending_id == 'subtractive':
			startColor = (startColor[0] * 255, startColor[1] * 255, startColor[2] * 255)
			endColor = (endColor[0] * 255, endColor[1] * 255, endColor[2] * 255)


		# endColor = (1, 1, 0)
		# startColor = (0, 0, 1)

		# endColor = (1.0, 1.0, 0.0)
		# startColor = (0.0, 0.0, 1.0)

		# endColor = (255, 255, 0)
		# startColor = (0, 0, 255)


		for i in range(steps + 1):
			lerpIter = None
			if i == 0:
				lerpIter = 1

			elif i == steps:
				lerpIter = 0

			else:
				lerpIter = 1 - (i / steps)

			# lerpIter = 0.5

			comboRatio_xyz = None

			if usableAdditiveOrSubtractiveColorBlending_id == 'additive':
				outputRatio_x = self.lerp(endColor[0], startColor[0], lerpIter)
				outputRatio_y = self.lerp(endColor[1], startColor[1], lerpIter)
				outputRatio_z = self.lerp(endColor[2], startColor[2], lerpIter)

				comboRatio_xyz = mathutils.Vector((outputRatio_x, outputRatio_y, outputRatio_z))

			elif usableAdditiveOrSubtractiveColorBlending_id == 'subtractive':
				outputRatio_x = spectral.spectral_mix(endColor, startColor, lerpIter)
				comboRatio_xyz = mathutils.Vector((outputRatio_x[0] / 255, outputRatio_x[1] / 255, outputRatio_x[2] / 255))

			val_gamma_correct_gradient_color_prop = bpy.context.scene.gamma_correct_gradient_color_prop

			if val_gamma_correct_gradient_color_prop == True:
				gammaCorrect = mathutils.Vector((2.2, 2.2, 2.2))
				gammaCorrect_r = pow(comboRatio_xyz.x, gammaCorrect.x)
				gammaCorrect_g = pow(comboRatio_xyz.y, gammaCorrect.y)
				gammaCorrect_b = pow(comboRatio_xyz.z, gammaCorrect.z)

				comboRatio_xyz = mathutils.Vector((gammaCorrect_r, gammaCorrect_g, gammaCorrect_b))

			allOutputRatios.append(comboRatio_xyz)

		self.makeGradientGrid_color(allOutputRatios)

		self.defaultColorSettings_UI()

		# print(allOutputRatios)
		# reversed_allOutputRatios = list(reversed(allOutputRatios))
		# print(reversed_allOutputRatios)

	def colorWheel_dynamic_inner(self, i, segments, center_x, center_y, lerpIter_outer, gradient_inner_circle_steps, myInputMesh, startColor, endColor, endColor_mix):

		print('i = ', i)

		outputRatio_x = self.lerp(endColor.x, startColor.x, lerpIter_outer)
		outputRatio_y = self.lerp(endColor.y, startColor.y, lerpIter_outer)
		outputRatio_z = self.lerp(endColor.z, startColor.z, lerpIter_outer)

		# endColor_white = mathutils.Vector((1.0, 1.0, 1.0))
		# endColor_black = mathutils.Vector((0.2, 0.2, 0.2))

		gradient_inner_circle_steps_usable = gradient_inner_circle_steps - 1

		for j in range(gradient_inner_circle_steps):
			finalOutputColors = []
			finalOutputColors_outer = []

			lerpIter_inner = None
			# lerpIter_outer = None

			if j == 0:
				if endColor_mix != mathutils.Vector((1.0, 1.0, 1.0)):
					continue
				lerpIter_inner = 1

			elif j == gradient_inner_circle_steps:
				lerpIter_inner = 0

			else:
				lerpIter_inner = 1 - (j / gradient_inner_circle_steps_usable)

			angle = 2 * math.pi * i / segments

			radius = 2
			lerpIter_inner_radius = radius * lerpIter_inner
			# radius_additional = 2.125
			# radius_additional = 2.5
			radius_additional = 2.35


			lerpIter_inner_radius_additional = radius_additional * lerpIter_inner
			x_additional = lerpIter_inner_radius_additional * math.cos(angle) + center_x
			y_additional = lerpIter_inner_radius_additional * math.sin(angle) + center_y

			x = lerpIter_inner_radius * math.cos(angle) + center_x
			y = lerpIter_inner_radius * math.sin(angle) + center_y

			if endColor_mix == mathutils.Vector((1.0, 1.0, 1.0)):
				outputRatio_x_01 = self.lerp(outputRatio_x, endColor_mix.x, lerpIter_inner)
				outputRatio_y_01 = self.lerp(outputRatio_y, endColor_mix.y, lerpIter_inner)
				outputRatio_z_01 = self.lerp(outputRatio_z, endColor_mix.z, lerpIter_inner)

			else:
				outputRatio_x_01 = self.lerp(endColor_mix.x, outputRatio_x, lerpIter_inner)
				outputRatio_y_01 = self.lerp(endColor_mix.y, outputRatio_y, lerpIter_inner)
				outputRatio_z_01 = self.lerp(endColor_mix.z, outputRatio_z, lerpIter_inner)

			val_gamma_correct_gradient_colorWheel_prop = bpy.context.scene.gamma_correct_gradient_colorWheel_prop

			# if val_gamma_correct_gradient_colorWheel_prop == True:
			# 	gammaCorrect = mathutils.Vector((2.2, 2.2, 2.2))
			# 	outputRatio_x_01 = pow(outputRatio_x_01, gammaCorrect.x)
			# 	outputRatio_y_01 = pow(outputRatio_y_01, gammaCorrect.y)
			# 	outputRatio_z_01 = pow(outputRatio_z_01, gammaCorrect.z)

			#always up down
			# textRaiseLowerZ = 0.05 * lerpIter_inner
			# if j % 2 == 0:
			# 	#even
			# 	textRaiseLowerZ *= 1
			# else:
			# 	#odd
			# 	textRaiseLowerZ *= -1

			#always middle except for near center column
			textRaiseLowerZ = 0

			# if j >= (gradient_inner_circle_steps[4]):
			# if j >= 4:

			if (-5 + (segments * .25)) <= i <= (5 + (segments * .25)):
				# textRaiseLowerZ = 0.15 * lerpIter_inner
				textRaiseLowerZ = 0.05 * lerpIter_inner
			elif (-5 + (segments * .75)) <= i <= (5 + (segments * .75)):
				# textRaiseLowerZ = 0.15 * lerpIter_inner
				textRaiseLowerZ = 0.05 * lerpIter_inner

			if i % 2 == 0:
				#even
				textRaiseLowerZ *= -1
			else:
				#odd
				textRaiseLowerZ *= 1

			comboRatio_xyz_final = mathutils.Vector((outputRatio_x_01, outputRatio_y_01, outputRatio_z_01))
			finalOutputColors.append(comboRatio_xyz_final)


			colorspace_18_hue_dict = {}

			additionalText = None

			if i == 0:
				additionalText = 'R'
			elif i == 1:
				additionalText = 'RO'
			elif i == 2:
				additionalText = 'OR'
			elif i == 3:
				additionalText = 'O'
			elif i == 4:
				additionalText = 'OY'
			elif i == 5:
				additionalText = 'YO'
			elif i == 6:
				additionalText = 'Y'
			elif i == 7:
				additionalText = 'YG'
			elif i == 8:
				additionalText = 'GY'
			elif i == 9:
				additionalText = 'G'
			elif i == 10:
				additionalText = 'GB'
			elif i == 11:
				additionalText = 'BG'
			elif i == 12:
				additionalText = 'B'
			elif i == 13:
				additionalText = 'BV'
			elif i == 14:
				additionalText = 'VB'
			elif i == 15:
				additionalText = 'V'
			elif i == 16:
				additionalText = 'VR'
			elif i == 17:
				additionalText = 'RV'

			if comboRatio_xyz_final == mathutils.Vector((1.0, 1.0, 1.0)):
				colorspace_18_hue_dict = {
					'identifier' : 'BW',
					'value' : comboRatio_xyz_final,
					'idx' : 0,
				}

			elif comboRatio_xyz_final <= mathutils.Vector((0.05, 0.05, 0.05)):
				colorspace_18_hue_dict = {
					'identifier' : 'BW',
					'value' : mathutils.Vector((0.05, 0.05, 0.05)),
					'idx' : 19,
				}

			# elif comboRatio_xyz_final <= mathutils.Vector((0.0, 0.0, 0.0)):
			# 	colorspace_18_hue_dict = {
			# 		'identifier' : 'BW',
			# 		'value' : comboRatio_xyz_final,
			# 		'idx' : 19,
			# 	}

			else:
				# print('in regular 18 hue dict addition for : ', additionalText, ' ', j)
				colorspace_18_hue_dict = {
					'identifier' : additionalText,
					'value' : comboRatio_xyz_final,
					# 'idx' : i,
					# 'idx' : j,
					'idx' : j + self.colorspace_18_continued_j,
					# 'idx' : j,
				}

			self.colorspace_18_hue_list.append(colorspace_18_hue_dict)

			# print(self.colorspace_18_hue_list)

			# for e in self.colorspace_18_hue_list:
			# 	# print(self.colorspace_18_hue_list[e][0])
			# 	print('identifier : ', e['identifier'])
			# 	# print(' ')
			# 	print('value : ', e['value'])
			# 	# print(' ')
			# 	print('idx : ', e['idx'])
			# 	# print(' ')

			# 	print(' ')

			#################################
			additionalTextUsable = None

			if self.val_gradient_circle_override == 0:
				additionalTextUsable = ''

			elif self.val_gradient_circle_override == 1:
				additionalTextUsable = additionalText

			self.makeGradientGrid_color_circular(finalOutputColors, x, y, myInputMesh, lerpIter_inner, textRaiseLowerZ, additionalTextUsable, x_additional, y_additional)

	def colorGradient_circular_preset18_0(self):
		#make preset
		self.val_gradient_circle_override = 1
		self.val_gradient_circle_override_side = 'outside'

		self.printColorGradient_circular()

		self.val_gradient_circle_override = 0
		self.val_gradient_circle_override_side = None

		self.colorWheelGradient_18_created = True

	def colorGradient_circular_preset18_1(self):
		#make preset
		self.val_gradient_circle_override = 1
		self.val_gradient_circle_override_side = 'inside'

		self.printColorGradient_circular()

		self.val_gradient_circle_override = 0
		self.val_gradient_circle_override_side = None

	def printColorGradient_circular(self):
		# outer circle steps 3
		# inner_circle_steps 10
		
		self.deselectAll()
		self.deleteAllObjects()
		self.mega_purge()
		
		self.textRef_all.clear()

		# for area in bpy.data.screens["Scripting"].areas:
		for area in bpy.data.screens["Layout"].areas:
		# for area in bpy.data.screens[bpy.context.window.screen].areas:
			if area.type == 'VIEW_3D':
				for space in area.spaces:
					if space.type == 'VIEW_3D':

						# space.overlay.grid_scale = 2

						usableToggle = None
						if space.overlay.show_floor == True:
							usableToggle = False

							space.overlay.show_floor = usableToggle
							space.overlay.show_axis_x = usableToggle
							space.overlay.show_axis_y = usableToggle
							space.overlay.show_axis_z = usableToggle
							space.overlay.show_cursor = usableToggle

		bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)
		bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')

		###########
		#DEFAULT CAMERA
		#############
		cam1_data = bpy.data.cameras.new('Camera')
		cam = bpy.data.objects.new('Camera', cam1_data)
		bpy.context.collection.objects.link(cam)

		###################################
		###### SET CAMERA POS / LOOK AT
		###################################
		self.myCam = bpy.data.objects["Camera"]

		# self.myCam.location = self.pos_camera_global
		self.myCam.location = mathutils.Vector((20, 0, 0))

		# bpy.context.object.data.type = 'ORTHO'
		self.myCam.data.type = 'ORTHO'
		self.myCam.data.ortho_scale = 11.7

		#near 8.5x11 printable ratio
		# bpy.context.scene.render.resolution_x = 3900
		# bpy.context.scene.render.resolution_y = 3000

		bpy.context.scene.render.resolution_x = 2550
		bpy.context.scene.render.resolution_y = 1970

		self.updateScene() # need

		self.look_at(self.myCam, self.myOrigin)

		self.myV = self.myCam.matrix_world.to_translation()
		self.myV.normalize()

		#####################
		### input mesh
		#####################

		usablePrimitiveType_gradient_id = 'grid'
		if usablePrimitiveType_gradient_id == 'grid':
			bpy.ops.mesh.primitive_grid_add()

		myInputMesh = bpy.context.active_object
		myInputMesh.select_set(1)
		myInputMesh.hide_set(1)
		# myInputMesh.hide_render = True

		#####################
		### grey background
		#####################
		bpy.context.view_layer.objects.active = myInputMesh
		myDupeGradient_bg = self.copyObject()
		myDupeGradient_bg.name = 'dupeGradient_background'
		# myDupeGradient_bg.scale = mathutils.Vector((5, 5, 5))
		myDupeGradient_bg.scale = mathutils.Vector((5, 8.5, 11))
		myDupeGradient_bg.location = mathutils.Vector((-1, 0, 0))
		myDupeGradient_bg.rotation_euler = mathutils.Vector((0, math.radians(90), 0))

		bpy.context.view_layer.objects.active = myDupeGradient_bg

		gamma_correct_gradient_colorWheel_prop = bpy.context.scene.gamma_correct_gradient_colorWheel_prop
		greyBG = 0.5
		if gamma_correct_gradient_colorWheel_prop == True:
			# lerpIter = pow(lerpIter, 1.0 / 2.2)
			greyBG = pow(greyBG, 2.2)

		mat1 = self.newShader("ShaderVisualizer_gradientBG", "emission", greyBG, greyBG, greyBG)

		bpy.context.active_object.data.materials.clear()
		bpy.context.active_object.data.materials.append(mat1)

		# Define circle parameters
		center_x = 0
		center_y = 0

		val_gradient_outer_circle_steps_prop = bpy.context.scene.gradient_outer_circle_steps_prop
		val_gradient_inner_circle_steps_prop = bpy.context.scene.gradient_inner_circle_steps_prop

		if self.val_gradient_circle_override == 1:
			val_gradient_outer_circle_steps_prop = 3
			val_gradient_inner_circle_steps_prop = 10

		#make preset

		divisor = 6
		segments = val_gradient_outer_circle_steps_prop * divisor  # Adjust for smoother circle

		countToDivisorMultiplier_list = []

		outerCircle_steps_usable = val_gradient_outer_circle_steps_prop - 1

		for i in range(divisor):
			for j in range(val_gradient_outer_circle_steps_prop):

				if j == 0:
					lerpIter = 1

				elif j == val_gradient_outer_circle_steps_prop:
					lerpIter = 0

				else:
					lerpIter = 1 - (j / outerCircle_steps_usable)

				countToDivisorMultiplier_list.append(lerpIter)

		# print('countToDivisorMultiplier_list = ', countToDivisorMultiplier_list)

		for i in range(segments):
			outputRatio_x = None
			outputRatio_y = None
			outputRatio_z = None
			allOutputRatios = []

			# usableTextRGBPrecision_items = bpy.context.scene.bl_rna.properties['text_rgb_precision_enum_prop'].enum_items
			# usableTextRGBPrecision_id = usableTextRGBPrecision_items[bpy.context.scene.text_rgb_precision_enum_prop].identifier

			# precisionVal = int(usableTextRGBPrecision_id)

			# if precisionVal == -1:
			# 	# precisionVal = 5
			# 	precisionVal = 3 # temp

			#ROYGBIV
			#Red (1, 0, 0), 
			# Orange (1, 0.5, 0), 
			# Yellow (1, 1, 0), 
			# Green (0, 1, 0), 
			# Blue (0, 0, 1), 
			# Violet (0.5, 0, 0.5)

			lerpIter_outer = countToDivisorMultiplier_list[i]

			# center_x = 0
			# center_y = 0
			# radius = 2
			# angle = 2 * math.pi * i / segments
			# center_x = radius * math.cos(angle) + center_x
			# center_y = radius * math.sin(angle) + center_y

			self.circular_gradient_text_counter = 0
			# self.colorspace_18_hue_list = []
			# self.colorspace_18_hue_list.clear()

			if 0 <= i < (segments / divisor):
				print('1 divisor : ', i)
				#RED TO ORANGE
				startColor = mathutils.Vector((1.0, 0.0, 0.0))
				endColor = mathutils.Vector((1, 0.5 - (1.0 / val_gradient_outer_circle_steps_prop), 0.0))
				# endColor_black = mathutils.Vector((0.0, 0.0, 0.0))
				endColor_black = mathutils.Vector((0.05, 0.05, 0.05))
				endColor_white = mathutils.Vector((1.0, 1.0, 1.0))

				center_x = 0
				center_y = 0
				radius = 2
				angle = 2 * math.pi * i / segments
				center_x = radius * math.cos(angle) + center_x
				center_y = radius * math.sin(angle) + center_y

				self.colorspace_18_continued_j = 1
				self.colorWheel_dynamic_inner(i, segments, center_x, center_y, lerpIter_outer, val_gradient_inner_circle_steps_prop, myInputMesh, startColor, endColor, endColor_white)

				center_x = 0
				center_y = 0

				self.colorspace_18_continued_j = val_gradient_inner_circle_steps_prop
				self.colorWheel_dynamic_inner(i, segments, center_x, center_y, lerpIter_outer, val_gradient_inner_circle_steps_prop, myInputMesh, startColor, endColor, endColor_black)
			
			elif (segments / divisor) <= i < (segments / (divisor / 2)):
				# print('2 divisor : ', i, ' lerpIter01 ', lerpIter)
				print('2 divisor : ', i)

				#ORANGE TO YELLOW
				startColor = mathutils.Vector((1.0, 0.5, 0.0))
				endColor = mathutils.Vector((1, 1 - (1.0 / val_gradient_outer_circle_steps_prop), 0.0)) #####
				# endColor_black = mathutils.Vector((0.0, 0.0, 0.0))
				endColor_black = mathutils.Vector((0.05, 0.05, 0.05))
				endColor_white = mathutils.Vector((1.0, 1.0, 1.0))

				center_x = 0
				center_y = 0
				radius = 2
				angle = 2 * math.pi * i / segments
				center_x = radius * math.cos(angle) + center_x
				center_y = radius * math.sin(angle) + center_y

				self.colorspace_18_continued_j = 1
				self.colorWheel_dynamic_inner(i, segments, center_x, center_y, lerpIter_outer, val_gradient_inner_circle_steps_prop, myInputMesh, startColor, endColor, endColor_white)

				center_x = 0
				center_y = 0

				self.colorspace_18_continued_j = val_gradient_inner_circle_steps_prop
				self.colorWheel_dynamic_inner(i, segments, center_x, center_y, lerpIter_outer, val_gradient_inner_circle_steps_prop, myInputMesh, startColor, endColor, endColor_black)

			elif (segments / (divisor / 2)) <= i < (segments / (divisor / 3)):
				# print('3 divisor : ', i, ' lerpIter01 ', lerpIter)
				print('3 divisor : ', i)

				#YELLOW TO GREEN
				startColor = mathutils.Vector((1.0, 1.0, 0.0))
				endColor = mathutils.Vector((1.0 / val_gradient_outer_circle_steps_prop, 1, 0.0)) #####
				# endColor = mathutils.Vector((0, 1 - (1.0 / val_gradient_outer_circle_steps_prop), 0.0)) #####
				# endColor_black = mathutils.Vector((0.0, 0.0, 0.0))
				endColor_black = mathutils.Vector((0.05, 0.05, 0.05))
				endColor_white = mathutils.Vector((1.0, 1.0, 1.0))

				center_x = 0
				center_y = 0
				radius = 2
				angle = 2 * math.pi * i / segments
				center_x = radius * math.cos(angle) + center_x
				center_y = radius * math.sin(angle) + center_y

				self.colorspace_18_continued_j = 1
				self.colorWheel_dynamic_inner(i, segments, center_x, center_y, lerpIter_outer, val_gradient_inner_circle_steps_prop, myInputMesh, startColor, endColor, endColor_white)

				center_x = 0
				center_y = 0

				self.colorspace_18_continued_j = val_gradient_inner_circle_steps_prop
				self.colorWheel_dynamic_inner(i, segments, center_x, center_y, lerpIter_outer, val_gradient_inner_circle_steps_prop, myInputMesh, startColor, endColor, endColor_black)

			elif (segments / (divisor / 3)) <= i < (segments / (divisor / 4)):
				print('4 divisor : ', i)

				#GREEN TO BLUE
				startColor = mathutils.Vector((0.0, 1.0, 0.0))
				endColor = mathutils.Vector((0, 1.0 / val_gradient_outer_circle_steps_prop, 1 - (1.0 / val_gradient_outer_circle_steps_prop))) #####
				# endColor_black = mathutils.Vector((0.0, 0.0, 0.0))
				endColor_black = mathutils.Vector((0.05, 0.05, 0.05))
				endColor_white = mathutils.Vector((1.0, 1.0, 1.0))

				center_x = 0
				center_y = 0
				radius = 2
				angle = 2 * math.pi * i / segments
				center_x = radius * math.cos(angle) + center_x
				center_y = radius * math.sin(angle) + center_y

				self.colorspace_18_continued_j = 1
				self.colorWheel_dynamic_inner(i, segments, center_x, center_y, lerpIter_outer, val_gradient_inner_circle_steps_prop, myInputMesh, startColor, endColor, endColor_white)

				center_x = 0
				center_y = 0

				self.colorspace_18_continued_j = val_gradient_inner_circle_steps_prop
				self.colorWheel_dynamic_inner(i, segments, center_x, center_y, lerpIter_outer, val_gradient_inner_circle_steps_prop, myInputMesh, startColor, endColor, endColor_black)

			elif (segments / (divisor / 4)) <= i < (segments / (divisor / 5)):
				print('5 divisor : ', i)

				#BLUE TO VIOLET
				startColor = mathutils.Vector((0.0, 0.0, 1.0))
				endColor = mathutils.Vector((0.5 - (1.0 / val_gradient_outer_circle_steps_prop), 0, 0.5 - (1.0 / val_gradient_outer_circle_steps_prop))) #####
				# endColor_black = mathutils.Vector((0.0, 0.0, 0.0))
				endColor_black = mathutils.Vector((0.05, 0.05, 0.05))
				endColor_white = mathutils.Vector((1.0, 1.0, 1.0))

				center_x = 0
				center_y = 0
				radius = 2
				angle = 2 * math.pi * i / segments
				center_x = radius * math.cos(angle) + center_x
				center_y = radius * math.sin(angle) + center_y

				self.colorspace_18_continued_j = 1
				self.colorWheel_dynamic_inner(i, segments, center_x, center_y, lerpIter_outer, val_gradient_inner_circle_steps_prop, myInputMesh, startColor, endColor, endColor_white)

				center_x = 0
				center_y = 0

				self.colorspace_18_continued_j = val_gradient_inner_circle_steps_prop
				self.colorWheel_dynamic_inner(i, segments, center_x, center_y, lerpIter_outer, val_gradient_inner_circle_steps_prop, myInputMesh, startColor, endColor, endColor_black)

			elif (segments / (divisor / 5)) <= i < (segments / (divisor / 6)):
				print('6 divisor : ', i)

				#VIOLET TO RED
				startColor = mathutils.Vector((0.5, 0.0, 0.5))
				# endColor = mathutils.Vector((1 - (1.0 / val_gradient_outer_circle_steps_prop), 0.0, 0.0)) #####
				endColor = mathutils.Vector((1 - (1.0 / val_gradient_outer_circle_steps_prop), 0.0, 1.0 / val_gradient_outer_circle_steps_prop)) #####
				# endColor_black = mathutils.Vector((0.0, 0.0, 0.0))
				endColor_black = mathutils.Vector((0.05, 0.05, 0.05))
				endColor_white = mathutils.Vector((1.0, 1.0, 1.0))

				center_x = 0
				center_y = 0
				radius = 2
				angle = 2 * math.pi * i / segments
				center_x = radius * math.cos(angle) + center_x
				center_y = radius * math.sin(angle) + center_y

				self.colorspace_18_continued_j = 1
				self.colorWheel_dynamic_inner(i, segments, center_x, center_y, lerpIter_outer, val_gradient_inner_circle_steps_prop, myInputMesh, startColor, endColor, endColor_white)

				center_x = 0
				center_y = 0

				self.colorspace_18_continued_j = val_gradient_inner_circle_steps_prop
				self.colorWheel_dynamic_inner(i, segments, center_x, center_y, lerpIter_outer, val_gradient_inner_circle_steps_prop, myInputMesh, startColor, endColor, endColor_black)

			# '''

		self.defaultColorSettings_UI()

		myInputMesh.hide_render = True

		self.deselectAll()

		myDupeGradient_bg.select_set(0)


		removeDuplicates_colorwheelList = []
		for c in self.colorspace_18_hue_list:
			if c not in removeDuplicates_colorwheelList:
				removeDuplicates_colorwheelList.append(c)

		self.colorspace_18_hue_list.clear()
		self.colorspace_18_hue_list = removeDuplicates_colorwheelList

		for e in self.colorspace_18_hue_list:
			print('18 HUE LIST GENERATION - identifier : ', e['identifier'], 'value : ', e['value'], 'idx : ', e['idx'])

	def makeGradientGrid_color_circular(self, gradientArray, xPos, yPos, myInputMesh, lerpIter_inner, textRaiseLowerZ, additionalText, x_additional, y_additional):
		#####################
		### each gradient face
		#####################
		gradientScale = 0.05
		raiseLowerZ = 0

		usableTextRGBPrecision_items = bpy.context.scene.bl_rna.properties['text_rgb_precision_enum_prop'].enum_items
		usableTextRGBPrecision_id = usableTextRGBPrecision_items[bpy.context.scene.text_rgb_precision_enum_prop].identifier

		precisionVal = int(usableTextRGBPrecision_id)

		if precisionVal == -1:
			precisionVal = 3

		# precisionVal = 1 # DEBUG
		precisionVal = 2 # DEBUG
		# precisionVal = 3 # DEBUG

		val_text_gradient_rotate_x_prop = bpy.context.scene.text_gradient_rotate_x_prop
		val_text_gradient_rotate_y_prop = bpy.context.scene.text_gradient_rotate_y_prop
		val_text_gradient_rotate_z_prop = bpy.context.scene.text_gradient_rotate_z_prop

		myRotation = mathutils.Vector((math.radians(90), math.radians(val_text_gradient_rotate_y_prop), math.radians(90)))

		for idx, i in enumerate(gradientArray):
			bpy.context.view_layer.objects.active = myInputMesh
			myDupeGradient = self.copyObject()
			myDupeGradient.name = additionalText + '_' + str(self.circular_gradient_text_counter) + '_dupeGradient_'

			gradientScale_lerped = self.clamp(gradientScale * lerpIter_inner, 0.025, 1)
			
			gradientScale_lerped = 0.075 #constant

			if self.circular_gradient_text_counter > 15:
				gradientScale_lerped = 0.045 #constant

			if self.circular_gradient_text_counter >= 17:
				gradientScale_lerped = 0.03 #constant

			myDupeGradient.scale = mathutils.Vector((gradientScale_lerped, gradientScale_lerped, gradientScale_lerped))

			gradient_startPos = mathutils.Vector((0, xPos, raiseLowerZ + yPos))
			myDupeGradient.location = gradient_startPos
	
			myDupeGradient.rotation_euler = mathutils.Vector((0, math.radians(90), 0))

			Ci = mathutils.Vector((i))	

			bpy.context.view_layer.objects.active = myDupeGradient
			mat1 = self.newShader("ShaderVisualizer_gradient_" + str(i), "emission", Ci.x, Ci.y, Ci.z)
			bpy.context.active_object.data.materials.clear()
			bpy.context.active_object.data.materials.append(mat1)

			#####################
			### text_add()
			#####################
			if precisionVal != -1:

				val_gamma_correct_gradient_colorWheel_prop = bpy.context.scene.gamma_correct_gradient_colorWheel_prop

				t = None

				Ci_gc_text = Ci

				primaryColorList = [mathutils.Vector((1.0, 0.0, 0.0)), mathutils.Vector((1.0, 1.0, 0.0)), mathutils.Vector((0.0, 0.0, 1.0))]

				alterTextLocation = mathutils.Vector((0.0, 0.0, 0.0))

				if val_gamma_correct_gradient_colorWheel_prop == True:
					gammaCorrect = mathutils.Vector((1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2))
					# gammaCorrect = mathutils.Vector((2.2, 2.2, 2.2))
					gammaCorrect_r = pow(Ci.x, gammaCorrect.x)
					gammaCorrect_g = pow(Ci.y, gammaCorrect.y)
					gammaCorrect_b = pow(Ci.z, gammaCorrect.z)

					Ci_gc_text = mathutils.Vector((gammaCorrect_r, gammaCorrect_g, gammaCorrect_b))

				elif val_gamma_correct_gradient_colorWheel_prop == False:
					t = '(' + str(round(Ci.x, precisionVal)) + ', ' + str(round(Ci.y, precisionVal)) + ', ' + str(round(Ci.z, precisionVal)) + ')'

					Ci_gc_text = t

				precisionVal = 1
				myFontScale = None
				t = ''

				if self.circular_gradient_text_counter == 0:
					t = additionalText ########

					myFontScale = 0.3

				else:
					myFontScale = 0.06

					if (self.circular_gradient_text_counter + 1) % 5 == 0:
						t = 'X'
						myFontScale = 0.06 #####


				# if (self.circular_gradient_text_counter + 1) == 10:
				# 	print('bingo for 10 : ', additionalText, ' ', Ci.x, Ci.y, Ci.z)

				#FONT OBJECT NUMBER IDX
				myFontCurve = bpy.data.curves.new(type="FONT", name="myFontCurve")
				myFontCurve.body = t

				myFontOb = bpy.data.objects.new(myDupeGradient.name + '_text', myFontCurve)
				myFontOb.data.align_x = 'CENTER'
				myFontOb.data.align_y = 'CENTER'

				textRaiseLowerZ = 0

				if self.circular_gradient_text_counter == 0:
					myFontOb.location = mathutils.Vector((1, 0, textRaiseLowerZ)) + mathutils.Vector((0, x_additional, y_additional)) ############!!!!!!!!

				else:
					myFontOb.location = myDupeGradient.location + mathutils.Vector((1, 0, 0))

				myFontOb.rotation_euler = myRotation

				myFontOb.scale = mathutils.Vector((myFontScale, myFontScale, myFontScale))
				myFontOb.data.body = t
				bpy.context.collection.objects.link(myFontOb)
				myFontOb.show_in_front = True
				self.textRef_all.append(myFontOb.name)
				bpy.context.view_layer.objects.active = myFontOb

				if self.circular_gradient_text_counter % 2 == 0:
					mat1 = self.newShader("ShaderVisualizer_gradient_text_" + str(i), "emission", 1, 1, 1)
				else:
					mat1 = self.newShader("ShaderVisualizer_gradient_text_" + str(i), "emission", 0, 0, 0)

				if self.circular_gradient_text_counter == 0:
					mat1 = self.newShader("ShaderVisualizer_gradient_text_" + str(i), "emission", 1, 1, 1)

				else:
					if additionalText == 'Y' or additionalText == 'YG' or additionalText == 'GY' or additionalText == 'G' or additionalText == 'GB' or additionalText == 'YO' or additionalText == 'OY' or additionalText == 'O':
						mat1 = self.newShader("ShaderVisualizer_gradient_text_" + str(i), "emission", 0, 0, 0)
					
					else:
						mat1 = self.newShader("ShaderVisualizer_gradient_text_" + str(i), "emission", 1, 1, 1)

				self.circular_gradient_text_counter += 1

				bpy.context.active_object.data.materials.clear()
				bpy.context.active_object.data.materials.append(mat1)

				bpy.context.view_layer.objects.active = myDupeGradient


	def makeGradientGrid_color_tri(self, gradientArray):

		self.deselectAll()
		self.deleteAllObjects()
		self.mega_purge()
		
		self.textRef_all.clear()

		bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)
		bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')

		###########
		#DEFAULT CAMERA
		#############
		cam1_data = bpy.data.cameras.new('Camera')
		cam = bpy.data.objects.new('Camera', cam1_data)
		bpy.context.collection.objects.link(cam)

		###################################
		###### SET CAMERA POS / LOOK AT
		###################################
		self.myCam = bpy.data.objects["Camera"]

		# self.myCam.location = self.pos_camera_global
		self.myCam.location = mathutils.Vector((20, 0, 0))


		# bpy.context.object.data.type = 'ORTHO'
		self.myCam.data.type = 'ORTHO'

		#near 8.5x11 printable ratio
		# bpy.context.scene.render.resolution_x = 3900
		# bpy.context.scene.render.resolution_y = 3000

		bpy.context.scene.render.resolution_x = 2550
		bpy.context.scene.render.resolution_y = 1970

		self.updateScene() # need

		self.look_at(self.myCam, self.myOrigin)

		self.myV = self.myCam.matrix_world.to_translation()
		self.myV.normalize()

		usablePrimitiveType_gradient_id = 'grid'

		if usablePrimitiveType_gradient_id == 'grid':
			bpy.ops.mesh.primitive_grid_add()

		myInputMesh = bpy.context.active_object
		myInputMesh.select_set(1)
		myInputMesh.hide_set(1)
		# myInputMesh.hide_render = True

		#####################
		### grey background
		#####################
		bpy.context.view_layer.objects.active = myInputMesh
		myDupeGradient_bg = self.copyObject()
		myDupeGradient_bg.name = 'dupeGradient_background'
		myDupeGradient_bg.scale = mathutils.Vector((5, 5, 5))
		myDupeGradient_bg.location = mathutils.Vector((-1, 0, 0))
		myDupeGradient_bg.rotation_euler = mathutils.Vector((0, math.radians(90), 0))

		bpy.context.view_layer.objects.active = myDupeGradient_bg

		gamma_correct_gradient_color_prop = bpy.context.scene.gamma_correct_gradient_color_prop
		greyBG = 0.5
		if gamma_correct_gradient_color_prop == True:
			# lerpIter = pow(lerpIter, 1.0 / 2.2)
			greyBG = pow(greyBG, 2.2)

		mat1 = self.newShader("ShaderVisualizer_gradientBG", "emission", greyBG, greyBG, greyBG)

		# mat1 = self.newShader("ShaderVisualizer_gradientBG", "emission", 0.5, 0.5, 0.5)
		bpy.context.active_object.data.materials.clear()
		bpy.context.active_object.data.materials.append(mat1)

		#####################
		### each gradient face
		#####################
		# gradientScale = 0.1
		# gradientScale = 0.15
		gradientScale = 0.175
		# rangeLength = 5
		# rangeLength = 12
		rangeLength = 15

		# locationMultiplierY = .5
		# locationMultiplierY = .35
		locationMultiplierY = .4
		# locationMultiplierZ = -.5
		locationMultiplierZ = -.6
		# locationMultiplierZ = -.75
		raiseLowerZ = 1
		usable_Z_Row = 0

		usableCurrLoc_Y = 0
		usableCurrRow_Z = 0
		startIdx = 0

		usableTextRGBPrecision_items = bpy.context.scene.bl_rna.properties['text_rgb_precision_enum_prop'].enum_items
		usableTextRGBPrecision_id = usableTextRGBPrecision_items[bpy.context.scene.text_rgb_precision_enum_prop].identifier

		precisionVal = int(usableTextRGBPrecision_id)

		if precisionVal == -1:
			precisionVal = 3

		# precisionVal = 1 # DEBUG
		precisionVal = 2 # DEBUG
		# precisionVal = 3 # DEBUG

		val_text_gradient_rotate_x_prop = bpy.context.scene.text_gradient_rotate_x_prop
		val_text_gradient_rotate_y_prop = bpy.context.scene.text_gradient_rotate_y_prop
		val_text_gradient_rotate_z_prop = bpy.context.scene.text_gradient_rotate_z_prop

		myRotation = mathutils.Vector((math.radians(90), math.radians(val_text_gradient_rotate_y_prop), math.radians(90)))

		for idx, i in enumerate(gradientArray):
			bpy.context.view_layer.objects.active = myInputMesh
			myDupeGradient = self.copyObject()
			myDupeGradient.name = 'dupeGradient_' + str(idx)
			myDupeGradient.scale = mathutils.Vector((gradientScale, gradientScale, gradientScale))

			myRange = range(startIdx, startIdx + rangeLength)
			usable_Z_Row = None

			if idx > myRange[-1]:
				startIdx = startIdx + rangeLength
				myRange = range(startIdx, startIdx + rangeLength)
				usableCurrLoc_Y = 0
				usableCurrRow_Z += 1
			
			if myRange[0] <= idx <= myRange[-1]:
				usable_Z_Row = usableCurrRow_Z
				output_Y = usableCurrLoc_Y # * yMult # * idx)
				usableCurrLoc_Y += 1

			gradient_startPos = mathutils.Vector((0, -2.8, 1.1)) #top left

			myDupeGradient.location = gradient_startPos + mathutils.Vector((0, output_Y * locationMultiplierY, raiseLowerZ + (locationMultiplierZ * usable_Z_Row)))

			myDupeGradient.rotation_euler = mathutils.Vector((0, math.radians(90), 0))

			Ci = mathutils.Vector((i))

			bpy.context.view_layer.objects.active = myDupeGradient
			mat1 = self.newShader("ShaderVisualizer_gradient_" + str(i), "emission", Ci.x, Ci.y, Ci.z)
			bpy.context.active_object.data.materials.clear()
			bpy.context.active_object.data.materials.append(mat1)

			#####################
			### text_add() (better text placement)
			#####################
			if precisionVal != -1:
				val_gamma_correct_gradient_color_prop = bpy.context.scene.gamma_correct_gradient_color_prop

				t = None

				if val_gamma_correct_gradient_color_prop == True:
					gammaCorrect = mathutils.Vector((1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2))
					# gammaCorrect = mathutils.Vector((2.2, 2.2, 2.2))
					gammaCorrect_r = pow(Ci.x, gammaCorrect.x)
					gammaCorrect_g = pow(Ci.y, gammaCorrect.y)
					gammaCorrect_b = pow(Ci.z, gammaCorrect.z)

					Ci_gc_text = mathutils.Vector((gammaCorrect_r, gammaCorrect_g, gammaCorrect_b))

					t = '(' + str(round(Ci_gc_text.x, precisionVal)) + ', ' + str(round(Ci_gc_text.y, precisionVal)) + ', ' + str(round(Ci_gc_text.z, precisionVal)) + ')'

				elif val_gamma_correct_gradient_color_prop == False:
					t = '(' + str(round(Ci.x, precisionVal)) + ', ' + str(round(Ci.y, precisionVal)) + ', ' + str(round(Ci.z, precisionVal)) + ')'

				myFontCurve = bpy.data.curves.new(type="FONT", name="myFontCurve")
				myFontCurve.body = t

				myFontOb = bpy.data.objects.new(myDupeGradient.name + '_text', myFontCurve)
				myFontOb.data.align_x = 'CENTER'
				myFontOb.data.align_y = 'CENTER'

				# textRaiseLower = 0.15
				# textRaiseLower = 0.175
				# textRaiseLower = 0.2
				# textRaiseLower = 0.12
				textRaiseLower = 0.23

				if idx % 2 == 0:
					#even
					# textRaiseLower *= 1
					# textRaiseLower *= 1
					pass
				else:
					#odd
					# textRaiseLower *= -1
					textRaiseLower += .06

				textRaiseLower *= -1

				myFontOb.location = myDupeGradient.location + mathutils.Vector((1, 0, textRaiseLower))
				myFontOb.rotation_euler = myRotation

				'''
				bpy.context.view_layer.objects.active = myDupeGradient
				me = bpy.context.active_object.data

				bm = bmesh.new()   # create an empty BMesh
				bm.from_mesh(me)   # fill it in from a Mesh

				outputSize = 0
				for f in bm.faces:
					# normalDir = f.normal
					# f = f 
					outputSize = f.calc_area()
					outputSize = (mathutils.Vector((outputSize, outputSize, outputSize)) * self.myV).x

				val_text_radius_0_prop = bpy.context.scene.text_radius_0_prop
				val_text_radius_1_prop = bpy.context.scene.text_radius_1_prop

				outputTextSize_usable = self.lerp(val_text_radius_0_prop, val_text_radius_1_prop, outputSize)
				'''

				# myFontScale = 0.075
				myFontScale = 0.06
				myFontOb.scale = mathutils.Vector((myFontScale, myFontScale, myFontScale))

				myFontOb.data.body = t

				bpy.context.collection.objects.link(myFontOb)

				myFontOb.show_in_front = True

				self.textRef_all.append(myFontOb.name)

				bpy.context.view_layer.objects.active = myFontOb

				mat1 = self.newShader("ShaderVisualizer_gradient_text_" + str(i), "emission", 0, 0, 0)
				# mat1 = self.newShader("ShaderVisualizer_gradient_text_" + str(i), "emission", 1, 1, 1)
				bpy.context.active_object.data.materials.clear()
				bpy.context.active_object.data.materials.append(mat1)

				bpy.context.view_layer.objects.active = myDupeGradient

		myInputMesh.hide_render = True


	def makeGradientGrid_color(self, gradientArray):

		self.deselectAll()
		self.deleteAllObjects()
		self.mega_purge()
		
		self.textRef_all.clear()

		bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)
		bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')

		###########
		#DEFAULT CAMERA
		#############
		cam1_data = bpy.data.cameras.new('Camera')
		cam = bpy.data.objects.new('Camera', cam1_data)
		bpy.context.collection.objects.link(cam)

		###################################
		###### SET CAMERA POS / LOOK AT
		###################################
		self.myCam = bpy.data.objects["Camera"]

		# self.myCam.location = self.pos_camera_global
		self.myCam.location = mathutils.Vector((20, 0, 0))


		# bpy.context.object.data.type = 'ORTHO'
		self.myCam.data.type = 'ORTHO'

		#near 8.5x11 printable ratio
		# bpy.context.scene.render.resolution_x = 3900
		# bpy.context.scene.render.resolution_y = 3000

		bpy.context.scene.render.resolution_x = 2550
		bpy.context.scene.render.resolution_y = 1970

		self.updateScene() # need

		self.look_at(self.myCam, self.myOrigin)

		self.myV = self.myCam.matrix_world.to_translation()
		self.myV.normalize()

		usablePrimitiveType_gradient_id = 'grid'

		if usablePrimitiveType_gradient_id == 'grid':
			bpy.ops.mesh.primitive_grid_add()

		myInputMesh = bpy.context.active_object
		myInputMesh.select_set(1)
		myInputMesh.hide_set(1)
		# myInputMesh.hide_render = True

		#####################
		### grey background
		#####################
		bpy.context.view_layer.objects.active = myInputMesh
		myDupeGradient_bg = self.copyObject()
		myDupeGradient_bg.name = 'dupeGradient_background'
		myDupeGradient_bg.scale = mathutils.Vector((5, 5, 5))
		myDupeGradient_bg.location = mathutils.Vector((-1, 0, 0))
		myDupeGradient_bg.rotation_euler = mathutils.Vector((0, math.radians(90), 0))

		bpy.context.view_layer.objects.active = myDupeGradient_bg

		gamma_correct_gradient_color_prop = bpy.context.scene.gamma_correct_gradient_color_prop
		greyBG = 0.5
		if gamma_correct_gradient_color_prop == True:
			# lerpIter = pow(lerpIter, 1.0 / 2.2)
			greyBG = pow(greyBG, 2.2)

		mat1 = self.newShader("ShaderVisualizer_gradientBG", "emission", greyBG, greyBG, greyBG)

		# mat1 = self.newShader("ShaderVisualizer_gradientBG", "emission", 0.5, 0.5, 0.5)
		bpy.context.active_object.data.materials.clear()
		bpy.context.active_object.data.materials.append(mat1)

		#####################
		### each gradient face
		#####################
		val_gradient_method0_rowRange_prop = bpy.context.scene.gradient_method0_rowRange_prop
		val_gradient_method0_size_prop = bpy.context.scene.gradient_method0_size_prop
		val_gradient_method0_spacing_prop = bpy.context.scene.gradient_method0_spacing_prop
		val_gradient_method0_height_prop = bpy.context.scene.gradient_method0_height_prop

		gradientScale = val_gradient_method0_size_prop
		rangeLength = val_gradient_method0_rowRange_prop

		# locationMultiplierY = .5
		# locationMultiplierY = .35
		# locationMultiplierY = .4
		locationMultiplierY = val_gradient_method0_spacing_prop

		# locationMultiplierY = .2
		# locationMultiplierZ = -.5
		# locationMultiplierZ = -.75
		# locationMultiplierZ = -.6
		locationMultiplierZ = val_gradient_method0_height_prop
		raiseLowerZ = 1
		usable_Z_Row = 0

		usableCurrLoc_Y = 0
		usableCurrRow_Z = 0
		startIdx = 0

		usableTextRGBPrecision_items = bpy.context.scene.bl_rna.properties['text_rgb_precision_enum_prop'].enum_items
		usableTextRGBPrecision_id = usableTextRGBPrecision_items[bpy.context.scene.text_rgb_precision_enum_prop].identifier

		precisionVal = int(usableTextRGBPrecision_id)

		if precisionVal == -1:
			precisionVal = 3

		# precisionVal = 1 # DEBUG
		precisionVal = 2 # DEBUG
		# precisionVal = 3 # DEBUG

		val_text_gradient_rotate_x_prop = bpy.context.scene.text_gradient_rotate_x_prop
		val_text_gradient_rotate_y_prop = bpy.context.scene.text_gradient_rotate_y_prop
		val_text_gradient_rotate_z_prop = bpy.context.scene.text_gradient_rotate_z_prop

		myRotation = mathutils.Vector((math.radians(90), math.radians(val_text_gradient_rotate_y_prop), math.radians(90)))

		for idx, i in enumerate(gradientArray):
			bpy.context.view_layer.objects.active = myInputMesh
			myDupeGradient = self.copyObject()
			myDupeGradient.name = 'dupeGradient_' + str(idx)
			myDupeGradient.scale = mathutils.Vector((gradientScale, gradientScale, gradientScale))

			myRange = range(startIdx, startIdx + rangeLength)
			usable_Z_Row = None

			if idx > myRange[-1]:
				startIdx = startIdx + rangeLength
				myRange = range(startIdx, startIdx + rangeLength)
				usableCurrLoc_Y = 0
				usableCurrRow_Z += 1
			
			if myRange[0] <= idx <= myRange[-1]:
				usable_Z_Row = usableCurrRow_Z
				output_Y = usableCurrLoc_Y # * yMult # * idx)
				usableCurrLoc_Y += 1

			gradient_startPos = mathutils.Vector((0, -2.8, 1.1)) #top left

			myDupeGradient.location = gradient_startPos + mathutils.Vector((0, output_Y * locationMultiplierY, raiseLowerZ + (locationMultiplierZ * usable_Z_Row)))

			myDupeGradient.rotation_euler = mathutils.Vector((0, math.radians(90), 0))

			Ci = mathutils.Vector((i))

			bpy.context.view_layer.objects.active = myDupeGradient
			mat1 = self.newShader("ShaderVisualizer_gradient_" + str(i), "emission", Ci.x, Ci.y, Ci.z)
			bpy.context.active_object.data.materials.clear()
			bpy.context.active_object.data.materials.append(mat1)

			#####################
			### text_add() (better text placement)
			#####################
			if precisionVal != -1:
				val_gamma_correct_gradient_color_prop = bpy.context.scene.gamma_correct_gradient_color_prop

				t = ''

				'''
				

				if val_gamma_correct_gradient_color_prop == True:
					gammaCorrect = mathutils.Vector((1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2))
					# gammaCorrect = mathutils.Vector((2.2, 2.2, 2.2))
					gammaCorrect_r = pow(Ci.x, gammaCorrect.x)
					gammaCorrect_g = pow(Ci.y, gammaCorrect.y)
					gammaCorrect_b = pow(Ci.z, gammaCorrect.z)

					Ci_gc_text = mathutils.Vector((gammaCorrect_r, gammaCorrect_g, gammaCorrect_b))

					t = '(' + str(round(Ci_gc_text.x, precisionVal)) + ', ' + str(round(Ci_gc_text.y, precisionVal)) + ', ' + str(round(Ci_gc_text.z, precisionVal)) + ')'

				elif val_gamma_correct_gradient_color_prop == False:
					t = '(' + str(round(Ci.x, precisionVal)) + ', ' + str(round(Ci.y, precisionVal)) + ', ' + str(round(Ci.z, precisionVal)) + ')'

				'''

				if (idx + 1) % 5 == 0:
					t = 'X'
					# myFontScale = 0.06 #####

				myFontCurve = bpy.data.curves.new(type="FONT", name="myFontCurve")
				myFontCurve.body = t

				myFontOb = bpy.data.objects.new(myDupeGradient.name + '_text', myFontCurve)
				myFontOb.data.align_x = 'CENTER'
				myFontOb.data.align_y = 'CENTER'

				# textRaiseLower = 0.15
				# textRaiseLower = 0.175
				# textRaiseLower = 0.2
				# textRaiseLower = 0.12
				# textRaiseLower = 0.23
				textRaiseLower = 0

				# if idx % 2 == 0:
				# 	#even
				# 	# textRaiseLower *= 1
				# 	# textRaiseLower *= 1
				# 	pass
				# else:
				# 	#odd
				# 	# textRaiseLower *= -1
				# 	textRaiseLower += .06






				textRaiseLower *= -1

				myFontOb.location = myDupeGradient.location + mathutils.Vector((1, 0, textRaiseLower))
				myFontOb.rotation_euler = myRotation

				'''
				bpy.context.view_layer.objects.active = myDupeGradient
				me = bpy.context.active_object.data

				bm = bmesh.new()   # create an empty BMesh
				bm.from_mesh(me)   # fill it in from a Mesh

				outputSize = 0
				for f in bm.faces:
					# normalDir = f.normal
					# f = f 
					outputSize = f.calc_area()
					outputSize = (mathutils.Vector((outputSize, outputSize, outputSize)) * self.myV).x

				val_text_radius_0_prop = bpy.context.scene.text_radius_0_prop
				val_text_radius_1_prop = bpy.context.scene.text_radius_1_prop

				outputTextSize_usable = self.lerp(val_text_radius_0_prop, val_text_radius_1_prop, outputSize)
				'''

				# myFontScale = 0.075
				myFontScale = 0.06
				myFontOb.scale = mathutils.Vector((myFontScale, myFontScale, myFontScale))

				myFontOb.data.body = t

				bpy.context.collection.objects.link(myFontOb)

				myFontOb.show_in_front = True

				self.textRef_all.append(myFontOb.name)

				bpy.context.view_layer.objects.active = myFontOb

				mat1 = self.newShader("ShaderVisualizer_gradient_text_" + str(i), "emission", 0, 0, 0)
				# mat1 = self.newShader("ShaderVisualizer_gradient_text_" + str(i), "emission", 1, 1, 1)
				bpy.context.active_object.data.materials.clear()
				bpy.context.active_object.data.materials.append(mat1)

				bpy.context.view_layer.objects.active = myDupeGradient

		myInputMesh.hide_render = True

	def step(self, edge, x):
		myOutput = None

		if x < edge:
			myOutput = 0.0

		else:
			myOutput = 1.0

		return myOutput

	def mix(self, x, y, a):
		return x * (1.0 - a) + y * a

	def lerp(self, a, b, t):
		return a + (b - a) * t

	def inverse_lerp(self, a, b, value):
		if a == b:
			return 0.0 if value <= a else 1.0
		return (value - a) / (b - a)

	def oren(self, NdotL, V, L, N, NdotV, roughness):
		return NdotL
	
	def final_Ci_output(self, aov_id, shadingPlane, mySplitFaceIndexUsable, inputDiffuse, inputDiffuseL2, spec, distanceL1, distanceL2, attenuation, attenuationL2, faceCenter_to_V_rayCast, faceCenter_to_L_rayCast, faceCenter):
		attenuation = 1.0 #temporary, outside sunlight

		Ks = 1
		diff_Cs_V = mathutils.Vector((1.0, 0.0, 0.0))
		diff_Cs_V2 = mathutils.Vector((0.0, 1.0, 0.0))

		inputDiff_V = mathutils.Vector((inputDiffuse, inputDiffuse, inputDiffuse))
		inputDiff_V2 = mathutils.Vector((inputDiffuseL2, inputDiffuseL2, inputDiffuseL2))

		finalSpec = spec * Ks
		finalSpec_V = mathutils.Vector((finalSpec, finalSpec, finalSpec))
		finalDiff_V = diff_Cs_V * inputDiff_V
		finalDiff_V2 = diff_Cs_V2 * inputDiff_V2

		ambientMultiplier = .0004
		ambient_V = mathutils.Vector((ambientMultiplier, ambientMultiplier, ambientMultiplier))

		Ci = None

		if aov_id == 'spec':
			Ci = ((finalSpec_V) * attenuation) ###
		elif aov_id == 'diffuse':
			Ci = ((finalDiff_V) * attenuation) ###
		elif aov_id == 'Ci':

			# val_is_metallic_prop = bpy.context.scene.is_metallic_prop
			# if val_is_metallic_prop == True:
			# 	Ci = ((finalDiff_V * finalSpec_V) * attenuation) ###

			# else:
			# 	Ci = ((finalDiff_V + finalSpec_V) * attenuation) ###
			# # Ci = ((finalDiff_V + finalSpec_V + (diff_Cs_V * ambient_V)) * attenuation) ###

			val_is_metallic_prop = bpy.context.scene.is_metallic_prop
			if val_is_metallic_prop == True:
				finalFromL1 = ((finalDiff_V * finalSpec_V) * attenuation) ###

			else:
				finalFromL1 = (finalDiff_V + finalSpec_V) * attenuation
				if val_is_metallic_prop == True:
					finalFromL1 = ((finalDiff_V * finalSpec_V) * attenuation) ###

			finalFromL2 = (finalDiff_V2)

			maxV10 = max(0, finalFromL1.x)
			maxV11 = max(0, finalFromL1.y)
			maxV12 = max(0, finalFromL1.z)

			maxV20 = max(0, finalFromL2.x)
			maxV21 = max(0, finalFromL2.y)
			maxV22 = max(0, finalFromL2.z)

			Ci = mathutils.Vector((maxV10, maxV11, maxV12)) + mathutils.Vector((maxV20, maxV21, maxV22)) ###

		elif aov_id == 'saturation_based_on_distance':

			val_is_metallic_prop = bpy.context.scene.is_metallic_prop
			# if val_is_metallic_prop == True:
			# 	finalFromL1 = ((finalDiff_V * finalSpec_V) * attenuation) ###

			# else:
				
			finalFromL1 = (finalDiff_V + finalSpec_V) * attenuation
			if val_is_metallic_prop == True:
				finalFromL1 = ((finalDiff_V * finalSpec_V) * attenuation) ###


			finalFromL2 = (finalDiff_V2)

			# maxV10 = max(0, finalFromL1.x)
			# maxV11 = max(0, finalFromL1.y)
			# maxV12 = max(0, finalFromL1.z)

			# maxV20 = max(0, finalFromL2.x)
			# maxV21 = max(0, finalFromL2.y)
			# maxV22 = max(0, finalFromL2.z)

			maxV10_clamped = self.clamp(finalFromL1.x, 0, 1)
			maxV11_clamped = self.clamp(finalFromL1.y, 0, 1)
			maxV12_clamped = self.clamp(finalFromL1.z, 0, 1)

			maxV20_clamped = self.clamp(finalFromL2.x, 0, 1)
			maxV21_clamped = self.clamp(finalFromL2.y, 0, 1)
			maxV22_clamped = self.clamp(finalFromL2.z, 0, 1)

			maxV10 = maxV10_clamped
			maxV11 = maxV11_clamped
			maxV12 = maxV12_clamped

			maxV20 = maxV20_clamped
			maxV21 = maxV21_clamped
			maxV22 = maxV22_clamped

			Ci = mathutils.Vector((maxV10, maxV11, maxV12)) + mathutils.Vector((maxV20, maxV21, maxV22)) ###
			# Ci = mathutils.Vector((maxV20, maxV21, maxV22)) ###

			Ci_stored = Ci

			distance_from_cam = (self.myV - faceCenter).length








			usableTextRGBPrecision_items = bpy.context.scene.bl_rna.properties['text_rgb_precision_enum_prop'].enum_items
			usableTextRGBPrecision_id = usableTextRGBPrecision_items[bpy.context.scene.text_rgb_precision_enum_prop].identifier

			precisionVal = int(usableTextRGBPrecision_id)

			Ci = mathutils.Vector((maxV10, maxV11, maxV12)) + mathutils.Vector((maxV20, maxV21, maxV22)) ###
			Ci_gc = Ci

			if precisionVal == -1:
				pass

			else:
				Ci_gc = mathutils.Vector((round(Ci_gc.x, precisionVal), round(Ci_gc.y, precisionVal), round(Ci_gc.z, precisionVal)))

				
			precisionValUsable = 5

			Ci_gc_rounded = mathutils.Vector((round(Ci_gc.x, precisionValUsable), round(Ci_gc.y, precisionValUsable), round(Ci_gc.z, precisionValUsable) ))

			min_distance = float('inf')
			closest_v = None
			closest_value = None
			closest_identifier = None
			closest_idx = None

			spectralMix = None

			comboColor = (mathutils.Vector((maxV10, maxV11, maxV12)) + mathutils.Vector((maxV20, maxV21, maxV22)))
			comboColor_clamp0 = self.clamp(comboColor[0], 0, 1)
			comboColor_clamp1 = self.clamp(comboColor[1], 0, 1)
			comboColor_clamp2 = self.clamp(comboColor[2], 0, 1)

			comboColor = mathutils.Vector((comboColor_clamp0, comboColor_clamp1, comboColor_clamp2))

			val_use_18_hue_colorspace_prop = bpy.context.scene.use_18_hue_colorspace_prop

			if val_use_18_hue_colorspace_prop == True:
				
				for v in self.colorspace_18_hue_list:
					distance = (comboColor - v['value']).length
					if distance < min_distance:
						min_distance = distance
						closest_value = v['value']
						closest_identifier = v['identifier']
						closest_idx = v['idx']

				# Ci_gc = mathutils.Vector((closest_value.x, closest_value.y, closest_value.z))
				# comboColor = mathutils.Vector((closest_value.r, closest_value.g, closest_value.b))
				comboColor = closest_value




			else:
				# greyscaleConversionVec = mathutils.Vector((.299, .587, .114))
				# greyscaleConversionVec = .299 * comboColor.x, .587 * comboColor.y, .114 * comboColor.z
				greyscaleConversionVec = (.299 * comboColor.x) + (.587 * comboColor.y) + (.114 * comboColor.z)
				# greyscaleConversionVec = mathutils.Vector((greyscaleConversionVec, greyscaleConversionVec, greyscaleConversionVec))

				closest_idx = 19 - ((1 - (greyscaleConversionVec * 10)) * 19)
				# closest_idx = 19 - closest_idx

				# comboColor = mathutils.Vector((0, 1, 0))
			













			distance_from_cam = (self.myCam.location - faceCenter).length

			oldMin = self.distanceFromCam_all_list[-1]
			oldMax = self.distanceFromCam_all_list[0]

			oldMin = self.distanceFromCam_raycastRenderable_list[-1]
			oldMax = self.distanceFromCam_raycastRenderable_list[0]
			# myRemap = 1 - self.remap_range(distance_from_cam, oldMin, oldMax, 0, 1) #!!!!!!!!!!!!!!!!!!!!
			myRemap = self.remap_range(distance_from_cam, oldMin, oldMax, 0, 1)

			# gammaCorrect = mathutils.Vector((1, 1, 1))
			gammaCorrect = mathutils.Vector((2.2, 2.2, 2.2))

			gammaCorrect_r = pow(myRemap, gammaCorrect.x)
			gammaCorrect_g = pow(myRemap, gammaCorrect.y)
			gammaCorrect_b = pow(myRemap, gammaCorrect.z)

			myRemap_gc = mathutils.Vector((gammaCorrect_r, gammaCorrect_g, gammaCorrect_b))

			realGreyscale = myRemap_gc

			closest_idx_usable = 19 - closest_idx
			closestIdxIn19 = float((closest_idx_usable * realGreyscale[0]) / 19)

			greyScaleColor = mathutils.Vector((closestIdxIn19, closestIdxIn19, closestIdxIn19))

			val_spectral_mix_multiplier_prop = bpy.context.scene.spectral_mix_multiplier_prop
			val_spectral_mix_multiplier2_prop = bpy.context.scene.spectral_mix_multiplier2_prop
			myTestAdjustFromUI_v = mathutils.Vector((val_spectral_mix_multiplier_prop, val_spectral_mix_multiplier_prop, val_spectral_mix_multiplier_prop))
		
			exactGrey = mathutils.Vector((0.5, 0.5, 0.5))
			testColor = mathutils.Vector((0.5, 0.3, 0.1))


			# spectralMix = spectral3_glsl.spectral_mix2(comboColor, 1, 1, exactGrey, greyScaleColor[0], 1) #********** ok
			# spectralMix = spectral3_glsl.spectral_mix2(comboColor, 1, 1, greyScaleColor, greyScaleColor[0], 1) #********** ok
			# spectralMix = spectral3_glsl.spectral_mix2(comboColor, 1, 1, exactGrey, .1 * greyScaleColor[0], 1) #********** good 3x
			# spectralMix = spectral3_glsl.spectral_mix2(comboColor, 1, 1, exactGrey, 3 * greyScaleColor[0], 1) #********** good 3x
			# spectralMix = spectral3_glsl.spectral_mix2(comboColor, 1, 1, exactGrey, 10 * greyScaleColor[0], 1) #********** good 10x
			# spectralMix = spectral3_glsl.spectral_mix2(comboColor, 1, 1, exactGrey, 2, 1) #********** good 10x
			# spectralMix = spectral3_glsl.spectral_mix2(comboColor, 1, 1, exactGrey, 6 * greyScaleColor[0], 1) #********** good 10x
			# spectralMix = spectral3_glsl.spectral_mix2(comboColor, 1, 1, exactGrey, val_spectral_mix_multiplier_prop * greyScaleColor[0], 1) #********** good 10x
			# spectralMix = spectral3_glsl.spectral_mix2(comboColor, 1, 1, exactGrey, val_spectral_mix_multiplier_prop * greyScaleColor[0], 1) #********** good 10x
			# spectralMix = spectral3_glsl.spectral_mix2(comboColor, 1, 1, exactGrey, val_spectral_mix_multiplier_prop + greyScaleColor[0], 1) #********** good 10x
			# spectralMix = spectral3_glsl.spectral_mix2(comboColor, 1, 1, exactGrey, val_spectral_mix_multiplier_prop + (10 * greyScaleColor[0]), 1) #********** good 10x
			# spectralMix = spectral3_glsl.spectral_mix2(comboColor, 1, 1, exactGrey, val_spectral_mix_multiplier_prop + (val_spectral_mix_multiplier2_prop + greyScaleColor[0]), 1) #********** good 10x
			# spectralMix = spectral3_glsl.spectral_mix2(comboColor, 1, 1, exactGrey, val_spectral_mix_multiplier_prop + (val_spectral_mix_multiplier2_prop * greyScaleColor[0]), 1) #********** good 10x



			# spectralMix = spectral3_glsl.spectral_mix2(comboColor, val_spectral_mix_multiplier_prop * (val_spectral_mix_multiplier2_prop + greyScaleColor[0]), 1, exactGrey, 1, 1) #********** good 10x
			# spectralMix = spectral3_glsl.spectral_mix2(comboColor, val_spectral_mix_multiplier_prop * greyScaleColor[0], 1, exactGrey, 1, 1) #********** good 10x
			# spectralMix = spectral3_glsl.spectral_mix2(comboColor, val_spectral_mix_multiplier_prop * greyScaleColor[0], 1, exactGrey, 1, 1) #********** good 10x

			# spectralMix = comboColor * (val_spectral_mix_multiplier_prop + (val_spectral_mix_multiplier2_prop * greyScaleColor[0]))

			# spectralMix = spectral3_glsl.spectral_mix2(myTestAdjustFromUI_v, 1, 1, exactGrey, val_spectral_mix_multiplier_prop + greyScaleColor[0], 1) #********** good 10x
			# spectralMix = spectral3_glsl.spectral_mix2(comboColor, 1, 1, exactGrey, val_spectral_mix_multiplier_prop * greyScaleColor[0], 1) #********** good 10x
			# spectralMix = myTestAdjustFromUI_v
			# spectralMix = comboColor
			
			# spectralMix = comboColor
			# self.aspect = val_spectral_mix_multiplier_prop

			# spectralMix = greyScaleColor * comboColor ### check vs this
			
			# spectralMix = closestValueV ###########
			# spectralMix = greyScaleColor ############
			# spectralMix = realGreyscale ###########
			# spectralMix = comboColor
			# spectralMix = realGreyscale * comboColor

			#############
			######## GOOD, ADJUSTABLE
			#############
			spectralMix = spectral3_glsl.spectral_mix2(comboColor, val_spectral_mix_multiplier_prop + (val_spectral_mix_multiplier2_prop * greyScaleColor[0]), 1, exactGrey, 1, 1) #********** use ####### 1 / 100

			val_gamma_correct_gradient_color_prop = True
			val_gamma_correct_gradient_color_prop = False

			if val_gamma_correct_gradient_color_prop == True:
				gammaCorrect = mathutils.Vector((2.2, 2.2, 2.2))
				gammaCorrect_r = pow(spectralMix.x, gammaCorrect.x)
				gammaCorrect_g = pow(spectralMix.y, gammaCorrect.y)
				gammaCorrect_b = pow(spectralMix.z, gammaCorrect.z)

				spectralMix = mathutils.Vector((gammaCorrect_r, gammaCorrect_g, gammaCorrect_b))

			Ci = spectralMix

		Ci_gc = Ci

		val_text_rotate_x_prop = bpy.context.scene.text_rotate_x_prop
		val_text_rotate_y_prop = bpy.context.scene.text_rotate_y_prop
		val_text_rotate_z_prop = bpy.context.scene.text_rotate_z_prop

		myRotation = self.myV * mathutils.Vector((math.radians(val_text_rotate_x_prop), math.radians(val_text_rotate_y_prop), math.radians(val_text_rotate_z_prop)))

		
		usableTextRGBPrecision_items = bpy.context.scene.bl_rna.properties['text_rgb_precision_enum_prop'].enum_items
		usableTextRGBPrecision_id = usableTextRGBPrecision_items[bpy.context.scene.text_rgb_precision_enum_prop].identifier

		precisionVal = int(usableTextRGBPrecision_id)

		for j in bpy.context.scene.objects:
			if j.name == shadingPlane:
				bpy.context.view_layer.objects.active = j

				#####################
				### text_add() (better text placement)
				#####################
				# '''
				val_use_18_hue_colorspace_prop = bpy.context.scene.use_18_hue_colorspace_prop

				# if precisionVal != -1:
				if val_use_18_hue_colorspace_prop == True:
					# precisionVal = 3

					min_distance = float('inf')
					closest_v = None
					closest_value = None
					closest_identifier = None
					closest_idx = None

					for v in self.colorspace_18_hue_list:
						distance = (Ci_gc - v['value']).length
						if distance < min_distance:
							min_distance = distance
							closest_value = v['value']
							closest_identifier = v['identifier']
							closest_idx = v['idx']

					# Ci_gc = mathutils.Vector((closest_value.x, closest_value.y, closest_value.z))
					usableCi = mathutils.Vector((closest_value.x, closest_value.y, closest_value.z))

					# t = '(' + str(round(Ci_gc.x, precisionVal)) + ', ' + str(round(Ci_gc.y, precisionVal)) + ', ' + str(round(Ci_gc.z, precisionVal)) + ')'
					# t = '(' + str(round(Ci_gc.x, precisionVal)) + ', ' + str(round(Ci_gc.y, precisionVal)) + ', ' + str(round(Ci_gc.z, precisionVal)) + ')'
					# t = '(' + str(round(usableCi.x, precisionVal)) + ', ' + str(round(usableCi.y, precisionVal)) + ', ' + str(round(usableCi.z, precisionVal)) + ')'
					t = closest_identifier + str(closest_idx)

					myFontCurve = bpy.data.curves.new(type="FONT", name="myFontCurve")
					myFontCurve.body = t

					myFontOb = bpy.data.objects.new(j.name + '_text', myFontCurve)
					myFontOb.data.align_x = 'CENTER'
					myFontOb.data.align_y = 'CENTER'

					myFontOb.location = j.location
					myFontOb.rotation_euler = myRotation

					bpy.context.view_layer.objects.active = j
					me = bpy.context.active_object.data

					bm = bmesh.new()   # create an empty BMesh
					bm.from_mesh(me)   # fill it in from a Mesh

					outputSize = 0
					for f in bm.faces:
						# normalDir = f.normal
						# f = f 
						outputSize = f.calc_area()
						outputSize = (mathutils.Vector((outputSize, outputSize, outputSize)) * self.myV).x

					val_text_radius_0_prop = bpy.context.scene.text_radius_0_prop
					val_text_radius_1_prop = bpy.context.scene.text_radius_1_prop

					outputTextSize_usable = self.lerp(val_text_radius_0_prop, val_text_radius_1_prop, outputSize)

					myFontOb.scale = mathutils.Vector((outputTextSize_usable, outputTextSize_usable, outputTextSize_usable))

					if Ci >= mathutils.Vector((0.05, 0.05, 0.05)):
						myFontOb.data.body = t

					else:
						myFontOb.data.body = 'x'

					bpy.context.collection.objects.link(myFontOb)

					if faceCenter_to_V_rayCast == True:
						myFontOb.show_in_front = True

					self.textRef_all.append(myFontOb.name)

				# '''

		mat1 = self.newShader("ShaderVisualizer_" + str(mySplitFaceIndexUsable), "emission", Ci_gc.x, Ci_gc.y, Ci_gc.z)
		# mat1 = self.newShader("ShaderVisualizer_" + str(mySplitFaceIndexUsable), "emission", .1, .1, .1)
		# mat1 = self.newShader("ShaderVisualizer_" + str(mySplitFaceIndexUsable), "emission", 0, 1, 0)
		bpy.context.active_object.data.materials.clear()
		bpy.context.active_object.data.materials.append(mat1)

	def renderPasses(self):
		###########
		#TO USE - THE SCENE MUST BE SAVED AND A FOLDER CALLED 'compositing_files' must be created in the save file directory
		###########

		#simple spec 7 steps
		simple_renderViewList_0 = ['N', 'L', 'NdotL']
		simple_renderViewList_1 = ['V', 'R', 'RdotV']
		simple_renderViewList_2 = ['RdotV + NdotL']

		#GGX 14 steps
		GGX_renderViewList_0 = ['N', 'L', 'NdotL']
		GGX_renderViewList_1 = ['V', 'H', 'HdotN', 'HdotL']
		GGX_renderViewList_1 = ['NdotV', 'ruff']
		GGX_renderViewList_1 = ['D', 'F', 'vis', 'GGX', 'GGX + NdotL']

		self.DoIt_part1_preprocess()
		
		#####################
		### SAVE RENDERS
		#####################
		mypath_N_persp = "//compositing_files/N_persp.png"
		mypath_N_view2 = "//compositing_files/N_view2.png"
		mypath_N_view3 = "//compositing_files/N_view3.png"

		mypath_L_persp = "//compositing_files/L_persp.png"
		mypath_L_view2 = "//compositing_files/L_view2.png"
		mypath_L_view3 = "//compositing_files/L_view3.png"

		mypath_NdotL_persp = "//compositing_files/NdotL_persp.png"
		mypath_NdotL_view2 = "//compositing_files/NdotL_view2.png"
		mypath_NdotL_view3 = "//compositing_files/NdotL_view3.png"
		
		mypath_V_persp = "//compositing_files/V_persp.png"
		mypath_V_view2 = "//compositing_files/V_view2.png"
		mypath_V_view3 = "//compositing_files/V_view3.png"
		
		#simple
		mypath_simple_R_persp = "//compositing_files/simple_R_persp.png"
		mypath_simple_R_view2 = "//compositing_files/simple_R_view2.png"
		mypath_simple_R_view3 = "//compositing_files/simple_R_view3.png"

		mypath_simple_RdotV_persp = "//compositing_files/simple_RdotV_persp.png"
		mypath_simple_RdotV_view2 = "//compositing_files/simple_RdotV_view2.png"
		mypath_simple_RdotV_view3 = "//compositing_files/simple_RdotV_view3.png"
		
		mypath_simple_combo_diffuse_persp = "//compositing_files/simple_combo_diffuse_persp.png"
		mypath_simple_combo_diffuse_view2 = "//compositing_files/simple_combo_diffuse_view2.png"
		mypath_simple_combo_diffuse_view3 = "//compositing_files/simple_combo_diffuse_view3.png"

		#GGX
		mypath_GGX_H_persp = "//compositing_files/GGX_H_persp.png"
		mypath_GGX_H_view2 = "//compositing_files/GGX_H_view2.png"
		mypath_GGX_H_view3 = "//compositing_files/GGX_H_view3.png"
		
		mypath_GGX_HdotN_persp = "//compositing_files/GGX_HdotN_persp.png"
		mypath_GGX_HdotN_view2 = "//compositing_files/GGX_HdotN_view2.png"
		mypath_GGX_HdotN_view3 = "//compositing_files/GGX_HdotN_view3.png"
		
		mypath_GGX_HdotL_persp = "//compositing_files/GGX_HdotL_persp.png"
		mypath_GGX_HdotL_view2 = "//compositing_files/GGX_HdotL_view2.png"
		mypath_GGX_HdotL_view3 = "//compositing_files/GGX_HdotL_view3.png"
		
		mypath_GGX_NdotV_persp = "//compositing_files/GGX_NdotV_persp.png"
		mypath_GGX_NdotV_view2 = "//compositing_files/GGX_NdotV_view2.png"
		mypath_GGX_NdotV_view3 = "//compositing_files/GGX_NdotV_view3.png"
		
		mypath_GGX_ruff_persp = "//compositing_files/GGX_ruff_persp.png"
		mypath_GGX_ruff_view2 = "//compositing_files/GGX_ruff_view2.png"
		mypath_GGX_ruff_view3 = "//compositing_files/GGX_ruff_view3.png"

		mypath_GGX_D_persp = "//compositing_files/GGX_D_persp.png"
		mypath_GGX_D_view2 = "//compositing_files/GGX_D_view2.png"
		mypath_GGX_D_view3 = "//compositing_files/GGX_D_view3.png"
		
		mypath_GGX_F_persp = "//compositing_files/GGX_F_persp.png"
		mypath_GGX_F_view2 = "//compositing_files/GGX_F_view2.png"
		mypath_GGX_F_view3 = "//compositing_files/GGX_F_view3.png"
		
		mypath_GGX_Vis_persp = "//compositing_files/GGX_Vis_persp.png"
		mypath_GGX_Vis_view2 = "//compositing_files/GGX_Vis_view2.png"
		mypath_GGX_Vis_view3 = "//compositing_files/GGX_Vis_view3.png"
		
		mypath_GGX_GGX_persp = "//compositing_files/GGX_GGX_persp.png"
		mypath_GGX_GGX_view2 = "//compositing_files/GGX_GGX_view2.png"
		mypath_GGX_GGX_view3 = "//compositing_files/GGX_GGX_view3.png"
		
		mypath_GGX_combo_diffuse_persp = "//compositing_files/GGX_combo_diffuse_persp.png"
		mypath_GGX_combo_diffuse_view2 = "//compositing_files/GGX_combo_diffuse_view2.png"
		mypath_GGX_combo_diffuse_view3 = "//compositing_files/GGX_combo_diffuse_view3.png"

		all_render_paths = [ 
			
			mypath_N_persp, 
			mypath_N_view2, 
			mypath_N_view3, 
		
			mypath_L_persp, 
			mypath_L_view2, 
			mypath_L_view3, 
			
			mypath_NdotL_persp,
			mypath_NdotL_view2,
			mypath_NdotL_view3,
					  
			mypath_simple_R_persp,
			mypath_simple_R_view2,
			mypath_simple_R_view3,

			mypath_simple_RdotV_persp, 
			mypath_simple_RdotV_view2, 
			mypath_simple_RdotV_view3, 
			
			mypath_simple_combo_diffuse_persp, 
			mypath_simple_combo_diffuse_view2, 
			mypath_simple_combo_diffuse_view3, 

			mypath_GGX_H_persp,
			mypath_GGX_H_view2,
			mypath_GGX_H_view3,

			mypath_GGX_HdotN_persp, 
			mypath_GGX_HdotN_view2, 
			mypath_GGX_HdotN_view3, 
			
			mypath_GGX_HdotL_persp, 
			mypath_GGX_HdotL_view2, 
			mypath_GGX_HdotL_view3, 
			
			mypath_GGX_NdotV_persp, 
			mypath_GGX_NdotV_view2, 
			mypath_GGX_NdotV_view3, 
			
			mypath_GGX_ruff_persp, 
			mypath_GGX_ruff_view2, 
			mypath_GGX_ruff_view3, 
			
			mypath_GGX_D_persp, 
			mypath_GGX_D_view2, 
			mypath_GGX_D_view3, 
			
			mypath_GGX_F_persp, 
			mypath_GGX_F_view2, 
			mypath_GGX_F_view3, 
			
			mypath_GGX_Vis_persp, 
			mypath_GGX_Vis_view2, 
			mypath_GGX_Vis_view3, 
			
			mypath_GGX_GGX_persp, 
			mypath_GGX_GGX_view2, 
			mypath_GGX_GGX_view3, 
			
			mypath_GGX_combo_diffuse_persp,
			mypath_GGX_combo_diffuse_view2,
			mypath_GGX_combo_diffuse_view3,

			mypath_V_persp, 
			mypath_V_view2, 
			mypath_V_view3, 
		]

		singleArrow_render_paths = [ 
			mypath_N_persp, 
			mypath_N_view2, 
			mypath_N_view3, 

			mypath_L_persp, 
			mypath_L_view2, 
			mypath_L_view3, 
			
			mypath_simple_R_persp, 
			mypath_simple_R_view2, 
			mypath_simple_R_view3, 
			
			mypath_GGX_H_persp,
			mypath_GGX_H_view2,
			mypath_GGX_H_view3,

			mypath_V_persp, 
			mypath_V_view2, 
			mypath_V_view3, 
		]

		#####################
		### SIMPLE SPEC RENDERS
		#####################
		bpy.context.scene.camera = self.myCam

		# self.renderPasses_simple = True

		camLocStored = self.myCam.location

		#select all faces
		bpy.ops.object.mode_set(mode="OBJECT")
		bpy.ops.object.select_all(action='SELECT')
		self.stages_selectfaces_UI()

		self.deselectAll()

		#######
		#Normal
		#######
		for area in bpy.context.screen.areas: 
			if area.type == 'VIEW_3D':
				for space in area.spaces: 
					if space.type == 'VIEW_3D':
						# space.shading.type = 'WIREFRAME'
						space.shading.type = 'MATERIAL'
						# space.shading.type = 'SOLID'



		# for area in bpy.data.screens["Scripting"].areas:
		for area in bpy.data.screens["Layout"].areas:
		# for area in bpy.data.screens[bpy.context.window.screen].areas:
			if area.type == 'VIEW_3D':
				for space in area.spaces:
					if space.type == 'VIEW_3D':

						# space.overlay.grid_scale = 2

						usableToggle = None
						if space.overlay.show_floor == True:
							usableToggle = False

							space.overlay.show_floor = usableToggle
							space.overlay.show_axis_x = usableToggle
							space.overlay.show_axis_y = usableToggle
							space.overlay.show_axis_z = usableToggle
							space.overlay.show_cursor = usableToggle

		self.textColorSettings_UI()

		# bpy.context.scene.view_settings.view_transform = 'Standard'
		bpy.context.scene.view_settings.look = 'AgX - Punchy'
		bpy.context.scene.view_settings.look = 'None'
		bpy.context.scene.render.use_multiview = False


		self.renderPasses_simple = False
		self.renderPasses_GGX = False

		for idx, i in enumerate(singleArrow_render_paths):
			
			# bpy.context.scene.render.resolution_x = 2550

			# bpy.context.scene.render.resolution_x = 1000
			# bpy.context.scene.render.resolution_y = 1000

			bpy.context.scene.render.resolution_x = 3000
			bpy.context.scene.render.resolution_y = 3000

			# bpy.context.scene.render.resolution_x = 2550
			# bpy.context.scene.render.resolution_y = 1970

			sideCamLoc = mathutils.Vector((-8, 0, 0))
			topCamLoc = mathutils.Vector((0, 0, 8))


			self.renderPasses_simple = True

			if idx == 0: #persp cam
				self.myCam.location = self.pos_camera_global
				self.doIt_part2_render()

			elif idx == 3 or idx == 6 or idx == 9 or idx == 12:
				self.myCam.location = self.pos_camera_global
				self.updateScene() # need
				self.look_at(self.myCam, self.myOrigin)

				self.stageIdx_plusMinus_UI(1)

			# elif idx == 1: #side cam
			elif idx == 1 or idx == 4 or idx == 7 or idx == 10 or idx == 13: #side cam
				self.myCam.location = sideCamLoc
				self.updateScene() # need
				self.look_at(self.myCam, self.myOrigin)

			elif idx == 2 or idx == 5 or idx == 8 or idx == 11 or idx == 14: #top cam
				self.myCam.location = topCamLoc
				self.updateScene() # need
				self.look_at(self.myCam, self.myOrigin)



			if idx != 1 or idx != 2 or idx != 3:
				self.mySun.hide_set(1)
				# self.myCubeCam.hide_set(1)



			if idx == 0 or idx == 1 or idx == 2:
				#show normal only, hide light
				# self.mySun.hide_set(1)
				self.myCubeCam.hide_set(1)

			if idx == 3 or idx == 4 or idx == 5:
				#show light
				# self.mySun.hide_set(0)
				self.myCubeCam.hide_set(1)

			if idx == 6 or idx == 7 or idx == 8:
				#show view
				# self.mySun.hide_set(1)

				if idx != 6:
					self.myCubeCam.hide_set(0)

			if idx == 9 or idx == 10 or idx == 11:
				#show R
				# self.mySun.hide_set(0)
				self.myCubeCam.hide_set(1)

			if idx == 12 or idx == 13 or idx == 14:
				#show H
				# self.mySun.hide_set(0)
				# self.myCubeCam.hide_set(1)

				if idx == 12:
					self.myCubeCam.hide_set(1)
				else:
					self.myCubeCam.hide_set(0)

			bpy.context.scene.render.filepath = i

			for area in bpy.context.screen.areas:
				if area.type == 'VIEW_3D':
					area.spaces[0].region_3d.view_perspective = 'CAMERA'
					break

			bpy.context.scene.render.film_transparent = True
			bpy.ops.render.opengl(write_still=True)

			# if idx == 3:
			# if idx == 10:
				# return

		bpy.context.scene.render.film_transparent = False

		####################
		### FINAL COMPOSITE
		####################
		self.deselectAll()
		self.deleteAllObjects()
		self.mega_purge()

		bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)
		bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')

		###########
		# DEFAULT CAMERA
		#############
		cam1_data = bpy.data.cameras.new('Camera')
		cam = bpy.data.objects.new('Camera', cam1_data)
		bpy.context.collection.objects.link(cam)

		###################################
		###### SET CAMERA POS / LOOK AT
		###################################
		self.myCam = bpy.data.objects["Camera"]

		# self.myCam.location = self.pos_camera_global
		self.myCam.location = mathutils.Vector((20, 0, 0))
		self.myCam.data.type = 'ORTHO'

		bpy.context.scene.render.resolution_x = 2550
		# bpy.context.scene.render.resolution_y = 1970 #####
		bpy.context.scene.render.resolution_y = 3300

		self.updateScene() # need
		self.look_at(self.myCam, self.myOrigin)

		#####################
		### input mesh
		#####################
		usablePrimitiveType_gradient_id = 'grid'
		if usablePrimitiveType_gradient_id == 'grid':
			bpy.ops.mesh.primitive_grid_add()

		myInputMesh = bpy.context.active_object
		myInputMesh.select_set(1)
		myInputMesh.hide_set(1)
		# myInputMesh.hide_render = True

		#####################
		### background
		#####################
		bpy.context.view_layer.objects.active = myInputMesh
		myDupeGradient_bg = self.copyObject()
		myDupeGradient_bg.name = 'dupeGradient_background'
		myDupeGradient_bg.scale = mathutils.Vector((5, 5, 5))
		myDupeGradient_bg.location = mathutils.Vector((-1, 0, 0))
		myDupeGradient_bg.rotation_euler = mathutils.Vector((0, math.radians(90), 0))

		bpy.context.view_layer.objects.active = myDupeGradient_bg

		gamma_correct_gradient_colorWheel_prop = bpy.context.scene.gamma_correct_gradient_colorWheel_prop
		# greyBG = 0.5
		# if gamma_correct_gradient_colorWheel_prop == True:
		# 	# lerpIter = pow(lerpIter, 1.0 / 2.2)
		# 	greyBG = pow(greyBG, 2.2)

		# mat1 = self.newShader("ShaderVisualizer_gradientBG", "emission", greyBG, greyBG, greyBG)
		mat1 = self.newShader("ShaderVisualizer_gradientBG", "emission", 1, 1, 1)

		bpy.context.active_object.data.materials.clear()
		bpy.context.active_object.data.materials.append(mat1)

		#####################
		## LOCATION VARIABLES
		#####################

		'''
		tileScale = 10000
		rangeLength = 3
		# locationMultiplierY = .4
		locationMultiplierY = .5
		# locationMultiplierZ = -.6
		locationMultiplierZ = -.15
		# raiseLowerZ = 1
		raiseLowerZ = 0

		usableCurrLoc_X = 0
		usableCurrLoc_Y = 0
		usableCurrRow_Z = 0
		startIdx = 0
		'''













		'''

		tileScale = 3000


		rangeLength = 3
		# locationMultiplierY = .4
		locationMultiplierY = 1.5
		# locationMultiplierZ = -.6
		# locationMultiplierZ = -.15
		locationMultiplierZ = -1.3
		# raiseLowerZ = 1
		raiseLowerZ = 0

		usableCurrLoc_X = 0
		usableCurrLoc_Y = 0
		usableCurrRow_Z = 0
		startIdx = 0

		# tile_startPos = mathutils.Vector((2, -2.0, 2.75)) #top left
		# tile_startPos = mathutils.Vector((2, -2.0, 2.7)) #top left
		tile_startPos = mathutils.Vector((2, -1.7, 2.6)) #top left

		'''





		# tileScale = 5000
		# tileScale = 3000
		# tileScale = 2000
		# tileScale = 2250


		rangeLength = 3
		# locationMultiplierY = .4
		locationMultiplierY = 1.5
		# locationMultiplierZ = -.6
		# locationMultiplierZ = -.15
		# locationMultiplierZ = -1.3
		locationMultiplierZ = -1.2
		# raiseLowerZ = 1
		raiseLowerZ = 0

		usableCurrLoc_X = 0
		usableCurrLoc_Y = 0
		usableCurrRow_Z = 0
		startIdx = 0

		# tile_startPos = mathutils.Vector((2, -2.0, 2.75)) #top left
		# tile_startPos = mathutils.Vector((2, -2.0, 2.7)) #top left
		tile_startPos = mathutils.Vector((2, -1.7, 2.4)) #top left

		tileScale = 1.3

		for idx, i in enumerate(singleArrow_render_paths):
			bpy.context.view_layer.objects.active = myInputMesh
			myTile = self.copyObject()
			myTile_name_0 = i.split('.png')[0]
			myTile_name_1 = myTile_name_0.split('/')[-1]

			# print('myTile_name_1 = ', myTile_name_1)

			myTile.name = myTile_name_1

			# myTile.scale = mathutils.Vector((bpy.context.scene.render.resolution_x / tileScale, bpy.context.scene.render.resolution_y / tileScale, 1))

			# myTile.scale *= mathutils.Vector((.5, .5, 1))
			# myTile.scale *= mathutils.Vector((.75, .75, 1))
			# myTile.scale *= mathutils.Vector((1, 1, 1))
			myTile.scale *= mathutils.Vector((tileScale, tileScale, 1))

			##############
			## LOCATION
			##############
			myRange = range(startIdx, startIdx + rangeLength)
			usable_Z_Row = None

			if idx > myRange[-1]:
				startIdx = startIdx + rangeLength
				myRange = range(startIdx, startIdx + rangeLength)
				usableCurrLoc_Y = 0
				usableCurrRow_Z += 1
			
			if myRange[0] <= idx <= myRange[-1]:
				usable_Z_Row = usableCurrRow_Z
				output_Y = usableCurrLoc_Y # * yMult # * idx)
				usableCurrLoc_Y += 1

			if idx > 0:
				usableCurrLoc_X -= .01


			# myTile.location = mathutils.Vector((2, 0, .1))
			# myTile.location = mathutils.Vector((2, -2.5, 2)) ######

			# tile_startPos = mathutils.Vector((0, -2.8, 1.1)) #top left
			# tile_startPos = mathutils.Vector((2, -2.5, 2)) #top left
			# tile_startPos = mathutils.Vector((2, -2.0, 2.75)) #top left

			# myTile.location = tile_startPos + mathutils.Vector((0, output_Y * locationMultiplierY, raiseLowerZ + (locationMultiplierZ * usable_Z_Row)))
			myTile.location = tile_startPos + mathutils.Vector((usableCurrLoc_X, output_Y * locationMultiplierY, raiseLowerZ + (locationMultiplierZ * usable_Z_Row)))


			#########

			myTile.rotation_euler = mathutils.Vector((math.radians(-90), math.radians(180), math.radians(-90)))

			bpy.context.view_layer.objects.active = myTile
			# mat1 = self.newShader("ShaderVisualizer_renderPasses_mypath_N_persp", "emission", 1, 1, 1)
			mat1 = self.newShader('ShaderVisualizer_renderPasses_' + myTile_name_1, 'emission', 1, 1, 1)
			
			bpy.context.active_object.data.materials.clear()
			bpy.context.active_object.data.materials.append(mat1)

			nodes = mat1.node_tree.nodes
			links = mat1.node_tree.links

			myImageTexture00 = nodes.new("ShaderNodeTexImage")
			myImageTexture00.image = bpy.data.images.load(i)
			myImageTexture00.image.colorspace_settings.name = "sRGB"

			mySepColor = nodes.new("ShaderNodeSeparateColor")
			links.new(myImageTexture00.outputs['Color'], nodes["Separate Color"].inputs[0])

			# myMath0 = nodes.new("Math")
			myMath0 = nodes.new("ShaderNodeMath")
			myMath0.operation = 'POWER'

			myMath1 = nodes.new("ShaderNodeMath")
			myMath1.operation = 'POWER'

			myMath2 = nodes.new("ShaderNodeMath")
			myMath2.operation = 'POWER'

			myMath0.inputs[1].default_value = 10
			myMath1.inputs[1].default_value = 10
			myMath2.inputs[1].default_value = 10

			links.new(mySepColor.outputs[0], myMath0.inputs[0])
			links.new(mySepColor.outputs[1], myMath1.inputs[0])
			links.new(mySepColor.outputs[2], myMath2.inputs[0])

			myCombineColor = nodes.new("ShaderNodeCombineColor")
			links.new(myMath0.outputs[0], myCombineColor.inputs[0])
			links.new(myMath1.outputs[0], myCombineColor.inputs[1])
			links.new(myMath2.outputs[0], myCombineColor.inputs[2])

			links.new(myCombineColor.outputs[0], nodes["Emission"].inputs[0])






			myPrincipled = nodes.new("ShaderNodeBsdfPrincipled")
			links.new(myCombineColor.outputs[0], myPrincipled.inputs[0])
			links.new(myImageTexture00.outputs[1], myPrincipled.inputs[4])
			links.new(myPrincipled.outputs[0], nodes['Material Output'].inputs[0])





			for area in bpy.context.screen.areas: 
				if area.type == 'VIEW_3D':
					for space in area.spaces: 
						if space.type == 'VIEW_3D':
							space.shading.type = 'MATERIAL'
							# space.shading.type = 'SOLID'

			self.defaultColorSettings_UI()

		self.renderPasses_simple = False
		self.renderPasses_GGX = False

	def skip_refresh_determine(self, mySplitFaceIndexUsable):
		#################################################
		#decide whether to continue and do a full refresh
		#################################################

		skip_refresh_override_recently_cleared_faces = False

		if self.recently_cleared_selFaces == True:
			skip_refresh_override_recently_cleared_faces = True
			self.recently_cleared_selFaces = False

		skip_refresh = False

		matCheck = bpy.data.materials.get("ShaderVisualizer_" + str(mySplitFaceIndexUsable))
		
		if matCheck: #material already exists...check if it is not selected for stage stepping
			for j in self.shadingStages_perFace_stepList:
				if (j["idx"]) == mySplitFaceIndexUsable:
					if j['idx'] not in self.shadingStages_selectedFaces:
						skip_refresh = True

		if self.skip_refresh_override_aov == True:
			skip_refresh = False

		if self.skip_refresh_override_RdotVpow == True:
			skip_refresh = False

		if self.skip_refresh_override_spectral == True:
			skip_refresh = False

		if self.skip_refresh_override_oren_roughness == True:
			skip_refresh = False

		if self.skip_refresh_override_GGX_roughness == True:
			skip_refresh = False

		if self.skip_refresh_override_GGX_fresnel == True:
			skip_refresh = False

		if self.skip_refresh_override_is_metallic == True:
			skip_refresh = False

		if self.skip_refresh_override_aniso_specular == True:
			skip_refresh = False

		if self.skip_refresh_override_aniso_rotation == True:
			skip_refresh = False

		if self.skip_refresh_override_aniso_roughnessX == True:
			skip_refresh = False

		if self.skip_refresh_override_aniso_roughnessY == True:
			skip_refresh = False

		if skip_refresh_override_recently_cleared_faces == True:
			skip_refresh = False

		if self.changedSpecularEquation_variables == True:
			skip_refresh = False

		if self.changedDiffuseEquation_variables == True:
			skip_refresh = False

		if self.skip_refresh_override_use_18_hue_colorspace == True:
			skip_refresh = False

		return skip_refresh
	
	def equation_dynamic_cubeH_creation(self, faceCenter, H, myCam):
		self.myCubeH_dupe.matrix_world = self.myCubeLight_og_Matrix
		
		bpy.context.view_layer.objects.active = self.myCubeH_dupe
		self.myCubeH_dupe.location = faceCenter
		
		self.updateScene()

		#look at direct
		rot_quat = H.to_track_quat('X', 'Z')

		# assume we're using euler rotation
		self.myCubeH_dupe.rotation_euler = rot_quat.to_euler()

		# #####################
		bpy.ops.object.mode_set(mode="OBJECT")
		self.deselectAll()
		self.myCubeH_dupe.select_set(1)

		myCubeH_dupe_Matrix_np = np.array(self.myCubeH_dupe.matrix_world)

		#scale to camera position
		self.objScaling_toMatchPosition_localSolve(self.myCubeH_dupe, self.myCubeH_og.name, myCam.matrix_world.translation, -1, 0, myCubeH_dupe_Matrix_np)

		self.updateScene()

		dynamicM = self.myCubeH_dupe.matrix_world
		return dynamicM

	def equation_dynamic_cubeN_creation(self, shadingPlane, faceCenter):
		# self.profile_stage2_07_a = datetime.now() ################

		self.myCubeN_dupe.matrix_world = self.myCubeN_og_Matrix

		for j in bpy.context.scene.objects:
			if j.name == shadingPlane:
				bpy.context.view_layer.objects.active = j

		normalDir = self.getFaceNormal()
		# myN = normalDir.normalized()

		bpy.context.view_layer.objects.active = self.myCubeN_dupe
		bpy.context.active_object.rotation_mode = 'QUATERNION'
		bpy.context.active_object.rotation_quaternion = normalDir.to_track_quat('X','Z')

		self.myCubeN_dupe.location = faceCenter

		#use x axis
		dynamicM = self.myCubeN_dupe.matrix_world

		# self.profile_stage2_07_b = datetime.now() - self.profile_stage2_07_a
		# self.profile_stage2_07_final += self.profile_stage2_07_b

		return dynamicM

	def equation_dynamic_cubeLight_creation(self, faceCenter, mySun):
		self.myCubeLight_dupe.matrix_world = self.myCubeLight_og_Matrix
		
		bpy.context.view_layer.objects.active = self.myCubeLight_dupe
		self.myCubeLight_dupe.location = faceCenter
		
		self.updateScene()
		self.look_at2(self.myCubeLight_dupe, self.pos_light_global_v)

		# #####################
		bpy.ops.object.mode_set(mode="OBJECT")
		self.deselectAll()
		self.myCubeLight_dupe.select_set(1)

		myCubeLight_dupe_Matrix_np = np.array(self.myCubeLight_dupe.matrix_world)

		self.objScaling_toMatchPosition_localSolve(self.myCubeLight_dupe, self.myCubeLight_og.name, mySun.matrix_world.translation, -1, 0, myCubeLight_dupe_Matrix_np)

		self.updateScene()

		dynamicM = self.myCubeLight_dupe.matrix_world
		return dynamicM

	def equation_dynamic_cubeV_creation(self, faceCenter, myCam):
		self.myCubeCam_dupe.matrix_world = self.myCubeLight_og_Matrix
		bpy.context.view_layer.objects.active = self.myCubeCam_dupe
		self.myCubeCam_dupe.location = faceCenter

		self.updateScene()
		
		self.look_at2(self.myCubeCam_dupe, self.pos_camera_global_v)

		# #####################
		bpy.ops.object.mode_set(mode="OBJECT")
		self.deselectAll()
		self.myCubeCam_dupe.select_set(1)

		myCubeCam_dupe_Matrix_np = np.array(self.myCubeCam_dupe.matrix_world)

		self.objScaling_toMatchPosition_localSolve(self.myCubeCam_dupe, self.myCubeLight_og.name, myCam.matrix_world.translation, 1, 0, myCubeCam_dupe_Matrix_np)

		self.updateScene()

		dynamicM = self.myCubeCam_dupe.matrix_world
		return dynamicM

	def equation_dynamic_cubeR_creation(self, defaultMatrix, R):
		self.myCubeR_dupe.matrix_world = defaultMatrix

		bpy.context.view_layer.objects.active = self.myCubeR_dupe

		#apply rotation
		bpy.context.active_object.rotation_mode = 'QUATERNION'
		bpy.context.active_object.rotation_quaternion = R.to_track_quat('X','Z')

		dynamicM = self.myCubeR_dupe.matrix_world

		return dynamicM


	def objScaling_toMatchPosition_localSolve(self, objToScale, objToScaleOrigName, toCoord, facingDirection, scaleMode, mWorld):
		global_coord = toCoord
		local_coord = mathutils.Matrix(mWorld.tolist()).inverted() @ global_coord
		loc_usable = local_coord.x
		bbx_og = bpy.data.objects[objToScaleOrigName].dimensions.x
		
		mySolve_ws = None

		if scaleMode == 0:
			mySolve_ws = (loc_usable / bbx_og)
		elif scaleMode == 1:
			mySolve_ws = (bbx_og / loc_usable)

		objToScale.scale = mathutils.Vector((facingDirection * mySolve_ws, 1, 1))

	def getFaceCenter(self, myObj, idx):
		# print('in get face center...')
		# myObj = bpy.context.active_object

		v0 = myObj.data.vertices[0]
		co_final = myObj.matrix_world @ v0.co

		v1 = myObj.data.vertices[1]
		c1_final = myObj.matrix_world @ v1.co

		v2 = myObj.data.vertices[2]
		c2_final = myObj.matrix_world @ v2.co

		centerX = (co_final.x + c1_final.x + c2_final.x) / 3
		centerY = (co_final.y + c1_final.y + c2_final.y) / 3
		centerZ = (co_final.z + c1_final.z + c2_final.z) / 3

		myCenter =  mathutils.Vector((centerX, centerY, centerZ))

		return myCenter

	def find_edge_loops(self, loop, max_loops = 1000):
		i = 0
		first_loop = loop
		while i < max_loops: 
			# Jump to adjacent face and walk two edges forward
			loop = loop.link_loop_next.link_loop_radial_next.link_loop_next
			loop.edge.select = True
			i += 1
			# If radial loop links back here, we're boundary, thus done        
			if loop == first_loop:
				break      
			
	def selectEdgeLoop(self, edgeIdx):
		# Get the active mesh
		me = bpy.context.active_object.data

		# Get a BMesh representation
		bm = bmesh.new()   # create an empty BMesh
		bm.from_mesh(me)   # fill it in from a Mesh
	
		bm.select_mode = {'EDGE'}

		bm.edges.ensure_lookup_table()

		for i in bm.edges:
			# print('i.index = ', i.index)
			# i.select = True
			if i.index == edgeIdx:
				i.select = True

		selected_edges = [ e for e in bm.edges if e.select ]

		for edge in selected_edges:
			# Get rings from "forward" loop
			self.find_edge_loops(edge.link_loops[0])
			# Get rings from "backward" loop
			# rings(edge.link_loops[1])

		bm.select_flush_mode()
		bpy.context.active_object.data.update()

		bpy.ops.object.mode_set(mode="OBJECT")
		# Finish up, write the bmesh back to the mesh

		bm.to_mesh(me)
		bm.free()  # free and prevent further access

		bpy.ops.object.mode_set(mode="EDIT")
		bpy.ops.mesh.select_mode(type="EDGE")

	def getFaceNormal(self):
		normalDir = None

		me = bpy.context.active_object.data

		bm = bmesh.new()   # create an empty BMesh
		bm.from_mesh(me)   # fill it in from a Mesh

		for f in bm.faces:
			normalDir = f.normal

		return normalDir

	def selectMultipleFace0(self, myArray):
		# Get the active mesh
		me = bpy.context.active_object.data

		# Get a BMesh representation
		bm = bmesh.new()   # create an empty BMesh
		bm.from_mesh(me)   # fill it in from a Mesh
	
		bm.faces.ensure_lookup_table()

		for f in bm.faces:
			# f.select = False
			f.select = True

		# for i in myArray:
		# 	for j in bm.faces:
		# 		# print('i.index = ', i.index)
		# 		if j.index == i:
		# 			j.select = True

		bm.select_flush_mode()
		bpy.context.active_object.data.update()

		bpy.ops.object.mode_set(mode="OBJECT")
		# Finish up, write the bmesh back to the mesh

		bm.to_mesh(me)
		bm.free()  # free and prevent further access

		bpy.ops.object.mode_set(mode="EDIT")
		bpy.ops.mesh.select_mode(type="FACE")

	def selectMultipleFace(self, myArray):
		# Get the active mesh
		me = bpy.context.active_object.data

		# Get a BMesh representation
		bm = bmesh.new()   # create an empty BMesh
		bm.from_mesh(me)   # fill it in from a Mesh
	
		bm.faces.ensure_lookup_table()

		for f in bm.faces:
			f.select = False

		for i in myArray:
			for j in bm.faces:
				# print('i.index = ', i.index)
				if j.index == i:
					j.select = True

		bm.select_flush_mode()
		bpy.context.active_object.data.update()

		bpy.ops.object.mode_set(mode="OBJECT")
		# Finish up, write the bmesh back to the mesh

		bm.to_mesh(me)
		bm.free()  # free and prevent further access

		bpy.ops.object.mode_set(mode="EDIT")
		bpy.ops.mesh.select_mode(type="FACE")

	def selectMultipleEdge(self, myArray):
		# Get the active mesh
		me = bpy.context.active_object.data

		# Get a BMesh representation
		bm = bmesh.new()   # create an empty BMesh
		bm.from_mesh(me)   # fill it in from a Mesh
	
		bm.select_mode = {'EDGE'}

		bm.edges.ensure_lookup_table()

		for i in myArray:
			for j in bm.edges:
				# print('i.index = ', i.index)
				if j.index == i:
					j.select = True

		bm.select_flush_mode()
		bpy.context.active_object.data.update()

		bpy.ops.object.mode_set(mode="OBJECT")
		# Finish up, write the bmesh back to the mesh

		bm.to_mesh(me)
		bm.free()  # free and prevent further access

		bpy.ops.object.mode_set(mode="EDIT")
		bpy.ops.mesh.select_mode(type="EDGE")		

	def modelArrow(self, inputAxis, amt):
		# Get the active mesh
		me = bpy.context.active_object.data

		# Get a BMesh representation
		bm = bmesh.new()   # create an empty BMesh
		bm.from_mesh(me)   # fill it in from a Mesh

		if 'inputAxis' != 'extend':
			bm.edges.ensure_lookup_table()

		selected_edges = [ e for e in bm.edges if e.select ]

		for edge in selected_edges:
			for v in edge.verts:
				#print(v)
				if inputAxis == 'x':
					# v.co.x += amt
					v.co.x = amt

				elif inputAxis == 'y':
					# v.co.y += amt
					v.co.y = amt

				elif inputAxis == 'z':
					# v.co.z += amt
					v.co.z = amt

		for i in selected_edges:
			i.select = False

		#########
		## FACES
		#########
		selected_faces = [ f for f in bm.faces if f.select ]
		for face in selected_faces:
			for v in face.verts:
				#print(v)
				if inputAxis == 'extend':
					# v.co.x += amt
					v.co.x = amt

		for i in selected_faces:
			i.select = False

		bm.select_flush_mode()
		bpy.context.active_object.data.update()

		bpy.ops.object.mode_set(mode="OBJECT")
		# Finish up, write the bmesh back to the mesh

		bm.to_mesh(me)
		bm.free()  # free and prevent further access

	def makeArrowFromCube(self, myObj, extend):
		#move out 5 in +Y (91, 93, 95)
		#move out 5 in -Y (66, 68, 70)
		#move 90 loop ring down in X

		bpy.context.view_layer.objects.active = myObj

		isExtending = 0
		if (extend == 0):
			isExtending = 0
		else:
			isExtending = 1

		edgeArray_0 = [91, 93, 95]
		self.selectMultipleEdge(edgeArray_0)
		self.modelArrow('y', self.arrow_wings)

		edgeArray_1 = [66, 68, 70]
		self.selectMultipleEdge(edgeArray_1)
		self.modelArrow('y', -self.arrow_wings)

		# return
		facesToSelect = [2, 37, 36, 31, 35, 34, 30, 33, 32]
		self.selectMultipleFace(facesToSelect)
		self.modelArrow('extend', self.len_arrow_frontExtend)

		self.selectEdgeLoop(90)
		self.modelArrow('x', .2)

		# return

		if (isExtending == 1):
			facesToSelect = [46, 49, 48, 47, 51, 50, 0, 53, 52]
			self.selectMultipleFace(facesToSelect)
			self.modelArrow('extend', extend) #extend back face to make longer

	def splitObjectIntoFacesFunc0(self, mesh):
		storedInputName = mesh.name

		allMeshes_pre = []
		allMeshes_post = []
		allMeshes_createdNow = []
		renamedAndSplitNodes = []

		for i in bpy.data.objects:
			allMeshes_pre.append(i)

		self.deselectAll()
		mesh.select_set(1)

		bpy.ops.object.mode_set(mode = 'EDIT')
		bpy.ops.mesh.select_all(action='SELECT')
		bpy.ops.mesh.edge_split()
		bpy.ops.mesh.separate(type='LOOSE')
		bpy.ops.object.mode_set(mode = 'OBJECT')
		bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
		bpy.ops.object.mode_set(mode = 'OBJECT')

		for i in bpy.data.objects:
			allMeshes_post.append(i)

		for i in allMeshes_post:
			if i not in allMeshes_pre:
				allMeshes_createdNow.append(i)

		allMeshes_createdNow.append(mesh)

		for idx, i in enumerate(allMeshes_createdNow):
			usableName = storedInputName + '_' + str(idx)
			i.name = usableName
			renamedAndSplitNodes.append(usableName)

		return renamedAndSplitNodes

	def normalize_Numpy(self, v):
		norm = np.linalg.norm(v)

		if norm == 0: 
			return v
			
		return v / norm

	def numpyReflectVector(self, vector, normal):
		'''
			Reflects a vector across a normal vector.

			Args:
			vector (numpy.ndarray): The vector to be reflected.
			normal (numpy.ndarray): The normal vector of the reflecting surface.

			Returns:
			numpy.ndarray: The reflected vector.
		'''

		# Ensure the normal vector is a unit vector

		normal = normal / np.linalg.norm(normal)

		# Calculate the projection of the vector onto the normal
		projection = np.dot(vector, normal) * normal

		# Calculate the reflected vector
		reflected_vector = vector - 2 * projection

		return reflected_vector
	
	##############
	#STAGE HELPERS
	##############
	def copyAndSet_arrow(self, idx, matrix, namePrefix, type):
		if type == "V":
			bpy.context.view_layer.objects.active = self.myCubeCam
		elif type == "N":
			bpy.context.view_layer.objects.active = self.myCubeN_og
		elif type == "R":
			bpy.context.view_layer.objects.active = self.myCubeR_og
		elif type == "L":
			bpy.context.view_layer.objects.active = self.myCubeLight_og
		elif type == "H":
			bpy.context.view_layer.objects.active = self.myCubeH_og
		elif type == "halfFOV_L":
			bpy.context.view_layer.objects.active = self.myCubeV_halfFov_L_og
		elif type == "halfFOV_R":
			bpy.context.view_layer.objects.active = self.myCubeV_halfFov_R_og
		elif type == "halfFOV_LB":
			bpy.context.view_layer.objects.active = self.myCubeV_halfFov_LB_og
		elif type == "halfFOV_LT":
			bpy.context.view_layer.objects.active = self.myCubeV_halfFov_LT_og
		elif type == "halfFOV_RT":
			bpy.context.view_layer.objects.active = self.myCubeV_halfFov_RT_og
		elif type == "halfFOV_RB":
			bpy.context.view_layer.objects.active = self.myCubeV_halfFov_RB_og

		arrow_instance = self.copyObject()

		# bpy.context.view_layer.objects.active = arrow_instance

		arrow_instance.name = namePrefix + str(idx)
		arrow_instance.hide_set(0)
		arrow_instance.matrix_world = matrix

		return arrow_instance


	def show_arrow_halfFOV_LB(self, mySplitFaceIndexUsable):
		nameToLookFor = 'cube_half_FOV_LB_instance_' + mySplitFaceIndexUsable

		for k in self.objectsToToggleOnOffLater:
			if k.name == nameToLookFor:
				# if k.hide_get() == 1:
				k.hide_set(0)

				return

		restored_HalfFov_LB_M_np = None

		for i in self.arrow_dynamic_instance_M_all_list_matrixOnly_debug:
			temp_idx = i['mySplitFaceIndexUsable']
			if temp_idx == mySplitFaceIndexUsable:
				HalfFov_LB_M_np = i['HalfFov_LB_M_np']
				restored_HalfFov_LB_M_np = mathutils.Matrix(HalfFov_LB_M_np.tolist())

		myCube_HalfFov_LB_instance = self.copyAndSet_arrow(mySplitFaceIndexUsable, restored_HalfFov_LB_M_np, 'cube_half_FOV_LB_instance_', 'halfFOV_LB')
		self.objectsToToggleOnOffLater.append(myCube_HalfFov_LB_instance)

		return myCube_HalfFov_LB_instance

	def show_arrow_halfFOV_LT(self, mySplitFaceIndexUsable):
		nameToLookFor = 'cube_half_FOV_LT_instance_' + mySplitFaceIndexUsable

		for k in self.objectsToToggleOnOffLater:
			if k.name == nameToLookFor:
				# if k.hide_get() == 1:
				k.hide_set(0)

				return

		restored_HalfFov_LT_M_np = None

		for i in self.arrow_dynamic_instance_M_all_list_matrixOnly_debug:
			temp_idx = i['mySplitFaceIndexUsable']
			if temp_idx == mySplitFaceIndexUsable:
				HalfFov_LT_M_np = i['HalfFov_LT_M_np']
				restored_HalfFov_LT_M_np = mathutils.Matrix(HalfFov_LT_M_np.tolist())

		myCube_HalfFov_LT_instance = self.copyAndSet_arrow(mySplitFaceIndexUsable, restored_HalfFov_LT_M_np, 'cube_half_FOV_LT_instance_', 'halfFOV_LT')
		self.objectsToToggleOnOffLater.append(myCube_HalfFov_LT_instance)

		return myCube_HalfFov_LT_instance

	def show_arrow_halfFOV_RT(self, mySplitFaceIndexUsable):
		nameToLookFor = 'cube_half_FOV_RT_instance_' + mySplitFaceIndexUsable

		for k in self.objectsToToggleOnOffLater:
			if k.name == nameToLookFor:
				# if k.hide_get() == 1:
				k.hide_set(0)

				return

		restored_HalfFov_RT_M_np = None

		for i in self.arrow_dynamic_instance_M_all_list_matrixOnly_debug:
			temp_idx = i['mySplitFaceIndexUsable']
			if temp_idx == mySplitFaceIndexUsable:
				HalfFov_RT_M_np = i['HalfFov_RT_M_np']
				restored_HalfFov_RT_M_np = mathutils.Matrix(HalfFov_RT_M_np.tolist())

		myCube_HalfFov_RT_instance = self.copyAndSet_arrow(mySplitFaceIndexUsable, restored_HalfFov_RT_M_np, 'cube_half_FOV_RT_instance_', 'halfFOV_RT')
		self.objectsToToggleOnOffLater.append(myCube_HalfFov_RT_instance)

		return myCube_HalfFov_RT_instance



	def show_arrow_halfFOV_RB(self, mySplitFaceIndexUsable):
		nameToLookFor = 'cube_half_FOV_RB_instance_' + mySplitFaceIndexUsable

		for k in self.objectsToToggleOnOffLater:
			if k.name == nameToLookFor:
				# if k.hide_get() == 1:
				k.hide_set(0)

				return

		restored_HalfFov_RB_M_np = None

		for i in self.arrow_dynamic_instance_M_all_list_matrixOnly_debug:
			temp_idx = i['mySplitFaceIndexUsable']
			if temp_idx == mySplitFaceIndexUsable:
				HalfFov_RB_M_np = i['HalfFov_RB_M_np']
				restored_HalfFov_RB_M_np = mathutils.Matrix(HalfFov_RB_M_np.tolist())

		myCube_HalfFov_RB_instance = self.copyAndSet_arrow(mySplitFaceIndexUsable, restored_HalfFov_RB_M_np, 'cube_half_FOV_RB_instance_', 'halfFOV_RB')
		self.objectsToToggleOnOffLater.append(myCube_HalfFov_RB_instance)

		return myCube_HalfFov_RB_instance










	def show_arrow_halfFOV_L(self, mySplitFaceIndexUsable):
		nameToLookFor = 'cube_half_FOV_L_instance_' + mySplitFaceIndexUsable

		for k in self.objectsToToggleOnOffLater:
			if k.name == nameToLookFor:
				# if k.hide_get() == 1:
				k.hide_set(0)

				return

		restored_HalfFov_L_M_np = None

		for i in self.arrow_dynamic_instance_M_all_list_matrixOnly_debug:
			temp_idx = i['mySplitFaceIndexUsable']
			if temp_idx == mySplitFaceIndexUsable:
				HalfFov_L_M_np = i['HalfFov_L_M_np']
				restored_HalfFov_L_M_np = mathutils.Matrix(HalfFov_L_M_np.tolist())

		myCube_HalfFov_L_instance = self.copyAndSet_arrow(mySplitFaceIndexUsable, restored_HalfFov_L_M_np, 'cube_half_FOV_L_instance_', 'halfFOV_L')
		self.objectsToToggleOnOffLater.append(myCube_HalfFov_L_instance)

		return myCube_HalfFov_L_instance
	
	def show_arrow_halfFOV_R(self, mySplitFaceIndexUsable):
		nameToLookFor = 'cube_half_FOV_R_instance_' + mySplitFaceIndexUsable

		for k in self.objectsToToggleOnOffLater:
			if k.name == nameToLookFor:
				# if k.hide_get() == 1:
				k.hide_set(0)

				return

		restored_HalfFov_R_M_np = None

		for i in self.arrow_dynamic_instance_M_all_list_matrixOnly_debug:
			temp_idx = i['mySplitFaceIndexUsable']
			if temp_idx == mySplitFaceIndexUsable:
				HalfFov_R_M_np = i['HalfFov_R_M_np']
				restored_HalfFov_R_M_np = mathutils.Matrix(HalfFov_R_M_np.tolist())

		myCube_HalfFov_R_instance = self.copyAndSet_arrow(mySplitFaceIndexUsable, restored_HalfFov_R_M_np, 'cube_half_FOV_R_instance_', 'halfFOV_R')
		self.objectsToToggleOnOffLater.append(myCube_HalfFov_R_instance)

		return myCube_HalfFov_R_instance	

	def show_arrow_H(self, shadingPlane, faceCenter, mySplitFaceIndexUsable):
		nameToLookFor = 'cubeH_instance_' + mySplitFaceIndexUsable

		for k in self.objectsToToggleOnOffLater:
			if k.name == nameToLookFor:
				# if k.hide_get() == 1:
				k.hide_set(0)

				return

		restored_H_M_np = None

		for i in self.arrow_dynamic_instance_M_all_list_matrixOnly:
			temp_idx = i['mySplitFaceIndexUsable']
			if temp_idx == mySplitFaceIndexUsable:
				H_M_np = i['H_M_np']
				restored_H_M_np = mathutils.Matrix(H_M_np.tolist())

		myCubeH_instance = self.copyAndSet_arrow(mySplitFaceIndexUsable, restored_H_M_np, 'cubeH_instance_', 'H')
		self.objectsToToggleOnOffLater.append(myCubeH_instance)

		return myCubeH_instance

	def show_arrow_N(self, shadingPlane, faceCenter, mySplitFaceIndexUsable):
		nameToLookFor = 'cubeN_instance_' + mySplitFaceIndexUsable

		for k in self.objectsToToggleOnOffLater:
			if k.name == nameToLookFor:
				# if k.hide_get() == 1:
				k.hide_set(0)

				return

		restored_N_M_np = None

		for i in self.arrow_dynamic_instance_M_all_list_matrixOnly:
			temp_idx = i['mySplitFaceIndexUsable']
			if temp_idx == mySplitFaceIndexUsable:
				N_M_np = i['N_M_np']
				restored_N_M_np = mathutils.Matrix(N_M_np.tolist())

		myCubeN_instance = self.copyAndSet_arrow(mySplitFaceIndexUsable, restored_N_M_np, 'cubeN_instance_', 'N')
		self.objectsToToggleOnOffLater.append(myCubeN_instance)

		return myCubeN_instance

	def show_arrow_V_to_faceCenter(self, faceCenter, mySplitFaceIndexUsable):
		nameToLookFor = 'faceCenterToV_rc_instance_' + mySplitFaceIndexUsable

		for k in self.objectsToToggleOnOffLater:
			if k.name == nameToLookFor:
				# if k.hide_get() == 1:
				k.hide_set(0)

				return

		########################

		restored_V_M_np = None

		for i in self.arrow_dynamic_instance_M_all_list_matrixOnly:
			temp_idx = i['mySplitFaceIndexUsable']
			if temp_idx == mySplitFaceIndexUsable:
				V_M_np = i['V_M_np']
				restored_V_M_np = mathutils.Matrix(V_M_np.tolist())
		
		myCubeV_instance = self.copyAndSet_arrow(mySplitFaceIndexUsable, restored_V_M_np, 'faceCenterToV_rc_instance_', 'V')

		self.objectsToToggleOnOffLater.append(myCubeV_instance)

		# outputMatrix = restored_V_M_np.to_quaternion() @ mathutils.Vector((-1.0, 0.0, 0.0)) #####
		# outputMatrix = restored_V_M_np.to_translation()
		# return outputMatrix

	def show_arrow_L_to_faceCenter(self, faceCenter, mySplitFaceIndexUsable):
		nameToLookFor = 'cubeLight_instance_' + mySplitFaceIndexUsable

		for k in self.objectsToToggleOnOffLater:
			if k.name == nameToLookFor:
				# if k.hide_get() == 1:
				k.hide_set(0)

				return

		########################

		restored_L_M_np = None

		for i in self.arrow_dynamic_instance_M_all_list_matrixOnly:
			temp_idx = i['mySplitFaceIndexUsable']
			if temp_idx == mySplitFaceIndexUsable:
				L_M_np = i['L_M_np']
				restored_L_M_np = mathutils.Matrix(L_M_np.tolist())
		
		myCubeLight_instance = self.copyAndSet_arrow(mySplitFaceIndexUsable, restored_L_M_np, 'cubeLight_instance_', 'L')
		self.objectsToToggleOnOffLater.append(myCubeLight_instance)

	def show_arrow_R(self, faceCenter, mySplitFaceIndexUsable, L, N):
		nameToLookFor = 'cubeR_instance_' + mySplitFaceIndexUsable

		for k in self.objectsToToggleOnOffLater:
			if k.name == nameToLookFor:
				# if k.hide_get() == 1:
				k.hide_set(0)

				return
				
		########################

		restored_R_M_np = None

		for i in self.arrow_dynamic_instance_M_all_list_matrixOnly:
			temp_idx = i['mySplitFaceIndexUsable']
			if temp_idx == mySplitFaceIndexUsable:
				R_M_np = i['R_M_np']
				restored_R_M_np = mathutils.Matrix(R_M_np.tolist())

		myCubeR_instance = self.copyAndSet_arrow(mySplitFaceIndexUsable, restored_R_M_np, 'cubeR_instance_', 'R')
		self.objectsToToggleOnOffLater.append(myCubeR_instance)

		return myCubeR_instance

	def setActiveStageMaterial(self, shadingPlane, idx, r, g, b):
		for j in bpy.context.scene.objects:
			if j.name == shadingPlane:
				bpy.context.view_layer.objects.active = j

		# bpy.context.view_layer.objects.active = 
		mat1 = self.newShader("ShaderVisualizer_" + str(idx), "emission", r, g, b)
		bpy.context.active_object.data.materials.clear()
		bpy.context.active_object.data.materials.append(mat1)

	def raycast_abj_scene(self, origin, direction, debugidx):
		for j in self.allNamesToToggleDuringRaycast:
			j.hide_set(1)

		objectsToToggleOnOffLater_stored = []
		for j in self.objectsToToggleOnOffLater:
			if j.hide_get() == 0:
				objectsToToggleOnOffLater_stored.append(j)
				
		for j in objectsToToggleOnOffLater_stored:
			j.hide_set(1)

		storedCubeCamState = 1
		if self.myCubeCam.hide_get() == 0:
			storedCubeCamState = 0
			self.myCubeCam.hide_set(1)

		myDepsgraph = bpy.context.view_layer.depsgraph
		dir_usable = direction
		origin_usable = origin

		# self.updateScene() ################
		# myDepsgraph.update()

		hit, loc, norm, idx, obj, mw = bpy.context.scene.ray_cast(myDepsgraph, origin_usable, dir_usable)

		######### OBJECT
		toReturn = None

		toDebug = '350'

		if hit:
			mySplitFaceIndexUsable_rayHit = obj.name.split('_', -1)[1]

			if debugidx == mySplitFaceIndexUsable_rayHit:
				toReturn = True
				# if debugidx == '236' or debugidx == '361' or debugidx == '296' or debugidx == '223':
				# 	print('TRUE for debugIdx, obj : ', debugidx, ' ', obj.name)

				# if debugidx == toDebug:
				# print('TRUE for debugIdx, obj : ', debugidx, ' ', obj.name)
			else:
				toReturn = False
				# if debugidx == '236' or debugidx == '361' or debugidx == '296' or debugidx == '223':

				# if debugidx == toDebug:
				# print('FALSE for debugIdx, obj : ', debugidx, ' ', obj.name)
	
		else:
			toReturn = False
			# if debugidx == '236' or debugidx == '361' or debugidx == '296' or debugidx == '223':

			if debugidx == toDebug:
				print('ray miss for debugIdx, obj : ', debugidx)


		for j in objectsToToggleOnOffLater_stored:
			j.hide_set(0)

		if storedCubeCamState == 0:
			self.myCubeCam.hide_set(0)

		for j in self.allNamesToToggleDuringRaycast:
			j.hide_set(0)

		# print('raycast ', debugidx, ' ', toReturn)

		return toReturn

class SCENE_PT_ABJ_Shader_Debugger_Panel(bpy.types.Panel):
	"""Creates a Panel in the scene context of the properties editor"""
	bl_label = "ABJ Shader Debugger"
	bl_idname = "SCENE_PT_ABJ_Shader_Debugger_Panel"
	bl_space_type = 'PROPERTIES'
	bl_region_type = 'WINDOW'
	bl_context = "scene"

	def draw(self, context):
		layout = self.layout

		######################################
		###### SPECTRAL COMPOSITOR
		######################################
		layout.label(text='SPECTRAL COMPOSITOR')
		row = layout.row()
		row.scale_y = 2.0 ###
		row.operator('shader.abj_shader_debugger_spectral_compositor_operator')

		row = layout.row()
		row.scale_y = 1.0 ###
		row.operator('shader.abj_shader_debugger_compositor_stock_operator')

		######################################
		###### STAGE 1
		######################################
		layout.label(text='PRE-PROCESS')
		row = layout.row()
		row.operator('shader.abj_shader_debugger_restorelight_operator')
		row.operator('shader.abj_shader_debugger_restorerxyz_operator')

		row = layout.row()
		row.scale_y = 2.0 ###
		row.operator('shader.abj_shader_debugger_randomlight_operator')

		row.operator('shader.abj_shader_debugger_randomrotation_operator')

		row = layout.row()
		row.prop(bpy.context.scene, 'min_shaded_prop', text="")

		row = layout.row()
		row.scale_y = 2.0 ###
		row.operator('shader.abj_shader_debugger_staticstage1_operator')

		######################################
		###### STAGE 2
		######################################
		layout.label(text='RENDER')
		row = layout.row()
		row.scale_y = 2.0 ###
		row.operator('shader.abj_shader_debugger_refreshstage2_operator')

		
		row = layout.row()
		row.prop(bpy.context.scene, 'spectral_mix_multiplier_prop')

		row = layout.row()
		row.prop(bpy.context.scene, 'spectral_mix_multiplier2_prop')

		layout.label(text='Render Passes')
		row = layout.row()
		row.scale_y = 1.0 ###
		row.operator('shader.abj_shader_debugger_renderpasses_operator')

		######################################
		###### WRITTEN
		######################################
		layout.label(text='WRITTEN')
		row = layout.row()
		row.scale_y = 2.0 ###
		row.operator('shader.abj_shader_debugger_writtenrender_operator')


		row = layout.row()
		row.prop(bpy.context.scene, 'written_aspect_prop')

		row = layout.row()
		row.prop(bpy.context.scene, 'written_fov_prop')

		row = layout.row()
		row.prop(bpy.context.scene, 'written_znear_prop')

		row = layout.row()
		row.prop(bpy.context.scene, 'written_zfar_prop')


		######################################
		###### STAGE IDX
		######################################
		layout.label(text='STAGE IDX')
		
		row = layout.row()
		row.scale_y = 2.0 ###
		row.operator('shader.abj_shader_debugger_stages_selectfaces_operator')
		row = layout.row()
		row.scale_y = 1.0 ###
		row.operator('shader.abj_shader_debugger_stageresetall_operator')
		row = layout.row()
		row.scale_y = 2.0 ###
		row.operator('shader.abj_shader_debugger_stageidxminus_operator')
		row.operator('shader.abj_shader_debugger_stageidxplus_operator')
		row.operator('shader.abj_shader_debugger_stageidxzero_operator')
		row = layout.row()
		row.scale_y = 1.0 ###
		row.operator('shader.abj_shader_debugger_stageidxprint_operator')
		
		layout.label(text='Diffuse:')
		row = layout.row()
		row.prop(bpy.context.scene, 'diffuse_equation_enum_prop', text="")
		row = layout.row()
		row.prop(bpy.context.scene, 'oren_roughness_prop')

		layout.label(text='Specular:')
		row = layout.row()
		row.prop(bpy.context.scene, 'specular_equation_enum_prop', text="")

		layout.label(text='GGX:')
		
		row = layout.row()
		row.prop(bpy.context.scene, 'ggx_roughness_prop')
		
		row = layout.row()
		row.prop(bpy.context.scene, 'ggx_fresnel_prop')

		row = layout.row()
		row.prop(bpy.context.scene, 'is_metallic_prop')

		row = layout.row()
		row.prop(bpy.context.scene, 'aniso_specular_prop')

		row = layout.row()
		row.prop(bpy.context.scene, 'aniso_rotation_prop')

		row = layout.row()
		row.prop(bpy.context.scene, 'aniso_roughnessX_prop')

		row = layout.row()
		row.prop(bpy.context.scene, 'aniso_roughnessY_prop')

		######################################
		###### R_DOT_V_POW
		######################################
		layout.label(text='Simple')
		row = layout.row()
		layout.label(text='R.V POW:')
		row = layout.row()
		row.scale_y = 1.0 ###
		row.prop(bpy.context.scene, 'r_dot_v_pow_enum_prop', text="")

		layout.label(text='Camera')
		row = layout.row()
		row.operator('shader.abj_shader_debugger_restorecameraview_operator')

		######################################
		###### AOV
		######################################
		layout.label(text='AOV')
		row = layout.row()
		row.scale_y = 1.0 ###
		row.prop(bpy.context.scene, 'aov_enum_prop', text="")

		######################################
		###### INPUT MESH SELECT
		######################################
		layout.label(text='INPUT MESH')
		row = layout.row()
		row.scale_y = 1.0 ###
		row.prop(bpy.context.scene, 'primitive_enum_prop', text="")
		row = layout.row()
		row.prop(bpy.context.scene, 'subdivision_toggle_enum_prop', text="")
		# row = layout.row()
		
		#show / hide arrows
		layout.label(text='Toggle 0')
		row = layout.row()
		row.scale_y = 1.0 ###
		row.operator('shader.abj_shader_debugger_showhidearrow_operator')
		row.operator('shader.abj_shader_debugger_showhidecubecam_operator')
		row.operator('shader.abj_shader_debugger_toggleextras_operator')
		row.operator('shader.abj_shader_debugger_showhidetext_operator')

		######################################
		###### TEXT
		######################################
		layout.label(text='TEXT')
		layout.label(text='Precision')
		row = layout.row()
		row.prop(bpy.context.scene, 'text_rgb_precision_enum_prop', text="")

		row = layout.row()
		row.prop(bpy.context.scene, 'text_radius_0_prop')
		
		row = layout.row()
		row.prop(bpy.context.scene, 'text_radius_1_prop')

		layout.label(text='Text Rxyz')
		row = layout.row()
		row.prop(bpy.context.scene, 'text_rotate_x_prop')
		row.prop(bpy.context.scene, 'text_rotate_y_prop')
		row.prop(bpy.context.scene, 'text_rotate_z_prop')

		######################################
		###### COLOR PRESETS
		######################################
		layout.label(text='COLOR PRESETS')
		row = layout.row()
		row.scale_y = 2.0 ###
		row.operator('shader.abj_shader_debugger_agx_color_operator')
		row.operator('shader.abj_shader_debugger_text_color_operator')
		row = layout.row()
		row.scale_y = 2.0 ###
		row.operator('shader.abj_shader_debugger_stereoscopic_color_operator')
		row.operator('shader.abj_shader_debugger_default_color_operator')

		######################################
		###### CONVIENENCE REDUNDANT CONTROLS
		######################################
		layout.label(text='PRE-PROCESS')
		row = layout.row()
		row.scale_y = 2.0 ###
		row.operator('shader.abj_shader_debugger_staticstage1_operator')
		
		layout.label(text='RENDER')
		row = layout.row()
		row.scale_y = 2.0 ###
		row.operator('shader.abj_shader_debugger_refreshstage2_operator')

		layout.label(text='Render Passes')
		row = layout.row()
		row.scale_y = 1.0 ###
		row.operator('shader.abj_shader_debugger_renderpasses_operator')

		######################################
		###### GRADIENT
		######################################
		row = layout.row()
		row.prop(bpy.context.scene, 'additive_or_subtractive_color_blending_enum_prop', text="")

		layout.label(text='Gradients Gamma Correct')
		row = layout.row()
		row.prop(bpy.context.scene, 'gamma_correct_gradient_color_prop')
		row = layout.row()
		row.prop(bpy.context.scene, 'gamma_correct_gradient_colorWheel_prop')

		layout.label(text='Gradients')

		row = layout.row()
		row.scale_y = 2.0 ###
		row.operator('shader.abj_shader_debugger_gradientcolor_operator')

		row = layout.row()
		row.prop(bpy.context.scene, 'gradient_color0_prop')
		row = layout.row()
		row.prop(bpy.context.scene, 'gradient_color1_prop')

		row = layout.row()
		row.prop(bpy.context.scene, 'gradient_method0_step_prop')
		row = layout.row()
		row.prop(bpy.context.scene, 'gradient_method0_rowRange_prop')
		row = layout.row()
		row.prop(bpy.context.scene, 'gradient_method0_size_prop')
		row = layout.row()
		row.prop(bpy.context.scene, 'gradient_method0_spacing_prop')
		row = layout.row()
		row.prop(bpy.context.scene, 'gradient_method0_height_prop')

		layout.label(text='Color Wheel')
		row = layout.row()
		row.prop(bpy.context.scene, 'gradient_outer_circle_steps_prop')
		row = layout.row()
		row.prop(bpy.context.scene, 'gradient_inner_circle_steps_prop')
		row = layout.row()
		row.scale_y = 2.0 ###
		row.operator('shader.abj_shader_debugger_gradientcolorwheel_operator')
		row = layout.row()
		row.operator('shader.abj_shader_debugger_preset18_0_operator')
		row = layout.row()
		row.prop(bpy.context.scene, 'use_18_hue_colorspace_prop')

		layout.label(text='Spectral Multi Blend')
		row = layout.row()
		row.prop(bpy.context.scene, 'spectral_multi_0_Blend_prop')
		row = layout.row()
		row.prop(bpy.context.scene, 'spectral_multi_0_Factor_prop')
		row = layout.row()
		row.prop(bpy.context.scene, 'spectral_multi_0_Tint_prop')



		row = layout.row()
		row.prop(bpy.context.scene, 'spectral_multi_1_Blend_prop')
		row = layout.row()
		row.prop(bpy.context.scene, 'spectral_multi_1_Factor_prop')
		row = layout.row()
		row.prop(bpy.context.scene, 'spectral_multi_1_Tint_prop')



		row = layout.row()
		row.prop(bpy.context.scene, 'spectral_multi_2_Blend_prop')
		row = layout.row()
		row.prop(bpy.context.scene, 'spectral_multi_2_Factor_prop')
		row = layout.row()
		row.prop(bpy.context.scene, 'spectral_multi_2_Tint_prop')


		row = layout.row()
		row.prop(bpy.context.scene, 'spectral_multi_3_Blend_prop')
		row = layout.row()
		row.prop(bpy.context.scene, 'spectral_multi_3_Factor_prop')
		row = layout.row()
		row.prop(bpy.context.scene, 'spectral_multi_3_Tint_prop')



		layout.label(text='Spectral Multiblend :')
		row = layout.row()
		row.prop(bpy.context.scene, 'spectral_multiblend_equation_enum_prop', text="")

		row = layout.row()
		row.scale_y = 2.0 ###
		row.operator('shader.abj_shader_debugger_spectralmultiblend_operator')

		layout.label(text='Gradient Text Rxyz')
		row = layout.row()
		row.prop(bpy.context.scene, 'text_gradient_rotate_x_prop')
		row.prop(bpy.context.scene, 'text_gradient_rotate_y_prop')
		row.prop(bpy.context.scene, 'text_gradient_rotate_z_prop')

		######################################
		###### BREAKPOINTS
		######################################
		layout.label(text='BREAKPOINTS')

		row = layout.row()
		row.scale_y = 1.0 ###
		row.prop(bpy.context.scene, 'breakpoint_override_enum_prop', text="")

		row = layout.row()
		# row.scale_y = 1.0 ###
		row.prop(bpy.context.scene, 'breakpoint_000_enum_prop', text="")
		row = layout.row()
		row.prop(bpy.context.scene, 'breakpoint_001_enum_prop', text="")
		row = layout.row()
		row.prop(bpy.context.scene, 'breakpoint_002_enum_prop', text="")
		row = layout.row()
		row.prop(bpy.context.scene, 'breakpoint_003_enum_prop', text="")
		row = layout.row()
		row.prop(bpy.context.scene, 'breakpoint_004_enum_prop', text="")
		row = layout.row()
		row.prop(bpy.context.scene, 'breakpoint_005_enum_prop', text="")
		row = layout.row()
		row.prop(bpy.context.scene, 'breakpoint_006_enum_prop', text="")
		row = layout.row()
		row.prop(bpy.context.scene, 'breakpoint_007_enum_prop', text="")
		row = layout.row()
		row.prop(bpy.context.scene, 'breakpoint_008_enum_prop', text="")
		row = layout.row()
		row.prop(bpy.context.scene, 'breakpoint_009_enum_prop', text="")
		row = layout.row()
		row.prop(bpy.context.scene, 'breakpoint_010_enum_prop', text="")
		row = layout.row()
		row.prop(bpy.context.scene, 'breakpoint_011_enum_prop', text="")
		row = layout.row()
		row.prop(bpy.context.scene, 'breakpoint_012_enum_prop', text="")
		row = layout.row()
		row.prop(bpy.context.scene, 'breakpoint_013_enum_prop', text="")
		row = layout.row()
		row.prop(bpy.context.scene, 'breakpoint_014_enum_prop', text="")
		row = layout.row()
		row.prop(bpy.context.scene, 'breakpoint_015_enum_prop', text="")
		row = layout.row()
		row.prop(bpy.context.scene, 'breakpoint_016_enum_prop', text="")
		row = layout.row()
		row.prop(bpy.context.scene, 'breakpoint_017_enum_prop', text="")
		row = layout.row()
		row.prop(bpy.context.scene, 'breakpoint_018_enum_prop', text="")
		row = layout.row()
		row.prop(bpy.context.scene, 'breakpoint_019_enum_prop', text="")
		row = layout.row()
		row.prop(bpy.context.scene, 'breakpoint_020_enum_prop', text="")
		row = layout.row()
		row.prop(bpy.context.scene, 'breakpoint_021_enum_prop', text="")
		row = layout.row()
		row.prop(bpy.context.scene, 'breakpoint_022_enum_prop', text="")
		row = layout.row()
		row.prop(bpy.context.scene, 'breakpoint_023_enum_prop', text="")
		row = layout.row()
		row.prop(bpy.context.scene, 'breakpoint_024_enum_prop', text="")
		row = layout.row()
		row.prop(bpy.context.scene, 'breakpoint_025_enum_prop', text="")

##########
#####
	
class SHADER_OT_RANDOMLIGHT(bpy.types.Operator):
	# if you create an operator class called MYSTUFF_OT_super_operator, the bl_idname should be mystuff.super_operator

	bl_label = 'rand L'
	bl_idname = 'shader.abj_shader_debugger_randomlight_operator'

	def execute(self, context):
		myABJ_SD_B.randomLight_UI()
		return {'FINISHED'}

class SHADER_OT_RANDOMROTATION(bpy.types.Operator):
	# if you create an operator class called MYSTUFF_OT_super_operator, the bl_idname should be mystuff.super_operator

	bl_label = 'rand Rxyz'
	bl_idname = 'shader.abj_shader_debugger_randomrotation_operator'

	def execute(self, context):
		myABJ_SD_B.randomRotation_UI()
		return {'FINISHED'}

class SHADER_OT_RESTORELIGHT(bpy.types.Operator):
	# if you create an operator class called MYSTUFF_OT_super_operator, the bl_idname should be mystuff.super_operator

	bl_label = 'restore L'
	bl_idname = 'shader.abj_shader_debugger_restorelight_operator'

	def execute(self, context):
		myABJ_SD_B.restoreLight_UI()
		return {'FINISHED'}
	
class SHADER_OT_RESTORERXYZ(bpy.types.Operator):
	# if you create an operator class called MYSTUFF_OT_super_operator, the bl_idname should be mystuff.super_operator

	bl_label = 'restore Rxyz'
	bl_idname = 'shader.abj_shader_debugger_restorerxyz_operator'

	def execute(self, context):
		myABJ_SD_B.restoreRxyz_UI()
		return {'FINISHED'}

class SHADER_OT_STATICSTAGE1(bpy.types.Operator):
	# if you create an operator class called MYSTUFF_OT_super_operator, the bl_idname should be mystuff.super_operator

	bl_label = 'stage 1'
	bl_idname = 'shader.abj_shader_debugger_staticstage1_operator'

	def execute(self, context):
		myABJ_SD_B.static_debugOnly_Stage1_UI()
		return {'FINISHED'}
	
class SHADER_OT_SPECTRAL_COMPOSITOR(bpy.types.Operator):
	# if you create an operator class called MYSTUFF_OT_super_operator, the bl_idname should be mystuff.super_operator

	bl_label = 'spectral compositor'
	bl_idname = 'shader.abj_shader_debugger_spectral_compositor_operator'

	def execute(self, context):
		myABJ_SD_B.spectral_compositor()
		return {'FINISHED'}
	

class SHADER_OT_COMPOSITOR_STOCK(bpy.types.Operator):
	# if you create an operator class called MYSTUFF_OT_super_operator, the bl_idname should be mystuff.super_operator

	bl_label = 'compositor stock'
	bl_idname = 'shader.abj_shader_debugger_compositor_stock_operator'

	def execute(self, context):
		myABJ_SD_B.spectral_compositor_stock()
		return {'FINISHED'}

class SHADER_OT_REFRESHSTAGE2(bpy.types.Operator):
	# if you create an operator class called MYSTUFF_OT_super_operator, the bl_idname should be mystuff.super_operator
	bl_label = 'refresh stage 2'
	bl_idname = 'shader.abj_shader_debugger_refreshstage2_operator'

	def execute(self, context):
		myABJ_SD_B.refreshPart2_UI()
		return {'FINISHED'}
	
class SHADER_OT_GRADIENTCOLOR(bpy.types.Operator):
	# if you create an operator class called MYSTUFF_OT_super_operator, the bl_idname should be mystuff.super_operator

	bl_label = 'gradient color'
	bl_idname = 'shader.abj_shader_debugger_gradientcolor_operator'

	def execute(self, context):
		myABJ_SD_B.printColorGradient()
		return {'FINISHED'}
	
class SHADER_OT_GRADIENTCOLORWHEEL(bpy.types.Operator):
	# if you create an operator class called MYSTUFF_OT_super_operator, the bl_idname should be mystuff.super_operator

	bl_label = 'gradient color wheel'
	bl_idname = 'shader.abj_shader_debugger_gradientcolorwheel_operator'

	def execute(self, context):
		myABJ_SD_B.printColorGradient_circular()
		return {'FINISHED'}
	
class SHADER_OT_SPECTRALMULTIBLEND(bpy.types.Operator):
	# if you create an operator class called MYSTUFF_OT_super_operator, the bl_idname should be mystuff.super_operator

	bl_label = 'spectral multi blend'
	bl_idname = 'shader.abj_shader_debugger_spectralmultiblend_operator'

	def execute(self, context):
		myABJ_SD_B.abj_spectral_multiblend()
		return {'FINISHED'}
	
class SHADER_OT_PRESET180(bpy.types.Operator):
	# if you create an operator class called MYSTUFF_OT_super_operator, the bl_idname should be mystuff.super_operator

	bl_label = 'color wheel preset 18 0'
	bl_idname = 'shader.abj_shader_debugger_preset18_0_operator'

	def execute(self, context):
		myABJ_SD_B.colorGradient_circular_preset18_0()
		return {'FINISHED'}
	
class SHADER_OT_PRESET181(bpy.types.Operator):
	# if you create an operator class called MYSTUFF_OT_super_operator, the bl_idname should be mystuff.super_operator

	bl_label = 'color wheel preset 18 1'
	bl_idname = 'shader.abj_shader_debugger_preset18_1_operator'

	def execute(self, context):
		myABJ_SD_B.colorGradient_circular_preset18_1()
		return {'FINISHED'}

class SHADER_OT_RENDERPASSES(bpy.types.Operator):
	# if you create an operator class called MYSTUFF_OT_super_operator, the bl_idname should be mystuff.super_operator

	bl_label = 'render passes'
	bl_idname = 'shader.abj_shader_debugger_renderpasses_operator'

	def execute(self, context):
		myABJ_SD_B.renderPasses()
		return {'FINISHED'}

##############
# Written
##############
class SHADER_OT_WRITTENRENDER(bpy.types.Operator):
	# if you create an operator class called MYSTUFF_OT_super_operator, the bl_idname should be mystuff.super_operator

	bl_label = 'written render'
	bl_idname = 'shader.abj_shader_debugger_writtenrender_operator'

	def execute(self, context):
		myABJ_SD_B.written_render()
		return {'FINISHED'}

##############
# Stage Index
##############
class SHADER_OT_STAGESSELECTFACES(bpy.types.Operator):
	bl_label = 'select face(s) +'
	bl_idname = 'shader.abj_shader_debugger_stages_selectfaces_operator'

	def execute(self, context):
		myABJ_SD_B.stages_selectfaces_UI()
		return {'FINISHED'}

class SHADER_OT_STAGEIDXMINUS(bpy.types.Operator):
	bl_label = 'stage -'
	bl_idname = 'shader.abj_shader_debugger_stageidxminus_operator'

	def execute(self, context):
		myABJ_SD_B.stageIdx_plusMinus_UI(-1)
		return {'FINISHED'}

class SHADER_OT_STAGEIDXPLUS(bpy.types.Operator):
	bl_label = 'stage +'
	bl_idname = 'shader.abj_shader_debugger_stageidxplus_operator'

	def execute(self, context):
		myABJ_SD_B.stageIdx_plusMinus_UI(1)
		return {'FINISHED'}

class SHADER_OT_STAGEIDXZERO(bpy.types.Operator):
	bl_label = 'stage 0'
	bl_idname = 'shader.abj_shader_debugger_stageidxzero_operator'

	def execute(self, context):
		myABJ_SD_B.stageIdx_zero_UI()
		return {'FINISHED'}

class SHADER_OT_STAGERESETALL(bpy.types.Operator):
	bl_label = 'clear selected faces'
	bl_idname = 'shader.abj_shader_debugger_stageresetall_operator'

	def execute(self, context):
		myABJ_SD_B.stage_resetAll_UI()
		return {'FINISHED'}

class SHADER_OT_STAGEIDXPRINT(bpy.types.Operator):
	bl_label = 'print stage info'
	bl_idname = 'shader.abj_shader_debugger_stageidxprint_operator'

	def execute(self, context):
		myABJ_SD_B.stageIdx_print_UI()
		return {'FINISHED'}

###############
class SHADER_OT_SHOWHIDETEXTTOGGLE(bpy.types.Operator):
	bl_label = 'tgl text'
	bl_idname = 'shader.abj_shader_debugger_showhidetext_operator'

	def execute(self, context):
		myABJ_SD_B.showhideText_UI()
		return {'FINISHED'}

class SHADER_OT_SHOWHIDEARROWTOGGLE(bpy.types.Operator):
	bl_label = 'tgl arrow'
	bl_idname = 'shader.abj_shader_debugger_showhidearrow_operator'

	def execute(self, context):
		myABJ_SD_B.showhideArrows_UI()
		return {'FINISHED'}

class SHADER_OT_SHOWHIDECUBECAM(bpy.types.Operator):
	bl_label = 'tgl cam'
	bl_idname = 'shader.abj_shader_debugger_showhidecubecam_operator'

	def execute(self, context):
		myABJ_SD_B.showhideCubeCam_UI()
		return {'FINISHED'}

class SHADER_OT_TOGGLEEXTRAS(bpy.types.Operator):
	bl_label = 'tgl extras'
	bl_idname = 'shader.abj_shader_debugger_toggleextras_operator'

	def execute(self, context):
		myABJ_SD_B.toggleExtras_UI()
		return {'FINISHED'}

##########################################
############# COLOR PRESETS ################
##########################################
class SHADER_OT_AGXCOLORSETTINGS(bpy.types.Operator):
	bl_label = 'agx'
	bl_idname = 'shader.abj_shader_debugger_agx_color_operator'

	def execute(self, context):
		myABJ_SD_B.agxColorSettings_UI()
		return {'FINISHED'}
	
class SHADER_OT_TEXTCOLORSETTINGS(bpy.types.Operator):
	bl_label = 'text'
	bl_idname = 'shader.abj_shader_debugger_text_color_operator'

	def execute(self, context):
		myABJ_SD_B.textColorSettings_UI()
		return {'FINISHED'}

class SHADER_OT_STEREOSCOPICCOLORSETTINGS(bpy.types.Operator):
	bl_label = 'stereo'
	bl_idname = 'shader.abj_shader_debugger_stereoscopic_color_operator'

	def execute(self, context):
		myABJ_SD_B.stereoscopicColorSettings_UI()
		return {'FINISHED'}

class SHADER_OT_DEFAULTCOLORSETTINGS(bpy.types.Operator):
	bl_label = 'default'
	bl_idname = 'shader.abj_shader_debugger_default_color_operator'

	def execute(self, context):
		myABJ_SD_B.defaultColorSettings_UI()
		return {'FINISHED'}

#################
class SHADER_OT_RESTORECAMVIEW(bpy.types.Operator):
	bl_label = 'restoreCameraView'
	bl_idname = 'shader.abj_shader_debugger_restorecameraview_operator'

	def execute(self, context):
		myABJ_SD_B.restoreCameraView_UI()
		return {'FINISHED'}
	

myABJ_SD_B = ABJ_Shader_Debugger() ######################
