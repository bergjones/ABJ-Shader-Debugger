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
		self.useRestoredRxyzValues = True
		self.breakEarlyForRandomLightAndRxyz = False

		self.chosen_diffuse_equation = 'oren'
		# self.chosen_specular_equation = 'simple'
		self.chosen_specular_equation = 'GGX'


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



		self.changedSpecularEquation_variables = False
		self.changedDiffuseEquation_variables = False
		self.skip_showing_visibility_raycast_check = True

		self.Ci_render_temp_list = []
		self.selectedFaceMat_temp_list = []

		self.runOnce_part2_preProcess = False
		self.debugV_223 = None
		self.myDebugFaces = []
		self.allNamesToToggleDuringRaycast = []
		self.arrow_dynamic_instance_M_all_list_matrixOnly = []
		self.shadingDict_global = {}
		self.shadingList_perFace = []
		self.shadingStages_all = []
		self.shadingStages_perFace_stepList = []
		self.shadingStages_selectedFaces = []
		self.textRef_all = []
		self.specTesterMatToggle = -1
		self.objectsToToggleOnOffLater = []
		self.debugStageIterPlusMinus = False
		self.recently_cleared_selFaces = False

		self.aov_stored = None
		self.rdotvpow_stored = None
		self.oren_roughness_stored = None
		self.ggx_roughness_stored = None
		self.ggx_fresnel_stored = None

		self.breakpointsOverrideToggle = False
		self.skip_refresh_override_aov = False
		self.skip_refresh_override_RdotVpow = False
		self.skip_refresh_override_oren_roughness = False
		self.skip_refresh_override_GGX_roughness = False
		self.skip_refresh_override_GGX_fresnel = False

		self.shadingPlane_sel_r = 0.0
		self.shadingPlane_sel_g = 0.0
		self.shadingPlane_sel_b = 1.0

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
	
	def stageIdx_print_UI(self):
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

		nodetree = bpy.context.scene.node_tree

		# clear default nodes
		for node in nodetree.nodes:
			nodetree.nodes.remove(node)

		# adding glare node
		node0 = nodetree.nodes.new("CompositorNodeRLayers")
		node0.location = (0,0)

		node1 = nodetree.nodes.new("CompositorNodeGlare")
		node1.location = (400,0)
		node1.glare_type = 'BLOOM'

		# adding compositor node
		node2 = nodetree.nodes.new("CompositorNodeComposite")
		node2.location = (800,0)

		# connecting nodes
		nodetree.links.new(node0.outputs["Image"],node1.inputs[0])
		nodetree.links.new(node1.outputs["Image"],node2.inputs[0])

		self.compositor_setup = True

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

	def DoIt_part1_preprocess(self):
		self.startTime_stage1 = datetime.now()

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

		self.runOnce_part2_preProcess = False

		aov_items = bpy.context.scene.bl_rna.properties['aov_enum_prop'].enum_items
		aov_id = aov_items[bpy.context.scene.aov_enum_prop].identifier
		self.aov_stored = aov_id

		rdotvpow_items = bpy.context.scene.bl_rna.properties['r_dot_v_pow_enum_prop'].enum_items
		rdotvpow_id = rdotvpow_items[bpy.context.scene.r_dot_v_pow_enum_prop].identifier
		self.rdotvpow_stored = rdotvpow_id

		val_oren_roughness_prop = bpy.context.scene.oren_roughness_prop
		self.oren_roughness_stored = val_oren_roughness_prop

		val_ggx_roughness_prop = bpy.context.scene.ggx_roughness_prop
		self.ggx_roughness_stored = val_ggx_roughness_prop

		val_ggx_fresnel_prop = bpy.context.scene.ggx_fresnel_prop
		self.ggx_fresnel_stored = val_ggx_fresnel_prop


		val_text_radius_0_prop = bpy.context.scene.text_radius_0_prop
		self.text_radius_0_stored = val_text_radius_0_prop

		val_text_radius_1_prop = bpy.context.scene.text_radius_1_prop
		self.text_radius_1_stored = val_text_radius_1_prop


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
		cam1_data = bpy.data.cameras.new('Camera')
		cam = bpy.data.objects.new('Camera', cam1_data)
		bpy.context.collection.objects.link(cam)

		###################################
		###### SET CAMERA POS / LOOK AT
		###################################
		self.myCam = bpy.data.objects["Camera"]

		self.myCam.location = self.pos_camera_global
		self.updateScene() # need

		self.look_at(self.myCam, self.myOrigin)

		self.myV = self.myCam.matrix_world.to_translation()
		self.myV.normalize()

		###################################
		###### INPUT MESH / XFORM ###########
		###################################
		usablePrimitiveType_items = bpy.context.scene.bl_rna.properties['primitive_enum_prop'].enum_items
		usablePrimitiveType_id = usablePrimitiveType_items[bpy.context.scene.primitive_enum_prop].identifier

		if usablePrimitiveType_id == 'cube':
			bpy.ops.mesh.primitive_cube_add()

		elif usablePrimitiveType_id == 'uv_sphere':
			bpy.ops.mesh.primitive_uv_sphere_add()

		elif usablePrimitiveType_id == 'ico_sphere':
			bpy.ops.mesh.primitive_ico_sphere_add()

		elif usablePrimitiveType_id == 'cylinder':
			bpy.ops.mesh.primitive_cylinder_add()

		elif usablePrimitiveType_id == 'cone':
			bpy.ops.mesh.primitive_cone_add()

		elif usablePrimitiveType_id == 'torus':
			bpy.ops.mesh.primitive_torus_add()

		elif usablePrimitiveType_id == 'monkey':
			bpy.ops.mesh.primitive_monkey_add()

		myInputMesh = bpy.context.active_object
		myInputMesh.select_set(1)

		# bpy.context.view_layer.objects.active = myInputMesh

		#TRIANGULATE
		# bpy.ops.object.modifier_add(type='TRIANGULATE')
		# bpy.ops.object.modifier_apply(modifier="Triangulate")

		####SUBDIVIDE
		usableSubDToggle_items = bpy.context.scene.bl_rna.properties['subdivision_toggle_enum_prop'].enum_items
		usablePrimitiveType_id = usableSubDToggle_items[bpy.context.scene.subdivision_toggle_enum_prop].identifier

		if usablePrimitiveType_id == 'subd_1':
			bpy.ops.object.modifier_add(type='SUBSURF')
			myObj = bpy.context.active_object
			myObj.modifiers["Subdivision"].levels = 1
			bpy.ops.object.modifier_apply(modifier="Subdivision")


		if self.useRestoredRxyzValues == True:
			bpy.ops.transform.rotate(value=math.radians(180), orient_axis='X', orient_type='GLOBAL')
			bpy.ops.transform.rotate(value=math.radians(180), orient_axis='Y', orient_type='GLOBAL')

		else:
			bpy.ops.transform.rotate(value=math.radians(180), orient_axis='X', orient_type='GLOBAL')
			bpy.ops.transform.rotate(value=math.radians(180), orient_axis='Y', orient_type='GLOBAL')
			bpy.ops.transform.rotate(value=math.radians(self.RandomRotationDegree), orient_axis=self.RandomRotationAxis, orient_type='GLOBAL')

		# bpy.ops.transform.rotate(value=math.radians(180), orient_axis='X', orient_type='GLOBAL')
		# # bpy.ops.transform.rotate(value=math.radians(180), orient_axis='Z', orient_type='GLOBAL')
		# # bpy.ops.transform.rotate(value=math.radians(0), orient_axis='Y', orient_type='GLOBAL')
		# bpy.ops.transform.rotate(value=math.radians(180), orient_axis='Y', orient_type='GLOBAL')
		# # bpy.ops.transform.rotate(value=math.radians(self.RandomRotationDegree), orient_axis=self.RandomRotationAxis, orient_type='GLOBAL')

		bpy.ops.object.transform_apply(location=1, rotation=1, scale=1)

		self.profile_stage1_06_b = str(datetime.now() - self.profile_stage1_06_a)
		if self.profileCode_part1 == True:
			print('~~~~~~~~~ self.profile_stage1_06_b = ', self.profile_stage1_06_b)

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
		self.mySun.name = "self.mySun"
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
		myR = -myL.reflect(myN)
		myH = (self.myV + myL).normalized()
		R_dot_V_control = max(self.myV.dot(myR), 0.0)
		N_dot_V = max(myN.dot(self.myV), 0.0)

		distance = (self.pos_light_global_v - pos).length
		attenuation = 1.0 / (distance * distance)

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

		self.changedSpecularEquation_variables = False
		self.changedDiffuseEquation_variables = False

		if self.rdotvpow_stored != rdotvpow_id:
			self.skip_refresh_override_RdotVpow = True
			self.rdotvpow_stored = rdotvpow_id
			self.changedSpecularEquation_variables = True

		if self.ggx_roughness_stored != val_ggx_roughness_prop:
			self.skip_refresh_override_GGX_roughness = True
			self.ggx_roughness_stored = val_ggx_fresnel_prop
			self.changedSpecularEquation_variables = True

		if self.ggx_fresnel_stored != val_ggx_fresnel_prop:
			self.skip_refresh_override_GGX_fresnel = True
			self.ggx_fresnel_stored = val_ggx_fresnel_prop
			self.changedSpecularEquation_variables = True

		usableSpecularEquationType_items = bpy.context.scene.bl_rna.properties['specular_equation_enum_prop'].enum_items
		usableSpecularEquationType_id = usableSpecularEquationType_items[bpy.context.scene.specular_equation_enum_prop].identifier

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
			self.skip_refresh_override_oren_roughness = False
			self.skip_refresh_override_GGX_roughness = False
			self.skip_refresh_override_GGX_fresnel = False

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
			V = self.myV

			faceCenter_to_V_rayCast = i['faceCenter_to_V_rayCast']
			faceCenter_to_L_rayCast = i['faceCenter_to_L_rayCast']

			if mySplitFaceIndexUsable in self.Ci_render_temp_list:
				finalDiffuse = 1

				if faceCenter_to_L_rayCast == False:
					finalDiffuse = 0

				if faceCenter_to_L_rayCast == True:
					if usableDiffuseEquationType_id == 'oren':
						finalDiffuse = self.oren(N_dot_L, V, L, N, N_dot_V, self.oren_roughness_stored)

					elif usableDiffuseEquationType_id == 'simple':
						finalDiffuse = N_dot_L

				self.final_Ci_output(aov_id, shadingPlane, mySplitFaceIndexUsable, finalDiffuse, spec, attenuation, faceCenter_to_V_rayCast, faceCenter_to_L_rayCast)

			elif mySplitFaceIndexUsable in self.selectedFaceMat_temp_list:
					self.setActiveStageMaterial(shadingPlane, mySplitFaceIndexUsable, self.shadingPlane_sel_r, self.shadingPlane_sel_g, self.shadingPlane_sel_b)

		# bpy.ops.object.mode_set(mode="OBJECT")

		#reset refresh override skips
		self.skip_refresh_override_RdotVpow = False
		self.skip_refresh_override_oren_roughness = False
		self.skip_refresh_override_GGX_roughness = False
		self.skip_refresh_override_GGX_fresnel = False

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

	def printGreyScaleGradient(self, steps):
		# self.test_gradientRows(0, 5)

		outputRatio = None
		allOutputRatios = []

		usableTextRGBPrecision_items = bpy.context.scene.bl_rna.properties['text_rgb_precision_enum_prop'].enum_items
		usableTextRGBPrecision_id = usableTextRGBPrecision_items[bpy.context.scene.text_rgb_precision_enum_prop].identifier

		precisionVal = int(usableTextRGBPrecision_id)

		if precisionVal == -1:
			# precisionVal = 5
			precisionVal = 3 # temp

		for i in range(steps + 1):
			if i == 0:
				lerpIter = 1

			elif i == steps:
				lerpIter = 0

			else:
				lerpIter = 1 - (i / steps)

			lerpIter = round(lerpIter, precisionVal)

			allOutputRatios.append(lerpIter)

		self.makeGradientGrid(allOutputRatios)

		self.defaultColorSettings_UI()

		# print(allOutputRatios)
		# reversed_allOutputRatios = list(reversed(allOutputRatios))
		# print(reversed_allOutputRatios)

	#This method calls a function from spectral.py, under MIT license (see file)
	def printColorGradient(self):

		usableTextRGBPrecision_items = bpy.context.scene.bl_rna.properties['text_rgb_precision_enum_prop'].enum_items
		usableTextRGBPrecision_id = usableTextRGBPrecision_items[bpy.context.scene.text_rgb_precision_enum_prop].identifier

		precisionVal = int(usableTextRGBPrecision_id)
		usableSteps = None

		if precisionVal == -1:
			usableSteps = 100

		elif precisionVal == 1:
			usableSteps = 10

		elif precisionVal == 2:
			usableSteps = 100

		elif precisionVal == 3:
			# usableSteps = 1000
			usableSteps = 100

		steps = 100
		steps = usableSteps
		# steps = 25

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

			if usableAdditiveOrSubtractiveColorBlending_id == 'additive':
				outputRatio_x = self.lerp(endColor[0], startColor[0], lerpIter)
				outputRatio_y = self.lerp(endColor[1], startColor[1], lerpIter)
				outputRatio_z = self.lerp(endColor[2], startColor[2], lerpIter)

				comboRatio_xyz = mathutils.Vector((outputRatio_x, outputRatio_y, outputRatio_z))

				gammaCorrect = mathutils.Vector((2.2, 2.2, 2.2))
				gammaCorrect_r = pow(comboRatio_xyz.x, gammaCorrect.x)
				gammaCorrect_g = pow(comboRatio_xyz.y, gammaCorrect.y)
				gammaCorrect_b = pow(comboRatio_xyz.z, gammaCorrect.z)

				comboRatio_xyz = mathutils.Vector((gammaCorrect_r, gammaCorrect_g, gammaCorrect_b))

				# comboRatio_xyz = mathutils.Vector((outputRatio_x, outputRatio_y, outputRatio_z))

			elif usableAdditiveOrSubtractiveColorBlending_id == 'subtractive':
				outputRatio_x = spectral.spectral_mix(endColor, startColor, lerpIter)
				comboRatio_xyz = mathutils.Vector((outputRatio_x[0] / 255, outputRatio_x[1] / 255, outputRatio_x[2] / 255))

				# gammaCorrect = mathutils.Vector((1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2))
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

	def colorWheel_dynamic_inner(self, i, segments, center_x, center_y, lerpIter_outer, gradient_inner_circle_steps, myInputMesh, startColor, endColor, endColor_black):
		outputRatio_x = self.lerp(endColor.x, startColor.x, lerpIter_outer)
		outputRatio_y = self.lerp(endColor.y, startColor.y, lerpIter_outer)
		outputRatio_z = self.lerp(endColor.z, startColor.z, lerpIter_outer)

		gradient_inner_circle_steps_usable = gradient_inner_circle_steps - 1

		for j in range(gradient_inner_circle_steps):
			finalOutputColors = []

			lerpIter_inner = None

			if j == 0:
				lerpIter_inner = 1

			elif j == gradient_inner_circle_steps:
				lerpIter_inner = 0

			else:
				lerpIter_inner = 1 - (j / gradient_inner_circle_steps_usable)

			angle = 2 * math.pi * i / segments
			# x = radius * math.cos(angle) + center_x
			# y = radius * math.sin(angle) + center_y

			# radius = 2.125
			radius = 2
			lerpIter_inner_radius = radius * lerpIter_inner

			x = lerpIter_inner_radius * math.cos(angle) + center_x
			y = lerpIter_inner_radius * math.sin(angle) + center_y

			# x = radius * lerpIter_inner * math.cos(angle) + center_x
			# y = radius * lerpIter_inner * math.sin(angle) + center_y

			outputRatio_x_01 = self.lerp(endColor_black.x, outputRatio_x, lerpIter_inner)
			outputRatio_y_01 = self.lerp(endColor_black.y, outputRatio_y, lerpIter_inner)
			outputRatio_z_01 = self.lerp(endColor_black.z, outputRatio_z, lerpIter_inner)

			# gammaCorrect = mathutils.Vector((2.2, 2.2, 2.2))
			# outputRatio_x_01 = pow(outputRatio_x_01, gammaCorrect.x)
			# outputRatio_y_01 = pow(outputRatio_y_01, gammaCorrect.y)
			# outputRatio_z_01 = pow(outputRatio_z_01, gammaCorrect.z)


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

			self.makeGradientGrid_color_circular(finalOutputColors, x, y, myInputMesh, lerpIter_inner, textRaiseLowerZ)

	def printColorGradient_circular(self):

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
		mat1 = self.newShader("ShaderVisualizer_gradientBG", "emission", 0.5, 0.5, 0.5)
		bpy.context.active_object.data.materials.clear()
		bpy.context.active_object.data.materials.append(mat1)

		# Define circle parameters
		center_x = 0
		center_y = 0

		val_gradient_outer_circle_steps_prop = bpy.context.scene.gradient_outer_circle_steps_prop
		val_gradient_inner_circle_steps_prop = bpy.context.scene.gradient_inner_circle_steps_prop

		# val_gradient_outer_circle_steps_prop = 10
		# val_gradient_outer_circle_steps_prop = 5

		# val_gradient_inner_circle_steps_prop = 25
		# val_gradient_inner_circle_steps_prop = 10
		# val_gradient_inner_circle_steps_prop = 5

		divisor = 7
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
			# Indigo (0.3, 0, 0.7), 
			# Violet (0.5, 0, 0.5)

			lerpIter_outer = countToDivisorMultiplier_list[i]

			if 0 <= i < (segments / divisor): 
				#RED TO ORANGE
				startColor = mathutils.Vector((1.0, 0.0, 0.0))
				# endColor = mathutils.Vector((1.0, 0.5, 0.0))
				endColor = mathutils.Vector((1 - (1.0 / 7.0), 0.5 - (1.0 / 7.0), 0.0))
				endColor_black = mathutils.Vector((0.0, 0.0, 0.0))

				self.colorWheel_dynamic_inner(i, segments, center_x, center_y, lerpIter_outer, val_gradient_inner_circle_steps_prop, myInputMesh, startColor, endColor, endColor_black)

			elif (segments / divisor) <= i < (segments / (divisor / 2)):
				# print('2 divisor : ', i, ' lerpIter01 ', lerpIter)

				#ORANGE TO YELLOW
				startColor = mathutils.Vector((1.0, 0.5, 0.0))
				# endColor = mathutils.Vector((1.0, 1.0, 0.0))
				endColor = mathutils.Vector((1 - (1.0 / 7.0), 1 - (1.0 / 7.0), 0.0))
				endColor_black = mathutils.Vector((0.0, 0.0, 0.0))

				self.colorWheel_dynamic_inner(i, segments, center_x, center_y, lerpIter_outer, val_gradient_inner_circle_steps_prop, myInputMesh, startColor, endColor, endColor_black)

			elif (segments / (divisor / 2)) <= i < (segments / (divisor / 3)):
				# print('3 divisor : ', i, ' lerpIter01 ', lerpIter)

				#YELLOW TO GREEN
				startColor = mathutils.Vector((1.0, 1.0, 0.0))
				# endColor = mathutils.Vector((0.0, 1.0, 0.0))
				endColor = mathutils.Vector((0, 1 - (1.0 / 7.0), 0.0))
				endColor_black = mathutils.Vector((0.0, 0.0, 0.0))

				self.colorWheel_dynamic_inner(i, segments, center_x, center_y, lerpIter_outer, val_gradient_inner_circle_steps_prop, myInputMesh, startColor, endColor, endColor_black)

			elif (segments / (divisor / 3)) <= i < (segments / (divisor / 4)):
				# print('4 divisor : ', i)

				#GREEN TO BLUE
				startColor = mathutils.Vector((0.0, 1.0, 0.0))
				# endColor = mathutils.Vector((0.0, 0.0, 1.0))
				endColor = mathutils.Vector((0, 0, 1 - (1.0 / 7.0)))
				endColor_black = mathutils.Vector((0.0, 0.0, 0.0))

				self.colorWheel_dynamic_inner(i, segments, center_x, center_y, lerpIter_outer, val_gradient_inner_circle_steps_prop, myInputMesh, startColor, endColor, endColor_black)

			elif (segments / (divisor / 4)) <= i < (segments / (divisor / 5)):
				# print('5 divisor : ', i)

				#BLUE TO INDIGO
				startColor = mathutils.Vector((0.0, 0.0, 1.0))
				# endColor = mathutils.Vector((0.3, 0.0, 0.7))
				endColor = mathutils.Vector((0.3 - (1.0 / 7.0), 0, 0.7 - (1.0 / 7.0)))
				endColor_black = mathutils.Vector((0.0, 0.0, 0.0))

				self.colorWheel_dynamic_inner(i, segments, center_x, center_y, lerpIter_outer, val_gradient_inner_circle_steps_prop, myInputMesh, startColor, endColor, endColor_black)

			elif (segments / (divisor / 5)) <= i < (segments / (divisor / 6)):
				# print('6 divisor : ', i)

				#INDIGO TO VIOLET
				startColor = mathutils.Vector((0.3, 0.0, 0.7))
				# endColor = mathutils.Vector((0.5, 0.0, 0.5))
				endColor = mathutils.Vector((0.5 - (1.0 / 7.0), 0.0, 0.5 - (1.0 / 7.0)))
				endColor_black = mathutils.Vector((0.0, 0.0, 0.0))

				self.colorWheel_dynamic_inner(i, segments, center_x, center_y, lerpIter_outer, val_gradient_inner_circle_steps_prop, myInputMesh, startColor, endColor, endColor_black)

			elif (segments / (divisor / 6)) <= i < (segments / (divisor / 7)):
				# print('7 divisor : ', i)

				#VIOLET TO RED
				startColor = mathutils.Vector((0.5, 0.0, 0.5))
				# endColor = mathutils.Vector((1.0, 0.0, 0.0))
				endColor = mathutils.Vector((1 - (1.0 / 7.0), 0.0, 0.0))
				endColor_black = mathutils.Vector((0.0, 0.0, 0.0))

				self.colorWheel_dynamic_inner(i, segments, center_x, center_y, lerpIter_outer, val_gradient_inner_circle_steps_prop, myInputMesh, startColor, endColor, endColor_black)

		self.defaultColorSettings_UI()

		myInputMesh.hide_render = True

		self.deselectAll()

		myDupeGradient_bg.select_set(0)

	def makeGradientGrid_color_circular(self, gradientArray, xPos, yPos, myInputMesh, lerpIter_inner, textRaiseLowerZ):
		#####################
		### each gradient face
		#####################
		# gradientScale = 0.175 ######
		# gradientScale = 0.1
		# gradientScale = 0.2
		gradientScale = 0.05
		# gradientScale = 0.03
		# raiseLowerZ = 1
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
			myDupeGradient.name = 'dupeGradient_' + str(idx)

			# myDupeGradient.scale = mathutils.Vector((gradientScale, gradientScale, gradientScale)) #####
			gradientScale_lerped = self.clamp(gradientScale * lerpIter_inner, 0.01, 1)
			myDupeGradient.scale = mathutils.Vector((gradientScale_lerped, gradientScale_lerped, gradientScale_lerped))

			gradient_startPos = mathutils.Vector((0, xPos, raiseLowerZ + yPos))
			myDupeGradient.location = gradient_startPos
	
			myDupeGradient.rotation_euler = mathutils.Vector((0, math.radians(90), 0))

			Ci_gc = mathutils.Vector((i))

			# if precisionVal == -1:
			# 	pass
			# else:
			# 	Ci_gc = mathutils.Vector((round(Ci_gc.x, precisionVal), round(Ci_gc.y, precisionVal), round(Ci_gc.z, precisionVal)))	

			bpy.context.view_layer.objects.active = myDupeGradient
			mat1 = self.newShader("ShaderVisualizer_gradient_" + str(i), "emission", Ci_gc.x, Ci_gc.y, Ci_gc.z)
			bpy.context.active_object.data.materials.clear()
			bpy.context.active_object.data.materials.append(mat1)

			#####################
			### text_add() (better text placement)
			#####################
			if precisionVal != -1:
				if lerpIter_inner >= (.25):

					# precisionVal = 3
					t = '(' + str(round(Ci_gc.x, precisionVal)) + ', ' + str(round(Ci_gc.y, precisionVal)) + ', ' + str(round(Ci_gc.z, precisionVal)) + ')'

					myFontCurve = bpy.data.curves.new(type="FONT", name="myFontCurve")
					myFontCurve.body = t

					myFontOb = bpy.data.objects.new(myDupeGradient.name + '_text', myFontCurve)
					myFontOb.data.align_x = 'CENTER'
					myFontOb.data.align_y = 'CENTER'

					myFontOb.location = myDupeGradient.location + mathutils.Vector((1, 0, textRaiseLowerZ))
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

					myFontScale = 0.03
					# myFontScale = 0.03 * lerpIter_inner

					# myFontOb_scale_clamped = self.clamp(myFontScale, 0.02, 1)
					# myFontOb_scale_clamped = self.clamp(myFontScale, 0.02, 1)

					myFontOb.scale = mathutils.Vector((myFontScale, myFontScale, myFontScale))
					# myFontOb.scale = mathutils.Vector((myFontOb_scale_clamped, myFontOb_scale_clamped, myFontOb_scale_clamped))

					myFontOb.data.body = t

					bpy.context.collection.objects.link(myFontOb)

					myFontOb.show_in_front = True

					self.textRef_all.append(myFontOb.name)

					bpy.context.view_layer.objects.active = myFontOb

					startColor = mathutils.Vector((0.0, 0.0, 0.0))
					endColor = mathutils.Vector((1.0, 1.0, 1.0))
					mat_output_color_x = self.lerp(endColor.x, startColor.x, lerpIter_inner)
					mat_output_color_y = self.lerp(endColor.y, startColor.y, lerpIter_inner)
					mat_output_color_z = self.lerp(endColor.z, startColor.z, lerpIter_inner)

					# if lerpIter_inner <= (3/10):
					# 	combo_mat_output_color = mathutils.Vector((1, 1, 1))

					# else:
					# 	combo_mat_output_color = mathutils.Vector((0, 0, 0))

					combo_mat_output_color = mathutils.Vector((0, 0, 0))
					
					# combo_mat_output_color = mathutils.Vector((mat_output_color_x, mat_output_color_y, mat_output_color_z))

					mat1 = self.newShader("ShaderVisualizer_gradient_text_" + str(i), "emission", combo_mat_output_color.x, combo_mat_output_color.y, combo_mat_output_color.z)
					# mat1 = self.newShader("ShaderVisualizer_gradient_text_" + str(i), "emission", 0, 0, 0)
					# mat1 = self.newShader("ShaderVisualizer_gradient_text_" + str(i), "emission", 1, 1, 0)
					bpy.context.active_object.data.materials.clear()
					bpy.context.active_object.data.materials.append(mat1)

					bpy.context.view_layer.objects.active = myDupeGradient

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
		mat1 = self.newShader("ShaderVisualizer_gradientBG", "emission", 0.5, 0.5, 0.5)
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

			# gradient_startPos = mathutils.Vector((0, -2.5, 1)) #top left
			# gradient_startPos = mathutils.Vector((0, -2.6, 1)) #top left
			# gradient_startPos = mathutils.Vector((0, -2.6, .9)) #top left
			# gradient_startPos = mathutils.Vector((0, -2.75, 1)) #top left
			# gradient_startPos = mathutils.Vector((0, -2.6, 1.1)) #top left
			gradient_startPos = mathutils.Vector((0, -2.8, 1.1)) #top left

			myDupeGradient.location = gradient_startPos + mathutils.Vector((0, output_Y * locationMultiplierY, raiseLowerZ + (locationMultiplierZ * usable_Z_Row)))

			myDupeGradient.rotation_euler = mathutils.Vector((0, math.radians(90), 0))





			# Ci_gc = mathutils.Vector((i, i, i))
			Ci_gc = mathutils.Vector((i))





			# gammaCorrect = mathutils.Vector((1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2))
			# # gammaCorrect = mathutils.Vector((2.2, 2.2, 2.2))
			# gammaCorrect_r = pow(Ci_gc.x, gammaCorrect.x)
			# gammaCorrect_g = pow(Ci_gc.y, gammaCorrect.y)
			# gammaCorrect_b = pow(Ci_gc.z, gammaCorrect.z)

			# Ci_gc_text = mathutils.Vector((gammaCorrect_r, gammaCorrect_g, gammaCorrect_b))












			# Ci_gc = mathutils.Vector((gradientArray[i]))
			# Ci_gc = mathutils.Vector((gradientArray[i].x, gradientArray[i].y, gradientArray[i].z))

			# if precisionVal == -1:
			# 	pass
			# else:
			# 	Ci_gc = mathutils.Vector((round(Ci_gc.x, precisionVal), round(Ci_gc.y, precisionVal), round(Ci_gc.z, precisionVal)))	

			bpy.context.view_layer.objects.active = myDupeGradient
			mat1 = self.newShader("ShaderVisualizer_gradient_" + str(i), "emission", Ci_gc.x, Ci_gc.y, Ci_gc.z)
			bpy.context.active_object.data.materials.clear()
			bpy.context.active_object.data.materials.append(mat1)

			#####################
			### text_add() (better text placement)
			#####################
			if precisionVal != -1:

				gammaCorrect = mathutils.Vector((1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2))
				# gammaCorrect = mathutils.Vector((2.2, 2.2, 2.2))
				gammaCorrect_r = pow(Ci_gc.x, gammaCorrect.x)
				gammaCorrect_g = pow(Ci_gc.y, gammaCorrect.y)
				gammaCorrect_b = pow(Ci_gc.z, gammaCorrect.z)

				Ci_gc_text = mathutils.Vector((gammaCorrect_r, gammaCorrect_g, gammaCorrect_b))




				# precisionVal = 3
				# t = '(' + str(round(Ci_gc.x, precisionVal)) + ', ' + str(round(Ci_gc.y, precisionVal)) + ', ' + str(round(Ci_gc.z, precisionVal)) + ')'
				t = '(' + str(round(Ci_gc_text.x, precisionVal)) + ', ' + str(round(Ci_gc_text.y, precisionVal)) + ', ' + str(round(Ci_gc_text.z, precisionVal)) + ')'

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


	def makeGradientGrid(self, gradientArray):

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
		mat1 = self.newShader("ShaderVisualizer_gradientBG", "emission", 0.5, 0.5, 0.5)
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

			# gradient_startPos = mathutils.Vector((0, -2.5, 1)) #top left
			# gradient_startPos = mathutils.Vector((0, -2.6, 1)) #top left
			# gradient_startPos = mathutils.Vector((0, -2.6, .9)) #top left
			# gradient_startPos = mathutils.Vector((0, -2.75, 1)) #top left
			# gradient_startPos = mathutils.Vector((0, -2.6, 1.1)) #top left
			gradient_startPos = mathutils.Vector((0, -2.8, 1.1)) #top left

			myDupeGradient.location = gradient_startPos + mathutils.Vector((0, output_Y * locationMultiplierY, raiseLowerZ + (locationMultiplierZ * usable_Z_Row)))

			myDupeGradient.rotation_euler = mathutils.Vector((0, math.radians(90), 0))

			Ci_gc = mathutils.Vector((i, i, i))

			# if precisionVal == -1:
			# 	pass
			# else:
			# 	Ci_gc = mathutils.Vector((round(Ci_gc.x, precisionVal), round(Ci_gc.y, precisionVal), round(Ci_gc.z, precisionVal)))	

			bpy.context.view_layer.objects.active = myDupeGradient
			mat1 = self.newShader("ShaderVisualizer_gradient_" + str(i), "emission", Ci_gc.x, Ci_gc.y, Ci_gc.z)
			bpy.context.active_object.data.materials.clear()
			bpy.context.active_object.data.materials.append(mat1)

			#####################
			### text_add() (better text placement)
			#####################
			if precisionVal != -1:
				# precisionVal = 3
				t = '(' + str(round(Ci_gc.x, precisionVal)) + ', ' + str(round(Ci_gc.y, precisionVal)) + ', ' + str(round(Ci_gc.z, precisionVal)) + ')'

				myFontCurve = bpy.data.curves.new(type="FONT", name="myFontCurve")
				myFontCurve.body = t

				myFontOb = bpy.data.objects.new(myDupeGradient.name + '_text', myFontCurve)
				myFontOb.data.align_x = 'CENTER'
				myFontOb.data.align_y = 'CENTER'

				# textRaiseLower = 0.15
				# textRaiseLower = 0.175
				# textRaiseLower = 0.2
				textRaiseLower = 0.12

				if idx % 2 == 0:
					#even
					textRaiseLower *= 1
				else:
					#odd
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
				bpy.context.active_object.data.materials.clear()
				bpy.context.active_object.data.materials.append(mat1)

				bpy.context.view_layer.objects.active = myDupeGradient

		myInputMesh.hide_render = True

	def gradient_UI(self):
		#####################
		##### GRADIENT
		#####################
		usableTextRGBPrecision_items = bpy.context.scene.bl_rna.properties['text_rgb_precision_enum_prop'].enum_items
		usableTextRGBPrecision_id = usableTextRGBPrecision_items[bpy.context.scene.text_rgb_precision_enum_prop].identifier

		precisionVal = int(usableTextRGBPrecision_id)
		usableSteps = None

		if precisionVal == -1:
			usableSteps = 100

		elif precisionVal == 1:
			usableSteps = 10

		elif precisionVal == 2:
			usableSteps = 100

		elif precisionVal == 3:
			# usableSteps = 1000
			usableSteps = 100

		self.printGreyScaleGradient(usableSteps)

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


	def final_Ci_output(self, aov_id, shadingPlane, mySplitFaceIndexUsable, inputDiffuse, spec, attenuation, faceCenter_to_V_rayCast, faceCenter_to_L_rayCast):
		attenuation = 1.0 #temporary, outside sunlight

		Ks = 1
		diff_Cs_V = mathutils.Vector((1.0, 0.0, 0.0))

		inputDiff_V = mathutils.Vector((inputDiffuse, inputDiffuse, inputDiffuse))

		finalSpec = spec * Ks
		finalSpec_V = mathutils.Vector((finalSpec, finalSpec, finalSpec))
		finalDiff_V = diff_Cs_V * inputDiff_V
		# finalDiff_V = diff_Cs_V

		ambientMultiplier = .0004
		ambient_V = mathutils.Vector((ambientMultiplier, ambientMultiplier, ambientMultiplier))

		Ci = None

		if aov_id == 'spec':
			Ci = ((finalSpec_V) * attenuation) ###
		elif aov_id == 'diffuse':
			Ci = ((finalDiff_V) * attenuation) ###
		elif aov_id == 'Ci':
			Ci = ((finalDiff_V + finalSpec_V) * attenuation) ###
			# Ci = ((finalDiff_V + finalSpec_V + (diff_Cs_V * ambient_V)) * attenuation) ###

		gammaCorrect = mathutils.Vector((1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2))
		gammaCorrect_r = pow(Ci.x, gammaCorrect.x)
		gammaCorrect_g = pow(Ci.y, gammaCorrect.y)
		gammaCorrect_b = pow(Ci.z, gammaCorrect.z)

		Ci_gc = mathutils.Vector((gammaCorrect_r, gammaCorrect_g, gammaCorrect_b))

		usableTextRGBPrecision_items = bpy.context.scene.bl_rna.properties['text_rgb_precision_enum_prop'].enum_items
		usableTextRGBPrecision_id = usableTextRGBPrecision_items[bpy.context.scene.text_rgb_precision_enum_prop].identifier

		precisionVal = int(usableTextRGBPrecision_id)

		if precisionVal == -1:
			pass

		else:
			Ci_gc = mathutils.Vector((round(Ci_gc.x, precisionVal), round(Ci_gc.y, precisionVal), round(Ci_gc.z, precisionVal)))



		val_text_rotate_x_prop = bpy.context.scene.text_rotate_x_prop
		val_text_rotate_y_prop = bpy.context.scene.text_rotate_y_prop
		val_text_rotate_z_prop = bpy.context.scene.text_rotate_z_prop



		# myRotation = self.myV * mathutils.Vector((0, 0, 180))
		# myRotation = self.myV * mathutils.Vector((0, 0, math.radians(180)))
		# myRotation = self.myV * mathutils.Vector((0, 0, math.radians(90)))



		myRotation = self.myV * mathutils.Vector((math.radians(val_text_rotate_x_prop), math.radians(val_text_rotate_y_prop), math.radians(val_text_rotate_z_prop)))
		
		if self.specTesterMatToggle == -1:
			for j in bpy.context.scene.objects:
				if j.name == shadingPlane:

					#####################
					### text_add() (better text placement)
					#####################
					if precisionVal != -1:
						# precisionVal = 3
						t = '(' + str(round(Ci_gc.x, precisionVal)) + ', ' + str(round(Ci_gc.y, precisionVal)) + ', ' + str(round(Ci_gc.z, precisionVal)) + ')'

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

						if Ci > mathutils.Vector((0, 0, 0)):
							myFontOb.data.body = t

						else:
							myFontOb.data.body = 'x'

						bpy.context.collection.objects.link(myFontOb)

						if faceCenter_to_V_rayCast == True:
							myFontOb.show_in_front = True

						self.textRef_all.append(myFontOb.name)

					bpy.context.view_layer.objects.active = j

			# mat1 = self.newShader("ShaderVisualizer_" + str(mySplitFaceIndexUsable), "emission", Ci.x, Ci.y, Ci.z)
			mat1 = self.newShader("ShaderVisualizer_" + str(mySplitFaceIndexUsable), "emission", Ci_gc.x, Ci_gc.y, Ci_gc.z)
			# mat1 = self.newShader("ShaderVisualizer_" + str(mySplitFaceIndexUsable), "emission", 1, 0, 0)
			bpy.context.active_object.data.materials.clear()
			bpy.context.active_object.data.materials.append(mat1)

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

		if self.skip_refresh_override_oren_roughness == True:
			skip_refresh = False

		if self.skip_refresh_override_GGX_roughness == True:
			skip_refresh = False

		if self.skip_refresh_override_GGX_fresnel == True:
			skip_refresh = False

		if skip_refresh_override_recently_cleared_faces == True:
			skip_refresh = False

		if self.changedSpecularEquation_variables == True:
			skip_refresh = False

		if self.changedDiffuseEquation_variables == True:
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

		arrow_instance = self.copyObject()

		# bpy.context.view_layer.objects.active = arrow_instance

		arrow_instance.name = namePrefix + str(idx)
		arrow_instance.hide_set(0)
		arrow_instance.matrix_world = matrix

		return arrow_instance

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

	def setActiveStageMaterial(self, shadingPlane, idx, r, g, b):
		if self.specTesterMatToggle == -1:
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

		######################################
		###### GRADIENT
		######################################
		row = layout.row()
		row.prop(bpy.context.scene, 'additive_or_subtractive_color_blending_enum_prop', text="")

		layout.label(text='Gradients')
		row = layout.row()
		row.scale_y = 2.0 ###
		row.operator('shader.abj_shader_debugger_gradient_operator')

		row = layout.row()
		row.scale_y = 2.0 ###
		row.operator('shader.abj_shader_debugger_gradientcolor_operator')

		row = layout.row()
		row.prop(bpy.context.scene, 'gradient_color0_prop')
		row = layout.row()
		row.prop(bpy.context.scene, 'gradient_color1_prop')



		layout.label(text='Color Wheel')
		row = layout.row()
		# row.prop(bpy.context.scene, 'gradient_outer_circle_steps_prop', text="")
		row.prop(bpy.context.scene, 'gradient_outer_circle_steps_prop')
		row = layout.row()
		row.prop(bpy.context.scene, 'gradient_inner_circle_steps_prop')
		# row.prop(bpy.context.scene, 'gradient_inner_circle_steps_prop', text="")
		row = layout.row()
		row.scale_y = 2.0 ###
		row.operator('shader.abj_shader_debugger_gradientcolorwheel_operator')

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

class SHADER_OT_REFRESHSTAGE2(bpy.types.Operator):
	# if you create an operator class called MYSTUFF_OT_super_operator, the bl_idname should be mystuff.super_operator
	bl_label = 'refresh stage 2'
	bl_idname = 'shader.abj_shader_debugger_refreshstage2_operator'

	def execute(self, context):
		myABJ_SD_B.refreshPart2_UI()
		return {'FINISHED'}
	
class SHADER_OT_GRADIENT(bpy.types.Operator):
	# if you create an operator class called MYSTUFF_OT_super_operator, the bl_idname should be mystuff.super_operator

	bl_label = 'gradient greyscale'
	bl_idname = 'shader.abj_shader_debugger_gradient_operator'

	def execute(self, context):
		myABJ_SD_B.gradient_UI()
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
	

# importlib.reload(splitABC)
# importlib.reload(GGX_hable_abj)
# # myGGX_hable_abj = GGX_hable_abj()
# myGGX_hable_abj = GGX_hable_abj.testPrint()

myABJ_SD_B = ABJ_Shader_Debugger() ######################

# if __name__ == "__main__":
# 	register()


'''
#########
# TO DO:
#########
- additional shading models (oren, glass, hair, subsurface, sheen, fuzz)
- Multiple lights

- gradient mixer (pencil, pen, greyscale paint, color pencil, color paint)

- subdivision gradient lerping blending sim
- samplePoints()

'''