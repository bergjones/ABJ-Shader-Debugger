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


from . import simple_spec_abj
from . import GGX_hable_abj

if "bpy" in locals():
	prefix = __package__ + '.'
	for name, module in sys.modules.copy().items():
		if name.startswith(prefix):
			basename = name.removeprefix(prefix)
			globals()[basename] = importlib.reload(module)

import bpy


from .GGX_hable_abj import myTest
from .simple_spec_abj import myEquation_simple_spec

myTest_class = myTest()
myEquation_simple_spec_class = myEquation_simple_spec()



# # print('sys.modules = ', sys.modules)

# import importlib, sys
# importlib.reload(sys.modules['ABJ_Shader_Debugger_for_Blender.GGX_hable_abj'])
# # importlib.reload(sys.modules['myTest'])
# from .GGX_hable_abj import myTest


# from .GGX_hable_abj import myTest

# import GGX_hable_abj

class ABJ_Shader_Debugger():
	def __init__(self):

		# print('~~~~~~~~~~~~~~~~')
		# print('sys.path = ', sys.path)
		# print('sys.modules = ', sys.modules)

		# myPrint2 = myTest()
		# myTestPrinting = myPrint2.testPrint()
		
		# print(myTestPrinting)

		# print(myPrint2.testPrint())

		# print('~~~~~~~~~~~~~~~~')

		# return

		self.debugV_223 = None
		self.myDebugFaces = []
		self.allNamesToToggleDuringRaycast = []
		self.myCube1_instance_M_all_list_matrixOnly = []
		self.shadingDict_global = {}
		self.shadingList_perFace = []
		self.shadingStages_all = []
		self.shadingStages_perFace_stepList = []
		self.shadingStages_selectedFaces = []
		self.specTesterMatToggle = -1
		self.objectsToToggleOnOffLater = []
		self.debugStageIterPlusMinus = False
		self.recently_cleared_selFaces = False
		self.aov_stored = None
		self.rdotvpow_stored = None
		self.breakpointsOverrideToggle = False

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

		self.spec_cutoff = 0.35
		self.diffuse_or_emission_og_shading = 'diffuse'
		# self.diffuse_or_emission_og_shading = 'emission'
		self.adjustedColors = False

		#instance
		self.myCubeLight_og = None
		self.myCube1_og = None
		self.myCube2_og = None

		self.myCubeLight_og_Matrix = None
		self.myCubeLight_og_Matrix_np = None
		self.myCube1_og_Matrix = None
		self.myCube1_og_Matrix_np = None
		self.myCube2_og_Matrix = None
		self.myCubeCam_og_Matrix = None
		self.myCubeCam_og_Matrix_np = None

		self.myCubeLight_dupe = None
		self.myCube1_dupe = None
		self.myCube2_dupe = None
		self.myCubeCam_dupe = None

		self.myCubeCam = None
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
		self.pos_light_global =  (0.766, 0.836, 0.427)
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
				usableStageCategory_items = bpy.context.scene.bl_rna.properties['shader_stages_enum_prop'].enum_items
				usableStageCategory_id = usableStageCategory_items[bpy.context.scene.shader_stages_enum_prop].identifier

				maxRange_usable = None

				if usableStageCategory_id == 'spec_with_arrow':
					maxRange_usable = 7

				elif usableStageCategory_id == 'spec_no_arrow':
					maxRange_usable = 4

				elif usableStageCategory_id == 'diffuse':
					maxRange_usable = 2

				elif usableStageCategory_id == 'cs':
					maxRange_usable = 5

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

			'''
			usableStageCategory_items = bpy.context.scene.bl_rna.properties['shader_stages_enum_prop'].enum_items
			usableStageCategory_id = usableStageCategory_items[bpy.context.scene.shader_stages_enum_prop].identifier

			maxRange_usable = None

			if usableStageCategory_id == 'spec_with_arrow':
				maxRange_usable = 7

			elif usableStageCategory_id == 'spec_no_arrow':
				maxRange_usable = 4

			elif usableStageCategory_id == 'diffuse':
				maxRange_usable = 2

			elif usableStageCategory_id == 'cs':
				maxRange_usable = 5

			#all
			if step == -1:
					
				for i in self.shadingStages_selectedFaces:
					for j in self.shadingStages_perFace_stepList:
						if j["idx"] == i: #mySplitFaceIndexUsable:
							j["stage"] = max(j["stage"] + step, 0)

			elif step == 1:
				for i in self.shadingStages_selectedFaces:
					for j in self.shadingStages_perFace_stepList:
						if j["idx"] == i: #mySplitFaceIndexUsable:
							j["stage"] = self.clamp(j["stage"] + step, 0, maxRange_usable)

			'''
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
		shadingDict_spec_with_arrow_visualization = {
			'description' : 'Basic Spec Visualization (with Arrow)',
			'stage_000' : 'N....show N arrow (cube1)',
			'stage_001' : 'V....show V arrow (myCubeCam)',
			'stage_002' : 'N_dot_V......show both myCube1 and myCubeCam',
			'stage_003' : 'N_dot_V over ortho compensate trick, so continue...raycast from faceCenter to V',
			'stage_004' : 'faceCenter_to_V_rayCast was TRUE so continue...raycast from faceCenter to L',
			'stage_005' : 'faceCenter_to_L_rayCast was TRUE so continue......show arrows N and L',
			'stage_009' : 'R.....show R arrow (cube2) along with N and L',
			'stage_010' : 'final shade',
		}

		for key, value in shadingDict_spec_with_arrow_visualization.items():
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

		for area in bpy.data.screens["Scripting"].areas:
		# for area in bpy.data.screens["Layout"].areas:
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
		# choice = 'cube1_instance'
		# choice = 'cube2_instance'
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

					if choice == 'cube1_instance':
						if self.adjustedColors == True:
							usableColor = mathutils.Vector((0.0, 1.0, 0.25)) #stereo
						else:
							usableColor = mathutils.Vector((1.0, 0.0, 0.0)) #agx

					elif choice == 'cube2_instance':
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

	def agxSettings_UI(self):
		bpy.context.scene.view_settings.view_transform = 'AgX'
		bpy.context.scene.view_settings.look = 'AgX - Punchy'
		bpy.context.scene.render.use_multiview = False

		self.adjustedColors = False
		self.stereo_retinal_rivalry_fix('cubeCam')
		self.stereo_retinal_rivalry_fix('cube1_instance')
		self.stereo_retinal_rivalry_fix('cube2_instance')
		self.stereo_retinal_rivalry_fix('light_instance')

	def stereoscopicSettings_UI(self):
		bpy.context.scene.view_settings.view_transform = 'Standard'
		bpy.context.scene.view_settings.look = 'None'
		bpy.context.scene.render.use_multiview = True

		self.adjustedColors = True
		self.stereo_retinal_rivalry_fix('cubeCam')
		self.stereo_retinal_rivalry_fix('cube1_instance')
		self.stereo_retinal_rivalry_fix('cube2_instance')
		self.stereo_retinal_rivalry_fix('light_instance')

	########
	def arrowCutoff_UI(self, mode):
		usableIncrement = 0.05
		if mode == "minus":
			self.spec_cutoff -= usableIncrement
		elif mode == "plus":
			self.spec_cutoff += usableIncrement
		elif mode == "reset":
			self.spec_cutoff = 0.35
			# self.spec_cutoff = 0.4

		print('self.spec_cutoff = ', self.spec_cutoff)

		self.doIt_part2_render()

	################
	def restoreCameraView_UI(self):
		restoredCamMatrix = mathutils.Matrix(self.world_mat_cam_stored_np.tolist())
		self.myCam.matrix_world = restoredCamMatrix

	def showhideCubeCam_UI(self):
		if self.myCubeCam.hide_get() == 1:
			self.myCubeCam.hide_set(0)
		else:
			self.myCubeCam.hide_set(1)

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

		#always positive Z value for now

		# print('self.pos_light_global NEW = ', self.pos_light_global)
		self.DoIt_part1_preprocess()

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

		self.DoIt_part1_preprocess()

	def static_debugOnly_Stage1_UI(self):
		#values from init
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
		bpy.ops.transform.resize(value=(2, 4, 2), orient_type='GLOBAL')

	def createArrowFullProcess(self, name, front_or_back_arrow_origin, doLookAt, lookAtPos, r_agx, g_agx, b_agx, r_stereo, g_stereo, b_stereo):
		# self.myCubeCam = self.createArrowFullProcess('myCubeCam', 'back', True, self.myOrigin, 0.0, 1.0, 0.9, 0.0, 1.0, 0.9)

		# self.myCubeLight_og = self.createArrowFullProcess('myCubeLight_og', 'front', False, self.myOrigin, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0)
		
		# self.myCube1_og = self.createArrowFullProcess('myCube1_og', 'back', False, self.myOrigin, 1.0, 0.0, 0.0, 0.0, 1.0, 0.25)


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
		bpy.ops.transform.resize(value=(self.len_cam_arrow, self.arrowWidth, self.arrowWidth), orient_type='GLOBAL')
		bpy.ops.object.transform_apply(location=1, rotation=1, scale=1)
		bpy.ops.object.mode_set(mode="EDIT")
		self.deselectAll_editMode()

		bpy.ops.object.mode_set(mode="OBJECT")

		self.makeArrowFromCube(outputArrow, self.len_reg_arrowExtendPos)
		bpy.ops.object.mode_set(mode="OBJECT") ###################
		self.deselectAll()
		outputArrow.select_set(1)
		bpy.ops.transform.resize(value=(self.scaleRegArrows, self.scaleRegArrows, self.scaleRegArrows), orient_type='GLOBAL')
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

		elif name == 'myCube1_og':
			self.myCube1_og_Matrix = outputArrow.matrix_world
			self.myCube1_og_Matrix_np = np.array(outputArrow.matrix_world)

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
		self.myCube1_instance_M_all_list_matrixOnly.clear()
		self.objectsToToggleOnOffLater.clear()

		self.myDebugFaces.clear()


		aov_items = bpy.context.scene.bl_rna.properties['aov_enum_prop'].enum_items
		aov_id = aov_items[bpy.context.scene.aov_enum_prop].identifier
		self.aov_stored = aov_id

		rdotvpow_items = bpy.context.scene.bl_rna.properties['r_dot_v_pow_enum_prop'].enum_items
		rdotvpow_id = rdotvpow_items[bpy.context.scene.r_dot_v_pow_enum_prop].identifier
		self.rdotvpow_stored = rdotvpow_id

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

		bpy.ops.transform.rotate(value=math.radians(180), orient_axis='X', orient_type='GLOBAL')
		bpy.ops.transform.rotate(value=math.radians(180), orient_axis='Z', orient_type='GLOBAL')
		bpy.ops.transform.rotate(value=math.radians(0), orient_axis='Y', orient_type='GLOBAL')
		bpy.ops.transform.rotate(value=math.radians(self.RandomRotationDegree), orient_axis=self.RandomRotationAxis, orient_type='GLOBAL')

		bpy.ops.object.transform_apply(location=1, rotation=1, scale=1)

		self.profile_stage1_06_b = str(datetime.now() - self.profile_stage1_06_a)
		if self.profileCode_part1 == True:
			print('~~~~~~~~~ self.profile_stage1_06_b = ', self.profile_stage1_06_b)

		########
		#Dupe for raycasting
		########
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
		bpy.ops.object.light_add(type='SUN', radius=1, align='WORLD', location=(self.pos_light_global), scale=(1, 1, 1))
		self.mySun = bpy.context.active_object
		self.mySun.name = "self.mySun"
		# self.mySun.hide_set(1)

		bpy.ops.object.transform_apply(location=1, rotation=1, scale=1) ##### thursday debug

		#############################################################################################
		################################# (INSTANCE ORIGINALS) ###############################
		#############################################################################################

		###############################
		######### CUBE LIGHT (DIRECTIONAL) ############
		###############################
		self.myCubeLight_og = self.createArrowFullProcess('myCubeLight_og', 'front', False, self.myOrigin, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0)

		###############################
		######### CUBE 1 ############
		###############################
		self.myCube1_og = self.createArrowFullProcess('myCube1_og', 'back', False, self.myOrigin, 1.0, 0.0, 0.0, 0.0, 1.0, 0.25)

		###############################
		######### CUBE 2 ############
		###############################
		bpy.context.view_layer.objects.active = self.myCube1_og
		self.myCube2_og = self.copyObject()
		self.myCube2_og.name = 'myCube2_og'

		self.myCube2_og_Matrix = self.myCube2_og.matrix_world

		####
		bpy.context.view_layer.objects.active = self.myCube2_og

		mat2 = None
		if self.adjustedColors == True:
			mat2 = self.newShader("cube2_og", self.diffuse_or_emission_og_shading, 0.2, 0.87, 1.0)
		elif self.adjustedColors == False:	
			mat2 = self.newShader("cube2_og", self.diffuse_or_emission_og_shading, 0.5, 0.0, 1.0) ###
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

		bpy.context.view_layer.objects.active = self.myCube1_og
		self.myCube1_dupe = self.copyObject()
		self.myCube1_dupe.name = 'myCube1_dupe'

		bpy.context.view_layer.objects.active = self.myCube2_og
		self.myCube2_dupe = self.copyObject()
		self.myCube2_dupe.name = 'myCube2_dupe'

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

			self.myCube1_dupe.matrix_world = self.myCube1_og_Matrix
			self.myCube2_dupe.matrix_world = self.myCube2_og_Matrix

			self.profile_stage1_07_b = datetime.now() - self.profile_stage1_07_a
			self.profile_stage1_07_final += self.profile_stage1_07_b

			###############################
			######### INFO (PER FACE) ############
			###############################
			self.profile_stage1_02_a = datetime.now() ################

			faceCenter = self.shadingPlane.location
			pos = self.shadingPlane.location

			myL = mathutils.Vector((self.pos_light_global_v - pos)).normalized()

			##################################################################################
			###################################### STORE SHADE PARAMS #####################################
			##################################################################################
			myEquation_simple_spec_class.equation_preProcess_00(myABJ_SD_B, mySplitFaceIndexUsable)

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
		self.myCube1_og.hide_set(1)
		self.myCube2_og.hide_set(1)

		self.myCubeLight_dupe.hide_set(1)
		self.myCube1_dupe.hide_set(1)
		self.myCube2_dupe.hide_set(1)

		# self.profile_stage1_05_b = datetime.now() - self.profile_stage1_05_a
		# if self.profileCode_part1 == True:
		# 	print('~~~~~~~~~ self.profile_stage1_05_b = ', self.profile_stage1_05_b)

		# self.profile_stage1_04_final += self.profile_stage1_05_b

		print('TIME TO COMPLETE stage 1 (preprocess) = ' + str(datetime.now() - self.startTime_stage1))
		print(' ')


		self.updateScene()

		##################
		# self.doIt_part2_render()
		###############

		self.myCubeCam.hide_set(1)
		bpy.ops.object.mode_set(mode="OBJECT")
		# bpy.context.space_data.lock_camera = True

		self.deselectAll()

		#####################
		##debug raycast miss
		####################
		# myToSelect = [1347, 1319, 824, 1317, 1785, 799, 785, 1793]
		# myToSelect = [236, 361, 223]
		# for i in myToSelect:
		# 	self.shadingStages_selectedFaces.append(str(i))

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
			bpy.context.view_layer.objects.active = self.myCube1_og
		elif type == "R":
			bpy.context.view_layer.objects.active = self.myCube2_og
		elif type == "L":
			bpy.context.view_layer.objects.active = self.myCubeLight_og

		arrow_instance = self.copyObject()

		# bpy.context.view_layer.objects.active = arrow_instance

		arrow_instance.name = namePrefix + str(idx)
		arrow_instance.hide_set(0)
		arrow_instance.matrix_world = matrix

		return arrow_instance

	def show_arrow_N(self, shadingPlane, faceCenter, mySplitFaceIndexUsable):
		nameToLookFor = 'cube1_instance_' + mySplitFaceIndexUsable

		for k in self.objectsToToggleOnOffLater:
			if k.name == nameToLookFor:
				# if k.hide_get() == 1:
				k.hide_set(0)

				return

		########################

		# for j in bpy.context.scene.objects:
		# 	for k in self.objectsToToggleOnOffLater:
		# 		if j.name == k.name:

		restored_N_M_np = None

		for i in self.myCube1_instance_M_all_list_matrixOnly:
			temp_idx = i['mySplitFaceIndexUsable']
			if temp_idx == mySplitFaceIndexUsable:
				N_M_np = i['N_M_np']
				restored_N_M_np = mathutils.Matrix(N_M_np.tolist())

		myCube1_instance = self.copyAndSet_arrow(mySplitFaceIndexUsable, restored_N_M_np, 'cube1_instance_', 'N')
		self.objectsToToggleOnOffLater.append(myCube1_instance)

	def show_arrow_V_to_faceCenter(self, faceCenter, mySplitFaceIndexUsable):
		nameToLookFor = 'faceCenterToV_rc_instance_' + mySplitFaceIndexUsable

		for k in self.objectsToToggleOnOffLater:
			if k.name == nameToLookFor:
				# if k.hide_get() == 1:
				k.hide_set(0)

				return

		########################

		restored_V_M_np = None

		for i in self.myCube1_instance_M_all_list_matrixOnly:
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

		for i in self.myCube1_instance_M_all_list_matrixOnly:
			temp_idx = i['mySplitFaceIndexUsable']
			if temp_idx == mySplitFaceIndexUsable:
				L_M_np = i['L_M_np']
				restored_L_M_np = mathutils.Matrix(L_M_np.tolist())
		
		myCubeLight_instance = self.copyAndSet_arrow(mySplitFaceIndexUsable, restored_L_M_np, 'cubeLight_instance_', 'L')
		self.objectsToToggleOnOffLater.append(myCubeLight_instance)

	def show_arrow_R(self, faceCenter, mySplitFaceIndexUsable, L, N):
		nameToLookFor = 'cube2_instance_' + mySplitFaceIndexUsable

		for k in self.objectsToToggleOnOffLater:
			if k.name == nameToLookFor:
				# if k.hide_get() == 1:
				k.hide_set(0)

				return
				
		########################

		restored_R_M_np = None

		for i in self.myCube1_instance_M_all_list_matrixOnly:
			temp_idx = i['mySplitFaceIndexUsable']
			if temp_idx == mySplitFaceIndexUsable:
				R_M_np = i['R_M_np']
				restored_R_M_np = mathutils.Matrix(R_M_np.tolist())

		myCube2_instance = self.copyAndSet_arrow(mySplitFaceIndexUsable, restored_R_M_np, 'cube2_instance_', 'R')
		self.objectsToToggleOnOffLater.append(myCube2_instance)

	def setActiveStageMaterial(self, shadingPlane, idx, r, g, b):
		if self.specTesterMatToggle == -1:
			for j in bpy.context.scene.objects:
				if j.name == shadingPlane:
					bpy.context.view_layer.objects.active = j

			# bpy.context.view_layer.objects.active = 
			mat1 = self.newShader("ShaderVisualizer_" + str(idx), "emission", r, g, b)
			bpy.context.active_object.data.materials.clear()
			bpy.context.active_object.data.materials.append(mat1)


	def doIt_part2_render(self):
		startTime = datetime.now()

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

		# for i in self.objectsToToggleOnOffLater:
		# 	try:
		# 		self.deleteSpecificObject(i.name)
		# 	except:
		# 		pass

		for i in self.objectsToToggleOnOffLater:
			i.hide_set(1)

		# self.objectsToToggleOnOffLater.clear()

		V = self.shadingDict_global['V']

		aov_items = bpy.context.scene.bl_rna.properties['aov_enum_prop'].enum_items
		aov_id = aov_items[bpy.context.scene.aov_enum_prop].identifier

		rdotvpow_items = bpy.context.scene.bl_rna.properties['r_dot_v_pow_enum_prop'].enum_items
		rdotvpow_id = rdotvpow_items[bpy.context.scene.r_dot_v_pow_enum_prop].identifier

		###print once variables
		printOnce_stage_000 = False
		printOnce_stage_001 = False
		printOnce_stage_002 = False
		printOnce_stage_003 = False
		printOnce_stage_004 = False
		printOnce_stage_005 = False
		printOnce_stage_006 = False
		printOnce_stage_007 = False

		skip_refresh_override = False

		if self.aov_stored != aov_id:
			self.aov_stored = aov_id
			skip_refresh_override = True

		if self.rdotvpow_stored != rdotvpow_id:
			self.rdotvpow_stored = rdotvpow_id
			skip_refresh_override = True

		if self.recently_cleared_selFaces == True:
			skip_refresh_override = True
			self.recently_cleared_selFaces = False

		self.profile_stage2_00_b = str(datetime.now() - self.profile_stage2_00_a)
		if self.profileCode_part2 == True:
			print('~~~~~~~~~ self.profile_stage2_00_b = ', self.profile_stage2_00_b)

		# self.myCube1_instance_M_all_list_matrixOnly.clear()


		self.profile_stage2_08_a = datetime.now() ################

		all_indicies_in_matrix_list = []
		for i in self.myCube1_instance_M_all_list_matrixOnly:
			all_indicies_in_matrix_list.append(i['mySplitFaceIndexUsable'])

		for i in self.shadingStages_selectedFaces:
		# for i in self.myDebugFaces:
			for j in self.shadingList_perFace:	
				if j["mySplitFaceIndexUsable"] == i: #mySplitFaceIndexUsable:

					################
					## PRE-PROCESS
					################

					mySplitFaceIndexUsable = j['mySplitFaceIndexUsable']

					#################################################
					#decide whether to continue and do a full refresh
					#################################################
					if mySplitFaceIndexUsable in all_indicies_in_matrix_list:
						continue

					shadingPlane = j['shadingPlane']
					faceCenter = j['faceCenter']
					# L = j['L']
					# N = j['N']
					R = j['R']

					N_M = self.dynamic_cube1_creation(shadingPlane, faceCenter, mySplitFaceIndexUsable)
					L_M = self.dynamic_cubeLight_creation(faceCenter, mySplitFaceIndexUsable, self.mySun)
					V_M = self.dynamic_cubeV_creation(faceCenter, mySplitFaceIndexUsable, self.myCam)
					R_M = self.dynamic_cube2_creation(faceCenter, mySplitFaceIndexUsable, N_M, R)

					self.updateScene() # need

					N_M_np = np.array(N_M)
					L_M_np = np.array(L_M)
					V_M_np = np.array(V_M)
					R_M_np = np.array(R_M)

					myCube1_instance_M_dict = {
						'mySplitFaceIndexUsable' : mySplitFaceIndexUsable,
						'N_M_np' : N_M_np,
						'L_M_np' : L_M_np,
						'V_M_np' : V_M_np,
						'R_M_np' : R_M_np,
					}

					self.myCube1_instance_M_all_list_matrixOnly.append(myCube1_instance_M_dict) ##########

		self.updateScene()

		self.profile_stage2_08_b = str(datetime.now() - self.profile_stage2_08_a)
		if self.profileCode_part2 == True:
			print('~~~~~~~~~ self.profile_stage2_08_b = ', self.profile_stage2_08_b)

		self.deselectAll()

		############################################################
		allNames = []
		self.allNamesToToggleDuringRaycast = []
		for i in self.shadingList_perFace:
			allNames.append(i['shadingPlane'])

		for i in bpy.context.scene.objects:
			if i.name not in allNames:
				if i.hide_get() == 0:
					self.allNamesToToggleDuringRaycast.append(i)

		# for i in allNamesToToggleDuringRaycast:
		# 	print(i)

		print('~~~~~~~~~~~~~~~~~~~ !!!!!!!!!!!! ~~~~~~~~~~~~~~~~~~~')
		print('~~~~~~~~~~~~~~~~~~~ !!!!!!!!!!!! allNamesToToggleDuringRaycast', self.allNamesToToggleDuringRaycast)
		print('~~~~~~~~~~~~~~~~~~~ !!!!!!!!!!!! ~~~~~~~~~~~~~~~~~~~')
		############################################################

		for i in self.shadingList_perFace:
			self.profile_stage2_01_a = datetime.now() ################

			mySplitFaceIndexUsable = i['mySplitFaceIndexUsable']

			#################################################
			#decide whether to continue and do a full refresh
			#################################################
			matCheck = bpy.data.materials.get("ShaderVisualizer_" + str(mySplitFaceIndexUsable))
			skip_refresh = False
			if matCheck: #material already exists...check if it is not selected for stage stepping
				for j in self.shadingStages_perFace_stepList:
					if (j["idx"]) == mySplitFaceIndexUsable:
						if j['idx'] not in self.shadingStages_selectedFaces:
							skip_refresh = True

			if skip_refresh_override == True:
				skip_refresh = False

			if skip_refresh == True:
				continue

			####################################

			shadingPlane = i['shadingPlane']
			faceCenter = i['faceCenter']
			N_dot_L = i['N_dot_L']
			N_dot_V = i['N_dot_V']
			R_dot_V = i['R_dot_V']
			attenuation = i['attenuation']
			L = i['L']
			N = i['N']
			R = i['R']

			spec = 0

			usableRdotVPow_items = bpy.context.scene.bl_rna.properties['r_dot_v_pow_enum_prop'].enum_items
			usableRdotVPow_id = usableRdotVPow_items[bpy.context.scene.r_dot_v_pow_enum_prop].identifier

			if usableRdotVPow_id == 'pow1':
				spec = pow(R_dot_V, 1) ##

			elif usableRdotVPow_id == 'pow2':
				spec = pow(R_dot_V, 2) ##

			elif usableRdotVPow_id == 'pow4':
				spec = pow(R_dot_V, 4) ##

			elif usableRdotVPow_id == 'pow8':
				spec = pow(R_dot_V, 8) ##

			elif usableRdotVPow_id == 'pow16':
				spec = pow(R_dot_V, 16) ##

			elif usableRdotVPow_id == 'pow32':
				spec = pow(R_dot_V, 32) ##

			elif usableRdotVPow_id == 'diffuse_only':
				spec = 0

			##############
			#######
			###########
			#debug
			# self.shadingStages_selectedFaces.clear()
			# self.shadingStages_selectedFaces.append('242')

			#local variables
			faceCenter_to_V_rayCast = None
			faceCenter_to_L_rayCast = None

			#visualize arrows if spec > cutoff
			cutoff = self.spec_cutoff

			self.profile_stage2_01_b = datetime.now() - self.profile_stage2_01_a
			self.profile_stage2_01_final += self.profile_stage2_01_b

			self.profile_stage2_04_a = datetime.now() ################

			N_dot_V_over_threshold_with_ortho_compensateTrick = None
			# if (N_dot_V <= 0):
			if (N_dot_V <= 0.1):
				N_dot_V_over_threshold_with_ortho_compensateTrick = False
			else:
				N_dot_V_over_threshold_with_ortho_compensateTrick = True

			if N_dot_V_over_threshold_with_ortho_compensateTrick == False:
				spec = 0

			else:
				#######################
				#RAYCAST AGAINST V
				#######################
				# direction_vector = point_b - point_a:
				# This line subtracts the coordinates of point_a from point_b, resulting in a vector pointing from point_a to point_b.
				V_toFace = mathutils.Vector(faceCenter - self.pos_camera_global_v).normalized()

				myRay_faceCenter_to_V = self.raycast_abj_scene(self.pos_camera_global_v, V_toFace, mySplitFaceIndexUsable) ########## good

				if myRay_faceCenter_to_V == True:
					faceCenter_to_V_rayCast = True
					# print('makes it to the cam, now cast again')

				else:
					faceCenter_to_V_rayCast = False
					# print('behind something else, discard')

				if faceCenter_to_V_rayCast == True:
					#######################
					#RAYCAST AGAINST L
					#######################
					if mySplitFaceIndexUsable == '350':
						print('raycast to L check...')
					myRay_faceCenter_to_L = self.raycast_abj_scene(self.pos_light_global_v, -L, mySplitFaceIndexUsable) 

					if myRay_faceCenter_to_L == True:
						faceCenter_to_L_rayCast = True

					else:
						faceCenter_to_L_rayCast = False

					if faceCenter_to_L_rayCast == True: 
						if (spec > cutoff):
							pass #end shader

					elif faceCenter_to_L_rayCast == False:
						spec = 0
							
				elif faceCenter_to_V_rayCast == False:
					spec = 0


			self.profile_stage2_04_b = datetime.now() - self.profile_stage2_04_a
			self.profile_stage2_04_final += self.profile_stage2_04_b

			self.profile_stage2_02_a = datetime.now() ################

			##################
			#STEPS FOR ALL
			##################
			maxRange_usable = None

			usableStageCategory_items = bpy.context.scene.bl_rna.properties['shader_stages_enum_prop'].enum_items
			usableStageCategory_id = usableStageCategory_items[bpy.context.scene.shader_stages_enum_prop].identifier

			if usableStageCategory_id == 'spec_with_arrow':
				maxRange_usable = 7

			elif usableStageCategory_id == 'spec_no_arrow':
				maxRange_usable = 4

			elif usableStageCategory_id == 'diffuse':
				maxRange_usable = 2

			elif usableStageCategory_id == 'cs':
				maxRange_usable = 5

			items_id_currentStage = None
			override = False

			for j in self.shadingStages_perFace_stepList:
				if (j["idx"]) == mySplitFaceIndexUsable:
					if j['idx'] not in self.shadingStages_selectedFaces:
						if usableStageCategory_id == 'spec_with_arrow':
							items_id_currentStage = maxRange_usable
							override = True

					elif j['idx'] in self.shadingStages_selectedFaces:
						currentStage = self.myBreakpointList[j['breakpoint_idx']]

						combo_currentStage = 'bpy.context.scene.' + currentStage
						items_currentStage = bpy.context.scene.bl_rna.properties[currentStage].enum_items
						items_id_currentStage = items_currentStage[eval(combo_currentStage)].identifier
						items_id_currentStage = int(items_id_currentStage)

			####################################
			#variables
			####################################
			usableStageCategory_items = bpy.context.scene.bl_rna.properties['shader_stages_enum_prop'].enum_items
			usableStageCategory_id = usableStageCategory_items[bpy.context.scene.shader_stages_enum_prop].identifier

			aov_items = bpy.context.scene.bl_rna.properties['aov_enum_prop'].enum_items
			aov_id = aov_items[bpy.context.scene.aov_enum_prop].identifier

			self.profile_stage2_02_b = datetime.now() - self.profile_stage2_02_a
			self.profile_stage2_02_final += self.profile_stage2_02_b
			# if self.profileCode_part2 == True:
				# print('~~~~~~~~~ self.profile_stage2_02_b = ', self.profile_stage2_02_b)

			###############
			### spec_with_arrow
			##############

			if usableStageCategory_id == 'spec_with_arrow':
				# if mySplitFaceIndexUsable == '242':
				# 	print('!!!!!!!!!!!!! items_id_currentStage = ', items_id_currentStage)
				# 	print('skip_refresh = ', skip_refresh)
				# 	print('faceCenter_to_L_rayCast for 242 = ', faceCenter_to_L_rayCast)
				# 	print('faceCenter_to_V_rayCast for 242 = ', faceCenter_to_V_rayCast)

				if items_id_currentStage == 0:
					if printOnce_stage_000 == False:
						print("'stage_000' : 'N....show N arrow (cube1)'")
						printOnce_stage_000 = True

					self.show_arrow_N(shadingPlane, faceCenter, mySplitFaceIndexUsable)

					self.myCubeCam.hide_set(1)

					self.setActiveStageMaterial(shadingPlane, mySplitFaceIndexUsable, self.shadingPlane_sel_r, self.shadingPlane_sel_g, self.shadingPlane_sel_b)

				elif items_id_currentStage == 1:
					if printOnce_stage_001 == False:
						print("'stage_001' : 'V....show V arrow (myCubeCam)'")
						printOnce_stage_001 = True

					self.myCubeCam.hide_set(0)

					self.setActiveStageMaterial(shadingPlane, mySplitFaceIndexUsable, self.shadingPlane_sel_r, self.shadingPlane_sel_g, self.shadingPlane_sel_b)

				elif items_id_currentStage == 2:
					if printOnce_stage_002 == False:
						print("'stage_002' : 'N_dot_V......show both myCube1 and myCubeCam'")
						printOnce_stage_002 = True

					self.show_arrow_N(shadingPlane, faceCenter, mySplitFaceIndexUsable)

					self.myCubeCam.hide_set(0)

					self.setActiveStageMaterial(shadingPlane, mySplitFaceIndexUsable, self.shadingPlane_sel_r, self.shadingPlane_sel_g, self.shadingPlane_sel_b)

				elif items_id_currentStage == 3:
					if N_dot_V_over_threshold_with_ortho_compensateTrick == False: #####
						if self.printDetailedInfo == True:
							print('N_dot_V_over_threshold_with_ortho_compensateTrick FAIL for ', mySplitFaceIndexUsable)
						self.aov_output(aov_id, shadingPlane, mySplitFaceIndexUsable, N_dot_L, spec, attenuation)

					elif N_dot_V_over_threshold_with_ortho_compensateTrick == True or override == True:
						if printOnce_stage_003 == False:
							print('N_dot_V over ortho compensate trick, so continue...', N_dot_V_over_threshold_with_ortho_compensateTrick)
							print("'stage_003' : 'raycast from faceCenter to V'")
							printOnce_stage_003 = True

						self.profile_stage2_03_a = datetime.now() ################

						self.show_arrow_V_to_faceCenter(faceCenter, mySplitFaceIndexUsable)

						self.setActiveStageMaterial(shadingPlane, mySplitFaceIndexUsable, self.shadingPlane_sel_r, self.shadingPlane_sel_g, self.shadingPlane_sel_b)

						self.profile_stage2_03_b = datetime.now() - self.profile_stage2_03_a
						self.profile_stage2_03_final += self.profile_stage2_03_b

						self.myCubeCam.hide_set(1)

				elif items_id_currentStage == 4:
					if faceCenter_to_V_rayCast == False: ####
						if self.printDetailedInfo == True:
							print('faceCenter_to_V_rayCast FAIL for ', mySplitFaceIndexUsable)
						self.aov_output(aov_id, shadingPlane, mySplitFaceIndexUsable, N_dot_L, spec, attenuation)

					elif faceCenter_to_V_rayCast == True or override == True:
							if printOnce_stage_004 == False:
								print('faceCenter_to_V_rayCast was TRUE so continue... ', faceCenter_to_V_rayCast)
								print("'stage_004' : 'raycast from faceCenter to L'")
								printOnce_stage_004 = True

							self.show_arrow_L_to_faceCenter(faceCenter, mySplitFaceIndexUsable)

							self.myCubeCam.hide_set(1)

							self.setActiveStageMaterial(shadingPlane, mySplitFaceIndexUsable, self.shadingPlane_sel_r, self.shadingPlane_sel_g, self.shadingPlane_sel_b)

				elif items_id_currentStage == 5:
					# if mySplitFaceIndexUsable == '242':
					# 	print('IN 005')
					# 	print('skip_refresh = ', skip_refresh)
					# 	print('printOnce_stage_005 = ', printOnce_stage_005)
					# 	print('faceCenter_to_L_rayCast for 242 = ', faceCenter_to_L_rayCast)
					# 	print('faceCenter_to_V_rayCast for 242 = ', faceCenter_to_V_rayCast)

					if faceCenter_to_L_rayCast == False: ####
						if self.printDetailedInfo == True:
							print('faceCenter_to_L_rayCast FAIL for ', mySplitFaceIndexUsable)
						self.aov_output(aov_id, shadingPlane, mySplitFaceIndexUsable, N_dot_L, spec, attenuation)

					elif faceCenter_to_L_rayCast == True or override == True:
						if printOnce_stage_005 == False:
							print('faceCenter_to_L_rayCast was TRUE so continue... ', faceCenter_to_V_rayCast)
							print("'stage_005' : 'show arrows (N, L)'")
							printOnce_stage_005 = True

						self.show_arrow_N(shadingPlane, faceCenter, mySplitFaceIndexUsable)

						##############

						self.show_arrow_L_to_faceCenter(faceCenter, mySplitFaceIndexUsable)

						self.myCubeCam.hide_set(1)

						self.setActiveStageMaterial(shadingPlane, mySplitFaceIndexUsable, self.shadingPlane_sel_r, self.shadingPlane_sel_g, self.shadingPlane_sel_b)

				elif items_id_currentStage == 6:
					if faceCenter_to_L_rayCast == True or override == True:
						if printOnce_stage_006 == False:
							print("'stage_006' : 'R.....show R arrow (cube2) along with N and L'")
							printOnce_stage_006 = True

						self.show_arrow_N(shadingPlane, faceCenter, mySplitFaceIndexUsable)
						self.show_arrow_L_to_faceCenter(faceCenter, mySplitFaceIndexUsable)
						self.show_arrow_R(faceCenter, mySplitFaceIndexUsable, L, N)

						self.myCubeCam.hide_set(1)

						self.setActiveStageMaterial(shadingPlane, mySplitFaceIndexUsable, self.shadingPlane_sel_r, self.shadingPlane_sel_g, self.shadingPlane_sel_b)

				elif items_id_currentStage == 7:
					if printOnce_stage_007 == False:
						print('stage_007 output AOV = ', aov_id)
						printOnce_stage_007 = True

					self.myCubeCam.hide_set(1)

					self.aov_output(aov_id, shadingPlane, mySplitFaceIndexUsable, N_dot_L, spec, attenuation)

			# if self.profileCode_part2 == True:
				# print('~~~~~~~~~ self.profile_stage2_03_b = ', self.profile_stage2_03_b)

		if self.profileCode_part2 == True:
			# print('~~~~~~~~~ self.profile_stage2_00_final = ', self.profile_stage2_00_final)
			# print('~~~~~~~~~ self.profile_stage2_01_final = ', self.profile_stage2_01_final)
			print('~~~~~~~~~ self.profile_stage2_02_final - raycast = ', self.profile_stage2_02_final)
			
			print('~~~~~~~~~ self.profile_stage2_03_final = ', self.profile_stage2_03_final)
			# print('~~~~~~~~~ self.profile_stage2_04_final = ', self.profile_stage2_04_final)
			print('~~~~~~~~~ self.profile_stage2_05_final = ', self.profile_stage2_05_final)

			# print('~~~~~~~~~ self.profile_stage2_06_final = ', self.profile_stage2_06_final)
			# print('~~~~~~~~~ self.profile_stage2_07_final = ', self.profile_stage2_07_final)
			# print('~~~~~~~~~ self.profile_stage2_08_final = ', self.profile_stage2_08_final)
			# print('~~~~~~~~~ self.profile_stage2_09_final = ', self.profile_stage2_09_final)
			# print('~~~~~~~~~ self.profile_stage2_10_final = ', self.profile_stage2_10_final)

			# print('~~~~~~~~~ self.profile_stage2_08_b = ', self.profile_stage2_08_b)

		# self.myCubeCam.hide_set(1)

		print('TIME TO COMPLETE (render) = ' + str(datetime.now() - startTime))
		print(' ')

	def dynamic_cube2_creation(self, faceCenter, mySplitFaceIndexUsable, defaultMatrix, R):
		self.myCube2_dupe.matrix_world = defaultMatrix

		bpy.context.view_layer.objects.active = self.myCube2_dupe

		#apply rotation
		bpy.context.active_object.rotation_mode = 'QUATERNION'
		bpy.context.active_object.rotation_quaternion = R.to_track_quat('X','Z')

		dynamicM = self.myCube2_dupe.matrix_world

		return dynamicM

	def dynamic_cube1_creation(self, shadingPlane, faceCenter, mySplitFaceIndexUsable):
		# self.profile_stage2_07_a = datetime.now() ################

		self.myCube1_dupe.matrix_world = self.myCube1_og_Matrix

		for j in bpy.context.scene.objects:
			if j.name == shadingPlane:
				bpy.context.view_layer.objects.active = j

		normalDir = self.getFaceNormal()
		# myN = normalDir.normalized()

		bpy.context.view_layer.objects.active = self.myCube1_dupe
		bpy.context.active_object.rotation_mode = 'QUATERNION'
		bpy.context.active_object.rotation_quaternion = normalDir.to_track_quat('X','Z')

		self.myCube1_dupe.location = faceCenter

		#use x axis
		dynamicM = self.myCube1_dupe.matrix_world

		# self.profile_stage2_07_b = datetime.now() - self.profile_stage2_07_a
		# self.profile_stage2_07_final += self.profile_stage2_07_b

		return dynamicM

	def dynamic_cubeV_creation(self, faceCenter, mySplitFaceIndexUsable, myCam):
		self.myCubeCam_dupe.matrix_world = self.myCubeLight_og_Matrix
		bpy.context.view_layer.objects.active = self.myCubeCam_dupe
		self.myCubeCam_dupe.location = faceCenter

		self.updateScene()
		
		self.look_at2(self.myCubeCam_dupe, self.pos_camera_global_v)

		# #####################
		bpy.ops.object.mode_set(mode="OBJECT")
		self.deselectAll()
		self.myCubeCam_dupe.select_set(1)

		myCubeLight_dupe_Matrix_np = np.array(self.myCubeCam_dupe.matrix_world)

		self.objScaling_toMatchPosition_localSolve(self.myCubeCam_dupe, self.myCubeLight_og.name, myCam.matrix_world.translation, 1, 0, myCubeLight_dupe_Matrix_np)

		self.updateScene()

		dynamicM = self.myCubeCam_dupe.matrix_world
		return dynamicM

	def dynamic_cubeLight_creation(self, faceCenter, mySplitFaceIndexUsable, mySun):
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

	def aov_output(self, aov_id, shadingPlane, mySplitFaceIndexUsable, N_dot_L, spec, attenuation):
		attenuation = 1.0 #temporary, outside sunlight

		Ks = 10
		Kl = 1
		finalDiff = N_dot_L
		finalSpec = spec * Ks

		if aov_id == 'spec':
			Ci = ((finalSpec) * attenuation * Kl) ###
		elif aov_id == 'diffuse':
			Ci = ((finalDiff) * attenuation * Kl) ###
		elif aov_id == 'Ci':
			Ci = ((finalDiff + finalSpec) * attenuation * Kl) ###

		Ci = pow(Ci, (1.0 / 2.2))

		if self.specTesterMatToggle == -1:
			for j in bpy.context.scene.objects:
				if j.name == shadingPlane:
					bpy.context.view_layer.objects.active = j

			mat1 = self.newShader("ShaderVisualizer_" + str(mySplitFaceIndexUsable), "emission", Ci, Ci, Ci)
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

		self.updateScene()
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

				if debugidx == toDebug:
					print('TRUE for debugIdx, obj : ', debugidx, ' ', obj.name)
			else:
				toReturn = False
				# if debugidx == '236' or debugidx == '361' or debugidx == '296' or debugidx == '223':

				if debugidx == toDebug:
					print('FALSE for debugIdx, obj : ', debugidx, ' ', obj.name)
	
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
		row.scale_y = 2.0 ###
		row.operator('shader.abj_shader_debugger_randomlight_operator')
		row.operator('shader.abj_shader_debugger_randomrotation_operator')

		row = layout.row()
		row.scale_y = 1.0 ###
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
		row.scale_y = 1.0 ###
		row.prop(bpy.context.scene, 'shader_stages_enum_prop', text="")

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

		######################################
		###### R_DOT_V_POW
		######################################
		layout.label(text='R.V POW:')
		row = layout.row()
		row.scale_y = 1.0 ###
		row.prop(bpy.context.scene, 'r_dot_v_pow_enum_prop', text="")

		layout.label(text='camera')
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
		layout.label(text='toggle 0')
		row = layout.row()
		row.scale_y = 1.0 ###
		row.operator('shader.abj_shader_debugger_showhidearrow_operator')
		row.operator('shader.abj_shader_debugger_showhidecubecam_operator')
		row.operator('shader.abj_shader_debugger_toggleextras_operator')

		layout.label(text='arrow cutoff')
		row = layout.row()
		row.operator('shader.abj_shader_debugger_arrowcutoffminus_operator')
		row.operator('shader.abj_shader_debugger_arrowcutoffplus_operator')
		row.operator('shader.abj_shader_debugger_arrowcutoffreset_operator')

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

		######################################
		###### COLOR PRESETS
		######################################
		layout.label(text='COLOR PRESETS')
		row = layout.row()
		row.scale_y = 2.0 ###
		row.operator('shader.abj_shader_debugger_agx_operator')
		row.operator('shader.abj_shader_debugger_stereoscopic_operator')

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

	bl_label = 'rand mesh rotation'
	bl_idname = 'shader.abj_shader_debugger_randomrotation_operator'

	def execute(self, context):
		myABJ_SD_B.randomRotation_UI()
		return {'FINISHED'}

class SHADER_OT_STATICSTAGE1(bpy.types.Operator):
	# if you create an operator class called MYSTUFF_OT_super_operator, the bl_idname should be mystuff.super_operator

	bl_label = 'debug (fixed)'
	bl_idname = 'shader.abj_shader_debugger_staticstage1_operator'

	def execute(self, context):
		myABJ_SD_B.static_debugOnly_Stage1_UI()
		return {'FINISHED'}

class SHADER_OT_REFRESHSTAGE2(bpy.types.Operator):
	bl_label = 'refresh stage 2'
	bl_idname = 'shader.abj_shader_debugger_refreshstage2_operator'

	def execute(self, context):
		myABJ_SD_B.refreshPart2_UI()
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
class SHADER_OT_SHOWHIDEARROWTOGGLE(bpy.types.Operator):
	bl_label = 'toggle arrow'
	bl_idname = 'shader.abj_shader_debugger_showhidearrow_operator'

	def execute(self, context):
		myABJ_SD_B.showhideArrows_UI()
		return {'FINISHED'}

class SHADER_OT_SHOWHIDECUBECAM(bpy.types.Operator):
	bl_label = 'cam tgl'
	bl_idname = 'shader.abj_shader_debugger_showhidecubecam_operator'

	def execute(self, context):
		myABJ_SD_B.showhideCubeCam_UI()
		return {'FINISHED'}

class SHADER_OT_TOGGLEEXTRAS(bpy.types.Operator):
	bl_label = 'toggle extras'
	bl_idname = 'shader.abj_shader_debugger_toggleextras_operator'

	def execute(self, context):
		myABJ_SD_B.toggleExtras_UI()
		return {'FINISHED'}

##########################################
############# CUTOFF ################
##########################################
class SHADER_OT_ARROWCUTOFFMINUS(bpy.types.Operator):
	bl_label = 'cutoff -'
	bl_idname = 'shader.abj_shader_debugger_arrowcutoffminus_operator'

	def execute(self, context):
		myABJ_SD_B.arrowCutoff_UI('minus')
		return {'FINISHED'}

class SHADER_OT_ARROWCUTOFFPLUS(bpy.types.Operator):
	bl_label = 'cutoff +'
	bl_idname = 'shader.abj_shader_debugger_arrowcutoffplus_operator'

	def execute(self, context):
		myABJ_SD_B.arrowCutoff_UI('plus')
		return {'FINISHED'}

class SHADER_OT_ARROWCUTOFFRESET(bpy.types.Operator):
	bl_label = 'cutoff R'
	bl_idname = 'shader.abj_shader_debugger_arrowcutoffreset_operator'

	def execute(self, context):
		myABJ_SD_B.arrowCutoff_UI('reset')
		return {'FINISHED'}

##########################################
############# COLOR PRESETS ################
##########################################
class SHADER_OT_AGXSETTINGS(bpy.types.Operator):
	bl_label = 'agx'
	bl_idname = 'shader.abj_shader_debugger_agx_operator'

	def execute(self, context):
		myABJ_SD_B.agxSettings_UI()
		return {'FINISHED'}

class SHADER_OT_STEREOSCOPICSETTINGS(bpy.types.Operator):
	bl_label = 'stereoscopic'
	bl_idname = 'shader.abj_shader_debugger_stereoscopic_operator'

	def execute(self, context):
		myABJ_SD_B.stereoscopicSettings_UI()
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
- Multiple lights
- light types
- additional shading models (ggx, oren, principled, new metallic)



	
##########
# DONE
##########
- step through selected faces at different rates
- cam arrow scaling
- light arrow scaling
- look at redundant calls to self.updateScene()
- Input Model Choose
- arrow cutoff +-
- randomize transform
- UI
- enum property
- enum for changing spec power through UI
- 3d stereo
- Set arrow materials
- fix matrix assignment (with numpy)
- look at changing materials on cube1/cube2/cubeL with self.adjustedColors toggle added to Agx / stereoscopic button


-2 constraints
Copy Rotation (camera)
Transformation (camera)
From: X min 1 / X max 10
To: X min 0.5 / X max 2

- write / show what the face arrows are trying to accomplish at any given time
	- cube1 is the normal
	- cube 2 is the reflection
	- cube L is the light
	- cube Cam is the eye
'''

#blender\intern\cycles\kernel\closure\bsdf.h
#blender\intern\cycles\kernel\integrator\subsurface_random_walk.h

#bpy.ops.script.reload()
# LOCATION = C:\Users\aleks\AppData\Roaming\Blender Foundation\Blender\4.3\scripts\addons\ABJ_Shader_Debugger_for_Blender
