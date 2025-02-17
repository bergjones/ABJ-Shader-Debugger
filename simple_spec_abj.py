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
import math
import mathutils
from datetime import datetime
import random
import numpy as np
import importlib
import sys
import copy

class myEquation_simple_spec:
	def __init__(self):
		super(myEquation_simple_spec, self).__init__()
		
	def testPrint(self):
			myString = '~~~~~~~~~~~ myString...from myEquation_simple_spec_002'
			return myString
	
		# def refreshPart2_UI(self):
		# 	self.print(myTest_class.testPrint())
		# 	self.print(myEquation_simple_spec_class.testPrint())

		# 	self.doIt_part2_render()
		# 	bpy.ops.object.mode_set(mode="OBJECT")

	def equation_part1_preProcess_00(self, abj_sd_b_instance, mySplitFaceIndexUsable):
		faceCenter = abj_sd_b_instance.shadingPlane.location
		pos = abj_sd_b_instance.shadingPlane.location

		myL = mathutils.Vector((abj_sd_b_instance.pos_light_global_v - pos)).normalized()

		##################################################################################
		###################################### STORE SHADE PARAMS #####################################
		##################################################################################
		bpy.context.view_layer.objects.active = abj_sd_b_instance.shadingPlane

		normalDir = abj_sd_b_instance.getFaceNormal() ################

		myN = normalDir.normalized()
	
		myReflectVec_cube2 = -myL.reflect(myN)

		########################
		########## DIFFUSE ########
		########################
		N_dot_L = max(np.dot(myN, myL), 0.0)

		########################
		########## SPEC ########
		########################
		myR = myReflectVec_cube2

		R_dot_V_control = max(abj_sd_b_instance.myV.dot(myR), 0.0)
		N_dot_V = max(myN.dot(abj_sd_b_instance.myV), 0.0)

		distance = (abj_sd_b_instance.pos_light_global_v - pos).length
		attenuation = 1.0 / (distance * distance)

		shadingDict_perFace = {
			'mySplitFaceIndexUsable' : mySplitFaceIndexUsable,
			'shadingPlane' : abj_sd_b_instance.shadingPlane.name,
			'faceCenter' : faceCenter,
			'N_dot_L' : N_dot_L,
			'N_dot_V' : N_dot_V,
			'R_dot_V' : R_dot_V_control,
			'attenuation' : attenuation,
			'L' : myL,
			'N' : myN,
			'R' : myR,
			'spec' : 0,
			'faceCenter_to_V_rayCast' : False,
			'faceCenter_to_L_rayCast' : False,
		}

		if abj_sd_b_instance.hideUnHideInitialNdotV == True:
			if N_dot_V <= 0.1:
				#Hide for potential raycast speed up
				abj_sd_b_instance.shadingPlane.hide_set(1)

		test_stagesDict_perFace0 = {
			'idx' : mySplitFaceIndexUsable,
			'shadingPlane' : abj_sd_b_instance.shadingPlane.name,
			# 'stage' : usableBreakpoint000_items_id,
			'stage' : 0,
			'breakpoint_idx' : 0,
		}

		abj_sd_b_instance.myDebugFaces.append(mySplitFaceIndexUsable)

		abj_sd_b_instance.shadingList_perFace.append(shadingDict_perFace)

		abj_sd_b_instance.shadingStages_perFace_stepList.append(test_stagesDict_perFace0)
		
		# '''

	def equation_part2_preProcess(self, abj_sd_b_instance):
		aov_items = bpy.context.scene.bl_rna.properties['aov_enum_prop'].enum_items
		aov_id = aov_items[bpy.context.scene.aov_enum_prop].identifier

		rdotvpow_items = bpy.context.scene.bl_rna.properties['r_dot_v_pow_enum_prop'].enum_items
		rdotvpow_id = rdotvpow_items[bpy.context.scene.r_dot_v_pow_enum_prop].identifier

		skip_refresh_override = False

		if abj_sd_b_instance.aov_stored != aov_id:
			abj_sd_b_instance.aov_stored = aov_id
			skip_refresh_override = True

		if abj_sd_b_instance.rdotvpow_stored != rdotvpow_id:
			abj_sd_b_instance.rdotvpow_stored = rdotvpow_id
			skip_refresh_override = True

		if abj_sd_b_instance.recently_cleared_selFaces == True:
			skip_refresh_override = True
			abj_sd_b_instance.recently_cleared_selFaces = False

		abj_sd_b_instance.profile_stage2_00_b = str(datetime.now() - abj_sd_b_instance.profile_stage2_00_a)
		if abj_sd_b_instance.profileCode_part2 == True:
			print('~~~~~~~~~ abj_sd_b_instance.profile_stage2_00_b = ', abj_sd_b_instance.profile_stage2_00_b)

		################

		abj_sd_b_instance.updateScene()

		abj_sd_b_instance.deselectAll()

		############################################################
		allNames = []
		abj_sd_b_instance.allNamesToToggleDuringRaycast = []
		for i in abj_sd_b_instance.shadingList_perFace:
			allNames.append(i['shadingPlane'])

		for i in bpy.context.scene.objects:
			if i.name not in allNames:
				if i.hide_get() == 0:
					abj_sd_b_instance.allNamesToToggleDuringRaycast.append(i)

		# for i in allNamesToToggleDuringRaycast:
		# 	print(i)

		# print('~~~~~~~~~~~~~~~~~~~ !!!!!!!!!!!! ~~~~~~~~~~~~~~~~~~~')
		# print('~~~~~~~~~~~~~~~~~~~ !!!!!!!!!!!! allNamesToToggleDuringRaycast', abj_sd_b_instance.allNamesToToggleDuringRaycast)
		# print('~~~~~~~~~~~~~~~~~~~ !!!!!!!!!!!! ~~~~~~~~~~~~~~~~~~~')
		############################################################

		# ray_cast_faceCenter_to_V_list = []
		# ray_cast_faceCenter_to_L_list = []

		ray_cast_faceCenter_to_V_dict_list = []
		ray_cast_faceCenter_to_L_dict_list = []

		# myNewList = []

		# deep_copied_list_perFace = None
		deep_copied_list_perFace_all = []

		for i in abj_sd_b_instance.shadingList_perFace:			
			abj_sd_b_instance.profile_stage2_02_a = datetime.now() ################

			mySplitFaceIndexUsable = i['mySplitFaceIndexUsable']

			#################################################
			#decide whether to continue and do a full refresh
			#################################################
			matCheck = bpy.data.materials.get("ShaderVisualizer_" + str(mySplitFaceIndexUsable))
			skip_refresh = False
			if matCheck: #material already exists...check if it is not selected for stage stepping
				for j in abj_sd_b_instance.shadingStages_perFace_stepList:
					if (j["idx"]) == mySplitFaceIndexUsable:
						if j['idx'] not in abj_sd_b_instance.shadingStages_selectedFaces:
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

			spec = i['spec']

			faceCenter_to_V_rayCast = i['faceCenter_to_V_rayCast']
			faceCenter_to_L_rayCast = i['faceCenter_to_L_rayCast']

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

			# print('~~~~~~ SPEC CTRL 0 = ', mySplitFaceIndexUsable, ' ', spec)

			deep_copied_list_perFace = copy.deepcopy(i) ####### DEEP COPY SPEC

			# continue

			##############
			#######
			###########
			#debug
			# abj_sd_b_instance.shadingStages_selectedFaces.clear()
			# abj_sd_b_instance.shadingStages_selectedFaces.append('242')

			#local variables
			faceCenter_to_V_rayCast = None
			faceCenter_to_L_rayCast = None

			#visualize arrows if spec > cutoff
			cutoff = abj_sd_b_instance.spec_cutoff

			abj_sd_b_instance.profile_stage2_02_b = datetime.now() - abj_sd_b_instance.profile_stage2_02_a
			abj_sd_b_instance.profile_stage2_02_final += abj_sd_b_instance.profile_stage2_02_b
			
			N_dot_V_over_threshold_with_ortho_compensateTrick = None
			# if (N_dot_V <= 0):
			if (N_dot_V <= 0.1):

				if abj_sd_b_instance.hideUnHideInitialNdotV == True:
					#Hide for potential ray_cast speed up
					for j in bpy.context.scene.objects:
						if j.name == shadingPlane:
							j.hide_set(1)

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
				V_toFace = mathutils.Vector(faceCenter - abj_sd_b_instance.pos_camera_global_v).normalized()

				ray_cast_faceCenter_to_V_dict = {
					'mySplitFaceIndexUsable' : mySplitFaceIndexUsable,
					'origin' : abj_sd_b_instance.pos_camera_global_v,
					'direction' : V_toFace,
				}

				ray_cast_faceCenter_to_V_dict_list.append(ray_cast_faceCenter_to_V_dict)

			deep_copied_list_perFace['spec'] = spec

			deep_copied_list_perFace_all.append(deep_copied_list_perFace) ###
		
		abj_sd_b_instance.shadingList_perFace = deep_copied_list_perFace_all[:] ###
		deep_copied_list_perFace_all.clear()

		deep_copied_list_perFace_all_02 = []
		deep_copied_list_perFace_02 = None

		abj_sd_b_instance.profile_stage2_03_a = datetime.now() ################
		deep_copied_list_02 = copy.deepcopy(abj_sd_b_instance.shadingList_perFace)

		for i in ray_cast_faceCenter_to_V_dict_list:

			mySplitFaceIndexUsable = i['mySplitFaceIndexUsable']
			origin = i['origin']
			direction = i['direction']
			myRay_faceCenter_to_V_result = abj_sd_b_instance.raycast_abj_scene(origin, direction, mySplitFaceIndexUsable) ########## good

			# myRay_faceCenter_to_V_result = abj_sd_b_instance.raycast_abj_scene(abj_sd_b_instance.pos_camera_global_v, V_toFace, mySplitFaceIndexUsable) ########## good

			if myRay_faceCenter_to_V_result == False:
				# print('behind something else, discard')

				for idx, j in enumerate(abj_sd_b_instance.shadingList_perFace):
					if j['mySplitFaceIndexUsable'] == i['mySplitFaceIndexUsable']:
						# deep_copied_list_perFace_02 = copy.deepcopy(j) ####### DEEP COPY SPEC

						deep_copied_list_02[idx]['spec'] = 0
						deep_copied_list_02[idx]['faceCenter_to_V_rayCast'] = myRay_faceCenter_to_V_result

			elif myRay_faceCenter_to_V_result == True:
				# print('makes it to the cam, now cast again')

				#######################
				#RAYCAST AGAINST L
				#######################
				usableDir = None
				for idx, j in enumerate(abj_sd_b_instance.shadingList_perFace):
					if j['mySplitFaceIndexUsable'] == i['mySplitFaceIndexUsable']:
						deep_copied_list_02[idx]['faceCenter_to_V_rayCast'] = myRay_faceCenter_to_V_result
						usableDir = -j['L']

				ray_cast_faceCenter_to_L_dict = {
					'mySplitFaceIndexUsable' : mySplitFaceIndexUsable,
					'origin' : abj_sd_b_instance.pos_light_global_v,
					# 'direction' : -L,
					'direction' : usableDir,
				}

				ray_cast_faceCenter_to_L_dict_list.append(ray_cast_faceCenter_to_L_dict)

			deep_copied_list_perFace_all_02.append(deep_copied_list_perFace_02) ###

		abj_sd_b_instance.shadingList_perFace = deep_copied_list_02[:] ###
		deep_copied_list_02.clear()

		deep_copied_list_perFace_all_03 = []
		deep_copied_list_perFace_03 = None

		deep_copied_list_03 = copy.deepcopy(abj_sd_b_instance.shadingList_perFace)

		for i in ray_cast_faceCenter_to_L_dict_list:
			mySplitFaceIndexUsable = i['mySplitFaceIndexUsable']
			origin = i['origin']
			direction = i['direction']

			# myRay_faceCenter_to_L_result = abj_sd_b_instance.raycast_abj_scene(abj_sd_b_instance.pos_light_global_v, -L, mySplitFaceIndexUsable)
			myRay_faceCenter_to_L_result = abj_sd_b_instance.raycast_abj_scene(origin, direction, mySplitFaceIndexUsable)

			if myRay_faceCenter_to_L_result == True:
				for idx, j in enumerate(abj_sd_b_instance.shadingList_perFace):
					if j['mySplitFaceIndexUsable'] == i['mySplitFaceIndexUsable']:
						deep_copied_list_03[idx]['faceCenter_to_L_rayCast'] = myRay_faceCenter_to_L_result

				# pass #end shader

			else:
				# spec = 0

				for idx, j in enumerate(abj_sd_b_instance.shadingList_perFace):
					if j['mySplitFaceIndexUsable'] == i['mySplitFaceIndexUsable']:
						deep_copied_list_03[idx]['spec'] = 0
						deep_copied_list_03[idx]['faceCenter_to_L_rayCast'] = myRay_faceCenter_to_L_result

			deep_copied_list_perFace_all_03.append(deep_copied_list_perFace_03) ###

		abj_sd_b_instance.shadingList_perFace = deep_copied_list_03[:] ###
		deep_copied_list_03.clear()

		abj_sd_b_instance.profile_stage2_03_b = datetime.now() - abj_sd_b_instance.profile_stage2_03_a
		abj_sd_b_instance.profile_stage2_03_final += abj_sd_b_instance.profile_stage2_03_b

	def equation_part2_preProcess_arrow_matricies(self, abj_sd_b_instance):
		all_indicies_in_matrix_list = []
		for i in abj_sd_b_instance.myCube1_instance_M_all_list_matrixOnly:
			all_indicies_in_matrix_list.append(i['mySplitFaceIndexUsable'])

		for i in abj_sd_b_instance.shadingStages_selectedFaces:
		# for i in abj_sd_b_instance.myDebugFaces:
			for j in abj_sd_b_instance.shadingList_perFace:	
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

					N_M = abj_sd_b_instance.dynamic_cube1_creation(shadingPlane, faceCenter, mySplitFaceIndexUsable)
					L_M = abj_sd_b_instance.dynamic_cubeLight_creation(faceCenter, mySplitFaceIndexUsable, abj_sd_b_instance.mySun)
					V_M = abj_sd_b_instance.dynamic_cubeV_creation(faceCenter, mySplitFaceIndexUsable, abj_sd_b_instance.myCam)
					R_M = abj_sd_b_instance.dynamic_cube2_creation(faceCenter, mySplitFaceIndexUsable, N_M, R)

					abj_sd_b_instance.updateScene() # need

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

					abj_sd_b_instance.myCube1_instance_M_all_list_matrixOnly.append(myCube1_instance_M_dict) ##########

	def equation_part3_switch_stages(self, abj_sd_b_instance, startTime):
		###print once variables
		printOnce_stage_000 = False
		printOnce_stage_001 = False
		printOnce_stage_002 = False
		printOnce_stage_003 = False
		printOnce_stage_004 = False
		printOnce_stage_005 = False
		printOnce_stage_006 = False
		printOnce_stage_007 = False

		for i in abj_sd_b_instance.shadingList_perFace:	

			mySplitFaceIndexUsable = i['mySplitFaceIndexUsable']

			#################################################
			#decide whether to continue and do a full refresh
			#################################################
			matCheck = bpy.data.materials.get("ShaderVisualizer_" + str(mySplitFaceIndexUsable))
			skip_refresh = False
			if matCheck: #material already exists...check if it is not selected for stage stepping
				for j in abj_sd_b_instance.shadingStages_perFace_stepList:
					if (j["idx"]) == mySplitFaceIndexUsable:
						if j['idx'] not in abj_sd_b_instance.shadingStages_selectedFaces:
							skip_refresh = True

			# if skip_refresh_override == True:
			# 	skip_refresh = False

			if skip_refresh == True:
				continue

			shadingPlane = i['shadingPlane']
			faceCenter = i['faceCenter']
			N_dot_L = i['N_dot_L']
			N_dot_V = i['N_dot_V']
			
			N_dot_V_over_threshold_with_ortho_compensateTrick = None
			if (N_dot_V <= 0.1):
				N_dot_V_over_threshold_with_ortho_compensateTrick = False
			else:
				N_dot_V_over_threshold_with_ortho_compensateTrick = True

			R_dot_V = i['R_dot_V']
			attenuation = i['attenuation']
			L = i['L']
			N = i['N']
			R = i['R']

			spec = i['spec']
			faceCenter_to_V_rayCast = i['faceCenter_to_V_rayCast']
			faceCenter_to_L_rayCast = i['faceCenter_to_L_rayCast']

			# #Hide for raycast speed up
			# if (N_dot_V <= 0.1):
			# 	for j in bpy.context.scene.objects:
			# 		if j.name == shadingPlane:
			# 			j.hide_set(0)

			abj_sd_b_instance.profile_stage2_04_a = datetime.now() ################

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

			for j in abj_sd_b_instance.shadingStages_perFace_stepList:
				if (j["idx"]) == mySplitFaceIndexUsable:
					if j['idx'] not in abj_sd_b_instance.shadingStages_selectedFaces:
						if usableStageCategory_id == 'spec_with_arrow':
							items_id_currentStage = maxRange_usable
							override = True

					elif j['idx'] in abj_sd_b_instance.shadingStages_selectedFaces:
						currentStage = abj_sd_b_instance.myBreakpointList[j['breakpoint_idx']]

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

			abj_sd_b_instance.profile_stage2_04_b = datetime.now() - abj_sd_b_instance.profile_stage2_04_a
			abj_sd_b_instance.profile_stage2_04_final += abj_sd_b_instance.profile_stage2_04_b
			# if abj_sd_b_instance.profileCode_part2 == True:
				# print('~~~~~~~~~ abj_sd_b_instance.profile_stage2_02_b = ', abj_sd_b_instance.profile_stage2_02_b)

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

					abj_sd_b_instance.show_arrow_N(shadingPlane, faceCenter, mySplitFaceIndexUsable)

					abj_sd_b_instance.myCubeCam.hide_set(1)

					abj_sd_b_instance.setActiveStageMaterial(shadingPlane, mySplitFaceIndexUsable, abj_sd_b_instance.shadingPlane_sel_r, abj_sd_b_instance.shadingPlane_sel_g, abj_sd_b_instance.shadingPlane_sel_b)

				elif items_id_currentStage == 1:
					if printOnce_stage_001 == False:
						print("'stage_001' : 'V....show V arrow (myCubeCam)'")
						printOnce_stage_001 = True

					abj_sd_b_instance.myCubeCam.hide_set(0)

					abj_sd_b_instance.setActiveStageMaterial(shadingPlane, mySplitFaceIndexUsable, abj_sd_b_instance.shadingPlane_sel_r, abj_sd_b_instance.shadingPlane_sel_g, abj_sd_b_instance.shadingPlane_sel_b)

				elif items_id_currentStage == 2:
					if printOnce_stage_002 == False:
						print("'stage_002' : 'N_dot_V......show both myCube1 and myCubeCam'")
						printOnce_stage_002 = True

					abj_sd_b_instance.show_arrow_N(shadingPlane, faceCenter, mySplitFaceIndexUsable)

					abj_sd_b_instance.myCubeCam.hide_set(0)

					abj_sd_b_instance.setActiveStageMaterial(shadingPlane, mySplitFaceIndexUsable, abj_sd_b_instance.shadingPlane_sel_r, abj_sd_b_instance.shadingPlane_sel_g, abj_sd_b_instance.shadingPlane_sel_b)

				elif items_id_currentStage == 3:
					if N_dot_V_over_threshold_with_ortho_compensateTrick == False: #####
						if abj_sd_b_instance.printDetailedInfo == True:
							print('N_dot_V_over_threshold_with_ortho_compensateTrick FAIL for ', mySplitFaceIndexUsable)

						self.equation_aov_output(aov_id, shadingPlane, mySplitFaceIndexUsable, N_dot_L, spec, attenuation, abj_sd_b_instance)

					elif N_dot_V_over_threshold_with_ortho_compensateTrick == True or override == True:
						if printOnce_stage_003 == False:
							print('N_dot_V over ortho compensate trick, so continue...', N_dot_V_over_threshold_with_ortho_compensateTrick)
							print("'stage_003' : 'raycast from faceCenter to V'")
							printOnce_stage_003 = True

						abj_sd_b_instance.profile_stage2_05_a = datetime.now() ################

						abj_sd_b_instance.show_arrow_V_to_faceCenter(faceCenter, mySplitFaceIndexUsable)

						abj_sd_b_instance.setActiveStageMaterial(shadingPlane, mySplitFaceIndexUsable, abj_sd_b_instance.shadingPlane_sel_r, abj_sd_b_instance.shadingPlane_sel_g, abj_sd_b_instance.shadingPlane_sel_b)

						abj_sd_b_instance.profile_stage2_05_b = datetime.now() - abj_sd_b_instance.profile_stage2_05_a
						abj_sd_b_instance.profile_stage2_05_final += abj_sd_b_instance.profile_stage2_05_b

						abj_sd_b_instance.myCubeCam.hide_set(1)

				elif items_id_currentStage == 4:
					if faceCenter_to_V_rayCast == False: ####
						if abj_sd_b_instance.printDetailedInfo == True:
							print('faceCenter_to_V_rayCast FAIL for ', mySplitFaceIndexUsable)

						self.equation_aov_output(aov_id, shadingPlane, mySplitFaceIndexUsable, N_dot_L, spec, attenuation, abj_sd_b_instance)

					elif faceCenter_to_V_rayCast == True or override == True:
							if printOnce_stage_004 == False:
								print('faceCenter_to_V_rayCast was TRUE so continue... ', faceCenter_to_V_rayCast)
								print("'stage_004' : 'raycast from faceCenter to L'")
								printOnce_stage_004 = True

							abj_sd_b_instance.show_arrow_L_to_faceCenter(faceCenter, mySplitFaceIndexUsable)

							abj_sd_b_instance.myCubeCam.hide_set(1)

							abj_sd_b_instance.setActiveStageMaterial(shadingPlane, mySplitFaceIndexUsable, abj_sd_b_instance.shadingPlane_sel_r, abj_sd_b_instance.shadingPlane_sel_g, abj_sd_b_instance.shadingPlane_sel_b)

				elif items_id_currentStage == 5:
					# if mySplitFaceIndexUsable == '242':
					# 	print('IN 005')
					# 	print('skip_refresh = ', skip_refresh)
					# 	print('printOnce_stage_005 = ', printOnce_stage_005)
					# 	print('faceCenter_to_L_rayCast for 242 = ', faceCenter_to_L_rayCast)
					# 	print('faceCenter_to_V_rayCast for 242 = ', faceCenter_to_V_rayCast)

					if faceCenter_to_L_rayCast == False: ####
						if abj_sd_b_instance.printDetailedInfo == True:
							print('faceCenter_to_L_rayCast FAIL for ', mySplitFaceIndexUsable)

						self.equation_aov_output(aov_id, shadingPlane, mySplitFaceIndexUsable, N_dot_L, spec, attenuation, abj_sd_b_instance)

					elif faceCenter_to_L_rayCast == True or override == True:
						if printOnce_stage_005 == False:
							print('faceCenter_to_L_rayCast was TRUE so continue... ', faceCenter_to_V_rayCast)
							print("'stage_005' : 'show arrows (N, L)'")
							printOnce_stage_005 = True

						abj_sd_b_instance.show_arrow_N(shadingPlane, faceCenter, mySplitFaceIndexUsable)

						##############

						abj_sd_b_instance.show_arrow_L_to_faceCenter(faceCenter, mySplitFaceIndexUsable)

						abj_sd_b_instance.myCubeCam.hide_set(1)

						abj_sd_b_instance.setActiveStageMaterial(shadingPlane, mySplitFaceIndexUsable, abj_sd_b_instance.shadingPlane_sel_r, abj_sd_b_instance.shadingPlane_sel_g, abj_sd_b_instance.shadingPlane_sel_b)

				elif items_id_currentStage == 6:
					if faceCenter_to_L_rayCast == True or override == True:
						if printOnce_stage_006 == False:
							print("'stage_006' : 'R.....show R arrow (cube2) along with N and L'")
							printOnce_stage_006 = True

						abj_sd_b_instance.show_arrow_N(shadingPlane, faceCenter, mySplitFaceIndexUsable)
						abj_sd_b_instance.show_arrow_L_to_faceCenter(faceCenter, mySplitFaceIndexUsable)
						abj_sd_b_instance.show_arrow_R(faceCenter, mySplitFaceIndexUsable, L, N)

						abj_sd_b_instance.myCubeCam.hide_set(1)

						abj_sd_b_instance.setActiveStageMaterial(shadingPlane, mySplitFaceIndexUsable, abj_sd_b_instance.shadingPlane_sel_r, abj_sd_b_instance.shadingPlane_sel_g, abj_sd_b_instance.shadingPlane_sel_b)

				elif items_id_currentStage == 7:
					if printOnce_stage_007 == False:
						print('stage_007 output AOV = ', aov_id)
						printOnce_stage_007 = True

					abj_sd_b_instance.profile_stage2_06_a = datetime.now() ################


					abj_sd_b_instance.myCubeCam.hide_set(1)

					self.equation_aov_output(aov_id, shadingPlane, mySplitFaceIndexUsable, N_dot_L, spec, attenuation, abj_sd_b_instance)

					abj_sd_b_instance.profile_stage2_06_b = datetime.now() - abj_sd_b_instance.profile_stage2_06_a
					abj_sd_b_instance.profile_stage2_06_final += abj_sd_b_instance.profile_stage2_06_b


			# if abj_sd_b_instance.profileCode_part2 == True:
				# print('~~~~~~~~~ abj_sd_b_instance.profile_stage2_03_b = ', abj_sd_b_instance.profile_stage2_03_b)

		if abj_sd_b_instance.profileCode_part2 == True:
			# print('~~~~~~~~~ abj_sd_b_instance.profile_stage2_00_final = ', abj_sd_b_instance.profile_stage2_00_final)
			# print('~~~~~~~~~ abj_sd_b_instance.profile_stage2_01_final = ', abj_sd_b_instance.profile_stage2_01_final)
			# print('~~~~~~~~~ abj_sd_b_instance.profile_stage2_02_final () = ', abj_sd_b_instance.profile_stage2_02_final)
			
			print('~~~~~~~~~ abj_sd_b_instance.profile_stage2_03_final (raycast) = ', abj_sd_b_instance.profile_stage2_03_final)
			print('~~~~~~~~~ abj_sd_b_instance.profile_stage2_04_final (set stage) = ', abj_sd_b_instance.profile_stage2_04_final)
			print('~~~~~~~~~ abj_sd_b_instance.profile_stage2_05_final (stage 3) = ', abj_sd_b_instance.profile_stage2_05_final)

			print('~~~~~~~~~ abj_sd_b_instance.profile_stage2_06_final (stage 7) = ', abj_sd_b_instance.profile_stage2_06_final)
			# print('~~~~~~~~~ abj_sd_b_instance.profile_stage2_07_final = ', abj_sd_b_instance.profile_stage2_07_final)
			# print('~~~~~~~~~ abj_sd_b_instance.profile_stage2_08_final = ', abj_sd_b_instance.profile_stage2_08_final)
			# print('~~~~~~~~~ abj_sd_b_instance.profile_stage2_09_final = ', abj_sd_b_instance.profile_stage2_09_final)
			# print('~~~~~~~~~ abj_sd_b_instance.profile_stage2_10_final = ', abj_sd_b_instance.profile_stage2_10_final)

			# print('~~~~~~~~~ abj_sd_b_instance.profile_stage2_08_b = ', abj_sd_b_instance.profile_stage2_08_b)

		# abj_sd_b_instance.myCubeCam.hide_set(1)

		print('TIME TO COMPLETE (render) = ' + str(datetime.now() - startTime))
		print(' ')


	def equation_aov_output(self, aov_id, shadingPlane, mySplitFaceIndexUsable, N_dot_L, spec, attenuation, abj_sd_b_instance):
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

		if abj_sd_b_instance.specTesterMatToggle == -1:
			for j in bpy.context.scene.objects:
				if j.name == shadingPlane:
					bpy.context.view_layer.objects.active = j

			mat1 = abj_sd_b_instance.newShader("ShaderVisualizer_" + str(mySplitFaceIndexUsable), "emission", Ci, Ci, Ci)
			bpy.context.active_object.data.materials.clear()
			bpy.context.active_object.data.materials.append(mat1)