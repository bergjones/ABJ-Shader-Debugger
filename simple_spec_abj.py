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

	def equation_part2_preProcess(self, abj_sd_b_instance):
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

		ray_cast_faceCenter_to_V_dict_list = []
		ray_cast_faceCenter_to_L_dict_list = []

		# deep_copied_list_perFace = None
		deep_copied_list_perFace_all = []

		for i in abj_sd_b_instance.shadingList_perFace:			
			abj_sd_b_instance.profile_stage2_02_a = datetime.now() ################

			mySplitFaceIndexUsable = i['mySplitFaceIndexUsable']

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

			deep_copied_list_perFace = copy.deepcopy(i) ####### DEEP COPY SPEC

			#local variables
			faceCenter_to_V_rayCast = None
			faceCenter_to_L_rayCast = None

			abj_sd_b_instance.profile_stage2_02_b = datetime.now() - abj_sd_b_instance.profile_stage2_02_a
			abj_sd_b_instance.profile_stage2_02_final += abj_sd_b_instance.profile_stage2_02_b
			
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
		for i in abj_sd_b_instance.arrow_dynamic_instance_M_all_list_matrixOnly:
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
					H = j['H']

					N_M = abj_sd_b_instance.equation_dynamic_cubeN_creation(shadingPlane, faceCenter)
					L_M = abj_sd_b_instance.equation_dynamic_cubeLight_creation(faceCenter, abj_sd_b_instance.mySun)
					V_M = abj_sd_b_instance.equation_dynamic_cubeV_creation(faceCenter, abj_sd_b_instance.myCam)
					R_M = abj_sd_b_instance.equation_dynamic_cubeR_creation(N_M, R)
					H_M = abj_sd_b_instance.equation_dynamic_cubeH_creation(faceCenter, H, abj_sd_b_instance.myCam)

					abj_sd_b_instance.updateScene() # need

					N_M_np = np.array(N_M)
					L_M_np = np.array(L_M)
					V_M_np = np.array(V_M)
					R_M_np = np.array(R_M)
					H_M_np = np.array(H_M)

					arrow_dynamic_instance_M_dict = {
						'mySplitFaceIndexUsable' : mySplitFaceIndexUsable,
						'N_M_np' : N_M_np,
						'L_M_np' : L_M_np,
						'V_M_np' : V_M_np,
						'R_M_np' : R_M_np,
						'H_M_np' : H_M_np,
					}

					abj_sd_b_instance.arrow_dynamic_instance_M_all_list_matrixOnly.append(arrow_dynamic_instance_M_dict) ##########
		
	def equation_part3_switch_stages(self, abj_sd_b_instance):
		###print once variables
		printOnce_stage_000 = False
		printOnce_stage_001 = False
		printOnce_stage_002 = False
		printOnce_stage_003 = False
		printOnce_stage_004 = False
		printOnce_stage_005 = False
		printOnce_stage_006 = False
		printOnce_stage_007 = False

		aov_items = bpy.context.scene.bl_rna.properties['aov_enum_prop'].enum_items
		aov_id = aov_items[bpy.context.scene.aov_enum_prop].identifier

		rdotvpow_items = bpy.context.scene.bl_rna.properties['r_dot_v_pow_enum_prop'].enum_items
		rdotvpow_id = rdotvpow_items[bpy.context.scene.r_dot_v_pow_enum_prop].identifier

		for i in abj_sd_b_instance.shadingList_perFace:	
			mySplitFaceIndexUsable = i['mySplitFaceIndexUsable']

			if abj_sd_b_instance.skip_refresh_determine(mySplitFaceIndexUsable) == True:
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
			if abj_sd_b_instance.skip_showing_visibility_raycast_check == True:
				maxRange_usable = 2
			else:
				maxRange_usable = 7

			items_id_currentStage = None
			override = False

			for j in abj_sd_b_instance.shadingStages_perFace_stepList:
				if (j["idx"]) == mySplitFaceIndexUsable:
					if j['idx'] not in abj_sd_b_instance.shadingStages_selectedFaces:
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
			aov_items = bpy.context.scene.bl_rna.properties['aov_enum_prop'].enum_items
			aov_id = aov_items[bpy.context.scene.aov_enum_prop].identifier

			abj_sd_b_instance.profile_stage2_04_b = datetime.now() - abj_sd_b_instance.profile_stage2_04_a
			abj_sd_b_instance.profile_stage2_04_final += abj_sd_b_instance.profile_stage2_04_b
			# if abj_sd_b_instance.profileCode_part2 == True:
				# print('~~~~~~~~~ abj_sd_b_instance.profile_stage2_02_b = ', abj_sd_b_instance.profile_stage2_02_b)

			###############
			### Simple Specular Breakdown
			##############
			if abj_sd_b_instance.renderPasses_simple == True:

				continue

				# abj_sd_b_instance.myCubeCam.hide_set(1)
				# abj_sd_b_instance.mySun.hide_set(1)
				# abj_sd_b_instance.mySun.hide_render = True

				if faceCenter_to_L_rayCast == False:
					continue

				else:

					if items_id_currentStage == 0:
						if printOnce_stage_000 == False:
							print("'stage_000' : 'show N'")
							printOnce_stage_000 = True

						myArrow = abj_sd_b_instance.show_arrow_N(shadingPlane, faceCenter, mySplitFaceIndexUsable)

						arrowScale = 0.1
						myArrow.scale = mathutils.Vector((.05, .2, 1)) ####

					elif items_id_currentStage == 1:
						if printOnce_stage_001 == False:
							print("'stage_001' : 'show L'")
							printOnce_stage_001 = True

						abj_sd_b_instance.show_arrow_L_to_faceCenter(faceCenter, mySplitFaceIndexUsable)

					elif items_id_currentStage == 2:
						if printOnce_stage_002 == False:
							print("'stage_002' : 'show R'")
							printOnce_stage_002 = True

						myArrow = abj_sd_b_instance.show_arrow_R(faceCenter, mySplitFaceIndexUsable, L, N)

						myArrow.scale = mathutils.Vector((.05, .2, 1)) ####

					elif items_id_currentStage == 3:
						if printOnce_stage_003 == False:
							print("'stage_003' : 'show H'")
							printOnce_stage_003 = True

						myArrow = abj_sd_b_instance.show_arrow_H(shadingPlane, faceCenter, mySplitFaceIndexUsable)
						# myArrow.scale = mathutils.Vector((.05, .2, 1)) ####
						# myArrow.scale = mathutils.Vector((.2, .8, 1)) ####
						myArrow.scale = mathutils.Vector((1, 1, 1)) ####

					elif items_id_currentStage == 4:
						if printOnce_stage_004 == False:
							print("'stage_004' : 'show V'")
							printOnce_stage_004 = True

							abj_sd_b_instance.myCubeCam.hide_set(0)

			else:
				if abj_sd_b_instance.skip_showing_visibility_raycast_check == True:
					if items_id_currentStage == 0:
						if printOnce_stage_000 == False:
							print("'stage_000' : 'N_dot_V......show both myCubeN and myCubeCam (V)'")
							printOnce_stage_000 = True

						abj_sd_b_instance.show_arrow_N(shadingPlane, faceCenter, mySplitFaceIndexUsable)
						abj_sd_b_instance.myCubeCam.hide_set(0)
						abj_sd_b_instance.selectedFaceMat_temp_list.append(mySplitFaceIndexUsable)

					elif items_id_currentStage == 1:
						if N_dot_V_over_threshold_with_ortho_compensateTrick == False: #####
							# if abj_sd_b_instance.printDetailedInfo == True:
								# print('N_dot_V_over_threshold_with_ortho_compensateTrick FAIL for ', mySplitFaceIndexUsable)

							abj_sd_b_instance.Ci_render_temp_list.append(mySplitFaceIndexUsable)

						elif N_dot_V_over_threshold_with_ortho_compensateTrick == True or override == True:
							if faceCenter_to_V_rayCast == False or faceCenter_to_L_rayCast == False: ####
								# if abj_sd_b_instance.printDetailedInfo == True:
									# print('faceCenter_to_V_rayCast FAIL for ', mySplitFaceIndexUsable)

								abj_sd_b_instance.Ci_render_temp_list.append(mySplitFaceIndexUsable)

							else:
								if printOnce_stage_001 == False:
									print('N_dot_V over ortho compensate trick, so continue...', N_dot_V_over_threshold_with_ortho_compensateTrick)
									print("'stage_001' : 'R.....show R arrow (cubeR) along with N and L'")
									printOnce_stage_001 = True

								abj_sd_b_instance.show_arrow_N(shadingPlane, faceCenter, mySplitFaceIndexUsable)
								abj_sd_b_instance.show_arrow_L_to_faceCenter(faceCenter, mySplitFaceIndexUsable)
								abj_sd_b_instance.show_arrow_R(faceCenter, mySplitFaceIndexUsable, L, N)

								abj_sd_b_instance.selectedFaceMat_temp_list.append(mySplitFaceIndexUsable)

					elif items_id_currentStage == 2:
						if printOnce_stage_002 == False:
							print('stage_002 output AOV = ', aov_id)
							printOnce_stage_002 = True

						abj_sd_b_instance.Ci_render_temp_list.append(mySplitFaceIndexUsable)

				else:
					if items_id_currentStage == 0:
						if printOnce_stage_000 == False:
							print("'stage_000' : 'N....show N arrow (cubeN)'")
							printOnce_stage_000 = True

						abj_sd_b_instance.show_arrow_N(shadingPlane, faceCenter, mySplitFaceIndexUsable)

						abj_sd_b_instance.myCubeCam.hide_set(1)

						abj_sd_b_instance.selectedFaceMat_temp_list.append(mySplitFaceIndexUsable)

					elif items_id_currentStage == 1:
						if printOnce_stage_001 == False:
							print("'stage_001' : 'V....show V arrow (myCubeCam)'")
							printOnce_stage_001 = True

						abj_sd_b_instance.myCubeCam.hide_set(0)

						abj_sd_b_instance.selectedFaceMat_temp_list.append(mySplitFaceIndexUsable)

					elif items_id_currentStage == 2:
						if printOnce_stage_002 == False:
							print("'stage_002' : 'N_dot_V......show both myCubeN and myCubeCam'")
							printOnce_stage_002 = True

						abj_sd_b_instance.show_arrow_N(shadingPlane, faceCenter, mySplitFaceIndexUsable)

						abj_sd_b_instance.myCubeCam.hide_set(0)

						abj_sd_b_instance.selectedFaceMat_temp_list.append(mySplitFaceIndexUsable)

					elif items_id_currentStage == 3:
						if N_dot_V_over_threshold_with_ortho_compensateTrick == False: #####
							if abj_sd_b_instance.printDetailedInfo == True:
								print('N_dot_V_over_threshold_with_ortho_compensateTrick FAIL for ', mySplitFaceIndexUsable)

							abj_sd_b_instance.Ci_render_temp_list.append(mySplitFaceIndexUsable)

						elif N_dot_V_over_threshold_with_ortho_compensateTrick == True or override == True:
							if printOnce_stage_003 == False:
								print('N_dot_V over ortho compensate trick, so continue...', N_dot_V_over_threshold_with_ortho_compensateTrick)
								print("'stage_003' : 'raycast from faceCenter to V'")
								printOnce_stage_003 = True

							abj_sd_b_instance.show_arrow_V_to_faceCenter(faceCenter, mySplitFaceIndexUsable)

							abj_sd_b_instance.myCubeCam.hide_set(1)

							abj_sd_b_instance.selectedFaceMat_temp_list.append(mySplitFaceIndexUsable)

					elif items_id_currentStage == 4:
						if faceCenter_to_V_rayCast == False: ####
							if abj_sd_b_instance.printDetailedInfo == True:
								print('faceCenter_to_V_rayCast FAIL for ', mySplitFaceIndexUsable)

							abj_sd_b_instance.Ci_render_temp_list.append(mySplitFaceIndexUsable)

						elif faceCenter_to_V_rayCast == True or override == True:
								if printOnce_stage_004 == False:
									print('faceCenter_to_V_rayCast was TRUE so continue... ', faceCenter_to_V_rayCast)
									print("'stage_004' : 'raycast from faceCenter to L'")
									printOnce_stage_004 = True

								abj_sd_b_instance.show_arrow_L_to_faceCenter(faceCenter, mySplitFaceIndexUsable)

								abj_sd_b_instance.selectedFaceMat_temp_list.append(mySplitFaceIndexUsable)

					elif items_id_currentStage == 5:
						if faceCenter_to_L_rayCast == False: ####
							if abj_sd_b_instance.printDetailedInfo == True:
								print('faceCenter_to_L_rayCast FAIL for ', mySplitFaceIndexUsable)

							abj_sd_b_instance.Ci_render_temp_list.append(mySplitFaceIndexUsable)

						elif faceCenter_to_L_rayCast == True or override == True:
							if printOnce_stage_005 == False:
								print('faceCenter_to_L_rayCast was TRUE so continue... ', faceCenter_to_V_rayCast)
								print("'stage_005' : 'show arrows (N, L)'")
								printOnce_stage_005 = True

							abj_sd_b_instance.show_arrow_N(shadingPlane, faceCenter, mySplitFaceIndexUsable)

							##############

							abj_sd_b_instance.show_arrow_L_to_faceCenter(faceCenter, mySplitFaceIndexUsable)

							abj_sd_b_instance.selectedFaceMat_temp_list.append(mySplitFaceIndexUsable)

					elif items_id_currentStage == 6:
						if faceCenter_to_L_rayCast == True or override == True:
							if printOnce_stage_006 == False:
								print("'stage_006' : 'R.....show R arrow (cubeR) along with N and L'")
								printOnce_stage_006 = True

							abj_sd_b_instance.show_arrow_N(shadingPlane, faceCenter, mySplitFaceIndexUsable)
							abj_sd_b_instance.show_arrow_L_to_faceCenter(faceCenter, mySplitFaceIndexUsable)
							abj_sd_b_instance.show_arrow_R(faceCenter, mySplitFaceIndexUsable, L, N)

							abj_sd_b_instance.selectedFaceMat_temp_list.append(mySplitFaceIndexUsable)

					elif items_id_currentStage == 7:
						if printOnce_stage_007 == False:
							print('stage_007 output AOV = ', aov_id)
							printOnce_stage_007 = True

						# abj_sd_b_instance.myCubeCam.hide_set(1)

						abj_sd_b_instance.Ci_render_temp_list.append(mySplitFaceIndexUsable)



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