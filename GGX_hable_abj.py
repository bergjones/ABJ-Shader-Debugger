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

'''
This file contains code derived from optimized-ggx.hlsl by John Hable, in the Public Domain, and is considered a derivative work under the GNU General Public License v3 (GPLv3). 

optimized-ggx.hlsl (Public Domain) by John Hable
http://filmicworlds.com/blog/optimizing-ggx-shaders-with-dotlh/

or

https://gist.github.com/alvalau/c4aa35069ff339da5e59e5ed9c4db192

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

class myEquation_GGX:
	def __init__(self):
		super(myEquation_GGX, self).__init__()

	def testPrint(self):
		myString = '~~~~~~~~~~~ myString...from TestPrint in GGX_hable_abj 006'
		return myString

	# This function is based on a public domain algorithm found at [http://filmicworlds.com/blog/optimizing-ggx-shaders-with-dotlh/ or https://gist.github.com/alvalau/c4aa35069ff339da5e59e5ed9c4db192]
	def G1V(self, dotNV, k):
		return 1.0 / (dotNV * (1.0 - k) + k)
	
	# This function is based on a public domain algorithm found at [http://filmicworlds.com/blog/optimizing-ggx-shaders-with-dotlh/ or https://gist.github.com/alvalau/c4aa35069ff339da5e59e5ed9c4db192]
	def LightingFuncGGX_REF(self, N, V, L, H, roughness, F0, abj_sd_b_instance):
		alpha = roughness * roughness

		# H = mathutils.Vector(V + L).normalized()

		dotNL = abj_sd_b_instance.clamp(N.dot(L), 0, 1)
		dotNV = abj_sd_b_instance.clamp(N.dot(V), 0, 1)
		dotNH = abj_sd_b_instance.clamp(N.dot(H), 0, 1)
		dotLH = abj_sd_b_instance.clamp(L.dot(H), 0, 1)

		# D = microfacet distrobution (GGX)
		alphaSqr = alpha * alpha
		pi = 3.14159
		denom = dotNH * dotNH * (alphaSqr - 1.0) + 1.0
		D = alphaSqr / (pi * denom * denom)

		# F = fresnel
		dotLH5 = pow(1.0 - dotLH, 5)
		F = F0 + (1.0 - F0) * (dotLH5)

		# V - shadowing
		k = alpha / 2.0
		vis = self.G1V(dotNL, k) * self.G1V(dotNV, k)

		specular = dotNL * D * F * vis
		# specular = abj_sd_b_instance.clamp(specular, 0, 1)

		return specular
	
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
			H = i['H']

			spec = i['spec']
			V = abj_sd_b_instance.myV

			faceCenter_to_V_rayCast = i['faceCenter_to_V_rayCast']
			faceCenter_to_L_rayCast = i['faceCenter_to_L_rayCast']

			ggx_roughness = bpy.context.scene.ggx_roughness_prop
			ggx_fresnel = bpy.context.scene.ggx_fresnel_prop

			spec = self.LightingFuncGGX_REF(N, V, L, H, ggx_roughness, ggx_fresnel, abj_sd_b_instance)

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

	# This function is based on a public domain algorithm found at [http://filmicworlds.com/blog/optimizing-ggx-shaders-with-dotlh/ or https://gist.github.com/alvalau/c4aa35069ff339da5e59e5ed9c4db192]
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
		printOnce_stage_008 = False
		printOnce_stage_009 = False
		printOnce_stage_010 = False
		printOnce_stage_011 = False
		printOnce_stage_012 = False
		printOnce_stage_013 = False
		printOnce_stage_014 = False
		printOnce_stage_015 = False
		printOnce_stage_016 = False
		printOnce_stage_017 = False
		printOnce_stage_018 = False
		printOnce_stage_019 = False
		printOnce_stage_020 = False
		printOnce_stage_021 = False
		printOnce_stage_022 = False
		printOnce_stage_023 = False
		printOnce_stage_024 = False
		printOnce_stage_025 = False
		printOnce_stage_026 = False
		printOnce_stage_027 = False
		printOnce_stage_028 = False
		printOnce_stage_029 = False
		printOnce_stage_030 = False
		printOnce_stage_031 = False
		printOnce_stage_032 = False
		printOnce_stage_033 = False
		printOnce_stage_034 = False
		printOnce_stage_035 = False
		printOnce_stage_036 = False

		aov_items = bpy.context.scene.bl_rna.properties['aov_enum_prop'].enum_items
		aov_id = aov_items[bpy.context.scene.aov_enum_prop].identifier

		for i in abj_sd_b_instance.shadingList_perFace:	
			mySplitFaceIndexUsable = i['mySplitFaceIndexUsable']

			if abj_sd_b_instance.skip_refresh_determine(mySplitFaceIndexUsable) == True:
				continue

			shadingPlane = i['shadingPlane']
			faceCenter = i['faceCenter']
			N_dot_L = i['N_dot_L']
			N_dot_V = i['N_dot_V']
			
			N_dot_V_over_threshold_with_ortho_compensateTrick = None
			# if (N_dot_V <= 0):
			if (N_dot_V <= 0.1):
				N_dot_V_over_threshold_with_ortho_compensateTrick = False
			else:
				N_dot_V_over_threshold_with_ortho_compensateTrick = True

			R_dot_V = i['R_dot_V']
			attenuation = i['attenuation']
			L = i['L']
			N = i['N']
			R = i['R']
			H = i['H']
			V = abj_sd_b_instance.myV

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
				maxRange_usable = 17
			else:
				maxRange_usable = 17

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


			#temp debug override
			# items_id_currentStage = maxRange_usable
			# override = True

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
			### GGX Breakdown
			##############
			# shadingDict_GGX_visualization = {
			# 	'description' : 'GGX Visualization',
			
			##############
			####### D - microfacet GGX distrobution
			##############
			#show N arrow

			#'V + L....show V arrow and L arrow',
			#'V + L + H....show V arrow and L arrow',

			# dotNH - show N and H
			# dotNH * dotNH
			# alphaSqr = alpha * alpha
			# denom = dotNH * dotNH * (alphaSqr - 1.0) + 1
			# D = pi * denom * denom
			# D = alphaSqr / (pi * denom * denom)
			# temp material for D

			##############
			####### F - Fresnel
			##############
			# ggx_roughness = bpy.context.scene.ggx_roughness_prop
			# ggx_fresnel = bpy.context.scene.ggx_fresnel_prop

			# LdotH - show L and H
			# dotLH5 = pow(1.0 - dotLH, 5)
			# F0
			# 1 - F0
			# F0 + (1.0 - F0)
			# F0 + (1.0 - F0) * (dotLH5)

			##############
			####### V - shadowing
			##############
			# roughness
			# alpha = roughness * roughness
			# k = alpha / 2.0
			# k0 = 1 - k

			########## G1V_01 = 1.0 / (dotNL * (1.0 - k) + k)
			# dotNL - show N and L
			# dotNL * k0
			# G1V_01 = dotNL * k0 + k

			########## G1V_02 = 1.0 / (dotNV * (1.0 - k) + k)
			# dotNV - show N and L
			# dotNV * k0
			# G1V_02 = dotNV * k0 + k

			# show G1V_01 and G1V_02
			# invert both - 1 / G1V_01 and 1 / G1V_02
			# multiply together
			# vis = 1 / G1V_01 * 1 / G1V_02

			##############
			####### Final
			##############

			# show dotNL - N and L
			# show D
			# show F
			# show vis

			# if N_dot_V_over_threshold_with_ortho_compensateTrick == True or override == True:
				# if faceCenter_to_V_rayCast == True or faceCenter_to_L_rayCast == True: ####

			if faceCenter_to_L_rayCast == True or override == True:
					##############
					### D
					##############
					if items_id_currentStage == 0:
						if printOnce_stage_000 == False:
							print("'stage_000' : 'N....show N arrow (cubeN)'")
							printOnce_stage_000 = True

						abj_sd_b_instance.show_arrow_N(shadingPlane, faceCenter, mySplitFaceIndexUsable)

						abj_sd_b_instance.myCubeCam.hide_set(1)

						abj_sd_b_instance.selectedFaceMat_temp_list.append(mySplitFaceIndexUsable)

					elif items_id_currentStage == 1:
						if printOnce_stage_001 == False:
							print("'stage_001' : 'V + L....show V arrow and L arrow'")
							printOnce_stage_001 = True

						abj_sd_b_instance.myCubeCam.hide_set(0) # V

						abj_sd_b_instance.show_arrow_L_to_faceCenter(faceCenter, mySplitFaceIndexUsable)

						abj_sd_b_instance.selectedFaceMat_temp_list.append(mySplitFaceIndexUsable)

					elif items_id_currentStage == 2:
						if printOnce_stage_002 == False:
							print("'stage_002' : 'V + L + H....show V arrow and L arrow and H arrow'")
							print("'stage_002' : 'H = mathutils.Vector(V + L).normalized()")
							printOnce_stage_002 = True

						abj_sd_b_instance.myCubeCam.hide_set(0) # V

						abj_sd_b_instance.show_arrow_L_to_faceCenter(faceCenter, mySplitFaceIndexUsable)

						abj_sd_b_instance.show_arrow_H(shadingPlane, faceCenter, mySplitFaceIndexUsable)

						abj_sd_b_instance.selectedFaceMat_temp_list.append(mySplitFaceIndexUsable)

					elif items_id_currentStage == 3:
						if printOnce_stage_003 == False:
							print("'stage_003' : 'dotNH....show N and H arrows'")
							printOnce_stage_003 = True

						abj_sd_b_instance.show_arrow_N(shadingPlane, faceCenter, mySplitFaceIndexUsable)

						abj_sd_b_instance.show_arrow_H(shadingPlane, faceCenter, mySplitFaceIndexUsable)

						abj_sd_b_instance.myCubeCam.hide_set(1)

						dotNH_temp = abj_sd_b_instance.clamp(N.dot(H), 0, 1)

						dotNH_temp = pow(dotNH_temp, (1.0 / 2.2))

						abj_sd_b_instance.setActiveStageMaterial(shadingPlane, mySplitFaceIndexUsable, dotNH_temp, dotNH_temp, dotNH_temp)

					elif items_id_currentStage == 4:
						if printOnce_stage_004 == False:
							print("'stage_004' : 'dotNH * dotNH'")
							printOnce_stage_004 = True

						dotNH_temp = abj_sd_b_instance.clamp(N.dot(H), 0, 1)
						dotNH2 = dotNH_temp * dotNH_temp
						dotNH2 = pow(dotNH2, (1.0 / 2.2))

						abj_sd_b_instance.setActiveStageMaterial(shadingPlane, mySplitFaceIndexUsable, dotNH2, dotNH2, dotNH2)

					elif items_id_currentStage == 5:
						if printOnce_stage_005 == False:
							print("'stage_005' : 'roughness'")
							printOnce_stage_005 = True

						roughness_temp = bpy.context.scene.ggx_roughness_prop

						roughness_temp = pow(roughness_temp, (1.0 / 2.2))

						abj_sd_b_instance.setActiveStageMaterial(shadingPlane, mySplitFaceIndexUsable, roughness_temp, roughness_temp, roughness_temp)	

					elif items_id_currentStage == 6:
						if printOnce_stage_006 == False:
							print("'stage_006' : 'alpha = roughness * roughness'")
							printOnce_stage_006 = True

						alpha_temp = bpy.context.scene.ggx_roughness_prop * bpy.context.scene.ggx_roughness_prop
						alpha_temp = pow(alpha_temp, (1.0 / 2.2))

						abj_sd_b_instance.setActiveStageMaterial(shadingPlane, mySplitFaceIndexUsable, alpha_temp, alpha_temp, alpha_temp)

					elif items_id_currentStage == 7:
						if printOnce_stage_007 == False:
							print("'stage_007' : 'alphaSqr = alpha * alpha'")
							printOnce_stage_007 = True

						alpha_temp = bpy.context.scene.ggx_roughness_prop * bpy.context.scene.ggx_roughness_prop
						alphaSqr = alpha_temp * alpha_temp
						alphaSqr = pow(alphaSqr, (1.0 / 2.2))
						abj_sd_b_instance.setActiveStageMaterial(shadingPlane, mySplitFaceIndexUsable, alphaSqr, alphaSqr, alphaSqr)

					elif items_id_currentStage == 8:
						if printOnce_stage_008 == False:
							print("'stage_008' : denom = dotNH * dotNH * (alphaSqr - 1.0) + 1")
							printOnce_stage_008 = True

						dotNH_temp = abj_sd_b_instance.clamp(N.dot(H), 0, 1)
						dotNH2 = dotNH_temp * dotNH_temp

						alpha_temp = bpy.context.scene.ggx_roughness_prop * bpy.context.scene.ggx_roughness_prop
						alphaSqr = alpha_temp * alpha_temp

						denom = dotNH2 * (alphaSqr - 1.0) + 1.0
						denom = pow(denom, (1.0 / 2.2))

						abj_sd_b_instance.setActiveStageMaterial(shadingPlane, mySplitFaceIndexUsable, denom, denom, denom)

					elif items_id_currentStage == 9:
						if printOnce_stage_009 == False:
							print("'stage_009' : D = alphaSqr / (pi * denom * denom)")
							printOnce_stage_009 = True

						dotNH_temp = abj_sd_b_instance.clamp(N.dot(H), 0, 1)
						dotNH2 = dotNH_temp * dotNH_temp

						alpha_temp = bpy.context.scene.ggx_roughness_prop * bpy.context.scene.ggx_roughness_prop
						alphaSqr = alpha_temp * alpha_temp

						denom = dotNH2 * (alphaSqr - 1.0) + 1.0
						pi = 3.14159
						D = alphaSqr / (pi * denom * denom)
						D = pow(D, (1.0 / 2.2))

						abj_sd_b_instance.setActiveStageMaterial(shadingPlane, mySplitFaceIndexUsable, D, D, D)

					##############
					### FRESNEL
					##############
					elif items_id_currentStage == 10:
						if printOnce_stage_010 == False:
							print("'stage_010' : dotLH....show L arrow and H arrow")
							print("'stage_010' : 'H = mathutils.Vector(V + L).normalized()")
							printOnce_stage_010 = True

						abj_sd_b_instance.show_arrow_L_to_faceCenter(faceCenter, mySplitFaceIndexUsable)

						abj_sd_b_instance.show_arrow_H(shadingPlane, faceCenter, mySplitFaceIndexUsable)

						dotLH_temp = abj_sd_b_instance.clamp(L.dot(H), 0, 1)
						dotLH_temp = pow(dotLH_temp, (1.0 / 2.2))

						abj_sd_b_instance.setActiveStageMaterial(shadingPlane, mySplitFaceIndexUsable, dotLH_temp, dotLH_temp, dotLH_temp)


					elif items_id_currentStage == 11:
						if printOnce_stage_011 == False:
							print("'stage_011' : dotLH5 = pow(1.0 - dotLH, 5)")
							printOnce_stage_011 = True

						abj_sd_b_instance.show_arrow_L_to_faceCenter(faceCenter, mySplitFaceIndexUsable)

						abj_sd_b_instance.show_arrow_H(shadingPlane, faceCenter, mySplitFaceIndexUsable)

						dotLH_temp = abj_sd_b_instance.clamp(L.dot(H), 0, 1)
						dotLH5_temp = pow(1.0 - dotLH_temp, 5)
						dotLH5_temp = pow(dotLH5_temp, (1.0 / 2.2))

						abj_sd_b_instance.setActiveStageMaterial(shadingPlane, mySplitFaceIndexUsable, dotLH5_temp, dotLH5_temp, dotLH5_temp)

					elif items_id_currentStage == 12:
						if printOnce_stage_012 == False:
							print("'stage_012' : F = F0 + (1.0 - F0) * (dotLH5)")
							printOnce_stage_012 = True

						dotLH_temp = abj_sd_b_instance.clamp(L.dot(H), 0, 1)
						dotLH5_temp = pow(1.0 - dotLH_temp, 5)

						F0 = bpy.context.scene.ggx_fresnel_prop
						F = F0 + (1.0 - F0) * (dotLH5_temp)
						F = pow(F, (1.0 / 2.2))

						abj_sd_b_instance.setActiveStageMaterial(shadingPlane, mySplitFaceIndexUsable, F, F, F)

					##############
					### V - Shadowing
					##############
					elif items_id_currentStage == 13:
						if printOnce_stage_013 == False:
							print("'stage_013' : k = alpha / 2.0")
							printOnce_stage_013 = True

						alpha_temp = bpy.context.scene.ggx_roughness_prop * bpy.context.scene.ggx_roughness_prop

						k = alpha_temp / 2.0
						k = pow(k, (1.0 / 2.2))

						abj_sd_b_instance.setActiveStageMaterial(shadingPlane, mySplitFaceIndexUsable, k, k, k)

					elif items_id_currentStage == 14:
						if printOnce_stage_014 == False:
							print("'stage_014' : G1V(dotNL, k)")
							print("'stage_014' : 1.0 / (dotNL * (1.0 - k) + k)")
							print("'stage_014' : k = alpha / 2.0")
							printOnce_stage_014 = True

						abj_sd_b_instance.show_arrow_N(shadingPlane, faceCenter, mySplitFaceIndexUsable)

						abj_sd_b_instance.show_arrow_L_to_faceCenter(faceCenter, mySplitFaceIndexUsable)

						alpha_temp = bpy.context.scene.ggx_roughness_prop * bpy.context.scene.ggx_roughness_prop

						k = alpha_temp / 2.0

						dotNL = abj_sd_b_instance.clamp(N.dot(L), 0, 1)

						myG1V_dotNL = self.G1V(dotNL, k)
						myG1V_dotNL = pow(myG1V_dotNL, (1.0 / 2.2))

						abj_sd_b_instance.setActiveStageMaterial(shadingPlane, mySplitFaceIndexUsable, myG1V_dotNL, myG1V_dotNL, myG1V_dotNL)

					elif items_id_currentStage == 15:
						if printOnce_stage_015 == False:
							print("'stage_015' : G1V(dotNV, k)")
							print("'stage_015' : 1.0 / (dotNV * (1.0 - k) + k)")
							print("'stage_015' : k = alpha / 2.0")
							printOnce_stage_015 = True

						abj_sd_b_instance.show_arrow_N(shadingPlane, faceCenter, mySplitFaceIndexUsable)

						abj_sd_b_instance.myCubeCam.hide_set(0) # V

						alpha_temp = bpy.context.scene.ggx_roughness_prop * bpy.context.scene.ggx_roughness_prop

						k = alpha_temp / 2.0

						dotNV = abj_sd_b_instance.clamp(N.dot(V), 0, 1)

						myG1V_dotNV = self.G1V(dotNV, k)
						myG1V_dotNV = pow(myG1V_dotNV, (1.0 / 2.2))

						abj_sd_b_instance.setActiveStageMaterial(shadingPlane, mySplitFaceIndexUsable, myG1V_dotNV, myG1V_dotNV, myG1V_dotNV)

					elif items_id_currentStage == 16:
						if printOnce_stage_016 == False:
							print("'stage_016' : vis = G1V(dotNL, k) * G1V(dotNV, k)")
							printOnce_stage_016 = True

						abj_sd_b_instance.show_arrow_N(shadingPlane, faceCenter, mySplitFaceIndexUsable)

						abj_sd_b_instance.myCubeCam.hide_set(0) # V

						abj_sd_b_instance.show_arrow_L_to_faceCenter(faceCenter, mySplitFaceIndexUsable)

						alpha_temp = bpy.context.scene.ggx_roughness_prop * bpy.context.scene.ggx_roughness_prop

						k = alpha_temp / 2.0

						dotNL = abj_sd_b_instance.clamp(N.dot(L), 0, 1)
						myG1V_dotNL = self.G1V(dotNL, k)

						dotNV = abj_sd_b_instance.clamp(N.dot(V), 0, 1)
						myG1V_dotNV = self.G1V(dotNV, k)

						vis = myG1V_dotNL * myG1V_dotNV
						vis = pow(vis, (1.0 / 2.2))

						abj_sd_b_instance.setActiveStageMaterial(shadingPlane, mySplitFaceIndexUsable, vis, vis, vis)

			if items_id_currentStage == 17:
				if printOnce_stage_017 == False:
					print('stage_017 output AOV = FINAL STAGE')
					print('stage_017 output AOV = dotNL * D * F * vis')
					print('stage_017 output AOV = ', aov_id)
					printOnce_stage_017 = True

				# abj_sd_b_instance.profile_stage2_06_a = datetime.now() ################

				# abj_sd_b_instance.myCubeCam.hide_set(1)

				abj_sd_b_instance.Ci_render_temp_list.append(mySplitFaceIndexUsable)

				# abj_sd_b_instance.profile_stage2_06_b = datetime.now() - abj_sd_b_instance.profile_stage2_06_a
				# abj_sd_b_instance.profile_stage2_06_final += abj_sd_b_instance.profile_stage2_06_b

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

	# This function is based on a public domain algorithm found at [http://filmicworlds.com/blog/optimizing-ggx-shaders-with-dotlh/ or https://gist.github.com/alvalau/c4aa35069ff339da5e59e5ed9c4db192]
	def equation_notes_for_study(self, abj_sd_b_instance):
		printableVersion = True

		if printableVersion == True:
			pass

			'''
			
			# if NdotV < 0.1 (ortho -> persp fix), spec = 0
			############## D......dotNH, roughness^3
			if items_id_currentStage == 0:
					("'N....show N arrow (cubeN)'")
			elif items_id_currentStage == 1:
					("'V + L....show V arrow and L arrow'")
			elif items_id_currentStage == 2:
					("V + L + H....show V arrow and L arrow and H arrow'")
					("'H = mathutils.Vector(V + L).normalized()")
			elif items_id_currentStage == 3:
					("'dotNH....show N and H arrows'")
			elif items_id_currentStage == 4:
					("'dotNH * dotNH'")
			elif items_id_currentStage == 5:
					("'roughness'")
			elif items_id_currentStage == 6:
					("'alpha = roughness * roughness'")
			elif items_id_currentStage == 7:
					("'alphaSqr = alpha * alpha'")
			elif items_id_currentStage == 8:
					("'denom = dotNH * dotNH * (alphaSqr - 1.0) + 1")
			elif items_id_currentStage == 9:
					("'D = alphaSqr / (pi * denom * denom)")
			############## FRESNEL......dotLH, F0
			elif items_id_currentStage == 10:
					("'dotLH....show L arrow and H arrow")
					("'H = mathutils.Vector(V + L).normalized()")
			elif items_id_currentStage == 11:
					("'dotLH5 = pow(1.0 - dotLH, 5)")
			elif items_id_currentStage == 12:
					("'F = F0 + (1.0 - F0) * (dotLH5)")
			############## V - Shadowing......dotNL, dotNV, roughness^2
			elif items_id_currentStage == 13:
					("'k = alpha / 2.0")
			elif items_id_currentStage == 14:
					("'G1V(dotNL, k)")
					("'1.0 / (dotNL * (1.0 - k) + k)")
			elif items_id_currentStage == 15:
					("'G1V(dotNV, k)")
					("'1.0 / (dotNV * (1.0 - k) + k)")
			elif items_id_currentStage == 16:
					("'vis = G1V(dotNL, k) * G1V(dotNV, k)")
			if items_id_currentStage == 17:
					('output AOV = dotNL * D * F * vis')

			'''


		'''
		if NdotV < 0.1 (ortho -> persp fix), spec = 0
		if raycast from V to faceCenter = False : spec 0
		if raycast from L to faceCenter = False : spec 0
		############## D......dotNH, roughness^3
		if items_id_currentStage == 0:
				print("'stage_000' : 'N....show N arrow (cubeN)'")
		elif items_id_currentStage == 1:
				print("'stage_001' : 'V + L....show V arrow and L arrow'")
		elif items_id_currentStage == 2:
				print("'stage_002' : 'V + L + H....show V arrow and L arrow and H arrow'")
				print("'stage_002' : 'H = mathutils.Vector(V + L).normalized()")
		elif items_id_currentStage == 3:
				print("'stage_003' : 'dotNH....show N and H arrows'")
		elif items_id_currentStage == 4:
				print("'stage_004' : 'dotNH * dotNH'")
		elif items_id_currentStage == 5:
				print("'stage_005' : 'roughness'")
		elif items_id_currentStage == 6:
				print("'stage_006' : 'alpha = roughness * roughness'")
		elif items_id_currentStage == 7:
				print("'stage_007' : 'alphaSqr = alpha * alpha'")
		elif items_id_currentStage == 8:
				print("'stage_008' : denom = dotNH * dotNH * (alphaSqr - 1.0) + 1")
		elif items_id_currentStage == 9:
				print("'stage_009' : D = alphaSqr / (pi * denom * denom)")
		############## FRESNEL......dotLH, F0
		elif items_id_currentStage == 10:
				print("'stage_010' : dotLH....show L arrow and H arrow")
				print("'stage_010' : 'H = mathutils.Vector(V + L).normalized()")
		elif items_id_currentStage == 11:
				print("'stage_011' : dotLH5 = pow(1.0 - dotLH, 5)")
		elif items_id_currentStage == 12:
				print("'stage_012' : F = F0 + (1.0 - F0) * (dotLH5)")
		############## V - Shadowing......dotNL, dotNV, roughness^2
		elif items_id_currentStage == 13:
				print("'stage_013' : k = alpha / 2.0")
		elif items_id_currentStage == 14:
				print("'stage_014' : G1V(dotNL, k)")
				print("'stage_014' : 1.0 / (dotNL * (1.0 - k) + k)")
		elif items_id_currentStage == 15:
				print("'stage_015' : G1V(dotNV, k)")
				print("'stage_015' : 1.0 / (dotNV * (1.0 - k) + k)")
		elif items_id_currentStage == 16:
				print("'stage_016' : vis = G1V(dotNL, k) * G1V(dotNV, k)")
		if items_id_currentStage == 17:
				print('stage_017 output AOV = dotNL * D * F * vis')

		'''