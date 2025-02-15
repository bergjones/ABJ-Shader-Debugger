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

	def equation_preProcess_00(self, abj_sd_b_instance, mySplitFaceIndexUsable):
		faceCenter = abj_sd_b_instance.shadingPlane.location
		pos = abj_sd_b_instance.shadingPlane.location

		myL = mathutils.Vector((abj_sd_b_instance.pos_light_global_v - pos)).normalized()

		##################################################################################
		###################################### STORE SHADE PARAMS #####################################
		##################################################################################
		bpy.context.view_layer.objects.active = abj_sd_b_instance.shadingPlane

		normalDir = abj_sd_b_instance.getFaceNormal()

		# abj_sd_b_instance.profile_stage1_02_b = datetime.now() - abj_sd_b_instance.profile_stage1_02_a
		# abj_sd_b_instance.profile_stage1_02_final += abj_sd_b_instance.profile_stage1_02_b

		# abj_sd_b_instance.profile_stage1_03_a = datetime.now() ################

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

		# abj_sd_b_instance.profile_stage1_03_b = datetime.now() - abj_sd_b_instance.profile_stage1_03_a
		# abj_sd_b_instance.profile_stage1_03_final += abj_sd_b_instance.profile_stage1_03_b

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
		}

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

	def equation_part2_00_working(self):
		pass

		'''
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

				myRay_faceCenter_to_V = self.raycast_abj_scene(shadingPlane, self.pos_camera_global_v, V_toFace, mySplitFaceIndexUsable) ########## good

				if myRay_faceCenter_to_V == True:
					faceCenter_to_V_rayCast = True
					# self.print('makes it to the cam, now cast again')

				else:
					faceCenter_to_V_rayCast = False
					# self.print('behind something else, discard')

				if faceCenter_to_V_rayCast == True:
					#######################
					#RAYCAST AGAINST L
					#######################
					myRay_faceCenter_to_L = self.raycast_abj_scene(myInputMesh_dupeForRaycast.name, self.pos_light_global_v, -L, mySplitFaceIndexUsable) ########## good

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


					

		def raycast_abj_scene(self, meshToCheck, origin, direction, debugidx):
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

			usableMesh = None
			for i in bpy.context.scene.objects:
				if i.name == meshToCheck:
					usableMesh = i

			myDepsgraph = bpy.context.view_layer.depsgraph
			dir_usable = direction
			origin_usable = origin

			# self.updateScene()
			# myDepsgraph.update() 

			hit, loc, norm, idx, obj, mw = bpy.context.scene.ray_cast(myDepsgraph, origin_usable, dir_usable)

			######### OBJECT
			toReturn = None

			if hit:
				mySplitFaceIndexUsable_rayHit = obj.name.split('_', -1)[1]

				if debugidx == mySplitFaceIndexUsable_rayHit:
					toReturn = True
					# if debugidx == '242':
					# 	self.print('TRUE for debugIdx, obj : ', debugidx, ' ', obj.name)
				else:
					toReturn = False
					if debugidx == '242':
						self.print('FALSE for debugIdx, obj : ', debugidx, ' ', obj.name)
		
			else:
				toReturn = False
				if debugidx == '242':
					self.print('ray miss for debugIdx, obj : ', debugidx)


			for j in objectsToToggleOnOffLater_stored:
				j.hide_set(0)

			if storedCubeCamState == 0:
				self.myCubeCam.hide_set(0)

			for j in self.allNamesToToggleDuringRaycast:
				j.hide_set(0)

			return toReturn
		'''



	def equation_part2_00(self):
		pass

		'''
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

				myRay_faceCenter_to_V = self.raycast_abj_scene(shadingPlane, self.pos_camera_global_v, V_toFace, mySplitFaceIndexUsable) ########## good

				if myRay_faceCenter_to_V == True:
					faceCenter_to_V_rayCast = True
					# self.print('makes it to the cam, now cast again')

				else:
					faceCenter_to_V_rayCast = False
					# self.print('behind something else, discard')

				if faceCenter_to_V_rayCast == True:
					#######################
					#RAYCAST AGAINST L
					#######################
					myRay_faceCenter_to_L = self.raycast_abj_scene(myInputMesh_dupeForRaycast.name, self.pos_light_global_v, -L, mySplitFaceIndexUsable) ########## good

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


					

		def raycast_abj_scene(self, meshToCheck, origin, direction, debugidx):
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

			usableMesh = None
			for i in bpy.context.scene.objects:
				if i.name == meshToCheck:
					usableMesh = i

			myDepsgraph = bpy.context.view_layer.depsgraph
			dir_usable = direction
			origin_usable = origin

			# self.updateScene()
			# myDepsgraph.update() 

			hit, loc, norm, idx, obj, mw = bpy.context.scene.ray_cast(myDepsgraph, origin_usable, dir_usable)

			######### OBJECT
			toReturn = None

			if hit:
				mySplitFaceIndexUsable_rayHit = obj.name.split('_', -1)[1]

				if debugidx == mySplitFaceIndexUsable_rayHit:
					toReturn = True
					# if debugidx == '242':
					# 	self.print('TRUE for debugIdx, obj : ', debugidx, ' ', obj.name)
				else:
					toReturn = False
					if debugidx == '242':
						self.print('FALSE for debugIdx, obj : ', debugidx, ' ', obj.name)
		
			else:
				toReturn = False
				if debugidx == '242':
					self.print('ray miss for debugIdx, obj : ', debugidx)


			for j in objectsToToggleOnOffLater_stored:
				j.hide_set(0)

			if storedCubeCamState == 0:
				self.myCubeCam.hide_set(0)

			for j in self.allNamesToToggleDuringRaycast:
				j.hide_set(0)

			return toReturn
		'''

	def equation_aov(self):
		pass

		'''
		elif items_id_currentStage == 7:
		if printOnce_stage_007 == False:
			self.print('stage_007 output AOV = ', aov_id)
			printOnce_stage_007 = True

		self.myCubeCam.hide_set(1)

		self.aov_output(aov_id, shadingPlane, mySplitFaceIndexUsable, N_dot_L, spec, attenuation)

		


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
		'''