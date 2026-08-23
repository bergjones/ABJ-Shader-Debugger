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
from datetime import datetime
import random
import numpy as np
import scipy
from scipy.spatial import Delaunay
import importlib
import sys
import copy

class myEquation_spectral_atmosphere:
	def __init__(self):
		super(myEquation_spectral_atmosphere, self).__init__()

	def atmospheric_rayleigh_mie_nishita_spectral_compositor(self, abj_sd_b_instance):
		abj_sd_b_instance.deselectAll()
		abj_sd_b_instance.deleteAllObjects()
		abj_sd_b_instance.mega_purge()

		bpy.context.scene.render.engine = 'CYCLES'
		bpy.context.scene.cycles.device = 'GPU'

		###########
		#DEFAULT CAMERA
		#############
		# abj_sd_b_instance.pos_camera_global = (10, 10, 10) #spectral
		# abj_sd_b_instance.pos_camera_global = (20, -30, 15) #spectral
		# abj_sd_b_instance.pos_camera_global = (5, -20, 10) #spectral ####
		# abj_sd_b_instance.pos_camera_global = (2, -15, 10) #spectral
		# abj_sd_b_instance.pos_camera_global = (2, -13.5, 8) #spectral
		abj_sd_b_instance.pos_camera_global = (13.7, -8, 0.76) #spectral ###atmospheric

		cam1_data = bpy.ops.object.camera_add(
			location=(abj_sd_b_instance.pos_camera_global),  # x, y, z coordinates
			# rotation=(0.0, 0.0, 0.0)   # x, y, z rotation in radians
			# rotation=(90, 0, 248)   # x, y, z rotation in radians
			# rotation=(math.degrees(90), math.degrees(0), math.degrees(248))   # x, y, z rotation in radians
			# rotation=(math.radians(90), math.radians(0), math.radians(248))   # x, y, z rotation in radians
		)

		abj_sd_b_instance.myCam = bpy.data.objects["Camera"]


		# loc_camera = cam.matrix_world.to_translation()

		# # 2. Compute direction vector and convert to quaternion (-Z forward, Y up)
		# direction = target_point - loc_camera
		# rot_quat = direction.to_track_quat('-Z', 'Y')

		# abj_sd_b_instance.myOrigin
		# abj_sd_b_instance.look_at(abj_sd_b_instance.myCam, abj_sd_b_instance.myOrigin)





		abj_sd_b_instance.myCam.data.clip_start = 1
		# abj_sd_b_instance.myCam.data.clip_start = .1
		# abj_sd_b_instance.myCam.data.clip_start = .5
		# abj_sd_b_instance.myCam.data.clip_end = 100
		abj_sd_b_instance.myCam.data.clip_end = 500

		abj_sd_b_instance.myCam.location = abj_sd_b_instance.pos_camera_global
		abj_sd_b_instance.updateScene() # need

		abj_sd_b_instance.look_at(abj_sd_b_instance.myCam, abj_sd_b_instance.myOrigin)

		# f = abj_sd_b_instance.abjNormalize_written(abj_sd_b_instance.myOrigin - abj_sd_b_instance.myCam.location)
		# abj_sd_b_instance.myV = -f

		bpy.context.scene.camera = abj_sd_b_instance.myCam


		bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
		group_name = "ABJ_Rayleigh_Atmosphere_Compositor"
		
		# Check if group already exists to prevent duplication
		ntree = None
		if group_name in bpy.data.node_groups:
			ntree = bpy.data.node_groups[group_name]
			# ntree.nodes.clear() 
		else:
			ntree = bpy.data.node_groups.new(name=group_name, type='CompositorNodeTree')

		bpy.context.scene.compositing_node_group = ntree

		for node in ntree.nodes:
			ntree.nodes.remove(node)

		# color_out = ntree.interface.new_socket(name="FragColor", in_out='OUT', socket_type='NodeSocketColor')
		ntree.interface.new_socket(name="Output", in_out='OUTPUT', socket_type='NodeSocketColor')

		input_node = ntree.nodes.new('NodeGroupInput')
		abj_sd_b_instance.nodeOut = ntree.nodes.new('NodeGroupOutput')

		node0 = ntree.nodes.new("CompositorNodeRLayers")
		abj_sd_b_instance.nodeViewer = ntree.nodes.new("CompositorNodeViewer")

		# Hardcoded Spectrometric Matrix Arrays (38 slices, 380nm - 750nm), 10nm slice
		RAYLEIGH_WEIGHTS = [
			4.7963, 4.3232, 3.9063, 3.5372, 3.2100, 2.9194, 2.6608, 2.4305, 2.2253, 2.0421,
			1.8783, 1.7314, 1.6000, 1.4822, 1.3764, 1.2813, 1.1957, 1.1184, 1.0483, 0.9474,
			0.8837, 0.8252, 0.7716, 0.7224, 0.6671, 0.6351, 0.5960, 0.5600, 0.5266, 0.4958,
			0.4672, 0.4406, 0.4160, 0.4031, 0.3720, 0.3519, 0.3332, 0.3160
		]

		MIE_WEIGHTS = [
			1.6111, 1.5583, 1.5085, 1.4614, 1.4168, 1.3745, 1.3344, 1.2963, 1.2600, 1.2254,
			1.1925, 1.1610, 1.1310, 1.1023, 1.0748, 1.0484, 1.0232, 0.9989, 0.9757, 0.9533,
			0.9318, 0.9111, 0.8912, 0.8720, 0.8536, 0.8358, 0.8186, 0.8021, 0.7861, 0.7707,
			0.7558, 0.7414, 0.7275, 0.7141, 0.7011, 0.6885, 0.6763, 0.664
		]

		# CIE D65 Standard Daylight Illuminant Spectrum across 38 slices (Unpolarized Source Spectrum)
		D65_ILLUMINANT = [
			49.9,  54.6,  82.8,  91.5,  93.4,  104.9, 117.1, 117.8, 114.9, 115.9,
			108.8, 109.4, 104.8, 105.5, 104.4, 102.1, 100.1, 96.3,  95.8,  90.0,
			89.6,  87.7,  83.3,  83.7,  80.0,  80.2,  82.3,  78.3,  69.7,  71.6,
			74.3,  67.9,  67.5,  64.3,  61.6,  60.4,  64.4,  63.5
		]

		# 1. CIE 1931 2-Degree Color Matching Functions (CMF) mapped to 38 slices (380nm - 750nm)
		# Values integrated per 10nm step to convert spectrum back into linear XYZ space.
		CIE_X = [
			0.0014, 0.0042, 0.0143, 0.0435, 0.1344, 0.2839, 0.3483, 0.3362, 0.2908, 0.1954, 
			0.0956, 0.0320, 0.0049, 0.0093, 0.0633, 0.1655, 0.2904, 0.4334, 0.5945, 0.7621, 
			0.9163, 1.0263, 1.0622, 1.0026, 0.8544, 0.6424, 0.4479, 0.2835, 0.1649, 0.0874, 
			0.0468, 0.0227, 0.0114, 0.0058, 0.0029, 0.0014, 0.0007, 0.0003
		]
		
		CIE_Y = [
			0.0000, 0.0001, 0.0004, 0.0012, 0.0040, 0.0116, 0.0230, 0.0380, 0.0600, 0.0910, 
			0.1390, 0.2080, 0.3230, 0.5030, 0.7100, 0.8620, 0.9540, 0.9950, 0.9950, 0.9520, 
			0.8700, 0.7570, 0.6310, 0.5030, 0.3810, 0.2650, 0.1750, 0.1070, 0.0610, 0.0320, 
			0.0170, 0.0082, 0.0041, 0.0021, 0.0010, 0.0005, 0.0002, 0.0001
		]
		
		CIE_Z = [
			0.0065, 0.0201, 0.0679, 0.2074, 0.6456, 1.3856, 1.7471, 1.7721, 1.6230, 1.2820, 
			0.8130, 0.4652, 0.2720, 0.1582, 0.0782, 0.0343, 0.0137, 0.0052, 0.0017, 0.0005, 
			0.0001, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 
			0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
		]

		#build sun arrow
		mySun_arrow = abj_sd_b_instance.createArrowFullProcess('mySunArrow', 'front', False, abj_sd_b_instance.myOrigin, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0)

		bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)

		abj_sd_b_instance.deselectAll()
		mySun_arrow.select_set(1)
		bpy.context.view_layer.objects.active = mySun_arrow
		# mySun_arrow.rotation_euler = mathutils.Vector((math.radians(0), math.radians(0), math.radians(90)))
		mySun_arrow.rotation_euler = mathutils.Vector((math.radians(0), math.radians(0), math.radians(-90)))
		mySun_arrow.location = mathutils.Vector((0, 0, 0))
		bpy.ops.object.transform_apply(location=1, rotation=1, scale=1)

		mySun_arrow.location = (-0.441291, 0.266461, 0)

		for area in bpy.context.screen.areas: 
			if area.type == 'VIEW_3D':
				for space in area.spaces: 
					if space.type == 'VIEW_3D':
						# space.shading.type = 'WIREFRAME'
						# space.shading.type = 'MATERIAL'
						# space.shading.type = 'SOLID'
						space.shading.type = 'RENDERED'

						# bpy.context.space_data.overlay.show_floor = True
						space.overlay.show_floor = True

		bpy.ops.mesh.primitive_cube_add()

		centerCube = bpy.context.active_object
		centerCube.select_set(1)
		centerCube.scale = mathutils.Vector((.1, .1, .1))
		bpy.ops.object.transform_apply(location=1, rotation=1, scale=1)
		centerCube.hide_set(1)
		centerCube.hide_render = True

		abj_sd_b_instance.deselectAll()
		mySun_arrow.select_set(1)
		bpy.context.view_layer.objects.active = mySun_arrow

		bpy.ops.object.constraint_add(type='DAMPED_TRACK')
		mySun_arrow.constraints["Damped Track"].target = centerCube
		# mySun_arrow.constraints["Damped Track"].track_axis = 'TRACK_X'
		# mySun_arrow.constraints["Damped Track"].track_axis = 'TRACK_NEGATIVE_Y'
		mySun_arrow.constraints["Damped Track"].track_axis = 'TRACK_Y'

		bpy.ops.object.constraint_add(type='LIMIT_LOCATION')
		mySun_arrow.constraints["Limit Location"].use_min_z = True
		mySun_arrow.constraints["Limit Location"].use_transform_limit = True

		mySun_arrow.hide_render = True

		bpy.context.space_data.context = 'SCENE'

		atmospheric_scale = ntree.nodes.new("ShaderNodeValue")
		# atmospheric_scale.outputs[0].default_value = 0.0005  # Lower default to balance raw meter units
		atmospheric_scale.outputs[0].default_value = 5  # Lower default to balance raw meter units
		atmospheric_scale.label = 'atmospheric scale'
		atmospheric_scale.use_custom_color = True
		atmospheric_scale.color = (1, 0, 0)

		########
		#### myL auto sync
		#######
		myL_usable = ntree.nodes.new("GeometryNodeObjectInfo")
		myL_usable.inputs[0].default_value = mySun_arrow

		subtractVec_myL = ntree.nodes.new("ShaderNodeVectorMath")
		subtractVec_myL.operation = 'SUBTRACT'
		ntree.links.new(myL_usable.outputs[1], subtractVec_myL.inputs[1])

		myL_norm = ntree.nodes.new("ShaderNodeVectorMath")
		myL_norm.operation = 'NORMALIZE'
		ntree.links.new(subtractVec_myL.outputs[0], myL_norm.inputs[0])

		myL_norm_sep = ntree.nodes.new("ShaderNodeSeparateXYZ")
		ntree.links.new(myL_norm.outputs[1], myL_norm_sep.inputs[0])

		myL = ntree.nodes.new("FunctionNodeInputVector")
		myL.vector[0] = 0.0
		myL.vector[1] = 0.0
		myL.vector[2] = 0.0
		myL.label = 'light dir'
		myL.use_custom_color = True
		myL.color = (1, 0, 0)


		group = bpy.data.node_groups['ABJ_Rayleigh_Atmosphere_Compositor']
		self.atmospheric_rayleigh_setup_sky_texture_vector_sync(group, myL, mySun_arrow, ntree, abj_sd_b_instance)
		abj_sd_b_instance.autoArrangeNodes(ntree)

		####
		# myV
		####
		inputCam = ntree.nodes.new("GeometryNodeObjectInfo")
		inputCam.inputs[0].default_value = abj_sd_b_instance.myCam

		subtractVec = ntree.nodes.new("ShaderNodeVectorMath")
		subtractVec.operation = 'SUBTRACT'
		ntree.links.new(inputCam.outputs[1], subtractVec.inputs[1])

		normalizeVec = ntree.nodes.new("ShaderNodeVectorMath")
		normalizeVec.operation = 'NORMALIZE'
		ntree.links.new(subtractVec.outputs[0], normalizeVec.inputs[0])

		myV = ntree.nodes.new("ShaderNodeVectorMath")
		myV.operation = 'MULTIPLY'
		myV.label = 'view dir'
		myV.inputs[1].default_value[0] = -1
		myV.inputs[1].default_value[1] = -1
		myV.inputs[1].default_value[2] = -1
		ntree.links.new(normalizeVec.outputs[0], myV.inputs[0])

		######
		sky_cap = ntree.nodes.new("ShaderNodeValue")
		sky_cap.outputs[0].default_value = 5000.0  # Prevents infinite horizon values from breaking loops
		sky_cap.label = 'sky threshold'
		sky_cap.use_custom_color = True
		sky_cap.color = (1, 0, 0)

		###########
		#DEPTH
		###########
		# add map range node
		node_mapRange = ntree.nodes.new("ShaderNodeMapRange")
		node_mapRange.location = (0,0)
		node_mapRange.label = 'depth_adjustable'
		node_mapRange.data_type = 'FLOAT'
		node_mapRange.clamp = True
		node_mapRange.inputs[1].default_value = 0
		node_mapRange.inputs[2].default_value = 3000
		# node_mapRange.inputs[3].default_value = -0.25
		# node_mapRange.inputs[4].default_value = 1

		node_mapRange.use_custom_color = True
		node_mapRange.color = (1, 0, 0)
		ntree.links.new(node0.outputs["Depth"], node_mapRange.inputs[0])

		# Clamp infinite background depth pixels to a measurable sky cap boundary
		depth_clamp = ntree.nodes.new('ShaderNodeMath')
		depth_clamp.operation = 'MINIMUM'
		# ntree.links.new(input_node.outputs['Render Pass Depth'], depth_clamp.inputs[0])
		ntree.links.new(node_mapRange.outputs[0], depth_clamp.inputs[0])
		ntree.links.new(sky_cap.outputs[0], depth_clamp.inputs[1])

		
		'''
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
		'''

		cosTheta = ntree.nodes.new("ShaderNodeVectorMath")
		cosTheta.operation = 'DOT_PRODUCT'
		ntree.links.new(myV.outputs[0], cosTheta.inputs[0])
		ntree.links.new(myL_norm.outputs[0], cosTheta.inputs[1])

		cos_sqr = ntree.nodes.new('ShaderNodeMath')
		cos_sqr.operation = 'MULTIPLY'
		cos_sqr.label = 'cos_sqr'
		ntree.links.new(cosTheta.outputs[0], cos_sqr.inputs[0])
		ntree.links.new(cosTheta.outputs[0], cos_sqr.inputs[1])

		phase_add = ntree.nodes.new('ShaderNodeMath')
		phase_add.label = 'phase_add'
		phase_add.operation = 'ADD'
		phase_add.inputs[0].default_value = 1.0
		ntree.links.new(cos_sqr.outputs[0], phase_add.inputs[1])

		rayleigh_phase = ntree.nodes.new('ShaderNodeMath')
		rayleigh_phase.label = 'rayleigh_phase'
		rayleigh_phase.operation = 'MULTIPLY'
		rayleigh_phase.inputs[0].default_value = 3.0 / (16.0 * math.pi)
		ntree.links.new(phase_add.outputs[0], rayleigh_phase.inputs[1])

		############
		#MIE
		############
		mie_scale = ntree.nodes.new("ShaderNodeValue")
		mie_scale.outputs[0].default_value = 5  # Lower default to balance raw meter units
		mie_scale.label = 'mie scale'
		mie_scale.use_custom_color = True
		mie_scale.color = (1, 0, 0)

		mie_anisotropy = ntree.nodes.new("ShaderNodeValue")
		mie_anisotropy.outputs[0].default_value = 0 
		mie_anisotropy.label = 'mie anisotropy'
		mie_anisotropy.use_custom_color = True
		mie_anisotropy.color = (1, 0, 0)

		g2 = ntree.nodes.new('ShaderNodeMath')
		g2.operation = 'MULTIPLY'
		g2.label = 'g2'
		ntree.links.new(mie_anisotropy.outputs[0], g2.inputs[0])
		ntree.links.new(mie_anisotropy.outputs[0], g2.inputs[1])

		one_minus_g2 = ntree.nodes.new('ShaderNodeMath')
		one_minus_g2.operation = 'SUBTRACT'
		one_minus_g2.outputs[0].default_value = 1 
		ntree.links.new(g2.outputs[0], one_minus_g2.inputs[1])

		mie_phase_0 = ntree.nodes.new('ShaderNodeMath')
		mie_phase_0.label = 'mie phase 0'
		mie_phase_0.operation = 'MULTIPLY'
		mie_phase_0.inputs[0].default_value = 1.0 / (4.0 * math.pi)
		ntree.links.new(one_minus_g2.outputs[0], mie_phase_0.inputs[1])

		#pow
		two_g = ntree.nodes.new('ShaderNodeMath')
		two_g.label = 'mie phase 0'
		two_g.operation = 'MULTIPLY'
		two_g.inputs[0].default_value = 2
		ntree.links.new(mie_anisotropy.outputs[0], two_g.inputs[1])

		two_g_mult_cos_theta = ntree.nodes.new('ShaderNodeMath')
		two_g_mult_cos_theta.label = 'two G mult cos theta'
		two_g_mult_cos_theta.operation = 'MULTIPLY'
		ntree.links.new(two_g.outputs[0], two_g_mult_cos_theta.inputs[0])
		ntree.links.new(cosTheta.outputs[0], two_g_mult_cos_theta.inputs[1])

		one_plus_g2 = ntree.nodes.new('ShaderNodeMath')
		one_plus_g2.operation = 'ADD'
		one_plus_g2.outputs[0].default_value = 1 
		ntree.links.new(g2.outputs[0], one_plus_g2.inputs[1])

		mie_phase_pow_subtract = ntree.nodes.new('ShaderNodeMath')
		mie_phase_pow_subtract.operation = 'SUBTRACT'
		ntree.links.new(one_plus_g2.outputs[0], mie_phase_pow_subtract.inputs[0])
		ntree.links.new(two_g_mult_cos_theta.outputs[0], mie_phase_pow_subtract.inputs[1])

		mie_pow = ntree.nodes.new("ShaderNodeMath")
		mie_pow.operation = 'POWER'
		mie_pow.label = 'node_pow'
		mie_pow.inputs[1].default_value = 1.5
		ntree.links.new(mie_phase_pow_subtract.outputs[0], mie_pow.inputs[0])

		mie_phase = ntree.nodes.new('ShaderNodeMath')
		mie_phase.label = 'mie phase 1'
		mie_phase.operation = 'DIVIDE'
		ntree.links.new(mie_phase_0.outputs[0], mie_phase.inputs[0])
		ntree.links.new(mie_pow.outputs[0], mie_phase.inputs[1])

		#################

		xyz_inscattering_accumulate = self.accumulate_spectral_atmosphere_38_p0(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, D65_ILLUMINANT, CIE_X, CIE_Y, CIE_Z, atmospheric_scale, mie_scale, depth_clamp, rayleigh_phase, mie_phase)
		xyz_transmission_accumulate = self.accumulate_spectral_atmosphere_38_p0_transmission(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, CIE_X, CIE_Y, CIE_Z, atmospheric_scale, mie_scale, depth_clamp)

		xyz_accumulate_mult_01 = ntree.nodes.new("ShaderNodeVectorMath")
		xyz_accumulate_mult_01.operation = 'MULTIPLY'
		xyz_accumulate_mult_01.inputs[1].default_value[0] = .01
		xyz_accumulate_mult_01.inputs[1].default_value[1] = .01
		xyz_accumulate_mult_01.inputs[1].default_value[2] = .01
		ntree.links.new(xyz_inscattering_accumulate.outputs[0], xyz_accumulate_mult_01.inputs[0])

		xyz_accumulate_mult_02 = ntree.nodes.new("ShaderNodeVectorMath")
		xyz_accumulate_mult_02.operation = 'MULTIPLY'
		xyz_accumulate_mult_02.inputs[1].default_value[0] = .01
		xyz_accumulate_mult_02.inputs[1].default_value[1] = .01
		xyz_accumulate_mult_02.inputs[1].default_value[2] = .01
		ntree.links.new(xyz_transmission_accumulate.outputs[0], xyz_accumulate_mult_02.inputs[0])

		spectral_atmospheric_xyz_to_srgb_end = self.spectral_compositor_xyz_to_srgb_atmospheric(ntree, xyz_accumulate_mult_01)
		spectral_atmospheric_xyz_to_srgb_end_transmission = self.spectral_compositor_xyz_to_srgb_atmospheric(ntree, xyz_accumulate_mult_02)

		abj_sd_b_instance.autoArrangeNodes(ntree)

		splitRGB = ntree.nodes.new("ShaderNodeSeparateXYZ")
		splitRGB.label = 'split transmission'
		ntree.links.new(spectral_atmospheric_xyz_to_srgb_end_transmission.outputs[0], splitRGB.inputs[0])

		split0 = self.spectral_compositor_clamp_01_idx(ntree, splitRGB, 0)
		split1 = self.spectral_compositor_clamp_01_idx(ntree, splitRGB, 1)
		split2 = self.spectral_compositor_clamp_01_idx(ntree, splitRGB, 2)

		transmissionClampedRGB = ntree.nodes.new("ShaderNodeCombineXYZ")
		ntree.links.new(split0.outputs[0], transmissionClampedRGB.inputs[0]) 
		ntree.links.new(split1.outputs[0], transmissionClampedRGB.inputs[1]) 
		ntree.links.new(split2.outputs[0], transmissionClampedRGB.inputs[2]) 

		inscattering_max = ntree.nodes.new("ShaderNodeVectorMath")
		inscattering_max.operation = 'MAXIMUM'
		inscattering_max.inputs[0].default_value[0] = 0
		inscattering_max.inputs[0].default_value[1] = 0
		inscattering_max.inputs[0].default_value[2] = 0
		ntree.links.new(spectral_atmospheric_xyz_to_srgb_end.outputs[0], inscattering_max.inputs[1])

		sceneMultTransmission = ntree.nodes.new("ShaderNodeVectorMath")
		sceneMultTransmission.operation = 'MULTIPLY'
		ntree.links.new(node0.outputs[0], sceneMultTransmission.inputs[0])
		ntree.links.new(transmissionClampedRGB.outputs[0], sceneMultTransmission.inputs[1])

		final_spectral_atmosphere_add = ntree.nodes.new("ShaderNodeVectorMath")
		final_spectral_atmosphere_add.operation = 'ADD'
		ntree.links.new(sceneMultTransmission.outputs[0], final_spectral_atmosphere_add.inputs[0])
		ntree.links.new(inscattering_max.outputs[0], final_spectral_atmosphere_add.inputs[1])

		self.spectral_compositor_debugging_exit_visualizer_atmospheric(ntree, final_spectral_atmosphere_add, 932, 633, abj_sd_b_instance)


	def spectral_compositor_debugging_exit_visualizer_atmospheric(self, nodetree, nodeToView, readPixelX, readPixelY, abj_sd_b_instance):

		# nodetree.links.new(nodeToView.outputs[0], abj_sd_b_instance.nodeOut.inputs[0]) ###### !!!!!!!!!!!
		# nodetree.links.new(nodeToView.outputs[0], abj_sd_b_instance.nodeViewer.inputs[0]) ###### !!!!!!!!!!!

		abj_sd_b_instance.autoArrangeNodes(nodetree)

		"""
		Appends a dedicated File Output node to the tail of your spectral compositor.
		This bypasses animation frame skipping by forcing Blender to serialize the 
		composited data to disk before advancing the animation timeline frame index.
		"""
		# Create an explicit file output hard-sink node
		file_out = nodetree.nodes.new('CompositorNodeOutputFile')
		file_out.label = "ABJ_Spectral_exr_out"
		file_out.use_custom_color = True
		file_out.color = (0, 1, 0)
		
		# Configure the path target destination
		file_out.directory = "//compositing_files//render_output/"
		file_out.file_name = "spectral_frame_######"

		file_out.format.media_type = 'IMAGE'

		# Force high-fidelity data preservation format tracking configurations
		file_out.format.file_format = 'PNG'
		# file_out.format.file_format = 'OPEN_EXR'
		# file_out.format.color_depth = '16' # 16-bit Float prevents any distance color banding
		# file_out.format.media_type = 'MULTI_LAYER_IMAGE'

		nodeToView_separated = nodetree.nodes.new("ShaderNodeSeparateXYZ")
		nodeToView_separated.label = 'separated final color'
		nodetree.links.new(nodeToView.outputs[0], nodeToView_separated.inputs[0])

		combine_color = nodetree.nodes.new("CompositorNodeCombineColor")
		combine_color.label = 'combine_final'
		nodetree.links.new(nodeToView_separated.outputs[0], combine_color.inputs[0])
		nodetree.links.new(nodeToView_separated.outputs[1], combine_color.inputs[1])
		nodetree.links.new(nodeToView_separated.outputs[2], combine_color.inputs[2])

		# slot_color = file_out.file_output_items.new("RGBA", "Final")
		slot_color = file_out.file_output_items.new("RGBA", "")

		nodetree.links.new(combine_color.outputs[0], file_out.inputs[0])
		
		print("[ABJ Debugger] Frame-skipping protection enabled via explicit File Output Sink node.")

		nodetree.links.new(combine_color.outputs[0], abj_sd_b_instance.nodeOut.inputs[0]) ###### !!!!!!!!!!!
		nodetree.links.new(combine_color.outputs[0], abj_sd_b_instance.nodeViewer.inputs[0]) ###### !!!!!!!!!!!

		abj_sd_b_instance.autoArrangeNodes(nodetree)
		# abj_sd_b_instance.autoArrangeNodes(worldtree)

		# abj_sd_b_instance.compositor_setup = True

		############################
		#the user must have saved a new, default scene to a file and have a folder called 'compositing_files' in that directory
		############################

		abj_sd_b_instance.updateScene() # need
		# abj_sd_b_instance.look_at(abj_sd_b_instance.myCam, myInputMesh3.location)

		bpy.context.scene.render.engine = 'CYCLES'
		bpy.context.scene.cycles.device = 'GPU'

		# bpy.context.scene.render.resolution_x = 3840
		# bpy.context.scene.render.resolution_y = 2160

		# bpy.context.scene.cycles.samples = 1024
		# bpy.context.scene.cycles.samples = 256
		bpy.context.scene.cycles.samples = 64

		bpy.context.scene.cycles.denoising_use_gpu = True

		# bpy.context.scene.view_settings.view_transform = 'AgX'
		# bpy.context.scene.view_settings.look = 'AgX - Punchy'

		# ##########
		# ## DEBUG_SPECTRAL_COMPOSITOR
		# #########
		# if "Cube" in bpy.data.objects:
		# 	cube_obj = bpy.data.objects["Cube"]
		# 	# Unlink and remove the object completely
		# 	bpy.data.objects.remove(cube_obj, do_unlink=True)

		# return

		######### generate test objects for demo

		# bpy.ops.mesh.primitive_cube_add(size=.5, location=(0, 0, 0))
		bpy.ops.mesh.primitive_uv_sphere_add(radius=.25, location=(0, 0, 0))
		base_obj = bpy.context.active_object
		bpy.ops.object.shade_smooth()


		# Shared mesh data for true instancing
		mesh_data = base_obj.data

		# Define X and Y positions for the instances
		import math
		positions = [(x * 3.0, y * 3.0) for x in range(-5, 6) for y in range(-5, 6)]

		# Deselect all
		bpy.ops.object.select_all(action='DESELECT')

		fixed_height = 0.0  # Keep all at the same height

		mat1 = abj_sd_b_instance.newShader("principled_test_grd", "principled", 0, 0, 1)

		for i, (px, py) in enumerate(positions):
			# Create a new object sharing the same mesh data (instance)
			new_obj = bpy.data.objects.new(f"Cube_Instance_{i}", mesh_data)
			
			# Link to the current scene collection
			bpy.context.collection.objects.link(new_obj)
			
			# Set position (keeping Z constant)
			new_obj.location = (px, py, fixed_height)


			bpy.context.active_object.data.materials.clear()
			bpy.context.active_object.data.materials.append(mat1)
			bpy.data.materials["principled_test_grd"].node_tree.nodes["Principled BSDF"].inputs[1].default_value = 1
			bpy.data.materials["principled_test_grd"].node_tree.nodes["Principled BSDF"].inputs[2].default_value = 0.323263

		########
		##### 00
		########
		bpy.ops.mesh.primitive_monkey_add()
		myInputMesh = bpy.context.active_object
		myInputMesh.select_set(1)
		# myInputMesh.location = mathutils.Vector((0, 0, 5))
		myInputMesh.location = mathutils.Vector((0, -3, -.2))
		myInputMesh.rotation_euler = mathutils.Vector((math.radians(0), math.radians(0), math.radians(90)))
		bpy.ops.object.shade_smooth()

		mat1 = abj_sd_b_instance.newShader("principled_test_00", "principled", 1, 0, 0)
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

		abj_sd_b_instance.autoArrangeNodes(mat.node_tree)

		########
		##### 01
		########
		bpy.ops.mesh.primitive_monkey_add()
		myInputMesh2 = bpy.context.active_object
		myInputMesh2.select_set(1)
		# myInputMesh2.location = mathutils.Vector((1, 8, 5))
		myInputMesh2.location = mathutils.Vector((-14, 8, -.8))
		myInputMesh2.rotation_euler = mathutils.Vector((math.radians(0), math.radians(0), math.radians(90)))
		bpy.ops.object.shade_smooth()

		mat1 = abj_sd_b_instance.newShader("principled_test_01", "principled", 1, 0, 0)
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

		abj_sd_b_instance.autoArrangeNodes(mat.node_tree)

		# mat1 = abj_sd_b_instance.newShader("greenM", "emission", 0, 1, 0)
		# mat1 = abj_sd_b_instance.newShader("greenM", "diffuse", 0, 1, 0)
		# bpy.context.active_object.data.materials.clear()
		# bpy.context.active_object.data.materials.append(mat1)

		########
		##### 02
		########
		bpy.ops.mesh.primitive_monkey_add()
		myInputMesh = bpy.context.active_object
		myInputMesh.select_set(1)
		myInputMesh.location = mathutils.Vector((-30, 24, -.3))
		myInputMesh.rotation_euler = mathutils.Vector((math.radians(0), math.radians(0), math.radians(90)))
		bpy.ops.object.shade_smooth()

		mat1 = abj_sd_b_instance.newShader("principled_test_02", "principled", 1, 0, 0)
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

		abj_sd_b_instance.autoArrangeNodes(mat.node_tree)

		########
		##### 03
		########
		bpy.ops.mesh.primitive_monkey_add()
		myInputMesh = bpy.context.active_object
		myInputMesh.select_set(1)
		myInputMesh.location = mathutils.Vector((-44, 47, -.23))
		myInputMesh.rotation_euler = mathutils.Vector((math.radians(0), math.radians(0), math.radians(90)))
		bpy.ops.object.shade_smooth()

		mat1 = abj_sd_b_instance.newShader("principled_test_03", "principled", 1, 0, 0)
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

		abj_sd_b_instance.autoArrangeNodes(mat.node_tree)

		####
		# CUBE GROUND
		####
		bpy.ops.mesh.primitive_cube_add()
		myInputMesh3 = bpy.context.active_object
		myInputMesh3.select_set(1)
		myInputMesh3.location = mathutils.Vector((0, 0, -5))
		myInputMesh3.scale = mathutils.Vector((20, 20, 2))

		# mat1 = abj_sd_b_instance.newShader("blueM", "emission", 0, 0, 1)
		# mat1 = abj_sd_b_instance.newShader("blueM", "diffuse", 0, 0, 1)
		# bpy.context.active_object.data.materials.clear()
		# bpy.context.active_object.data.materials.append(mat1)

		mat1 = abj_sd_b_instance.newShader("principled_test_grd", "principled", 0, 0, 1)
		bpy.context.active_object.data.materials.clear()
		bpy.context.active_object.data.materials.append(mat1)
		bpy.data.materials["principled_test_grd"].node_tree.nodes["Principled BSDF"].inputs[1].default_value = 1
		bpy.data.materials["principled_test_grd"].node_tree.nodes["Principled BSDF"].inputs[2].default_value = 0.323263

		abj_sd_b_instance.autoArrangeNodes(mat.node_tree)

		return

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

		abj_sd_b_instance.look_at(abj_sd_b_instance.myCam, myInputMesh2.location)

		nodetree.links.new(nodeToView.outputs[0], abj_sd_b_instance.nodeOut.inputs[0]) ###### !!!!!!!!!!!
		nodetree.links.new(nodeToView.outputs[0], abj_sd_b_instance.nodeViewer.inputs[0]) ###### !!!!!!!!!!!

		abj_sd_b_instance.autoArrangeNodes(nodetree)
		abj_sd_b_instance.autoArrangeNodes(worldtree)

		abj_sd_b_instance.compositor_setup = True

		########################################################
		#write node to disc and read pixel
		########################################################
		bpy.context.scene.view_settings.view_transform = 'AgX'
		bpy.context.scene.view_settings.look = 'AgX - Punchy'

		bpy.context.scene.render.use_multiview = False

		# temp_filepath = bpy.path.abspath("//compositor_pixel_temp.png")
		temp_filepath = "//compositing_files/readSingleOutputPixel.png"
		original_filepath = bpy.context.scene.render.filepath

		bpy.context.scene.render.filepath = temp_filepath

		# 2. Render to write the compositor result out to disk
		bpy.ops.render.render(write_still=True)

		return

		# Restore original render filepath
		bpy.context.scene.render.filepath = original_filepath

		# 3. Load the saved image back via bpy.data.images to inspect pixel data safely
		img = bpy.data.images.load(temp_filepath, check_existing=False)

		# bpy.data.images[temp_filepath].reload()
		# "E:/projects_3d/ABJ_Shader_Debugger_for_Blender/scenes/compositing_files/readSingleOutputPixel.png"
		# bpy.data.images["E:/projects_3d/ABJ_Shader_Debugger_for_Blender/scenes/compositing_files/readSingleOutputPixel.png"].reload()

		# bpy.ops.image.reload()

		abj_sd_b_instance.updateScene()

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


	def atmospheric_rayleigh_setup_sky_texture_vector_sync(self, compositor_node_group, myL, mySun_arrow, ntree, abj_sd_b_instance):
		"""
		Connects a FunctionNodeInputVector node from your compositor group 
		to the World Shader's Sky Texture node using automated scripted drivers.
		"""
		# 1. Target the Sky Texture node inside the active World Shader tree
		if not bpy.context.scene.world or not bpy.context.scene.world.node_tree:
			print("[ABJ Debugger Error] No active World Node Tree found to sync.")
			return
			
		world_tree = bpy.context.scene.world.node_tree

		world_tree.nodes.clear()

		output_node_world = world_tree.nodes.new(type="ShaderNodeOutputWorld")

		sky_node = next((n for n in world_tree.nodes if n.type == 'TEX_SKY'), None)
		
		if not sky_node:
			print("[ABJ Debugger Warning] No Sky Texture node found in World Shader. Creating one...")
			sky_node = world_tree.nodes.new('ShaderNodeTexSky')

		sky_node.sun_elevation = 0
		bpy.context.scene.view_settings.exposure = -3
		
		world_tree.links.new(sky_node.outputs["Color"], output_node_world.inputs["Surface"])

		abj_sd_b_instance.autoArrangeNodes(world_tree)

		# -------------------------------------------------------------------------
		# DRIVER A: SUN ELEVATION on sky texture
		# -------------------------------------------------------------------------
		# Clear existing driver if present to prevent duplication
		sky_node.driver_remove("sun_elevation")
		driver = sky_node.driver_add("sun_elevation").driver
		driver.type = 'SCRIPTED'
		
		# Hook the 'z' value of your input vector to a variable named 'z'
		var_z = driver.variables.new()
		var_z.name = 'var'
		var_z.type = 'SINGLE_PROP'

		var_z.targets[0].id_type = 'OBJECT' 
		var_z.targets[0].id = mySun_arrow
		var_z.targets[0].data_path = "location[2]"

		driver.expression = "clamp(var * 10, 0, radians(180))"

		#########################################################################
		# -------------------------------------------------------------------------
		# DRIVER B: myL
		# -------------------------------------------------------------------------
		node = ntree.nodes.get(myL.name)

		mySync = node.driver_add("vector", 0)
		driver_myL = mySync.driver
		driver_myL.type = 'SCRIPTED'

		var_z_myL = driver_myL.variables.new()
		var_z_myL.name = 'var'
		var_z_myL.type = 'SINGLE_PROP'

		var_z_myL.targets[0].id_type = 'OBJECT' 
		var_z_myL.targets[0].id = mySun_arrow
		var_z_myL.targets[0].data_path = "location[2]"

		driver_myL.expression = "clamp(degrees(var * 10), 0, 180)"

		#########################################################################
		# -------------------------------------------------------------------------
		# DRIVER C: SUN ROTATION
		# -------------------------------------------------------------------------
		sky_node.driver_remove("sun_rotation")
		driver_rot = sky_node.driver_add("sun_rotation").driver
		driver_rot.type = 'SCRIPTED'

		var_x = driver_rot.variables.new()
		var_x.name = 'varX'
		var_x.type = 'SINGLE_PROP'
		var_x.targets[0].id_type = 'OBJECT' 
		var_x.targets[0].id = mySun_arrow
		var_x.targets[0].data_path = "location[0]"

		var_y = driver_rot.variables.new()
		var_y.name = 'varY'
		var_y.type = 'SINGLE_PROP'
		var_y.targets[0].id_type = 'OBJECT' 
		var_y.targets[0].id = mySun_arrow
		var_y.targets[0].data_path = "location[1]"

		driver_rot.expression = "atan2(radians(varX), radians(varY))" 

	#This function is based on spectral3_glsl.py, under MIT license by Ronald van Wijnen (see file)
	def spectral_compositor_xyz_to_srgb_atmospheric(self, nodetree, xyz_combo):

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

		return dotProdCombo_custom

	def spectral_compositor_clamp_01_idx(self, nodetree, inputNode, idx):
		node_min = nodetree.nodes.new("ShaderNodeMath")
		node_min.operation = 'MINIMUM'
		node_min.inputs[1].default_value = 1
		nodetree.links.new(inputNode.outputs[idx], node_min.inputs[0])
		# nodetree.links.new(inputNode.outputs['Result'], node_min.inputs[0])

		node_max = nodetree.nodes.new("ShaderNodeMath")
		node_max.operation = 'MAXIMUM'
		node_max.inputs[0].default_value = 0
		nodetree.links.new(node_min.outputs[0], node_max.inputs[1])

		return node_max


	def accumulate_spectral_atmosphere_slice_transmission(self, ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, atmospheric_scale, mie_scale, depth_clamp, idx):
		#Base Extinction (Outscattering) calculation for this slice
		beta_r = ntree.nodes.new('ShaderNodeMath')
		beta_r.operation = 'MULTIPLY'
		beta_r.inputs[0].default_value = RAYLEIGH_WEIGHTS[idx]
		ntree.links.new(atmospheric_scale.outputs[0], beta_r.inputs[1])

		beta_m = ntree.nodes.new('ShaderNodeMath')
		beta_m.operation = 'MULTIPLY'
		beta_m.inputs[0].default_value = MIE_WEIGHTS[idx]
		ntree.links.new(mie_scale.outputs[0], beta_m.inputs[1])

		beta_total = ntree.nodes.new('ShaderNodeMath')
		beta_total.operation = 'MULTIPLY'
		ntree.links.new(beta_r.outputs[0], beta_total.inputs[0])
		ntree.links.new(beta_m.outputs[0], beta_total.inputs[1])

		neg_beta_total = ntree.nodes.new('ShaderNodeMath')
		neg_beta_total.operation = 'MULTIPLY'
		neg_beta_total.inputs[0].default_value = -1
		ntree.links.new(beta_total.outputs[0], neg_beta_total.inputs[1])

		#Outscattering Transmission calculation (Beer-Lambert Law)
		exp_mult = ntree.nodes.new('ShaderNodeMath')
		exp_mult.operation = 'MULTIPLY'
		ntree.links.new(neg_beta_total.outputs[0], exp_mult.inputs[0])
		ntree.links.new(depth_clamp.outputs[0], exp_mult.inputs[1])

		transmission = ntree.nodes.new('ShaderNodeMath')
		transmission.operation = 'EXPONENT'
		ntree.links.new(exp_mult.outputs[0], transmission.inputs[0])

		############
		return transmission

		
	def accumulate_spectral_atmosphere_slice_inscattered(self, ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, D65_ILLUMINANT, atmospheric_scale, mie_scale, depth_clamp, rayleigh_phase, mie_phase, idx):
		#Base Extinction (Outscattering) calculation for this slice
		beta_r = ntree.nodes.new('ShaderNodeMath')
		beta_r.operation = 'MULTIPLY'
		beta_r.inputs[0].default_value = RAYLEIGH_WEIGHTS[idx]
		ntree.links.new(atmospheric_scale.outputs[0], beta_r.inputs[1])

		beta_m = ntree.nodes.new('ShaderNodeMath')
		beta_m.operation = 'MULTIPLY'
		beta_m.inputs[0].default_value = MIE_WEIGHTS[idx]
		ntree.links.new(mie_scale.outputs[0], beta_m.inputs[1])

		beta_total = ntree.nodes.new('ShaderNodeMath')
		beta_total.operation = 'MULTIPLY'
		ntree.links.new(beta_r.outputs[0], beta_total.inputs[0])
		ntree.links.new(beta_m.outputs[0], beta_total.inputs[1])

		neg_beta_total = ntree.nodes.new('ShaderNodeMath')
		neg_beta_total.operation = 'MULTIPLY'
		neg_beta_total.inputs[0].default_value = -1
		ntree.links.new(beta_total.outputs[0], neg_beta_total.inputs[1])

		#Outscattering Transmission calculation (Beer-Lambert Law)
		exp_mult = ntree.nodes.new('ShaderNodeMath')
		exp_mult.operation = 'MULTIPLY'
		ntree.links.new(neg_beta_total.outputs[0], exp_mult.inputs[0])
		ntree.links.new(depth_clamp.outputs[0], exp_mult.inputs[1])

		transmission = ntree.nodes.new('ShaderNodeMath')
		transmission.operation = 'EXPONENT'
		ntree.links.new(exp_mult.outputs[0], transmission.inputs[0])

		############

		scatteredEnergy_0 = ntree.nodes.new('ShaderNodeMath')
		scatteredEnergy_0.operation = 'MULTIPLY'
		ntree.links.new(beta_r.outputs[0], scatteredEnergy_0.inputs[0])
		ntree.links.new(rayleigh_phase.outputs[0], scatteredEnergy_0.inputs[1])

		scatteredEnergy_1 = ntree.nodes.new('ShaderNodeMath')
		scatteredEnergy_1.operation = 'MULTIPLY'
		ntree.links.new(beta_m.outputs[0], scatteredEnergy_1.inputs[0])
		ntree.links.new(mie_phase.outputs[0], scatteredEnergy_1.inputs[1])

		scatteredEnergy_add = ntree.nodes.new('ShaderNodeMath')
		scatteredEnergy_add.operation = 'ADD'
		ntree.links.new(scatteredEnergy_0.outputs[0], scatteredEnergy_add.inputs[0])
		ntree.links.new(scatteredEnergy_1.outputs[0], scatteredEnergy_add.inputs[1])

		#Accumulate single-scattered light inside this slice array
		one_minus_transmission = ntree.nodes.new('ShaderNodeMath')
		one_minus_transmission.operation = 'SUBTRACT'
		one_minus_transmission.inputs[0].default_value = 1
		ntree.links.new(transmission.outputs[0], one_minus_transmission.inputs[1])

		beta_max = ntree.nodes.new("ShaderNodeMath")
		beta_max.operation = 'MAXIMUM'
		beta_max.inputs[0].default_value = 1e-6
		ntree.links.new(beta_total.outputs[0], beta_max.inputs[1])

		divide_oneMinusTransmission_by_betaMax = ntree.nodes.new("ShaderNodeMath")
		divide_oneMinusTransmission_by_betaMax.operation = 'DIVIDE'
		ntree.links.new(one_minus_transmission.outputs[0], divide_oneMinusTransmission_by_betaMax.inputs[0])
		ntree.links.new(beta_total.outputs[0], divide_oneMinusTransmission_by_betaMax.inputs[1])

		slice_inscattered_0 = ntree.nodes.new('ShaderNodeMath')
		slice_inscattered_0.operation = 'MULTIPLY'
		slice_inscattered_0.inputs[0].default_value = D65_ILLUMINANT[idx]
		ntree.links.new(scatteredEnergy_add.outputs[0], slice_inscattered_0.inputs[1])

		slice_inscattered_1 = ntree.nodes.new('ShaderNodeMath')
		slice_inscattered_1.operation = 'MULTIPLY'
		ntree.links.new(slice_inscattered_0.outputs[0], slice_inscattered_1.inputs[0])
		ntree.links.new(divide_oneMinusTransmission_by_betaMax.outputs[0], slice_inscattered_1.inputs[1])

		return slice_inscattered_1
	
	def accumulate_spectral_atmosphere_38_p0_transmission(self, ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, CIE_X, CIE_Y, CIE_Z, atmospheric_scale, mie_scale, depth_clamp):
		xyz_start = ntree.nodes.new("FunctionNodeInputVector")
		xyz_start.vector[0] = 0.0
		xyz_start.vector[1] = 0.0
		xyz_start.vector[2] = 0.0

		transmission_00 = self.accumulate_spectral_atmosphere_slice_transmission(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, atmospheric_scale, mie_scale, depth_clamp, 0)
		accumulate_00 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, transmission_00, 0, xyz_start)

		transmission_01 = self.accumulate_spectral_atmosphere_slice_transmission(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, atmospheric_scale, mie_scale, depth_clamp, 1)
		accumulate_01 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, transmission_01, 1, accumulate_00)

		transmission_02 = self.accumulate_spectral_atmosphere_slice_transmission(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, atmospheric_scale, mie_scale, depth_clamp, 2)
		accumulate_02 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, transmission_02, 2, accumulate_01)

		transmission_03 = self.accumulate_spectral_atmosphere_slice_transmission(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, atmospheric_scale, mie_scale, depth_clamp, 3)
		accumulate_03 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, transmission_03, 3, accumulate_02)

		transmission_04 = self.accumulate_spectral_atmosphere_slice_transmission(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, atmospheric_scale, mie_scale, depth_clamp, 4)
		accumulate_04 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, transmission_04, 4, accumulate_03)

		transmission_05 = self.accumulate_spectral_atmosphere_slice_transmission(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, atmospheric_scale, mie_scale, depth_clamp, 5)
		accumulate_05 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, transmission_05, 5, accumulate_04)

		transmission_06 = self.accumulate_spectral_atmosphere_slice_transmission(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, atmospheric_scale, mie_scale, depth_clamp, 6)
		accumulate_06 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, transmission_06, 6, accumulate_05)

		transmission_07 = self.accumulate_spectral_atmosphere_slice_transmission(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, atmospheric_scale, mie_scale, depth_clamp, 7)
		accumulate_07 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, transmission_07, 7, accumulate_06)

		transmission_08 = self.accumulate_spectral_atmosphere_slice_transmission(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, atmospheric_scale, mie_scale, depth_clamp, 8)
		accumulate_08 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, transmission_08, 8, accumulate_07)

		transmission_09 = self.accumulate_spectral_atmosphere_slice_transmission(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, atmospheric_scale, mie_scale, depth_clamp, 9)
		accumulate_09 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, transmission_09, 9, accumulate_08)

		transmission_10 = self.accumulate_spectral_atmosphere_slice_transmission(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, atmospheric_scale, mie_scale, depth_clamp, 10)
		accumulate_10 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, transmission_10, 10, accumulate_09)

		transmission_11 = self.accumulate_spectral_atmosphere_slice_transmission(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, atmospheric_scale, mie_scale, depth_clamp, 11)
		accumulate_11 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, transmission_11, 11, accumulate_10)

		transmission_12 = self.accumulate_spectral_atmosphere_slice_transmission(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, atmospheric_scale, mie_scale, depth_clamp, 12)
		accumulate_12 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, transmission_12, 12, accumulate_11)

		transmission_13 = self.accumulate_spectral_atmosphere_slice_transmission(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, atmospheric_scale, mie_scale, depth_clamp, 13)
		accumulate_13 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, transmission_13, 13, accumulate_12)

		transmission_14 = self.accumulate_spectral_atmosphere_slice_transmission(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, atmospheric_scale, mie_scale, depth_clamp, 14)
		accumulate_14 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, transmission_14, 14, accumulate_13)

		transmission_15 = self.accumulate_spectral_atmosphere_slice_transmission(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, atmospheric_scale, mie_scale, depth_clamp, 15)
		accumulate_15 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, transmission_15, 15, accumulate_14)

		transmission_16 = self.accumulate_spectral_atmosphere_slice_transmission(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, atmospheric_scale, mie_scale, depth_clamp, 16)
		accumulate_16 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, transmission_16, 16, accumulate_15)

		transmission_17 = self.accumulate_spectral_atmosphere_slice_transmission(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, atmospheric_scale, mie_scale, depth_clamp, 17)
		accumulate_17 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, transmission_17, 17, accumulate_16)

		transmission_18 = self.accumulate_spectral_atmosphere_slice_transmission(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, atmospheric_scale, mie_scale, depth_clamp, 18)
		accumulate_18 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, transmission_18, 18, accumulate_17)

		transmission_19 = self.accumulate_spectral_atmosphere_slice_transmission(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, atmospheric_scale, mie_scale, depth_clamp, 19)
		accumulate_19 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, transmission_19, 19, accumulate_18)

		transmission_20 = self.accumulate_spectral_atmosphere_slice_transmission(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, atmospheric_scale, mie_scale, depth_clamp, 20)
		accumulate_20 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, transmission_20, 20, accumulate_19)

		transmission_21 = self.accumulate_spectral_atmosphere_slice_transmission(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, atmospheric_scale, mie_scale, depth_clamp, 21)
		accumulate_21 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, transmission_21, 21, accumulate_20)

		transmission_22 = self.accumulate_spectral_atmosphere_slice_transmission(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, atmospheric_scale, mie_scale, depth_clamp, 22)
		accumulate_22 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, transmission_22, 22, accumulate_21)

		transmission_23 = self.accumulate_spectral_atmosphere_slice_transmission(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, atmospheric_scale, mie_scale, depth_clamp, 23)
		accumulate_23 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, transmission_23, 23, accumulate_22)

		transmission_24 = self.accumulate_spectral_atmosphere_slice_transmission(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, atmospheric_scale, mie_scale, depth_clamp, 24)
		accumulate_24 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, transmission_24, 24, accumulate_23)

		transmission_25 = self.accumulate_spectral_atmosphere_slice_transmission(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, atmospheric_scale, mie_scale, depth_clamp, 25)
		accumulate_25 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, transmission_25, 25, accumulate_24)

		transmission_26 = self.accumulate_spectral_atmosphere_slice_transmission(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, atmospheric_scale, mie_scale, depth_clamp, 26)
		accumulate_26 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, transmission_26, 26, accumulate_25)

		transmission_27 = self.accumulate_spectral_atmosphere_slice_transmission(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, atmospheric_scale, mie_scale, depth_clamp, 27)
		accumulate_27 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, transmission_27, 27, accumulate_26)

		transmission_28 = self.accumulate_spectral_atmosphere_slice_transmission(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, atmospheric_scale, mie_scale, depth_clamp, 28)
		accumulate_28 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, transmission_28, 28, accumulate_27)

		transmission_29 = self.accumulate_spectral_atmosphere_slice_transmission(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, atmospheric_scale, mie_scale, depth_clamp, 29)
		accumulate_29 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, transmission_29, 29, accumulate_28)

		transmission_30 = self.accumulate_spectral_atmosphere_slice_transmission(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, atmospheric_scale, mie_scale, depth_clamp, 30)
		accumulate_30 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, transmission_30, 30, accumulate_29)

		transmission_31 = self.accumulate_spectral_atmosphere_slice_transmission(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, atmospheric_scale, mie_scale, depth_clamp, 31)
		accumulate_31 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, transmission_31, 31, accumulate_30)

		transmission_32 = self.accumulate_spectral_atmosphere_slice_transmission(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, atmospheric_scale, mie_scale, depth_clamp, 32)
		accumulate_32 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, transmission_32, 32, accumulate_31)

		transmission_33 = self.accumulate_spectral_atmosphere_slice_transmission(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, atmospheric_scale, mie_scale, depth_clamp, 33)
		accumulate_33 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, transmission_33, 33, accumulate_32)

		transmission_34 = self.accumulate_spectral_atmosphere_slice_transmission(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, atmospheric_scale, mie_scale, depth_clamp, 34)
		accumulate_34 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, transmission_34, 34, accumulate_33)

		transmission_35 = self.accumulate_spectral_atmosphere_slice_transmission(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, atmospheric_scale, mie_scale, depth_clamp, 35)
		accumulate_35 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, transmission_35, 35, accumulate_34)

		transmission_36 = self.accumulate_spectral_atmosphere_slice_transmission(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, atmospheric_scale, mie_scale, depth_clamp, 36)
		accumulate_36 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, transmission_36, 36, accumulate_35)

		transmission_37 = self.accumulate_spectral_atmosphere_slice_transmission(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, atmospheric_scale, mie_scale, depth_clamp, 37)
		accumulate_37 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, transmission_37, 37, accumulate_36)

		accumulate_37.label = 'transmission XYZ accumulated'
		accumulate_37.use_custom_color = True
		accumulate_37.color = (0, 0, 1)

		return accumulate_37
	
	def accumulate_spectral_atmosphere_38_p0(self, ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, D65_ILLUMINANT, CIE_X, CIE_Y, CIE_Z, atmospheric_scale, mie_scale, depth_clamp, rayleigh_phase, mie_phase):
		xyz_start = ntree.nodes.new("FunctionNodeInputVector")
		xyz_start.vector[0] = 0.0
		xyz_start.vector[1] = 0.0
		xyz_start.vector[2] = 0.0

		inscattered_00 = self.accumulate_spectral_atmosphere_slice_inscattered(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, D65_ILLUMINANT, atmospheric_scale, mie_scale, depth_clamp, rayleigh_phase, mie_phase, 0)
		accumulate_00 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, inscattered_00, 0, xyz_start)

		inscattered_01 = self.accumulate_spectral_atmosphere_slice_inscattered(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, D65_ILLUMINANT, atmospheric_scale, mie_scale, depth_clamp, rayleigh_phase, mie_phase, 1)
		accumulate_01 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, inscattered_01, 1, accumulate_00)

		inscattered_02 = self.accumulate_spectral_atmosphere_slice_inscattered(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, D65_ILLUMINANT, atmospheric_scale, mie_scale, depth_clamp, rayleigh_phase, mie_phase, 2)
		accumulate_02 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, inscattered_02, 2, accumulate_01)

		inscattered_03 = self.accumulate_spectral_atmosphere_slice_inscattered(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, D65_ILLUMINANT, atmospheric_scale, mie_scale, depth_clamp, rayleigh_phase, mie_phase, 3)
		accumulate_03 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, inscattered_03, 3, accumulate_02)

		inscattered_04 = self.accumulate_spectral_atmosphere_slice_inscattered(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, D65_ILLUMINANT, atmospheric_scale, mie_scale, depth_clamp, rayleigh_phase, mie_phase, 4)
		accumulate_04 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, inscattered_04, 4, accumulate_03)

		inscattered_05 = self.accumulate_spectral_atmosphere_slice_inscattered(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, D65_ILLUMINANT, atmospheric_scale, mie_scale, depth_clamp, rayleigh_phase, mie_phase, 5)
		accumulate_05 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, inscattered_05, 5, accumulate_04)

		inscattered_06 = self.accumulate_spectral_atmosphere_slice_inscattered(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, D65_ILLUMINANT, atmospheric_scale, mie_scale, depth_clamp, rayleigh_phase, mie_phase, 6)
		accumulate_06 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, inscattered_06, 6, accumulate_05)

		inscattered_07 = self.accumulate_spectral_atmosphere_slice_inscattered(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, D65_ILLUMINANT, atmospheric_scale, mie_scale, depth_clamp, rayleigh_phase, mie_phase, 7)
		accumulate_07 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, inscattered_07, 7, accumulate_06)

		inscattered_08 = self.accumulate_spectral_atmosphere_slice_inscattered(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, D65_ILLUMINANT, atmospheric_scale, mie_scale, depth_clamp, rayleigh_phase, mie_phase, 8)
		accumulate_08 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, inscattered_08, 8, accumulate_07)

		inscattered_09 = self.accumulate_spectral_atmosphere_slice_inscattered(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, D65_ILLUMINANT, atmospheric_scale, mie_scale, depth_clamp, rayleigh_phase, mie_phase, 9)
		accumulate_09 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, inscattered_09, 9, accumulate_08)

		inscattered_10 = self.accumulate_spectral_atmosphere_slice_inscattered(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, D65_ILLUMINANT, atmospheric_scale, mie_scale, depth_clamp, rayleigh_phase, mie_phase, 10)
		accumulate_10 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, inscattered_10, 10, accumulate_09)

		inscattered_11 = self.accumulate_spectral_atmosphere_slice_inscattered(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, D65_ILLUMINANT, atmospheric_scale, mie_scale, depth_clamp, rayleigh_phase, mie_phase, 11)
		accumulate_11 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, inscattered_11, 11, accumulate_10)

		inscattered_12 = self.accumulate_spectral_atmosphere_slice_inscattered(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, D65_ILLUMINANT, atmospheric_scale, mie_scale, depth_clamp, rayleigh_phase, mie_phase, 12)
		accumulate_12 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, inscattered_12, 12, accumulate_11)

		inscattered_13 = self.accumulate_spectral_atmosphere_slice_inscattered(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, D65_ILLUMINANT, atmospheric_scale, mie_scale, depth_clamp, rayleigh_phase, mie_phase, 13)
		accumulate_13 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, inscattered_13, 13, accumulate_12)

		inscattered_14 = self.accumulate_spectral_atmosphere_slice_inscattered(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, D65_ILLUMINANT, atmospheric_scale, mie_scale, depth_clamp, rayleigh_phase, mie_phase, 14)
		accumulate_14 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, inscattered_14, 14, accumulate_13)

		inscattered_15 = self.accumulate_spectral_atmosphere_slice_inscattered(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, D65_ILLUMINANT, atmospheric_scale, mie_scale, depth_clamp, rayleigh_phase, mie_phase, 15)
		accumulate_15 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, inscattered_15, 15, accumulate_14)

		inscattered_16 = self.accumulate_spectral_atmosphere_slice_inscattered(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, D65_ILLUMINANT, atmospheric_scale, mie_scale, depth_clamp, rayleigh_phase, mie_phase, 16)
		accumulate_16 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, inscattered_16, 16, accumulate_15)

		inscattered_17 = self.accumulate_spectral_atmosphere_slice_inscattered(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, D65_ILLUMINANT, atmospheric_scale, mie_scale, depth_clamp, rayleigh_phase, mie_phase, 17)
		accumulate_17 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, inscattered_17, 17, accumulate_16)

		inscattered_18 = self.accumulate_spectral_atmosphere_slice_inscattered(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, D65_ILLUMINANT, atmospheric_scale, mie_scale, depth_clamp, rayleigh_phase, mie_phase, 18)
		accumulate_18 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, inscattered_18, 18, accumulate_17)

		inscattered_19 = self.accumulate_spectral_atmosphere_slice_inscattered(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, D65_ILLUMINANT, atmospheric_scale, mie_scale, depth_clamp, rayleigh_phase, mie_phase, 19)
		accumulate_19 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, inscattered_19, 19, accumulate_18)

		inscattered_20 = self.accumulate_spectral_atmosphere_slice_inscattered(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, D65_ILLUMINANT, atmospheric_scale, mie_scale, depth_clamp, rayleigh_phase, mie_phase, 20)
		accumulate_20 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, inscattered_20, 20, accumulate_19)

		inscattered_21 = self.accumulate_spectral_atmosphere_slice_inscattered(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, D65_ILLUMINANT, atmospheric_scale, mie_scale, depth_clamp, rayleigh_phase, mie_phase, 21)
		accumulate_21 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, inscattered_21, 21, accumulate_20)

		inscattered_22 = self.accumulate_spectral_atmosphere_slice_inscattered(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, D65_ILLUMINANT, atmospheric_scale, mie_scale, depth_clamp, rayleigh_phase, mie_phase, 22)
		accumulate_22 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, inscattered_22, 22, accumulate_21)

		inscattered_23 = self.accumulate_spectral_atmosphere_slice_inscattered(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, D65_ILLUMINANT, atmospheric_scale, mie_scale, depth_clamp, rayleigh_phase, mie_phase, 23)
		accumulate_23 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, inscattered_23, 23, accumulate_22)

		inscattered_24 = self.accumulate_spectral_atmosphere_slice_inscattered(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, D65_ILLUMINANT, atmospheric_scale, mie_scale, depth_clamp, rayleigh_phase, mie_phase, 24)
		accumulate_24 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, inscattered_24, 24, accumulate_23)

		inscattered_25 = self.accumulate_spectral_atmosphere_slice_inscattered(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, D65_ILLUMINANT, atmospheric_scale, mie_scale, depth_clamp, rayleigh_phase, mie_phase, 25)
		accumulate_25 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, inscattered_25, 25, accumulate_24)

		inscattered_26 = self.accumulate_spectral_atmosphere_slice_inscattered(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, D65_ILLUMINANT, atmospheric_scale, mie_scale, depth_clamp, rayleigh_phase, mie_phase, 26)
		accumulate_26 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, inscattered_26, 26, accumulate_25)

		inscattered_27 = self.accumulate_spectral_atmosphere_slice_inscattered(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, D65_ILLUMINANT, atmospheric_scale, mie_scale, depth_clamp, rayleigh_phase, mie_phase, 27)
		accumulate_27 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, inscattered_27, 27, accumulate_26)

		inscattered_28 = self.accumulate_spectral_atmosphere_slice_inscattered(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, D65_ILLUMINANT, atmospheric_scale, mie_scale, depth_clamp, rayleigh_phase, mie_phase, 28)
		accumulate_28 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, inscattered_28, 28, accumulate_27)

		inscattered_29 = self.accumulate_spectral_atmosphere_slice_inscattered(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, D65_ILLUMINANT, atmospheric_scale, mie_scale, depth_clamp, rayleigh_phase, mie_phase, 29)
		accumulate_29 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, inscattered_29, 29, accumulate_28)

		inscattered_30 = self.accumulate_spectral_atmosphere_slice_inscattered(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, D65_ILLUMINANT, atmospheric_scale, mie_scale, depth_clamp, rayleigh_phase, mie_phase, 30)
		accumulate_30 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, inscattered_30, 30, accumulate_29)

		inscattered_31 = self.accumulate_spectral_atmosphere_slice_inscattered(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, D65_ILLUMINANT, atmospheric_scale, mie_scale, depth_clamp, rayleigh_phase, mie_phase, 31)
		accumulate_31 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, inscattered_31, 31, accumulate_30)

		inscattered_32 = self.accumulate_spectral_atmosphere_slice_inscattered(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, D65_ILLUMINANT, atmospheric_scale, mie_scale, depth_clamp, rayleigh_phase, mie_phase, 32)
		accumulate_32 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, inscattered_32, 32, accumulate_31)

		inscattered_33 = self.accumulate_spectral_atmosphere_slice_inscattered(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, D65_ILLUMINANT, atmospheric_scale, mie_scale, depth_clamp, rayleigh_phase, mie_phase, 33)
		accumulate_33 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, inscattered_33, 33, accumulate_32)

		inscattered_34 = self.accumulate_spectral_atmosphere_slice_inscattered(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, D65_ILLUMINANT, atmospheric_scale, mie_scale, depth_clamp, rayleigh_phase, mie_phase, 34)
		accumulate_34 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, inscattered_34, 34, accumulate_33)

		inscattered_35 = self.accumulate_spectral_atmosphere_slice_inscattered(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, D65_ILLUMINANT, atmospheric_scale, mie_scale, depth_clamp, rayleigh_phase, mie_phase, 35)
		accumulate_35 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, inscattered_35, 35, accumulate_34)

		inscattered_36 = self.accumulate_spectral_atmosphere_slice_inscattered(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, D65_ILLUMINANT, atmospheric_scale, mie_scale, depth_clamp, rayleigh_phase, mie_phase, 36)
		accumulate_36 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, inscattered_36, 36, accumulate_35)

		inscattered_37 = self.accumulate_spectral_atmosphere_slice_inscattered(ntree, RAYLEIGH_WEIGHTS, MIE_WEIGHTS, D65_ILLUMINANT, atmospheric_scale, mie_scale, depth_clamp, rayleigh_phase, mie_phase, 37)
		accumulate_37 = self.accumulate_spectral_atmosphere_38_p1(ntree, CIE_X, CIE_Y, CIE_Z, inscattered_37, 37, accumulate_36)

		accumulate_37.label = 'inscatter XYZ accumulated'
		accumulate_37.use_custom_color = True
		accumulate_37.color = (0, 0, 1)

		return accumulate_37

	def accumulate_spectral_atmosphere_38_p1(self, ntree, CIE_x, CIE_y, CIE_z, inscatteredNode, idx, inputAdd):
		inscatter_mult_by_CIE = ntree.nodes.new("ShaderNodeVectorMath")
		inscatter_mult_by_CIE.operation = 'MULTIPLY'
		inscatter_mult_by_CIE.inputs[1].default_value[0] = CIE_x[idx]
		inscatter_mult_by_CIE.inputs[1].default_value[1] = CIE_y[idx]
		inscatter_mult_by_CIE.inputs[1].default_value[2] = CIE_z[idx]
		ntree.links.new(inscatteredNode.outputs[0], inscatter_mult_by_CIE.inputs[0])

		node_add = ntree.nodes.new("ShaderNodeVectorMath")
		node_add.operation = 'ADD'
		ntree.links.new(inputAdd.outputs[0], node_add.inputs[0])
		ntree.links.new(inscatter_mult_by_CIE.outputs[0], node_add.inputs[1])

		return node_add
