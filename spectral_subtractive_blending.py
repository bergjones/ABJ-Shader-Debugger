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

class myEquation_spectral_Kubelka_Munk:
	def __init__(self):
		super(myEquation_spectral_Kubelka_Munk, self).__init__()


	#This function is based on spectral3_glsl.py, under MIT license by Ronald van Wijnen (see file)
	def kubelka_munk_subtractive_mixing_spectral_compositor(self, myABJ_SD_B):

		myABJ_SD_B.deselectAll()
		myABJ_SD_B.deleteAllObjects()
		myABJ_SD_B.mega_purge()

		bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True

		compGroupName = 'spectralCompositor_kubla_munk'

		nodetree = bpy.data.node_groups.new(compGroupName, "CompositorNodeTree")
		bpy.context.scene.compositing_node_group = nodetree

		for node in nodetree.nodes:
			nodetree.nodes.remove(node)

		node0 = nodetree.nodes.new("CompositorNodeRLayers")

		nodetree.interface.new_socket(name="Output", in_out='OUTPUT', socket_type='NodeSocketColor')
		myABJ_SD_B.nodeOut = nodetree.nodes.new("NodeGroupOutput")

		myABJ_SD_B.nodeViewer = nodetree.nodes.new("CompositorNodeViewer")

		# myABJ_SD_B.spectral_compositor_debugging_exit_visualizer(nodetree, node0, 932, 633, myABJ_SD_B)
		# return

		###################
		# CONSTANTS
		###################
		# SPECTRAL_GAMMA
		myABJ_SD_B.node_spectral_gamma = nodetree.nodes.new("ShaderNodeValue")
		myABJ_SD_B.node_spectral_gamma.outputs[0].default_value = 2.4
		# myABJ_SD_B.node_spectral_gamma.outputs[0].default_value = 2.2

		# SPECTRAL_EPSILON
		myABJ_SD_B.node_spectral_epsilon = nodetree.nodes.new("ShaderNodeValue")
		myABJ_SD_B.node_spectral_epsilon.outputs[0].default_value = 0.0000000000000001

		#SPECTRAL_SIZE
		myABJ_SD_B.node_spectral_size = nodetree.nodes.new("FunctionNodeInputInt")
		myABJ_SD_B.node_spectral_size.integer = 38

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

		lrgb1_r = self.spectral_compositor_uncompand(nodetree, node0, 0, myABJ_SD_B)
		lrgb1_g = self.spectral_compositor_uncompand(nodetree, node0, 1, myABJ_SD_B)
		lrgb1_b = self.spectral_compositor_uncompand(nodetree, node0, 2, myABJ_SD_B)

		nodetree.links.new(lrgb1_r.outputs[0], node_lrgb1_combo.inputs[0])
		nodetree.links.new(lrgb1_g.outputs[0], node_lrgb1_combo.inputs[1])
		nodetree.links.new(lrgb1_b.outputs[0], node_lrgb1_combo.inputs[2])

		# print('READPIXEL : node_lrgb1_combo')
		# self.spectral_compositor_debugging_exit_visualizer(nodetree, node_lrgb1_combo, 932, 633, myABJ_SD_B)
		# return

		##############
		## lrgb2
		##############
		node_lrgb2_combo = nodetree.nodes.new("ShaderNodeCombineXYZ")

		lrgb2_r = self.spectral_compositor_uncompand(nodetree, node_color2, 0, myABJ_SD_B)
		lrgb2_g = self.spectral_compositor_uncompand(nodetree, node_color2, 1, myABJ_SD_B)
		lrgb2_b = self.spectral_compositor_uncompand(nodetree, node_color2, 2, myABJ_SD_B)

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
			spectralR_divide.label = 'spectralR_divide'
			nodetree.links.new(ksMix_R1_and_R2.outputs[0], spectralR_divide.inputs[0])
			nodetree.links.new(totalConcentration.outputs[0], spectralR_divide.inputs[1])

			R[i] = self.spectral_compositor_KM(nodetree, spectralR_divide)

		reflectanceEnd = self.spectral_compositor_reflectance_to_xyz_p0(nodetree, R)
		reflectanceEnd.label = 'reflectanceEnd'
		reflectanceEnd.use_custom_color = True
		reflectanceEnd.color = (0, 0, 1)

		# print('READPIXEL : reflectanceEnd')
		# self.spectral_compositor_debugging_exit_visualizer(nodetree, reflectanceEnd, 932, 633, myABJ_SD_B)
		# return

		spectral_xyz_to_srgb_end = self.spectral_compositor_xyz_to_srgb(nodetree, reflectanceEnd, myABJ_SD_B)

		print('READPIXEL : spectral_xyz_to_srgb_end')
		self.spectral_compositor_debugging_exit_visualizer(nodetree, spectral_xyz_to_srgb_end, 932, 633, myABJ_SD_B)
		return

	def sub_spectral_compositor_stock(self, myABJ_SD_B):
		print('to do')
		return

		myABJ_SD_B.deselectAll()
		myABJ_SD_B.deleteAllObjects()
		myABJ_SD_B.mega_purge()

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

		nodetree.links.new(nodeToView.outputs[0], myABJ_SD_B.nodeOut.inputs[0]) ###### !!!!!!!!!!!
		nodetree.links.new(nodeToView.outputs[0], myABJ_SD_B.nodeViewer.inputs[0]) ###### !!!!!!!!!!!

		myABJ_SD_B.autoArrangeNodes(nodetree)
		# myABJ_SD_B.autoArrangeNodes(worldtree)

		myABJ_SD_B.compositor_setup = True

		return

	def spectral_compositor_debugging_exit_visualizer(self, nodetree, nodeToView, readPixelX, readPixelY, myABJ_SD_B):
		
		nodetree.links.new(nodeToView.outputs[0], myABJ_SD_B.nodeOut.inputs[0]) ###### !!!!!!!!!!!
		nodetree.links.new(nodeToView.outputs[0], myABJ_SD_B.nodeViewer.inputs[0]) ###### !!!!!!!!!!!

		myABJ_SD_B.autoArrangeNodes(nodetree)
		# myABJ_SD_B.autoArrangeNodes(worldtree)

		myABJ_SD_B.compositor_setup = True

		# return

		############################
		#the user must have saved a new, default scene to a file and have a folder called 'compositing_files' in that directory
		############################

		###########
		#DEFAULT CAMERA
		#############
		# myABJ_SD_B.pos_camera_global = (10, 10, 10) #spectral
		# myABJ_SD_B.pos_camera_global = (20, -30, 15) #spectral
		# myABJ_SD_B.pos_camera_global = (5, -20, 10) #spectral ####
		# myABJ_SD_B.pos_camera_global = (2, -15, 10) #spectral
		myABJ_SD_B.pos_camera_global = (2, -13.5, 8) #spectral

		cam1_data = bpy.ops.object.camera_add(
			location=(myABJ_SD_B.pos_camera_global),  # x, y, z coordinates
			rotation=(0.0, 0.0, 0.0)   # x, y, z rotation in radians
		)

		myABJ_SD_B.myCam = bpy.data.objects["Camera"]
		myABJ_SD_B.myCam.data.clip_start = 1
		# # myABJ_SD_B.myCam.data.clip_start = .1
		# # myABJ_SD_B.myCam.data.clip_start = .5
		myABJ_SD_B.myCam.data.clip_end = 100
		# myABJ_SD_B.myCam.data.clip_end = 500

		bpy.context.scene.camera = myABJ_SD_B.myCam

		myABJ_SD_B.updateScene() # need
		# myABJ_SD_B.look_at(myABJ_SD_B.myCam, myInputMesh3.location)

		bpy.context.scene.render.engine = 'CYCLES'
		bpy.context.scene.cycles.device = 'GPU'

		# bpy.context.scene.render.resolution_x = 3840
		# bpy.context.scene.render.resolution_y = 2160

		# bpy.context.scene.cycles.samples = 1024
		bpy.context.scene.cycles.samples = 64

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

		mat1 = myABJ_SD_B.newShader("principled_test_00", "principled", 1, 0, 0)
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

		myABJ_SD_B.autoArrangeNodes(mat.node_tree)

		########
		##### 01
		########
		bpy.ops.mesh.primitive_monkey_add()
		myInputMesh2 = bpy.context.active_object
		myInputMesh2.select_set(1)
		myInputMesh2.location = mathutils.Vector((1, 8, 5))
		bpy.ops.object.shade_smooth()

		mat1 = myABJ_SD_B.newShader("principled_test_01", "principled", 1, 0, 0)
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

		myABJ_SD_B.autoArrangeNodes(mat.node_tree)

		# mat1 = myABJ_SD_B.newShader("greenM", "emission", 0, 1, 0)
		# mat1 = myABJ_SD_B.newShader("greenM", "diffuse", 0, 1, 0)
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

		mat1 = myABJ_SD_B.newShader("principled_test_02", "principled", 1, 0, 0)
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

		myABJ_SD_B.autoArrangeNodes(mat.node_tree)

		########
		##### 03
		########
		bpy.ops.mesh.primitive_monkey_add()
		myInputMesh = bpy.context.active_object
		myInputMesh.select_set(1)
		myInputMesh.location = mathutils.Vector((5, 16, 5))
		bpy.ops.object.shade_smooth()

		mat1 = myABJ_SD_B.newShader("principled_test_03", "principled", 1, 0, 0)
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

		myABJ_SD_B.autoArrangeNodes(mat.node_tree)

		####
		# CUBE GROUND
		####
		bpy.ops.mesh.primitive_cube_add()
		myInputMesh3 = bpy.context.active_object
		myInputMesh3.select_set(1)
		myInputMesh3.location = mathutils.Vector((0, 0, -5))
		myInputMesh3.scale = mathutils.Vector((20, 20, 2))

		# mat1 = myABJ_SD_B.newShader("blueM", "emission", 0, 0, 1)
		# mat1 = myABJ_SD_B.newShader("blueM", "diffuse", 0, 0, 1)
		# bpy.context.active_object.data.materials.clear()
		# bpy.context.active_object.data.materials.append(mat1)

		mat1 = myABJ_SD_B.newShader("principled_test_grd", "principled", 0, 0, 1)
		bpy.context.active_object.data.materials.clear()
		bpy.context.active_object.data.materials.append(mat1)
		bpy.data.materials["principled_test_grd"].node_tree.nodes["Principled BSDF"].inputs[1].default_value = 1
		bpy.data.materials["principled_test_grd"].node_tree.nodes["Principled BSDF"].inputs[2].default_value = 0.323263

		myABJ_SD_B.autoArrangeNodes(mat.node_tree)

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

		myABJ_SD_B.look_at(myABJ_SD_B.myCam, myInputMesh2.location)

		nodetree.links.new(nodeToView.outputs[0], myABJ_SD_B.nodeOut.inputs[0]) ###### !!!!!!!!!!!
		nodetree.links.new(nodeToView.outputs[0], myABJ_SD_B.nodeViewer.inputs[0]) ###### !!!!!!!!!!!

		myABJ_SD_B.autoArrangeNodes(nodetree)
		myABJ_SD_B.autoArrangeNodes(worldtree)

		myABJ_SD_B.compositor_setup = True

		########################################################
		#write node to disc and read pixel
		########################################################
		# bpy.context.scene.view_settings.view_transform = 'Standard'
		# bpy.context.scene.view_settings.look = 'AgX - Punchy'
		# bpy.context.scene.view_settings.look = 'None'

		bpy.context.scene.view_settings.view_transform = 'AgX'
		bpy.context.scene.view_settings.look = 'AgX - Punchy'

		bpy.context.scene.render.use_multiview = False

		return

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

		myABJ_SD_B.updateScene()

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
	def spectral_compositor_xyz_to_srgb(self, nodetree, xyz_combo, myABJ_SD_B):

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
		# self.spectral_compositor_debugging_exit_visualizer(nodetree, node_dotProd_R, 932, 633, myABJ_SD_B)
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

		return self.spectral_compositor_linear_to_srgb(nodetree, dotProdCombo_custom, myABJ_SD_B)

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
	
	def spectral_compositor_clamp_0180(self, nodetree, inputNode):
		node_min = nodetree.nodes.new("ShaderNodeMath")
		node_min.operation = 'MINIMUM'
		node_min.inputs[1].default_value = 180
		nodetree.links.new(inputNode.outputs[0], node_min.inputs[0])

		node_max = nodetree.nodes.new("ShaderNodeMath")
		node_max.operation = 'MAXIMUM'
		node_max.inputs[0].default_value = 0
		nodetree.links.new(node_min.outputs[0], node_max.inputs[1])

		return node_max

	#This function is based on spectral3_glsl.py, under MIT license by Ronald van Wijnen (see file)
	def spectral_compositor_linear_to_srgb(self, nodetree, dotProdCombo, myABJ_SD_B):
		dotProdCombo_sep = nodetree.nodes.new("ShaderNodeSeparateXYZ")
		dotProdCombo_sep.label = 'dotProdCombo_sep'
		nodetree.links.new(dotProdCombo.outputs[0], dotProdCombo_sep.inputs[0])

		compandR = self.spectral_compositor_compand(nodetree, dotProdCombo_sep, 0, myABJ_SD_B)
		compandG = self.spectral_compositor_compand(nodetree, dotProdCombo_sep, 1, myABJ_SD_B)
		compandB = self.spectral_compositor_compand(nodetree, dotProdCombo_sep, 2, myABJ_SD_B)

		compandR_clamped = self.spectral_compositor_clamp_01(nodetree, compandR)
		compandG_clamped = self.spectral_compositor_clamp_01(nodetree, compandG)
		compandB_clamped = self.spectral_compositor_clamp_01(nodetree, compandB)

		finalColor = nodetree.nodes.new("ShaderNodeCombineXYZ")

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
	def spectral_compositor_uncompand(self, nodetree, inputCombo, idx, myABJ_SD_B):
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
		nodetree.links.new(myABJ_SD_B.node_spectral_gamma.outputs[0], node_math5.inputs[1])

		nodetree.links.new(node_math5.outputs[0], node_mix1.inputs['B'])

		return node_mix1
	
	#This function is based on spectral3_glsl.py, under MIT license by Ronald van Wijnen (see file)
	def spectral_compositor_compand(self, nodetree, inputNode, idx, myABJ_SD_B):
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
		nodetree.links.new(myABJ_SD_B.node_spectral_gamma.outputs[0], node_math3.inputs[1])

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
		# self.spectral_compositor_debugging_exit_visualizer(nodetree, node_cmy_debug_combo, 932, 633, myABJ_SD_B)
		# # self.spectral_compositor_debugging_exit_visualizer(nodetree, node_rgb_debug_combo, 932, 633, myABJ_SD_B)
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

