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

bl_info = {
	"name": "ABJ Shader Debugger for Blender",
	"author" : "Aleksander Berg-Jones",
	"version" : (1, 0),
	"blender": (4, 5, 2),
	"location": "Scene",
	"description": "Shader Debugger",
	"warning": "",
	"doc_url": "",
	"tracker_url": "",	
	"category": "Scene",
}

import bpy
import sys
import importlib

from .abj_shader_debugger_for_blender_main import ABJ_Shader_Debugger, SCENE_PT_ABJ_Shader_Debugger_Panel, SHADER_OT_RANDOMLIGHT, SHADER_OT_RANDOMROTATION, SHADER_OT_RESTORELIGHT, SHADER_OT_RESTORERXYZ, SHADER_OT_STATICSTAGE1, SHADER_OT_REFRESHSTAGE2, SHADER_OT_STAGESSELECTFACES, SHADER_OT_STAGEIDXMINUS, SHADER_OT_STAGEIDXPLUS, SHADER_OT_STAGEIDXZERO, SHADER_OT_STAGEIDXPRINT, SHADER_OT_STAGERESETALL, SHADER_OT_SHOWHIDETEXTTOGGLE, SHADER_OT_SHOWHIDEARROWTOGGLE, SHADER_OT_SHOWHIDECUBECAM, SHADER_OT_RESTORECAMVIEW, SHADER_OT_AGXCOLORSETTINGS, SHADER_OT_TEXTCOLORSETTINGS, SHADER_OT_STEREOSCOPICCOLORSETTINGS, SHADER_OT_DEFAULTCOLORSETTINGS, SHADER_OT_TOGGLEEXTRAS, SHADER_OT_GRADIENTCOLOR, SHADER_OT_GRADIENTCOLORWHEEL, SHADER_OT_RENDERPASSES, SHADER_OT_WRITTENRENDER, SHADER_OT_PRESET1, SHADER_OT_SPECTRALMULTIBLEND

if "bpy" in locals():
	prefix = __package__ + '.'
	for name, module in sys.modules.copy().items():
		if name.startswith(prefix):
			basename = name.removeprefix(prefix)
			globals()[basename] = importlib.reload(module)

import bpy

classes = [
	# ABJ_Shader_Debugger,
	SCENE_PT_ABJ_Shader_Debugger_Panel,

	SHADER_OT_RANDOMLIGHT,
	SHADER_OT_RANDOMROTATION,
	SHADER_OT_RESTORELIGHT,
	SHADER_OT_RESTORERXYZ,
	SHADER_OT_STATICSTAGE1,

	SHADER_OT_REFRESHSTAGE2,

	SHADER_OT_STAGESSELECTFACES,
	SHADER_OT_STAGEIDXMINUS,
	SHADER_OT_STAGEIDXPLUS,
	SHADER_OT_STAGEIDXZERO,
	SHADER_OT_STAGEIDXPRINT,
	SHADER_OT_STAGERESETALL,

	SHADER_OT_SHOWHIDETEXTTOGGLE,
	SHADER_OT_SHOWHIDEARROWTOGGLE,
	SHADER_OT_SHOWHIDECUBECAM,

	SHADER_OT_RESTORECAMVIEW,
	SHADER_OT_AGXCOLORSETTINGS,
	SHADER_OT_TEXTCOLORSETTINGS,
	SHADER_OT_STEREOSCOPICCOLORSETTINGS,
	SHADER_OT_DEFAULTCOLORSETTINGS,
	
	SHADER_OT_TOGGLEEXTRAS,
	SHADER_OT_GRADIENTCOLOR,
	SHADER_OT_GRADIENTCOLORWHEEL,

	SHADER_OT_RENDERPASSES,

	SHADER_OT_WRITTENRENDER,

	SHADER_OT_PRESET1,

	SHADER_OT_SPECTRALMULTIBLEND,
]

def register():
	for c in classes:
		# pass
		bpy.utils.register_class(c)

	# '''

	bpy.types.Scene.written_aspect_prop = bpy.props.FloatProperty(min=0.0, max=4.0, default=1.0, name='aspect')

	bpy.types.Scene.written_fov_prop = bpy.props.FloatProperty(min=0.0, max=180.0, default=60.0, name='fov')

	bpy.types.Scene.written_znear_prop = bpy.props.FloatProperty(min=0.0, max=500, default=1.0, name='znear')

	bpy.types.Scene.written_zfar_prop = bpy.props.FloatProperty(min=20, max=1000, default=250, name='zfar')


	bpy.types.Scene.gradient_outer_circle_steps_prop = bpy.props.IntProperty(min=0, max=20, default=10, name='outer_circle_steps')


	bpy.types.Scene.gamma_correct_gradient_color_prop = bpy.props.BoolProperty(default=(True), name='gamma_correct_color')
	bpy.types.Scene.gamma_correct_gradient_colorWheel_prop = bpy.props.BoolProperty(default=(True), name='gamma_correct_colorWheel')

	default_gradientColor_0 = (0, 0, 0)
	min_gradientColor_0 = 0
	max_gradientColor_0 = 1
	bpy.types.Scene.gradient_color0_prop = bpy.props.FloatVectorProperty(default=default_gradientColor_0, min=min_gradientColor_0, max=max_gradientColor_0, name='rgb_0')

	default_gradientColor_1 = (0, 0, 0)
	min_gradientColor_1 = 0
	max_gradientColor_1 = 1
	bpy.types.Scene.gradient_color1_prop = bpy.props.FloatVectorProperty(default=default_gradientColor_1, min=min_gradientColor_1, max=max_gradientColor_1, name='rgb_1')




	default_spectralMultiBlend_0 = (1, 0, 0)
	bpy.types.Scene.spectral_multi_0_Blend_prop = bpy.props.FloatVectorProperty(default=default_spectralMultiBlend_0, min=0, max=1, precision=3, name='multi_0')

	bpy.types.Scene.spectral_multi_0_Factor_prop = bpy.props.FloatProperty(min=0, max=100, default=1, precision=3, name='factor_0')
	bpy.types.Scene.spectral_multi_0_Tint_prop = bpy.props.FloatProperty(min=0, max=1000, default=1, precision=3, name='tint_0')


	default_spectralMultiBlend_1 = (1, 1, 0)
	bpy.types.Scene.spectral_multi_1_Blend_prop = bpy.props.FloatVectorProperty(default=default_spectralMultiBlend_1, min=0, max=1, precision=3, name='multi_1')

	bpy.types.Scene.spectral_multi_1_Factor_prop = bpy.props.FloatProperty(min=0, max=100, default=1, precision=3, name='factor_1')
	bpy.types.Scene.spectral_multi_1_Tint_prop = bpy.props.FloatProperty(min=0, max=1000, default=1, precision=3, name='tint_1')

	default_spectralMultiBlend_2 = (0, 0, 1)
	bpy.types.Scene.spectral_multi_2_Blend_prop = bpy.props.FloatVectorProperty(default=default_spectralMultiBlend_2, min=0, max=1, precision=3, name='multi_2')

	bpy.types.Scene.spectral_multi_2_Factor_prop = bpy.props.FloatProperty(min=0, max=100, default=1, precision=3, name='factor_2')
	bpy.types.Scene.spectral_multi_2_Tint_prop = bpy.props.FloatProperty(min=0, max=1000, default=1, precision=3, name='tint_2')


	default_spectralMultiBlend_3 = (0, 0, 0)
	bpy.types.Scene.spectral_multi_3_Blend_prop = bpy.props.FloatVectorProperty(default=default_spectralMultiBlend_3, min=0, max=1, precision=3, name='multi_3')

	bpy.types.Scene.spectral_multi_3_Factor_prop = bpy.props.FloatProperty(min=0, max=100, default=1, precision=3, name='factor_3')
	bpy.types.Scene.spectral_multi_3_Tint_prop = bpy.props.FloatProperty(min=0, max=1000, default=1, precision=3, name='tint_3')



	
	bpy.types.Scene.gradient_outer_circle_steps_prop = bpy.props.IntProperty(min=0, max=20, default=10, name='outer_circle_steps')
	bpy.types.Scene.gradient_inner_circle_steps_prop = bpy.props.IntProperty(min=0, max=20, default=10, name='inner_circle_steps')

	bpy.types.Scene.min_shaded_prop = bpy.props.IntProperty(min=0, max=200, default=50, name='min_shaded')

	bpy.types.Scene.oren_roughness_prop = bpy.props.FloatProperty(min=0.0, max=1.0, default=0.5, name='oren_roughness')
	bpy.types.Scene.ggx_roughness_prop = bpy.props.FloatProperty(min=0.0, max=1.0, default=0.1, name='ggx_roughness')
	bpy.types.Scene.ggx_fresnel_prop = bpy.props.FloatProperty(min=0.0, max=1.0, default=0.1, name='ggx_fresnel')

	bpy.types.Scene.text_radius_0_prop = bpy.props.FloatProperty(min=0.0, max=1.0, default=0.005, name='text_radius_0')
	bpy.types.Scene.text_radius_1_prop = bpy.props.FloatProperty(min=0.0, max=1.0, default=0.6, name='text_radius_1')

	bpy.types.Scene.text_rotate_x_prop = bpy.props.FloatProperty(min=0.0, max=360.0, default=0.0, name='rx')
	bpy.types.Scene.text_rotate_y_prop = bpy.props.FloatProperty(min=0.0, max=360.0, default=0.0, name='ry')
	bpy.types.Scene.text_rotate_z_prop = bpy.props.FloatProperty(min=0.0, max=360.0, default=270.0, name='rz')

	bpy.types.Scene.text_gradient_rotate_x_prop = bpy.props.FloatProperty(min=0.0, max=360.0, default=0.0, name='rx')
	bpy.types.Scene.text_gradient_rotate_y_prop = bpy.props.FloatProperty(min=0.0, max=360.0, default=0.0, name='ry')
	bpy.types.Scene.text_gradient_rotate_z_prop = bpy.props.FloatProperty(min=0.0, max=360.0, default=0.0, name='rz')

	# [(identifier, name, description, icon, number), ...]
	additive_or_subtractive_color_blending_enum_items = (
			('additive', 'additive', 'additive'),
			('subtractive', 'subtractive', 'subtractive'),
		)
	
	bpy.types.Scene.additive_or_subtractive_color_blending_enum_prop = bpy.props.EnumProperty(
		name='additive_or_subtractive',
		description="additive_or_subtractive",
		items=additive_or_subtractive_color_blending_enum_items,
		default='subtractive',
	)

	r_dot_v_pow_enum_items = (
			('pow1', 'pow_1', 'R_dot_V pow 1'),
			('pow2', 'pow_2', 'R_dot_V pow 2'),
			('pow4', 'pow_4', 'R_dot_V pow 4'),
			('pow8', 'pow_8', 'R_dot_V pow 8'),
			('pow16', 'pow_16', 'R_dot_V pow 16'),
			('pow32', 'pow_32', 'R_dot_V pow 32'),
			('diffuse_only', 'diffuse_only', 'display diffuse only'),
		)

	bpy.types.Scene.r_dot_v_pow_enum_prop = bpy.props.EnumProperty(
		name='r_dot_v_pow',
		description="r_dot_v_pow",
		items=r_dot_v_pow_enum_items,
		default='pow8',
		# default='pow2',
		# default='pow4',
	)

	primitive_select_enum_items = (
			('cube', 'cube', 'cube primitive'),
			('uv_sphere', 'uv_sphere', 'uv_sphere primitive'),
			('ico_sphere', 'ico_sphere', 'ico_sphere primitive'),
			('cylinder', 'cylinder', 'cylinder primitive'),
			('cone', 'cone', 'cone primitive'),
			('torus', 'torus', 'torus primitive'),
			('monkey', 'monkey', 'monkey primitive'),
	)

	bpy.types.Scene.primitive_enum_prop = bpy.props.EnumProperty(
		name='primitive_select',
		description="primitive_select",
		items=primitive_select_enum_items,
		default='monkey',
	)


	spectral_multiblend_equation_enum_items = (
			('2', '2', '2'),
			('3', '3', '3'),
			('4', '4', '4'),
	)

	bpy.types.Scene.spectral_multiblend_equation_enum_prop = bpy.props.EnumProperty(
		name='spectral_multiblend_equation',
		description="spectral_multiblend_equation",
		items=spectral_multiblend_equation_enum_items,
		default='2',
	)	

	diffuse_equation_enum_items = (
			('oren', 'oren', 'oren'),
			('simple', 'simple', 'simple'),
	)

	bpy.types.Scene.diffuse_equation_enum_prop = bpy.props.EnumProperty(
		name='diffuse_equation',
		description="diffuse_equation",
		items=diffuse_equation_enum_items,
		default='oren',
	)	

	specular_equation_enum_items = (
			('GGX', 'GGX', 'GGX'),
			('simple', 'simple', 'simple'),
	)

	bpy.types.Scene.specular_equation_enum_prop = bpy.props.EnumProperty(
		name='specular_equation',
		description="specular_equation",
		items=specular_equation_enum_items,
		default='GGX',
	)

	text_rgb_precision_enum_items = (
			('-1', '-1', '-1'),
			('1', '1', '1'),
			('2', '2', '2'),
			('3', '3', '3'),
	)

	bpy.types.Scene.text_rgb_precision_enum_prop = bpy.props.EnumProperty(
		name='text_rgb_precision',
		description="text_rgb_precision",
		items=text_rgb_precision_enum_items,
		default='-1',
	)


	aov_items = (
		('spec', 'spec', 'spec'),
		('diffuse', 'diffuse', 'diffuse'),
		('Ci', 'Ci', 'Ci'),
	)

	bpy.types.Scene.aov_enum_prop = bpy.props.EnumProperty(
		name='aov',
		description="aov",
		items=aov_items,
		default='Ci',
	)

	breakpoint_override_items = (
		('regular', 'regular', 'regular'),
		('override', 'override', 'override'),
	)

	bpy.types.Scene.breakpoint_override_enum_prop = bpy.props.EnumProperty(
		name='breakpoint_override',
		description="breakpoint_override",
		items=breakpoint_override_items,
		default='regular',
	)

	subdivision_toggle_items = (
		('subd_0', 'subd_0', 'subd_0'),
		('subd_1', 'subd_1', 'subd_1'),
	)

	bpy.types.Scene.subdivision_toggle_enum_prop = bpy.props.EnumProperty(
		name='subdivision_toggle',
		description="subdivision_toggle",
		items=subdivision_toggle_items,
		default='subd_0',
	)

	#BREAKPOINT 000
	breakpoint_enum_items = (
		('-1', '-1', '-1'),
		('000', '000', '000'),
		('001', '001', '001'),
		('002', '002', '002'),
		('003', '003', '003'),
		('004', '004', '004'),
		('005', '005', '005'),
		('006', '006', '006'),
		('007', '007', '007'),
		('008', '008', '008'),
		('009', '009', '009'),
		('010', '010', '010'),
		('011', '011', '011'),
		('012', '012', '012'),
		('013', '013', '013'),
		('014', '014', '014'),
		('015', '015', '015'),
		('016', '016', '016'),
		('017', '017', '017'),
		('018', '018', '018'),
		('019', '019', '019'),
		('020', '020', '020'),
		('021', '021', '021'),
		('022', '022', '022'),
		('023', '023', '023'),
		('024', '024', '024'),
		('025', '025', '025'),
	)

	bpy.types.Scene.breakpoint_000_enum_prop = bpy.props.EnumProperty(
		name='breakpoint_000',
		description="breakpoint_000",
		items=breakpoint_enum_items,
		default='000',
	)

	bpy.types.Scene.breakpoint_001_enum_prop = bpy.props.EnumProperty(
		name='breakpoint_001',
		description="breakpoint_001",
		items=breakpoint_enum_items,
		default='001',
		# default='004',
	)

	bpy.types.Scene.breakpoint_002_enum_prop = bpy.props.EnumProperty(
		name='breakpoint_002',
		description="breakpoint_002",
		items=breakpoint_enum_items,
		default='002',
	)

	bpy.types.Scene.breakpoint_003_enum_prop = bpy.props.EnumProperty(
		name='breakpoint_003',
		description="breakpoint_003",
		items=breakpoint_enum_items,
		default='003',
	)

	bpy.types.Scene.breakpoint_004_enum_prop = bpy.props.EnumProperty(
		name='breakpoint_004',
		description="breakpoint_004",
		items=breakpoint_enum_items,
		default='004',
	)

	bpy.types.Scene.breakpoint_005_enum_prop = bpy.props.EnumProperty(
		name='breakpoint_005',
		description="breakpoint_005",
		items=breakpoint_enum_items,
		default='005',
	)

	bpy.types.Scene.breakpoint_006_enum_prop = bpy.props.EnumProperty(
		name='breakpoint_006',
		description="breakpoint_006",
		items=breakpoint_enum_items,
		default='006',
	)

	bpy.types.Scene.breakpoint_007_enum_prop = bpy.props.EnumProperty(
		name='breakpoint_007',
		description="breakpoint_007",
		items=breakpoint_enum_items,
		default='007',
	)

	bpy.types.Scene.breakpoint_008_enum_prop = bpy.props.EnumProperty(
		name='breakpoint_008',
		description="breakpoint_008",
		items=breakpoint_enum_items,
		default='008',
	)

	bpy.types.Scene.breakpoint_009_enum_prop = bpy.props.EnumProperty(
		name='breakpoint_009',
		description="breakpoint_009",
		items=breakpoint_enum_items,
		default='009',
	)

	bpy.types.Scene.breakpoint_010_enum_prop = bpy.props.EnumProperty(
		name='breakpoint_010',
		description="breakpoint_010",
		items=breakpoint_enum_items,
		default='010',
	)

	bpy.types.Scene.breakpoint_011_enum_prop = bpy.props.EnumProperty(
		name='breakpoint_011',
		description="breakpoint_011",
		items=breakpoint_enum_items,
		default='011',
	)

	bpy.types.Scene.breakpoint_012_enum_prop = bpy.props.EnumProperty(
		name='breakpoint_012',
		description="breakpoint_012",
		items=breakpoint_enum_items,
		default='012',
	)

	bpy.types.Scene.breakpoint_013_enum_prop = bpy.props.EnumProperty(
		name='breakpoint_013',
		description="breakpoint_013",
		items=breakpoint_enum_items,
		default='013',
	)

	bpy.types.Scene.breakpoint_014_enum_prop = bpy.props.EnumProperty(
		name='breakpoint_014',
		description="breakpoint_014",
		items=breakpoint_enum_items,
		default='014',
	)

	bpy.types.Scene.breakpoint_015_enum_prop = bpy.props.EnumProperty(
		name='breakpoint_015',
		description="breakpoint_015",
		items=breakpoint_enum_items,
		default='015',
	)

	bpy.types.Scene.breakpoint_016_enum_prop = bpy.props.EnumProperty(
		name='breakpoint_016',
		description="breakpoint_016",
		items=breakpoint_enum_items,
		default='016',
	)

	bpy.types.Scene.breakpoint_017_enum_prop = bpy.props.EnumProperty(
		name='breakpoint_017',
		description="breakpoint_017",
		items=breakpoint_enum_items,
		default='017',
	)

	bpy.types.Scene.breakpoint_018_enum_prop = bpy.props.EnumProperty(
		name='breakpoint_018',
		description="breakpoint_018",
		items=breakpoint_enum_items,
		default='-1',
	)

	bpy.types.Scene.breakpoint_019_enum_prop = bpy.props.EnumProperty(
		name='breakpoint_019',
		description="breakpoint_019",
		items=breakpoint_enum_items,
		default='-1',
	)

	bpy.types.Scene.breakpoint_020_enum_prop = bpy.props.EnumProperty(
		name='breakpoint_020',
		description="breakpoint_020",
		items=breakpoint_enum_items,
		default='-1',
	)

	bpy.types.Scene.breakpoint_021_enum_prop = bpy.props.EnumProperty(
		name='breakpoint_021',
		description="breakpoint_021",
		items=breakpoint_enum_items,
		default='-1',
	)

	bpy.types.Scene.breakpoint_022_enum_prop = bpy.props.EnumProperty(
		name='breakpoint_022',
		description="breakpoint_022",
		items=breakpoint_enum_items,
		default='-1',
	)

	bpy.types.Scene.breakpoint_023_enum_prop = bpy.props.EnumProperty(
		name='breakpoint_023',
		description="breakpoint_023",
		items=breakpoint_enum_items,
		default='-1',
	)

	bpy.types.Scene.breakpoint_024_enum_prop = bpy.props.EnumProperty(
		name='breakpoint_024',
		description="breakpoint_024",
		items=breakpoint_enum_items,
		default='-1',
	)

	bpy.types.Scene.breakpoint_025_enum_prop = bpy.props.EnumProperty(
		name='breakpoint_025',
		description="breakpoint_025",
		items=breakpoint_enum_items,
		default='-1',
	)
	# '''


def unregister():
	for c in classes:
		bpy.utils.unregister_class(c)

	# '''

	del bpy.types.Scene.written_aspect_prop
	del bpy.types.Scene.written_fov_prop
	del bpy.types.Scene.written_znear_prop
	del bpy.types.Scene.written_zfar_prop

	del bpy.types.Scene.gamma_correct_gradient_color_prop
	del bpy.types.Scene.gamma_correct_gradient_colorWheel_prop

	del bpy.types.Scene.gradient_color0_prop
	del bpy.types.Scene.gradient_color1_prop

	del bpy.types.Scene.spectral_multi_0_Blend_prop
	del bpy.types.Scene.spectral_multi_0_Factor_prop
	del bpy.types.Scene.spectral_multi_0_Tint_prop

	del bpy.types.Scene.spectral_multi_1_Blend_prop
	del bpy.types.Scene.spectral_multi_1_Factor_prop
	del bpy.types.Scene.spectral_multi_1_Tint_prop

	del bpy.types.Scene.spectral_multi_2_Blend_prop
	del bpy.types.Scene.spectral_multi_2_Factor_prop
	del bpy.types.Scene.spectral_multi_2_Tint_prop

	del bpy.types.Scene.spectral_multi_3_Blend_prop
	del bpy.types.Scene.spectral_multi_3_Factor_prop
	del bpy.types.Scene.spectral_multi_3_Tint_prop

	del bpy.types.Scene.gradient_outer_circle_steps_prop
	del bpy.types.Scene.gradient_inner_circle_steps_prop

	del bpy.types.Scene.min_shaded_prop
	del bpy.types.Scene.oren_roughness_prop
	
	del bpy.types.Scene.ggx_roughness_prop
	del bpy.types.Scene.ggx_fresnel_prop

	del bpy.types.Scene.additive_or_subtractive_color_blending_enum_prop

	del bpy.types.Scene.r_dot_v_pow_enum_prop
	del bpy.types.Scene.primitive_enum_prop

	del bpy.types.Scene.breakpoint_override_enum_prop
	del bpy.types.Scene.subdivision_toggle_enum_prop

	del bpy.types.Scene.aov_enum_prop

	del bpy.types.Scene.diffuse_equation_enum_prop
	del bpy.types.Scene.spectral_multiblend_equation_enum_prop
	del bpy.types.Scene.specular_equation_enum_prop

	del bpy.types.Scene.text_rgb_precision_enum_prop

	del bpy.types.Scene.text_radius_0_prop
	del bpy.types.Scene.text_radius_1_prop
	del bpy.types.Scene.text_rotate_x_prop
	del bpy.types.Scene.text_rotate_y_prop
	del bpy.types.Scene.text_rotate_z_prop

	del bpy.types.Scene.text_gradient_rotate_x_prop
	del bpy.types.Scene.text_gradient_rotate_y_prop
	del bpy.types.Scene.text_gradient_rotate_z_prop

	del bpy.types.Scene.breakpoint_000_enum_prop
	del bpy.types.Scene.breakpoint_001_enum_prop
	del bpy.types.Scene.breakpoint_002_enum_prop
	del bpy.types.Scene.breakpoint_003_enum_prop
	del bpy.types.Scene.breakpoint_004_enum_prop
	del bpy.types.Scene.breakpoint_005_enum_prop
	del bpy.types.Scene.breakpoint_006_enum_prop
	del bpy.types.Scene.breakpoint_007_enum_prop
	del bpy.types.Scene.breakpoint_008_enum_prop
	del bpy.types.Scene.breakpoint_009_enum_prop
	del bpy.types.Scene.breakpoint_010_enum_prop
	del bpy.types.Scene.breakpoint_011_enum_prop
	del bpy.types.Scene.breakpoint_012_enum_prop
	del bpy.types.Scene.breakpoint_013_enum_prop
	del bpy.types.Scene.breakpoint_014_enum_prop
	del bpy.types.Scene.breakpoint_015_enum_prop
	del bpy.types.Scene.breakpoint_016_enum_prop
	del bpy.types.Scene.breakpoint_017_enum_prop
	del bpy.types.Scene.breakpoint_018_enum_prop
	del bpy.types.Scene.breakpoint_019_enum_prop
	del bpy.types.Scene.breakpoint_020_enum_prop
	del bpy.types.Scene.breakpoint_021_enum_prop
	del bpy.types.Scene.breakpoint_022_enum_prop
	del bpy.types.Scene.breakpoint_023_enum_prop
	del bpy.types.Scene.breakpoint_024_enum_prop
	del bpy.types.Scene.breakpoint_025_enum_prop

	# '''