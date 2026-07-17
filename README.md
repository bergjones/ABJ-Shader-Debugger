###
Latest update
###

The latest update to ABJ Shader Debugger demonstrates the concept of color saturation based on distance with spectral blending, in Blender 5.2.0 compositor nodes. This allows the effect to work on the entire image as adjustable post processing, with a scripted demo. You must save a new, default scene file first and have a folder named 'compositing_files' in that directory. The spectral equation is based on the excellent Spectral.js, but I have translated it most recently into compositor nodes and in previous commits to regular bpy.

###
Features
###

View stages of a shader with arrows and text while it is shown in the viewport as opposed to your only option being the final output stage. This addon was originally developed for use with improving my paintings. Choose a input mesh from one of the primitives and break it up into "faces" that are shaded individually with an emissive shader. Preprocess. You can randomly rotate the mesh and set random light placement. You can choose faces to step through in any order by setting the index value on breakpoint enums and then step through them with Plus (+) or Minus (-) buttons. 

For now, choose between Simple Specular (RdotV) (7 steps) and GGX (17 steps).

You can step through stages in direct order, 1 to N, or 2-4-6, etc. You can set what AOV you want to see, for now spec, diffuse, or combination (Ci)

To note : This debugger clears your entire scene each time you pre-process. This addon requires numpy to be installed and resides in the Scene panel.

This add on was developed on Blender 5.2.0
