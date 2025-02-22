View stages of a shader with arrows and text while it is shown in the viewport as opposed to your only option being the final output stage. This addon was originally developed for use with improving my traditional realist paintings where I have to really understand what is going on for to be able to paint in the realist style with tradtional mediums. Choose a input mesh from one of the primitives and break it up into "faces" that are shaded individually with an emissive shader. There are preprocessing stages. You can randomly rotate the mesh and set random light placement. You can choose faces to step through in any order by setting the index value on breakpoint enums and then step through them with Plus (+) or Minus (-) buttons. 

For now, choose between Simple Specular (RdotV) (7 steps) and GGX (17 steps).

You can step through stages in direct order, 1 to N, or 2-4-6, etc. You can set what AOV you want to see, for now spec, diffuse, or combination (Ci)

To note : This debugger clears your entire scene each time you pre-process. This addon requires numpy to be installed and resides in the Scene panel.

I made this debugger as a way to quiz myself on the stages of shaders and how they are solved so I can improve my traditional painting skills as I am very into painting. I also added stereoscopic color presets to help reduce retinal rivalry for use with anaglyph glasses so I can also look at this in 3d since there are many instances in traditional painting where I am looking at a 3d, not 2d, subject.

I am looking to complete the shaders that I have and add more. This add on was developed on Blender 4.3.2
