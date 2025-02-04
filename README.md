Here's how it works : You choose a input mesh from one of the primitives and break it up into "faces" that are shaded individually with an emissive shader. There is a initial preprocessing step and it only works on one object. You can randomly rotate the mesh and set random light placement. You can choose faces to step through in any order by setting the index value on breakpoint enums and then step through them with Plus (+) or Minus (-) buttons. For example the example shader which I based the script on is is Diffuse (NdotL) and Spec (RdotV). I broke it up into 7 steps. You can step through 1 to 7, or 2-4-6, or go directly to 5-7. You can set what AOV you want to see, in this example simple diffuse or simple spec.

To note : This debugger clears your entire scene each time you pre-process. It is crucial that you read and edit the script. It requires numpy to be installed.

This version is greatly expanded from what I posted last month for Maya. This version is a legacy addon in the Scene panel and depends on the script tab being open so information can be printed into it.

I made this debugger as a way to quiz myself on the stages of shaders and how they are solved so I can improve my traditional painting skills as I am very into painting. For that reason I also added stereoscopic color presets to help reduce retinal rivalry for use with anaglyph glasses so I can also look at this in 3d since there are many instances in traditional painting where I am looking at a 3d, not 2d, subject.

I am looking to add more shaders from a GL engine I wrote as well as adding support for more multiple lights.
