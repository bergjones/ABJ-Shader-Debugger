***LATEST UPDATE ***

###
Stable Neo-Hookean TR-BDF2 Continuum Physics with JAX Differentiation
###

The latest update to ABJ Shader Debugger demonstrates stable neo-hookean, tr-bdf2 stimestep continuum physics built upon the idea of using JAX JIT for differentiating parameters. Demonstrated on a sphere dropping onto a plane, the volume preservation and adjustable, high TR-BDF2 excitability are evident. While there is self-collision, the sphere is simulated as a plain sphere and the displacement is added in shader space. My plans for the future are first of all moving to an signed distance field SDF sphere instead of polygon for faster speeds and the possibility of far more verts in the sim as well as working on arbitrary collisions.

###
Features
###

View stages of a shader with arrows and text while it is shown in the viewport as opposed to your only option being the final output stage. This addon was originally developed for use with improving my paintings. Choose a input mesh from one of the primitives and break it up into "faces" that are shaded individually with an emissive shader. Preprocess. You can randomly rotate the mesh and set random light placement. You can choose faces to step through in any order by setting the index value on breakpoint enums and then step through them with Plus (+) or Minus (-) buttons. 

This add on was developed on Blender 5.2.1
