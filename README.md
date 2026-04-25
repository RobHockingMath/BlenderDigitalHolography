How to use:

If you already have a blender screne, load and run the "create_hogel_lightfield_rig.py" script to add the hogel images rendering rig into the scene.  If you don't have a scene, load and run "create_rubiks_cube.py" to create one.  

You may want to take a look at hogel_params.py and adapt it to your setup.

Then, load and run the render_hogels_from_rig.py script to start rendering hogel images.  This might take quite some time, depending on scene complexity and the number of hogels.  Rather than measuring progress within Blender, I find it easier
to just look at the folder "hogel_renders" that the images are being written to.  Periodically check how many files are in it (there will be one per hogel, so compute ahead of time how many hogels you have).  

You can then preview what your hologram should look like by running from the terminal "python hologram_lightfield_viewer.py".  Run "pip install pygame moderngl pillow numpy" from the terminal to make sure you have everything.

IN ITS CURRENT FORM, THIS ONLY WORKS IF THE ENTIRE SCENE IS BEHIND THE HOLOGRAM PLANE.
