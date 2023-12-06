About this project:

The task is to develop a toolset in blender that can digitize road markings from point clouds. It should do this semi-automatically, meaning it helps the user save time and effort with techniques such as snapping, correcting, auto completing etc.

Optional is: a web environment for the interface, implementing AI assistance or at least making the code AI ready.

INSTRUCTIONS
--------------------------------------------------------------------------------------
Confirmed to work with blender versions 3.4-3.6. Python used: 3.10.12

How to install as add-on: Blender->Edit->Preferences->add-ons->install->add-on.zip
Changed add-on scripts? -> make new zip file of add-on folder. Often needs blender restart to prevent errrors

To develop, use in-blender script. To do so, open text-> full script.py. Then run with alt+p
To save time, it has Auto load auto.laz function, will quickly import a laz file called auto.laz on script execution.

Libraries: full script.py has functions to install multiple libraries at once. Alternatively use: from pip._internal import main and then main(['install','libraryname']) in blender console.

Manual import takes 1-3 minutes first time, but 60-70% faster subsequent times. ~2Gb+ .laz are unstable.
Export to shapefile can take 5-10 minutes. Lower points percentage to 10-20% to speed it up. 

''Points percentage'' changes displayed point amount to help performance, NOT the amount of points. It affects export to shapefile.
"max z" cuts off height (meters), since we only need roads. Set to 0 to not cut. Improves performance a lot.

Use center point cloud to find the cloud.
Use Create Point Cloud Object to create blender object of the cloud.

Get point color & intensity print useful info around every left mouseclick in viewport.

Intensity threshold: this needs to be changed constantly, since every point cloud is different, and some road marks are less white (intensity) than others. ~140 for vague road marks, ~160 for normal, ~200 for bright ones. Too high-> wont detect, Too low-> region growing will leave the road mark (is bad)

Marking z = height of all drawn shapes, 0 could mean they are harder to see
Draw line/width: simple line drawn at every left mouse click.

! important press ESC key  to stop the drawing/marker functions. !

Simple fill: creates meshes based on the points found
Complex fill: marks all points found, combines it into a shape (can mark complex symbols like bicycles)
Selection fill: performs complex fill between 2 mouse clicks
Auto mark: performs complex fill on entire point cloud, up to a max.

Fixed Triangle/rectangle markers: fixed shapes
Triangle/rectangle markers: Try to match actual road mark. click multiple triangles/rectangles in order to connect them.
Auto Triangle/rectangle markers: click the FIRST and LAST one, then it finds all in-between, and connects them all.

Curved line marker: has optional snapping which tries to move the line to nearby road mark.
Auto curvedline: draws curved polyline as far as it can go, it uses middle points to center itself.

Save shapes: saves a small picture of every shape drawn from now on in road_mark_images
Show dots: some functions create small feedback dots, useful for debugging

Full script has undo/redo buttons for drawn shapes because ctrl+z also undos code.. The add-on does not have those buttons, there can just use ctrl+z/ctrl+shift+z

More detailed instructions and full documentation are being made.
-----------------------------------------------------------------------------------

Issues:

snap_line_to_road_mark does not function. (Snap correction is important to have for this project..)

''Dirty'' point clouds are challenging, region growing fails often..It includes points that it should not. Solution-> AI to recognize shapes.
I tried openCV shape recognition, but could not figure it out, the shapes found using region growing are too imperfect and complex to find contours, and openCV is technically not AI anyway.
Training CNN on road mark shapes such as zebra crossing, shark teeth and symbols like bicycles -> requires large dataset. Could work in python script using TensorFlow. No time to try it out.



 




