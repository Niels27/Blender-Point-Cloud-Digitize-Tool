ABOUT
--------------------------------------------------------------------------------------
The goal of this project is to develop a toolset in blender that can digitize road markings from point clouds. It should do this semi-automatically, meaning it helps the user save time and effort with techniques such as snapping, correcting, auto completing etc.
Optional is making the toolset web based and implementing AI assistance.

INSTRUCTIONS
--------------------------------------------------------------------------------------
Confirmed to work with blender versions 3.4, 3.5, 3.6. Python used: 3.10.12

Before you start, save the blender project first, otherwise filepaths wont work!

How to install as add-on: Blender->Edit->Preferences->add-ons->install->add-on.zip -> enable it

To develop faster, use in-blender script. To do so, open text-> full script.py. Then run with alt+p (overwrites the add-on)
To save time, it has Auto load auto.laz function, this will quickly imports a laz file called auto.laz on every script execution. Blender does not remember data between executions!

Installing Libraries: full script.py has functions to install multiple libraries at once. Alternatively use: from pip._internal import main, main(['install','libraryname']) in blender console.

Manual import takes 1-3 minutes first time, but ~70% faster subsequent times. 
Export to shapefile can take couple minutes. Lower points percentage to ~10% to speed it up. 


FUNCTIONS
----------------------------------------------------------------------------------------------------------------
! important ! press ESC key to stop the operators. Be careful left-clicking around while an operator is active !

Points percentage: changes the displayed point amount to help performance, does NOT modify the amount of points. It DOES affect points exported to shapefile.

max z: cuts off height (meters) from ground level, since we only need roads. Set to 0 to not cut. Improves performance a lot. (Recommended 0.5)

Center point cloud: Centers point cloud in viewport.

Create Point Cloud Object: creates a blender object of the cloud. If BlenderGIS add-on is installed, it's ''export to shapefile'' will work on this object after selecting it.

Click Info: prints useful info of every left mouseclick in viewport.

Intensity threshold: this needs to be changed constantly, since every point cloud is different, and some road marks are less white (intensity) than others. ~140 for vague road marks, ~160 for normal, ~200 for bright ones. Too high-> wont detect, Too low-> region growing will leave the road mark. (is bad, blender might freeze for a while)

Marking height: height of all drawn shapes, 0 could mean they are harder to see.

Simple marker: creates meshes based on the points found by region growing.

Complex marker: marks all points found by region growing, combines it into a shape.

Selection marker: performs complex fill between 2 mouse clicks.

Auto mark: performs complex fill on entire point cloud, up to a max.

Fixed Triangle/rectangle markers: simple fixed shapes of standardized size

Triangle/rectangle markers: tries to match actual road mark. click multiple triangles/rectangles in order to connect them.

Auto Triangle/rectangle markers: click the first and last one, then it finds all in-between, and connects them all.

Line marker: has optional snapping which tries to move the line to nearby road mark when clicking near said road mark, or center it on a road mark when clicking on said road mark.

Auto line marker: draws curved polyline as far as it can go, it uses middle points to center itself.

Curb marker: finds curbs between 2 mouse clicks, then marks the top and bottom using 2 lines.

Save shapes: saves a small picture of every shape drawn from there on out in road_mark_images.

Show dots: some functions create small red feedback dots, useful for debugging.

-----------------------------------------------------------------------------------

Issues:

- Digitizing symbols like the bicycle symbol is hard to do, it is not a geometrical shape and hard to mark.
- Dirty/noisy point clouds are challenging, region growing fails often..It includes points that it should not. Solution-> AI to recognize shapes.
  openCV shape recognition could work, but the shapes found using region growing are too imperfect and complex to find contours, and openCV is technically not AI anyway..
  Training CNN on road mark shapes such as zebra crossing, shark teeth and symbols like bicycles is better but requires large dataset and time to train. Could work in python script using TensorFlow. 
- Webbased PoC (proof of concept) uses a (zipped) SHAPEFILE to load a point cloud in a web enviroment, using Flask and Three.js. Then, it sends coordinates of mouseclicks to blender using sockets.
  Currently does not work.  

Warnings:

- The add-on can save a bunch of (big) files in the project folder, watch out for disk space!
- Blender can freeze when clicking with too low intensity_threshold. It is not crashing, it will eventually work again. Region growing function has a time limit variable, currently 15s.
- If the add-on code is modified, remove the add-on, restart blender, make new zip of add-on folder, reinstall that add-on zip. Else errors will occur from the old add-on!











 




