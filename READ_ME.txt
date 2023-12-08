ABOUT
--------------------------------------------------------------------------------------
The goal of this project is to develop a toolset in blender that can digitize road markings from point clouds. It should do this semi-automatically, meaning it helps the user save time and effort with techniques such as snapping, correcting, auto completing etc.
Optional is making the toolset web based and implementing AI assistance (or at least making the toolset AI ready).

INSTRUCTIONS
--------------------------------------------------------------------------------------
Confirmed to work with blender versions 3.4-3.6. Python used: 3.10.12

How to install as add-on: Blender->Edit->Preferences->add-ons->install->add-on.zip -> enable it

To develop faster, use in-blender script. To do so, open text-> full script.py. Then run with alt+p (overwrites the add-on)
To save time, it has Auto load auto.laz function, this will quickly imports a laz file called auto.laz on every script execution. Otherwise need to manually import again on every script execution to get point cloud data back.
(turned off by default)

Installing Libraries: full script.py has functions to install multiple libraries at once. Alternatively use: from pip._internal import main, main(['install','libraryname']) in blender console.

Manual import takes 1-3 minutes first time, but ~70% faster subsequent times. ~2Gb+ .laz could be unstable. 
Export to shapefile can take couple minutes. Lower points percentage to ~10% to speed it up. 

Full script has undo/redo buttons for drawn shapes because ctrl+z also undos code.. The add-on does not have those buttons, there can just use ctrl+z/ctrl+shift+z

<More detailed instructions and full documentation are being made>

FUNCTIONS
----------------------------------------------------------------------------------------------------------------
! important press ESC key to stop the operators. !

Points percentage: changes the displayed point amount to help performance, does NOT modify the amount of points. It DOES affect points exported to shapefile.

max z: cuts off height (meters) from ground level, since we only need roads. Set to 0 to not cut. Improves performance a lot. (Recommended 0.5)

Center point cloud: Centers point cloud in viewport.

Create Point Cloud Object: creates a blender object of the cloud. If BlenderGIS add-on is installed, it's ''export to shapefile'' will work on this object after selecting it.

Get point color & intensity: prints useful info of every left mouseclick in viewport.

Intensity threshold: this needs to be changed constantly, since every point cloud is different, and some road marks are less white (intensity) than others. ~140 for vague road marks, ~160 for normal, ~200 for bright ones. Too high-> wont detect, Too low-> region growing will leave the road mark. (is bad, blender might freeze for a while)

Marking z: height of all drawn shapes, 0 could mean they are harder to see.

Draw line/width: simple free line of adjustable width and color drawn at every left mouse click.

Simple fill: creates meshes based on the points found.

Complex fill: marks all points found, combines it into a shape (can mark complex symbols like bicycles)

Selection fill: performs complex fill between 2 mouse clicks.

Auto mark: performs complex fill on entire point cloud, up to a max.

Fixed Triangle/rectangle markers: simple fixed shapes of standardized size

Triangle/rectangle markers: tries to match actual road mark. click multiple triangles/rectangles in order to connect them.

Auto Triangle/rectangle markers: click the FIRST and LAST one, then it finds all in-between, and connects them all.

Snapping line marker: has optional snapping which tries to move the line to nearby road mark when clicking near said road mark, or center it on a road mark when clicking on said road mark.

Auto curvedline: draws curved polyline as far as it can go, it uses middle points to center itself.

Save shapes: saves a small picture of every shape drawn from there on out in road_mark_images.

Show dots: some functions create small red feedback dots, useful for debugging.

-----------------------------------------------------------------------------------

Issues:

- Digitizing symbols like the bicycle symbol is hard to do, it is not a geometrical shape and hard to mark.
- Dirty/noisy point clouds are challenging, region growing fails often..It includes points that it should not. Solution-> AI to recognize shapes.
  openCV shape recognition could work, but the shapes found using region growing are too imperfect and complex to find contours, and openCV is technically not AI anyway..
  Training CNN on road mark shapes such as zebra crossing, shark teeth and symbols like bicycles is better but requires large dataset and time to train. Could work in python script using TensorFlow. 

- If intensity threshold is too low, region growing might keep going for too long. It needs either a time limit, a total points limit, a distance limit or some way to interrupt it.
- Line snapping is not perfect
- Panel is too long (?) might miss buttons since you have to scroll down.
- Point cloud should be moved to (0.0.0). Now it is near impossible to find without the center point cloud button.
- The remove markings button removes ALL objects. There should be cleaner way to tag road marking objects and only remove those.

- Developing in full script.py is easier to test, but it is annoying to have to then copy those changes to the add-on code too.. On the other hand, developing in the add-on code takes too long to debug..
- The included auto.laz has triangles, road lines, bicycle symbols, speed limit marking, speedbump markings, middle lines.
  It is the fastest to test because of its small size. But it is missing: zebra crossing, arrow markings. There might be better point clouds to test on
- Webbased PoC (proof of concept) uses a (zipped) SHAPEFILE to load a point cloud in a web enviroment, using Flask and Three.js. Then, it sends coordinates of mouseclicks to blender using sockets.
  Currently does not work. Better to use geoborg enviroment(?) 

Warnings:

- The add-on can save a bunch of (big) files in the project folder, watch out for disk space!
- Blender can freeze when clicking with too low intensity_threshold. It is not crashing, it will eventually work again.
- If the add-on code is modified, remove the add-on, restart blender, make new zip of add-on folder, reinstall that add-on zip. Else errors will occur from the old add-on!
- Do not import manually when auto import is turned on. 
- Import point clouds manualy after enabling the add-on or else nothing works.










 




