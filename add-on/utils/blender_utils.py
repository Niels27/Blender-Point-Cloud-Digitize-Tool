#library imports
import sys
import os
import bpy
from bpy_extras import view3d_utils
import gpu
import bmesh
from gpu_extras.batch import batch_for_shader
import numpy as np
import open3d as o3d
import bgl
import blf
import time
import math
import laspy as lp
import scipy
import shapely
from scipy.spatial import KDTree, cKDTree, ConvexHull
from scipy.spatial.distance import cdist
from mathutils import Vector, Matrix
import mathutils
import pickle
import gzip
import pandas as pd
import geopandas as gpd
import copy
import subprocess
import json
from shapely.geometry import Point

#global variables
save_json=False
point_cloud_name="Point cloud"
point_cloud_point_size=1

#Blender utility functions
#Singleton class to save point cloud data
class GetPointCloudData:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GetPointCloudData, cls).__new__(cls)
            cls._instance.point_coords = None
            cls._instance.point_colors = None
            cls._instance.points_kdtree = None
            cls._instance.original_coords = None
            
        return cls._instance
    
    #Function to load the point cloud, store it's data and draw it using openGL, optimized version
    def pointcloud_load_optimized(self,path, point_size, sparsity_value,z_height_cut_off):
        
        start_time = time.time()
        print("Started loading point cloud.."),
        global use_pickled_kdtree,point_cloud_name,point_cloud_point_size, save_json
        overwrite_data= bpy.context.scene.overwrite_existing_data
        
        base_file_name = os.path.basename(path)
        point_cloud_name = base_file_name
        directory_path = os.path.dirname(path)
        blend_file_path = bpy.data.filepath
        if blend_file_path:
            directory = os.path.dirname(blend_file_path)
        else:
        #Prompt the user to save the file first 
            print("please save blender project first!")
            return
        directory = os.path.dirname(blend_file_path)
        JSON_data_path = os.path.join(directory, "JSON files")
        shapefiles_data_path = os.path.join(directory, "shapefiles")
        stored_data_path = os.path.join(directory, "stored data")
        file_name_points = base_file_name + "_points.npy"
        file_name_colors = base_file_name + "_colors.npy"
        file_name_avg_coords = base_file_name + "_avgCoords.npy"
        file_name_kdtree = base_file_name + "_kdtree.joblib"
        file_name_kdtree_pickle = base_file_name + "_kdtree.gz"
    
        if not os.path.exists(stored_data_path):
            os.mkdir(stored_data_path)
        
        if not os.path.exists(os.path.join(stored_data_path, file_name_points)) or overwrite_data:
            
            point_cloud = lp.read(path) 
            ground_code = 2 #the ground is usually 2

            print(f"Using classification code for GROUND: {ground_code}")
            use_ground_points_only=bpy.context.scene.ground_only
            if use_ground_points_only:
                # Filter points based on classification
                ground_points_mask = point_cloud.classification == ground_code
                if ground_points_mask.any():
                    # Applying the ground points mask
                    points_a = np.vstack((point_cloud.x[ground_points_mask], 
                                        point_cloud.y[ground_points_mask], 
                                        point_cloud.z[ground_points_mask])).transpose()
                    colors_a = np.vstack((point_cloud.red[ground_points_mask], 
                                        point_cloud.green[ground_points_mask], 
                                        point_cloud.blue[ground_points_mask])).transpose() / 65535
                else:
                    print("classification ", ground_code, " not found")
            else: 
                points_a = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
                colors_a = np.vstack((point_cloud.red, point_cloud.green, point_cloud.blue)).transpose() / 65535
                
            #Convert points to float32
            points_a = points_a.astype(np.float32)
            #Convert colors to uint8
            colors_a = (colors_a * 255).astype(np.uint8)
            
            #Sort the points based on the Z coordinate
            sorted_points = points_a[points_a[:, 2].argsort()]

            #Determine the cutoff index for the lowest %
            cutoff_index = int(len(sorted_points) * 0.1)

            #Calculate the average Z value of the lowest % of points
            road_base_level = np.mean(sorted_points[:cutoff_index, 2])

            print("Estimated road base level:", road_base_level)
            
            #Store the original coordinates 
            self.original_coords = np.copy(points_a)
            
            if(z_height_cut_off>0):
                #Filter points with Z coordinate > 0.5
                print("Number of points before filtering:", len(points_a))
                mask = points_a[:, 2] <= (road_base_level+z_height_cut_off)
                points_a = points_a[mask]
                colors_a = colors_a[mask]
                print("Number of points after filtering:", len(points_a))
                
            #Shifting coords
            points_a_avg = np.mean(points_a, axis=0)
            points_a = points_a - points_a_avg
       
            #Storing Shifting coords 
            np.save(os.path.join(stored_data_path, file_name_avg_coords), points_a_avg)
        
            #Storing the centered coordinate arrays as npy file
            np.save(os.path.join(stored_data_path, file_name_points), points_a)
            np.save(os.path.join(stored_data_path, file_name_colors), colors_a)
                    
        else:
            points_a = np.load(os.path.join(stored_data_path, file_name_points))
            colors_a = np.load(os.path.join(stored_data_path, file_name_colors))
            self.original_coords = np.load(os.path.join(stored_data_path, file_name_avg_coords))
            
        #Store point data and colors globally
        self.point_coords = points_a
        self.point_colors = colors_a
        point_cloud_point_size = point_size
        
        print("point cloud loaded in: ", time.time() - start_time)
        
        step = int(1 / sparsity_value)

        #Evenly sample points using the provided sparsity value
        reduced_points = points_a[::step]
        reduced_colors = colors_a[::step]
            
        #Save json file of point cloud data
        if save_json:
            export_as_json(reduced_points,reduced_colors,JSON_data_path,point_cloud_name)
  
        #Function to save KD-tree with pickle and gzip
        def save_kdtree_pickle_gzip(file_path, kdtree):
            with gzip.open(file_path, 'wb', compresslevel=1) as f:  # compresslevel from 1-9, low-high compression
                pickle.dump(kdtree, f)
        #Function to load KD-tree with pickle and gzip
        def load_kdtree_pickle_gzip(file_path):
            with gzip.open(file_path, 'rb') as f:
                return pickle.load(f)  
            
        use_pickled_kdtree=True
        if use_pickled_kdtree:
            #KDTree handling
            kdtree_pickle_path = os.path.join(stored_data_path, file_name_kdtree_pickle)
            if not os.path.exists(kdtree_pickle_path) or overwrite_data:
                #Create the kdtree if it doesn't exist
                print("creating cKDTree..")
                start_time = time.time()
                self.points_kdtree = cKDTree(np.array(self.point_coords))
                save_kdtree_pickle_gzip(kdtree_pickle_path, self.points_kdtree)
                print("Compressed cKD-tree created at:", kdtree_pickle_path," in:",time)
            else:
                print("kdtree found at: ",kdtree_pickle_path, "loading..")
                self.points_kdtree = load_kdtree_pickle_gzip(kdtree_pickle_path)
                print("Compressed cKD-tree loaded from gzip file in:", time.time() - start_time)
        else:  
            #KDTree handling
            kdtree_path = os.path.join(stored_data_path, file_name_kdtree)
            self.points_kdtree = load_kdtree_from_file(kdtree_path)
            if not os.path.exists(kdtree_pickle_path) or bpy.types.Scene.overwrite_existing_data == False:
                #create the kdtree if it doesn't exist
                self.points_kdtree = cKDTree(np.array(self.point_coords))
                print("kdtree created in: ", time.time() - start_time)
                #Save the kdtree to a file
                save_kdtree_to_file(kdtree_path, self.points_kdtree)
                print("kdtree saved in: ", time.time() - start_time, "at", kdtree_path)
            
        try: 
            redraw_viewport()
            draw_handler = bpy.app.driver_namespace.get('my_draw_handler')
            
            if draw_handler is None:
                #colors_ar should be in uint8 format
                reduced_colors = reduced_colors / 255.0  #Normalize to 0-1 range
                #Converting to tuple 
                coords = tuple(map(tuple, reduced_points))
                colors = tuple(map(tuple, reduced_colors))
                
                shader = gpu.shader.from_builtin('3D_FLAT_COLOR')
                batch = batch_for_shader(
                    shader, 'POINTS',
                    {"pos": coords, "color": colors}
                )
                
                # the draw function
                def draw():
                    gpu.state.point_size_set(point_size)
                    bgl.glEnable(bgl.GL_DEPTH_TEST)
                    batch.draw(shader)
                    bgl.glDisable(bgl.GL_DEPTH_TEST)
                    
                #Define draw handler to acces the drawn point cloud later on
                draw_handler = bpy.types.SpaceView3D.draw_handler_add(draw, (), 'WINDOW', 'POST_VIEW')
                #Store the draw handler reference in the driver namespace
                bpy.app.driver_namespace['my_draw_handler'] = draw_handler
                
                #Calculate the bounding box of the point cloud
                min_coords = np.min(self.point_coords, axis=0)
                max_coords = np.max(self.point_coords, axis=0)
                bbox_center = (min_coords + max_coords) / 2
                
                #Get the active 3D view
                for area in bpy.context.screen.areas:
                    if area.type == 'VIEW_3D':
                        break
                        
                #Set the view to look at the bounding box center from above at a certain height
                view3d = area.spaces[0]
                camera_height=50
                view3d.region_3d.view_location = (bbox_center[0], bbox_center[1], camera_height)  #X, Y, z meters height
                #view3d.region_3d.view_rotation = bpy.context.scene.camera.rotation_euler  #Maintaining the current rotation
                view3d.region_3d.view_distance = camera_height  #Distance from the view point
                print("openGL point cloud drawn in:",time.time() - start_time,"using ",sparsity_value*100," percent of points (",len(reduced_points),") points") 
                
            else:
                print("Draw handler already exists, skipping drawing")
        except Exception as e:
            #Handle any other exceptions that might occur
            print(f"An error occurred: {e}")     
                                
#Function to Check whether the mouseclick happened in the viewport or elsewhere    
def is_mouse_in_3d_view(context, event):
    
    #Identify the 3D Viewport area and its regions
    view_3d_area = next((area for area in context.screen.areas if area.type == 'VIEW_3D'), None)
    if view_3d_area is not None:
        toolbar_region = next((region for region in view_3d_area.regions if region.type == 'TOOLS'), None)
        ui_region = next((region for region in view_3d_area.regions if region.type == 'UI'), None)
        view_3d_window_region = next((region for region in view_3d_area.regions if region.type == 'WINDOW'), None)

        #Check if the mouse is inside the 3D Viewport's window region
        if view_3d_window_region is not None:
            mouse_inside_view3d = (
                view_3d_window_region.x < event.mouse_x < view_3d_window_region.x + view_3d_window_region.width and 
                view_3d_window_region.y < event.mouse_y < view_3d_window_region.y + view_3d_window_region.height
            )
            
            #Exclude areas occupied by the toolbar or UI regions
            if toolbar_region is not None:
                mouse_inside_view3d &= not (
                    toolbar_region.x < event.mouse_x < toolbar_region.x + toolbar_region.width and 
                    toolbar_region.y < event.mouse_y < toolbar_region.y + toolbar_region.height
                )
            if ui_region is not None:
                mouse_inside_view3d &= not (
                    ui_region.x < event.mouse_x < ui_region.x + ui_region.width and 
                    ui_region.y < event.mouse_y < ui_region.y + ui_region.height
                )
            
            return mouse_inside_view3d

    return False  #Default to False if checks fail.        

#function to get 3D click point
def get_click_point_in_3d(context, event):
    # Convert the mouse position to 3D space
    coord_3d = view3d_utils.region_2d_to_location_3d(
        context.region, context.space_data.region_3d,
        (event.mouse_region_x, event.mouse_region_y),
        Vector((0, 0, 0))
    )
    return coord_3d

#Function to load KDTree from a file
def load_kdtree_from_file(file_path):
    if os.path.exists(file_path):
        print("Existing kdtree found. Loading...")
        start_time = time.time()
        with open(file_path, 'r') as file:
            kdtree_data = json.load(file)
        #Convert the loaded points back to a Numpy array
        points = np.array(kdtree_data['points'])
        print("Loaded kdtree in: %s seconds" % (time.time() - start_time),"from: ",file_path)
        return cKDTree(points)
    else:
        return None

#Function to save KDTree to a file
def save_kdtree_to_file(file_path, kdtree):
    kdtree_data = {
        'points': kdtree.data.tolist()  #Convert Numpy array to Python list
    }
    with open(file_path, 'w') as file:
        json.dump(kdtree_data, file)
        
        if bpy.context.object:
            bpy.ops.object.select_all(action='DESELECT')
            bpy.context.view_layer.objects.active = bpy.context.object
            bpy.context.object.select_set(True)
            bpy.ops.object.delete()
    print("saved kdtree to",file_path)

#Function to store obj state
def prepare_object_for_export(obj):
    
    # Check if the necessary context attributes are available
    if 'selected_objects' not in dir(bpy.context) or 'view_layer' not in dir(bpy.context):
        print("Required context attributes are not available.")
        return None

    # check to ensure the object is valid
    if obj is None:
        print("Invalid object for export.")
        return None
    
    try:
        bpy.context.view_layer.objects.active = obj  #Set as active object
        bpy.ops.object.mode_set(mode='OBJECT') #force object mode
        bpy.ops.object.select_all(action='DESELECT')  #Deselect all objects
        obj.select_set(True)  #Select the current object

        set_origin_to_geometry_center(obj)
        
        save_shape_checkbox = bpy.context.scene.save_shape
        if(save_shape_checkbox):
            save_shape_as_image(obj)
            
        save_obj_checkbox = bpy.context.scene.save_obj
        if(save_obj_checkbox):
            export_shape_as_obj(obj, obj.name)
            
    except Exception as e:
        print(f"Error during preparation for export: {e}")
        return None
        
#Function to export objects as OBJ files 
def export_shape_as_obj(obj, name):
    
    #global properties from context
    marking_color = bpy.context.scene.marking_color 
    transparency = bpy.context.scene.marking_transparency
    shape_material = bpy.data.materials.new(name="shape_material")
    shape_material.diffuse_color = (marking_color[0], marking_color[1], marking_color[2], transparency)

    #Assign the material to the object
    if len(obj.data.materials) > 0:
        obj.data.materials[0] = shape_material
    else:
        obj.data.materials.append(shape_material)
        
    #Get the path of the current Blender file
    blend_file_path = bpy.data.filepath
    directory = os.path.dirname(blend_file_path)

    #Create the 'shape objects' directory if it doesn't exist
    shapes_dir = os.path.join(directory, "shape objects")
    if not os.path.exists(shapes_dir):
        os.makedirs(shapes_dir)

    #Define the path for the OBJ file
    obj_file_path = os.path.join(shapes_dir, f"{name}.obj")

    #Select the object
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    #Export the object to an OBJ file
    bpy.ops.export_scene.obj(filepath=obj_file_path, use_selection=True,use_materials=True)
    
    save_obj_for_webviewer=True
    if(save_obj_for_webviewer):
        shapes_dir = os.path.join(directory, "webbased poc/objects")
        obj_file_path = os.path.join(shapes_dir, "shape.obj")
        bpy.ops.export_scene.obj(filepath=obj_file_path, use_selection=True,use_materials=True)
        
    obj.select_set(False)
    
#Function to set origin to geometry center based on an object
def set_origin_to_geometry_center(obj):
    if obj.type == 'MESH':
        #For mesh objects, use the built-in function
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    else:
        #For non-mesh objects, calculate the bounding box center manually
        local_bbox_center = sum((0.125 * Vector(v) for v in obj.bound_box), Vector((0, 0, 0)))
        global_bbox_center = obj.matrix_world @ local_bbox_center

        #Move the object so that the bounding box center is at the world origin
        obj.location = obj.location - global_bbox_center

        #Adjust object's mesh data to new origin
        if hasattr(obj.data, "transform"):
            obj.data.transform(Matrix.Translation(global_bbox_center))
            if hasattr(obj.data, "update"):
                obj.data.update()

#Function to force top view in the viewport
def set_view_to_top(context):
    
    #Find the first 3D View
    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            #Get the 3D View
            space_data = area.spaces.active

            #Get the current rotation in Euler angles
            current_euler = space_data.region_3d.view_rotation.to_euler()

            #Set the Z-axis rotation to 0, retaining X and Y rotations
            new_euler = mathutils.Euler((math.radians(0), 0, current_euler.z), 'XYZ')
            space_data.region_3d.view_rotation = new_euler.to_quaternion()
            
            #Update the view
            area.tag_redraw()
            break
   
#Function to clears the viewport and delete the draw handler
def redraw_viewport():
    
    #global draw_handler  #Reference the global variable
    draw_handler = bpy.app.driver_namespace.get('my_draw_handler')
    
    if draw_handler is not None:
        #Remove the handler reference, stopping the draw calls
        bpy.types.SpaceView3D.draw_handler_remove(draw_handler, 'WINDOW')
        #draw_handler = None
        del bpy.app.driver_namespace['my_draw_handler']

        print("Draw handler removed successfully.")
        print("Stopped drawing the point cloud.")

    #Redraw the 3D view to reflect the removal of the point cloud
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()        
                
    print("viewport redrawn")
    
#Function to check if a mouseclick is on a white object
def is_click_on_white(self, context, location, neighbors=6):
    pointcloud_data = GetPointCloudData()
    point_colors = pointcloud_data.point_colors
    points_kdtree=  pointcloud_data.points_kdtree
    intensity_threshold = context.scene.intensity_threshold

    #Define the number of nearest neighbors to search for
    num_neighbors = neighbors
    
    #Use the k-d tree to find the nearest points to the click location
    _, nearest_indices = points_kdtree.query([location], k=num_neighbors)
    
    average_intensity=get_average_intensity(nearest_indices,point_colors)

    print(average_intensity)

    #If the average intensity is above the threshold, return True (click is on a "white" object)
    if average_intensity > intensity_threshold:
        return True
    else:
        print("Intensity threshold not met")
        return False

#Function to export the point cloud as a shapefile   
def export_as_shapefile(points,points_percentage=100,epsg_value=28992):
    
    global point_cloud_name
    start_time=time.time()
    num_points=len(points)
    num_points_to_keep = math.ceil(num_points * (points_percentage/100))
    step = math.ceil(num_points / num_points_to_keep)
    points = points[::step]
    
    print("exporting as shapefile using ",points_percentage," percent of points: ","(",len(points)," points)")
    point_geometries = [Point(x, y, z) for x, y, z in points]
    crs = 'EPSG:' + str(epsg_value)
    gdf = gpd.GeoDataFrame(geometry=point_geometries, crs=crs)
    print("exported as a shapefile in: ",time.time()-start_time," Saving to file...")
    
    #Get the directory of the current Blender file
    blend_file_path = bpy.data.filepath
    directory = os.path.dirname(blend_file_path)

    #Create a folder 'road_mark_images' if it doesn't exist
    shapefile_dir = os.path.join(directory, 'shapefiles')
    if not os.path.exists(shapefile_dir):
        os.makedirs(shapefile_dir)
    #Define the path for the output shapefile
    output_shapefile_path = os.path.join(shapefile_dir, f"{point_cloud_name}_shapefile")
    gdf.to_file(output_shapefile_path)
    print("saved shapefile to: ",shapefile_dir," in: ",time.time()-start_time)

#Function to export point cloud as JSON    
def export_as_json(point_coords,point_colors,JSON_data_path,point_cloud_name):
    start_time = time.time()
    print("exporting point cloud data as JSON")
    #Adjusting the structure to match the expected format
    point_cloud_data = [
        {
            'x': round(float(point[0]), 2), 
            'y': round(float(point[1]), 2), 
            'z': round(float(point[2]), 2), 
            'color': {'r': int(color[0]), 'g': int(color[1]), 'b': int(color[2])}
        } for point, color in zip(point_coords, point_colors)
    ]

    #Save as compact JSON to reduce file size
    json_data = json.dumps(point_cloud_data, separators=(',', ':')).encode('utf-8')

    #Defines file paths
    json_file_path = os.path.join(JSON_data_path, f"{point_cloud_name}_points_colors.json.gz")

    #Write to JSON file
    print("Compressing JSON...")
    with gzip.open(json_file_path, 'wb') as f:
        f.write(json_data)

    print("Combined JSON file compressed and saved at: ", JSON_data_path, "in: ", time.time() - start_time, "seconds")
             
#Function to install libraries from a list using pip
def install_libraries(library_list):
    for library in library_list:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', library])
            print(f"Successfully installed {library}")
        except subprocess.CalledProcessError as e:
            print(f"Error installing {library}: {e}")
#Function to update libraries from a list using pip            
def update_libraries(library_list):
    for library in library_list:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install','--upgrade', library])
            print(f"Successfully updated {library}")
        except subprocess.CalledProcessError as e:
            print(f"Error updating {library}: {e}")
#Function to uninstall libraries from a list using pip            
def uninstall_libraries(library_list):
    for library in library_list:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', library])
            print(f"Successfully uninstall {library}")
        except subprocess.CalledProcessError as e:
            print(f"Error uninstall {library}: {e}")   
                               
#module imports
from .math_utils import get_average_intensity
from .shape_recognition_utils import save_shape_as_image