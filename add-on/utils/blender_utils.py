#library imports
import sys
import os
import bpy
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
use_pickled_kdtree=True
save_json=False
point_cloud_name="Point cloud"
point_cloud_point_size=1

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
    def pointcloud_load_optimized(self,path, point_size, sparsity_value,points_percentage,z_height_cut_off):
        
        start_time = time.time()
        print("Started loading point cloud.."),
        global use_pickled_kdtree,point_cloud_name,point_cloud_point_size, save_json

        base_file_name = os.path.basename(path)
        point_cloud_name = base_file_name
        directory_path = os.path.dirname(path)
        blend_file_path = bpy.data.filepath
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
        
        if not os.path.exists(os.path.join(stored_data_path, file_name_points)):
            point_cloud = lp.read(path)
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
            #Creates a cube at the centroid location
            #bpy.ops.mesh.primitive_cube_add(size=2, enter_editmode=False, align='WORLD', location=points_a_avg)
            points_a = points_a - points_a_avg
            #print(points_a[:5]) #Print the first 5 points
            
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
        
        if sparsity_value == 1 and points_percentage<100: 
            #Calculate sparsity value based on the desired percentage
            desired_sparsity = int(1 / (points_percentage / 100))

            #Evenly sample points based on the calculated sparsity
            reduced_indices = range(0, len(points_a), desired_sparsity)
            reduced_points = points_a[reduced_indices]
            reduced_colors = colors_a[reduced_indices]
        else:
            #Evenly sample points using the provided sparsity value
            reduced_points = points_a[::sparsity_value]
            reduced_colors = colors_a[::sparsity_value]
            
        #Save json file of point cloud data
        if save_json:
            export_as_json(reduced_points,reduced_colors,JSON_data_path,point_cloud_name,points_percentage)
  
        #Function to save KD-tree with pickle and gzip
        def save_kdtree_pickle_gzip(file_path, kdtree):
            with gzip.open(file_path, 'wb', compresslevel=1) as f:  # compresslevel from 1-9, low-high compression
                pickle.dump(kdtree, f)
        #Function to load KD-tree with pickle and gzip
        def load_kdtree_pickle_gzip(file_path):
            with gzip.open(file_path, 'rb') as f:
                return pickle.load(f)  

        if use_pickled_kdtree:
            #KDTree handling
            kdtree_pickle_path = os.path.join(stored_data_path, file_name_kdtree_pickle)
            if not os.path.exists(kdtree_pickle_path):
                #Create the kdtree if it doesn't exist
                print("creating cKDTree..")
                self.points_kdtree = cKDTree(np.array(self.point_coords))
                save_kdtree_pickle_gzip(kdtree_pickle_path, self.points_kdtree)
                print("Compressed cKD-tree saved at:", kdtree_pickle_path)  
            else:
                print("kdtree found, loading..")
                self.points_kdtree = load_kdtree_pickle_gzip(kdtree_pickle_path)
                print("Compressed cKD-tree loaded from gzip file in:", time.time() - start_time)
        else:  
            #KDTree handling
            kdtree_path = os.path.join(stored_data_path, file_name_kdtree)
            self.points_kdtree = load_kdtree_from_file(kdtree_path)
            if not os.path.exists(kdtree_pickle_path):
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
                
                print("openGL point cloud drawn in:",time.time() - start_time,"using ",points_percentage," percent of points (",len(reduced_points),") points") 
                
            else:
                print("Draw handler already exists, skipping drawing")
        except Exception as e:
            #Handle any other exceptions that might occur
            print(f"An error occurred: {e}")     
                                
#Checks whether the mouseclick happened in the viewport or elsewhere    
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

def store_object_state(obj):

    #bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.objects.active = obj  #Set as active object
    bpy.ops.object.select_all(action='DESELECT')  #Deselect all objects
    obj.select_set(True)  #Select the current object

    set_origin_to_geometry_center(obj)
    save_shape_checkbox = bpy.context.scene.save_shape
    if(save_shape_checkbox):
        save_shape_as_image(obj)
    #Storing object state
    obj_state = {
        'name': obj.name,
        'location': obj.location.copy(),
        'rotation': obj.rotation_euler.copy(),
        'scale': obj.scale.copy(),
        'mesh': obj.data.copy() 
    }
    

#Set origin to geometry center based on object type   
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

#Clears the viewport and deletes the draw handler
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
    
#function to check if a mouseclick is on a white object
def is_click_on_white(self, context, location, points_kdtree,point_colors,neighbors=5):
    
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

#exports point cloud as JSON    
def export_as_json(point_coords,point_colors,JSON_data_path,point_cloud_name,points_percentage):
    start_time = time.time()
    print("exporting point cloud data as JSON with",points_percentage, "percent of points")
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
             
def dv_points_reader(filepath):
    data = pd.read_csv(filepath)
    x_coord = np.asarray(data['x'])
    y_coord = np.asarray(data['y'])
    z_coord = np.asarray(data['z'])
    indices = np.asarray(data['index'])

    points_coord = np.vstack((x_coord, y_coord, z_coord)).T

    points_coord_avg = np.mean(points_coord, axis=0)
    points_coord = points_coord - points_coord_avg

    return points_coord, indices, points_coord_avg

def dv_shapefile_reader(filepath, index, avg_shift, coord_z):

    gdf = gpd.read_file(filepath)
    gdf["Index_"] = gdf["Index_"].astype(int)

    selected_geometries = gdf[gdf['Index_'] == index]['geometry']

    line_coordinates = list(selected_geometries[index].coords)

    first_point = line_coordinates[0]
    second_point = line_coordinates[1]
    line_start = np.array([first_point[0] - avg_shift[0], first_point[1] - avg_shift[1], coord_z])
    line_end = np.array([second_point[0] - avg_shift[0], second_point[1] - avg_shift[1], coord_z])

    return line_start, line_end 

#installs libraries from a list using pip
def install_libraries(library_list):
    for library in library_list:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', library])
            print(f"Successfully installed {library}")
        except subprocess.CalledProcessError as e:
            print(f"Error installing {library}: {e}")
            
def update_libraries(library_list):
    for library in library_list:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install','--upgrade', library])
            print(f"Successfully updated {library}")
        except subprocess.CalledProcessError as e:
            print(f"Error updating {library}: {e}")
            
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