# list of libraries to install
library_list = [
    'numpy',
    'open3d',
    'laspy[laszip]',
    'scipy',
    'mathutils',
    'pandas',
    'geopandas',
    'shapely',
    'scikit-learn',
    'joblib',
    'opencv-python'
]

#imports
import sys
print(sys.executable)
print(sys.version)
import subprocess
import os 

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

#install_libraries()    
#update_libraries() 
#uninstall_libraries()  

#imports
import bpy
import gpu
import bmesh
from bpy.props import StringProperty, IntProperty, FloatProperty, BoolProperty
from gpu_extras.batch import batch_for_shader
from bpy_extras import view3d_utils
import numpy as np
import open3d as o3d
import bgl
import blf
import time
import math
import laspy as lp
import scipy
import shapely
from scipy.spatial import KDTree
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from mathutils import Vector, Matrix
import mathutils
import pickle
import gzip
import pandas as pd
import geopandas as gpd
from bl_ui.space_toolsystem_common import ToolSelectPanelHelper
import copy
from bpy import context
from bpy_extras.view3d_utils import region_2d_to_location_3d
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree
from collections import deque
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from joblib import dump, load 
import json
import cv2
from sklearn.cluster import DBSCAN 
from scipy.interpolate import UnivariateSpline, make_interp_spline,CubicSpline
from shapely.geometry import Point

#Global variables 
point_coords = None # The coordinates of the point cloud, stored as a numpy array
point_colors = None # The colors of the point cloud, stored as a numpy array
original_coords = None # The original coords of the point cloud, before shifting
points_kdtree = None #The loaded ckdtree of coords that all functions can access
collection_name = "Collection" #the default collection name in blender
point_cloud_name= None # Used when storing files related to current point cloud
point_cloud_point_size =  1 # The size of the points in the point cloud
shape_counter=1 #Keeps track of amount of shapes currently in viewport
auto_las_file_path = "C:/Users/Niels/OneDrive/stage hawarIT cloud/point clouds/auto.laz" # Add path here for a laz file name auto.laz
use_pickled_kdtree=True #compress files to save disk space
save_json=False #generate a json file of point cloud data

#Keeps track of all objects created/removed for undo/redo functions
undo_stack = []
redo_stack = []

#Global variable to keep track of the last processed index, for numbering road marks
last_processed_index = 0

# Global variable to keep track of the active operator
active_operator = None

#Function to load the point cloud, store it's data and draw it using openGL           
def pointcloud_load(path, point_size, sparsity_value):
    
    start_time = time.time()
    
    global point_coords, point_colors, original_coords, points_kdtree, point_cloud_name 
   
    base_file_name = os.path.basename(path)
    point_cloud_name = base_file_name
    directory_path = os.path.dirname(path)
    saved_data_path = os.path.join(directory_path, "Stored Data")
    file_name_points = base_file_name + "_points.npy"
    file_name_colors = base_file_name + "_colors.npy"
    file_name_avg_coords = base_file_name + "_avgCoords.npy"
    file_name_kdtree = base_file_name + "_kdtree.joblib"  #File name for the kdtree

    #Create the folder where all the stored data will exist
    if not os.path.exists(saved_data_path):
        os.mkdir(saved_data_path)
    
    #Check if file is numpy file of the point cloud exist or not
    if not os.path.exists(os.path.join(saved_data_path, file_name_points)):
        point_cloud = lp.read(path)
        points_a = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
        colors_a = np.vstack((point_cloud.red, point_cloud.green, point_cloud.blue)).transpose() / 65535

        ####--------------- Shifting coords ---------------####
        points_a_avg = np.mean(points_a, axis=0)
        points_a = points_a - points_a_avg

        ####--------------- Storing Shifting coords ---------------####
        np.save(os.path.join(saved_data_path, file_name_avg_coords), points_a_avg)

        ####--------------- Storing the arrays as npy file ---------------####
        np.save(os.path.join(saved_data_path, file_name_points), points_a)
        np.save(os.path.join(saved_data_path, file_name_colors), colors_a)

    else:
        points_a = np.load(os.path.join(saved_data_path, file_name_points))
        colors_a = np.load(os.path.join(saved_data_path, file_name_colors))
    
    print("point cloud loaded in: ", time.time() - start_time)
    
    ####--------------- Checking sparsity ---------------####
    point_cloud_display_sparsity = sparsity_value
    #print("sparsity: ", sparsity_value)
    
    if point_cloud_display_sparsity == 1:
        points_ar = points_a
        colors_ar = colors_a
    else:
        points_ar = points_a[:: point_cloud_display_sparsity]
        colors_ar = colors_a[:: point_cloud_display_sparsity]
  
    # Store point data and colors globally
    point_coords = points_ar
    point_colors = colors_ar*255
    original_coords = points_a
    point_cloud_point_size = point_size
    
    # Check if the kdtree file exists
    if not os.path.exists(os.path.join(saved_data_path, file_name_kdtree)):
        #Create the kdtree if it doesn't exist
        points_kdtree = cKDTree(np.array(points_ar))
   
        #Save the kdtree to a file
        dump(points_kdtree, os.path.join(saved_data_path, file_name_kdtree))
    else:
        #Load the kdtree from the file
        points_kdtree = load(os.path.join(saved_data_path, file_name_kdtree))
     
    print("kdtree loaded in: ", time.time() - start_time)
    
    try: 
        draw_handler = bpy.app.driver_namespace.get('my_draw_handler')
        
        if draw_handler is None:

            #Converting to tuple 
            coords = tuple(map(tuple, points_ar))
            colors = tuple(map(tuple, colors_ar))
            
            shader = gpu.shader.from_builtin('3D_FLAT_COLOR')
            batch = batch_for_shader(
                shader, 'POINTS',
                {"pos": coords, "color": colors}
            )
            
            #Make sure the viewport is cleared before rendering
            redraw_viewport()
            
            #Draw the point cloud using opengl
            def draw():
                gpu.state.point_size_set(point_size)
                bgl.glEnable(bgl.GL_DEPTH_TEST)
                batch.draw(shader)
                bgl.glDisable(bgl.GL_DEPTH_TEST)
                
            #Define draw handler to acces the drawn point cloud later on
            draw_handler = bpy.types.SpaceView3D.draw_handler_add(draw, (), 'WINDOW', 'POST_VIEW')
            #Store the draw handler reference in the driver namespace
            bpy.app.driver_namespace['my_draw_handler'] = draw_handler
            
            print("openGL point cloud drawn in:",time.time() - start_time) 
            
        else:
            print("Draw handler already exists, skipping drawing")
    except Exception as e:
        # Handle any other exceptions that might occur
        print(f"An error occurred: {e}")

#Function to load the point cloud, store it's data and draw it using openGL, optimized version
def pointcloud_load_optimized(path, point_size, sparsity_value):
    
    start_time = time.time()
    print("Started loading point cloud.."),
    global point_coords, point_colors, original_coords, points_kdtree,use_pickled_kdtree,point_cloud_name,point_cloud_point_size
    points_percentage=context.scene.points_percentage
    z_height_cut_off=context.scene.z_height_cut_off
    
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
    blend_file_path = bpy.data.filepath
  
    if not os.path.exists(stored_data_path):
        os.mkdir(stored_data_path)
    
    if not os.path.exists(os.path.join(stored_data_path, file_name_points)):
        point_cloud = lp.read(path)
        points_a = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
        colors_a = np.vstack((point_cloud.red, point_cloud.green, point_cloud.blue)).transpose() / 65535
        # Convert points to float32
        points_a = points_a.astype(np.float32)
        # Convert colors to uint8
        colors_a = (colors_a * 255).astype(np.uint8)
        
        # Sort the points based on the Z coordinate
        sorted_points = points_a[points_a[:, 2].argsort()]

        # Determine the cutoff index for the lowest %
        cutoff_index = int(len(sorted_points) * 0.1)

        # Calculate the average Z value of the lowest % of points
        road_base_level = np.mean(sorted_points[:cutoff_index, 2])

        print("Estimated road base level:", road_base_level)
        
        if(z_height_cut_off>0):
            # Filter points with Z coordinate > 0.5
            print("Number of points before filtering:", len(points_a))
            mask = points_a[:, 2] <= (road_base_level+z_height_cut_off)
            points_a = points_a[mask]
            colors_a = colors_a[mask]
            print("Number of points after filtering:", len(points_a))
        
        # Store the original coordinates 
        original_coords = np.copy(points_a)
               
        #Shifting coords
        points_a_avg = np.mean(points_a, axis=0)
        # Create a cube at the centroid location
        #bpy.ops.mesh.primitive_cube_add(size=2, enter_editmode=False, align='WORLD', location=points_a_avg)
        points_a = points_a - points_a_avg
        #print(points_a[:5])
        
        #Storing Shifting coords 
        np.save(os.path.join(stored_data_path, file_name_avg_coords), points_a_avg)
    
        #Storing the centered coordinate arrays as npy file
        np.save(os.path.join(stored_data_path, file_name_points), points_a)
        np.save(os.path.join(stored_data_path, file_name_colors), colors_a)
                
    else:
        points_a = np.load(os.path.join(stored_data_path, file_name_points))
        colors_a = np.load(os.path.join(stored_data_path, file_name_colors))
        original_coords = np.load(os.path.join(stored_data_path, file_name_avg_coords))
        
    # Store point data and colors globally
    point_coords = points_a
    point_colors = colors_a
    point_cloud_point_size = point_size
    
    print("point cloud loaded in: ", time.time() - start_time)
    
    if sparsity_value == 1 and points_percentage<100: 
        # Calculate sparsity value based on the desired percentage
        desired_sparsity = int(1 / (points_percentage / 100))

        # Evenly sample points based on the calculated sparsity
        reduced_indices = range(0, len(points_a), desired_sparsity)
        reduced_points = points_a[reduced_indices]
        reduced_colors = colors_a[reduced_indices]
    else:
        # Evenly sample points using the provided sparsity value
        reduced_points = points_a[::sparsity_value]
        reduced_colors = colors_a[::sparsity_value]
        
    #Save json file of point cloud data
    if save_json:
        save_as_json(reduced_points,reduced_colors,JSON_data_path,point_cloud_name,points_percentage)
        #this function creates more read able json files, but is slower 
        #save_as_json(point_coords,point_colors)

    # Function to save KD-tree with pickle and gzip
    def save_kdtree_pickle_gzip(file_path, kdtree):
        with gzip.open(file_path, 'wb', compresslevel=1) as f:  #  compresslevel from 1-9, low-high compression
            pickle.dump(kdtree, f)
    # Function to load KD-tree with pickle and gzip
    def load_kdtree_pickle_gzip(file_path):
        with gzip.open(file_path, 'rb') as f:
            return pickle.load(f)  

    if(use_pickled_kdtree):
        # KDTree handling
        kdtree_pickle_path = os.path.join(stored_data_path, file_name_kdtree_pickle)
        if points_kdtree is None:
            # Create the kdtree if it doesn't exist
            points_kdtree = cKDTree(np.array(point_coords))
            save_kdtree_pickle_gzip(kdtree_pickle_path, points_kdtree)
            print("Compressed KD-tree saved at:", kdtree_pickle_path)  
        else:
            points_kdtree = load_kdtree_pickle_gzip(kdtree_pickle_path)
            print("Compressed KD-tree loaded from gzip file")
    else:  
        # KDTree handling
        kdtree_path = os.path.join(stored_data_path, file_name_kdtree)
        points_kdtree = load_kdtree_from_file(kdtree_path)
        if points_kdtree is None:
            # Create the kdtree if it doesn't exist
            points_kdtree = cKDTree(np.array(point_coords))
            print("kdtree created in: ", time.time() - start_time)
            # Save the kdtree to a file
            save_kdtree_to_file(kdtree_path, points_kdtree)
            print("kdtree saved in: ", time.time() - start_time, "at", kdtree_path)
         
    try: 
        draw_handler = bpy.app.driver_namespace.get('my_draw_handler')
        
        if draw_handler is None:
            # colors_ar should be in uint8 format
            reduced_colors = reduced_colors / 255.0  # Normalize to 0-1 range
            #Converting to tuple 
            coords = tuple(map(tuple, reduced_points))
            colors = tuple(map(tuple, reduced_colors))
            
            shader = gpu.shader.from_builtin('3D_FLAT_COLOR')
            batch = batch_for_shader(
                shader, 'POINTS',
                {"pos": coords, "color": colors}
            )
            
            # Inside the draw function
            def draw():
                gpu.state.point_size_set(point_size)
                bgl.glEnable(bgl.GL_DEPTH_TEST)
                batch.draw(shader)
                bgl.glDisable(bgl.GL_DEPTH_TEST)
                
            #Define draw handler to acces the drawn point cloud later on
            draw_handler = bpy.types.SpaceView3D.draw_handler_add(draw, (), 'WINDOW', 'POST_VIEW')
            #Store the draw handler reference in the driver namespace
            bpy.app.driver_namespace['my_draw_handler'] = draw_handler
            
            print("openGL point cloud drawn in:",time.time() - start_time,"using ",points_percentage," percent of points, ",len(reduced_points)," points") 
            
        else:
            print("Draw handler already exists, skipping drawing")
    except Exception as e:
        # Handle any other exceptions that might occur
        print(f"An error occurred: {e}")     
                             
class CreatePointCloudObjectOperator(bpy.types.Operator):
    
    bl_idname = "custom.create_point_cloud_object"
    bl_label = "Create point cloud object"
    
    global point_coords, point_colors, point_cloud_point_size, collection_name
    
    def create_point_cloud_object(self, points_ar, colors_ar, point_size, collection_name):
    
      #Create a new mesh
      mesh = bpy.data.meshes.new("Point Cloud Mesh")

      #Link the mesh to the object
      obj = bpy.data.objects.new("Point Cloud Object", mesh)

      #Link the object to the specified collection
      collection = bpy.data.collections.get(collection_name)
      if collection:
        collection.objects.link(obj)

      #Link the mesh to the scene
      bpy.context.scene.collection.objects.link(obj)

      #Set the mesh vertices
      mesh.from_pydata(points_ar, [], [])

      #Create a new material for the object
      material = bpy.data.materials.new("Point Cloud Material")
      material.use_nodes = True  # Enable material nodes

      #Clear default nodes
      material.node_tree.nodes.clear()

      #Create an emission shader node
      emission_node = material.node_tree.nodes.new(type='ShaderNodeEmission')
      emission_node.location = (0, 0)

      #Create a Shader to RGB node to convert vertex colors to shader input
      shader_to_rgb = material.node_tree.nodes.new(type='ShaderNodeShaderToRGB')
      shader_to_rgb.location = (200, 0)

      #Connect the emission shader to the shader and the RGB node
      material.node_tree.links.new(emission_node.outputs['Emission'], shader_to_rgb.inputs['Shader'])

      #Connect the shader to RGB node to the material output
      material_output_node = material.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
      material_output_node.location = (400, 0)
      material.node_tree.links.new(shader_to_rgb.outputs['Color'], material_output_node.inputs['Surface'])
 
      emission_node.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)  
      emission_node.inputs['Strength'].default_value = 1.0  

      #Check if the mesh has vertex colors
      if len(colors_ar) == len(points_ar):
        # Give the mesh has loop colors
        if not mesh.vertex_colors.active:
          mesh.vertex_colors.new()

        # Iterate over the loops and assign vertex colors
        color_layer = mesh.vertex_colors.active.data
        for i, loop in enumerate(mesh.loops):
          color = colors_ar[i] if i < len(colors_ar) else (1.0, 1.0, 1.0)
          #color*=255
          color_layer[loop.index].color = color + (1.0,)

      # Assign the material to the mesh
      if mesh.materials:
        mesh.materials[0] = material
      else:
        mesh.materials.append(material)
        
      # After the object is created, store it 
      #store_object_state(obj)
      return obj 
   
    def execute(self, context):
        start_time = time.time()
        self.create_point_cloud_object(point_coords,point_colors, point_cloud_point_size, collection_name)
        print("--- %s seconds ---" % (time.time() - start_time))
        return {'FINISHED'}
    
#Defines an Operator for drawing a free thick straight line in the viewport using mouseclicks
class DrawStraightFatLineOperator(bpy.types.Operator):
    
    bl_idname = "view3d.line_drawer"
    bl_label = "Draw Straight Line"
    prev_end_point = None

    def modal(self, context, event):
        
        if event.type == 'LEFTMOUSE':
            if event.value == 'RELEASE':
                self.draw_line(context, event)
                return {'RUNNING_MODAL'}
        elif event.type == 'RIGHTMOUSE' or event.type == 'ESC':
            return {'CANCELLED'}

        return {'PASS_THROUGH'}

    def invoke(self, context, event):
       
        self.prev_end_point = None
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}
        #Render rectangles for creating a thick line

    def draw_line(self, context, event):
        
        marking_color = context.scene.marking_color
        width = context.scene.fatline_width
        extra_z_height = context.scene.extra_z_height
        
        view3d = context.space_data
        region = context.region
        region_3d = context.space_data.region_3d

        if self.prev_end_point:
            coord_3d_start = self.prev_end_point
        else:
            coord_3d_start = view3d_utils.region_2d_to_location_3d(region, region_3d, (event.mouse_region_x, event.mouse_region_y), Vector((0, 0, 0)))
            coord_3d_start.z += extra_z_height  # Add to the z dimension to prevent clipping

        coord_3d_end = view3d_utils.region_2d_to_location_3d(region, region_3d, (event.mouse_region_x, event.mouse_region_y), Vector((0, 0, 0)))
        coord_3d_end.z += extra_z_height  

        # Create a new mesh object for the line
        mesh = bpy.data.meshes.new(name="Line Mesh")
        obj = bpy.data.objects.new("Thin Line", mesh)
        
      
        
        # Link it to scene
        bpy.context.collection.objects.link(obj)
        
        # Create mesh from python data
        bm = bmesh.new()

        # Add vertices
        bmesh.ops.create_vert(bm, co=coord_3d_start)
        bmesh.ops.create_vert(bm, co=coord_3d_end)

        # Add an edge between the vertices
        bm.edges.new(bm.verts)

        # Update and free bmesh to improve memory performance
        bm.to_mesh(mesh)
        bm.free()

        # Create a material for the line and set its color
        material = bpy.data.materials.new(name="Line Material")
        material.diffuse_color = marking_color
        obj.data.materials.append(material)

        self.prev_end_point = coord_3d_end
          # After the object is created, store it 
        store_object_state(obj)
        # Create a rectangle object on top of the line
        create_rectangle_line_object(coord_3d_start, coord_3d_end)
        

    def cancel(self, context):
        if context.object:
            bpy.ops.object.select_all(action='DESELECT')
            context.view_layer.objects.active = context.object
            context.object.select_set(True)
            bpy.ops.object.delete()    

#Prints the point cloud coordinates and the average color & intensity around mouse click        
class GetPointsInfoOperator(bpy.types.Operator):
    bl_idname = "view3d.select_points"
    bl_label = "Get Points information"

    def modal(self, context, event):
        
        global point_coords, point_colors
        
        if event.type == 'MOUSEMOVE':  
            self.mouse_inside_view3d = is_mouse_in_3d_view(context, event)


        if event.type == 'LEFTMOUSE' and event.value == 'PRESS'and self.mouse_inside_view3d:
            if context.area and context.area.type == 'VIEW_3D':
                # Get the mouse coordinates
                x, y = event.mouse_region_x, event.mouse_region_y
                # Convert 2D mouse coordinates to 3D view coordinates
                view3d = context.space_data
                region = context.region
                region_3d = context.space_data.region_3d
                location = region_2d_to_location_3d(region, region_3d, (x, y), (0, 0, 0))

                # Get the z coordinate from 3d space
                z = location.z

                # Perform nearest-neighbor search
                radius=5
                _, nearest_indices = points_kdtree.query([location], k=radius)
                nearest_colors = [point_colors[i] for i in nearest_indices[0]]

                average_intensity = get_average_intensity(nearest_indices[0])
                # Calculate the average color
                average_color = np.mean(nearest_colors, axis=0)
                #average_color*=255
                
                clicked_on_white = "Clicked on roadmark" if is_click_on_white(self, context, location) else "No roadmark detected"
                    
                print("clicked on x,y,z: ",x,y,z,"Average Color:", average_color,"Average intensity: ",average_intensity,clicked_on_white)
            else:
                return {'PASS_THROUGH'}
            
        elif event.type == 'ESC':
            return {'CANCELLED'}  # Stop the operator when ESCAPE is pressed

        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        if context.area.type == 'VIEW_3D':
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            return {'CANCELLED'}

#Draws simple shapes to mark road markings     
class SimpleMarkOperator(bpy.types.Operator):
    bl_idname = "view3d.mark_fast"
    bl_label = "Mark Road Markings fast"

    _is_running = False  # Class variable to check if the operator is already running
    
    def modal(self, context, event):
        global point_coords, point_colors, points_kdtree
        intensity_threshold = context.scene.intensity_threshold
        
        if event.type == 'MOUSEMOVE':  
            self.mouse_inside_view3d = is_mouse_in_3d_view(context, event)
            
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS'and self.mouse_inside_view3d:
            
            start_time = time.time()
            # Get the mouse coordinates
            x, y = event.mouse_region_x, event.mouse_region_y
            # Convert 2D mouse coordinates to 3D view coordinates
            view3d = context.space_data
            region = context.region
            region_3d = context.space_data.region_3d
            location = region_2d_to_location_3d(region, region_3d, (x, y), (0, 0, 0))

            # Get the z coordinate from 3D space
            z = location.z

            # Do a nearest-neighbor search
            num_neighbors = 16  # Number of neighbors 
            radius = 50
            _, nearest_indices = points_kdtree.query([location], k=num_neighbors)
        
            rectangle_coords = []
            
            # Get the average intensity of the nearest points
            average_intensity = get_average_intensity(nearest_indices[0])
           
             # Get the average color of the nearest points
            average_color = get_average_color(nearest_indices[0])
             
            print("average color: ", average_color,"average intensity: " ,average_intensity)
            
            # Check if the average intensity indicates a road marking (white)
            if average_intensity > intensity_threshold:
                # Region growing algorithm
                checked_indices = set()
                indices_to_check = list(nearest_indices[0])
                print("Region growing started")
                while indices_to_check:   
                    current_index = indices_to_check.pop()
                    if current_index not in checked_indices:
                        checked_indices.add(current_index)
                        intensity = np.average(point_colors[current_index]) #* 255  # grayscale
                        if intensity>intensity_threshold:
                            rectangle_coords.append(point_coords[current_index])
                            _, neighbor_indices = points_kdtree.query([point_coords[current_index]], k=radius)
                            indices_to_check.extend(neighbor_index for neighbor_index in neighbor_indices[0] if neighbor_index not in checked_indices)

                print("Region growing completed", time.time()-start_time)
                
            
            else:
                print("no road markings found")
                
            if rectangle_coords:
                # Create a single mesh for the combined  rectangles
                create_shape(rectangle_coords,shape_type="unkown")
                
        
        elif event.type == 'ESC':
            SimpleMarkOperator._is_running = False
            print("Operation was cancelled")
            return {'CANCELLED'}  # Stop when ESCAPE is pressed

        return {'PASS_THROUGH'}

    
    def invoke(self, context, event):
        if SimpleMarkOperator._is_running:
            self.report({'WARNING'}, "Operator is already running")
            return {'CANCELLED'}  # Do not run the operator if it's already running

        if context.area.type == 'VIEW_3D':
            SimpleMarkOperator._is_running = True  # Set the flag to indicate the operator is running
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            return {'CANCELLED'}

    def cancel(self, context):
        SimpleMarkOperator._is_running = False  # Reset the flag when the operator is cancelled
        print("Operator was properly cancelled")  # Debug message
        return {'CANCELLED'}
        
#Draws complexer shaped road markings using many tiny squares, which then get combined          
class ComplexMarkOperator(bpy.types.Operator):
    bl_idname = "view3d.mark_complex"
    bl_label = "Mark complex Road Markings"
    _is_running = False  # Class variable to check if the operator is already running
    
    def modal(self, context, event):
        
        global point_coords, point_colors, points_kdtree
        intensity_threshold = context.scene.intensity_threshold
        if event.type == 'MOUSEMOVE':  
            self.mouse_inside_view3d = is_mouse_in_3d_view(context, event) 
        clicked=None
        
        if not clicked:
            if event.type == 'LEFTMOUSE' and event.value == 'PRESS'and self.mouse_inside_view3d:
                clicked=True
                start_time = time.time()
                # Get the mouse coordinates
                x, y = event.mouse_region_x, event.mouse_region_y
                # Convert 2D mouse coordinates to 3D view coordinates
                view3d = context.space_data
                region = context.region
                region_3d = context.space_data.region_3d
                location = region_2d_to_location_3d(region, region_3d, (x, y), (0, 0, 0))

                # Get the z coordinate from 3D space
                z = location.z

                # Do a nearest-neighbor search
                num_neighbors = 16  # Number of neighbors 
                radius = 100
                _, nearest_indices = points_kdtree.query([location], k=num_neighbors)
            
                rectangle_coords = []
                
                # Get the average intensity of the nearest points
                average_intensity = get_average_intensity(nearest_indices[0])
                           
                 # Get the average color of the nearest points
                average_color = get_average_color(nearest_indices[0])
                    
                print("average color: ", average_color,"average intensity: " ,average_intensity)
                
                # Check if the average intensity indicates a road marking (white)
                if average_intensity > intensity_threshold:
                    # Region growing algorithm
                    checked_indices = set()
                    indices_to_check = list(nearest_indices[0])
                    print("Region growing started")
                    while indices_to_check:   
                        current_index = indices_to_check.pop()
                        if current_index not in checked_indices:
                            checked_indices.add(current_index)
                            intensity = np.average(point_colors[current_index]) #* 255  # grayscale
                            if intensity>intensity_threshold:
                                rectangle_coords.append(point_coords[current_index])
                                _, neighbor_indices = points_kdtree.query([point_coords[current_index]], k=radius)
                                indices_to_check.extend(neighbor_index for neighbor_index in neighbor_indices[0] if neighbor_index not in checked_indices)

                    print("Region growing completed", time.time()-start_time)
                    
                
                else:
                    print("no road markings found")
                clicked=False    
                
                if rectangle_coords:
                    # Create a single mesh for the combined rectangles
                    create_dots_shape(rectangle_coords)
                      
            elif event.type == 'ESC':
                ComplexMarkOperator._is_running = False  # Reset the flag when the operator stops
                print("Operation was cancelled")  
                return {'CANCELLED'}  # Stop when ESCAPE is pressed

            return {'PASS_THROUGH'}

    def invoke(self, context, event):
        if ComplexMarkOperator._is_running:
            self.report({'WARNING'}, "Operator is already running")
            return {'CANCELLED'}  # Do not run the operator if it's already running

        if context.area.type == 'VIEW_3D':
            ComplexMarkOperator._is_running = True  # Set the flag to indicate the operator is running
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            return {'CANCELLED'}

    def cancel(self, context):
        ComplexMarkOperator._is_running = False  # Reset the flag when the operator is cancelled
        print("Operator was properly cancelled")  # Debug message
        return {'CANCELLED'}
        
#Operator to scans the entire point cloud for road markings, then mark them   
class FindALlRoadMarkingsOperator(bpy.types.Operator):
    bl_idname = "custom.find_all_road_marks"
    bl_label = "Finds all road marks"

    def execute(self, context):
        global last_processed_index
        
        markings_threshold = context.scene.markings_threshold
        start_time = time.time()
        print("Start auto detecting up to",markings_threshold, "road markings.. this could take a while")
        
        global point_coords, point_colors, points_kdtree
        intensity_threshold = context.scene.intensity_threshold
       
        point_threshold = 100
        checked_indices = set()
        all_white_object_coords = []
        white_objects_count = 0 
        radius = 100
        n = 1
        max_road_marks = markings_threshold
        
        # Start loop from the last processed index
        for idx, color in enumerate(point_colors[::n][last_processed_index:], start=last_processed_index):
            if white_objects_count >= max_road_marks:
                break
            if idx in checked_indices:
                continue

            intensity = np.average(color) #* 255  
            if intensity > intensity_threshold:
                rectangle_coords = []
                indices_to_check = [idx]
                while indices_to_check:
                    current_index = indices_to_check.pop()
                    if current_index not in checked_indices:
                        checked_indices.add(current_index)
                        intensity = np.average(point_colors[current_index]) #* 255
                        if intensity > intensity_threshold:
                            rectangle_coords.append(point_coords[current_index])
                            _, neighbor_indices = points_kdtree.query([point_coords[current_index]], k=radius)
                            indices_to_check.extend(neighbor_index for neighbor_index in neighbor_indices[0] if neighbor_index not in checked_indices)
                
                # Check point count before adding to draw list
                if len(rectangle_coords) >= point_threshold:
                    all_white_object_coords.append(rectangle_coords)
                    white_objects_count += 1  # Increment counter when valid white object is found

        # Update the last processed index
        last_processed_index = idx + 1
        
        print("finished detecting, found: ", white_objects_count, "road marks in: ", time.time() - start_time)
        start_time = time.time()
        
        for white_object_coords in all_white_object_coords:
            create_dots_shape(white_object_coords)
        
        print("rendered shapes in: ", time.time() - start_time)
        
        return {'FINISHED'}
        
#Run the detection logic only within a selection made by the user with 2 mouseclicks   
class SelectionDetectionOpterator(bpy.types.Operator):
    bl_idname = "view3d.selection_detection"
    bl_label = "Detect White Objects in Region"

    click_count = 0
    region_corners = []

    def modal(self, context, event):
        
        global point_coords, point_colors, points_kdtree 
        if event.type == 'MOUSEMOVE':  
            self.mouse_inside_view3d = is_mouse_in_3d_view(context, event)
            
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS'and self.mouse_inside_view3d:
            # Get the mouse coordinates
            x, y = event.mouse_region_x, event.mouse_region_y
            # Convert 2D mouse coordinates to 3D view coordinates
            view3d = context.space_data
            region = context.region
            region_3d = context.space_data.region_3d
            location = region_2d_to_location_3d(region, region_3d, (x, y), (0, 0, 0))

            # Nearest-neighbor search from the point cloud
            _, closest_indices = points_kdtree.query([location], k=20)
            closest_point = point_coords[closest_indices[0][0]]  # get the closest point

            self.region_corners.append(closest_point)  # store the point cloud coordinate
            self.click_count += 1
            if self.click_count >= 2:
                # Find and visualize white objects within the specified region
                self.find_white_objects_within_region()
                return {'FINISHED'}
            for obj in bpy.context.scene.objects:
                if "BoundingBox" in obj.name:
                    bpy.data.objects.remove(obj)
        elif event.type == 'ESC':
            return {'CANCELLED'} 

        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        if context.area.type == 'VIEW_3D':
            # Reset the variables when the operator is run
            self.click_count = 0
            self.region_corners = []
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            return {'CANCELLED'}
    
    def find_white_objects_within_region(self):
       
        global point_coords, point_colors, points_kdtree,intensity_threshold

        # Define bounding box limits
        min_corner = np.min(self.region_corners, axis=0)
        max_corner = np.max(self.region_corners, axis=0)
        print("rectangle drawn within these 2 points: ",min_corner, max_corner)
        # Create a bounding box for visualization
        self.create_bounding_box(min_corner, max_corner)
        # Filter points based on bounding box
        within_bbox = np.all(np.logical_and(min_corner <= point_coords, point_coords <= max_corner), axis=1)
        filtered_points = point_coords[within_bbox]
        filtered_colors = point_colors[within_bbox]
        filtered_kdtree = cKDTree(filtered_points)
        
        print("Number of points in the bounding box:", len(filtered_points))
        
        # Parameters
        point_threshold = 100
        radius = 100
        max_white_objects = 100
        intensity_threshold=intensity_threshold
        # Intensity calculation
        intensities = np.mean(filtered_colors, axis=1) #* 255  
        checked_indices = set()
        all_white_object_coords = []
        white_objects_count = 0 

        for idx, intensity in enumerate(intensities):
            if white_objects_count >= max_white_objects:
                break
            
            if idx in checked_indices or intensity <= intensity_threshold:
                continue

            # Region growing algorithm
            rectangle_coords = []
            indices_to_check = [idx]
            while indices_to_check:
                current_index = indices_to_check.pop()
                if current_index not in checked_indices:
                    checked_indices.add(current_index)
                    intensity = np.average(filtered_colors[current_index]) #* 255
                    if intensity > intensity_threshold:
                        rectangle_coords.append(filtered_points[current_index])
                        _, neighbor_indices = filtered_kdtree.query([filtered_points[current_index]], k=radius)
                        indices_to_check.extend(neighbor_index for neighbor_index in neighbor_indices[0] if neighbor_index not in checked_indices)

            # Check point count before adding to draw list
            if len(rectangle_coords) >= point_threshold:
                all_white_object_coords.append(rectangle_coords)
                white_objects_count += 1  # Increment counter when valid white object is found
                
        print("road marks found: ", white_objects_count)
        # Visualize detected white objects
        for white_object_coords in all_white_object_coords:
            create_dots_shape(white_object_coords)  
            
    #Creates and draws the selection rectangle in the viewport         
    def create_bounding_box(self, min_corner, max_corner):
        # Create a new mesh
        mesh = bpy.data.meshes.new(name="BoundingBox")
        obj = bpy.data.objects.new("BoundingBox", mesh)

        # Link it to scene
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        # Construct bounding box vertices and edges
        verts = [(min_corner[0], min_corner[1], min_corner[2]),
                 (max_corner[0], min_corner[1], min_corner[2]),
                 (max_corner[0], max_corner[1], min_corner[2]),
                 (min_corner[0], max_corner[1], min_corner[2]),
                 (min_corner[0], min_corner[1], max_corner[2]),
                 (max_corner[0], min_corner[1], max_corner[2]),
                 (max_corner[0], max_corner[1], max_corner[2]),
                 (min_corner[0], max_corner[1], max_corner[2])]

        edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                 (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]

        # Create mesh
        mesh.from_pydata(verts, edges, [])
        mesh.update()
        
        
        # After the object is created, store it 
        store_object_state(obj)
        
        TriangleMarkOperator ._is_running = False  # Reset the flag when the operator is cancelled
        print("Operator was properly cancelled")  # Debug message
        return {'CANCELLED'}

class AutoTriangleMarkOperator(bpy.types.Operator):
    bl_idname = "custom.auto_mark_triangle"
    bl_label = "Auto Mark Triangle"
    
    _is_running = False  # Class variable to check if the operator is already running
    _triangles = []  # List to store the triangle vertices
    _simulated_clicks = 0  # Count of simulated clicks
    _found_triangles = 0   # Count of triangles found
    _processed_indices = set()
                
    def modal(self, context, event):
        global point_coords, point_colors, points_kdtree
        intensity_threshold = context.scene.intensity_threshold
        
        if event.type == 'MOUSEMOVE':  
            self.mouse_inside_view3d = is_mouse_in_3d_view(context, event)
            
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS'and self.mouse_inside_view3d:
            
            start_time = time.time()
            # Get the mouse coordinates
            x, y = event.mouse_region_x, event.mouse_region_y
            # Convert 2D mouse coordinates to 3D view coordinates
            view3d = context.space_data
            region = context.region
            region_3d = context.space_data.region_3d
            location = region_2d_to_location_3d(region, region_3d, (x, y), (0, 0, 0))

            # Get the z coordinate from 3D space
            z = location.z

            # Do a nearest-neighbor search
            num_neighbors = 16  # Number of neighbors 
            radius = 50
            _, nearest_indices = points_kdtree.query([location], k=num_neighbors)
        
            triangle_coords = []
            
            # Get the average intensity of the nearest points
            average_intensity = get_average_intensity(nearest_indices[0])
           
             # Get the average color of the nearest points
            average_color = get_average_color(nearest_indices[0])
             
            print("average color: ", average_color,"average intensity: " ,average_intensity)
            
            # Check if the average intensity indicates a road marking (white)
            if average_intensity > intensity_threshold:
                # Region growing algorithm
                checked_indices = set()
                indices_to_check = list(nearest_indices[0])
                print("Region growing started")
                while indices_to_check:   
                    current_index = indices_to_check.pop()
                    if current_index not in checked_indices:
                        checked_indices.add(current_index)
                        intensity = np.average(point_colors[current_index]) #* 255  # grayscale
                        if intensity>intensity_threshold:
                            triangle_coords.append(point_coords[current_index])
                            _, neighbor_indices = points_kdtree.query([point_coords[current_index]], k=radius)
                            indices_to_check.extend(neighbor_index for neighbor_index in neighbor_indices[0] if neighbor_index not in checked_indices)

                print("Region growing completed", time.time()-start_time)
         
            
            else:
                print("no road markings found")
                

            if triangle_coords:
                #filters out bad points
                filtered_triangle_coords=filter_noise_with_dbscan(triangle_coords)
                self._processed_indices.update(checked_indices)
                triangle_vertices = create_flexible_triangle(filtered_triangle_coords)
                self._triangles.append(triangle_vertices)
                create_shape(filtered_triangle_coords, shape_type="triangle", vertices=triangle_vertices)

                if len(self._triangles) == 2:
                    outer_corners= self.find_outermost_corners(self._triangles[0], self._triangles[1])
                    create_polyline("triangles_base_line",outer_corners)
                    self.perform_automatic_marking(context, intensity_threshold,outer_corners)
                
        elif event.type == 'ESC':
            self.cancel(context)
            return {'CANCELLED'}  # Stop when ESCAPE is pressed
        return {'RUNNING_MODAL'}
    
    @staticmethod
    def find_outermost_corners(triangle1, triangle2):
        max_distance = 0
        outermost_points = (None, None)

        for point1 in triangle1:
            for point2 in triangle2:
                distance = np.linalg.norm(np.array(point1) - np.array(point2))
                if distance > max_distance:
                    max_distance = distance
                    outermost_points = (point1, point2)
        return outermost_points
    
    def perform_automatic_marking(self, context, intensity_threshold,outer_corners):
        if len(self._triangles) >= 2:
            centers = [np.mean(triangle, axis=0) for triangle in self._triangles[:2]]
            middle_points = self.interpolate_line(centers[0], centers[1])
            for point in middle_points:
                self.simulate_click_and_grow(point, context, intensity_threshold, outer_corners)
            #Move triangle down to the base line
            for triangle in self._triangles:
                move_blender_triangle_objects(triangle, outer_corners[0], outer_corners[1])
            print("Moved TriangleS")
                
                
    def simulate_click_and_grow(self, location, context, intensity_threshold, outer_corners):
        global point_coords, point_colors, points_kdtree

        _, nearest_indices = points_kdtree.query([location], k=16)
        average_intensity = get_average_intensity(nearest_indices[0])
        average_color = get_average_color(nearest_indices[0])

        if (average_intensity > intensity_threshold) and not self._processed_indices.intersection(nearest_indices[0]):
            # Proceed only if the intensity is above the threshold and the area hasn't been processed yet
            checked_indices = set()
            indices_to_check = list(nearest_indices[0])

            while indices_to_check:
                current_index = indices_to_check.pop()
                if current_index not in checked_indices:
                    checked_indices.add(current_index)
                    intensity = np.average(point_colors[current_index]) #* 255
                    if intensity > intensity_threshold:
                        _, neighbor_indices = points_kdtree.query([point_coords[current_index]], k=50)
                        indices_to_check.extend(neighbor_index for neighbor_index in neighbor_indices[0] if neighbor_index not in checked_indices)

            if checked_indices:
                points = [point_coords[i] for i in checked_indices]
                filtered_points = filter_noise_with_dbscan(points)
                self._processed_indices.update(checked_indices)
                triangle_vertices = create_flexible_triangle(filtered_points)
                self._triangles.append(triangle_vertices)
                self._found_triangles += 1
                move_triangle_to_line(triangle_vertices, outer_corners[0], outer_corners[1])
                create_shape(filtered_points, shape_type="triangle", vertices=triangle_vertices)
        
    def interpolate_line(self, start, end, num_points=50):
        # Generate points along the line between start and end
        return [start + t * (end - start) for t in np.linspace(0, 1, num_points)]

    def invoke(self, context, event):
        if AutoTriangleMarkOperator ._is_running:
            self.report({'WARNING'}, "Operator is already running")
            self.cancel(context)
            return {'CANCELLED'}

        if context.area.type == 'VIEW_3D':
            #clean up
            AutoTriangleMarkOperator._triangles = []
            AutoTriangleMarkOperator._processed_indices = set()
            AutoTriangleMarkOperator ._is_running = True  # Set the flag to indicate the operator is running
            AutoTriangleMarkOperator._found_triangles = 0
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            return {'CANCELLED'}

    def cancel(self, context):
        AutoTriangleMarkOperator ._is_running = False  # Reset the flag when the operator is cancelled
        print("Operator was properly cancelled")  # Debug message
        return {'CANCELLED'}

class TriangleMarkOperator(bpy.types.Operator):
    bl_idname = "custom.mark_triangle"
    bl_label = "Mark Triangle"
    
    _is_running = False  # Class variable to check if the operator is already running
    _triangles = []  # List to store the triangle vertices
    _processed_indices = set()
    _last_outer_corner = None  # Initialize the last outer corner here   
         
    def modal(self, context, event):
        global point_coords, point_colors, points_kdtree
        intensity_threshold = context.scene.intensity_threshold
        extra_z_height = context.scene.extra_z_height
        if event.type == 'MOUSEMOVE':  
            self.mouse_inside_view3d = is_mouse_in_3d_view(context, event)
            
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS' and self.mouse_inside_view3d:
            # Process the mouse click
            self.process_mouse_click(context, event,intensity_threshold)

        elif event.type == 'ESC':
            self.cancel(context)
            return {'CANCELLED'}  # Stop when ESCAPE is pressed
        
        return {'RUNNING_MODAL'}

    def process_mouse_click(self, context,event, intensity_threshold):
        # Get the mouse coordinates
        x, y = event.mouse_region_x, event.mouse_region_y
        location = region_2d_to_location_3d(context.region, context.space_data.region_3d, (x, y), (0, 0, 0))
        triangle_coords=[]
        # Nearest-neighbor search
        _, nearest_indices = points_kdtree.query([location], k=16)
        average_intensity = get_average_intensity(nearest_indices[0])
        average_color = get_average_color(nearest_indices[0])
        if average_intensity > intensity_threshold:
            # Region growing algorithm
            checked_indices = set()
            indices_to_check = list(nearest_indices[0])
            
            while indices_to_check:   
                current_index = indices_to_check.pop()
                if current_index not in checked_indices:
                    checked_indices.add(current_index)
                    intensity = np.average(point_colors[current_index]) #* 255
                    if intensity > intensity_threshold:
                        triangle_coords.append(point_coords[current_index])
                        _, neighbor_indices = points_kdtree.query([point_coords[current_index]], k=50)
                        indices_to_check.extend(neighbor_index for neighbor_index in neighbor_indices[0] if neighbor_index not in checked_indices)

            if triangle_coords:
  
                #current_triangle_coords=[point_coords[i] for i in checked_indices]
                filtered_current_triangle_coords=filter_noise_with_dbscan(triangle_coords)
                self._processed_indices.update(checked_indices)
                current_triangle_vertices = create_flexible_triangle(filtered_current_triangle_coords)
                self._triangles.append(current_triangle_vertices)
                    
                if len(self._triangles) >= 2:
                    #Check if _last_outer_corner is initialized
                    if self._last_outer_corner is None:
                        outer_corners = self.find_outermost_corners(self._triangles[-2], self._triangles[-1])
                        # Ensure both corners are in the correct format
                        #outer_corners = [list(corner) for corner in outer_corners]
                        self._last_outer_corner = outer_corners[1]
                    else:
                        #Use the last outer corner and find the new one
                        new_outer_corner = self.find_outermost_corner(self._triangles[-1], self._last_outer_corner)
                        outer_corners = [self._last_outer_corner, new_outer_corner]
                        self._last_outer_corner = new_outer_corner

                    #Ensure outer_corners contains two points, each  a list or tuple
                    if all(isinstance(corner, (list, tuple)) for corner in outer_corners):
                        create_polyline("triangles_base_line", outer_corners)
                        current_triangle_vertices = move_triangle_to_line(current_triangle_vertices, outer_corners[0],outer_corners[1])
                        print("Moved Triangle To Line")
                    else:
                        print("Error: outer_corners does not contain valid points")
                        
                # Convert all vertices to lists containing three coordinates
                for vertex in current_triangle_vertices:
                    if not isinstance(vertex, (list, tuple)) or len(vertex) != 3:
                        # Convert vertex to a list
                        vertex = list(vertex)
                        
                new_triangle_coords=[]  
                for vertex in current_triangle_vertices:
                    new_triangle_coords+=[(vertex[0], vertex[1], vertex[2])]
                create_shape(new_triangle_coords, shape_type="triangle",vertices=current_triangle_vertices)

    @staticmethod
    def find_outermost_corners(triangle1, triangle2):
        max_distance = 0
        outermost_points = (None, None)

        for point1 in triangle1:
            for point2 in triangle2:
                distance = np.linalg.norm(np.array(point1) - np.array(point2))
                if distance > max_distance:
                    max_distance = distance
                    outermost_points = (point1, point2)
        return outermost_points
    @staticmethod
    def find_outermost_corner(triangle, reference_point):
        max_distance = 0
        outermost_point = None

        for point in triangle:
            distance = np.linalg.norm(np.array(point) - np.array(reference_point))
            if distance > max_distance:
                max_distance = distance
                outermost_point = point

        return outermost_point

    def cancel(self, context):
        TriangleMarkOperator._is_running = False
        print("Operation was cancelled")
        return {'CANCELLED'}

    def invoke(self, context, event):
        if TriangleMarkOperator._is_running:
            self.report({'WARNING'}, "Operator is already running")
            self.cancel(context)
            return {'CANCELLED'}

        if context.area.type == 'VIEW_3D':
            # Reset the state
            TriangleMarkOperator._triangles = []
            TriangleMarkOperator._processed_indices = set()
            TriangleMarkOperator._last_outer_corner = None
            TriangleMarkOperator._is_running = True
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            return {'CANCELLED'}        
        
class RectangleMarkOperator(bpy.types.Operator):
    bl_idname = "custom.mark_rectangle"
    bl_label = "Mark Rectangle"
    
    _is_running = False  # Class variable to check if the operator is already running
    
    def modal(self, context, event):
        global point_coords, point_colors, points_kdtree
        intensity_threshold = context.scene.intensity_threshold
        
        if event.type == 'MOUSEMOVE':  
            self.mouse_inside_view3d = is_mouse_in_3d_view(context, event)
            
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS'and self.mouse_inside_view3d:
            
            start_time = time.time()
            # Get the mouse coordinates
            x, y = event.mouse_region_x, event.mouse_region_y
            # Convert 2D mouse coordinates to 3D view coordinates
            view3d = context.space_data
            region = context.region
            region_3d = context.space_data.region_3d
            location = region_2d_to_location_3d(region, region_3d, (x, y), (0, 0, 0))

            # Get the z coordinate from 3D space
            z = location.z

            # Do a nearest-neighbor search
            num_neighbors = 16  # Number of neighbors 
            radius = 50
            _, nearest_indices = points_kdtree.query([location], k=num_neighbors)
        
            rectangle_coords = []
            
            # Get the average intensity of the nearest points
            average_intensity = get_average_intensity(nearest_indices[0])
           
             # Get the average color of the nearest points
            average_color = get_average_color(nearest_indices[0])
             
            print("average color: ", average_color,"average intensity: " ,average_intensity)
            
            # Check if the average intensity indicates a road marking (white)
            if average_intensity > intensity_threshold:
                # Region growing algorithm
                checked_indices = set()
                indices_to_check = list(nearest_indices[0])
                print("Region growing started")
                while indices_to_check:   
                    current_index = indices_to_check.pop()
                    if current_index not in checked_indices:
                        checked_indices.add(current_index)
                        intensity = np.average(point_colors[current_index]) #* 255  # grayscale
                        if intensity>intensity_threshold:
                            rectangle_coords.append(point_coords[current_index])
                            _, neighbor_indices = points_kdtree.query([point_coords[current_index]], k=radius)
                            indices_to_check.extend(neighbor_index for neighbor_index in neighbor_indices[0] if neighbor_index not in checked_indices)

                print("Region growing completed", time.time()-start_time)
                
            
            else:
                print("no road markings found")
                
            if rectangle_coords:
                # Create a single mesh for the combined  rectangles
                create_shape(rectangle_coords,shape_type="rectangle")
                
        
        elif event.type == 'ESC':
            SimpleMarkOperator._is_running = False
            print("Operation was cancelled")
            return {'CANCELLED'}  # Stop when ESCAPE is pressed

        return {'PASS_THROUGH'}

    
    def invoke(self, context, event):
        if SimpleMarkOperator._is_running:
            self.report({'WARNING'}, "Operator is already running")
            self.cancel(context)

        if context.area.type == 'VIEW_3D':
            SimpleMarkOperator._is_running = True  # Set the flag to indicate the operator is running
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            return {'CANCELLED'}

    def cancel(self, context):
        SimpleMarkOperator._is_running = False  # Reset the flag when the operator is cancelled
        print("Operator was properly cancelled")  # Debug message
        return {'CANCELLED'}

class AutoRectangleMarkOperator(bpy.types.Operator):
    bl_idname = "custom.auto_mark_rectangle"
    bl_label = "Auto Mark rectangle"
    
    _is_running = False  # Class variable to check if the operator is already running
    _rectangles = []  # List to store the rectangle vertices
    _simulated_clicks = 0  # Count of simulated clicks
    _found_rectangles = 0   # Count of triangles found
    _processed_indices = set()
                
    def modal(self, context, event):
        global point_coords, point_colors, points_kdtree
        intensity_threshold = context.scene.intensity_threshold
        
        if event.type == 'MOUSEMOVE':  
            self.mouse_inside_view3d = is_mouse_in_3d_view(context, event)
            
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS'and self.mouse_inside_view3d:
            
            start_time = time.time()
            # Get the mouse coordinates
            x, y = event.mouse_region_x, event.mouse_region_y
            # Convert 2D mouse coordinates to 3D view coordinates
            view3d = context.space_data
            region = context.region
            region_3d = context.space_data.region_3d
            location = region_2d_to_location_3d(region, region_3d, (x, y), (0, 0, 0))

            # Get the z coordinate from 3D space
            z = location.z

            # Do a nearest-neighbor search
            num_neighbors = 16  # Number of neighbors 
            radius = 50
            _, nearest_indices = points_kdtree.query([location], k=num_neighbors)
        
            rectangle_coords = []
            
            # Get the average intensity of the nearest points
            average_intensity = get_average_intensity(nearest_indices[0])
           
             # Get the average color of the nearest points
            average_color = get_average_color(nearest_indices[0])
             
            print("average color: ", average_color,"average intensity: " ,average_intensity)
            
            # Check if the average intensity indicates a road marking (white)
            if average_intensity > intensity_threshold:
                # Region growing algorithm
                checked_indices = set()
                indices_to_check = list(nearest_indices[0])
                print("Region growing started")
                while indices_to_check:   
                    current_index = indices_to_check.pop()
                    if current_index not in checked_indices:
                        checked_indices.add(current_index)
                        intensity = np.average(point_colors[current_index]) #* 255  # grayscale
                        if intensity>intensity_threshold:
                            rectangle_coords.append(point_coords[current_index])
                            _, neighbor_indices = points_kdtree.query([point_coords[current_index]], k=radius)
                            indices_to_check.extend(neighbor_index for neighbor_index in neighbor_indices[0] if neighbor_index not in checked_indices)

                print("Region growing completed", time.time()-start_time)
         
            
            else:
                print("no road markings found")
                

            if rectangle_coords:
                #filters out bad points
                #filtered_rectangle_coords=filter_noise_with_dbscan(rectangle_coords)
                self._processed_indices.update(checked_indices)
                rectangle_vertices = create_flexible_rectangle(rectangle_coords)
                self._rectangles.append(rectangle_vertices)
                create_shape(rectangle_coords, shape_type="rectangle")

                if len(self._rectangles) == 2:
                    #center_points= self.find_center_points(self._rectangles[0], self._rectangles[1])
                    self.perform_automatic_marking(context, intensity_threshold)
                
        elif event.type == 'ESC':
            self.cancel(context)
            return {'CANCELLED'}  # Stop when ESCAPE is pressed
        
        return {'RUNNING_MODAL'}
    
    @staticmethod
    def find_center_points(rectangle1, rectangle2):
        max_distance = 0
        center_points = (None, None)

        for point1 in rectangle1:
            for point2 in rectangle2:
                distance = np.linalg.norm(np.array(point1) - np.array(point2))
                if distance > max_distance:
                    max_distance = distance
                    center_points = (point1, point2)
        return center_points
    
    def perform_automatic_marking(self, context, intensity_threshold):
        print("2 rectangles found, starting automatic marking..")
        if len(self._rectangles) >= 2:
            centers = [np.mean(rectangle, axis=0) for rectangle in self._rectangles[:2]]
            middle_points = self.interpolate_line(centers[0], centers[1])
            for point in middle_points:
                mark_point(point,"ZebraCrossing",size=0.1)
                self.simulate_click_and_grow(point, context, intensity_threshold)            
                
    def simulate_click_and_grow(self, location, context, intensity_threshold):
        global point_coords, point_colors, points_kdtree

        _, nearest_indices = points_kdtree.query([location], k=16)
        average_intensity = get_average_intensity(nearest_indices[0])
        average_color = get_average_color(nearest_indices[0])

        if (average_intensity > intensity_threshold) and not self._processed_indices.intersection(nearest_indices[0]):
            # Proceed only if the intensity is above the threshold and the area hasn't been processed yet
            checked_indices = set()
            indices_to_check = list(nearest_indices[0])

            while indices_to_check:
                current_index = indices_to_check.pop()
                if current_index not in checked_indices:
                    checked_indices.add(current_index)
                    intensity = np.average(point_colors[current_index]) #* 255
                    if intensity > intensity_threshold:
                        _, neighbor_indices = points_kdtree.query([point_coords[current_index]], k=50)
                        indices_to_check.extend(neighbor_index for neighbor_index in neighbor_indices[0] if neighbor_index not in checked_indices)

            if checked_indices:
                rectangle_points = [point_coords[i] for i in checked_indices]
                #filtered_points = filter_noise_with_dbscan(points)
                self._processed_indices.update(checked_indices)
                rectangle_vertices = create_flexible_rectangle(rectangle_points)
                self._rectangles.append(rectangle_vertices)
                self._found_rectangles += 1
                create_shape(rectangle_points, shape_type="rectangle")
        
    def interpolate_line(self, start, end, num_points=20):
        # Generate points along the line between start and end
        return [start + t * (end - start) for t in np.linspace(0, 1, num_points)]


    
    def invoke(self, context, event):
        if AutoRectangleMarkOperator ._is_running:
            self.report({'WARNING'}, "Operator is already running")
            self.cancel(context)

        if context.area.type == 'VIEW_3D':
            AutoRectangleMarkOperator ._is_running = True  # Set the flag to indicate the operator is running
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            return {'CANCELLED'}

    def cancel(self, context):
        AutoRectangleMarkOperator ._is_running = False  # Reset the flag when the operator is cancelled
        print("Operator was properly cancelled")  # Debug message
        return {'CANCELLED'}
        
class CurvedLineMarkOperator(bpy.types.Operator):
    bl_idname = "custom.mark_curved_line"
    bl_label = "Mark curved line"
    
    prev_end_point = None
    _is_running = False  
    
    def modal(self, context, event):
        

        if event.type == 'LEFTMOUSE' and is_mouse_in_3d_view(context, event):
            if event.value == 'RELEASE':
                draw_line(self, context, event)
                return {'RUNNING_MODAL'}
        elif event.type == 'RIGHTMOUSE' or event.type == 'ESC':
            return {'CANCELLED'}

        return {'PASS_THROUGH'}

    def invoke(self, context, event):

        if CurvedLineMarkOperator._is_running:
            self.report({'WARNING'}, "Operator is already running")
            self.cancel(context)
            return {'CANCELLED'}
        else:
            self.prev_end_point = None
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        
        if context.area.type == 'VIEW_3D':
            CurvedLineMarkOperator._is_running = True
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            return {'CANCELLED'}

 
    def cancel(self, context):
        
        '''if context.object:
            bpy.ops.object.select_all(action='DESELECT')
            context.view_layer.objects.active = context.object
            context.object.select_set(True)
            bpy.ops.object.delete() '''   
            
        CurvedLineMarkOperator._is_running = False
        print("Operator was properly cancelled")
        return {'CANCELLED'}
    
def draw_line(self, context, event):

    if not hasattr(self, 'click_counter'):
        self.click_counter = 0

    marking_color = context.scene.marking_color
    width = context.scene.fatline_width
    intensity_threshold = context.scene.intensity_threshold
    extra_z_height = context.scene.extra_z_height
    #user_input=context.scene.user_input_result
    
    view3d = context.space_data
    region = context.region
    region_3d = context.space_data.region_3d
    
    wrong_click=""
    
    #Convert the mouse position to a 3D location for the end point of the line
    coord_3d_end = view3d_utils.region_2d_to_location_3d(region, region_3d, (event.mouse_region_x, event.mouse_region_y), Vector((0, 0, 0)))
    coord_3d_end.z += extra_z_height  # Add to the z dimension to prevent clipping

    #Check if the current click is on a white road mark
    on_white_end = is_click_on_white(self, context, coord_3d_end)
    self.click_counter += 1
    print(f"Mouseclick {self.click_counter} is {'on' if on_white_end else 'not on'} a white road mark.")

    if self.prev_end_point:
        #Use the previous end point as the start point for the new line segment
        coord_3d_start = self.prev_end_point

        #Check if the previous click was on a white road mark
        on_white_start = is_click_on_white(self, context, coord_3d_start)

        #Determine and print the outcome
        if on_white_start and on_white_end:
            print(f"Mouseclick {self.click_counter - 1} and Mouseclick {self.click_counter}are both on a white road mark.") 
            wrong_click="none" 
            coord_3d_start, coord_3d_end = snap_to_road_mark(self, context, coord_3d_start,coord_3d_end, click_to_correct=wrong_click)   
            create_rectangle_line_object(coord_3d_start, coord_3d_end)
            
        elif on_white_start:
            print(f"Mouseclick {self.click_counter - 1} is on a white road mark,Mouseclick {self.click_counter} is not.")
            wrong_click="second"
            bpy.ops.wm.correction_pop_up('INVOKE_DEFAULT', start_point=coord_3d_start, end_point=coord_3d_end, click_to_correct=wrong_click)
                
        elif on_white_end:
            print(f"Mouseclick {self.click_counter - 1} is not on a white road mark, Mouseclick {self.click_counter} is on.")
            wrong_click="first"
            bpy.ops.wm.correction_pop_up('INVOKE_DEFAULT', start_point=coord_3d_start, end_point=coord_3d_end, click_to_correct=wrong_click)

    #Update the previous end point to be the current one for the next click
    self.prev_end_point = coord_3d_end
    
def snap_to_road_mark(self, context, first_click_point, last_click_point, click_to_correct,region_radius=2):
    
    intensity_threshold = context.scene.intensity_threshold
    
    global point_coords, point_colors, points_kdtree        

    #Get the direction vector between the two clicks and its perpendicular
    direction = (last_click_point - first_click_point).normalized()
    perp_direction = direction.cross(Vector((0, 0, 1))).normalized()

    #Find the index of the last click point in the point cloud
    _, idx = points_kdtree.query([last_click_point], k=1)
         
    def region_grow(start_point, radius, threshold):
        checked_indices = set()
        indices_to_check = [start_point]
        region_points = []
        #visualize_search_radius(last_click_point, radius)
        while indices_to_check:
            current_index = indices_to_check.pop()
            if current_index not in checked_indices:
                checked_indices.add(current_index)
                point_intensity = np.average(point_colors[current_index]) #* 255
                if point_intensity > threshold:
                    region_points.append(point_coords[current_index])
                    _, neighbor_indices = points_kdtree.query([point_coords[current_index]], k=radius)
                    indices_to_check.extend(neighbor_index for neighbor_index in neighbor_indices[0] if neighbor_index not in checked_indices)
        return region_points
    
    def find_outward_points(region_points, direction):
        #Project all points to the direction vector and find the most outward points
        projections = [np.dot(point, direction) for point in region_points]
        min_proj_index = np.argmin(projections)
        max_proj_index = np.argmax(projections)
        return region_points[min_proj_index], region_points[max_proj_index]
        
    def snap_last_point(_first_click_point, _last_click_point):
        
        #Perform region growing on the last click point
        region = region_grow(idx[0], region_radius, intensity_threshold)
        if region:
            edge1, edge2 = find_outward_points(region, perp_direction)

            #Calculate the new click point based on the edges
            _last_click_point = (edge1 + edge2) * 0.5
            _last_click_point = Vector((_last_click_point[0], _last_click_point[1], _last_click_point[2]))
        else:
            print("No points found to project.")
        mark_point(_first_click_point,"_first_click_point",0.02)
        mark_point(_last_click_point,"_last_click_point",0.02)
        return _first_click_point, _last_click_point
    
    def snap_first_point(_first_click_point, _last_click_point):
        
        # Perform region growing on the last click point
        region = region_grow(idx[0], region_radius, intensity_threshold)
        if region:
            edge1, edge2 = find_outward_points(region, perp_direction)
            mark_point(edge1,"edge1",0.02)
            mark_point(edge2,"edge2",0.02)
            # Calculate the new click point based on the edges
            _first_click_point = (edge1 + edge2) * 0.5
            _first_click_point = Vector((_first_click_point[0], _first_click_point[1], _first_click_point[2]))
        else:
            print("No points found to project.")
            
        mark_point(_first_click_point,"_first_click_point",0.02)
        mark_point(_last_click_point,"_last_click_point",0.02)
        return _first_click_point, _last_click_point

    if(click_to_correct=="none"): 
        new_first_click_point, new_last_click_point = snap_last_point(first_click_point,last_click_point)
        return new_first_click_point, new_last_click_point    
    elif(click_to_correct=="first"):
        new_first_click_point, new_last_click_point = snap_first_point(first_click_point,last_click_point)
        print("first point corrected")
        return new_first_click_point, new_last_click_point   
    elif(click_to_correct=="second"):
        new_first_click_point, new_last_click_point = snap_last_point(first_click_point,last_click_point)
        print("second point corrected")
        return new_first_click_point, new_last_click_point    
   
   
   

# Custom operator for the pop-up dialog
class CorrectionPopUpOperator(bpy.types.Operator):
    bl_idname = "wm.correction_pop_up"
    bl_label = "Confirm correction pop up"

    start_point: bpy.props.FloatVectorProperty()
    end_point: bpy.props.FloatVectorProperty()
    click_to_correct: bpy.props.StringProperty()
    
    action: bpy.props.EnumProperty(
        items=[
            ('DRAW', "Draw Line", "Draw the line anyway"),
            ('CORRECT', "Correct Line", "Try to correct the line"),
            ('CANCEL', "Cancel", "Cancel the drawing")
        ],
        default='DRAW',
    )
    # Define the custom draw method
    def draw(self, context):
        layout = self.layout
        col = layout.column()
        
        # Add custom buttons to the UI
        col.label(text=f"{self.click_to_correct} Click(s) might be incorrect")
        col.label(text="Choose an action:")
        col.separator()
        
        # Use 'props_enum' to create buttons for each enum property
        layout.props_enum(self, "action")
        
    def execute(self, context):
        # Access the stored data to perform the correction
        coord_3d_start = Vector(self.start_point)
        coord_3d_end = Vector(self.end_point)
        click_to_correct = self.click_to_correct
        
        #print("User chose to", "draw" if self.action == 'DRAW' else "correct")
        # Based on the user's choice, either draw or initiate a correction process
        context.scene.user_input_result = self.action
       
        if self.action == 'CORRECT':
            coord_3d_start, coord_3d_end = snap_to_road_mark(self,context, coord_3d_start, coord_3d_end, click_to_correct)
            create_rectangle_line_object(coord_3d_start, coord_3d_end)
        
        elif self.action == ('CANCEL'):
            print("Canceled line drawing")
            return {'CANCELLED'}
        
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)
            
class AutoCurvedLineOperator(bpy.types.Operator):
    bl_idname = "custom.auto_curved_line"
    bl_label = "Mark curved line" 

    
    def modal(self, context, event):
        global point_coords, point_colors, points_kdtree
        intensity_threshold = context.scene.intensity_threshold
        
        if event.type == 'MOUSEMOVE':  
            self.mouse_inside_view3d = is_mouse_in_3d_view(context, event)
            
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS'and self.mouse_inside_view3d:
            
            start_time = time.time()
            # Get the mouse coordinates
            x, y = event.mouse_region_x, event.mouse_region_y
            # Convert 2D mouse coordinates to 3D view coordinates
            view3d = context.space_data
            region = context.region
            region_3d = context.space_data.region_3d
            location = region_2d_to_location_3d(region, region_3d, (x, y), (0, 0, 0))

            # Get the z coordinate from 3D space
            z = location.z

            # Do a nearest-neighbor search
            num_neighbors = 16  # Number of neighbors 
            radius = 50
            _, nearest_indices = points_kdtree.query([location], k=num_neighbors)
        
            rectangle_coords = []
            
            # Get the average intensity of the nearest points
            average_intensity = get_average_intensity(nearest_indices[0])
           
             # Get the average color of the nearest points
            average_color = get_average_color(nearest_indices[0])
             
            print("average color: ", average_color,"average intensity: " ,average_intensity)
            
            # Check if the average intensity indicates a road marking (white)
            if average_intensity > intensity_threshold:
                # Region growing algorithm
                checked_indices = set()
                indices_to_check = list(nearest_indices[0])
                print("Region growing started")
                while indices_to_check:   
                    current_index = indices_to_check.pop()
                    if current_index not in checked_indices:
                        checked_indices.add(current_index)
                        intensity = np.average(point_colors[current_index]) #* 255  # grayscale
                        if intensity>intensity_threshold:
                            rectangle_coords.append(point_coords[current_index])
                            _, neighbor_indices = points_kdtree.query([point_coords[current_index]], k=radius)
                            indices_to_check.extend(neighbor_index for neighbor_index in neighbor_indices[0] if neighbor_index not in checked_indices)

                print("Region growing completed", time.time()-start_time)
                
            else:
                print("no road markings found")
                
            if rectangle_coords:
                # Create a single mesh for the combined  rectangles
                create_shape(rectangle_coords,shape_type="curved line")
                
        elif event.type == 'ESC':
            SimpleMarkOperator._is_running = False
            print("Operation was cancelled")
            return {'CANCELLED'}  # Stop when ESCAPE is pressed

        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        if SimpleMarkOperator._is_running:
            self.report({'WARNING'}, "Operator is already running")
            self.cancel(context)
            return {'CANCELLED'}

        if context.area.type == 'VIEW_3D':
            SimpleMarkOperator._is_running = True  # Set the flag to indicate the operator is running
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            return {'CANCELLED'}

    def cancel(self, context):
        SimpleMarkOperator._is_running = False  # Reset the flag when the operator is cancelled
        print("Operator was properly cancelled")  # Debug message
        return {'CANCELLED'}

class FixedTriangleMarkOperator(bpy.types.Operator):
    bl_idname = "custom.mark_fixed_triangle"
    bl_label = "mark a fixed triangle"

    def modal(self, context, event):
        
        global point_coords, point_colors
        
        if event.type == 'MOUSEMOVE':  
            self.mouse_inside_view3d = is_mouse_in_3d_view(context, event)


        if event.type == 'LEFTMOUSE' and event.value == 'PRESS'and self.mouse_inside_view3d:
            if context.area and context.area.type == 'VIEW_3D':
                # Get the mouse coordinates
                x, y = event.mouse_region_x, event.mouse_region_y
                # Convert 2D mouse coordinates to 3D view coordinates
                view3d = context.space_data
                region = context.region
                region_3d = context.space_data.region_3d
                location = region_2d_to_location_3d(region, region_3d, (x, y), (0, 0, 0))

                # Get the z coordinate from 3d space
                z = location.z
                draw_fixed_triangle(context, location, size=0.5)
          
            else:
                return {'PASS_THROUGH'}
            
        elif event.type == 'ESC':
            return {'CANCELLED'}  # Stop the operator when ESCAPE is pressed

        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        if context.area.type == 'VIEW_3D':
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            return {'CANCELLED'}
        
class FixedRectangleMarkOperator(bpy.types.Operator):
    bl_idname = "custom.mark_fixed_rectangle"
    bl_label = "mark a fixed rectangle"

    def modal(self, context, event):
        
        global point_coords, point_colors
        
        if event.type == 'MOUSEMOVE':  
            self.mouse_inside_view3d = is_mouse_in_3d_view(context, event)


        if event.type == 'LEFTMOUSE' and event.value == 'PRESS'and self.mouse_inside_view3d:
            if context.area and context.area.type == 'VIEW_3D':
                # Get the mouse coordinates
                x, y = event.mouse_region_x, event.mouse_region_y
                # Convert 2D mouse coordinates to 3D view coordinates
                view3d = context.space_data
                region = context.region
                region_3d = context.space_data.region_3d
                location = region_2d_to_location_3d(region, region_3d, (x, y), (0, 0, 0))

                # Get the z coordinate from 3d space
                z = location.z
                create_fixed_square(context, location, size=0.5)
          
            else:
                return {'PASS_THROUGH'}
            
        elif event.type == 'ESC':
            return {'CANCELLED'}  # Stop the operator when ESCAPE is pressed

        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        if context.area.type == 'VIEW_3D':
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            return {'CANCELLED'}  
        
def get_average_intensity(indices):
    # If indices is a NumPy array with more than one dimension, flatten it
    if isinstance(indices, np.ndarray) and indices.ndim > 1:
        indices = indices.flatten()

    # If indices is a scalar, convert it to a list with a single element
    if np.isscalar(indices):
        indices = [indices]

    total_intensity = 0.0
    point_amount = len(indices)
    for index in indices:
        
        intensity = np.average(point_colors[index]) #* 255  
        total_intensity += intensity

    return total_intensity / point_amount

def get_average_color(indices):
    point_amount=len(indices)
    average_color = np.zeros(3, dtype=float)
    for index in indices:
        color = point_colors[index] #* 255  # rgb
        average_color += color
    average_color /= point_amount
    return average_color

def create_flexible_triangle(coords):
    
    #filter bad points
    #coords=filter_noise_with_dbscan(coords)
    # Convert coords to numpy array for efficient operations
    coords_np = np.array(coords)
    
    #calculate the pairwise distances
    pairwise_distances = np.linalg.norm(coords_np[:, np.newaxis] - coords_np, axis=2)
    
    #find the two points that are the furthest apart
    max_dist_indices = np.unravel_index(np.argmax(pairwise_distances), pairwise_distances.shape)
    vertex1 = coords_np[max_dist_indices[0]]
    vertex2 = coords_np[max_dist_indices[1]]
    
    # for each point, compute its distance to the line formed by vertex1 and vertex2
    line_vector = vertex2 - vertex1
    line_vector /= np.linalg.norm(line_vector)  # normalize
    max_distance = 0
    third_vertex = None
    for point in coords_np:
        diff = point - vertex1
        proj = np.dot(diff, line_vector) * line_vector
        distance_to_line = np.linalg.norm(diff - proj)
        if distance_to_line > max_distance:
            max_distance = distance_to_line
            third_vertex = point

    return [vertex1.tolist(), vertex2.tolist(), third_vertex.tolist()]

def create_flexible_rectangle(coords):
    
    hull = ConvexHull(coords)
    vertices = np.array([coords[v] for v in hull.vertices])
    centroid = np.mean(vertices, axis=0)
    north = max(vertices, key=lambda p: p[1])
    south = min(vertices, key=lambda p: p[1])
    east = max(vertices, key=lambda p: p[0])
    west = min(vertices, key=lambda p: p[0])
    return [north, east, south, west]

def draw_fixed_triangle(context, location, size=1.0):
    
    extra_z_height = context.scene.extra_z_height
    # Create new mesh and object
    mesh = bpy.data.meshes.new('FixedTriangle')
    obj = bpy.data.objects.new('Fixed Triangle', mesh)

    # Link object to scene
    bpy.context.collection.objects.link(obj)
    
    # Set object location
    obj.location = (location.x, location.y, extra_z_height)

    # Create mesh data
    bm = bmesh.new()

    # Add vertices
    bm.verts.new((0, 0, 0))  # First vertex at the click location
    bm.verts.new((size, 0, 0))  # Second vertex size units along the x-axis
    bm.verts.new((size / 2, size * (3 ** 0.5) / 2, 0))  # Third vertex to form an equilateral triangle

    # Create a face
    bm.faces.new(bm.verts)

    # Write the bmesh back to the mesh
    bm.to_mesh(mesh)
    bm.free()

    # Add a material to the object
    mat = bpy.data.materials.new(name="TriangleMaterial")
    mat.diffuse_color = (1, 0, 0, 1)  # Red color with full opacity
    obj.data.materials.append(mat)   
    
def create_fixed_triangle(coords, side_length=0.5):
     # Convert coords to numpy array for efficient operations
    coords_np = np.array(coords)

    # Reference vertex (first vertex)
    vertex1 = coords_np[0]

    # Normal vector of the plane defined by the original triangle
    normal_vector = np.cross(coords_np[1] - vertex1, coords_np[2] - vertex1)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)  # Normalize the normal vector

    # Direction vector for the second vertex
    dir_vector = coords_np[1] - vertex1
    dir_vector = dir_vector / np.linalg.norm(dir_vector) * side_length

    # Calculate the position of the second vertex
    vertex2 = vertex1 + dir_vector

    # Direction vector for the third vertex
    # Use cross product to find a perpendicular vector in the plane
    perp_vector = np.cross(normal_vector, dir_vector)
    perp_vector = perp_vector / np.linalg.norm(perp_vector) * side_length

    # Angle for equilateral triangle (60 degrees)
    angle_rad = np.deg2rad(60)

    # Calculate the position of the third vertex
    vertex3 = vertex1 + np.cos(angle_rad) * dir_vector + np.sin(angle_rad) * perp_vector

    return [vertex1.tolist(), vertex2.tolist(), vertex3.tolist()]
def create_fixed_square(context, location, size=1.0):
    # Create new mesh and object
    mesh = bpy.data.meshes.new('FixedSquare')
    obj = bpy.data.objects.new('Fixed Square', mesh)

    # Link object to scene
    bpy.context.collection.objects.link(obj)
    
    # Set object location
    obj.location = location

    # Create mesh data
    bm = bmesh.new()

    # Add vertices for a square
    half_size = size / 2
    v1 = bm.verts.new((half_size, half_size, 0))  # Top Right
    v2 = bm.verts.new((-half_size, half_size, 0))  # Top Left
    v3 = bm.verts.new((-half_size, -half_size, 0))  # Bottom Left
    v4 = bm.verts.new((half_size, -half_size, 0))  # Bottom Right

    # Ensure lookup table is updated before we access vertices by index
    bm.verts.ensure_lookup_table()

    # Create a face
    bm.faces.new((v1, v2, v3, v4))

    # Write the bmesh back to the mesh
    bm.to_mesh(mesh)
    bm.free()

    # Add a material to the object
    mat = bpy.data.materials.new(name="SquareMaterial")
    mat.diffuse_color = (1, 0, 0, 1)  # Red color with full opacity
    obj.data.materials.append(mat)
      
def create_polyline(name, points, width=0.01, color=(1, 0, 0, 1)):
    # Create a new curve data object
    curve_data = bpy.data.curves.new(name, type='CURVE')
    curve_data.dimensions = '3D'

    # Create a new spline in the curve
    polyline = curve_data.splines.new('POLY')
    polyline.points.add(len(points) - 1)  # The new spline has no points by default, add them

    # Assign the points to the spline
    for i, point in enumerate(points):
        polyline.points[i].co = (*point, 1)

    # Create a new object with the curve
    curve_obj = bpy.data.objects.new(name, curve_data)
    bpy.context.collection.objects.link(curve_obj)

    # Set up the curve bevel for width
    curve_data.bevel_depth = width
    curve_data.bevel_resolution = 0

    # Create a new material with the given color
    mat = bpy.data.materials.new(name + "_Mat")
    mat.diffuse_color = color
    curve_obj.data.materials.append(mat)
    store_object_state(curve_obj)
    return curve_obj

#Define a function to create a single mesh for combined rectangles
def create_shape(coords_list, shape_type,vertices=None):
    
    start_time = time.time()
    marking_color = bpy.context.scene.marking_color 
    transparency = bpy.context.scene.marking_transparency
    line_width = context.scene.fatline_width
    shape_coords = None  # Default to original coordinates
    coords_list=filter_noise_with_dbscan(coords_list)
    
    if shape_type == "triangle":
        #flexible_coords = create_flexible_triangle(coords_list)
        vertices=create_fixed_triangle(vertices)
        obj=create_mesh_with_material(
            "Triangle Shape", vertices,
            marking_color, transparency)
        create_triangle_outline(vertices)
        
    elif shape_type == "flexible triangle":
        vertices = create_flexible_triangle(coords_list)
        obj=create_mesh_with_material(
            "flexible triangle", vertices,
            marking_color, transparency)
        create_triangle_outline(vertices)
        print("Drawing flexible triangle")
             
    elif shape_type == "rectangle":
        print("Drawing rectangle")
        shape_coords = create_flexible_rectangle(coords_list)
        #shape_coords=create_fixed_rectangle_old(shape_coords)
        obj=create_mesh_with_material(
            "rectangle Shape", shape_coords,
            marking_color, transparency)
        
    elif shape_type == "curved line":
        print("Drawing curved line")
        middle_points = create_middle_points(coords_list)

        fixed_length_points, total_length, segments = create_fixed_length_segments(middle_points)
        print(f"Total line length: {total_length:.2f} meters")
        print(f"Segmented lines drawn: {segments}")

        obj=create_polyline("Poly Line", middle_points, width=line_width, color=(marking_color[0], marking_color[1], marking_color[2], transparency))
   
    else:
        print("Drawing unkown Shape")
        obj=create_mesh_with_material(
            "Unkown Shape", coords_list,
            marking_color, transparency)
        

    store_object_state(obj)
    print(f"Rendered {shape_type} shape in: {time.time() - start_time:.2f} seconds")
  
def create_mesh_with_material(obj_name, shape_coords, marking_color, transparency):
    
    extra_z_height = context.scene.extra_z_height
    shape_coords = [(x, y, z + extra_z_height) for x, y, z in shape_coords]
        
    mesh = bpy.data.meshes.new(obj_name + "_mesh")
    obj = bpy.data.objects.new(obj_name, mesh)
    bpy.context.collection.objects.link(obj)

    bm = bmesh.new()
    for coords in shape_coords:
        bm.verts.new(coords)
    bmesh.ops.convex_hull(bm, input=bm.verts)
    bm.to_mesh(mesh)
    bm.free()

    # Create a new material for the object
    mat = bpy.data.materials.new(name=obj_name + "_material")
    mat.diffuse_color = (marking_color[0], marking_color[1], marking_color[2], transparency)
    mat.use_nodes = True
    mat.blend_method = 'BLEND'

    principled_node = next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    principled_node.inputs['Alpha'].default_value = transparency

    obj.data.materials.append(mat)
    return obj

#Function to create a colored, resizable line object on top of the line      
def create_rectangle_line_object(start, end):
    
    context = bpy.context
    marking_color = context.scene.marking_color
    transparency = context.scene.marking_transparency
    extra_z_height = context.scene.extra_z_height
    width = context.scene.fatline_width
    # Calculate the direction vector and its length
    direction = end - start
    length = direction.length

    direction.normalize()

    # Calculate the rectangle's width
    orthogonal = direction.cross(Vector((0, 0, 1)))
    orthogonal.normalize()
    orthogonal *= width / 2

    # Calculate the rectangle's vertices with an increase in the z-axis by extra_z_height
    v1 = start + orthogonal + Vector((0, 0, extra_z_height))
    v2 = start - orthogonal + Vector((0, 0, extra_z_height))
    v3 = end - orthogonal + Vector((0, 0, extra_z_height))
    v4 = end + orthogonal + Vector((0, 0, extra_z_height))

    # Create a new mesh object for the rectangle
    mesh = bpy.data.meshes.new(name="Rectangle Mesh")
    obj = bpy.data.objects.new("Rectangle Line", mesh)

    # Link it to the scene
    bpy.context.collection.objects.link(obj)

    # Create mesh from python data
    bm = bmesh.new()

    # Add vertices
    bmesh.ops.create_vert(bm, co=v1)
    bmesh.ops.create_vert(bm, co=v2)
    bmesh.ops.create_vert(bm, co=v3)
    bmesh.ops.create_vert(bm, co=v4)

    # Add faces
    bm.faces.new(bm.verts)

    # Update and free bmesh to reduce memory usage
    bm.to_mesh(mesh)
    bm.free()

    # Create a material for the rectangle and set its color
    material = bpy.data.materials.new(name="Rectangle Material")
    
    # Set the color with alpha for transparency
    material.diffuse_color = (marking_color[0], marking_color[1], marking_color[2], transparency)

    # Adjust the material settings to enable transparency
    material.use_nodes = True
    material.blend_method = 'BLEND'  # Use alpha blend mode

    # Set the Principled BSDF shader's alpha value
    principled_bsdf = next(node for node in material.node_tree.nodes if node.type == 'BSDF_PRINCIPLED')
    principled_bsdf.inputs['Alpha'].default_value = transparency
    
    # Assign the material to the object
    obj.data.materials.append(material)

    # After the object is created, store it 
    store_object_state(obj)

    return obj

# Define a function to create multiple squares on top of detected points, then combines them into one shape
def create_dots_shape(coords_list):
    
    start_time=time.time()
    global shape_counter
    
    marking_color=context.scene.marking_color
    transparency = bpy.context.scene.marking_transparency
    extra_z_height = context.scene.extra_z_height
    
    # Create a new mesh and link it to the scene
    mesh = bpy.data.meshes.new("Combined Shape")
    obj = bpy.data.objects.new("Dots Shape", mesh)
    bpy.context.collection.objects.link(obj)

    bm = bmesh.new()

    square_size = 0.025  # Size of each square
    z_offset = extra_z_height  # Offset in Z coordinate
    max_gap = 10  # Maximum gap size to fill

    #filters out bad points
    coords_list = filter_noise_with_dbscan(coords_list)
    
    # Sort the coordinates by distance
    coords_list.sort(key=lambda coords: (coords[0]**2 + coords[1]**2 + coords[2]**2)**0.5)

    for i in range(len(coords_list)):
        if i > 0:
            # Calculate the distance to the previous point
            gap = ((coords_list[i][0] - coords_list[i-1][0])**2 +
                   (coords_list[i][1] - coords_list[i-1][1])**2 +
                   (coords_list[i][2] - coords_list[i-1][2])**2)**0.5
            if gap > max_gap:
                # If the gap is too large, create a new mesh for the previous group of points
                bm.to_mesh(mesh)
                bm.clear()
                # Update the internal index table of the BMesh
                bm.verts.ensure_lookup_table()

        # Create a square at the current point with an adjusted Z coordinate
        square_verts = [
            bm.verts.new(coords_list[i] + (-square_size / 2, -square_size / 2, z_offset)),
            bm.verts.new(coords_list[i] + (-square_size / 2, square_size / 2, z_offset)),
            bm.verts.new(coords_list[i] + (square_size / 2, square_size / 2, z_offset)),
            bm.verts.new(coords_list[i] + (square_size / 2, -square_size / 2, z_offset)),
        ]

        # Create a face for the square
        bm.faces.new(square_verts)

    # Create a mesh for the last group of points
    bm.to_mesh(mesh)
    bm.free()

    # Create a new material for the combined shape
    shape_material = bpy.data.materials.new(name="shape material")
    shape_material.diffuse_color = (marking_color[0], marking_color[1], marking_color[2], transparency)
    # Enable transparency in the material settings
    shape_material.use_nodes = True
    shape_material.blend_method = 'BLEND'

    # Find the Principled BSDF node and set its alpha value
    principled_node = next(n for n in shape_material.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    principled_node.inputs['Alpha'].default_value = transparency
    
    # Assign the material to the object
    if len(obj.data.materials) > 0:
        # If the object already has materials, replace the first one with the  material
        obj.data.materials[0] = shape_material
    else:
        # add the material to the object
        obj.data.materials.append(shape_material)
        
    obj.color = marking_color  # Set viewport display color 
    shape_counter+=1

    #After the object is created, store it 
    store_object_state(obj)
    
#Checks whether the mouseclick happened in the viewport or elsewhere    
def is_mouse_in_3d_view(context, event):
    
    # Identify the 3D Viewport area and its regions
    view_3d_area = next((area for area in context.screen.areas if area.type == 'VIEW_3D'), None)
    if view_3d_area is not None:
        toolbar_region = next((region for region in view_3d_area.regions if region.type == 'TOOLS'), None)
        ui_region = next((region for region in view_3d_area.regions if region.type == 'UI'), None)
        view_3d_window_region = next((region for region in view_3d_area.regions if region.type == 'WINDOW'), None)

        # Check if the mouse is inside the 3D Viewport's window region
        if view_3d_window_region is not None:
            mouse_inside_view3d = (
                view_3d_window_region.x < event.mouse_x < view_3d_window_region.x + view_3d_window_region.width and 
                view_3d_window_region.y < event.mouse_y < view_3d_window_region.y + view_3d_window_region.height
            )
            
            # Exclude areas occupied by the toolbar or UI regions
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

    return False  # Default to False if checks fail.        
 
def filter_noise_with_dbscan(coords_list, eps=0.04, min_samples=20):
    
    #DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords_list)

    #Create a mask for the points belonging to clusters (excluding noise labeled as -1)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    #Filter the coordinates: keep only those points that are part of a cluster
    filtered_coords = [coords for coords, is_core in zip(coords_list, core_samples_mask) if is_core]

    #Print the filtering results
    original_count = len(coords_list)
    filtered_count = len(filtered_coords)
    removed_count = original_count - filtered_count
    if(removed_count>0 and original_count>3):
        print(f"Points have been filtered. Original amount: {original_count}, Removed: {removed_count}")

    return filtered_coords

def mark_point(point, name="point", size=0.05):
    
    show_dots=context.scene.show_dots
    
    if show_dots:
        # Create a cube to mark the point
        bpy.ops.mesh.primitive_cube_add(size=size, location=point)
        marker = bpy.context.active_object
        marker.name = name
        
        # Create a new material with the specified color
        mat = bpy.data.materials.new(name="MarkerMaterial")
        mat.diffuse_color = (1.0, 0.0, 0.0, 1.0)  # Red color
        mat.use_nodes = False  

        # Assign it to the cube
        if len(marker.data.materials):
            marker.data.materials[0] = mat
        else:
            marker.data.materials.append(mat)

        store_object_state(marker)

def is_click_on_white(self, context, location):
    global points_kdtree, point_colors
    intensity_threshold = context.scene.intensity_threshold

    # Define the number of nearest neighbors to search for
    num_neighbors = 5
    
    # Use the k-d tree to find the nearest points to the click location
    _, nearest_indices = points_kdtree.query([location], k=num_neighbors)
    
    average_intensity=get_average_intensity(nearest_indices)

    print(average_intensity)

    # If the average intensity is above the threshold, return True (click is on a "white" object)
    if average_intensity > intensity_threshold:
        return True
    else:
        print("Intensity threshold not met")
        return False
    
def create_triangle_outline(vertices):
    # Create a new mesh and object for the triangle outline
    mesh = bpy.data.meshes.new(name="TriangleOutline")
    obj = bpy.data.objects.new("Triangle Outline", mesh)

    # Link the object to the scene
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Define edges for the triangle outline
    edges = [(0, 1), (1, 2), (2, 0)]

    # Create the mesh data
    mesh.from_pydata(vertices, edges, [])  # No faces
    mesh.update()

    # Ensure the object scale is applied
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    
    '''wireframe_modifier = obj.modifiers.new(name="Wireframe", type='WIREFRAME')
    wireframe_modifier.thickness = 0.1 # Adjust this value for desired thickness'''
    
    store_object_state(obj)
    
    return obj

def create_middle_points(coords_list, num_segments=10):
    coords_np = np.array(coords_list)

    # Identify the points with extreme x values (leftmost and rightmost)
    leftmost_x = coords_np[:, 0].min()
    rightmost_x = coords_np[:, 0].max()

    leftmost_points = coords_np[coords_np[:, 0] == leftmost_x]
    rightmost_points = coords_np[coords_np[:, 0] == rightmost_x]

    # Identify the top and bottom points among the leftmost and rightmost points
    top_left = leftmost_points[leftmost_points[:, 1].argmax()]
    bottom_left = leftmost_points[leftmost_points[:, 1].argmin()]
    top_right = rightmost_points[rightmost_points[:, 1].argmax()]
    bottom_right = rightmost_points[rightmost_points[:, 1].argmin()]

    # Initialize the middle points list with the leftmost middle point
    middle_points = [(top_left + bottom_left) / 2]

    # Divide the remaining line into segments
    segment_width = (rightmost_x - leftmost_x) / (num_segments - 1)

    for i in range(1, num_segments):
        # Determine the segment boundaries
        x_min = leftmost_x + i * segment_width
        x_max = leftmost_x + (i + 1) * segment_width

        # Filter points in the current segment
        segment_points = coords_np[(coords_np[:, 0] >= x_min) & (coords_np[:, 0] < x_max)]

        if len(segment_points) > 0:
            # Find the top and bottom points in this segment
            top_point = segment_points[segment_points[:, 1].argmax()]
            bottom_point = segment_points[segment_points[:, 1].argmin()]

            # Calculate the middle point
            middle_point = (top_point + bottom_point) / 2
            middle_points.append(middle_point)
            mark_point(middle_point,"middle_point")
    # Add the rightmost middle point at the end
    middle_points.append((top_right + bottom_right) / 2)


    mark_point(top_left,"top_left")
    mark_point(top_right,"top_right")
    mark_point(bottom_left,"bottom_left")
    mark_point(bottom_right,"bottom_right")
    
    return middle_points
                
#Operator to remove all drawn markings from the scene collection
class RemoveAllMarkingsOperator(bpy.types.Operator):
    bl_idname = "custom.remove_all_markings"
    bl_label = "Remove All Lines"

    def execute(self, context):
        global shape_counter
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_by_type(type='CURVE')
        for obj in bpy.context.scene.objects:
            #if "Line" in obj.name or "Combined" in obj.name or "Plane" in obj.name or "BoundingBox" in obj.name:
            bpy.data.objects.remove(obj)
        bpy.ops.object.delete()
        print("All markings cleared")
        shape_counter = 1
        return {'FINISHED'}
    
class RemovePointCloudOperator(bpy.types.Operator):
    """Remove the point cloud."""

    bl_idname = "custom.remove_point_cloud"
    bl_label = "Remove Point Cloud"

    def execute(self, context):
        
        #set the rendering of the openGL point cloud to off
        redraw_viewport()
       
        # Find and remove the object with the name "Point Cloud Object"
        for obj in bpy.context.scene.objects:
            if "Point Cloud"in obj.name:
                bpy.data.objects.remove(obj)
                break
      
        return {'FINISHED'}
 
class LAS_OT_OpenOperator(bpy.types.Operator):
    
    bl_idname = "wm.las_open"
    bl_label = "Open LAS File"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    
    
    def execute(self, context):
        start_time = time.time()
        redraw_viewport()
        bpy.context.scene["Filepath to the loaded pointcloud"] = self.filepath
        sparsity_value = bpy.context.scene.sparsity_value
        point_size = bpy.context.scene.point_size
        pointcloud_load_optimized(self.filepath, point_size, sparsity_value)
        print("Opened LAS file: ", self.filepath,"in %s seconds" % (time.time() - start_time))
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
class LAS_OT_AutoOpenOperator(bpy.types.Operator):
    bl_idname = "wm.las_auto_open"
    bl_label = "Auto Open LAS File"
    def execute(self, context):

        if not os.path.exists(auto_las_file_path):
            print("Error: The file", auto_las_file_path, "does not exist.")
            return {'CANCELLED'}

        sparsity_value = bpy.context.scene.sparsity_value
        point_size = bpy.context.scene.point_size
        print("Opening LAS file:", auto_las_file_path)
        pointcloud_load_optimized(auto_las_file_path, point_size, sparsity_value)
        print("Finished opening LAS file:", auto_las_file_path)
        return {'FINISHED'}

# Function to load KDTree from a file
def load_kdtree_from_file(file_path):
    if os.path.exists(file_path):
        print("Existing kdtree found. Loading...")
        start_time = time.time()
        with open(file_path, 'r') as file:
            kdtree_data = json.load(file)
        # Convert the loaded points back to a Numpy array
        points = np.array(kdtree_data['points'])
        print("Loaded kdtree in: %s seconds" % (time.time() - start_time),"from: ",file_path)
        return cKDTree(points)
    else:
        return None

# Function to save KDTree to a file
def save_kdtree_to_file(file_path, kdtree):
    kdtree_data = {
        'points': kdtree.data.tolist()  # Convert Numpy array to Python list
    }
    with open(file_path, 'w') as file:
        json.dump(kdtree_data, file)
        

        if context.object:
            bpy.ops.object.select_all(action='DESELECT')
            context.view_layer.objects.active = context.object
            context.object.select_set(True)
            bpy.ops.object.delete()
    print("saved kdtree to",file_path)

def store_object_state(obj):
    
    #bpy.context.view_layer.objects.active = obj  # Make the object the active object
    #bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS') # Set the origin to the center of the object's bounding box
 
    #bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.objects.active = obj  # Set as active object
    bpy.ops.object.select_all(action='DESELECT')  # Deselect all objects
    obj.select_set(True)  # Select the current object

    set_origin_to_geometry_center(obj)

    save_shape_as_image(obj)
    # Storing object state
    obj_state = {
        'name': obj.name,
        'location': obj.location.copy(),
        'rotation': obj.rotation_euler.copy(),
        'scale': obj.scale.copy(),
        'mesh': obj.data.copy() 
    }
    
    undo_stack.append(obj_state) 
    # Clear the redo stack
    redo_stack.clear()     

# Set origin to geometry center based on object type   
def set_origin_to_geometry_center(obj):
    if obj.type == 'MESH':
        # For mesh objects, use the built-in function
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    else:
        # For non-mesh objects, calculate the bounding box center manually
        local_bbox_center = sum((0.125 * Vector(v) for v in obj.bound_box), Vector((0, 0, 0)))
        global_bbox_center = obj.matrix_world @ local_bbox_center

        # Move the object so that the bounding box center is at the world origin
        obj.location = obj.location - global_bbox_center

        # Adjust object's mesh data to new origin
        if hasattr(obj.data, "transform"):
            obj.data.transform(Matrix.Translation(global_bbox_center))
            if hasattr(obj.data, "update"):
                obj.data.update()

# Clears the viewport and deletes the draw handler
def redraw_viewport():
    
    #global draw_handler  # Reference the global variable
    draw_handler = bpy.app.driver_namespace.get('my_draw_handler')
    
    if draw_handler is not None:
        # Remove the handler reference, stopping the draw calls
        bpy.types.SpaceView3D.draw_handler_remove(draw_handler, 'WINDOW')
        #draw_handler = None
        del bpy.app.driver_namespace['my_draw_handler']

        print("Draw handler removed successfully.")
        print("Stopped drawing the point cloud.")

    # Redraw the 3D view to reflect the removal of the point cloud
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()        
                
    print("viewport redrawn")

class CenterPointCloudOperator(bpy.types.Operator):
    bl_idname = "custom.center_pointcloud"
    bl_label = "Center the Point Cloud in Viewport"

    def execute(self, context):
       
        global point_coords

        # Calculate the bounding box of the point cloud
        min_coords = np.min(point_coords, axis=0)
        max_coords = np.max(point_coords, axis=0)
        bbox_center = (min_coords + max_coords) / 2

        # Get the active 3D view
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                break
        else:
            self.report({'WARNING'}, "No 3D View found")
            return {'CANCELLED'}

        # Set the view to look at the bounding box center from above at a height of 10 meters
        view3d = area.spaces[0]
        view3d.region_3d.view_location = (bbox_center[0], bbox_center[1], 10)  # X, Y, 10 meters height
        #view3d.region_3d.view_rotation = bpy.context.scene.camera.rotation_euler  # Maintaining the current rotation
        view3d.region_3d.view_distance = 10  # Distance from the view point

        return {'FINISHED'}

class ExportToShapeFileOperator(bpy.types.Operator):
    bl_idname = "custom.export_to_shapefile"
    bl_label = "Export to Shapefile"
    bl_description = "Export the current point cloud to a shapefile"

    def execute(self, context):
        global point_coords
        # Call the function to export the point cloud data to a shapefile
        export_as_shapefile(point_coords,10)

        # Return {'FINISHED'} to indicate that the operation was successful
        return {'FINISHED'}
    
def save_as_json(point_coords,point_colors,JSON_data_path,point_cloud_name,points_percentage):
    start_time = time.time()
    print("exporting point cloud data as JSON with",points_percentage, "percent of points")
    # Convert NumPy float32 values to Python float for JSON serialization
    point_cloud_data = [{'coords': [round(float(coord), 2) for coord in point], 'color': [int(clr) for clr in color]} for point, color in zip(point_coords, point_colors)]

    # Save as compact JSON (without pretty-printing)
    json_data = json.dumps(point_cloud_data, separators=(',', ':')).encode('utf-8')

    # Define file paths
    json_file_path = os.path.join(JSON_data_path, point_cloud_name + "_points_colors.json.gz")

    # Write to JSON file
    print("compressing JSON..")
    with gzip.open(json_file_path, 'wb') as f:
        f.write(json_data)

    print("Combined JSON file compressed and saved at: ", JSON_data_path, "in: ", time.time() - start_time, "seconds")
    
    '''start_time = time.time()
    print("exporting point cloud data as JSON with",points_percentage, "percent of points")
    # Convert NumPy float32 values to Python float for JSON serialization
    point_cloud_data = [{'coords': [round(float(coord), 2) for coord in point], 'color': [int(clr) for clr in color]} for point, color in zip(point_coords, point_colors)]
    # Save as compact JSON (without pretty-printing)
    json_data = json.dumps(point_cloud_data, separators=(',', ':'))
    # Define file paths
    json_file_path = os.path.join(JSON_data_path, point_cloud_name + "_points_colors.json")
    # Write to JSON file
    with open(json_file_path, 'w') as json_file:
        json_file.write(json_data)
    print("Combined JSON file saved at: ", JSON_data_path, "in: ", time.time() - start_time, "seconds")'''

    '''start_time = time.time()
    # Convert NumPy float32 values to Python float for JSON serialization
    point_cloud_data = [{'coords': [float(coord) for coord in point], 'color': [int(clr) for clr in color]} for point, color in zip(point_coords, point_colors)]
    # Save as JSON
    json_data = json.dumps(point_cloud_data)
    # Define file paths
    json_file_path = os.path.join(JSON_data_path, point_cloud_name + "_points_colors.json")
    # Write to JSON file
    with open(json_file_path, 'w') as json_file:
        json_file.write(json_data)
    print("Combined JSON file saved at: ", json_file_path, "in: ", time.time() - start_time, "seconds")'''
                    
class OBJECT_OT_simple_undo(bpy.types.Operator):
    bl_idname = "object.simple_undo"
    bl_label = "Simple Undo"
    
    def execute(self, context):
        if undo_stack:
            obj_state = undo_stack.pop()
            for obj in bpy.context.scene.objects:
                if obj_state['name'] in obj.name:
                    bpy.data.objects.remove(obj)
                    redo_stack.append(obj_state)
        return {'FINISHED'}
        
class OBJECT_OT_simple_redo(bpy.types.Operator):
    bl_idname = "object.simple_redo"
    bl_label = "Simple Redo"
    
    def execute(self, context):
        if redo_stack:
            obj_state = redo_stack.pop()
            undo_stack.append(obj_state)

            mesh = obj_state['mesh']
            obj = bpy.data.objects.new(obj_state['name'], mesh)

            bpy.context.collection.objects.link(obj)

            obj.location = obj_state['location']
            obj.rotation_euler = obj_state['rotation']
            obj.scale = obj_state['scale']

            # Ensure the object is updated in the 3D view
            obj.update_tag()
            
        return {'FINISHED'}
    
# Panel for the Road Marking Digitizer
class DIGITIZE_PT_Panel(bpy.types.Panel):
    bl_label = "Road Marking Digitizer"
    bl_idname = "DIGITIZE_PT_Panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Digitizing Tool'

    def draw(self, context):
        
        layout = self.layout
        scene = context.scene
        
        layout.operator("wm.las_open", text="Import Point Cloud")
        layout.operator("custom.export_to_shapefile", text="export to shapefile")  
        layout.prop(scene, "points_percentage")
        layout.prop(scene, "z_height_cut_off")
        layout.operator("custom.center_pointcloud", text="Center Point Cloud")
        layout.operator("custom.create_point_cloud_object", text="Create Point Cloud object")
        layout.operator("custom.remove_point_cloud", text="Remove point cloud")
        layout.operator("custom.remove_all_markings", text="Remove All markings")
        
        row = layout.row(align=True)
        layout.operator("view3d.select_points", text="Get point color & intensity")
        layout.prop(scene, "intensity_threshold")
        
        row = layout.row()
        layout.prop(scene, "marking_color")
        layout.prop(scene, "marking_transparency")
        layout.prop(scene, "extra_z_height")
        #layout.operator("custom.draw_straight_line", text="Draw Simple Line")
        
        row = layout.row(align=True)
        row.operator("view3d.line_drawer", text="Draw Line")
        row.prop(scene, "fatline_width")
 
        row = layout.row(align=True)
        layout.operator("view3d.mark_fast", text="Simple fill marker")
        layout.operator("view3d.mark_complex", text="Complex fill marker")
        layout.operator("view3d.selection_detection", text="Selection fill Marker")
        row = layout.row()
        row.operator("custom.find_all_road_marks", text="Auto Mark")
        row.prop(scene, "markings_threshold")
        
        layout.operator("custom.mark_fixed_triangle", text="fixed triangle marker")
        layout.operator("custom.mark_fixed_rectangle", text="fixed rectangle marker")
       
        layout.operator("custom.mark_triangle", text="triangle marker")
        layout.operator("custom.auto_mark_triangle", text="auto triangle marker")
        layout.operator("custom.mark_rectangle", text="rectangle marker")
        layout.operator("custom.auto_mark_rectangle", text="auto rectangle marker")
        layout.operator("custom.mark_curved_line", text="curved line marker") 
        layout.operator("custom.auto_curved_line", text="auto curved line")  
        row = layout.row(align=True)
        row.operator("object.simple_undo", text="Undo")
        row.operator("object.simple_redo", text="Redo")
        row = layout.row()
        row.prop(scene, "auto_load")
        row = layout.row()
        row.prop(scene, "save_shape") 
        row = layout.row()
        row.prop(scene, "show_dots")
        row = layout.row()
        
        
         # Dummy space
        for _ in range(5): 
            layout.label(text="")
            
# Register the operators and panel
def register():
    bpy.utils.register_class(LAS_OT_OpenOperator)
    bpy.utils.register_class(LAS_OT_AutoOpenOperator)
    bpy.utils.register_class(CreatePointCloudObjectOperator)
   # bpy.utils.register_class(DrawStraightLineOperator)
    bpy.utils.register_class(DrawStraightFatLineOperator)
    bpy.utils.register_class(RemoveAllMarkingsOperator)
    bpy.utils.register_class(DIGITIZE_PT_Panel)
    bpy.types.Scene.point_size = IntProperty(name="POINT SIZE",
                                      default=1)
    bpy.types.Scene.sparsity_value = IntProperty(name="SPARSITY VALUE",
                                      default=1)
    bpy.utils.register_class(RemovePointCloudOperator)
    bpy.utils.register_class(GetPointsInfoOperator)
    
    bpy.utils.register_class(SimpleMarkOperator)
    bpy.utils.register_class(ComplexMarkOperator)
    bpy.utils.register_class(SelectionDetectionOpterator)
    bpy.utils.register_class(AutoTriangleMarkOperator) 
    bpy.utils.register_class(TriangleMarkOperator) 
    bpy.utils.register_class(FixedTriangleMarkOperator) 
    bpy.utils.register_class(FixedRectangleMarkOperator)
    bpy.utils.register_class(RectangleMarkOperator)
    bpy.utils.register_class(AutoRectangleMarkOperator)
    bpy.utils.register_class(AutoCurvedLineOperator)
    bpy.utils.register_class(CurvedLineMarkOperator)
    bpy.utils.register_class(CorrectionPopUpOperator)
    bpy.utils.register_class(CenterPointCloudOperator)
    bpy.utils.register_class(ExportToShapeFileOperator)

    bpy.types.Scene.intensity_threshold = bpy.props.FloatProperty(
        name="Intensity Threshold",
        description="Minimum intensity threshold",
        default=160,  # Default value
        min=0,  # Minimum value
        subtype='UNSIGNED'  
    )
   
    bpy.types.Scene.markings_threshold = bpy.props.IntProperty(
        name="Max:",
        description="Maximum markings amount",
        default=30,  # Default value
        min=1, # Minimum value
        max=100, # Max value  
        subtype='UNSIGNED' 
    )
    bpy.types.Scene.points_percentage = bpy.props.IntProperty(
        name="Points percentage:",
        description="Percentage of points rendered",
        default=50,  # Default value
        min=1, # Minimum value
        max=100, # Max value  
        subtype='UNSIGNED' 
    )
    bpy.types.Scene.fatline_width = bpy.props.FloatProperty(
        name="Width",
        description="Fat Line Width",
        default=0.15,
        min=0.01, max=10,  #min and max width
        subtype='NONE'  
        
    )
    bpy.types.Scene.marking_color = bpy.props.FloatVectorProperty(
        name="Marking Color",
        subtype='COLOR',
        description="Select Marking color",
        default=(1, 0, 0, 1),  # Default is red
        min=0.0, max=1.0,  # Colors range from 0 to 1
        size=4
        
    )
    bpy.types.Scene.marking_color = bpy.props.FloatVectorProperty(
        name="Marking Color",
        subtype='COLOR',
        description="Select Marking color",
        default=(1, 0, 0, 1),  # Default is red
        min=0.0, max=1.0,  # Colors range from 0 to 1
        size=4
        
    )
    bpy.types.Scene.marking_transparency = bpy.props.FloatProperty(
        name="Marking Transparency",
        description="Set the transparency for the marking (0.0 fully transparent, 1.0 fully opaque)",
        default=0.5,  # Default transparency is 50%
        min=0.0, max=1.0  # Transparency can range from 0.0 to 1.0
    )
    bpy.types.Scene.user_input_result = bpy.props.StringProperty(
    name="User Input Result",
    description="Stores the result from the user input pop-up",
)
    bpy.types.Scene.save_shape = bpy.props.BoolProperty(
        name="Save Shapes",
        description="Toggle saving shapes",
        default=False,
        subtype='UNSIGNED'  
    )
    bpy.types.Scene.auto_load = bpy.props.BoolProperty(
        name="Auto load auto.laz",
        description="Toggle auto loading auto.laz",
        default=False,
        subtype='UNSIGNED'  
    )
    bpy.types.Scene.show_dots = bpy.props.BoolProperty(
        name="Show Dots",
        description="Toggle showing dots",
        default=False,
        subtype='UNSIGNED'  
    )
    bpy.types.Scene.z_height_cut_off = bpy.props.FloatProperty(
        name="max z",
        description="height to cut off z",
        default=0,
        subtype='UNSIGNED'  
    )
    bpy.types.Scene.extra_z_height = bpy.props.FloatProperty(
        name="marking z",
        description="extra height of all objects",
        default=0,
        subtype='UNSIGNED'  
    )
    bpy.utils.register_class(FindALlRoadMarkingsOperator)
    bpy.utils.register_class(OBJECT_OT_simple_undo)
    bpy.utils.register_class(OBJECT_OT_simple_redo)
                                     
def unregister():
    
    bpy.utils.unregister_class(LAS_OT_OpenOperator) 
    bpy.utils.unregister_class(LAS_OT_AutoOpenOperator)
   # bpy.utils.unregister_class(DrawStraightLineOperator)
    bpy.utils.unregister_class(DrawStraightFatLineOperator)
    bpy.utils.unregister_class(RemoveAllMarkingsOperator)
    bpy.utils.unregister_class(DIGITIZE_PT_Panel)
    bpy.utils.unregister_class(RemovePointCloudOperator)
    bpy.utils.unregister_class(GetPointsInfoOperator)
    
    bpy.utils.unregister_class(SimpleMarkOperator)
    bpy.utils.unregister_class(ComplexMarkOperator)
    bpy.utils.unregister_class(SelectionDetectionOpterator)
    bpy.utils.unregister_class(FindALlRoadMarkingsOperator)
    
    bpy.utils.unregister_class(FixedTriangleMarkOperator)
    bpy.utils.unregister_class(FixedRectangleMarkOperator) 
    bpy.utils.unregister_class(TriangleMarkOperator)
    bpy.utils.unregister_class(AutoTriangleMarkOperator)
    bpy.utils.unregister_class(RectangleMarkOperator) 
    bpy.utils.unregister_class(AutoRectangleMarkOperator) 
    bpy.utils.unregister_class(AutoCurvedLineOperator)
    bpy.utils.unregister_class(CurvedLineMarkOperator)
    bpy.utils.unregister_class(CreatePointCloudObjectOperator)
    bpy.utils.unregister_class(CorrectionPopUpOperator)
    bpy.utils.unregister_class(CenterPointCloudOperator)
    bpy.utils.unregister_class(ExportToShapeFileOperator)
    
    del bpy.types.Scene.marking_transparency
    del bpy.types.Scene.marking_color
    del bpy.types.Scene.intensity_threshold
    del bpy.types.Scene.markings_threshold
    del bpy.types.Scene.fatline_width
    del bpy.types.Scene.user_input_result
    del bpy.types.Scene.save_shape
    del bpy.types.Scene.auto_load
    del bpy.types.Scene.show_dots
    del bpy.types.Scene.z_height_cut_off
    del bpy.types.Scene.extra_z_height
    del bpy.types.Scene.points_percentage
    
    bpy.utils.unregister_class(OBJECT_OT_simple_undo)
    bpy.utils.unregister_class(OBJECT_OT_simple_redo)
                 
if __name__ == "__main__":
    register()
    
    if(context.scene.auto_load):
        bpy.ops.wm.las_auto_open()
        
 
#triangle mark functions
def move_triangle_to_line(triangle, line_start, line_end):
    # Convert inputs to numpy arrays for easier calculations
    triangle_np = np.array(triangle)
    line_start_np = np.array(line_start)
    line_end_np = np.array(line_end)

    # Identify the base vertices (the two closest to the line)
    base_vertex_indices = find_base_vertices(triangle_np, line_start_np, line_end_np)
    base_vertices = triangle_np[base_vertex_indices]

    # Find the closest points on the line for the base vertices
    closest_points = [closest_point(vertex, line_start_np, line_end_np) for vertex in base_vertices]

    # Move the base vertices to the closest points on the line
    triangle_np[base_vertex_indices] = closest_points

    # Calculate the height of the triangle to reposition the third vertex
    third_vertex_index = 3 - sum(base_vertex_indices)  #  indices should be 0, 1, 2
    height_vector = triangle_np[third_vertex_index] - np.mean(base_vertices, axis=0)
    triangle_np[third_vertex_index] = np.mean(closest_points, axis=0) + height_vector

    return triangle_np.tolist()

def find_base_vertices(triangle, line_start, line_end):
    distances = [np.linalg.norm(closest_point(vertex, line_start, line_end) - vertex) for vertex in triangle]
    sorted_indices = np.argsort(distances)
    return sorted_indices[:2]  # Indices of the two closest vertices

def find_closest_vertex_to_line(triangle, line_start, line_end):
    min_distance = float('inf')
    closest_vertex_index = -1

    for i, vertex in enumerate(triangle):
        closest_point_on_line = closest_point(vertex, line_start, line_end)
        distance = np.linalg.norm(vertex - closest_point_on_line)
        if distance < min_distance:
            min_distance = distance
            closest_vertex_index = i

    return closest_vertex_index

def find_base_vertex(triangle, line_start, line_end):
    min_distance = float('inf')
    base_vertex = None
    base_index = None

    for i, vertex in enumerate(triangle):
        closest_point_on_line = closest_point(vertex, line_start, line_end)
        distance = np.linalg.norm(vertex - closest_point_on_line)
        if distance < min_distance:
            min_distance = distance
            base_vertex = vertex
            base_index = i

    return base_vertex, base_index

def closest_point(point, line_start, line_end):
    line_vec = line_end - line_start
    point_vec = point - line_start
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    point_vec_scaled = point_vec / line_len
    t = np.dot(line_unitvec, point_vec_scaled)    
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = line_vec * t
    return line_start + nearest

def move_blender_triangle_objects(new_vertices, line_start, line_end):
    for obj in bpy.data.objects:
        if "Triangle Shape" in obj.name and obj.type == 'MESH':
            if len(obj.data.vertices) >= 3:
                # Assuming the object represents a triangle
                current_triangle = [obj.data.vertices[i].co for i in range(3)]
                moved_triangle = move_triangle_to_line(current_triangle, line_start, line_end)

                # Update the vertices of the mesh
                for i, vertex in enumerate(obj.data.vertices[:3]):
                    vertex.co = moved_triangle[i]
            else:
                print(f"Object '{obj.name}' does not have enough vertices")
                   
#opencv
def save_shape_as_image(obj):
    
    obj_name=obj.name
    save_shape_checkbox = context.scene.save_shape
    if obj_name =="Thin Line":
        return
    
    if save_shape_checkbox:
        # Ensure the object exists
        if not obj:
            raise ValueError(f"Object {obj_name} not found.")
        
        # Get the directory of the current Blender file
        blend_file_path = bpy.data.filepath
        directory = os.path.dirname(blend_file_path)

        # Create a folder 'road_mark_images' if it doesn't exist
        images_dir = os.path.join(directory, 'road_mark_images')
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)


        # Set up rendering
        bpy.context.scene.render.engine = 'CYCLES'  # or 'BLENDER_EEVEE'
        bpy.context.scene.render.image_settings.file_format = 'JPEG'
        bpy.context.scene.render.resolution_x = 256
        bpy.context.scene.render.resolution_y = 256
        bpy.context.scene.render.resolution_percentage = 100

        # Set up camera
        cam = bpy.data.cameras.new("Camera")
        cam_ob = bpy.data.objects.new("Camera", cam)
        bpy.context.scene.collection.objects.link(cam_ob)
        bpy.context.scene.camera = cam_ob

        # Use orthographic camera
        cam.type = 'ORTHO'

        # Calculate the bounding box of the object
        local_bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        min_corner = Vector(local_bbox_corners[0])
        max_corner = Vector(local_bbox_corners[6])

        # Position the camera
        bbox_center = (min_corner + max_corner) / 2
        bbox_size = max_corner - min_corner
        cam_ob.location = bbox_center + Vector((0, 0, max(bbox_size.x, bbox_size.y, bbox_size.z)))

        # Adjust the orthographic scale to 75%
        cam.ortho_scale = 1.33 * max(bbox_size.x, bbox_size.y)

        # Point the camera downward
        cam_ob.rotation_euler = (0, 0, 0)

        # Set up lighting
        light = bpy.data.lights.new(name="Light", type='POINT')
        light_ob = bpy.data.objects.new(name="Light", object_data=light)
        bpy.context.scene.collection.objects.link(light_ob)
        light_ob.location = cam_ob.location + Vector((0, 0, 2))
        light.energy = 50
        light.energy += 100* max(0, cam_ob.location.z-1)
        print("light energy: ",light.energy)
        #light.energy=10
        
        # Set object material to bright white
        mat = bpy.data.materials.new(name="WhiteMaterial")
        mat.diffuse_color = (1, 1, 1, 1)  # White color
        obj.data.materials.clear()
        obj.data.materials.append(mat)

        # Set world background to black
        bpy.context.scene.world.use_nodes = True
        bpy.context.scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)  # Black color
        
        # Render and save the image with object's name
        file_path = os.path.join(images_dir, f'{obj_name}.png')
        bpy.context.scene.render.filepath = file_path
        bpy.ops.render.render(write_still=True)
        print("saved image to: ",file_path)
        # Cleanup: delete the created camera, light, and material
        bpy.data.objects.remove(cam_ob)
        bpy.data.objects.remove(light_ob)
        bpy.data.materials.remove(mat)
        print("deleted camera, light, and material")

#Opencv shape detection from points    
def detect_shape_from_points(points, from_bmesh=False, scale_factor=100):

    if from_bmesh:
        # Convert bmesh vertices to numpy array
        coords_list = np.array([(point.x, point.y, point.z) for point in points])
    else:
        coords_list = np.array(points)
    
    coords_list = filter_noise_with_dbscan(coords_list)
    #Convert the floating points to integers
    int_coords = np.round(coords_list).astype(int)

    #Remove the Z-coordinate 
    xy_coords = int_coords[:, :2]  

    #Scale the coordinates to increase the shape size
    scaled_coords = xy_coords * scale_factor
    
    #Find min and max bounds for the coordinates for the canvas size
    min_vals = xy_coords.min(axis=0) * scale_factor  
    max_vals = xy_coords.max(axis=0) * scale_factor  
    
    #Create a binary image
    img = np.zeros((max_vals[1] - min_vals[1] + scale_factor, max_vals[0] - min_vals[0] + scale_factor), dtype=np.uint8)
    #Apply gaussian blur
    img = cv2.GaussianBlur(img, (5, 5), 0)
    #Shift points based on min_vals to fit within the image
    shifted_points = scaled_coords  - min_vals + 5  

    #Reshape points to the format needed by fillPoly 
    reshaped_points = shifted_points.reshape((-1, 1, 2))

    #Draw the shape filled (as white) on the image
    cv2.fillPoly(img, [reshaped_points], 255)

    #Find contours and select the contour with the maximum area 
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    print(len(contours) , "contours found")
    #Draw all the contours found on an image
    contour_image = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)  # Convert to a 3-channel image 
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 1)  # Draws contours in green 

    display_contour=False
    save_countor=True
    
    if(display_contour):
        #Display the image with contours
        cv2.imshow("Contours", contours)
        cv2.waitKey(0)  
        cv2.destroyAllWindows()
        
    if(save_countor):
        # Save the image
        output_path = output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'contours', 'contours.png')
        cv2.imwrite(output_path, contour_image)
        print("contour saved at ",output_path)
    
    #Approximate the contour to a simpler shape
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    #Based on the number of vertices, determine the shape
    num_vertices = len(approx)
    print("number of vertices detected: ",num_vertices)
    shape = "unknown"
    if num_vertices == 3:
        shape="rectangle"
    if num_vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspectRatio = float(w) / h
        if 0.95 <= aspectRatio <= 1.05:
            shape = "square"
        else:
            shape = "rectangle"

        
    #Print the detected shape
    print(f"Detected shape: {shape}")

    return shape

#not used
def draw_polyline_from_points(points, name, color=(0.8, 0.3, 0.3), thickness=0.02):
    # Create a new curve and a new object using that curve
    curve_data = bpy.data.curves.new(name=name, type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.fill_mode = 'FULL'
    curve_data.bevel_depth = thickness

    # Create a new spline within that curve
    polyline = curve_data.splines.new('POLY')
    polyline.points.add(len(points) - 1)  # The new spline has no points by default, add the number needed

    # Assign the point coordinates to the spline points
    for i, point in enumerate(points):
        polyline.points[i].co = (point[0], point[1], point[2], 1)  # The fourth value is the 'w' for the homogeneous coordinate

    # Create the object
    curve_object = bpy.data.objects.new(name, curve_data)
    bpy.context.collection.objects.link(curve_object)

    # Assign a color to the curve
    material = bpy.data.materials.new(name=name + "Material")
    material.diffuse_color = (*color, 1)  # RGB + Alpha
    curve_object.data.materials.append(material)

def get_principal_component(points):
    # Compute the centroid of the points
    centroid = np.mean(points, axis=0)
    
    # Center the data points
    centered_points = points - centroid
    
    # Compute the covariance matrix
    covariance_matrix = np.cov(centered_points, rowvar=False)
    
    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # Sort the eigenvectors based on the eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    
    # Extract the principal component
    principal_component = eigenvectors[:, sorted_indices[0]]
    
    return principal_component

def align_shapes(original_coords, perfect_coords):
    # Compute centroids
    original_centroid = np.mean(original_coords, axis=0)
    perfect_centroid = np.mean(perfect_coords, axis=0)

    # Compute vectors from centroids to vertices
    original_vectors = original_coords - original_centroid
    perfect_vectors = perfect_coords - perfect_centroid

    # Normalize vectors
    original_directions = original_vectors / np.linalg.norm(original_vectors, axis=1)[:, None]
    perfect_directions = perfect_vectors / np.linalg.norm(perfect_vectors, axis=1)[:, None]

    # Match vertices
    matched_vertices = []
    for p_dir in perfect_directions:
        similarities = np.dot(original_directions, p_dir)
        best_match_idx = np.argmax(similarities)
        matched_vertices.append(original_coords[best_match_idx])

    return matched_vertices
      
def perpendicular_bisector_from_line(start, end):
    # Ensure that the lines are not parallel by adding a small value to the denominator
    midpoint = (start + end) / 2
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    # To avoid division by zero
    if np.abs(dy) < 1e-10:
        dy += 1e-10
    slope = -dx / dy
    intercept = midpoint[1] - slope * midpoint[0]
    return slope, intercept

def intersection_of_lines(line1, line2):
    # If the slopes are too close, they might be considered parallel
    slope1, intercept1 = line1
    slope2, intercept2 = line2
    # If the lines are parallel return a point at infinity
    if np.isclose(slope1, slope2):
        return np.array([np.inf, np.inf])
    x_intersect = (intercept2 - intercept1) / (slope1 - slope2)
    y_intersect = slope1 * x_intersect + intercept1
    return np.array([x_intersect, y_intersect])

def find_circumcenter(point1, point2, point3):
    bisector1 = perpendicular_bisector_from_line(point1, point2)
    bisector2 = perpendicular_bisector_from_line(point1, point3)
    circumcenter = intersection_of_lines(bisector1, bisector2)
    return circumcenter

def find_radius(circumcenter, point):
    return np.linalg.norm(circumcenter - point)

def create_arc_points(circumcenter, radius, start_angle, end_angle, num_points, z_coord):
    # Generate points on the arc in 2D
    arc_points_2d = [
        circumcenter + radius * np.array([np.cos(angle), np.sin(angle)])
        for angle in np.linspace(start_angle, end_angle, num_points)
    ]
    # Add the Z-coordinate to each 2D point to make it 3D
    arc_points_3d = [(*point, z_coord) for point in arc_points_2d]
    
    return arc_points_3d
    
def find_circle_through_3_points(point1, point2, point3):
    # Convert to numpy arrays for easier manipulation
    point1 = np.array(point1)
    point2 = np.array(point2)
    point3 = np.array(point3)

    # Create the two perpendicular bisectors of the triangle formed by the points
    bisector1 = perpendicular_bisector_from_line(point1, point2)
    bisector2 = perpendicular_bisector_from_line(point2, point3)

    # Find the intersection of the two bisectors, which is the circumcenter
    circumcenter = intersection_of_lines(bisector1, bisector2)

    # The radius is the distance from the circumcenter to any of the three points
    radius = np.linalg.norm(circumcenter - point1[:2])  # Use only x and y for distance

    return circumcenter, radius        
        
def interpolate_curve_old(coords):
    coords_np = np.array(coords)
    sorted_coords = coords_np[np.argsort(coords_np[:, 0])]
    x = sorted_coords[:, 0]
    y = sorted_coords[:, 1]
    
    # Create a smooth curve through the points
    t = np.linspace(0, 1, len(x))
    t_new = np.linspace(0, 1, 100)  # Increase this for more resolution
    spl = make_interp_spline(t, np.c_[x, y], k=3)  # k=3 for cubic spline
    curve = spl(t_new)
    
    return curve             
    
def create_bezier_spline(bezier_data, point1, control_point, point2):
    spline = bezier_data.splines.new('BEZIER')
    spline.bezier_points.add(2)
    
    spline.bezier_points[0].co = point1
    spline.bezier_points[0].handle_right_type = 'FREE'
    spline.bezier_points[0].handle_left_type = 'FREE'

    spline.bezier_points[1].co = control_point
    spline.bezier_points[1].handle_right_type = 'FREE'
    spline.bezier_points[1].handle_left_type = 'FREE'
    
    spline.bezier_points[2].co = point2
    spline.bezier_points[2].handle_right_type = 'FREE'
    spline.bezier_points[2].handle_left_type = 'FREE'
    
    # Set the handles to align them with the control points for smooth curves
    spline.bezier_points[0].handle_right = spline.bezier_points[1].co
    spline.bezier_points[2].handle_left = spline.bezier_points[1].co

def convert_curve_object_to_mesh(curve_obj):
    # This function converts a curve object to a mesh object
    mesh_data = curve_obj.to_mesh()
    mesh_obj = bpy.data.objects.new(curve_obj.name + '_mesh', mesh_data)
    bpy.context.collection.objects.link(mesh_obj)
    return mesh_obj

def convert_curve_to_mesh_old(curve_obj):
    
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Select the curve object
    bpy.context.view_layer.objects.active = curve_obj
    curve_obj.select_set(True)
    
    # Convert the curve to a mesh
    bpy.ops.object.convert(target='MESH')

def draw_curved_line_shape(top_left, top_right, highest_top, bottom_left, bottom_right, lowest_bottom):
    start_time = time.time()
    marking_color = bpy.context.scene.marking_color 
    transparency = bpy.context.scene.marking_transparency

    # Ensure object mode
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    # Create the Bezier curves
    bezier_top = bpy.data.curves.new('bezier_top', 'CURVE')
    create_bezier_spline(bezier_top, Vector(top_left), Vector(highest_top), Vector(top_right))

    bezier_bottom = bpy.data.curves.new('bezier_bottom', 'CURVE')
    create_bezier_spline(bezier_bottom, Vector(bottom_left), Vector(lowest_bottom), Vector(bottom_right))

    # Add the Bezier curves to the scene as objects
    top_curve_obj = bpy.data.objects.new('TopBezierCurve', bezier_top)
    bottom_curve_obj = bpy.data.objects.new('BottomBezierCurve', bezier_bottom)
    bpy.context.collection.objects.link(top_curve_obj)
    bpy.context.collection.objects.link(bottom_curve_obj)

    bpy.context.view_layer.update()

    # Convert the Bezier curves to meshes
    convert_curve_to_mesh(top_curve_obj)
    convert_curve_to_mesh(bottom_curve_obj)

    bpy.ops.object.select_all(action='DESELECT')

    # Select the curve objects
    top_curve_obj.select_set(True)
    bottom_curve_obj.select_set(True)
    bpy.context.view_layer.objects.active = top_curve_obj

    # Join the two objects into one mesh
    bpy.ops.object.join()

    #  top_curve_obj contains the mesh data for both curves
    # Create a bmesh from this object
    bm = bmesh.new()
    bm.from_mesh(top_curve_obj.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.001)
    bmesh.ops.convex_hull(bm, input=bm.verts)
    bm.to_mesh(top_curve_obj.data)
    bm.free()

    # Create a new material for the combined shape
    shape_material = bpy.data.materials.new(name="shape color")
    shape_material.diffuse_color = (marking_color[0], marking_color[1], marking_color[2], transparency)

    # Enable transparency in the material settings
    shape_material.use_nodes = True
    shape_material.blend_method = 'BLEND'

    # Find the Principled BSDF node and set its alpha value
    principled_node = next(n for n in shape_material.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    principled_node.inputs['Alpha'].default_value = transparency

    # Assign the material to the object
    if len(top_curve_obj.data.materials) > 0:
        top_curve_obj.data.materials[0] = shape_material
    else:
        top_curve_obj.data.materials.append(shape_material)

    # After the object is created, store it 
   
    print("rendered curved line shape in: ", time.time() - start_time)

    
    start_time = time.time()
    marking_color = bpy.context.scene.marking_color 
    transparency = bpy.context.scene.marking_transparency
    
    # Create a new mesh and link it to the scene
    mesh_data = bpy.data.meshes.new("curved_line_shape_mesh")
    curve_obj = bpy.data.objects.new("Curved Line Shape", mesh_data)
    bpy.context.collection.objects.link(curve_obj)
    
    # Start a bmesh instance
    bm = bmesh.new()

    # Create the top and bottom Bezier curves
    bezier_top = bpy.data.curves.new('bezier_top', 'CURVE')
    bezier_bottom = bpy.data.curves.new('bezier_bottom', 'CURVE')
    
    # Function to create a bezier spline
    def create_bezier_spline(bezier_data, point1, control_point, point2):
        spline = bezier_data.splines.new('BEZIER')
        spline.bezier_points.add(2)
        
        spline.bezier_points[0].co = point1
        spline.bezier_points[0].handle_right_type = 'FREE'
        spline.bezier_points[0].handle_left_type = 'FREE'

        spline.bezier_points[1].co = control_point
        spline.bezier_points[1].handle_right_type = 'FREE'
        spline.bezier_points[1].handle_left_type = 'FREE'
        
        spline.bezier_points[2].co = point2
        spline.bezier_points[2].handle_right_type = 'FREE'
        spline.bezier_points[2].handle_left_type = 'FREE'
        
        # Set the handles to align them with the control points for smooth curves
        spline.bezier_points[0].handle_right = spline.bezier_points[1].co
        spline.bezier_points[2].handle_left = spline.bezier_points[1].co

        return spline

    # Create the top bezier spline
    create_bezier_spline(bezier_top, Vector(top_left), Vector(highest_top), Vector(top_right))
    
    # Create the bottom bezier spline
    create_bezier_spline(bezier_bottom, Vector(bottom_left), Vector(lowest_bottom), Vector(bottom_right))

    # Add the bezier curves to the scene
    top_curve_obj = bpy.data.objects.new('TopBezierCurve', bezier_top)
    bottom_curve_obj = bpy.data.objects.new('BottomBezierCurve', bezier_bottom)
    bpy.context.collection.objects.link(top_curve_obj)
    bpy.context.collection.objects.link(bottom_curve_obj)

    # Convert the curves to a mesh and join them
    bpy.context.view_layer.objects.active = top_curve_obj
    bpy.ops.object.convert(target='MESH')
    bpy.context.view_layer.objects.active = bottom_curve_obj
    bpy.ops.object.convert(target='MESH')
    
    bpy.ops.object.select_all(action='DESELECT')
    top_curve_obj.select_set(True)
    bottom_curve_obj.select_set(True)
    bpy.context.view_layer.objects.active = top_curve_obj
    
    # Join the two objects into one mesh
    bpy.ops.object.join()
    
    # Update the bmesh with the new vertices
    bm.from_mesh(top_curve_obj.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.001)
    bmesh.ops.convex_hull(bm, input=bm.verts)
    bm.to_mesh(mesh_data)
    bm.free()

    # Delete the curve objects as they are no longer needed
    bpy.data.objects.remove(top_curve_obj)
    bpy.data.objects.remove(bottom_curve_obj)

    # Create a new material for the combined shape
    shape_material = bpy.data.materials.new(name="shape color")
    shape_material.diffuse_color = (marking_color[0], marking_color[1], marking_color[2], transparency) 

    # Enable transparency in the material settings
    shape_material.use_nodes = True
    shape_material.blend_method = 'BLEND'

    # Find the Principled BSDF node and set its alpha value
    principled_node = next(n for n in shape_material.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    principled_node.inputs['Alpha'].default_value = transparency
    
    # Assign the material to the object
    if len(obj.data.materials) > 0:
        # If the object already has materials, replace the first one with the new material
        obj.data.materials[0] = shape_material
    else:
        # Otherwise, add the new material to the object
        obj.data.materials.append(shape_material)
    
    # After the object is created, store it 
    store_object_state(obj)
    print("rendered curved line shape in: ", time.time()-start_time)
    
def create_bezier_spline_from_points(curveData, point1, point2, point3):
    # Create a new spline in that curve
    spline = curveData.splines.new(type='BEZIER')
    spline.bezier_points.add(3)
    
    # Assign the points to the spline points
    spline.bezier_points[0].co = point1
    spline.bezier_points[1].co = point2
    spline.bezier_points[2].co = point3
    
    # Handle types must be set to 'FREE' to allow custom handle positions
    for bp in spline.bezier_points:
        bp.handle_right_type = 'FREE'
        bp.handle_left_type = 'FREE'

    # Set the handles to halfway between the endpoints and the control point
    spline.bezier_points[0].handle_right = spline.bezier_points[0].co + (spline.bezier_points[1].co - spline.bezier_points[0].co) / 2
    spline.bezier_points[2].handle_left = spline.bezier_points[2].co + (spline.bezier_points[1].co - spline.bezier_points[2].co) / 2

def convert_curve_to_mesh(curve_obj):
    # Ensure we're in object mode and deselect all objects
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')

    # Select the curve object and make sure it's the active object
    curve_obj.select_set(True)
    bpy.context.view_layer.objects.active = curve_obj

    # Convert the curve to a mesh
    bpy.ops.object.convert(target='MESH')
    
    # Now the curve_obj has become a mesh object, return it
    return curve_obj

def draw_curved_rectangle(top_left, highest_top, top_right, bottom_left, lowest_bottom, bottom_right):
    # Create the Bezier curves
    bezier_top = bpy.data.curves.new('bezier_top', 'CURVE')
    bezier_bottom = bpy.data.curves.new('bezier_bottom', 'CURVE')
    
    create_bezier_spline_from_points(bezier_top, Vector(top_left), Vector(highest_top), Vector(top_right))
    create_bezier_spline_from_points(bezier_bottom, Vector(bottom_left), Vector(lowest_bottom), Vector(bottom_right))
    
    # Create objects for the curves and link them to the scene
    top_curve_obj = bpy.data.objects.new('TopBezierCurve', bezier_top)
    bpy.context.collection.objects.link(top_curve_obj)

    bottom_curve_obj = bpy.data.objects.new('BottomBezierCurve', bezier_bottom)
    bpy.context.collection.objects.link(bottom_curve_obj)

    # Make sure we're in object mode
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.update()

    # Convert the curves to meshes
    top_mesh_obj = convert_curve_to_mesh(top_curve_obj)
    bottom_mesh_obj = convert_curve_to_mesh(bottom_curve_obj)
    
    # Join the two mesh objects into one
    bpy.context.view_layer.objects.active = top_mesh_obj
    top_mesh_obj.select_set(True)
    bottom_mesh_obj.select_set(True)
    bpy.ops.object.join()  # This joins selected objects into the active one
    
    # Ensure the new mesh is selected and make it active
    combined_mesh_obj = bpy.context.view_layer.objects.active
    
    # Create the edges to close the rectangle, if needed
    bm = bmesh.new()
    bm.from_mesh(combined_mesh_obj.data)
    
    # Add the closing edges (if not already present)
    verts_top = [v for v in bm.verts if v.co.z == max(v.co.z for v in bm.verts)]
    verts_bottom = [v for v in bm.verts if v.co.z == min(v.co.z for v in bm.verts)]
    bmesh.ops.contextual_create(bm, geom=[verts_top[-1], verts_bottom[0]])
    bmesh.ops.contextual_create(bm, geom=[verts_top[0], verts_bottom[-1]])
    
    bm.to_mesh(combined_mesh_obj.data)
    bm.free()
    
    # Update mesh with new geometry
    combined_mesh_obj.data.update()

    return combined_mesh_obj

def lerp(v0, v1, t):
    return (1 - t) * v0 + t * v1

def interpolate_curve(p1, p2, p3, divisions):
    # Convert tuples to Vectors for easier manipulation
    p1, p2, p3 = map(Vector, [p1, p2, p3])

    # Create a list to store the new points
    curve_points = []

    # Interpolate along the curve
    for i in range(divisions + 1):
        t = i / divisions
        # Quadratic Bezier Curve formula
        point = (1 - t)**2 * p1 + 2 * (1 - t) * t * p2 + t**2 * p3
        curve_points.append(point)

    return curve_points

def create_dotted_curved_line(p1, p2, p3, p4, p5, p6, divisions=5):
    # Interpolate the top and bottom curves
    top_curve_points = interpolate_curve(p1, p3, p2, divisions)
    bottom_curve_points = interpolate_curve(p4, p6, p5, divisions)
    
    top_dotted_line, bottom_dotted_line = interpolate_curve(p1, p3, p2, divisions), interpolate_curve(p4, p6, p5, divisions)
    draw_dotted_line(top_dotted_line, "TopDottedLine")
    draw_dotted_line(bottom_dotted_line, "BottomDottedLine")
    
    # Combine the points to form two dotted lines
    return top_curve_points, bottom_curve_points

def draw_dotted_line(points, name):
    # Create a new mesh
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)

    # Link it to the scene
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)

    # Create mesh from given verts, edges, and faces
    mesh.from_pydata(points, [], [])

    # Update mesh with new data
    mesh.update()

def quadratic_bezier(p0, p1, p2, t):
    """
    Calculate the quadratic Bezier curve point at t [0, 1]
    """
    return (1 - t)**2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2

def create_dotted_line(p0, p1, p2, divisions):
    """
    Create a dotted line from p0 to p2 through p1.
    """
    dotted_line = []
    for i in range(divisions + 1):
        t = i / divisions
        point = quadratic_bezier(p0, p1, p2, t)
        dotted_line.append(point)
    return dotted_line

def create_dotted_curved_lines(top_left, top_right, highest_top, bottom_left, bottom_right, lowest_bottom, divisions=5):
    # Convert to numpy for easier calculations
    top_left = np.array(top_left)
    top_right = np.array(top_right)
    highest_top = np.array(highest_top)
    bottom_left = np.array(bottom_left)
    bottom_right = np.array(bottom_right)
    lowest_bottom = np.array(lowest_bottom)

    # Create the top dotted lines from left to middle and right to middle
    top_dotted_line_left = create_dotted_line(top_left, highest_top, highest_top, divisions)
    top_dotted_line_right = create_dotted_line(highest_top, highest_top, top_right, divisions)

    # Merge and remove the duplicate middle point
    top_dotted_line = top_dotted_line_left[:-1] + top_dotted_line_right

    # Create the bottom dotted lines from left to middle and right to middle
    bottom_dotted_line_left = create_dotted_line(bottom_left, lowest_bottom, lowest_bottom, divisions)
    bottom_dotted_line_right = create_dotted_line(lowest_bottom, lowest_bottom, bottom_right, divisions)

    # Merge and remove the duplicate middle point
    bottom_dotted_line = bottom_dotted_line_left[:-1] + bottom_dotted_line_right

    # Return both dotted lines
    return top_dotted_line, bottom_dotted_line

def create_mesh_object_from_lines(top_line, bottom_line, z_height, color, transparency, mesh_name="CurvedShape"):
    mesh_data = bpy.data.meshes.new(mesh_name)
    mesh_obj = bpy.data.objects.new(mesh_name, mesh_data)
    bpy.context.collection.objects.link(mesh_obj)
    
    # Create a new BMesh
    bm = bmesh.new()
    
    # Add top line vertices and bottom line vertices
    top_verts = [bm.verts.new((x, y, z + z_height)) for x, y, z in top_line]
    bottom_verts = [bm.verts.new((x, y, z + z_height)) for x, y, z in bottom_line]
    
    # Ensure we have the same number of vertices in top and bottom lines
    if len(top_verts) != len(bottom_verts):
        raise ValueError("Top and bottom lines must have the same number of points")
    
    # Create edges by connecting consecutive vertices
    top_edges = [bm.edges.new((top_verts[i], top_verts[i + 1])) for i in range(len(top_verts) - 1)]
    bottom_edges = [bm.edges.new((bottom_verts[i], bottom_verts[i + 1])) for i in range(len(bottom_verts) - 1)]
    
    # Create faces between the top and bottom edges
    for i in range(len(top_verts) - 1):
        bm.faces.new((top_verts[i], top_verts[i + 1], bottom_verts[i + 1], bottom_verts[i]))
    
    # Finish up, write the bmesh back to the mesh
    bm.to_mesh(mesh_data)
    bm.free()
    
    # Create a new material with the given color and transparency
    shape_material = bpy.data.materials.new(name="ShapeMaterial")
    shape_material.diffuse_color = (*color, transparency)
    shape_material.use_nodes = True
    shape_material.blend_method = 'BLEND'
    principled_node = shape_material.node_tree.nodes["Principled BSDF"]
    principled_node.inputs['Alpha'].default_value = transparency
    
    # Assign the material to the object
    mesh_obj.data.materials.append(shape_material)
    
    return mesh_obj

def create_fixed_rectangle_old(coords, width=None, height=None):
    
    coords_np = np.array(coords)
    centroid = coords_np.mean(axis=0)
    principal_direction = get_principal_component(coords_np[:, :2])

    if width is None or height is None:
        min_vals = coords_np.min(axis=0)
        max_vals = coords_np.max(axis=0)
        width = max_vals[0] - min_vals[0] if width is None else width
        height = max_vals[1] - min_vals[1] if height is None else height

    half_width = width / 2
    half_height = height / 2

    rectangle_coords = np.array([
        [-half_width, -half_height],  
        [half_width, -half_height],  
        [half_width, half_height],   
        [-half_width, half_height]   
    ])

    theta = np.arctan2(principal_direction[1], principal_direction[0])
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    rotated_coords = rectangle_coords.dot(rotation_matrix)
    final_coords = rotated_coords + centroid[:2]
    final_coords = [(x, y, centroid[2]) for x, y in final_coords]

    return final_coords

def create_fixed_triangle_old(coords, size=None):
    #Find the average of the points 
    coords_np = np.array(coords)
    centroid = coords_np.mean(axis=0)

    distances = np.sqrt(np.sum((coords_np - centroid) ** 2, axis=1))
    avg_distance = np.mean(distances)

    triangle_coords = []
    for i in range(3):
        angle = 2 * np.pi / 3 * i  # 120 degrees difference
        new_point = centroid + np.array([np.cos(angle), np.sin(angle), 0]) * avg_distance
        triangle_coords.append(new_point.tolist())
    
    aligned_coords = align_shapes(coords, triangle_coords)
    return aligned_coords

def calculate_curved_line(top_left, top_right, highest_top, bottom_left, bottom_right, lowest_bottom):
    #  3D coordinates for all points
    top_left = np.array(top_left)
    top_right = np.array(top_right)
    highest_top = np.array(highest_top)
    bottom_left = np.array(bottom_left)
    bottom_right = np.array(bottom_right)
    lowest_bottom = np.array(lowest_bottom)
    
    # Find circumcenters and radii for the top and bottom arcs
    circumcenter_top, radius_top = find_circle_through_3_points(top_left, highest_top, top_right)
    circumcenter_bottom, radius_bottom = find_circle_through_3_points(bottom_left, lowest_bottom, bottom_right)
    
    # Get the Z-coordinates for the top and bottom arcs
    z_coord_top = highest_top[2]  # Z-coordinate for the top arc
    z_coord_bottom = lowest_bottom[2]  # Z-coordinate for the bottom arc
    
    # Calculate angles for the arc endpoints
    angle_start_top = np.arctan2(top_left[1] - circumcenter_top[1], top_left[0] - circumcenter_top[0])
    angle_end_top = np.arctan2(top_right[1] - circumcenter_top[1], top_right[0] - circumcenter_top[0])
    angle_start_bottom = np.arctan2(bottom_left[1] - circumcenter_bottom[1], bottom_left[0] - circumcenter_bottom[0])
    angle_end_bottom = np.arctan2(bottom_right[1] - circumcenter_bottom[1], bottom_right[0] - circumcenter_bottom[0])
    
    # Create arc points for the top and bottom edges
    num_points = 20  # Adjust the number of points as needed for smoothness
    arc_points_top = create_arc_points(circumcenter_top, radius_top, angle_start_top, angle_end_top, num_points, z_coord_top)
    arc_points_bottom = create_arc_points(circumcenter_bottom, radius_bottom, angle_start_bottom, angle_end_bottom, num_points, z_coord_bottom)
    
    # Combine points to create the curved rectangle
    # Ensure the order of points forms a continuous loop; this may require additional points along the corners
    curved_rectangle_points = arc_points_top + [top_right.tolist(), bottom_right.tolist()] + arc_points_bottom[::-1] + [bottom_left.tolist(), top_left.tolist()]

    for point in curved_rectangle_points:
        mark_point(point)
        
    draw_polyline_from_points(arc_points_top, "TopArc", color=(1, 0, 0))  
    draw_polyline_from_points(arc_points_bottom, "BottomArc", color=(0, 1, 0))  
    #return curved_rectangle_points

def visualize_search_radius(location, search_radius, name="SearchRadius"):
    # Check if the mesh already exists. If it does, remove it.
    if name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)

    # Create a new mesh and a new object
    mesh = bpy.data.meshes.new(name + "Mesh")
    obj = bpy.data.objects.new(name, mesh)

    # Link the object to the scene
    bpy.context.collection.objects.link(obj)

    # Set the object's location to the target location
    obj.location = location

    # Create a sphere to represent the search radius
    bpy.ops.object.select_all(action='DESELECT')  # Deselect all objects
    bpy.context.view_layer.objects.active = obj   # Make the new object the active object
    obj.select_set(True)                          # Select the new object

    # Enter edit mode to create the sphere
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.primitive_uv_sphere_add(radius=search_radius, location=location)
    bpy.ops.object.mode_set(mode='OBJECT')  # Exit edit mode

    # We can also make the sphere transparent to not occlude the view
    mat = bpy.data.materials.new(name="SearchRadiusMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes["Principled BSDF"].inputs["Alpha"].default_value = 0.3  # Set transparency to 70%
    obj.data.materials.append(mat)
    obj.show_transparent = True

    # Finally, update the scene
    bpy.context.view_layer.update()
    bpy.context.view_layer.objects.active = obj  # Make the object the active object
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS') # Set the origin to the center of the object's bounding box
    return obj

def create_fixed_length_segments(points, segment_length=1.0):
    # function generates points on a polyline with fixed segment lengths
    extended_points = [Vector(points[0])]  # Start with the first point
    total_distance = 0  # Keep track of the total distance
    segment_count = 0  # Count the number of full segments

    for i in range(1, len(points)):
        start_point = Vector(points[i - 1])
        end_point = Vector(points[i])
        segment_vector = end_point - start_point
        segment_distance = segment_vector.length
        total_distance += segment_distance

        # Normalize the segment vector
        segment_vector.normalize()

        # Generate points at fixed intervals between start and end
        while segment_distance > segment_length:
            new_point = start_point + segment_vector * segment_length
            extended_points.append(new_point)
            start_point = new_point
            segment_distance -= segment_length
            segment_count += 1

        # Add the last point if it doesn't fit a full segment
        if segment_distance > 0:
            extended_points.append(end_point)

    # Adjust the last segment if it's not a full segment
    if total_distance % segment_length != 0:
        extended_points[-1] = extended_points[-2] + segment_vector * (total_distance % segment_length)

    return extended_points, total_distance, segment_count + 1  # Include the last partial segment

def create_curved_line(coords):
    
    coords_np = np.array(coords)
    
    # Calculate the centroid
    centroid = np.mean(coords_np, axis=0)
    
    # Sort points by x for interpolation
    sorted_points = coords_np[np.argsort(coords_np[:, 0])]
    
    # Determine the top and bottom points
    top_points = sorted_points[sorted_points[:, 1] > centroid[1]]
    bottom_points = sorted_points[sorted_points[:, 1] <= centroid[1]]
    
    # Create cubic spline interpolations
    top_spline = CubicSpline(top_points[:, 0], top_points[:, 1])
    bottom_spline = CubicSpline(bottom_points[:, 0], bottom_points[:, 1])
    
    # Sample points from the spline
    x_vals = np.linspace(np.min(coords_np[:, 0]), np.max(coords_np[:, 0]), len(coords_np))
    top_y_vals = top_spline(x_vals)
    bottom_y_vals = bottom_spline(x_vals)
    
    # Combine the top and bottom points to get the final shape
    rectangle_coords = np.column_stack((x_vals, top_y_vals))
    rectangle_coords = np.vstack((rectangle_coords, np.column_stack((x_vals[::-1], bottom_y_vals[::-1]))))
    
    return rectangle_coords.tolist()

    # Create a new curve data object
    curve_data = bpy.data.curves.new(name, type='CURVE')
    curve_data.dimensions = '3D'

    # Create a new spline in the curve
    spline = curve_data.splines.new('BEZIER')
    spline.bezier_points.add(2)  # We need exactly three points for the Bezier spline

    # Assign the points to the spline. The handles will be automatically calculated.
    spline.bezier_points[0].co = points[0]
    spline.bezier_points[0].handle_left_type = 'AUTO'
    spline.bezier_points[0].handle_right_type = 'AUTO'

    spline.bezier_points[1].co = points[1]
    spline.bezier_points[1].handle_left_type = 'AUTO'
    spline.bezier_points[1].handle_right_type = 'AUTO'

    spline.bezier_points[2].co = points[2]
    spline.bezier_points[2].handle_left_type = 'AUTO'
    spline.bezier_points[2].handle_right_type = 'AUTO'

    # Create a new object with the curve
    curve_obj = bpy.data.objects.new(name, curve_data)
    bpy.context.collection.objects.link(curve_obj)

    # Set the curve to use a full path and set the bevel depth for width
    curve_data.bevel_depth = width

    # Create a new material with the given color
    mat = bpy.data.materials.new(name + "_Mat")
    mat.diffuse_color = color
    curve_obj.data.materials.append(mat)

    # Ensure the curve data is updated with the new information
    #curve_data.update()

    return curve_obj
   
def find_closest_pairs(top_half, bottom_half):
    pairs = []
    for top_point in top_half:
        distances = np.sqrt(np.sum((bottom_half - top_point) ** 2, axis=1))
        closest_index = np.argmin(distances)
        closest_point = bottom_half[closest_index]
        pairs.append((top_point, closest_point))
    return pairs

    middle_points = []
    for top_point, bottom_point in pairs:
        middle_point = (top_point + bottom_point) / 2
        middle_points.append(middle_point)
        mark_point(middle_point, "middle_point")
    return middle_points

def find_evenly_distributed_pairs(coords_list, num_pairs):
    coords_np = np.array(coords_list)
    coords_y = coords_np[:, 1]

    top_mask = coords_y >= np.median(coords_y)
    top_half = coords_np[top_mask]
    bottom_half = coords_np[~top_mask]

    top_half_sorted = top_half[np.argsort(top_half[:, 0])]
    bottom_half_sorted = bottom_half[np.argsort(bottom_half[:, 0])]

    pairs = find_closest_pairs(top_half_sorted, bottom_half_sorted)
    selected_pairs = pairs[::len(pairs) // num_pairs][:num_pairs]

    return selected_pairs

def locate_middle_points(coords_list, num_points=3):
    coords_np = np.array(coords_list)

    # Find corner points
    top_left = coords_np[coords_np[:, 1].argmax()]
    top_right = coords_np[coords_np[:, 0].argmax()]
    bottom_left = coords_np[coords_np[:, 1].argmin()]
    bottom_right = coords_np[coords_np[:, 0].argmin()]

    # Calculate vectors for the long sides of the rectangle
    left_side_vector = bottom_left - top_left
    right_side_vector = bottom_right - top_right

    # Project points onto the long side vectors to find extreme points
    def project_and_find_extremes(points, base_point, vector):
        projections = np.dot(points - base_point, vector) / np.linalg.norm(vector)
        min_idx = np.argmin(projections)
        max_idx = np.argmax(projections)
        return points[min_idx], points[max_idx]

    top_left_proj, bottom_left_proj = project_and_find_extremes(coords_np, top_left, left_side_vector)
    top_right_proj, bottom_right_proj = project_and_find_extremes(coords_np, top_right, right_side_vector)

    # Function to interpolate points on a line
    def interpolate_points(start, end, num):
        return [start + (end - start) * i / (num - 1) for i in range(num)]

    # Interpolate points on top and bottom edges
    left_points = interpolate_points(top_left_proj, bottom_left_proj, num_points)
    right_points = interpolate_points(top_right_proj, bottom_right_proj, num_points)

    # Pair points based on proximity
    pairs = list(zip(left_points, right_points))

    # Calculate middle points
    middle_points = [(left + right) / 2 for left, right in pairs]

    mark_point(top_left,"top_left")
    mark_point(top_right,"top_right")
    mark_point(bottom_left,"bottom_left")
    mark_point(bottom_right,"bottom_right")
    
    for point in left_points:
        mark_point(point,"left_points")
    for point in right_points:
        mark_point(point,"right_points")
    for point in middle_points:
        mark_point(point,"middle")
        
    return middle_points


    coords_np = np.array(coords_list)
    points_kdtree = KDTree(coords_np)
    
    def find_closest_point(point, kdtree):
        distance, index = kdtree.query(point)
        return kdtree.data[index]

    def find_corner_points(coords_np):
        top_left = coords_np[coords_np[:, 1].argmax()]
        top_right = coords_np[coords_np[:, 0].argmax()]
        bottom_right = coords_np[coords_np[:, 1].argmin()]
        bottom_left = coords_np[coords_np[:, 0].argmin()]
        return top_left, top_right, bottom_left, bottom_right

    top_left, top_right, bottom_left, bottom_right = find_corner_points(coords_np)

    # Snap the corner points to the nearest points in coords_list
    top_left = find_closest_point(top_left, points_kdtree)
    top_right = find_closest_point(top_right, points_kdtree)
    bottom_left = find_closest_point(bottom_left, points_kdtree)
    bottom_right = find_closest_point(bottom_right, points_kdtree)

    # Ensure there are at least two middle points
    num_middle_points = max(num_middle_points, 2)

    def interpolate_points(p1, p2, num_points):
        if num_points <= 2:
            return [p1, p2]
        
        step = (p2 - p1) / (num_points - 1)
        return [p1 + step * i for i in range(num_points)]

    # Directly calculate the middle points
    middle_points = []
    for i in range(num_middle_points):
        top_point = interpolate_points(top_left, top_right, num_middle_points)[i]
        bottom_point = interpolate_points(bottom_left, bottom_right, num_middle_points)[i]
        middle_point = (top_point + bottom_point) / 2
        middle_points.append(middle_point)
        mark_point(top_point,"top_point")
        mark_point(bottom_point,"bottom_point")
        mark_point(middle_point,"middle_point")

    # Ensure end points are added
    middle_points.insert(0, (top_left + bottom_left) / 2)
    middle_points.append((top_right + bottom_right) / 2)

    return middle_points

def create_segmented_middle_points(coords_list, segment_length=1):
    coords_np = np.array(coords_list)

    # Identify the points with extreme x values (leftmost and rightmost)
    leftmost_x = coords_np[:, 0].min()
    rightmost_x = coords_np[:, 0].max()

    leftmost_points = coords_np[coords_np[:, 0] == leftmost_x]
    rightmost_points = coords_np[coords_np[:, 0] == rightmost_x]

    # Identify the top and bottom points among the leftmost and rightmost points
    top_left = leftmost_points[leftmost_points[:, 1].argmax()]
    bottom_left = leftmost_points[leftmost_points[:, 1].argmin()]
    top_right = rightmost_points[rightmost_points[:, 1].argmax()]
    bottom_right = rightmost_points[rightmost_points[:, 1].argmin()]

    # Initialize the middle points list
    middle_points = []

    # Total length of the line
    total_length = rightmost_x - leftmost_x

    # Handle the case when the total length is shorter than the segment length
    if total_length <= segment_length:
        # Add a middle point between the leftmost and rightmost points
        middle_points.append((top_left + bottom_left) / 2)
        middle_points.append((top_right + bottom_right) / 2)
        return middle_points

    # Calculate the number of segments based on the segment length
    num_segments = int(np.ceil(total_length / segment_length))

    for i in range(num_segments):
        # Determine the segment boundaries
        x_min = leftmost_x + i * segment_length
        x_max = min(leftmost_x + (i + 1) * segment_length, rightmost_x)

        # Filter points in the current segment
        segment_points = coords_np[(coords_np[:, 0] >= x_min) & (coords_np[:, 0] <= x_max)]

        if len(segment_points) > 0:
            # Find the top and bottom points in this segment
            top_point = segment_points[segment_points[:, 1].argmax()]
            bottom_point = segment_points[segment_points[:, 1].argmin()]

            # Calculate the middle point
            middle_point = (top_point + bottom_point) / 2
            middle_points.append(middle_point)

    # Ensure there is a middle point at the very end
    if middle_points[-1][0] < rightmost_x:
        middle_points.append((top_right + bottom_right) / 2)

    return middle_points

def export_as_shapefile(points,points_percentage=100,epsg_value=28992):
    
    global point_cloud_name
    start_time=time.time()
    num_points=len(points)
    num_points_to_keep = math.ceil(num_points * (points_percentage/100))
    step = math.ceil(num_points / num_points_to_keep)
    points = points[::step]
    
    print("exporting as shapefile using ",points_percentage," percent of points: ",len(points)," points")
    point_geometries = [Point(x, y, z) for x, y, z in points]
    crs = 'EPSG:' + str(epsg_value)
    gdf = gpd.GeoDataFrame(geometry=point_geometries, crs=crs)
    print("Exported as a shapefile in: ",time.time()-start_time)
    
    # Get the directory of the current Blender file
    blend_file_path = bpy.data.filepath
    directory = os.path.dirname(blend_file_path)

    # Create a folder 'road_mark_images' if it doesn't exist
    shapefile_dir = os.path.join(directory, 'shapefiles')
    if not os.path.exists(shapefile_dir):
        os.makedirs(shapefile_dir)
    # Define the path for the output shapefile
    output_shapefile_path = os.path.join(shapefile_dir, f"{point_cloud_name}_shapefile")
    gdf.to_file(output_shapefile_path)
    print("saved shapefile to: ",shapefile_dir," in: ",time.time()-start_time)
         
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
   
def save_as_json_pretty(points, colors):
    # Convert data to a list of dictionaries
    point_cloud_data = []
    for point, color in zip(points, colors):
        point_data = {
            'x': point[0],
            'y': point[1],
            'z': point[2],
            'color': {
                'r': color[0],
                'g': color[1],
                'b': color[2]
            }
        }
        point_cloud_data.append(point_data)

    # Convert to JSON
    json_data = json.dumps(point_cloud_data)

    # Save to a JSON file
    with open('point_cloud_data.json', 'w') as file:
        file.write(json_data)
        
