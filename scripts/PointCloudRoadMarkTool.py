# list of libraries to install
library_list = [
    'numpy',
    'open3d',
    'laspy[laszip]',
    'scipy',
    'mathutils',
    'pandas',
    'geopandas',
    'scikit-learn',
    'joblib',
    'opencv-python'
]

#imports
import sys
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
            
# uncomment to install libraries 
#install_libraries('opencv-python')    
#uninstall_libraries('library name'):

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
from scipy.spatial import KDTree
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from mathutils import Vector
import mathutils
import pickle
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

#Global variables 
point_coords = None
point_colors = None
original_coords = None
points_kdtree = None #The loaded ckdtree of coords that all functions can access
collection_name = "Collection"
point_cloud_point_size =  1
shape_counter=1 #Keeps track of amount of shapes currently in viewport
objects_z_height=10 #Height of all markings
auto_load=True # Automatically imports a file called auto.laz on every execution 
auto_las_file_path = "C:/Users/Niels/OneDrive/stage hawarIT cloud/point clouds/auto.laz" # Add path here for a laz file name auto.laz
 
#Keeps track of all objects created/removed for undo/redo functions
undo_stack = []
redo_stack = []

#Global variable to keep track of the last processed index, for numbering road marks by different functions
last_processed_index = 0

# Global variable to keep track of the active operator
active_operator = None

#Function to load the point cloud, store it's data and draw it using openGL           
def pointcloud_load(path, point_size, sparsity_value):
    
    start_time = time.time()
    
    global point_coords, point_colors, original_coords, points_kdtree #draw_handler
   
    base_file_name = os.path.basename(path)
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
    point_colors = colors_ar
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
             
def create_point_cloud_object(points_ar, colors_ar, point_size, collection_name):
    
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
          color*=255
          color_layer[loop.index].color = color + (1.0,)

      # Assign the material to the mesh
      if mesh.materials:
        mesh.materials[0] = material
      else:
        mesh.materials.append(material)
        
      # After the object is created, store it 
      store_object_state(obj)
      return obj

class CreatePointCloudObjectOperator(bpy.types.Operator):
    
    bl_idname = "custom.create_point_cloud_object"
    bl_label = "Create point cloud object"
    
    global point_coords, point_colors, point_cloud_point_size, collection_name
     
    def execute(self, context):
        start_time = time.time()
        create_point_cloud_object(point_coords, point_colors, point_cloud_point_size, collection_name)
        print("--- %s seconds ---" % (time.time() - start_time))
        return {'FINISHED'}
    

#Defines an Operator for drawing a free straight line in the viewport using mouseclicks
class DrawStraightLineOperator(bpy.types.Operator):
    bl_idname = "custom.draw_straight_line"
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

    def draw_line(self, context, event):
        view3d = context.space_data
        region = context.region
        region_3d = view3d.region_3d
        marking_color = context.scene.marking_color  # make sure you have this defined or accessible
        global objects_z_height
       
        # Convert the 2D mouse position to a 3D coordinate
        if self.prev_end_point:
            coord_3d_start = self.prev_end_point
        else:
            coord_3d_start = view3d_utils.region_2d_to_location_3d(region, region_3d, (event.mouse_region_x, event.mouse_region_y), Vector((0, 0, 0)))
            coord_3d_start.z = objects_z_height  # Set the Z coordinate

        coord_3d_end = view3d_utils.region_2d_to_location_3d(region, region_3d, (event.mouse_region_x, event.mouse_region_y), Vector((0, 0, 0)))
        coord_3d_end.z = objects_z_height  # Set the Z coordinate

        # Create a new mesh object for the line
        mesh = bpy.data.meshes.new(name="Line Mesh")
        obj = bpy.data.objects.new("Line Object", mesh)

        # Link it to scene
        bpy.context.collection.objects.link(obj)
        
        # Create mesh from python data
        bm = bmesh.new()

        # Add vertices
        bm.verts.new(coord_3d_start)
        bm.verts.new(coord_3d_end)

        # Add an edge between the vertices
        bm.edges.new(bm.verts)

        # Update and free bmesh for memory performance
        bm.to_mesh(mesh)
        bm.free()

        # Create a material for the line and set its color
        material = bpy.data.materials.new(name="Line Material")
        material.diffuse_color = marking_color  # make sure this color is set accordingly
        obj.data.materials.append(material)

        self.prev_end_point = coord_3d_end
        
        
        # After the object is created, store it 
        store_object_state(obj)
        
    def cancel(self, context):
        if context.object:
            bpy.ops.object.select_all(action='DESELECT')
            context.view_layer.objects.active = context.object
            context.object.select_set(True)
            bpy.ops.object.delete()
            

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
        
        view3d = context.space_data
        region = context.region
        region_3d = context.space_data.region_3d

        if self.prev_end_point:
            coord_3d_start = self.prev_end_point
        else:
            coord_3d_start = view3d_utils.region_2d_to_location_3d(region, region_3d, (event.mouse_region_x, event.mouse_region_y), Vector((0, 0, 0)))
            coord_3d_start.z += objects_z_height  # Add to the z dimension to prevent clipping

        coord_3d_end = view3d_utils.region_2d_to_location_3d(region, region_3d, (event.mouse_region_x, event.mouse_region_y), Vector((0, 0, 0)))
        coord_3d_end.z += objects_z_height  

        # Create a new mesh object for the line
        mesh = bpy.data.meshes.new(name="Line Mesh")
        obj = bpy.data.objects.new("Line Object", mesh)
        
        # After the object is created, store it 
        store_object_state(obj)
        
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

        # Create a rectangle object on top of the line
        create_rectangle_object(coord_3d_start, coord_3d_end, width)
        

    def cancel(self, context):
        if context.object:
            bpy.ops.object.select_all(action='DESELECT')
            context.view_layer.objects.active = context.object
            context.object.select_set(True)
            bpy.ops.object.delete()    

#Function to create a colored, resizable line object on top of the line      
def create_rectangle_object(start, end, width):
    
        context = bpy.context
        marking_color = context.scene.marking_color
        transparency = context.scene.marking_transparency
        # Calculate the direction vector and its length
        direction = end - start
        length = direction.length

        direction.normalize()

        # Calculate the rectangle's width
        orthogonal = direction.cross(Vector((0, 0, 1)))
        orthogonal.normalize()
        orthogonal *= width / 2

        # Calculate the rectangle's vertices
        v1 = start + orthogonal
        v2 = start - orthogonal
        v3 = end - orthogonal
        v4 = end + orthogonal

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
                radius=20
                _, nearest_indices = points_kdtree.query([location], k=radius)
                nearest_colors = [point_colors[i] for i in nearest_indices[0]]

                average_intensity = get_average_intensity(nearest_indices[0])
                # Calculate the average color
                average_color = np.mean(nearest_colors, axis=0)
                average_color*=255
                
                print("clicked on x,y,z: ",x,y,z,"Average Color:", average_color,"Average intensity: ",average_intensity)
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
    total_intensity = 0.0
    point_amount= len(indices)
    for index in indices:
        intensity = np.average(point_colors[index]) * 255  # grayscale
        total_intensity += intensity
    return total_intensity / point_amount

def get_average_color(indices):
    point_amount=len(indices)
    average_color = np.zeros(3, dtype=float)
    for index in indices:
        color = point_colors[index] * 255  # rgb
        average_color += color
    average_color /= point_amount
    return average_color

#Draws simple shapes to mark road markings     
class FastMarkOperator(bpy.types.Operator):
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
            if average_intensity > intensity_threshold or np.all(average_color > 160):
                # Region growing algorithm
                checked_indices = set()
                indices_to_check = list(nearest_indices[0])
                print("Region growing started")
                while indices_to_check:   
                    current_index = indices_to_check.pop()
                    if current_index not in checked_indices:
                        checked_indices.add(current_index)
                        intensity = np.average(point_colors[current_index]) * 255  # grayscale
                        if intensity>intensity_threshold:
                            rectangle_coords.append(point_coords[current_index])
                            _, neighbor_indices = points_kdtree.query([point_coords[current_index]], k=radius)
                            indices_to_check.extend(neighbor_index for neighbor_index in neighbor_indices[0] if neighbor_index not in checked_indices)

                print("Region growing completed", time.time()-start_time)
                
            
            else:
                print("no road markings found")
                
            if rectangle_coords:
                # Create a single mesh for the combined  rectangles
                create_combined_shape(rectangle_coords)
                
        
        elif event.type == 'ESC':
            FastMarkOperator._is_running = False
            print("Operation was cancelled")
            return {'CANCELLED'}  # Stop when ESCAPE is pressed

        return {'PASS_THROUGH'}

    
    def invoke(self, context, event):
        if FastMarkOperator._is_running:
            self.report({'WARNING'}, "Operator is already running")
            return {'CANCELLED'}  # Do not run the operator if it's already running

        if context.area.type == 'VIEW_3D':
            FastMarkOperator._is_running = True  # Set the flag to indicate the operator is running
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            return {'CANCELLED'}

    def cancel(self, context):
        FastMarkOperator._is_running = False  # Reset the flag when the operator is cancelled
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
                
                #rgb treshold
                color_treshhold=160
                
                print("average color: ", average_color,"average intensity: " ,average_intensity)
                
                # Check if the average intensity indicates a road marking (white)
                if average_intensity > intensity_threshold or np.all(average_color > color_treshhold):
                    # Region growing algorithm
                    checked_indices = set()
                    indices_to_check = list(nearest_indices[0])
                    print("Region growing started")
                    while indices_to_check:   
                        current_index = indices_to_check.pop()
                        if current_index not in checked_indices:
                            checked_indices.add(current_index)
                            intensity = np.average(point_colors[current_index]) * 255  # grayscale
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
                    create_combined_dots_shape(rectangle_coords)
                      
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

            intensity = np.average(color) * 255  
            if intensity > intensity_threshold:
                rectangle_coords = []
                indices_to_check = [idx]
                while indices_to_check:
                    current_index = indices_to_check.pop()
                    if current_index not in checked_indices:
                        checked_indices.add(current_index)
                        intensity = np.average(point_colors[current_index]) * 255
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
            create_combined_dots_shape(white_object_coords)
        
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
        # Your global variables (adjust as per actual implementation)
        global point_coords, point_colors, points_kdtree 

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
        intensity_threshold = 160  
        point_threshold = 100
        radius = 100
        max_white_objects = 100

        # Intensity calculation
        intensities = np.mean(filtered_colors, axis=1) * 255  
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
                    intensity = np.average(filtered_colors[current_index]) * 255
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
            create_combined_dots_shape(white_object_coords)  
            
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
             
# Define a function to create multiple squares on top of detected points, then combines them into one shape
def create_combined_dots_shape(coords_list):
    
    start_time=time.time()
    global shape_counter, objects_z_height
    
    marking_color=context.scene.marking_color
    transparency = bpy.context.scene.marking_transparency
    
    # Create a new mesh and link it to the scene
    mesh = bpy.data.meshes.new("Combined Shape")
    obj = bpy.data.objects.new("Combined Shape", mesh)
    bpy.context.collection.objects.link(obj)

    bm = bmesh.new()

    square_size = 0.025  # Size of each square
    z_offset = objects_z_height  # Offset in Z coordinate
    max_gap = 10  # Maximum gap size to fill

    #filters out bad points
    distance_between_cords = 0.03
    required_surrounded_cords= 10
    coords_list = filter_noise_with_dbscan(coords_list, distance_between_cords, required_surrounded_cords)
    
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
        # Otherwise, add the material to the object
        obj.data.materials.append(shape_material)
        
    obj.color = marking_color  # Set viewport display color 
    shape_counter+=1
    
    #After the object is created, store it 
    store_object_state(obj)

#Define a function to create a single mesh for combined rectangles
def create_combined_shape(coords_list):
    
    start_time = time.time()
    marking_color = bpy.context.scene.marking_color 
    transparency = bpy.context.scene.marking_transparency

    # Create a new mesh and link it to the scene
    mesh = bpy.data.meshes.new("Combined Shape")
    obj = bpy.data.objects.new("Combined Shape", mesh)
    bpy.context.collection.objects.link(obj)
    
    # Create a bmesh object
    bm = bmesh.new()

    #filters out bad points
    distance_between_cords = 0.03
    required_surrounded_cords= 10
    coords_list = filter_noise_with_dbscan(coords_list, distance_between_cords, required_surrounded_cords)
    
    # Add vertices from coords_list to the bmesh 
    for coords in coords_list:
        #adjusted_coords  = (coords[0], coords[1], coords[2])  
        bm.verts.new(coords )

    # Create the convex hull using these vertices
    bmesh.ops.convex_hull(bm, input=bm.verts)

    # Extract the vertices from the bmesh for shape detection
    bm_vertices = [vert.co for vert in bm.verts]

    # Detect the shape using the bmesh vertices
    shape_type = detect_shape_from_points(bm_vertices, from_bmesh=True)
  
    # Create new coordinates for the shape based on the detection
    if shape_type == "triangle":
        new_coords_list = create_perfect_triangle(coords_list)
    elif shape_type == "rectangle":
        new_coords_list = create_perfect_rectangle(coords_list)
    else:
        new_coords_list = coords_list  # If no recognizable shape, default to original coordinates

    # Update the bmesh with the new coordinates
    bm.clear()
    for coords in new_coords_list:
        bm.verts.new(coords)
    bmesh.ops.convex_hull(bm, input=bm.verts)
    bm.to_mesh(mesh)
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
    if len(obj.data.materials) > 0:
        # If the object already has materials, replace the first one with the new material
        obj.data.materials[0] = shape_material
    else:
        # Otherwise, add the new material to the object
        obj.data.materials.append(shape_material)
    
    # After the object is created, store it 
    store_object_state(obj)
    print("rendered road mark shape in: ", time.time()-start_time)

    
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

def create_perfect_rectangle(coords):
    
    #Find the center of the points
    coords_np = np.array(coords)
    center = coords_np.mean(axis=0)

    #Find the width and height based on the furthest points
    min_vals = coords_np.min(axis=0)
    max_vals = coords_np.max(axis=0)
    width = max_vals[0] - min_vals[0]
    height = max_vals[1] - min_vals[1]

    half_width = width / 2
    half_height = height / 2

    #Create coordinates for a perfect rectangle
    rectangle_coords = [
        (center[0] - half_width, center[1] - half_height, center[2]),  # bottom left
        (center[0] + half_width, center[1] - half_height, center[2]),  # bottom right
        (center[0] + half_width, center[1] + half_height, center[2]),  # top right
        (center[0] - half_width, center[1] + half_height, center[2])   # top left
    ]

    return rectangle_coords

def create_perfect_triangle(coords):
    
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

    return triangle_coords

def filter_noise_with_dbscan(coords_list, eps=0.05, min_samples=25):
    
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

    print(f"Points have been filtered. Original amount: {original_count}, Removed: {removed_count}")

    return filtered_coords
             
#Operator to remove all drawn markings from the scene collection
class RemoveAllMarkingsOperator(bpy.types.Operator):
    bl_idname = "custom.remove_all_markings"
    bl_label = "Remove All Lines"

    def execute(self, context):
        global rshape_counter
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_by_type(type='CURVE')
        for obj in bpy.context.scene.objects:
            if "Line" in obj.name or "Combined" in obj.name or "Plane" in obj.name or "BoundingBox" in obj.name:
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
        redraw_viewport()
        start_time = time.time()
        bpy.context.scene["Filepath to the loaded pointcloud"] = self.filepath
        sparsity_value = bpy.context.scene.sparsity_value
        point_size = bpy.context.scene.point_size
        pointcloud_load(self.filepath, point_size, sparsity_value)
        print("Opening LAS file: ", self.filepath)
        print("--- %s seconds ---" % (time.time() - start_time))
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
        pointcloud_load(auto_las_file_path, point_size, sparsity_value)
        print("Finished opening LAS file:", auto_las_file_path)
        return {'FINISHED'}

# Function to load KDTree from a file
def load_kdtree_from_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            kdtree_data = json.load(file)
        # Convert the loaded points back to a Numpy array
        points = np.array(kdtree_data['points'])
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


    
def store_object_state(obj):
    
    # Storing object state
    obj_state = {
        'name': obj.name,
        'location': obj.location.copy(),
        'rotation': obj.rotation_euler.copy(),
        'scale': obj.scale.copy(),
        'mesh': obj.data.copy() 
    }
    
    undo_stack.append(obj_state) 
    # Clear the redo stack since we have a new action
    redo_stack.clear()  
    
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
        layout.operator("custom.create_point_cloud_object", text="Create Point Cloud object")
        layout.operator("custom.remove_point_cloud", text="Remove point cloud")
        layout.operator("custom.remove_all_markings", text="Remove All markings")
        
        row = layout.row(align=True)
        layout.operator("view3d.select_points", text="Get point color & intensity")
        layout.prop(scene, "intensity_threshold")
        
        row = layout.row()
        layout.prop(scene, "marking_color")
        layout.prop(scene, "marking_transparency")
        #layout.operator("custom.draw_straight_line", text="Draw Simple Line")
        
        row = layout.row(align=True)
        row.operator("view3d.line_drawer", text="Draw Line")
        row.prop(scene, "fatline_width")
 
        row = layout.row(align=True)
        layout.operator("view3d.mark_fast", text="Simple marker")
        layout.operator("view3d.mark_complex", text="Complex shape marker")
        layout.operator("view3d.selection_detection", text="Selection Marker")
        
        row = layout.row()
        row.operator("custom.find_all_road_marks", text="Auto Mark")
        row.prop(scene, "markings_threshold")
        
        row = layout.row()
        row.operator("object.simple_undo", text="Undo")
        row.operator("object.simple_redo", text="Redo")
               
         # Dummy space
        for _ in range(20):  # you can increase or decrease this number
            layout.label(text="")
            
# Register the operators and panel
def register():
    bpy.utils.register_class(LAS_OT_OpenOperator)
    bpy.utils.register_class(LAS_OT_AutoOpenOperator)
    bpy.utils.register_class(DrawStraightLineOperator)
    bpy.utils.register_class(DrawStraightFatLineOperator)
    bpy.utils.register_class(RemoveAllMarkingsOperator)
    bpy.utils.register_class(DIGITIZE_PT_Panel)
    bpy.types.Scene.point_size = IntProperty(name="POINT SIZE",
                                      default=1)
    bpy.types.Scene.sparsity_value = IntProperty(name="SPARSITY VALUE",
                                      default=1)
    bpy.utils.register_class(RemovePointCloudOperator)
    bpy.utils.register_class(GetPointsInfoOperator)
    
    bpy.utils.register_class(FastMarkOperator)
    bpy.utils.register_class(ComplexMarkOperator)
    bpy.utils.register_class(SelectionDetectionOpterator)
    bpy.utils.register_class(CreatePointCloudObjectOperator)
    
    """bpy.types.Scene.operator_is_running = bpy.props.BoolProperty(
        name="Operator Is Running",
        description="A flag to indicate whether the operator is running",
        default=False
    )"""
    
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
  
    bpy.utils.register_class(FindALlRoadMarkingsOperator)
    bpy.utils.register_class(OBJECT_OT_simple_undo)
    bpy.utils.register_class(OBJECT_OT_simple_redo)
  
                                      
def unregister():
    
    bpy.utils.unregister_class(LAS_OT_OpenOperator) 
    bpy.utils.unregister_class(DrawStraightLineOperator)
    bpy.utils.unregister_class(DrawStraightFatLineOperator)
    bpy.utils.unregister_class(RemoveAllMarkingsOperator)
    bpy.utils.unregister_class(DIGITIZE_PT_Panel)
    bpy.utils.unregister_class(RemovePointCloudOperator)
    bpy.utils.unregister_class(GetPointsInfoOperator)
    bpy.utils.unregister_class(FastMarkOperator)
    bpy.utils.unregister_class(ComplexMarkOperator)
    bpy.utils.unregister_class(SelectionDetectionOpterator)
    
    bpy.utils.unregister_class(CreatePointCloudObjectOperator)
    bpy.utils.unregister_class(LAS_OT_AutoOpenOperator)
    bpy.utils.unregister_class(FindALlRoadMarkingsOperator)
    del bpy.types.Scene.marking_transparency
    del bpy.types.Scene.marking_color
    del bpy.types.Scene.intensity_threshold
    del bpy.types.Scene.markings_threshold
    del bpy.types.Scene.fatline_width
    #del bpy.types.Scene.operator_is_running
    bpy.utils.unregister_class(OBJECT_OT_simple_undo)
    bpy.utils.unregister_class(OBJECT_OT_simple_redo)
    
def uninstall_libraries(library_list):
    for library in library_list:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', library])
            print(f"Successfully uninstall {library}")
        except subprocess.CalledProcessError as e:
            print(f"Error uninstall {library}: {e}") 
              
if __name__ == "__main__":
    register()
    
    if(auto_load):
        bpy.ops.wm.las_auto_open()
    
  