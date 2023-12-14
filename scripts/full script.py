#list of libraries to install
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
from bpy import context
from bpy_extras.view3d_utils import region_2d_to_location_3d
import gpu
import bmesh
from bpy.props import StringProperty, IntProperty, FloatProperty, BoolProperty
from gpu_extras.batch import batch_for_shader
from bpy_extras import view3d_utils
import numpy as np
import open3d as o3d
import bgl
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
from bl_ui.space_toolsystem_common import ToolSelectPanelHelper
from collections import deque
from multiprocessing import Pool
from joblib import dump, load 
import json
import cv2
from sklearn.cluster import DBSCAN 
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from shapely.geometry import Point

#Global variables 
point_coords = None #The coordinates of the point cloud, stored as a numpy array
point_colors = None #The colors of the point cloud, stored as a numpy array
original_coords = None #The original coords of the point cloud, before shifting
points_kdtree = None #The loaded ckdtree of coords that all functions can access
collection_name = "Collection" #the default collection name in blender
point_cloud_name= None #Used when storing files related to current point cloud
point_cloud_point_size =  1 #The size of the points in the point cloud
shape_counter=1 #Keeps track of amount of shapes drawn, used to number them
auto_las_file_path =os.path.dirname(bpy.data.filepath)+'/auto.laz' #path  for a laz file name auto.laz
use_pickled_kdtree=True #compress files to save disk space
save_json=False #generate a json file of point cloud data

#Keeps track of all objects created/removed for undo/redo functions
undo_stack = []
redo_stack = []

#Global variable to keep track of the last processed index, for numbering road marks
last_processed_index = 0

#Global variable to keep track of the active operator
active_operator = None

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
        
        if(z_height_cut_off>0):
            #Filter points with Z coordinate > 0.5
            print("Number of points before filtering:", len(points_a))
            mask = points_a[:, 2] <= (road_base_level+z_height_cut_off)
            points_a = points_a[mask]
            colors_a = colors_a[mask]
            print("Number of points after filtering:", len(points_a))
        
        #Store the original coordinates 
        original_coords = np.copy(points_a)
               
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
        original_coords = np.load(os.path.join(stored_data_path, file_name_avg_coords))
        
    #Store point data and colors globally
    point_coords = points_a
    point_colors = colors_a
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
        #this function creates more read able json files, but is slower 
        #save_as_json(point_coords,point_colors)

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
            points_kdtree = cKDTree(np.array(point_coords))
            save_kdtree_pickle_gzip(kdtree_pickle_path, points_kdtree)
            print("Compressed cKD-tree saved at:", kdtree_pickle_path)  
        else:
            print("kdtree found, loading..")
            points_kdtree = load_kdtree_pickle_gzip(kdtree_pickle_path)
            print("Compressed cKD-tree loaded from gzip file in:", time.time() - start_time)
    else:  
        #KDTree handling
        kdtree_path = os.path.join(stored_data_path, file_name_kdtree)
        points_kdtree = load_kdtree_from_file(kdtree_path)
        if not os.path.exists(kdtree_pickle_path):
            #create the kdtree if it doesn't exist
            points_kdtree = cKDTree(np.array(point_coords))
            print("kdtree created in: ", time.time() - start_time)
            #Save the kdtree to a file
            save_kdtree_to_file(kdtree_path, points_kdtree)
            print("kdtree saved in: ", time.time() - start_time, "at", kdtree_path)
         
    try: 
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
            min_coords = np.min(point_coords, axis=0)
            max_coords = np.max(point_coords, axis=0)
            bbox_center = (min_coords + max_coords) / 2
            
            #Get the active 3D view
            for area in bpy.context.screen.areas:
                if area.type == 'VIEW_3D':
                    break
                    
            #Set the view to look at the bounding box center from above at a height of 10 meters
            view3d = area.spaces[0]
            view3d.region_3d.view_location = (bbox_center[0], bbox_center[1], 10)  #X, Y, 10 meters height
            #view3d.region_3d.view_rotation = bpy.context.scene.camera.rotation_euler  #Maintaining the current rotation
            view3d.region_3d.view_distance = 10  #Distance from the view point
            
            print("openGL point cloud drawn in:",time.time() - start_time,"using ",points_percentage," percent of points (",len(reduced_points),") points") 
            
        else:
            print("Draw handler already exists, skipping drawing")
    except Exception as e:
        #Handle any other exceptions that might occur
        print(f"An error occurred: {e}")     
                             
class CreatePointCloudObjectOperator(bpy.types.Operator):
    
    bl_idname = "custom.create_point_cloud_object"
    bl_label = "Create point cloud object"
    
    global point_coords, point_colors, point_cloud_point_size, collection_name
    
    #creates a blender object from the point cloud data
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
      material.use_nodes = True  #Enable material nodes

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
        #Give the mesh has loop colors
        if not mesh.vertex_colors.active:
          mesh.vertex_colors.new()

        #Iterate over the loops and assign vertex colors
        color_layer = mesh.vertex_colors.active.data
        for i, loop in enumerate(mesh.loops):
          color = colors_ar[i] if i < len(colors_ar) else (1.0, 1.0, 1.0)
          #color*=255
          color_layer[loop.index].color = color + (1.0,)

      #Assign the material to the mesh
      if mesh.materials:
        mesh.materials[0] = material
      else:
        mesh.materials.append(material)
        
      #After the object is created, store it 
      #store_object_state(obj)
      return obj 
   
    def execute(self, context):
        start_time = time.time()
        self.create_point_cloud_object(point_coords,point_colors, point_cloud_point_size, collection_name)
        print("--- %s seconds ---" % (time.time() - start_time))
        return {'FINISHED'}

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
                #Get the mouse coordinates
                x, y = event.mouse_region_x, event.mouse_region_y
                #Convert 2D mouse coordinates to 3D view coordinates
                view3d = context.space_data
                region = context.region
                region_3d = context.space_data.region_3d
                location = region_2d_to_location_3d(region, region_3d, (x, y), (0, 0, 0))

                #Get the z coordinate from 3d space
                z = location.z

                #Perform nearest-neighbor search
                radius=5
                _, nearest_indices = points_kdtree.query([location], k=radius)
                nearest_colors = [point_colors[i] for i in nearest_indices[0]]

                average_intensity = get_average_intensity(nearest_indices[0])
                #Calculate the average color
                average_color = np.mean(nearest_colors, axis=0)
                
                clicked_on_white = "Clicked on roadmark" if is_click_on_white(self, context, location) else "No roadmark detected"
                    
                print("clicked on x,y,z: ",x,y,z,"Average Color:", average_color,"Average intensity: ",average_intensity,clicked_on_white)
            else:
                return {'PASS_THROUGH'}
            
        elif event.type == 'ESC':
            return {'CANCELLED'}  #Stop the operator when ESCAPE is pressed

        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        if context.area.type == 'VIEW_3D':
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            return {'CANCELLED'}    
        
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
            coord_3d_start.z += extra_z_height  #Add to the z dimension to prevent clipping

        coord_3d_end = view3d_utils.region_2d_to_location_3d(region, region_3d, (event.mouse_region_x, event.mouse_region_y), Vector((0, 0, 0)))
        coord_3d_end.z += extra_z_height  

        #Create a new mesh object for the line
        mesh = bpy.data.meshes.new(name="Line Mesh")
        obj = bpy.data.objects.new("Thin Line", mesh)
        
        #Link it to scene
        bpy.context.collection.objects.link(obj)
        
        #Create mesh from python data
        bm = bmesh.new()

        #Add vertices
        bmesh.ops.create_vert(bm, co=coord_3d_start)
        bmesh.ops.create_vert(bm, co=coord_3d_end)

        #Add an edge between the vertices
        bm.edges.new(bm.verts)

        #Update and free bmesh to improve memory performance
        bm.to_mesh(mesh)
        bm.free()

        #Create a material for the line and set its color
        material = bpy.data.materials.new(name="Line Material")
        material.diffuse_color = marking_color
        obj.data.materials.append(material)

        self.prev_end_point = coord_3d_end
          #After the object is created, store it 
        store_object_state(obj)
        #Create a rectangle object on top of the line
        create_rectangle_line_object(coord_3d_start, coord_3d_end)
        

    def cancel(self, context):
        if context.object:
            bpy.ops.object.select_all(action='DESELECT')
            context.view_layer.objects.active = context.object
            context.object.select_set(True)
            bpy.ops.object.delete()    


#Draws simple shapes to mark road markings     
class SimpleMarkOperator(bpy.types.Operator):
    bl_idname = "view3d.mark_fast"
    bl_label = "Mark Road Markings fast"

    _is_running = False  #Class variable to check if the operator is already running
    
    def modal(self, context, event):
        global point_coords, point_colors, points_kdtree
        intensity_threshold = context.scene.intensity_threshold
        
        if event.type == 'MOUSEMOVE':  
            self.mouse_inside_view3d = is_mouse_in_3d_view(context, event)
            
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS'and self.mouse_inside_view3d:
            
            start_time = time.time()
            #Get the mouse coordinates
            x, y = event.mouse_region_x, event.mouse_region_y
            #Convert 2D mouse coordinates to 3D view coordinates
            view3d = context.space_data
            region = context.region
            region_3d = context.space_data.region_3d
            location = region_2d_to_location_3d(region, region_3d, (x, y), (0, 0, 0))

            #Get the z coordinate from 3D space
            z = location.z

            #Do a nearest-neighbor search
            num_neighbors = 16  #Number of neighbors 
            radius = 50
            _, nearest_indices = points_kdtree.query([location], k=num_neighbors)
        
            rectangle_coords = []
            
            #Get the average intensity of the nearest points
            average_intensity = get_average_intensity(nearest_indices[0])
           
             #Get the average color of the nearest points
            average_color = get_average_color(nearest_indices[0])
             
            print("average color: ", average_color,"average intensity: " ,average_intensity)
            
            #Check if the average intensity indicates a road marking (white)
            if average_intensity > intensity_threshold:
                #Region growing algorithm
                checked_indices = set()
                indices_to_check = list(nearest_indices[0])
                print("Region growing started")
                while indices_to_check:   
                    current_index = indices_to_check.pop()
                    if current_index not in checked_indices:
                        checked_indices.add(current_index)
                        intensity = np.average(point_colors[current_index]) #* 255  #grayscale
                        if intensity>intensity_threshold:
                            rectangle_coords.append(point_coords[current_index])
                            _, neighbor_indices = points_kdtree.query([point_coords[current_index]], k=radius)
                            indices_to_check.extend(neighbor_index for neighbor_index in neighbor_indices[0] if neighbor_index not in checked_indices)

                print("Region growing completed", time.time()-start_time)
                
            
            else:
                print("no road markings found")
                
            if rectangle_coords:
                #Create a single mesh for the combined  rectangles
                create_shape(rectangle_coords,shape_type="unkown")
                
        
        elif event.type == 'ESC':
            return self.cancel(context)

        return {'PASS_THROUGH'}

    
    def invoke(self, context, event):
        if SimpleMarkOperator._is_running:
            self.report({'WARNING'}, "Operator is already running")
            return {'CANCELLED'}  #Do not run the operator if it's already running

        if context.area.type == 'VIEW_3D':
            SimpleMarkOperator._is_running = True  #Set the flag to indicate the operator is running
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            return {'CANCELLED'}

    def cancel(self,context):
        SimpleMarkOperator._is_running = False  #Reset the flag when the operator is cancelled
        print("Operator was properly cancelled")  #Debug message
        return {'CANCELLED'}
        
#Draws complexer shaped road markings using many tiny squares, which then get combined          
class ComplexMarkOperator(bpy.types.Operator):
    bl_idname = "view3d.mark_complex"
    bl_label = "Mark complex Road Markings"
    _is_running = False  #Class variable to check if the operator is already running
    
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
                #Get the mouse coordinates
                x, y = event.mouse_region_x, event.mouse_region_y
                #Convert 2D mouse coordinates to 3D view coordinates
                view3d = context.space_data
                region = context.region
                region_3d = context.space_data.region_3d
                location = region_2d_to_location_3d(region, region_3d, (x, y), (0, 0, 0))

                #Get the z coordinate from 3D space
                z = location.z

                #Do a nearest-neighbor search
                num_neighbors = 16  #Number of neighbors 
                radius = 100
                _, nearest_indices = points_kdtree.query([location], k=num_neighbors)
            
                rectangle_coords = []
                
                #Get the average intensity of the nearest points
                average_intensity = get_average_intensity(nearest_indices[0])
                           
                 #Get the average color of the nearest points
                average_color = get_average_color(nearest_indices[0])
                    
                print("average color: ", average_color,"average intensity: " ,average_intensity)
                
                #Check if the average intensity indicates a road marking (white)
                if average_intensity > intensity_threshold:
                    #Region growing algorithm
                    checked_indices = set()
                    indices_to_check = list(nearest_indices[0])
                    print("Region growing started")
                    while indices_to_check:   
                        current_index = indices_to_check.pop()
                        if current_index not in checked_indices:
                            checked_indices.add(current_index)
                            intensity = np.average(point_colors[current_index]) #* 255  #grayscale
                            if intensity>intensity_threshold:
                                rectangle_coords.append(point_coords[current_index])
                                _, neighbor_indices = points_kdtree.query([point_coords[current_index]], k=radius)
                                indices_to_check.extend(neighbor_index for neighbor_index in neighbor_indices[0] if neighbor_index not in checked_indices)

                    print("Region growing completed", time.time()-start_time)
                    
                else:
                    print("no road markings found")
                clicked=False    
                
                if rectangle_coords:
                    #Create a single mesh for the combined rectangles
                    create_dots_shape(rectangle_coords,"rectangle dots shape")
                      
            elif event.type == 'ESC':
                ComplexMarkOperator._is_running = False  #Reset the flag when the operator stops
                print("Operation was cancelled")  
                return {'CANCELLED'}  #Stop when ESCAPE is pressed

            return {'PASS_THROUGH'}

    def invoke(self, context, event):
        if ComplexMarkOperator._is_running:
            self.report({'WARNING'}, "Operator is already running")
            return {'CANCELLED'}  #Do not run the operator if it's already running

        if context.area.type == 'VIEW_3D':
            ComplexMarkOperator._is_running = True  #Set the flag to indicate the operator is running
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            return {'CANCELLED'}

    def cancel(self, context):
        ComplexMarkOperator._is_running = False  #Reset the flag when the operator is cancelled
        print("Operator was properly cancelled")  #Debug message
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
        
        #Start loop from the last processed index
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
                
                #Check point count before adding to draw list
                if len(rectangle_coords) >= point_threshold:
                    all_white_object_coords.append(rectangle_coords)
                    white_objects_count += 1  #Increment counter when valid white object is found

        #Update the last processed index
        last_processed_index = idx + 1
        
        print("finished detecting, found: ", white_objects_count, "road marks in: ", time.time() - start_time)
        start_time = time.time()
        
        for white_object_coords in all_white_object_coords:
            create_dots_shape(white_object_coords,"auto road mark")
        
        print("rendered shapes in: ", time.time() - start_time)
        return {'FINISHED'}

class CurbDetectionOperator(bpy.types.Operator):
    bl_idname = "custom.curb_detection_operator"
    bl_label = "Curb Detection Operator"

    click_count = 0
    first_click_point = None
    second_click_point = None
    _is_running = False  #Class variable to check if the operator is already running
    
    def modal(self, context, event):
        global point_coords, point_colors, points_kdtree
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            self.click_count += 1

            # Convert the mouse position to 3D space
            coord_3d = view3d_utils.region_2d_to_location_3d(
                context.region, context.space_data.region_3d,
                (event.mouse_region_x, event.mouse_region_y),
                Vector((0, 0, 0))
            )

            if self.click_count == 1:
                self.first_click_point = coord_3d
                print("first click at", self.first_click_point)
            else:
                self.second_click_point = coord_3d
                print("second click at", self.second_click_point,"starting curb detection between mouseclicks")
                start_time = time.time()
                self.detect_curb_points(points_kdtree, self.first_click_point, self.second_click_point)
                self._is_running = False
                end_time = time.time()
                print(f"Detection time: {end_time - start_time} seconds.")
                # Set the first click point to the second click point for the next detection
                self.first_click_point = self.second_click_point

        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            return self.cancel(context)

        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        self.click_count = 0
        context.window_manager.modal_handler_add(self)
        if CurbDetectionOperator._is_running:
            self.report({'WARNING'}, "Operator is already running")
            return self.cancel(context)  #Do not run the operator if it's already running
        self._is_running = True
        return {'RUNNING_MODAL'}

    def detect_curb_points(self, points_kdtree, first_click_point, second_click_point):
        
        print("Starting curb detection...")  # Debug
        curb_points_indices = []

        line_direction = np.array(second_click_point) - np.array(first_click_point)
        line_length = np.linalg.norm(line_direction)
        line_direction /= line_length  # Normalize
        perp_direction = np.array([-line_direction[1], line_direction[0], 0])
        corridor_width = 0.4
        num_samples = 100 
        neighbor_search_distance = 0.2

        print(f"Line Direction: {line_direction}, Perpendicular Direction: {perp_direction}")  # Debug

        for i in range(num_samples):
            t = i / (num_samples - 1)
            sample_point = np.array(first_click_point) + t * line_direction * line_length
            if(i==1 ):
                mark_point(sample_point, "curb start", 0.1)
            if(i==num_samples-1 ):
                mark_point(sample_point, "curb end", 0.1)
            # Query KDTree for points within the corridor width around the sample point
            indices = points_kdtree.query_ball_point(sample_point, corridor_width / 2)

            for idx in indices:
                point = point_coords[idx]

                # Check neighbors to the left and right
                left_neighbor = points_kdtree.query_ball_point(point - perp_direction * neighbor_search_distance, 0.1)
                right_neighbor = points_kdtree.query_ball_point(point + perp_direction * neighbor_search_distance, 0.1)

                if not left_neighbor or not right_neighbor:
                    curb_points_indices.append(idx)

        # Extract unique indices as curb points may be found multiple times
        unique_indices = list(set(curb_points_indices))
        curb_points = [point_coords[idx] for idx in unique_indices]
        
        # Extract x, y, and z coordinates
        x_coords = [p[0] for p in curb_points]
        y_coords = [p[1] for p in curb_points]
        z_coords = [p[2] for p in curb_points]

        # Calculate the median for each coordinate
        median_x = np.median(x_coords)
        median_y = np.median(y_coords)
        median_z = np.median(z_coords)
    
        if curb_points:
            median_curb_point = Vector((median_x, median_y, median_z))
            self.draw_curb_line(first_click_point, median_curb_point, second_click_point)
            
        print(f"Total unique curb points found: {len(curb_points)}")              
        create_dots_shape(curb_points,"curb shape",False,True)
        
    def draw_curb_line(self, first_click_point, avg_curb_point, second_click_point):
        # Create a new mesh and object
        mesh = bpy.data.meshes.new(name="CurbLine")
        obj = bpy.data.objects.new("CurbLine", mesh)

        # Link the object to the scene
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        # Create a bmesh, add vertices, and create the edge
        bm = bmesh.new()
        v1 = bm.verts.new(first_click_point)
        v2 = bm.verts.new(avg_curb_point)
        v3 = bm.verts.new(second_click_point)
        bm.edges.new((v1, v2))
        bm.edges.new((v2, v3))

        # Update and free the bmesh
        bm.to_mesh(mesh)
        bm.free()
           
    def cancel(self,context):
        CurbDetectionOperator._is_running = False  #Reset the flag when the operator is cancelled
        print("Operator was properly cancelled")  #Debug message
        return {'CANCELLED'}   
     
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
            #Get the mouse coordinates
            x, y = event.mouse_region_x, event.mouse_region_y
            #Convert 2D mouse coordinates to 3D view coordinates
            view3d = context.space_data
            region = context.region
            region_3d = context.space_data.region_3d
            location = region_2d_to_location_3d(region, region_3d, (x, y), (0, 0, 0))

            #Nearest-neighbor search from the point cloud
            _, closest_indices = points_kdtree.query([location], k=20)
            closest_point = point_coords[closest_indices[0][0]]  #get the closest point

            self.region_corners.append(closest_point)  #store the point cloud coordinate
            self.click_count += 1
            if self.click_count >= 2:
                #Find and visualize white objects within the specified region
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
            #Reset the variables when the operator is run
            self.click_count = 0
            self.region_corners = []
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            return {'CANCELLED'}
    
    def find_white_objects_within_region(self):
       
        global point_coords, point_colors, points_kdtree,intensity_threshold

        #Define bounding box limits
        min_corner = np.min(self.region_corners, axis=0)
        max_corner = np.max(self.region_corners, axis=0)
        print("rectangle drawn within these 2 points: ",min_corner, max_corner)
        #Create a bounding box for visualization
        self.create_bounding_box(min_corner, max_corner)
        #Filter points based on bounding box
        within_bbox = np.all(np.logical_and(min_corner <= point_coords, point_coords <= max_corner), axis=1)
        filtered_points = point_coords[within_bbox]
        filtered_colors = point_colors[within_bbox]
        filtered_kdtree = cKDTree(filtered_points)
        
        print("Number of points in the bounding box:", len(filtered_points))
        
        #Parameters
        point_threshold = 100
        radius = 100
        max_white_objects = 100
        intensity_threshold=intensity_threshold
        #Intensity calculation
        intensities = np.mean(filtered_colors, axis=1) #* 255  
        checked_indices = set()
        all_white_object_coords = []
        white_objects_count = 0 

        for idx, intensity in enumerate(intensities):
            if white_objects_count >= max_white_objects:
                break
            
            if idx in checked_indices or intensity <= intensity_threshold:
                continue

            #Region growing algorithm
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

            #Check point count before adding to draw list
            if len(rectangle_coords) >= point_threshold:
                all_white_object_coords.append(rectangle_coords)
                white_objects_count += 1  #Increment counter when valid white object is found
                
        print("road marks found: ", white_objects_count)
        #Visualize detected white objects
        for white_object_coords in all_white_object_coords:
            create_dots_shape(white_object_coords,"selection road mark")  
            
    #Creates and draws the selection rectangle in the viewport         
    def create_bounding_box(self, min_corner, max_corner):
        #Create a new mesh
        mesh = bpy.data.meshes.new(name="BoundingBox")
        obj = bpy.data.objects.new("BoundingBox", mesh)

        #Link it to scene
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        #Construct bounding box vertices and edges
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

        #Create mesh
        mesh.from_pydata(verts, edges, [])
        mesh.update()
        
        
        #After the object is created, store it 
        store_object_state(obj)
        
        TriangleMarkOperator ._is_running = False  #Reset the flag when the operator is cancelled
        print("Operator was properly cancelled")  #Debug message
        return {'CANCELLED'}

class AutoTriangleMarkOperator(bpy.types.Operator):
    bl_idname = "custom.auto_mark_triangle"
    bl_label = "Auto Mark Triangle"
    
    _is_running = False  #Class variable to check if the operator is already running
    _triangles = []  #List to store the triangle vertices
    _simulated_clicks = 0  #Count of simulated clicks
    _found_triangles = 0   #Count of triangles found
    _processed_indices = set()
                
    def modal(self, context, event):
        global point_coords, point_colors, points_kdtree
        intensity_threshold = context.scene.intensity_threshold
        
        if event.type == 'MOUSEMOVE':  
            self.mouse_inside_view3d = is_mouse_in_3d_view(context, event)
            
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS'and self.mouse_inside_view3d:
            
            start_time = time.time()
            #Get the mouse coordinates
            x, y = event.mouse_region_x, event.mouse_region_y
            #Convert 2D mouse coordinates to 3D view coordinates
            view3d = context.space_data
            region = context.region
            region_3d = context.space_data.region_3d
            location = region_2d_to_location_3d(region, region_3d, (x, y), (0, 0, 0))

            #Get the z coordinate from 3D space
            z = location.z

            #Do a nearest-neighbor search
            num_neighbors = 16  #Number of neighbors 
            radius = 50
            _, nearest_indices = points_kdtree.query([location], k=num_neighbors)
        
            triangle_coords = []
            
            #Get the average intensity of the nearest points
            average_intensity = get_average_intensity(nearest_indices[0])
           
             #Get the average color of the nearest points
            average_color = get_average_color(nearest_indices[0])
             
            print("average color: ", average_color,"average intensity: " ,average_intensity)
            
            #Check if the average intensity indicates a road marking (white)
            if average_intensity > intensity_threshold:
                #Region growing algorithm
                checked_indices = set()
                indices_to_check = list(nearest_indices[0])
                print("Region growing started")
                while indices_to_check:   
                    current_index = indices_to_check.pop()
                    if current_index not in checked_indices:
                        checked_indices.add(current_index)
                        intensity = np.average(point_colors[current_index]) #* 255  #grayscale
                        if intensity>intensity_threshold:
                            triangle_coords.append(point_coords[current_index])
                            _, neighbor_indices = points_kdtree.query([point_coords[current_index]], k=radius)
                            indices_to_check.extend(neighbor_index for neighbor_index in neighbor_indices[0] if neighbor_index not in checked_indices)

                print("Region growing completed", time.time()-start_time)
         
            else:
                print("no road markings found")
                
            if triangle_coords:
                
                filtered_triangle_coords=filter_noise_with_dbscan(triangle_coords)
                self._processed_indices.update(checked_indices)
                triangle_vertices = create_flexible_triangle(filtered_triangle_coords)
                self._triangles.append(triangle_vertices)
                create_shape(filtered_triangle_coords, shape_type="triangle", vertices=triangle_vertices)

                if(len(self._triangles)) >= 2:
                    outer_corners= self.find_closest_corners(self._triangles[0], self._triangles[1])
                    self.perform_automatic_marking(context, intensity_threshold, outer_corners)
                    
        elif event.type == 'ESC':
            self.cancel(context)
            return {'CANCELLED'}  #Stop when ESCAPE is pressed
        return {'RUNNING_MODAL'}
    
   
    def find_closest_corners(self, triangle1, triangle2):
        # Find the closest corners between two flat triangles
        min_distance = float('inf')
        closest_corners = (None, None)

        for corner1 in triangle1:
            for corner2 in triangle2:
                distance = np.linalg.norm(np.array(corner1[:2]) - np.array(corner2[:2]))  # Only consider X and Y
                if distance < min_distance:
                    min_distance = distance
                    closest_corners = (corner1, corner2)

        return closest_corners

    
    def perform_automatic_marking(self, context, intensity_threshold, outer_corners):
        line_points = []
        # Calculate centers before popping the final triangle
        centers = [np.mean(triangle, axis=0) for triangle in self._triangles]

        # This is the second user click, mark it as the final triangle
        final_triangle = self._triangles.pop()

        # Automatically mark the triangles in between
        middle_points = self.interpolate_line(centers[0], centers[1])
        for point in middle_points:
            self.simulate_click_and_grow(point, context, intensity_threshold, outer_corners)

        # Add the final triangle back to the end of the list
        self._triangles.append(final_triangle)

        # Create segmented lines between the bottom corners of each triangle
        for i in range(1, len(self._triangles)):
            prev_triangle = self._triangles[i - 1]
            current_triangle = self._triangles[i]
  
            closest_corners = self.find_closest_corners(prev_triangle, current_triangle)
            line_points.extend(closest_corners)
            create_polyline("segment_" + str(i), closest_corners)

        if line_points:
            create_polyline("continuous_line", line_points) #create 1 line of out all the segments
        
    def find_bottom_corners(self, triangle, previous_triangle=None, next_triangle=None):
    
        if not previous_triangle and not next_triangle:
            # If there are no neighbors, return any two points as bottom corners
            return triangle[:2]

        bottom_corners = []
        if previous_triangle:
            closest_point = min(triangle, key=lambda pt: min(np.linalg.norm(np.array(pt) - np.array(prev_pt)) for prev_pt in previous_triangle))
            bottom_corners.append(closest_point)

        if next_triangle:
            closest_point = min(triangle, key=lambda pt: min(np.linalg.norm(np.array(pt) - np.array(next_pt)) for next_pt in next_triangle))
            if closest_point not in bottom_corners:
                bottom_corners.append(closest_point)

        # If only one bottom corner was found (e.g., for the first or last triangle)
        if len(bottom_corners) < 2:
            # Add the farthest corner in the triangle from the first bottom corner
            farthest_point = max(triangle, key=lambda pt: np.linalg.norm(np.array(pt) - np.array(bottom_corners[0])))
            bottom_corners.append(farthest_point)

        return bottom_corners
            
    def simulate_click_and_grow(self, location, context, intensity_threshold, outer_corners):
        global point_coords, point_colors, points_kdtree

        _, nearest_indices = points_kdtree.query([location], k=16)
        average_intensity = get_average_intensity(nearest_indices[0])
        average_color = get_average_color(nearest_indices[0])

        if (average_intensity > intensity_threshold) and not self._processed_indices.intersection(nearest_indices[0]):
            #Proceed only if the intensity is above the threshold and the area hasn't been processed yet
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
                #move_triangle_to_line(triangle_vertices, outer_corners[0], outer_corners[1])
                #create_shape(filtered_points, shape_type="triangle", vertices=triangle_vertices)
                create_shape(filtered_points, shape_type="flexible triangle", vertices=triangle_vertices)
                
    def interpolate_line(self, start, end, num_points=50):
        #Generate points along the line between start and end
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
            AutoTriangleMarkOperator ._is_running = True  #Set the flag to indicate the operator is running
            AutoTriangleMarkOperator._found_triangles = 0
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            return {'CANCELLED'}

    def cancel(self, context):
        AutoTriangleMarkOperator ._is_running = False  #Reset the flag when the operator is cancelled
        print("Operator was properly cancelled")  #Debug message
        return {'CANCELLED'}

class TriangleMarkOperator(bpy.types.Operator):
    bl_idname = "custom.mark_triangle"
    bl_label = "Mark Triangle"
    
    _is_running = False  #Class variable to check if the operator is already running
    _triangles = []  #List to store the triangle vertices
    _processed_indices = set()
    _last_outer_corner = None  #Initialize the last outer corner here   
         
    def modal(self, context, event):
        global point_coords, point_colors, points_kdtree
        intensity_threshold = context.scene.intensity_threshold
        extra_z_height = context.scene.extra_z_height
        if event.type == 'MOUSEMOVE':  
            self.mouse_inside_view3d = is_mouse_in_3d_view(context, event)
            
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS' and self.mouse_inside_view3d:
            #Process the mouse click
            self.process_mouse_click(context, event,intensity_threshold)

        elif event.type == 'ESC':
            self.cancel(context)
            return {'CANCELLED'}  #Stop when ESCAPE is pressed
        
        return {'RUNNING_MODAL'}

    def process_mouse_click(self, context,event, intensity_threshold):
        #Get the mouse coordinates
        x, y = event.mouse_region_x, event.mouse_region_y
        location = region_2d_to_location_3d(context.region, context.space_data.region_3d, (x, y), (0, 0, 0))
        triangle_coords=[]
        #Nearest-neighbor search
        _, nearest_indices = points_kdtree.query([location], k=16)
        average_intensity = get_average_intensity(nearest_indices[0])
        average_color = get_average_color(nearest_indices[0])
        if average_intensity > intensity_threshold:
            #Region growing algorithm
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
                        #Ensure both corners are in the correct format
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
                        
                #Convert all vertices to lists containing three coordinates
                for vertex in current_triangle_vertices:
                    if not isinstance(vertex, (list, tuple)) or len(vertex) != 3:
                        #Convert vertex to a list
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
            #Reset the state
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
    
    _is_running = False  #Class variable to check if the operator is already running
    
    def modal(self, context, event):
        global point_coords, point_colors, points_kdtree
        intensity_threshold = context.scene.intensity_threshold
        
        if event.type == 'MOUSEMOVE':  
            self.mouse_inside_view3d = is_mouse_in_3d_view(context, event)
            
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS'and self.mouse_inside_view3d:
            
            start_time = time.time()
            #Get the mouse coordinates
            x, y = event.mouse_region_x, event.mouse_region_y
            #Convert 2D mouse coordinates to 3D view coordinates
            view3d = context.space_data
            region = context.region
            region_3d = context.space_data.region_3d
            location = region_2d_to_location_3d(region, region_3d, (x, y), (0, 0, 0))

            #Get the z coordinate from 3D space
            z = location.z

            #Do a nearest-neighbor search
            num_neighbors = 16  #Number of neighbors 
            radius = 50
            _, nearest_indices = points_kdtree.query([location], k=num_neighbors)
        
            rectangle_coords = []
            
            #Get the average intensity of the nearest points
            average_intensity = get_average_intensity(nearest_indices[0])
           
             #Get the average color of the nearest points
            average_color = get_average_color(nearest_indices[0])
             
            print("average color: ", average_color,"average intensity: " ,average_intensity)
            
            #Check if the average intensity indicates a road marking (white)
            if average_intensity > intensity_threshold:
                #Region growing algorithm
                checked_indices = set()
                indices_to_check = list(nearest_indices[0])
                print("Region growing started")
                while indices_to_check:   
                    current_index = indices_to_check.pop()
                    if current_index not in checked_indices:
                        checked_indices.add(current_index)
                        intensity = np.average(point_colors[current_index]) #* 255  #grayscale
                        if intensity>intensity_threshold:
                            rectangle_coords.append(point_coords[current_index])
                            _, neighbor_indices = points_kdtree.query([point_coords[current_index]], k=radius)
                            indices_to_check.extend(neighbor_index for neighbor_index in neighbor_indices[0] if neighbor_index not in checked_indices)

                print("Region growing completed", time.time()-start_time)
                
            
            else:
                print("no road markings found")
                
            if rectangle_coords:
                #Create a single mesh for the combined  rectangles
                create_shape(rectangle_coords,shape_type="rectangle")
                
        
        elif event.type == 'ESC':
            SimpleMarkOperator._is_running = False
            print("Operation was cancelled")
            return {'CANCELLED'}  #Stop when ESCAPE is pressed

        return {'PASS_THROUGH'}

    
    def invoke(self, context, event):
        if SimpleMarkOperator._is_running:
            self.report({'WARNING'}, "Operator is already running")
            self.cancel(context)

        if context.area.type == 'VIEW_3D':
            SimpleMarkOperator._is_running = True  #Set the flag to indicate the operator is running
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            return {'CANCELLED'}

    def cancel(self, context):
        SimpleMarkOperator._is_running = False  #Reset the flag when the operator is cancelled
        print("Operator was properly cancelled")  #Debug message
        return {'CANCELLED'}

class AutoRectangleMarkOperator(bpy.types.Operator):
    bl_idname = "custom.auto_mark_rectangle"
    bl_label = "Auto Mark rectangle"
    
    _is_running = False  #Class variable to check if the operator is already running
    _rectangles = []  #List to store the rectangle vertices
    _simulated_clicks = 0  #Count of simulated clicks
    _found_rectangles = 0   #Count of triangles found
    _processed_indices = set()
                
    def modal(self, context, event):
        global point_coords, point_colors, points_kdtree
        intensity_threshold = context.scene.intensity_threshold
        
        if event.type == 'MOUSEMOVE':  
            self.mouse_inside_view3d = is_mouse_in_3d_view(context, event)
            
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS'and self.mouse_inside_view3d:
            
            start_time = time.time()
            #Get the mouse coordinates
            x, y = event.mouse_region_x, event.mouse_region_y
            #Convert 2D mouse coordinates to 3D view coordinates
            view3d = context.space_data
            region = context.region
            region_3d = context.space_data.region_3d
            location = region_2d_to_location_3d(region, region_3d, (x, y), (0, 0, 0))

            #Get the z coordinate from 3D space
            z = location.z

            #Do a nearest-neighbor search
            num_neighbors = 16  #Number of neighbors 
            radius = 50
            _, nearest_indices = points_kdtree.query([location], k=num_neighbors)
        
            rectangle_coords = []
            
            #Get the average intensity of the nearest points
            average_intensity = get_average_intensity(nearest_indices[0])
           
             #Get the average color of the nearest points
            average_color = get_average_color(nearest_indices[0])
             
            print("average color: ", average_color,"average intensity: " ,average_intensity)
            
            #Check if the average intensity indicates a road marking (white)
            if average_intensity > intensity_threshold:
                #Region growing algorithm
                checked_indices = set()
                indices_to_check = list(nearest_indices[0])
                print("Region growing started")
                while indices_to_check:   
                    current_index = indices_to_check.pop()
                    if current_index not in checked_indices:
                        checked_indices.add(current_index)
                        intensity = np.average(point_colors[current_index]) #* 255  #grayscale
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
            return {'CANCELLED'}  #Stop when ESCAPE is pressed
        
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
            #Proceed only if the intensity is above the threshold and the area hasn't been processed yet
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
        #Generate points along the line between start and end
        return [start + t * (end - start) for t in np.linspace(0, 1, num_points)]


    
    def invoke(self, context, event):
        if AutoRectangleMarkOperator ._is_running:
            self.report({'WARNING'}, "Operator is already running")
            self.cancel(context)

        if context.area.type == 'VIEW_3D':
            AutoRectangleMarkOperator ._is_running = True  #Set the flag to indicate the operator is running
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            return {'CANCELLED'}

    def cancel(self, context):
        AutoRectangleMarkOperator ._is_running = False  #Reset the flag when the operator is cancelled
        print("Operator was properly cancelled")  #Debug message
        return {'CANCELLED'}
        
class SnappingLineMarkOperator(bpy.types.Operator):
    bl_idname = "custom.mark_snapping_line"
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

        if SnappingLineMarkOperator._is_running:
            self.report({'WARNING'}, "Operator is already running")
            self.cancel(context)
            return {'CANCELLED'}
        else:
            self.prev_end_point = None
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
 
    def cancel(self, context):
        
        if context.object:
            bpy.ops.object.select_all(action='DESELECT')
            context.view_layer.objects.active = context.object
            context.object.select_set(True)
            bpy.ops.object.delete()  
            
        SnappingLineMarkOperator._is_running = False
        print("Operator was properly cancelled")
        return {'CANCELLED'}
    
def draw_line(self, context, event):
    if not hasattr(self, 'click_counter'):
        self.click_counter = 0

    snap_to_road_mark = context.scene.snap_to_road_mark
    extra_z_height = context.scene.extra_z_height
    
    view3d = context.space_data
    region = context.region
    region_3d = context.space_data.region_3d
    
    # Convert the mouse position to a 3D location for the end point of the line
    coord_3d_end = view3d_utils.region_2d_to_location_3d(region, region_3d, (event.mouse_region_x, event.mouse_region_y), Vector((0, 0, 0)))
    coord_3d_end.z += extra_z_height

    self.click_counter += 1

    # Check if the current click is on a white road mark
    on_white = is_click_on_white(self, context, coord_3d_end)

    if snap_to_road_mark and self.click_counter > 1:
        # Calculate the direction vector
        direction_vector = (self.prev_end_point - coord_3d_end).normalized()
        search_range = 0.5

        # Find the center of the cluster near the second click point
        cluster_center = find_cluster_center(context, coord_3d_end, direction_vector, search_range)
        if cluster_center is not None:
            coord_3d_end = cluster_center  # Move the second click point to the cluster center

    # Create or update the line
    if self.prev_end_point is not None:
        create_rectangle_line_object(self.prev_end_point, coord_3d_end)

    self.prev_end_point = coord_3d_end  # Update the previous end point


def find_cluster_center(context, click_point, direction, range):
    intensity_threshold = context.scene.intensity_threshold
    global point_coords, point_colors, points_kdtree

    # Define the search bounds and find points within the bounds
    upper_bound = click_point + direction * range
    lower_bound = click_point - direction * range
    indices = points_kdtree.query_ball_point([upper_bound, lower_bound], range)
    indices = [i for sublist in indices for i in sublist]
    potential_points = np.array(point_coords)[indices]
    high_intensity_points = potential_points[np.average(point_colors[indices], axis=1) > intensity_threshold]

    if len(high_intensity_points) > 0:
        # Find the extremal points
        min_x = np.min(high_intensity_points[:, 0])
        max_x = np.max(high_intensity_points[:, 0])
        min_y = np.min(high_intensity_points[:, 1])
        max_y = np.max(high_intensity_points[:, 1])

        # Calculate the center of these extremal points
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        center_z = np.mean(high_intensity_points[:, 2])  # Average Z value
  
        mark_point(Vector((center_x, center_y, center_z)))
        return Vector((center_x, center_y, center_z))

    return None
  
# Finds a high-intensity points cluster near the click point and calculate its center.  
def find_cluster_points(context, click_point, direction, range):
    intensity_threshold = context.scene.intensity_threshold
    global point_coords, point_colors, points_kdtree

    # Define the search bounds
    upper_bound = click_point + direction * range
    lower_bound = click_point - direction * range

    # Find all points within the search bounds
    indices = points_kdtree.query_ball_point([upper_bound, lower_bound], range)
    indices = [i for sublist in indices for i in sublist]  # Flatten the list
    potential_points = np.array(point_coords)[indices]
    high_intensity_points = potential_points[np.average(point_colors[indices], axis=1) > intensity_threshold]

    # Limit to a certain number of points for performance
    if len(high_intensity_points) > 0:
        return high_intensity_points[:500]

    return None


#Custom operator for the pop-up dialog
class CorrectionPopUpOperator(bpy.types.Operator):
    bl_idname = "wm.correction_pop_up"
    bl_label = "Confirm correction pop up"

    start_point: bpy.props.FloatVectorProperty()
    end_point: bpy.props.FloatVectorProperty()
    click_to_correct: bpy.props.StringProperty()
    
    action: bpy.props.EnumProperty(
        items=[
            ('CONTINUE', "Draw Line", "Continue drawing the line"),
            ('STOP', "Stop", "stop the drawing"),
        ],
        default='STOP',
    )
    #Define the custom draw method
    def draw(self, context):
        layout = self.layout
        col = layout.column()
        
        #Add custom buttons to the UI
        col.label(text="Continue drawing the line?")
        col.label(text="Choose an action:")
        col.separator()
        
        #Use 'props_enum' to create buttons for each enum property
        layout.props_enum(self, "action")
        
    def execute(self, context):
        #Access the stored data to perform the correction
        coord_3d_start = Vector(self.start_point)
        coord_3d_end = Vector(self.end_point)
        
        #Based on the user's choice, either draw or start a correction process
        context.scene.user_input_result = self.action
       
        if self.action == 'CONTINUE':
            print("Trying to continue the line..")
        
        elif self.action == ('STOP'):
            print("Ended line drawing")
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
            #Get the mouse coordinates
            x, y = event.mouse_region_x, event.mouse_region_y
            #Convert 2D mouse coordinates to 3D view coordinates
            view3d = context.space_data
            region = context.region
            region_3d = context.space_data.region_3d
            location = region_2d_to_location_3d(region, region_3d, (x, y), (0, 0, 0))

            #Get the z coordinate from 3D space
            z = location.z

            #Do a nearest-neighbor search
            num_neighbors = 16  #Number of neighbors 
            radius = 50
            _, nearest_indices = points_kdtree.query([location], k=num_neighbors)
        
            rectangle_coords = []
            
            #Get the average intensity of the nearest points
            average_intensity = get_average_intensity(nearest_indices[0])
           
             #Get the average color of the nearest points
            average_color = get_average_color(nearest_indices[0])
             
            print("average color: ", average_color,"average intensity: " ,average_intensity)
            
            #Check if the average intensity indicates a road marking (white)
            if average_intensity > intensity_threshold:
                #Region growing algorithm
                checked_indices = set()
                indices_to_check = list(nearest_indices[0])
                print("Region growing started")
                while indices_to_check:   
                    current_index = indices_to_check.pop()
                    if current_index not in checked_indices:
                        checked_indices.add(current_index)
                        intensity = np.average(point_colors[current_index]) #* 255  #grayscale
                        if intensity>intensity_threshold:
                            rectangle_coords.append(point_coords[current_index])
                            _, neighbor_indices = points_kdtree.query([point_coords[current_index]], k=radius)
                            indices_to_check.extend(neighbor_index for neighbor_index in neighbor_indices[0] if neighbor_index not in checked_indices)

                print("Region growing completed", time.time()-start_time)
                
            else:
                print("no road markings found")
                
            if rectangle_coords:
                #Create a single mesh for the combined  rectangles
                create_shape(rectangle_coords,shape_type="curved line")
                
        elif event.type == 'ESC':
            SimpleMarkOperator._is_running = False
            print("Operation was cancelled")
            return {'CANCELLED'}  #Stop when ESCAPE is pressed

        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        if SimpleMarkOperator._is_running:
            self.report({'WARNING'}, "Operator is already running")
            self.cancel(context)
            return {'CANCELLED'}

        if context.area.type == 'VIEW_3D':
            SimpleMarkOperator._is_running = True  #Set the flag to indicate the operator is running
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            return {'CANCELLED'}

    def cancel(self, context):
        SimpleMarkOperator._is_running = False  #Reset the flag when the operator is cancelled
        print("Operator was properly cancelled")  #Debug message
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
                #Get the mouse coordinates
                x, y = event.mouse_region_x, event.mouse_region_y
                #Convert 2D mouse coordinates to 3D view coordinates
                view3d = context.space_data
                region = context.region
                region_3d = context.space_data.region_3d
                location = region_2d_to_location_3d(region, region_3d, (x, y), (0, 0, 0))

                #Get the z coordinate from 3d space
                z = location.z
                draw_fixed_triangle(context, location, size=0.5)
          
            else:
                return {'PASS_THROUGH'}
            
        elif event.type == 'ESC':
            return {'CANCELLED'}  #Stop the operator when ESCAPE is pressed

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
                #Get the mouse coordinates
                x, y = event.mouse_region_x, event.mouse_region_y
                #Convert 2D mouse coordinates to 3D view coordinates
                view3d = context.space_data
                region = context.region
                region_3d = context.space_data.region_3d
                location = region_2d_to_location_3d(region, region_3d, (x, y), (0, 0, 0))

                #Get the z coordinate from 3d space
                z = location.z
                create_fixed_square(context, location, size=0.5)
          
            else:
                return {'PASS_THROUGH'}
            
        elif event.type == 'ESC':
            return {'CANCELLED'}  #Stop the operator when ESCAPE is pressed

        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        if context.area.type == 'VIEW_3D':
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            return {'CANCELLED'}  
        
def get_average_intensity(indices):
    #If indices is a NumPy array with more than one dimension, flatten it
    if isinstance(indices, np.ndarray) and indices.ndim > 1:
        indices = indices.flatten()

    #If indices is a scalar, convert it to a list with a single element
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
        color = point_colors[index] #* 255  #rgb
        average_color += color
    average_color /= point_amount
    return average_color

#triangle mark functions
def move_triangle_to_line(triangle, line_start, line_end):
    #Convert inputs to numpy arrays for easier calculations
    triangle_np = np.array(triangle)
    line_start_np = np.array(line_start)
    line_end_np = np.array(line_end)

    #Identify the base vertices (the two closest to the line)
    base_vertex_indices = find_base_vertices(triangle_np, line_start_np, line_end_np)
    base_vertices = triangle_np[base_vertex_indices]

    #Find the closest points on the line for the base vertices
    closest_points = [closest_point(vertex, line_start_np, line_end_np) for vertex in base_vertices]

    #Move the base vertices to the closest points on the line
    triangle_np[base_vertex_indices] = closest_points

    #Calculate the height of the triangle to reposition the third vertex
    third_vertex_index = 3 - sum(base_vertex_indices)  # indices should be 0, 1, 2
    height_vector = triangle_np[third_vertex_index] - np.mean(base_vertices, axis=0)
    triangle_np[third_vertex_index] = np.mean(closest_points, axis=0) + height_vector

    return triangle_np.tolist()

def find_base_vertices(triangle, line_start, line_end):
    distances = [np.linalg.norm(closest_point(vertex, line_start, line_end) - vertex) for vertex in triangle]
    sorted_indices = np.argsort(distances)
    return sorted_indices[:2]  #Indices of the two closest vertices

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
                
                current_triangle = [obj.data.vertices[i].co for i in range(3)]
                moved_triangle = move_triangle_to_line(current_triangle, line_start, line_end)

                #Update the vertices of the mesh
                for i, vertex in enumerate(obj.data.vertices[:3]):
                    vertex.co = moved_triangle[i]
            else:
                print(f"Object '{obj.name}' does not have enough vertices")
                                    
def create_flexible_triangle(coords):

    #Convert coords to numpy array for efficient operations
    coords_np = np.array(coords)
    
    #calculate the pairwise distances
    pairwise_distances = np.linalg.norm(coords_np[:, np.newaxis] - coords_np, axis=2)
    
    #find the two points that are the furthest apart
    max_dist_indices = np.unravel_index(np.argmax(pairwise_distances), pairwise_distances.shape)
    vertex1 = coords_np[max_dist_indices[0]]
    vertex2 = coords_np[max_dist_indices[1]]
    
    #for each point, compute its distance to the line formed by vertex1 and vertex2
    line_vector = vertex2 - vertex1
    line_vector /= np.linalg.norm(line_vector)  #normalize
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

def draw_fixed_triangle(context, location, size=1.0):
    
    extra_z_height = context.scene.extra_z_height
    #Create new mesh and object
    mesh = bpy.data.meshes.new('FixedTriangle')
    obj = bpy.data.objects.new('Fixed Triangle', mesh)

    #Link object to scene
    bpy.context.collection.objects.link(obj)
    
    #Set object location
    obj.location = (location.x, location.y, extra_z_height)

    #Create mesh data
    bm = bmesh.new()

    #Add vertices
    bm.verts.new((0, 0, 0))  #First vertex at the click location
    bm.verts.new((size, 0, 0))  #Second vertex size units along the x-axis
    bm.verts.new((size / 2, size * (3 ** 0.5) / 2, 0))  #Third vertex to form an equilateral triangle

    #Create a face
    bm.faces.new(bm.verts)

    #Write the bmesh back to the mesh
    bm.to_mesh(mesh)
    bm.free()

    #Add a material to the object
    mat = bpy.data.materials.new(name="TriangleMaterial")
    mat.diffuse_color = (1, 0, 0, 1)  #Red color with full opacity
    obj.data.materials.append(mat)   
    
def create_fixed_triangle(coords, side_length=0.5):
     #Convert coords to numpy array for efficient operations
    coords_np = np.array(coords)

    #Reference vertex (first vertex)
    vertex1 = coords_np[0]

    #Normal vector of the plane defined by the original triangle
    normal_vector = np.cross(coords_np[1] - vertex1, coords_np[2] - vertex1)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)  #Normalize the normal vector

    #Direction vector for the second vertex
    dir_vector = coords_np[1] - vertex1
    dir_vector = dir_vector / np.linalg.norm(dir_vector) * side_length

    #Calculate the position of the second vertex
    vertex2 = vertex1 + dir_vector

    #Direction vector for the third vertex
    #Use cross product to find a perpendicular vector in the plane
    perp_vector = np.cross(normal_vector, dir_vector)
    perp_vector = perp_vector / np.linalg.norm(perp_vector) * side_length

    #Angle for equilateral triangle (60 degrees)
    angle_rad = np.deg2rad(60)

    #Calculate the position of the third vertex
    vertex3 = vertex1 + np.cos(angle_rad) * dir_vector + np.sin(angle_rad) * perp_vector

    return [vertex1.tolist(), vertex2.tolist(), vertex3.tolist()]

def create_flexible_rectangle(coords):
    
    hull = ConvexHull(coords)
    vertices = np.array([coords[v] for v in hull.vertices])
    centroid = np.mean(vertices, axis=0)
    north = max(vertices, key=lambda p: p[1])
    south = min(vertices, key=lambda p: p[1])
    east = max(vertices, key=lambda p: p[0])
    west = min(vertices, key=lambda p: p[0])
    return [north, east, south, west]

def create_fixed_square(context, location, size=1.0):
    #Create new mesh and object
    mesh = bpy.data.meshes.new('FixedSquare')
    obj = bpy.data.objects.new('Fixed Square', mesh)

    #Link object to scene
    bpy.context.collection.objects.link(obj)
    
    #Set object location
    obj.location = location

    #Create mesh data
    bm = bmesh.new()

    #Add vertices for a square
    half_size = size / 2
    v1 = bm.verts.new((half_size, half_size, 0))  #Top Right
    v2 = bm.verts.new((-half_size, half_size, 0))  #Top Left
    v3 = bm.verts.new((-half_size, -half_size, 0))  #Bottom Left
    v4 = bm.verts.new((half_size, -half_size, 0))  #Bottom Right

    #Ensure lookup table is updated before we access vertices by index
    bm.verts.ensure_lookup_table()

    #Create a face
    bm.faces.new((v1, v2, v3, v4))

    #Write the bmesh back to the mesh
    bm.to_mesh(mesh)
    bm.free()

    #Add a material to the object
    mat = bpy.data.materials.new(name="SquareMaterial")
    mat.diffuse_color = (1, 0, 0, 1)  #Red color with full opacity
    obj.data.materials.append(mat)
      
def create_polyline(name, points, width=0.01, color=(1, 0, 0, 1)):
    #Create a new curve data object
    curve_data = bpy.data.curves.new(name, type='CURVE')
    curve_data.dimensions = '3D'

    #Create a new spline in the curve
    polyline = curve_data.splines.new('POLY')
    polyline.points.add(len(points) - 1)  #The new spline has no points by default, add them

    #Assign the points to the spline
    for i, point in enumerate(points):
        polyline.points[i].co = (*point, 1)

    #Create a new object with the curve
    curve_obj = bpy.data.objects.new(name, curve_data)
    bpy.context.collection.objects.link(curve_obj)

    #Set up the curve bevel for width
    curve_data.bevel_depth = width
    curve_data.bevel_resolution = 0

    #Create a new material with the given color
    mat = bpy.data.materials.new(name + "_Mat")
    mat.diffuse_color = color
    curve_obj.data.materials.append(mat)
    store_object_state(curve_obj)
    return curve_obj

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

#Define a function to create a single mesh for combined rectangles
def create_shape(coords_list, shape_type,vertices=None):
    
    start_time = time.time()
    marking_color = bpy.context.scene.marking_color 
    transparency = bpy.context.scene.marking_transparency
    line_width = context.scene.fatline_width
    shape_coords = None  #Default to original coordinates
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

    #Create a new material for the object
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
    #Calculate the direction vector and its length
    direction = end - start
    length = direction.length

    direction.normalize()

    #Calculate the rectangle's width
    orthogonal = direction.cross(Vector((0, 0, 1)))
    orthogonal.normalize()
    orthogonal *= width / 2

    #Calculate the rectangle's vertices with an increase in the z-axis by extra_z_height
    v1 = start + orthogonal + Vector((0, 0, extra_z_height))
    v2 = start - orthogonal + Vector((0, 0, extra_z_height))
    v3 = end - orthogonal + Vector((0, 0, extra_z_height))
    v4 = end + orthogonal + Vector((0, 0, extra_z_height))

    #Create a new mesh object for the rectangle
    mesh = bpy.data.meshes.new(name="Rectangle Mesh")
    obj = bpy.data.objects.new("Rectangle Line", mesh)

    #Link it to the scene
    bpy.context.collection.objects.link(obj)

    #Create mesh from python data
    bm = bmesh.new()

    #Add vertices
    bmesh.ops.create_vert(bm, co=v1)
    bmesh.ops.create_vert(bm, co=v2)
    bmesh.ops.create_vert(bm, co=v3)
    bmesh.ops.create_vert(bm, co=v4)

    #Add faces
    bm.faces.new(bm.verts)

    #Update and free bmesh to reduce memory usage
    bm.to_mesh(mesh)
    bm.free()

    #Create a material for the rectangle and set its color
    material = bpy.data.materials.new(name="Rectangle Material")
    
    #Set the color with alpha for transparency
    material.diffuse_color = (marking_color[0], marking_color[1], marking_color[2], transparency)

    #Adjust the material settings to enable transparency
    material.use_nodes = True
    material.blend_method = 'BLEND'  #Use alpha blend mode

    #Set the Principled BSDF shader's alpha value
    principled_bsdf = next(node for node in material.node_tree.nodes if node.type == 'BSDF_PRINCIPLED')
    principled_bsdf.inputs['Alpha'].default_value = transparency
    
    #Assign the material to the object
    obj.data.materials.append(material)

    #After the object is created, store it 
    store_object_state(obj)

    return obj

#Define a function to create multiple squares on top of detected points, then combines them into one shape
def create_dots_shape(coords_list,name="Dots Shape", flat_shape=True, filter_points=True):
    start_time = time.time()
    global shape_counter
    
    marking_color = context.scene.marking_color
    transparency = context.scene.marking_transparency
    extra_z_height = context.scene.extra_z_height

    # Create a new mesh and link it to the scene
    mesh = bpy.data.meshes.new("Combined Shape")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    bm = bmesh.new()

    square_size = 0.025  # Size of each square
    max_gap = 10  # Maximum gap size to fill

    coords_list.sort(key=lambda coords: (coords[0]**2 + coords[1]**2 + coords[2]**2)**0.5)
    
    if filter_points:
        #filters out bad points
        coords_list = filter_noise_with_dbscan(coords_list)
        
    for i, coords in enumerate(coords_list):
        if i > 0:
            # Calculate the distance to the previous point
            gap = ((coords[0] - coords_list[i-1][0])**2 +
                   (coords[1] - coords_list[i-1][1])**2 +
                   (coords[2] - coords_list[i-1][2])**2)**0.5
            if gap > max_gap:
                # If the gap is too large, create a new mesh for the previous group of points
                bm.to_mesh(mesh)
                bm.clear()
                bm.verts.ensure_lookup_table()

        z_coord = coords[2] if not flat_shape else extra_z_height
        # Create a square at the current point
        square_verts = [
            bm.verts.new(Vector((coords[0] - square_size / 2, coords[1] - square_size / 2, z_coord))),
            bm.verts.new(Vector((coords[0] - square_size / 2, coords[1] + square_size / 2, z_coord))),
            bm.verts.new(Vector((coords[0] + square_size / 2, coords[1] + square_size / 2, z_coord))),
            bm.verts.new(Vector((coords[0] + square_size / 2, coords[1] - square_size / 2, z_coord))),
        ]

        bm.faces.new(square_verts)

    # Create a mesh for the last group of points
    bm.to_mesh(mesh)
    bm.free()

    # Create and assign the material
    shape_material = bpy.data.materials.new(name="shape material")
    shape_material.diffuse_color = (marking_color[0], marking_color[1], marking_color[2], transparency)
    shape_material.use_nodes = True
    shape_material.blend_method = 'BLEND'
    principled_node = next(n for n in shape_material.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    principled_node.inputs['Alpha'].default_value = transparency
    obj.data.materials.append(shape_material)

    obj.color = marking_color
    shape_counter += 1

    # Store the object state
    store_object_state(obj)

    print(f"Dots shape created in {time.time() - start_time} seconds.")
    return obj

    
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

#function to filter bad points 
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

#function to draw tiny marks on a given point
def mark_point(point, name="point", size=0.05):
    
    show_dots=context.scene.show_dots
    
    if show_dots:
        # Check if point has only 2 components (X and Y) and add a Z component of 0
        if len(point) == 2:
            point = (point[0], point[1], 0)
            
        #Create a cube to mark the point
        bpy.ops.mesh.primitive_cube_add(size=size, location=point)
        marker = bpy.context.active_object
        marker.name = name
        
        #Create a new material with the specified color
        mat = bpy.data.materials.new(name="MarkerMaterial")
        mat.diffuse_color = (1.0, 0.0, 0.0, 1.0)  #Red color
        mat.use_nodes = False  

        #Assign it to the cube
        if len(marker.data.materials):
            marker.data.materials[0] = mat
        else:
            marker.data.materials.append(mat)

        store_object_state(marker)

#function to check if a mouseclick is on a white object
def is_click_on_white(self, context, location, neighbors=5):
    global points_kdtree, point_colors
    intensity_threshold = context.scene.intensity_threshold

    #Define the number of nearest neighbors to search for
    num_neighbors = neighbors
    
    #Use the k-d tree to find the nearest points to the click location
    _, nearest_indices = points_kdtree.query([location], k=num_neighbors)
    
    average_intensity=get_average_intensity(nearest_indices)

    print(average_intensity)

    #If the average intensity is above the threshold, return True (click is on a "white" object)
    if average_intensity > intensity_threshold:
        return True
    else:
        print("Intensity threshold not met")
        return False
    
def create_triangle_outline(vertices):
    #Create a new mesh and object for the triangle outline
    mesh = bpy.data.meshes.new(name="TriangleOutline")
    obj = bpy.data.objects.new("Triangle Outline", mesh)

    #Link the object to the scene
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    #Define edges for the triangle outline
    edges = [(0, 1), (1, 2), (2, 0)]

    #Create the mesh data
    mesh.from_pydata(vertices, edges, [])  #No faces
    mesh.update()

    #Ensure the object scale is applied
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    
    #wireframe thickness, does not work
    '''wireframe_modifier = obj.modifiers.new(name="Wireframe", type='WIREFRAME')
    wireframe_modifier.thickness = 0.1 #Adjust this value for desired thickness'''
    
    store_object_state(obj)
    
    return obj

#function to calculate middle points of a line
def create_middle_points(coords_list, num_segments=10):
    coords_np = np.array(coords_list)

    #Identify the points with extreme x values (leftmost and rightmost)
    leftmost_x = coords_np[:, 0].min()
    rightmost_x = coords_np[:, 0].max()

    leftmost_points = coords_np[coords_np[:, 0] == leftmost_x]
    rightmost_points = coords_np[coords_np[:, 0] == rightmost_x]

    #Identify the top and bottom points among the leftmost and rightmost points
    top_left = leftmost_points[leftmost_points[:, 1].argmax()]
    bottom_left = leftmost_points[leftmost_points[:, 1].argmin()]
    top_right = rightmost_points[rightmost_points[:, 1].argmax()]
    bottom_right = rightmost_points[rightmost_points[:, 1].argmin()]

    #Initialize the middle points list with the leftmost middle point
    middle_points = [(top_left + bottom_left) / 2]

    #Divide the remaining line into segments
    segment_width = (rightmost_x - leftmost_x) / (num_segments - 1)

    for i in range(1, num_segments):
        #Determine the segment boundaries
        x_min = leftmost_x + i * segment_width
        x_max = leftmost_x + (i + 1) * segment_width

        #Filter points in the current segment
        segment_points = coords_np[(coords_np[:, 0] >= x_min) & (coords_np[:, 0] < x_max)]

        if len(segment_points) > 0:
            #Find the top and bottom points in this segment
            top_point = segment_points[segment_points[:, 1].argmax()]
            bottom_point = segment_points[segment_points[:, 1].argmin()]

            #Calculate the middle point
            middle_point = (top_point + bottom_point) / 2
            middle_points.append(middle_point)
            mark_point(middle_point,"middle_point")
            
    #Add the rightmost middle point at the end
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
       
        #Find and remove the object with the name "Point Cloud Object"
        for obj in bpy.context.scene.objects:
            if "Point Cloud"in obj.name:
                bpy.data.objects.remove(obj)
                break
      
        return {'FINISHED'}
 
class LAS_OT_OpenOperator(bpy.types.Operator):
    
    bl_idname = "wm.las_open"
    bl_label = "Open LAS/LAZ File"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    
    
    def execute(self, context):
        start_time = time.time()
        redraw_viewport()
        bpy.context.scene["Filepath to the loaded pointcloud"] = self.filepath
        sparsity_value = bpy.context.scene.sparsity_value
        point_size = bpy.context.scene.point_size
        pointcloud_load_optimized(self.filepath, point_size, sparsity_value)
        print("Opened LAS/LAZ file: ", self.filepath,"in %s seconds" % (time.time() - start_time))
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
        

        if context.object:
            bpy.ops.object.select_all(action='DESELECT')
            context.view_layer.objects.active = context.object
            context.object.select_set(True)
            bpy.ops.object.delete()
    print("saved kdtree to",file_path)

def store_object_state(obj):

    #bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.objects.active = obj  #Set as active object
    bpy.ops.object.select_all(action='DESELECT')  #Deselect all objects
    obj.select_set(True)  #Select the current object

    set_origin_to_geometry_center(obj)

    save_shape_as_image(obj)
    #Storing object state
    obj_state = {
        'name': obj.name,
        'location': obj.location.copy(),
        'rotation': obj.rotation_euler.copy(),
        'scale': obj.scale.copy(),
        'mesh': obj.data.copy() 
    }
    
    undo_stack.append(obj_state) 
    #Clear the redo stack
    redo_stack.clear()     

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

#operator to center the point cloud in the viewport
class CenterPointCloudOperator(bpy.types.Operator):
    bl_idname = "custom.center_pointcloud"
    bl_label = "Center the Point Cloud in Viewport"

    def execute(self, context):
       
        global point_coords

        #Calculate the bounding box of the point cloud
        min_coords = np.min(point_coords, axis=0)
        max_coords = np.max(point_coords, axis=0)
        bbox_center = (min_coords + max_coords) / 2

        #Get the active 3D view
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                break
        else:
            self.report({'WARNING'}, "No 3D View found")
            return {'CANCELLED'}

        #Set the view to look at the bounding box center from above at a height of 10 meters
        view3d = area.spaces[0]
        view3d.region_3d.view_location = (bbox_center[0], bbox_center[1], 10)  #X, Y, 10 meters height
        #view3d.region_3d.view_rotation = bpy.context.scene.camera.rotation_euler  #Maintaining the current rotation
        view3d.region_3d.view_distance = 10  #Distance from the view point

        return {'FINISHED'}

#exports point cloud as shp file
class ExportToShapeFileOperator(bpy.types.Operator):
    bl_idname = "custom.export_to_shapefile"
    bl_label = "Export to Shapefile"
    bl_description = "Export the current point cloud to a shapefile"

    def execute(self, context):
        global point_coords
        points_percentage=context.scene.points_percentage
        #Call the function to export the point cloud data to a shapefile
        export_as_shapefile(point_coords,points_percentage)

        #Return {'FINISHED'} to indicate that the operation was successful
        return {'FINISHED'}
                 
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

            #Ensure the object is updated in the 3D view
            obj.update_tag()
            
        return {'FINISHED'}
    
#Panel for the Road Marking Digitizer
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
        
        row = layout.row()
        row.prop(scene, "snap_to_road_mark")
        row.prop(scene, "fatline_width")
        layout.operator("custom.mark_snapping_line", text="line marker") 
        layout.operator("custom.auto_curved_line", text="auto curved line") 
        layout.operator("custom.curb_detection_operator", text="curb detection") 
        
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
        
         #Dummy space
        for _ in range(5): 
            layout.label(text="")
            
#Register the operators and panel
def register():
    bpy.utils.register_class(LAS_OT_OpenOperator)
    bpy.utils.register_class(LAS_OT_AutoOpenOperator)
    bpy.utils.register_class(CreatePointCloudObjectOperator)
    bpy.utils.register_class(DrawStraightFatLineOperator)
    bpy.utils.register_class(RemoveAllMarkingsOperator)
    bpy.utils.register_class(DIGITIZE_PT_Panel)
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
    bpy.utils.register_class(SnappingLineMarkOperator)
    bpy.utils.register_class(CurbDetectionOperator)
    bpy.utils.register_class(CorrectionPopUpOperator)
    bpy.utils.register_class(CenterPointCloudOperator)
    bpy.utils.register_class(ExportToShapeFileOperator)
    bpy.utils.register_class(FindALlRoadMarkingsOperator)
    bpy.utils.register_class(OBJECT_OT_simple_undo)
    bpy.utils.register_class(OBJECT_OT_simple_redo)
    
    bpy.types.Scene.point_size = IntProperty(name="POINT SIZE",
                                      default=1)
    bpy.types.Scene.sparsity_value = IntProperty(name="SPARSITY VALUE",
                                      default=1)
    bpy.types.Scene.intensity_threshold = bpy.props.FloatProperty(
        name="Intensity Threshold",
        description="Minimum intensity threshold",
        default=160,  #Default value
        min=0,#Minimum value
        max=255,#Max value
        subtype='UNSIGNED'  
    )
    bpy.types.Scene.markings_threshold = bpy.props.IntProperty(
        name="Max:",
        description="Maximum markings amount for auto marker",
        default=30,  #Default value
        min=1, #Minimum value
        max=100, #Max value  
        subtype='UNSIGNED' 
    )
    bpy.types.Scene.points_percentage = bpy.props.IntProperty(
        name="Points percentage:",
        description="Percentage of points rendered",
        default=50,  #Default value
        min=1, #Minimum value
        max=100, #Max value  
        subtype='UNSIGNED' 
    )
    bpy.types.Scene.fatline_width = bpy.props.FloatProperty(
        name="Line width",
        description="Fat Line Width",
        default=0.10,
        min=0.01, max=10,  #min and max width
        subtype='NONE'     
    )
    bpy.types.Scene.marking_color = bpy.props.FloatVectorProperty(
        name="Marking Color",
        subtype='COLOR',
        description="Select Marking color",
        default=(1, 0, 0, 1),  #Default is red
        min=0.0, max=1.0,  #Colors range from 0 to 1
        size=4
        
    )
    bpy.types.Scene.marking_color = bpy.props.FloatVectorProperty(
        name="Marking Color",
        subtype='COLOR',
        description="Select Marking color",
        default=(1, 0, 0, 1),  #Default is red
        min=0.0, max=1.0,  #Colors range from 0 to 1
        size=4
    )
    bpy.types.Scene.marking_transparency = bpy.props.FloatProperty(
        name="Marking Transparency",
        description="Set the transparency for the marking (0.0 fully transparent, 1.0 fully opaque)",
        default=1,  #Default transparency is 100%
        min=0.0, max=1.0  #Transparency can range from 0.0 to 1.0
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
        name="max height",
        description="height to cut off z",
        default=0.5,
        subtype='UNSIGNED'  
    )
    bpy.types.Scene.extra_z_height = bpy.props.FloatProperty(
        name="marking z",
        description="extra height of all objects",
        default=0.1,
        subtype='UNSIGNED'  
    )
    bpy.types.Scene.snap_to_road_mark= bpy.props.BoolProperty(
        name="snap line",
        description="Snaps user drawn shape to roadmark",
        default=True,
        subtype='UNSIGNED'  
    )
                                    
def unregister():
    
    bpy.utils.unregister_class(LAS_OT_OpenOperator) 
    bpy.utils.unregister_class(LAS_OT_AutoOpenOperator)
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
    bpy.utils.unregister_class(SnappingLineMarkOperator)
    bpy.utils.unregister_class(CurbDetectionOperator)
    
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
    del bpy.types.Scene.snap_to_road_mark
    del bpy.types.Scene.z_height_cut_off
    del bpy.types.Scene.extra_z_height
    del bpy.types.Scene.points_percentage
    
    bpy.utils.unregister_class(OBJECT_OT_simple_undo)
    bpy.utils.unregister_class(OBJECT_OT_simple_redo)
                 
if __name__ == "__main__":
    register()

    if(context.scene.auto_load):
        bpy.ops.wm.las_auto_open()
        
 
#opencv functions
def save_shape_as_image(obj):
    
    obj_name=obj.name
    save_shape_checkbox = context.scene.save_shape
    if obj_name =="Thin Line":
        return
    
    if save_shape_checkbox:
        #Ensure the object exists
        if not obj:
            raise ValueError(f"Object {obj_name} not found.")
        
        #Get the directory of the current Blender file
        blend_file_path = bpy.data.filepath
        directory = os.path.dirname(blend_file_path)

        #Create a folder 'road_mark_images' if it doesn't exist
        images_dir = os.path.join(directory, 'road_mark_images')
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)


        #Set up rendering
        bpy.context.scene.render.engine = 'CYCLES'  #or 'BLENDER_EEVEE'
        bpy.context.scene.render.image_settings.file_format = 'JPEG'
        bpy.context.scene.render.resolution_x = 256
        bpy.context.scene.render.resolution_y = 256
        bpy.context.scene.render.resolution_percentage = 100

        #Set up camera
        cam = bpy.data.cameras.new("Camera")
        cam_ob = bpy.data.objects.new("Camera", cam)
        bpy.context.scene.collection.objects.link(cam_ob)
        bpy.context.scene.camera = cam_ob

        #Use orthographic camera
        cam.type = 'ORTHO'

        #Calculate the bounding box of the object
        local_bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        min_corner = Vector(local_bbox_corners[0])
        max_corner = Vector(local_bbox_corners[6])

        #Position the camera
        bbox_center = (min_corner + max_corner) / 2
        bbox_size = max_corner - min_corner
        cam_ob.location = bbox_center + Vector((0, 0, max(bbox_size.x, bbox_size.y, bbox_size.z)))

        #Adjust the orthographic scale to 75%
        cam.ortho_scale = 1.33 * max(bbox_size.x, bbox_size.y)

        #Point the camera downward
        cam_ob.rotation_euler = (0, 0, 0)

        #Set up lighting
        light = bpy.data.lights.new(name="Light", type='POINT')
        light_ob = bpy.data.objects.new(name="Light", object_data=light)
        bpy.context.scene.collection.objects.link(light_ob)
        light_ob.location = cam_ob.location + Vector((0, 0, 2))
        light.energy = 50
        light.energy += 100* max(0, cam_ob.location.z-1)
        print("light energy: ",light.energy)
        #light.energy=10
        
        #Set object material to bright white
        mat = bpy.data.materials.new(name="WhiteMaterial")
        mat.diffuse_color = (1, 1, 1, 1)  #White color
        obj.data.materials.clear()
        obj.data.materials.append(mat)

        #Set world background to black
        bpy.context.scene.world.use_nodes = True
        bpy.context.scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)  #Black color
        
        #Render and save the image with object's name
        file_path = os.path.join(images_dir, f'{obj_name}.png')
        bpy.context.scene.render.filepath = file_path
        bpy.ops.render.render(write_still=True)
        print("saved image to: ",file_path)
        #Cleanup: delete the created camera, light, and material
        bpy.data.objects.remove(cam_ob)
        bpy.data.objects.remove(light_ob)
        bpy.data.materials.remove(mat)
        print("deleted camera, light, and material")

#Opencv shape detection from points    
def detect_shape_from_points(points, from_bmesh=False, scale_factor=100):

    if from_bmesh:
        #Convert bmesh vertices to numpy array
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
    contour_image = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)  #Convert to a 3-channel image 
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 1)  #Draws contours in green 

    display_contour=False
    save_countor=True
    
    if(display_contour):
        #Display the image with contours
        cv2.imshow("Contours", contours)
        cv2.waitKey(0)  
        cv2.destroyAllWindows()
        
    if(save_countor):
        #Save the image
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
   
        
