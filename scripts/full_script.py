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
    'opencv-python',
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
                          
#install_libraries("")    

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
collection_name = "Collection" #the default collection name in blender
point_cloud_name= None #Used when storing files related to current point cloud
point_cloud_point_size =  1 #The size of the points in the point cloud
shape_counter=1 #Keeps track of amount of shapes drawn, used to number them
auto_las_file_path =os.path.dirname(bpy.data.filepath)+'/auto.laz' #path  for a laz file name auto.laz
save_json=False #generate a json file of point cloud data
last_processed_index = 0 #Global variable to keep track of the last processed index, for numbering road marks




#Utility operators
#Operator to import las/laz files                         
class LAS_OT_OpenOperator(bpy.types.Operator):
    
    bl_idname = "wm.las_open"
    bl_label = "Open LAS/LAZ File"
    bl_description = "Import a LAS/LAZ file"
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    
    def execute(self, context):
        start_time = time.time()
        bpy.context.scene.auto_load = False
        bpy.context.scene["Filepath to the loaded pointcloud"] = self.filepath
        sparsity_value = bpy.context.scene.sparsity_value
        point_size = bpy.context.scene.point_size
        z_height_cut_off=bpy.context.scene.z_height_cut_off
        pointcloud_data = GetPointCloudData()
        pointcloud_data.pointcloud_load_optimized(self.filepath, point_size, sparsity_value,z_height_cut_off)
        print("Opened LAS/LAZ file: ", self.filepath,"in %s seconds" % (time.time() - start_time))
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
#Operator to remove all drawn markings from the scene collection
class RemoveAllMarkingsOperator(bpy.types.Operator):
    bl_idname = "custom.remove_all_markings"
    bl_label = "Remove All Lines"
    bl_description = "Removes all markings from the scene"
    
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

#Operator to remove the point cloud     
class RemovePointCloudOperator(bpy.types.Operator):

    bl_idname = "custom.remove_point_cloud"
    bl_label = "Remove Point Cloud"
    bl_description = "Stops rendering OpenGL point cloud"
    
    def execute(self, context):
        
        #set the rendering of the openGL point cloud to off
        redraw_viewport()
       
        #Find and remove the object with the name "Point Cloud Object"
        for obj in bpy.context.scene.objects:
            if "Point Cloud"in obj.name:
                bpy.data.objects.remove(obj)
                break
      
        return {'FINISHED'}
 
#Operator to print the point cloud coordinates and the average color & intensity around mouse click        
class GetPointsInfoOperator(bpy.types.Operator):
    bl_idname = "view3d.select_points"
    bl_label = "Get Points information"
    bl_description= "Get point info around mouse click such as coordinates, color and intensity"
    
    def modal(self, context, event):
        
        set_view_to_top(context)
        pointcloud_data = GetPointCloudData()
        point_coords = pointcloud_data.point_coords
        point_colors = pointcloud_data.point_colors
        points_kdtree=  pointcloud_data.points_kdtree
        
        if event.type == 'MOUSEMOVE':  
            self.mouse_inside_view3d = is_mouse_in_3d_view(context, event)


        if event.type == 'LEFTMOUSE' and event.value == 'PRESS'and self.mouse_inside_view3d:
            if context.area and context.area.type == 'VIEW_3D':
                
                #Get the mouse coordinates
                x, y = event.mouse_region_x, event.mouse_region_y
                #Convert 2D mouse coordinates to 3D view coordinates
                region = context.region
                region_3d = context.space_data.region_3d
                location = region_2d_to_location_3d(region, region_3d, (x, y), (0, 0, 0))

                #Get the z coordinate from 3d space
                z = location.z

                #Perform nearest-neighbor search
                radius=6
                _, nearest_indices = points_kdtree.query([location], k=radius)
                nearest_colors = [point_colors[i] for i in nearest_indices[0]]

                average_intensity = get_average_intensity(nearest_indices[0],point_colors)
                #Calculate the average color
                average_color = np.mean(nearest_colors, axis=0)
                
                clicked_on_white = "Clicked on roadmark" if is_click_on_white(self, context, location) else "No roadmark detected"
                    
                print("clicked on x,y,z: ",x,y,z,"Average intensity: ",average_intensity,clicked_on_white,"Average Color:", average_color,)
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
       
#Operator to center the point cloud in the viewport
class CenterPointCloudOperator(bpy.types.Operator):
    bl_idname = "custom.center_pointcloud"
    bl_label = "Center the Point Cloud in Viewport"
    bl_description = "Center the point cloud in the viewport"
    
    def execute(self, context):
       
        pointcloud_data = GetPointCloudData()
        point_coords = pointcloud_data.point_coords

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

        viewport_height=20
        #Set the view to look at the bounding box center from above at a height of x meters
        view3d = area.spaces[0]
        view3d.region_3d.view_location = (bbox_center[0], bbox_center[1], 10) 
        #view3d.region_3d.view_rotation = bpy.context.scene.camera.rotation_euler  #Maintaining the current rotation
        view3d.region_3d.view_distance = viewport_height  #Distance from the view point

        # Ensure there is an active camera in the scene
        if bpy.context.scene.camera:
            bpy.context.scene.camera.data.type = 'ORTHO' #Set the camera type to Orthographic, alternatively is PERSP for perspective
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                # Access the 3D View's region data
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        # Set the viewport to Orthographic projection
                        space.region_3d.view_perspective = 'ORTHO'
                        break  # Exit the loop once the first 3D view is found and set
                    
        return {'FINISHED'}

#Operator to exports point cloud as shp file
class ExportToShapeFileOperator(bpy.types.Operator):
    
    bl_idname = "custom.export_to_shapefile"
    bl_label = "Export to Shapefile"
    bl_description = "Export the current point cloud to a shapefile"

    def execute(self, context):
        
        pointcloud_data = GetPointCloudData()
        point_coords = pointcloud_data.point_coords
        points_percentage=context.scene.points_percentage
        export_as_shapefile(point_coords,points_percentage)
        return {'FINISHED'}

#Operator to create a point cloud object from the loaded point cloud    
class CreatePointCloudObjectOperator(bpy.types.Operator):
    
    bl_idname = "custom.create_point_cloud_object"
    bl_label = "Create point cloud object"
    bl_description= "Create a point cloud object from the loaded point cloud"
    
    global point_cloud_point_size, collection_name

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
      prepare_object_for_export(obj)  
      return obj 
   
    def execute(self, context):
        start_time = time.time()
        pointcloud_data = GetPointCloudData()
        point_coords=pointcloud_data.point_coords
        point_colors=pointcloud_data.point_colors
        self.create_point_cloud_object(point_coords,point_colors, point_cloud_point_size, collection_name)
        print("--- %s seconds ---" % (time.time() - start_time))
        return {'FINISHED'}

#Operator for the pop-up dialog
class PopUpOperator(bpy.types.Operator):
    bl_idname = "wm.correction_pop_up"
    bl_label = "Confirm correction pop up"
    bl_description = "Pop up to confirm"
    
    average_intensity: bpy.props.FloatProperty()
    adjust_action: bpy.props.StringProperty()
    
    action: bpy.props.EnumProperty(
        items=[
            ('CONTINUE', "Yes", "Yes"),
            ('STOP', "No", "No"),
        ],
        default='CONTINUE',
    )
    #Define the custom draw method
    def draw(self, context):
        layout = self.layout
        col = layout.column()
        
        #Add custom buttons to the UI
        if self.adjust_action=='LOWER':
            col.label(text="No road marks found. Try with lower threshold?")
        elif self.adjust_action=='HIGHER':
            col.label(text="Threshold might be too low, Try with higher threshold?")
        col.label(text="Choose an action:")
        col.separator()
        
        #Use 'props_enum' to create buttons for each enum property
        layout.props_enum(self, "action")
        
    def execute(self, context):

        #Based on the user's choice, perform the action
        context.scene.user_input_result = self.action
       
        if self.action == 'CONTINUE':
           if self.adjust_action=='LOWER':
               old_threshold=context.scene.intensity_threshold
               context.scene.intensity_threshold=self.average_intensity-20 #lower the threshold to the average intensity around the mouseclick
               print("changed intensity threshold from: ",old_threshold,"to: ",self.average_intensity," please try again")
           elif self.adjust_action=='HIGHER':
               old_threshold=context.scene.intensity_threshold
               context.scene.intensity_threshold=self.average_intensity-30 #higher the threshold to the average intensity around the mouseclick 
               print("changed intensity threshold from: ",old_threshold,"to: ",self.average_intensity," please try again")
        elif self.action == 'STOP':
            return {'CANCELLED'}
        
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)




#Digitizing operators
#Operator for drawing a free thick straight line in the viewport using mouseclicks
class DrawStraightFatLineOperator(bpy.types.Operator):
    
    bl_idname = "view3d.line_drawer"
    bl_label = "Draw Straight Line"
    bl_description= "Draw a straight line in the viewport"
    prev_end_point = None
    
    def modal(self, context, event):
       
        set_view_to_top(context)

        if event.type == 'LEFTMOUSE':
            if event.value == 'RELEASE':
                self.draw_line(context, event)
                return {'RUNNING_MODAL'}
        elif event.type == 'RIGHTMOUSE' or event.type == 'ESC':
            #If escape is pressed, stop the operator 
            return self.cancel(self, context)

        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        
        self.prev_end_point = None
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def draw_line(self, context, event):
        
        marking_color = context.scene.marking_color
        width = context.scene.fatline_width
        extra_z_height = context.scene.extra_z_height
        
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
        prepare_object_for_export(obj)
        self.prev_end_point = coord_3d_end

        #Create a rectangle object on top of the line
        create_rectangle_line_object(coord_3d_start, coord_3d_end)

    def cancel(self, context):
        DrawStraightFatLineOperator._is_running = False  #Reset the flag when the operator is cancelled
        print("Operator was properly cancelled")  #Debug message
        return {'CANCELLED'}
    
#Operator to draw simple shapes in the viewport using mouseclicks   
class SimpleMarkOperator(bpy.types.Operator):
    bl_idname = "custom.mark_fast"
    bl_label = "Mark Road Markings fast"
    bl_description= "Mark shapes with simple shapes"
    _is_running = False  #Class variable to check if the operator is already running
    
    def modal(self, context, event):
    
        set_view_to_top(context)
        pointcloud_data = GetPointCloudData()
        point_coords = pointcloud_data.point_coords
        point_colors = pointcloud_data.point_colors
        points_kdtree=  pointcloud_data.points_kdtree
        intensity_threshold = context.scene.intensity_threshold
        show_popup = context.scene.adjust_intensity_popup
        
        if event.type == 'MOUSEMOVE':  
            self.mouse_inside_view3d = is_mouse_in_3d_view(context, event)
            
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS'and self.mouse_inside_view3d:

            #Get the mouse coordinates
            x, y = event.mouse_region_x, event.mouse_region_y
            #Convert 2D mouse coordinates to 3D view coordinates
            region = context.region
            region_3d = context.space_data.region_3d
            location = region_2d_to_location_3d(region, region_3d, (x, y), (0, 0, 0))

            #Do a nearest-neighbor search
            num_neighbors = 16  #Number of neighbors 
            radius = 50
            _, nearest_indices = points_kdtree.query([location], k=num_neighbors)
        
            region_growth_coords = []
            
            #Get the average intensity of the nearest points
            average_intensity = get_average_intensity(nearest_indices[0],point_colors)
           
             #Get the average color of the nearest points
            average_color = get_average_color(nearest_indices[0], point_colors)
             
            print("average color: ", average_color,"average intensity: " ,average_intensity)
            
            intensity_difference_threshold=50
            #Check if the average intensity indicates a road marking (white)
            if average_intensity > intensity_threshold:
                #if the average intensity is way higher than the threshold, give a warning
                if(average_intensity-intensity_threshold>intensity_difference_threshold) and show_popup:
                    bpy.ops.wm.correction_pop_up('INVOKE_DEFAULT', average_intensity=average_intensity, adjust_action='HIGHER')
                    return {'PASS_THROUGH'}
                #Region growing algorithm
                region_growth_coords,checked_indices=region_growing(point_coords, point_colors, points_kdtree, nearest_indices, radius, intensity_threshold, region_growth_coords)
            else:
                print("no road markings found")
                #if the average intensity is way lower than the threshold, give a warning
                if show_popup:
                    bpy.ops.wm.correction_pop_up('INVOKE_DEFAULT', average_intensity=average_intensity, adjust_action='LOWER')
                return {'PASS_THROUGH'}
                
            if region_growth_coords:
                #Create a single mesh for the combined  rectangles
                create_shape(region_growth_coords,shape_type="unkown")
                
        elif event.type == 'ESC':
            SimpleMarkOperator._is_running = False
            print("Operation was cancelled")
            return {'CANCELLED'}  #Stop when ESCAPE is pressed

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

    def cancel(self, context):
        SimpleMarkOperator._is_running = False  #Reset the flag when the operator is cancelled
        print("Operator was properly cancelled")  #Debug message
        return {'CANCELLED'}
        
#Operator to draw complex shapes on mouseclicks using tiny squares, which then get combined into a single mesh         
class ComplexMarkOperator(bpy.types.Operator):
    bl_idname = "custom.mark_complex"
    bl_label = "Mark complex Road Markings"
    bl_description= "Mark shapes using multiple points"
    
    #External coordinates to use when calling the operator from outside
    external_x: bpy.props.FloatProperty(name="External X")
    external_y: bpy.props.FloatProperty(name="External Y")
    external_z: bpy.props.FloatProperty(name="External Z")
    
    _is_running = False  #Class variable to check if the operator is already running
    
    def modal(self, context, event):
        
        set_view_to_top(context)
        pointcloud_data = GetPointCloudData()
        point_coords = pointcloud_data.point_coords
        point_colors = pointcloud_data.point_colors
        points_kdtree=  pointcloud_data.points_kdtree
        intensity_threshold = context.scene.intensity_threshold
        show_popup = context.scene.adjust_intensity_popup
        
        if event.type == 'MOUSEMOVE':  
            self.mouse_inside_view3d = is_mouse_in_3d_view(context, event) 

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS'and self.mouse_inside_view3d:
            #Get the mouse coordinates
            x, y = event.mouse_region_x, event.mouse_region_y
            #Convert 2D mouse coordinates to 3D view coordinates
            region = context.region
            region_3d = context.space_data.region_3d
            location = region_2d_to_location_3d(region, region_3d, (x, y), (0, 0, 0))
            self.process_location( context, location, intensity_threshold,points_kdtree,point_coords,point_colors,show_popup)
                
        #If escape is pressed, stop the operator 
        elif event.type == 'ESC':
            return self.cancel(context)  

        return {'PASS_THROUGH'}

    def process_location(self, context, location, intensity_threshold,points_kdtree,point_coords,point_colors,show_popup):
       
        #Do a nearest-neighbor search
        num_neighbors = 16  #Number of neighbors 
        radius = 100
        _, nearest_indices = points_kdtree.query([location], k=num_neighbors)
    
        region_growth_coords = []
        
        #Get the average intensity of the nearest points
        average_intensity = get_average_intensity(nearest_indices[0],point_colors)
                    
            #Get the average color of the nearest points
        average_color = get_average_color(nearest_indices[0],point_colors)
            
        print("average color: ", average_color,"average intensity: " ,average_intensity)
        
        intensity_difference_threshold=50
        #Check if the average intensity indicates a road marking (white)
        if average_intensity > intensity_threshold:
            #if the average intensity is way higher than the threshold, give a warning
            if(average_intensity-intensity_threshold>intensity_difference_threshold) and show_popup:
                bpy.ops.wm.correction_pop_up('INVOKE_DEFAULT', average_intensity=average_intensity, adjust_action='HIGHER') 
                return {'PASS_THROUGH'}
            #Region growing algorithm
            region_growth_coords,checked_indices=region_growing(point_coords, point_colors, points_kdtree, nearest_indices, radius, intensity_threshold, region_growth_coords)
        else:
            print("no road markings found")
            #if the average intensity is way lower than the threshold, give a warning
            if show_popup:
                bpy.ops.wm.correction_pop_up('INVOKE_DEFAULT', average_intensity=average_intensity, adjust_action='LOWER')
            return {'PASS_THROUGH'}
        
        if region_growth_coords:
            #Create a single mesh for the combined rectangles
            create_dots_shape(region_growth_coords)
        
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
    
    def execute(self, context):
        # Use the external coordinates if provided
        if hasattr(self, 'external_x') and hasattr(self, 'external_y') and hasattr(self, 'external_z'):
            location = Vector((self.external_x, self.external_y, 0))
            
            pointcloud_data = GetPointCloudData()
            point_coords = pointcloud_data.point_coords
            point_colors = pointcloud_data.point_colors
            points_kdtree=  pointcloud_data.points_kdtree
            intensity_threshold = context.scene.intensity_threshold
            show_popup = context.scene.adjust_intensity_popup
            
            self.process_location(context, location, intensity_threshold,points_kdtree,point_coords,point_colors,show_popup)

        return {'FINISHED'}
       
#Operator to scans the entire point cloud for road markings, then mark them   
class FindALlRoadMarkingsOperator(bpy.types.Operator):
    bl_idname = "custom.find_all_road_marks"
    bl_label = "Finds all road marks"
    bl_description="Finds all road marks up to a max and marks them"

    def execute(self, context):
        global last_processed_index
        
        markings_threshold = context.scene.markings_threshold
        start_time = time.time()
        print("Start auto detecting up to",markings_threshold, "road markings.. this could take a while")
        
        pointcloud_data = GetPointCloudData()
        point_coords = pointcloud_data.point_coords
        point_colors = pointcloud_data.point_colors
        points_kdtree=  pointcloud_data.points_kdtree

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

            intensity = np.average(color)   
            if intensity > intensity_threshold:
                region_growth_coords = []
                indices_to_check = [idx]
                while indices_to_check:
                    current_index = indices_to_check.pop()
                    if current_index not in checked_indices:
                        checked_indices.add(current_index)
                        intensity = np.average(point_colors[current_index]) 
                        if intensity > intensity_threshold:
                            region_growth_coords.append(point_coords[current_index])
                            _, neighbor_indices = points_kdtree.query([point_coords[current_index]], k=radius)
                            indices_to_check.extend(neighbor_index for neighbor_index in neighbor_indices[0] if neighbor_index not in checked_indices)
                
                #Check point count before adding to draw list
                if len(region_growth_coords) >= point_threshold:
                    all_white_object_coords.append(region_growth_coords)
                    white_objects_count += 1  #Increment counter when valid white object is found

        #Update the last processed index
        last_processed_index = idx + 1
        
        print("finished detecting, found: ", white_objects_count, "road marks in: ", time.time() - start_time)
        start_time = time.time()
        
        for white_object_coords in all_white_object_coords:
            create_dots_shape(white_object_coords)
        
        print("rendered shapes in: ", time.time() - start_time)
        
        return {'FINISHED'}

#Operator to detect curbs between 2 mouseclicks and then draw a line between them        
class CurbDetectionOperator(bpy.types.Operator):
    bl_idname = "custom.curb_detection_operator"
    bl_label = "Curb Detection Operator"
    bl_description="Finds curbs between 2 mouseclicks and marks them"
    
    click_count = 0
    first_click_point = None
    second_click_point = None
    _is_running = False  #Class variable to check if the operator is already running
    
    def modal(self, context, event):
        
        set_view_to_top(context)
        pointcloud_data = GetPointCloudData()
        points_kdtree=  pointcloud_data.points_kdtree
        point_coords = pointcloud_data.point_coords
        extra_z_height = context.scene.extra_z_height
        
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            self.click_count += 1

            #Convert the mouse position to 3D space
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
                curb_points,median_curb_point = self.detect_curb_points(points_kdtree,point_coords, self.first_click_point, self.second_click_point)
                
                if curb_points:
                    avg_lowest_point, avg_highest_point = calculate_adjusted_extreme_points(curb_points)
                    # Draw lines at the adjusted highest and lowest curb points
                    self.draw_curb_line(self.first_click_point, self.second_click_point, avg_highest_point, median_curb_point)
                    self.draw_curb_line(self.first_click_point, self.second_click_point, avg_lowest_point, median_curb_point)
                else:
                    print("No curb points found")
                    # Draw a single line from start to end point at default height
                    self.draw_curb_line(self.first_click_point, self.second_click_point, extra_z_height)
                    
                self._is_running = False
                end_time = time.time()
                print(f"Marked curb in: {end_time - start_time} seconds.")
                #Set the first click point to the second click point for the next detection
                self.first_click_point = self.second_click_point

        #If escape is pressed, stop the operator 
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
    
    #Detects the curb points between 2 mouseclicks, based on points without neighbors to the left and right
    def detect_curb_points(self, points_kdtree,point_coords, first_click_point, second_click_point):
            
        print("Starting curb detection...") 
        curb_points_indices = []

        line_direction = np.array(second_click_point) - np.array(first_click_point)
        line_length = np.linalg.norm(line_direction)
        line_direction /= line_length  # Normalize
        perp_direction = np.array([-line_direction[1], line_direction[0], 0])
        corridor_width = 0.25
        neighbor_search_distance = 0.2

        middle_point = (np.array(first_click_point) + np.array(second_click_point)) / 2
        half_num_samples = int(5 * line_length)  # Half the number of samples on each side

        #Function to check and add curb points
        def check_and_add_curb_points(sample_point):
            indices = points_kdtree.query_ball_point(sample_point, corridor_width / 2)
            for idx in indices:
                point = point_coords[idx]
                left_neighbor = points_kdtree.query_ball_point(point - perp_direction * neighbor_search_distance, 0.05)
                right_neighbor = points_kdtree.query_ball_point(point + perp_direction * neighbor_search_distance, 0.05)
                if not left_neighbor or not right_neighbor:
                    curb_points_indices.append(idx)

        #Start sampling from the middle and expand outwards
        for i in range(half_num_samples):
            t = i / half_num_samples
            sample_point_left = middle_point - t * line_direction * line_length / 2
            sample_point_right = middle_point + t * line_direction * line_length / 2

            check_and_add_curb_points(sample_point_left)
            check_and_add_curb_points(sample_point_right)

            if len(curb_points_indices) >= 1000:  # Stop if the limit is reached
                break

        #Extract unique indices as curb points may be found multiple times
        unique_indices = list(set(curb_points_indices))
        curb_points = [point_coords[idx] for idx in unique_indices]

        #Calculate the median for each coordinate
        median_x = np.median([p[0] for p in curb_points])
        median_y = np.median([p[1] for p in curb_points])
        median_z = np.median([p[2] for p in curb_points])
        median_curb_point = Vector((median_x, median_y, median_z))

        print(f"Total unique curb points found: {len(curb_points)}")   
        #visualize the curb points           
        create_dots_shape(curb_points,"curb shape",False)
        return curb_points, median_curb_point

    #Function to draw a line between 2 points, trough the median curb point       
    def draw_curb_line(self, first_click_point, second_click_point, curb_height, median_curb_point=None):
        # Create a new mesh and object
        mesh = bpy.data.meshes.new(name="Curb Line")
        obj = bpy.data.objects.new("Curb Line", mesh)

        # Link the object to the scene
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        if(median_curb_point is None):
            median_curb_point = (first_click_point + second_click_point) / 2
                                       
        # Adjust the z-coordinates of the first, median, and second points to curb_height
        adjusted_first_click = Vector((first_click_point.x, first_click_point.y, curb_height))
        adjusted_median_point = Vector((median_curb_point.x, median_curb_point.y, curb_height))
        adjusted_second_click = Vector((second_click_point.x, second_click_point.y, curb_height))

        # Create a bmesh, add vertices, and create the edges
        bm = bmesh.new()
        v1 = bm.verts.new(adjusted_first_click)
        v2 = bm.verts.new(adjusted_median_point)
        v3 = bm.verts.new(adjusted_second_click)
        bm.edges.new((v1, v2))
        bm.edges.new((v2, v3))

        # Update and free the bmesh
        bm.to_mesh(mesh)
        bm.free()
           
    def cancel(self,context):

        CurbDetectionOperator._is_running = False  #Reset the flag when the operator is cancelled
        print("Operator was properly cancelled")  #Debug message
        return {'CANCELLED'}   
                 
#Operator to find markings only within a selection made by the user with 2 mouseclicks   
class SelectionDetectionOpterator(bpy.types.Operator):
    bl_idname = "view3d.selection_detection"
    bl_label = "Detect White Objects in Region"
    bl_description = "Detect white objects within a region made by 2 mouseclicks"
    
    click_count = 0
    region_corners = []

    def modal(self, context, event):
        
        set_view_to_top(context)
        pointcloud_data = GetPointCloudData()
        point_coords = pointcloud_data.point_coords
        point_colors = pointcloud_data.point_colors
        points_kdtree=  pointcloud_data.points_kdtree
        
        if event.type == 'MOUSEMOVE':  
            self.mouse_inside_view3d = is_mouse_in_3d_view(context, event)
            
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS'and self.mouse_inside_view3d:
            #Get the mouse coordinates
            x, y = event.mouse_region_x, event.mouse_region_y
            #Convert 2D mouse coordinates to 3D view coordinates
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
                self.find_white_objects_within_region(context,point_coords,point_colors)
                return {'FINISHED'}
            for obj in bpy.context.scene.objects:
                if "BoundingBox" in obj.name:
                    bpy.data.objects.remove(obj)
        #If escape is pressed, stop the operator 
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
    #Function to find white objects within a region
    def find_white_objects_within_region(self, context,point_coords,point_colors):
       
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
        intensity_threshold=context.scene.intensity_threshold
        #Intensity calculation
        intensities = np.mean(filtered_colors, axis=1)   
        checked_indices = set()
        all_white_object_coords = []
        white_objects_count = 0 

        for idx, intensity in enumerate(intensities):
            if white_objects_count >= max_white_objects:
                break
            
            if idx in checked_indices or intensity <= intensity_threshold:
                continue

            #Region growing algorithm
            region_growth_coords = []
            indices_to_check = [idx]
            while indices_to_check:
                current_index = indices_to_check.pop()
                if current_index not in checked_indices:
                    checked_indices.add(current_index)
                    intensity = np.average(filtered_colors[current_index]) 
                    if intensity > intensity_threshold:
                        region_growth_coords.append(filtered_points[current_index])
                        _, neighbor_indices = filtered_kdtree.query([filtered_points[current_index]], k=radius)
                        indices_to_check.extend(neighbor_index for neighbor_index in neighbor_indices[0] if neighbor_index not in checked_indices)

            #Check point count before adding to draw list
            if len(region_growth_coords) >= point_threshold:
                all_white_object_coords.append(region_growth_coords)
                white_objects_count += 1  #Increment counter when valid white object is found
                
        print("road marks found: ", white_objects_count)
        #Visualize detected white objects
        for white_object_coords in all_white_object_coords:
            create_dots_shape(white_object_coords)  
            
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
        
        prepare_object_for_export(obj)

        TriangleMarkOperator ._is_running = False  #Reset the flag when the operator is cancelled
        print("Operator was properly cancelled")  #Debug message
        return {'CANCELLED'}

#Operator to automatically mark triangles between 2 mouseclicks  
class AutoTriangleMarkOperator(bpy.types.Operator):
    
    bl_idname = "custom.auto_mark_triangle"
    bl_label = "Auto Mark Triangle"
    bl_description="Marks multiple triangles by clicking on the first and last triangle"
    _is_running = False  #Class variable to check if the operator is already running
    _triangles = []  #List to store the triangle vertices
    _simulated_clicks = 0  #Count of simulated clicks
    _found_triangles = 0   #Count of triangles found
    _processed_indices = set()
             
    def modal(self, context, event):
        
        set_view_to_top(context)
        pointcloud_data = GetPointCloudData()
        point_coords = pointcloud_data.point_coords
        point_colors = pointcloud_data.point_colors
        points_kdtree=  pointcloud_data.points_kdtree
        intensity_threshold = context.scene.intensity_threshold
        
        if event.type == 'MOUSEMOVE':  
            self.mouse_inside_view3d = is_mouse_in_3d_view(context, event)
            
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS'and self.mouse_inside_view3d:
            
            #Get the mouse coordinates
            x, y = event.mouse_region_x, event.mouse_region_y
            #Convert 2D mouse coordinates to 3D view coordinates
            region = context.region
            region_3d = context.space_data.region_3d
            location = region_2d_to_location_3d(region, region_3d, (x, y), (0, 0, 0))

            #Get the z coordinate from 3D space
            z = location.z

            #Do a nearest-neighbor search
            num_neighbors = 16  #Number of neighbors 
            radius = 50
            _, nearest_indices = points_kdtree.query([location], k=num_neighbors)
        
            region_growth_coords = []
            
            #Get the average intensity of the nearest points
            average_intensity = get_average_intensity(nearest_indices[0],point_colors)
           
             #Get the average color of the nearest points
            average_color = get_average_color(nearest_indices[0],point_colors)
             
            print("average color: ", average_color,"average intensity: " ,average_intensity)
            
            #Check if the average intensity indicates a road marking (white)
            if average_intensity > intensity_threshold:
                #Region growing algorithm
                region_growth_coords,checked_indices=region_growing(point_coords, point_colors, points_kdtree, nearest_indices, radius, intensity_threshold, region_growth_coords)         
            else:
                print("no road markings found")
                
            if region_growth_coords:
                
                filtered_triangle_coords=filter_noise_with_dbscan(region_growth_coords,eps=bpy.context.scene.filter_distance, min_samples=bpy.context.scene.filter_neighbors)
                self._processed_indices.update(checked_indices)
                triangle_vertices = create_flexible_triangle(filtered_triangle_coords)
                self._triangles.append(triangle_vertices)
                create_shape(filtered_triangle_coords, shape_type="triangle", vertices=triangle_vertices)

                if(len(self._triangles)) >= 2:
                    outer_corners= self.find_closest_corners(self._triangles[0], self._triangles[1])
                    self.perform_automatic_marking(context, intensity_threshold,point_coords,point_colors,points_kdtree )
        #If escape is pressed, stop the operator         
        elif event.type == 'ESC':
            self.cancel(context)
            return {'CANCELLED'}  #Stop when ESCAPE is pressed
        return {'RUNNING_MODAL'}
    
    #Function to find the closest corners between 2 triangles
    def find_closest_corners(self, triangle1, triangle2):
        #Find the closest corners between two flat triangles
        min_distance = float('inf')
        closest_corners = (None, None)

        for corner1 in triangle1:
            for corner2 in triangle2:
                distance = np.linalg.norm(np.array(corner1[:2]) - np.array(corner2[:2]))  #Only consider X and Y
                if distance < min_distance:
                    min_distance = distance
                    closest_corners = (corner1, corner2)

        return closest_corners

    #Function to automatically mark triangles
    def perform_automatic_marking(self, context, intensity_threshold,point_coords,point_colors,points_kdtree):
        line_points = []
        #Calculate centers before popping the final triangle
        centers = [np.mean(triangle, axis=0) for triangle in self._triangles]

        #This is the second user click, mark it as the final triangle
        final_triangle = self._triangles.pop()

        #Automatically mark the triangles in between
        middle_points = self.interpolate_line(centers[0], centers[1])
        for point in middle_points:
            self.simulate_click_and_grow(point, context, intensity_threshold, point_coords,point_colors,points_kdtree)

        #Add the final triangle back to the end of the list
        self._triangles.append(final_triangle)

        #Create segmented lines between the bottom corners of each triangle
        for i in range(1, len(self._triangles)):
            prev_triangle = self._triangles[i - 1]
            current_triangle = self._triangles[i]
  
            closest_corners = self.find_closest_corners(prev_triangle, current_triangle)
            line_points.extend(closest_corners)
            create_polyline(closest_corners,"segment_" + str(i) )

        if line_points:
            create_polyline(line_points,"continuous_line") #create 1 line of out all the segments
            
    #Function to simulate a click and grow on a point        
    def simulate_click_and_grow(self, location, context, intensity_threshold, point_coords,point_colors,points_kdtree):
        
        _, nearest_indices = points_kdtree.query([location], k=16)
        average_intensity = get_average_intensity(nearest_indices[0],point_colors)

        if (average_intensity > intensity_threshold) and not self._processed_indices.intersection(nearest_indices[0]):
            #Proceed if the intensity is above the threshold and the area hasn't been processed yet
            checked_indices = set()
            indices_to_check = list(nearest_indices[0])

            while indices_to_check:
                current_index = indices_to_check.pop()
                if current_index not in checked_indices:
                    checked_indices.add(current_index)
                    intensity = np.average(point_colors[current_index]) 
                    if intensity > intensity_threshold:
                        _, neighbor_indices = points_kdtree.query([point_coords[current_index]], k=50)
                        indices_to_check.extend(neighbor_index for neighbor_index in neighbor_indices[0] if neighbor_index not in checked_indices)

            if checked_indices:
                points = [point_coords[i] for i in checked_indices]
                filtered_points = filter_noise_with_dbscan(points,eps=bpy.context.scene.filter_distance, min_samples=bpy.context.scene.filter_neighbors)
                self._processed_indices.update(checked_indices)
                triangle_vertices = create_flexible_triangle(filtered_points)
                self._triangles.append(triangle_vertices)
                self._found_triangles += 1
                create_shape(filtered_points, shape_type="triangle", vertices=triangle_vertices)
                
    def interpolate_line(self, start, end, num_points=100):
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

#Operator to mark triangles at every mouse click
class TriangleMarkOperator(bpy.types.Operator):
    bl_idname = "custom.mark_triangle"
    bl_label = "Mark Triangle"
    bl_description="Marks a triangle"
    _is_running = False  #Class variable to check if the operator is already running
    _triangles = []  #List to store the triangle vertices
    _processed_indices = set()
    _last_outer_corner = None  #Initialize the last outer corner here   
    
    def modal(self, context, event):
        
        set_view_to_top(context)
        pointcloud_data = GetPointCloudData()
        point_coords = pointcloud_data.point_coords
        point_colors = pointcloud_data.point_colors
        points_kdtree=  pointcloud_data.points_kdtree
        intensity_threshold = context.scene.intensity_threshold
        extra_z_height = context.scene.extra_z_height
        
        if event.type == 'MOUSEMOVE':  
            self.mouse_inside_view3d = is_mouse_in_3d_view(context, event)
            
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS' and self.mouse_inside_view3d:
            #Process the mouse click
            self.create_triangle(context, event,intensity_threshold,point_coords,point_colors,points_kdtree)
        #If escape is pressed, stop the operator 
        elif event.type == 'ESC':
            self.cancel(context)
            return {'CANCELLED'}  #Stop when ESCAPE is pressed
        
        return {'RUNNING_MODAL'}
    
    #Function to create a triangle
    def create_triangle(self, context,event, intensity_threshold,point_coords,point_colors,points_kdtree):
        #Get the mouse coordinates
        x, y = event.mouse_region_x, event.mouse_region_y
        location = region_2d_to_location_3d(context.region, context.space_data.region_3d, (x, y), (0, 0, 0))
        region_growth_coords=[]
        search_radius=20
        #Nearest-neighbor search
        _, nearest_indices = points_kdtree.query([location], k=16)
        average_intensity = get_average_intensity(nearest_indices[0],point_colors)
        if average_intensity > intensity_threshold:
            #Region growing algorithm
            region_growth_coords,checked_indices=region_growing(point_coords, point_colors, points_kdtree, nearest_indices, search_radius, intensity_threshold, region_growth_coords)         

            if region_growth_coords:
  
                #current_triangle_coords=[point_coords[i] for i in checked_indices]
                filtered_current_triangle_coords=filter_noise_with_dbscan(region_growth_coords,eps=bpy.context.scene.filter_distance, min_samples=bpy.context.scene.filter_neighbors)
                self._processed_indices.update(checked_indices)
                current_triangle_vertices = create_flexible_triangle(filtered_current_triangle_coords)
                self._triangles.append(current_triangle_vertices)
                    
                if len(self._triangles) >= 2:
                    #Check if _last_outer_corner is initialized
                    if self._last_outer_corner is None:
                        outer_corners = self.find_outermost_corners(self._triangles[-2], self._triangles[-1])
                        self._last_outer_corner = outer_corners[1]
                    else:
                        #Use the last outer corner and find the new one
                        new_outer_corner = self.find_outermost_corner(self._triangles[-1], self._last_outer_corner)
                        outer_corners = [self._last_outer_corner, new_outer_corner]
                        self._last_outer_corner = new_outer_corner

                    #Ensure outer_corners contains two points, each  a list or tuple
                    if all(isinstance(corner, (list, tuple)) for corner in outer_corners):
                        create_polyline(outer_corners,"triangles_base_line")
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
                
    #Function to find the outermost corners of a triangle
    def find_outermost_corners(self,triangle1, triangle2):
        max_distance = 0
        outermost_points = (None, None)

        for point1 in triangle1:
            for point2 in triangle2:
                distance = np.linalg.norm(np.array(point1) - np.array(point2))
                if distance > max_distance:
                    max_distance = distance
                    outermost_points = (point1, point2)
        return outermost_points
    
    #Function to find the outermost corner of a triangle compared to a reference point
    def find_outermost_corner(self,triangle, reference_point):
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

#Operator to mark rectangles at every mouse click        
class RectangleMarkOperator(bpy.types.Operator):
    bl_idname = "custom.mark_rectangle"
    bl_label = "Mark Rectangle"
    bl_description="Mark a rectangle"
    _is_running = False  #Class variable to check if the operator is already running
    
    def modal(self, context, event):
        
        set_view_to_top(context)
        pointcloud_data = GetPointCloudData()
        point_coords = pointcloud_data.point_coords
        point_colors = pointcloud_data.point_colors
        points_kdtree=  pointcloud_data.points_kdtree
        intensity_threshold = context.scene.intensity_threshold
        
        if event.type == 'MOUSEMOVE':  
            self.mouse_inside_view3d = is_mouse_in_3d_view(context, event)
            
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS'and self.mouse_inside_view3d:
            
            #Get the mouse coordinates
            x, y = event.mouse_region_x, event.mouse_region_y
            #Convert 2D mouse coordinates to 3D view coordinates
            region = context.region
            region_3d = context.space_data.region_3d
            location = region_2d_to_location_3d(region, region_3d, (x, y), (0, 0, 0))

            #Do a nearest-neighbor search
            num_neighbors = 16  #Number of neighbors 
            radius = 50
            _, nearest_indices = points_kdtree.query([location], k=num_neighbors)
        
            region_growth_coords = []
            
            #Get the average intensity of the nearest points
            average_intensity = get_average_intensity(nearest_indices[0],point_colors)
           
             #Get the average color of the nearest points
            average_color = get_average_color(nearest_indices[0],point_colors)
             
            print("average color: ", average_color,"average intensity: " ,average_intensity)
            
            #Check if the average intensity indicates a road marking (white)
            if average_intensity > intensity_threshold:
                #Region growing algorithm
                region_growth_coords,checked_indices=region_growing(point_coords, point_colors, points_kdtree, nearest_indices, radius, intensity_threshold, region_growth_coords)                
            
            else:
                print("no road markings found")
                
            if region_growth_coords:
                #Create a single mesh for the combined  rectangles
                create_shape(region_growth_coords,shape_type="rectangle")
                
        #If escape is pressed, stop the operator 
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

#Operator to automatically mark rectangles between 2 clicked rectangles
class AutoRectangleMarkOperator(bpy.types.Operator):
    bl_idname = "custom.auto_mark_rectangle"
    bl_label = "Auto Mark rectangle"
    bl_description = "Automatically mark multiple rectangles by clicking on the first and last rectangle"
    
    _is_running = False  #Class variable to check if the operator is already running
    _rectangles = []  #List to store the rectangle vertices
    _simulated_clicks = 0  #Count of simulated clicks
    _found_rectangles = 0   #Count of triangles found
    _processed_indices = set()
              
    def modal(self, context, event):
        
        set_view_to_top(context)
        pointcloud_data = GetPointCloudData()
        point_coords = pointcloud_data.point_coords
        point_colors = pointcloud_data.point_colors
        points_kdtree=  pointcloud_data.points_kdtree
        intensity_threshold = context.scene.intensity_threshold
        
        if event.type == 'MOUSEMOVE':  
            self.mouse_inside_view3d = is_mouse_in_3d_view(context, event)
            
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS'and self.mouse_inside_view3d:
            
            #Get the mouse coordinates
            x, y = event.mouse_region_x, event.mouse_region_y
            #Convert 2D mouse coordinates to 3D view coordinates
            region = context.region
            region_3d = context.space_data.region_3d
            location = region_2d_to_location_3d(region, region_3d, (x, y), (0, 0, 0))

            #Do a nearest-neighbor search
            num_neighbors = 16  #Number of neighbors 
            radius = 50
            _, nearest_indices = points_kdtree.query([location], k=num_neighbors)
        
            region_growth_coords = []
            
            #Get the average intensity of the nearest points
            average_intensity = get_average_intensity(nearest_indices[0],point_colors)
           
             #Get the average color of the nearest points
            average_color = get_average_color(nearest_indices[0],point_colors)
             
            print("average color: ", average_color,"average intensity: " ,average_intensity)
            
            #Check if the average intensity indicates a road marking (white)
            if average_intensity > intensity_threshold:
                #Region growing algorithm
                region_growth_coords,checked_indices=region_growing(point_coords, point_colors, points_kdtree, nearest_indices, radius, intensity_threshold, region_growth_coords)         
            
            else:
                print("no road markings found")
            
            if region_growth_coords:
                self._processed_indices.update(checked_indices)
                rectangle_vertices = create_flexible_rectangle(region_growth_coords)
                self._rectangles.append(rectangle_vertices)
                create_shape(region_growth_coords, shape_type="rectangle",vertices=None, filter_coords=True)

                if len(self._rectangles) == 2:
                    
                    self.perform_automatic_marking(context, intensity_threshold,point_coords,point_colors,points_kdtree)
        #If escape is pressed, stop the operator            
        elif event.type == 'ESC':
            self.cancel(context)
            return {'CANCELLED'}  #Stop when ESCAPE is pressed
        
        return {'RUNNING_MODAL'}
    
    #Function to find the center points of 2 given rectangles
    def find_center_points(self,rectangle1, rectangle2):
        max_distance = 0
        center_points = (None, None)

        for point1 in rectangle1:
            for point2 in rectangle2:
                distance = np.linalg.norm(np.array(point1) - np.array(point2))
                if distance > max_distance:
                    max_distance = distance
                    center_points = (point1, point2)
        return center_points
    
    #Function to automatically mark rectangles
    def perform_automatic_marking(self, context, intensity_threshold,point_coords,point_colors,points_kdtree):
        print("2 rectangles found, starting automatic marking..")
        if len(self._rectangles) >= 2:
            centers = [np.mean(rectangle, axis=0) for rectangle in self._rectangles[:2]]
            middle_points = self.interpolate_line(centers[0], centers[1])
            for point in middle_points:
                mark_point(point,"ZebraCrossing",size=0.1)
                self.simulate_click_and_grow(point, context, intensity_threshold, point_coords,point_colors,points_kdtree)  
                         
    #Function to simulate a click and grow on a point        
    def simulate_click_and_grow(self, location, context, intensity_threshold, point_coords,point_colors,points_kdtree):
        
        _, nearest_indices = points_kdtree.query([location], k=16)
        average_intensity = get_average_intensity(nearest_indices[0],point_colors)
        average_color = get_average_color(nearest_indices[0],point_colors)

        if (average_intensity > intensity_threshold) and not self._processed_indices.intersection(nearest_indices[0]):
            #Proceed only if the intensity is above the threshold and the area hasn't been processed yet
            checked_indices = set()
            indices_to_check = list(nearest_indices[0])

            while indices_to_check:
                current_index = indices_to_check.pop()
                if current_index not in checked_indices:
                    checked_indices.add(current_index)
                    intensity = np.average(point_colors[current_index]) 
                    if intensity > intensity_threshold:
                        _, neighbor_indices = points_kdtree.query([point_coords[current_index]], k=50)
                        indices_to_check.extend(neighbor_index for neighbor_index in neighbor_indices[0] if neighbor_index not in checked_indices)

            if checked_indices:
                rectangle_points = [point_coords[i] for i in checked_indices]
                filtered_points = filter_noise_with_dbscan(rectangle_points,eps=bpy.context.scene.filter_distance, min_samples=bpy.context.scene.filter_neighbors)
                self._processed_indices.update(checked_indices)
                rectangle_vertices = create_flexible_rectangle(rectangle_points)
                self._rectangles.append(rectangle_vertices)
                self._found_rectangles += 1
                create_shape(filtered_points, shape_type="rectangle")
        
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

#Operator for drawing a line between mouse clicks with optional soft snapping to nearby road mark      
class SnappingLineMarkOperator(bpy.types.Operator):
    bl_idname = "custom.mark_snapping_line"
    bl_label = "Mark curved line"
    bl_description= "Draws a line between 2 mouse clicks"
    
    prev_end_point = None
    _is_running = False  
    
    def modal(self, context, event):
        
        set_view_to_top(context)
        pointcloud_data = GetPointCloudData()
        point_coords = pointcloud_data.point_coords
        point_colors = pointcloud_data.point_colors
        points_kdtree=  pointcloud_data.points_kdtree
        if event.type == 'LEFTMOUSE' and is_mouse_in_3d_view(context, event):
            if event.value == 'RELEASE':
                draw_line(self, context, event, point_coords, point_colors, points_kdtree)
                return {'RUNNING_MODAL'}
        #If escape is pressed, stop the operator 
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

#Operator to automatically mark dashed lines after clicking on 2
class DashedLineMarkingOperator(bpy.types.Operator):
    bl_idname = "custom.dashed_line_marking_operator"
    bl_label = "Dashed Line Marking Operator"
    bl_description = "Finds and marks repeating dash lines after clicking on 2 dash lines"

    click_count = 0
    first_cluster_center = None
    second_cluster_center = None
    _is_running = False
    
    def modal(self, context, event):
        
        set_view_to_top(context)
        pointcloud_data = GetPointCloudData()
        point_coords = pointcloud_data.point_coords
        point_colors = pointcloud_data.point_colors
        points_kdtree=  pointcloud_data.points_kdtree
        intensity_threshold = context.scene.intensity_threshold
        
        if event.type == 'MOUSEMOVE':  
            self.mouse_inside_view3d = is_mouse_in_3d_view(context, event)
    
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            self.click_count += 1
            #Get the mouse coordinates
            x, y = event.mouse_region_x, event.mouse_region_y
            #Convert 2D mouse coordinates to 3D view coordinates
            region = context.region
            region_3d = context.space_data.region_3d
            click_point = region_2d_to_location_3d(region, region_3d, (x, y), (0, 0, 0))

            #Get the z coordinate from 3D space
            z = click_point.z

            if self.click_count == 1:
                
                self.first_cluster_center = self.find_cluster_center(points_kdtree, point_coords, point_colors, click_point, intensity_threshold,)
           
            elif self.click_count == 2:
                
                self.second_cluster_center = self.find_cluster_center(points_kdtree, point_coords, point_colors, click_point, intensity_threshold,)

                if self.first_cluster_center is not None and self.second_cluster_center is not None: #make sure the clusters are not empty
                    
                    self.connect_clusters(self.first_cluster_center, self.second_cluster_center)
                     # Continue finding and connecting additional clusters
                    self.extend_dashed_lines(points_kdtree, point_coords, point_colors, self.second_cluster_center, self.first_cluster_center, intensity_threshold)
               
                self._is_running = False
                return {'FINISHED'}
            
        #Stop the operator of ESCAPE is pressed
        elif event.type in {'ESC'}:
            return self.cancel(context)

        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        self.click_count = 0
        context.window_manager.modal_handler_add(self)
        if DashedLineMarkingOperator._is_running:
            self.report({'WARNING'}, "Operator is already running")
            return self.cancel(context)
        self._is_running = True
        return {'RUNNING_MODAL'}

    def find_cluster_center(self, points_kdtree, point_coords, point_colors, click_point, intensity_threshold, search_radius=0.3):
       
        #select points within the specified radius of the click point
        indices = points_kdtree.query_ball_point(click_point, search_radius)
        
        if not indices:
            print("No points found near the click point")
            return None

        #Filter points based on the intensity threshold
        filtered_points = [point_coords[i] for i in indices if np.average(point_colors[i]) >= intensity_threshold]
        print("Number of cluster points found above ",int(intensity_threshold),f" intensity: {len(filtered_points)}")
        
        if not filtered_points:
            print("No points above the intensity threshold")
            return None
        
        # Check if filtered_points is empty before creating a dots shape
        if len(filtered_points) > 0:
            create_dots_shape(filtered_points, "dash line", True)
        else:
            print("No points to create a shape from")
            return None

        #Calculate the median for each coordinate
        median_x = np.median([p[0] for p in filtered_points])
        median_y = np.median([p[1] for p in filtered_points])
        median_z = np.median([p[2] for p in filtered_points])

        #Combine medians to form the median point
        median_point = Vector((median_x, median_y, median_z))
        mark_point(median_point, "dash line center",size=0.1)

        return median_point

    def connect_clusters(self, first_cluster_center, second_cluster_center):
        
        if first_cluster_center is None or second_cluster_center is None:
            print("One of the cluster centers is None. Skipping line creation.")
            return
        
        #Create a new mesh and object
        mesh = bpy.data.meshes.new(name="Dash Line")
        obj = bpy.data.objects.new("Dash Line", mesh)

        #Link the object to the scene
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        #Create a bmesh, add vertices, and create the edge
        bm = bmesh.new()
        v1 = bm.verts.new(first_cluster_center)
        v2 = bm.verts.new(second_cluster_center)
        bm.edges.new((v1, v2))

        #Update and free the bmesh
        bm.to_mesh(mesh)
        bm.free()

    def extend_dashed_lines(self, points_kdtree, point_coords, point_colors, start_cluster_center, end_cluster_center, intensity_threshold, search_radius=0.3):
        if start_cluster_center is None or end_cluster_center is None:
            print("One of the cluster centers is None. Cannot extend dashed lines.")
            return
        
        #Calculate initial line direction and length
        line_direction = np.array(end_cluster_center) - np.array(start_cluster_center)
        line_length = np.linalg.norm(line_direction)
        line_direction /= line_length  #Normalize 
        # Reverse the line direction
        line_direction = -line_direction

        print(f"Line direction: {line_direction}, Line length: {line_length}")
    
        current_search_point = np.array(end_cluster_center)
        print("searching for more dash lines..")
        while True:
            
            #Move to the next search point
            current_search_point += line_direction * line_length
            
            #mark_point(current_search_point, "Current Search Point")
            print(f"Searching for a new cluster at: {current_search_point}")
        
            #Find a new cluster at this point
            new_cluster_center = self.find_cluster_center(points_kdtree, point_coords, point_colors, current_search_point, intensity_threshold, search_radius=0.3)
            
            if new_cluster_center is None:
                print("No more clusters found")
                break  #No more clusters found terminate the search
            
            mark_point(new_cluster_center, "dash line center",size=0.1)

            #Connect the previous cluster center to the new cluster center
            self.connect_clusters(end_cluster_center, new_cluster_center)

            #Update the end cluster center for the next iteration
            end_cluster_center = new_cluster_center
            
    def cancel(self, context):

        DashedLineMarkingOperator._is_running = False  #Reset the flag when the operator is cancelled
        print("Operator was properly cancelled")  #Debug message
        return {'CANCELLED'}
    
#Operator to automatically mark a curved line at mouseclick            
class AutoCurvedLineOperator(bpy.types.Operator):
    bl_idname = "custom.auto_curved_line"
    bl_label = "Mark curved line" 
    bl_description = "Automatically draws a curved line"
    
    def modal(self, context, event):
        
        set_view_to_top(context)
        pointcloud_data = GetPointCloudData()
        point_coords = pointcloud_data.point_coords
        point_colors = pointcloud_data.point_colors
        points_kdtree=  pointcloud_data.points_kdtree
        intensity_threshold = context.scene.intensity_threshold
        
        if event.type == 'MOUSEMOVE':  
            self.mouse_inside_view3d = is_mouse_in_3d_view(context, event)
            
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS'and self.mouse_inside_view3d:
            
            #Get the mouse coordinates
            x, y = event.mouse_region_x, event.mouse_region_y
            #Convert 2D mouse coordinates to 3D view coordinates
            region = context.region
            region_3d = context.space_data.region_3d
            location = region_2d_to_location_3d(region, region_3d, (x, y), (0, 0, 0))

            #Do a nearest-neighbor search
            num_neighbors = 16  #Number of neighbors 
            radius = 50
            _, nearest_indices = points_kdtree.query([location], k=num_neighbors)
        
            region_growth_coords = []
            
            #Get the average intensity of the nearest points
            average_intensity = get_average_intensity(nearest_indices[0],point_colors)
           
             #Get the average color of the nearest points
            average_color = get_average_color(nearest_indices[0],point_colors)
             
            print("average color: ", average_color,"average intensity: " ,average_intensity)
            
             #Check if the average intensity indicates a road marking (white)
            if average_intensity > intensity_threshold:
                region_growth_coords,checked_indices = region_growing(point_coords, point_colors, points_kdtree, nearest_indices, radius, intensity_threshold, region_growth_coords)
                
            else:
                print("no road markings found")
                
            if region_growth_coords:
                #Create a single mesh for the combined  rectangles
                create_shape(region_growth_coords,shape_type="curved line")
                
        elif event.type == 'ESC':
            return self.cancel(context)  #Stop when ESCAPE is pressed
        
        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        if SimpleMarkOperator._is_running:
            self.report({'WARNING'}, "Operator is already running")
            return self.cancel(context)

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

#Operator to draw a triangle of a fixed size at every mouseclick
class FixedTriangleMarkOperator(bpy.types.Operator):
    bl_idname = "custom.mark_fixed_triangle"
    bl_label = "mark a fixed triangle"
    bl_description = "Draws a fixed triangle"
    #Properties to receive external coordinates
    external_x: bpy.props.FloatProperty(name="External X")
    external_y: bpy.props.FloatProperty(name="External Y")
    external_z: bpy.props.FloatProperty(name="External Z")

    def modal(self, context, event):
        set_view_to_top(context)
      
        if event.type == 'MOUSEMOVE':  
            self.mouse_inside_view3d = is_mouse_in_3d_view(context, event)

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS'and self.mouse_inside_view3d:
            if context.area and context.area.type == 'VIEW_3D':
                #Get the mouse coordinates
                x, y = event.mouse_region_x, event.mouse_region_y
                region = context.region
                region_3d = context.space_data.region_3d
                location = region_2d_to_location_3d(region, region_3d, (x, y), (0, 0, 0))

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
        
    def execute(self, context):
        # Use the external coordinates if provided
        if hasattr(self, 'external_x') and hasattr(self, 'external_y') and hasattr(self, 'external_z'):
            location = Vector((self.external_x, self.external_y, self.external_z))
            draw_fixed_triangle(context, location, size=0.5)
            print("Triangle drawn at", location)
        return {'FINISHED'}
    
#Operator to draw a rectangle of a fixed size at every mouseclick        
class FixedRectangleMarkOperator(bpy.types.Operator):
    bl_idname = "custom.mark_fixed_rectangle"
    bl_label = "mark a fixed rectangle"
    bl_description = "Draws a fixed rectangle"
    
    def modal(self, context, event):
        set_view_to_top(context)
        if event.type == 'MOUSEMOVE':  
            self.mouse_inside_view3d = is_mouse_in_3d_view(context, event)

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS'and self.mouse_inside_view3d:
            if context.area and context.area.type == 'VIEW_3D':
                #Get the mouse coordinates
                x, y = event.mouse_region_x, event.mouse_region_y
                #Convert 2D mouse coordinates to 3D view coordinates
                region = context.region
                region_3d = context.space_data.region_3d
                location = region_2d_to_location_3d(region, region_3d, (x, y), (0, 0, 0))

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

    #bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.objects.active = obj  #Set as active object
    bpy.ops.object.select_all(action='DESELECT')  #Deselect all objects
    obj.select_set(True)  #Select the current object

    set_origin_to_geometry_center(obj)
    save_shape_checkbox = bpy.context.scene.save_shape
    if(save_shape_checkbox):
        save_shape_as_image(obj)

    save_obj_checkbox = bpy.context.scene.save_obj
    if(save_obj_checkbox):
        export_shape_as_obj(obj, obj.name)
        
#Function to export objects as OBJ files 
def export_shape_as_obj(obj, name):
    
    marking_color = bpy.context.scene.marking_color 
    transparency = bpy.context.scene.marking_transparency
    shape_material = bpy.data.materials.new(name="shape_material")
    shape_material.diffuse_color = (marking_color[0], marking_color[1], marking_color[2], transparency)

    # Assign the material to the object
    if len(obj.data.materials) > 0:
        obj.data.materials[0] = shape_material
    else:
        obj.data.materials.append(shape_material)
        
    # Get the path of the current Blender file
    blend_file_path = bpy.data.filepath
    directory = os.path.dirname(blend_file_path)

    # Create the 'shape objects' directory if it doesn't exist
    shapes_dir = os.path.join(directory, "shape objects")
    if not os.path.exists(shapes_dir):
        os.makedirs(shapes_dir)

    # Define the path for the OBJ file
    obj_file_path = os.path.join(shapes_dir, f"{name}.obj")

    # Select the object
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Export the object to an OBJ file
    bpy.ops.export_scene.obj(filepath=obj_file_path, use_selection=True,use_materials=True)
    
    # Deselect the object
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
                             



#Blender digitizing functions 
#Function to create a flexible triangle shape out of coordinates                                    
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

#Function to draw a fixed triangle shape out of coordinates
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

#Function to create a fixed triangle shape out of coordinates    
def create_fixed_triangle(coords, side_length=0.5):
     #Convert coords to numpy array for efficient operations
    coords_np = np.array(coords)

    #Reference vertex 
    vertex1 = coords_np[0]

    #Normal vector of the plane defined by the original triangle
    normal_vector = np.cross(coords_np[1] - vertex1, coords_np[2] - vertex1)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)  #Normalize the normal vector

    #direction vector for the second vertex
    dir_vector = coords_np[1] - vertex1
    dir_vector = dir_vector / np.linalg.norm(dir_vector) * side_length

    #Calculate the position of the second vertex
    vertex2 = vertex1 + dir_vector

 
    #use cross product to find a perpendicular vector 
    perp_vector = np.cross(normal_vector, dir_vector)
    perp_vector = perp_vector / np.linalg.norm(perp_vector) * side_length

    #Angle for equilateral triangle (60 degrees)
    angle_rad = np.deg2rad(60)

    #Calculate the position of the third vertex
    vertex3 = vertex1 + np.cos(angle_rad) * dir_vector + np.sin(angle_rad) * perp_vector

    return [vertex1.tolist(), vertex2.tolist(), vertex3.tolist()]

#Function to create a flexible rectangle shape out of coordinates
def create_flexible_rectangle(coords):
    
    extra_z_height = bpy.context.scene.extra_z_height
    coords_np = np.array(coords)

    #Find minimum and maximum X and Y coordinates ignoring Z-coordinate if present
    min_x = np.min(coords_np[:, 0])
    max_x = np.max(coords_np[:, 0])
    min_y = np.min(coords_np[:, 1])
    max_y = np.max(coords_np[:, 1])

    #Create rectangle corners 
    top_left = (min_x, max_y, extra_z_height)
    top_right = (max_x, max_y, extra_z_height)
    bottom_right = (max_x, min_y, extra_z_height)
    bottom_left = (min_x, min_y, extra_z_height)

    return [top_left, top_right, bottom_right, bottom_left]

#Function to create a fixed rectangle shape at a location
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

#Function to create a polyline going trough given points      
def create_polyline(points,name='Poly Line', width=0.01, color=(1, 0, 0, 1)):
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
    return curve_obj

#Function to create segments of a given size out of points
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

        #Normalize the segment vector
        segment_vector.normalize()

        #Generate points at fixed intervals between start and end
        while segment_distance > segment_length:
            new_point = start_point + segment_vector * segment_length
            extended_points.append(new_point)
            start_point = new_point
            segment_distance -= segment_length
            segment_count += 1

        # Add the last point if it doesn't fit a full segment
        if segment_distance > 0:
            extended_points.append(end_point)

    #Adjust the last segment if it's not a full segment
    if total_distance % segment_length != 0:
        extended_points[-1] = extended_points[-2] + segment_vector * (total_distance % segment_length)

    return extended_points, total_distance, segment_count + 1  # Include the last partial segment

#Function to create different shapes out of points
def create_shape(coords_list, shape_type,vertices=None,filter_coords=True):
    
    start_time = time.time()
    marking_color = bpy.context.scene.marking_color 
    transparency = bpy.context.scene.marking_transparency
    line_width = bpy.context.scene.fatline_width
    shape_coords = None  #Default to original coordinates
    if filter_coords:
        coords_list=filter_noise_with_dbscan(coords_list,eps=bpy.context.scene.filter_distance, min_samples=bpy.context.scene.filter_neighbors)
    
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

        obj=create_polyline(middle_points,  width=line_width, color=(marking_color[0], marking_color[1], marking_color[2], transparency))
   
    else:
        print("Drawing unkown Shape")
        obj=create_mesh_with_material(
            "Unkown Shape", coords_list,
            marking_color, transparency)
        
    print(f"Rendered {shape_type} shape in: {time.time() - start_time:.2f} seconds")
    prepare_object_for_export(obj)
    
#Function to create a mesh object
def create_mesh_with_material(obj_name, shape_coords, marking_color, transparency):
    
    extra_z_height = bpy.context.scene.extra_z_height
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
    prepare_object_for_export(obj)
    return obj

#Function to draw a line between two points with optional snapping
def draw_line(self, context, event,point_coords, point_colors, points_kdtree):
    if not hasattr(self, 'click_counter'):
        self.click_counter = 0

    snap_to_road_mark = context.scene.snap_to_road_mark
    extra_z_height = context.scene.extra_z_height
    
    region = context.region
    region_3d = context.space_data.region_3d
    
    #Convert the mouse position to a 3D location for the end point of the line
    coord_3d_end = view3d_utils.region_2d_to_location_3d(region, region_3d, (event.mouse_region_x, event.mouse_region_y), Vector((0, 0, 0)))
    coord_3d_end.z += extra_z_height

    self.click_counter += 1

    #Check if the current click is on a white road mark
    on_white = is_click_on_white(self, context, coord_3d_end)

    if snap_to_road_mark and self.click_counter > 1:
        #Calculate the direction vector
        direction_vector = (self.prev_end_point - coord_3d_end).normalized()
        search_range = 0.5

        #Find the center of the cluster near the second click point
        cluster_center = find_cluster_center(context, coord_3d_end, direction_vector, search_range,point_coords, point_colors, points_kdtree)
        if cluster_center is not None:
            coord_3d_end = cluster_center  #Move the second click point to the cluster center

    #Create or update the line
    if self.prev_end_point is not None:
        create_rectangle_line_object(self.prev_end_point, coord_3d_end)

    self.prev_end_point = coord_3d_end  #Update the previous end point

#Function to determine the center of a cluster of points
def find_cluster_center(context, click_point, direction, range,point_coords, point_colors, points_kdtree):
    intensity_threshold = context.scene.intensity_threshold
   
    # Define the search bounds and find points within the bounds
    upper_bound = click_point + direction * range
    lower_bound = click_point - direction * range
    indices = points_kdtree.query_ball_point([upper_bound, lower_bound], range)
    indices = [i for sublist in indices for i in sublist]
    potential_points = np.array(point_coords)[indices]
    high_intensity_points = potential_points[np.average(point_colors[indices], axis=1) > intensity_threshold]

    if len(high_intensity_points) > 0:
        # Find the outer points
        min_x = np.min(high_intensity_points[:, 0])
        max_x = np.max(high_intensity_points[:, 0])
        min_y = np.min(high_intensity_points[:, 1])
        max_y = np.max(high_intensity_points[:, 1])

        # Calculate the center of these extremal points
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        center_z = np.mean(high_intensity_points[:, 2])  # Average Z value
  
        mark_point(Vector((center_x, center_y, center_z)), "cluster_center")
        return Vector((center_x, center_y, center_z))

    return None

#Function to create a colored, resizable line shape between 2 points 
def create_rectangle_line_object(start, end):
    
    context = bpy.context
    marking_color = context.scene.marking_color
    transparency = context.scene.marking_transparency
    extra_z_height = context.scene.extra_z_height
    width = context.scene.fatline_width
    #Calculate the direction vector and its length
    direction = end - start

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

    #Set the Principled BSDF shader alpha value
    principled_bsdf = next(node for node in material.node_tree.nodes if node.type == 'BSDF_PRINCIPLED')
    principled_bsdf.inputs['Alpha'].default_value = transparency
    
    #Assign the material to the object
    obj.data.materials.append(material)
    prepare_object_for_export(obj)

    return obj

#Function to create multiple squares on top of detected points, then combines them into one shape
def create_dots_shape(coords_list,name="Dots Shape", filter_points=True):
    
    global shape_counter
    
    marking_color=bpy.context.scene.marking_color
    transparency = bpy.context.scene.marking_transparency
    extra_z_height = bpy.context.scene.extra_z_height
    
    #Create a new mesh and link it to the scene
    mesh = bpy.data.meshes.new("Combined Shape")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    bm = bmesh.new()

    square_size = 0.025  #Size of each square
    z_offset = extra_z_height  #Offset in Z coordinate
    max_gap = 10  #Maximum gap size to fill

    if filter_points:
        #filters out bad points
        coords_list = filter_noise_with_dbscan(coords_list,eps=bpy.context.scene.filter_distance, min_samples=bpy.context.scene.filter_neighbors)
    
    #Sort the coordinates by distance
    coords_list.sort(key=lambda coords: (coords[0]**2 + coords[1]**2 + coords[2]**2)**0.5)

    for i in range(len(coords_list)):
        if i > 0:
            #Calculate the distance to the previous point
            gap = ((coords_list[i][0] - coords_list[i-1][0])**2 +
                   (coords_list[i][1] - coords_list[i-1][1])**2 +
                   (coords_list[i][2] - coords_list[i-1][2])**2)**0.5
            if gap > max_gap:
                #If the gap is too large, create a new mesh for the previous group of points
                bm.to_mesh(mesh)
                bm.clear()
                #Update the internal index table of the BMesh
                bm.verts.ensure_lookup_table()

        #Create a square at the current point with an adjusted Z coordinate
        square_verts = [
            bm.verts.new(coords_list[i] + (-square_size / 2, -square_size / 2, z_offset)),
            bm.verts.new(coords_list[i] + (-square_size / 2, square_size / 2, z_offset)),
            bm.verts.new(coords_list[i] + (square_size / 2, square_size / 2, z_offset)),
            bm.verts.new(coords_list[i] + (square_size / 2, -square_size / 2, z_offset)),
        ]

        #Create a face for the square
        bm.faces.new(square_verts)

    #Create a mesh for the last group of points
    bm.to_mesh(mesh)
    bm.free()

    #Create a new material for the combined shape
    shape_material = bpy.data.materials.new(name="shape material")
    shape_material.diffuse_color = (marking_color[0], marking_color[1], marking_color[2], transparency)
    #Enable transparency in the material settings
    shape_material.use_nodes = True
    shape_material.blend_method = 'BLEND'

    #Find the Principled BSDF node and set its alpha value
    principled_node = next(n for n in shape_material.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    principled_node.inputs['Alpha'].default_value = transparency
    
    #Assign the material to the object
    if len(obj.data.materials) > 0:
        #If the object already has materials, replace the first one with the  material
        obj.data.materials[0] = shape_material
    else:
        #add the material to the object
        obj.data.materials.append(shape_material)
        
    obj.color = marking_color  #Set viewport display color 
    shape_counter+=1
    prepare_object_for_export(obj)
    
#Function to draw tiny marks on a given point
def mark_point(point, name="point", size=0.05):
    
    show_dots=bpy.context.scene.show_dots
    
    if show_dots:
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

#Function to create a triangle outline
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
    
    prepare_object_for_export(obj)
    return obj

#Function to find nearby road marks and then snap the line, made out of 2 click points, to it
def snap_line_to_road_mark(self, context, first_click_point, last_click_point,point_coords, point_colors, points_kdtree,region_radius=10):
    
    intensity_threshold = context.scene.intensity_threshold       

    #Get the direction vector between the two clicks and its perpendicular
    direction = (last_click_point - first_click_point).normalized()
    perp_direction = direction.cross(Vector((0, 0, 1))).normalized()

    #Find the index of the last click point in the point cloud
    _, nearest_indices = points_kdtree.query([last_click_point], k=1)
         
    #Function to find the most outward points of a region
    def find_outward_points(region_points, direction):
        #Project all points to the direction vector and find the most outward points
        projections = [np.dot(point, direction) for point in region_points]
        min_proj_index = np.argmin(projections)
        max_proj_index = np.argmax(projections)
        return region_points[min_proj_index], region_points[max_proj_index]
    #Function to snap the last click point to a road mark    
    def snap_last_point(_first_click_point, _last_click_point):
       
        #Perform region growing on the last click point
        region_growth_coords,checked_indices=region_growing(point_coords, point_colors, points_kdtree, nearest_indices, region_radius, intensity_threshold, region_growth_coords)         

        if region_growth_coords:
            edge1, edge2 = find_outward_points(region_growth_coords, perp_direction)

            #Calculate the new click point based on the edges
            _last_click_point = (edge1 + edge2) * 0.5
            _last_click_point = Vector((_last_click_point[0], _last_click_point[1], _last_click_point[2]))
        else:
            print("No points found to project.")
        mark_point(_first_click_point,"_first_click_point",0.02)
        mark_point(_last_click_point,"_last_click_point",0.02)
        return _first_click_point, _last_click_point
    
    new_first_click_point, new_last_click_point = snap_last_point(first_click_point, last_click_point)
    print("Snapped to road mark")
    return new_first_click_point, new_last_click_point




#Math functions
#function to filter bad points
def filter_noise_with_dbscan(coords_list, eps=0.04, min_samples=20):
    #DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords_list)

    # Create a mask for the points belonging to clusters (excluding noise labeled as -1)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    # Filter the coordinates: keep only those points that are part of a cluster
    filtered_coords = [
        coords for coords, is_core in zip(coords_list, core_samples_mask) if is_core
    ]

    #Print the filtering results
    original_count = len(coords_list)
    filtered_count = len(filtered_coords)
    removed_count = original_count - filtered_count
    if removed_count > 0 and original_count > 3:
        print(
            f"Points have been filtered. Original amount: {original_count}, Removed: {removed_count}"
        )

    return filtered_coords

#Function to find the average intensity of given points
def get_average_intensity(indices, point_colors):
    #if indices is a NumPy array with more than one dimension flatten it
    if isinstance(indices, np.ndarray) and indices.ndim > 1:
        indices = indices.flatten()

    #if indices is a scalar, convert it to a list with a single element
    if np.isscalar(indices):
        indices = [indices]

    #gather all intensity values
    intensities = [np.average(point_colors[index]) for index in indices]

    #Sort the intensities
    sorted_intensities = sorted(intensities)

    #discard the lowest 50% of values to remove outliers
    upper_half = sorted_intensities[len(sorted_intensities) // 2:]

    #return the average intensity of the upper half
    return sum(upper_half) / len(upper_half) if upper_half else 0

#Function to find the average color of given points
def get_average_color(indices,point_colors):
    point_amount=len(indices)
    average_color = np.zeros(3, dtype=float)
    for index in indices:
        color = point_colors[index]   #rgb
        average_color += color
    average_color /= point_amount
    return average_color
 
#Function to move triangle to a line
def move_triangle_to_line(triangle, line_start, line_end):
    # Convert inputs to numpy arrays for easier calculations
    triangle_np = np.array(triangle)
    line_start_np = np.array(line_start)
    line_end_np = np.array(line_end)

    # Identify the base vertices (the two closest to the line)
    base_vertex_indices = find_base_vertices(triangle_np, line_start_np, line_end_np)
    base_vertices = triangle_np[base_vertex_indices]

    # Find the closest points on the line for the base vertices
    closest_points = [
        closest_point(vertex, line_start_np, line_end_np) for vertex in base_vertices
    ]

    # Move the base vertices to the closest points on the line
    triangle_np[base_vertex_indices] = closest_points

    #Calculate the height of the triangle to reposition the third vertex
    third_vertex_index = 3 - sum(base_vertex_indices)  # indices should be 0, 1, 2
    height_vector = triangle_np[third_vertex_index] - np.mean(base_vertices, axis=0)
    triangle_np[third_vertex_index] = np.mean(closest_points, axis=0) + height_vector

    return triangle_np.tolist()

#Function to find the base vertices of a triangle
def find_base_vertices(triangle, line_start, line_end):
    distances = [
        np.linalg.norm(closest_point(vertex, line_start, line_end) - vertex)
        for vertex in triangle
    ]
    sorted_indices = np.argsort(distances)
    return sorted_indices[:2]  # Indices of the two closest vertices

#Function to find the closest vertex to a line
def find_closest_vertex_to_line(triangle, line_start, line_end):
    min_distance = float("inf")
    closest_vertex_index = -1

    for i, vertex in enumerate(triangle):
        closest_point_on_line = closest_point(vertex, line_start, line_end)
        distance = np.linalg.norm(vertex - closest_point_on_line)
        if distance < min_distance:
            min_distance = distance
            closest_vertex_index = i

    return closest_vertex_index

#Function to find the base vertex of a triangle
def find_base_vertex(triangle, line_start, line_end):
    min_distance = float("inf")
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

#Function to find the closest point on a line to a given point
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

#Function to find the center of a cluster of points
def find_cluster_center(context, click_point, direction, range,point_coords, point_colors, points_kdtree):
    intensity_threshold = context.scene.intensity_threshold
 
    #Define the search bounds and find points within the bounds
    upper_bound = click_point + direction * range
    lower_bound = click_point - direction * range
    indices = points_kdtree.query_ball_point([upper_bound, lower_bound], range)
    indices = [i for sublist in indices for i in sublist]
    potential_points = np.array(point_coords)[indices]
    high_intensity_points = potential_points[np.average(point_colors[indices], axis=1) > intensity_threshold]

    if len(high_intensity_points) > 0:
        #Find the extremal points
        min_x = np.min(high_intensity_points[:, 0])
        max_x = np.max(high_intensity_points[:, 0])
        min_y = np.min(high_intensity_points[:, 1])
        max_y = np.max(high_intensity_points[:, 1])

        #Calculate the center of these extremal points
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        center_z = np.mean(high_intensity_points[:, 2])  #Average Z value
  
        mark_point(Vector((center_x, center_y, center_z)))
        return Vector((center_x, center_y, center_z))

    return None
  
#Function to find a high-intensity points cluster near the click point and calculate its center.  
def find_cluster_points(context, click_point, direction, range,point_coords,point_colors,points_kdtree):
    intensity_threshold = context.scene.intensity_threshold
    #Define the search bounds
    upper_bound = click_point + direction * range
    lower_bound = click_point - direction * range

    #Find all points within the search bounds
    indices = points_kdtree.query_ball_point([upper_bound, lower_bound], range)
    indices = [i for sublist in indices for i in sublist]  #Flatten the list
    potential_points = np.array(point_coords)[indices]
    high_intensity_points = potential_points[np.average(point_colors[indices], axis=1) > intensity_threshold]

    #Limit to a certain number of points for performance
    if len(high_intensity_points) > 0:
        return high_intensity_points[:500]

    return None

# function to calculate middle points of a line
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

    # Initialize the middle points list with the leftmost middle point
    middle_points = [(top_left + bottom_left) / 2]

    #Divide the remaining line into segments
    segment_width = (rightmost_x - leftmost_x) / (num_segments - 1)

    for i in range(1, num_segments):
        # Determine the segment boundaries
        x_min = leftmost_x + i * segment_width
        x_max = leftmost_x + (i + 1) * segment_width

        # Filter points in the current segment
        segment_points = coords_np[
            (coords_np[:, 0] >= x_min) & (coords_np[:, 0] < x_max)
        ]

        if len(segment_points) > 0:
            # Find the top and bottom points in this segment
            top_point = segment_points[segment_points[:, 1].argmax()]
            bottom_point = segment_points[segment_points[:, 1].argmin()]

            # Calculate the middle point
            middle_point = (top_point + bottom_point) / 2
            middle_points.append(middle_point)
            mark_point(middle_point, "middle_point")

    # Add the rightmost middle point at the end
    middle_points.append((top_right + bottom_right) / 2)

    mark_point(top_left, "top_left")
    mark_point(top_right, "top_right")
    mark_point(bottom_left, "bottom_left")
    mark_point(bottom_right, "bottom_right")

    return middle_points

#Function to Find the four corner points of the rectangle formed by the given points.
def find_rectangle_corners(points):
    # Extract X and Y coordinates
    x_coords = points[:, 0]
    y_coords = points[:, 1]

    # Find extremal values for X and Y coordinates
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)

    # Define corners based on extremal points
    bottom_left = np.array([min_x, min_y])
    bottom_right = np.array([max_x, min_y])
    top_right = np.array([max_x, max_y])
    top_left = np.array([min_x, max_y])

    # Combine corners into a single array
    corners = np.array([bottom_left, bottom_right, top_right, top_left])
    for corner in corners:
        mark_point(corner, "corner")
    return corners

#Function to calculate the middle line of the rectangle formed by the corners.
def calculate_middle_line(corners):
    # Calculate the midpoints of opposite sides
    midpoint_left = (
        corners[0] + corners[3]
    ) / 2  # Midpoint of bottom_left and top_left
    midpoint_right = (
        corners[1] + corners[2]
    ) / 2  # Midpoint of bottom_right and top_right
    mark_point(midpoint_left, "midpoint1")
    mark_point(midpoint_right, "midpoint2")
    return midpoint_left, midpoint_right

# function to snap the drawn line to the center line of the rectangle formed by the cluster.
def snap_line_to_center_line(first_click_point, second_click_point, cluster):
    corners = find_rectangle_corners(cluster)
    line_start, line_end = calculate_middle_line(corners)

    #Ensure line_start and line_end are 3D vectors
    line_start = Vector((line_start[0], line_start[1], 0))
    line_end = Vector((line_end[0], line_end[1], 0))

    new_first_click_point = line_start
    new_second_click_point = line_end

    #Calculate the direction vector as a 3D vector
    direction = (new_second_click_point - new_first_click_point).normalized()

    #Ensure the line keeps its original length
    original_length = (second_click_point - first_click_point).length
    new_second_click_point = new_first_click_point + direction * original_length

    return new_first_click_point, new_second_click_point

#Function to calculate the extreme points without outliers
def calculate_adjusted_extreme_points(points):
    
    if len(points) < 20:
        #For fewer than 20 points handle differently
        z_coords = [p[2] for p in points]
        return min(z_coords), max(z_coords)

    #Sort curb points based on their z-coordinate
    sorted_points = sorted(points, key=lambda p: p[2])

    #Determine the indices for top and bottom 10%
    ten_percent_index = len(sorted_points) // 10
    bottom_10_percent = sorted_points[:ten_percent_index]
    top_10_percent = sorted_points[-ten_percent_index:]

    #Discard the most extreme 50% within those ranges
    remaining_bottom = bottom_10_percent[len(bottom_10_percent) // 2:]
    remaining_top = top_10_percent[:len(top_10_percent) // 2]

    #calculate the average of the remaining points, prevent division by zero
    avg_lowest_point = sum((p[2] for p in remaining_bottom), 0.0) / len(remaining_bottom) if len(remaining_bottom) > 0 else 0
    avg_highest_point = sum((p[2] for p in remaining_top), 0.0) / len(remaining_top) if len(remaining_top) > 0 else 0

    return avg_lowest_point, avg_highest_point

#Function that defines a region growing algoritm
def region_growing(point_coords,point_colors,points_kdtree,nearest_indices,radius,intensity_threshold,region_growth_coords):
    # Region growing algorithm
    start_time = time.time()
    checked_indices = set()
    indices_to_check = list(nearest_indices[0])
    print("Region growing started")
    while indices_to_check:
        current_time = time.time()
        #Check if 30 seconds have passed
        if current_time - start_time > 15:
            print("Region growing stopped due to time limit.")
            break
        current_index = indices_to_check.pop()
        if current_index not in checked_indices:
            checked_indices.add(current_index)
            intensity = np.average(point_colors[current_index])    #grayscale
            if intensity > intensity_threshold:
                region_growth_coords.append(point_coords[current_index])
                _, neighbor_indices = points_kdtree.query(
                    [point_coords[current_index]], k=radius
                )
                indices_to_check.extend(
                    neighbor_index
                    for neighbor_index in neighbor_indices[0]
                    if neighbor_index not in checked_indices
                )
    print("Region growing completed in: ", time.time() - start_time)
    return region_growth_coords, checked_indices



#AI functions
#function to save the shape as an image
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
    
        if blend_file_path:
            directory = os.path.dirname(blend_file_path)
        else:
        #Prompt the user to save the file first or set a default directory
            print("please save blender project first!")
            return

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

#Function that uses Opencv for shape detection done on points    
def detect_shape_from_points(points, from_bmesh=False, scale_factor=100):

    if from_bmesh:
        #Convert bmesh vertices to numpy array
        coords_list = np.array([(point.x, point.y, point.z) for point in points])
    else:
        coords_list = np.array(points)
    
    #coords_list = filter_noise_with_dbscan(coords_list,eps=bpy.context.scene.filter_distance, min_samples=bpy.context.scene.filter_neighbors)
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



#full script only operators
#Operator to automatically import a LAS file to speed up testing
class LAS_OT_AutoOpenOperator(bpy.types.Operator):
    bl_idname = "wm.las_auto_open"
    bl_label = "Auto Open LAS File"
    def execute(self, context):

        if not os.path.exists(auto_las_file_path):
            print("Error: The file", auto_las_file_path, "does not exist.")
            return {'CANCELLED'}

        sparsity_value = bpy.context.scene.sparsity_value
        point_size = bpy.context.scene.point_size
        points_percentage=bpy.context.scene.points_percentage
        z_height_cut_off=bpy.context.scene.z_height_cut_off
        print("Opening LAS file:", auto_las_file_path)
        pointcloud_data = GetPointCloudData()
        pointcloud_data.pointcloud_load_optimized(auto_las_file_path, point_size, sparsity_value,z_height_cut_off)
        print("Finished opening LAS file:", auto_las_file_path)
        return {'FINISHED'}
     
     
         
#UI     
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
        
        row_text = self.layout.row(align=True)
        row_text.label(text="                                  Point Cloud Tools")
        
        row = layout.row(align=True)
        row.operator("wm.las_open", text="Import point cloud")
        row.prop(scene, "sparsity_value")
        

        row = layout.row(align=True)
        row.prop(scene, "ground_only")
        row.prop(scene, "z_height_cut_off")
        
        row = layout.row(align=True)
        row.operator("custom.export_to_shapefile", text="Export shapefile")  
        row.prop(scene, "points_percentage")
        
        layout.operator("custom.create_point_cloud_object", text="Create point cloud object")
        row = layout.row(align=True)
        row.operator("custom.center_pointcloud", text="Center point cloud")
        row.operator("custom.remove_point_cloud", text="Remove point cloud")
        
        row = layout.row(align=True)
        row.operator("view3d.select_points", text="Get click info")
        
        row_text = self.layout.row(align=True)
        row_text.label(text="                                      Markers")
        
        row = layout.row(align=True)
        row.operator("custom.mark_fast", text="Simple marker")
        row.operator("custom.mark_complex", text="Complex marker")
        layout.operator("view3d.selection_detection", text="Selection marker")
        row = layout.row()
        row.operator("custom.find_all_road_marks", text="Auto mark")
        row.prop(scene, "markings_threshold")
        
        row = layout.row(align=True)
        row.operator("custom.mark_fixed_triangle", text="Fixed triangle")
        row.operator("custom.mark_fixed_rectangle", text="Fixed rectangle")
        row = layout.row(align=True)
        row.operator("custom.mark_triangle", text="Triangles marker")
        row.operator("custom.auto_mark_triangle", text="Auto triangles marker")
        row = layout.row(align=True)
        row.operator("custom.mark_rectangle", text="Rectangles marker")
        row.operator("custom.auto_mark_rectangle", text="Auto rectangles marker")
        

        row = layout.row()
        row.operator("custom.mark_snapping_line", text="Line marker") 
        row.operator("custom.auto_curved_line", text="Auto line marker") 
        row = layout.row()
        row.prop(scene, "snap_to_road_mark")
        row.prop(scene, "fatline_width")
       
        layout.operator("custom.dashed_line_marking_operator", text="Dash line marker") 
        layout.operator("custom.curb_detection_operator", text="Curb marker") 
        
        row_text = self.layout.row(align=True)
        row_text.label(text="                                     Marker Settings")
        
        layout.operator("custom.remove_all_markings", text="Remove all markings")
        row = layout.row(align=True)
        row.prop(scene, "extra_z_height")
        row.prop(scene, "marking_transparency")
        layout.prop(scene, "marking_color")
        row = layout.row(align=True)
        row.prop(scene, "filter_distance")
        row.prop(scene, "filter_neighbors")
        
        layout.prop(scene, "intensity_threshold")
        
        row_text = self.layout.row(align=True)
        row_text.label(text="                                   Extra Options")
        
        row = layout.row()
        row.prop(scene, "auto_load")
        row = layout.row()
        row.prop(scene, "overwrite_existing_data")
        row = layout.row()
        row.prop(scene, "show_dots")
        row = layout.row()
        row.prop(scene, "adjust_intensity_popup")
        row = layout.row()
        row.prop(scene, "save_shape") 
        row = layout.row()
        row.prop(scene, "save_obj") 

         #Dummy space
        for _ in range(10): 
            layout.label(text="")   
            

            
#Initialization            
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
    bpy.utils.register_class(PopUpOperator)
    bpy.utils.register_class(CenterPointCloudOperator)
    bpy.utils.register_class(ExportToShapeFileOperator)
    bpy.utils.register_class(FindALlRoadMarkingsOperator)
    bpy.utils.register_class(DashedLineMarkingOperator)
    
    
    bpy.types.Scene.point_size = IntProperty(name="POINT SIZE",default=1)
    bpy.types.Scene.intensity_threshold = bpy.props.FloatProperty(
        name="Intensity threshold",
        description="Minimum intensity threshold",
        default=160, 
        min=0,
        max=255,
        subtype='UNSIGNED'  
    )
    bpy.types.Scene.markings_threshold = bpy.props.IntProperty(
        name="Max:",
        description="Maximum markings amount for auto marker",
        default=5,  
        min=1, 
        max=100,   
        subtype='UNSIGNED' 
    )
    bpy.types.Scene.points_percentage = bpy.props.IntProperty(
        name="with point %:",
        description="Percentage of points to export",
        default=2,  
        min=1, 
        max=100, 
        subtype='UNSIGNED' 
    )
    bpy.types.Scene.sparsity_value = bpy.props.FloatProperty(
        name="Sparsity:",
        description="sparsity of points rendered",
        default=0.2, 
        min=0.01, 
        max=1,
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
        name="Marking color",
        subtype='COLOR',
        description="Select Marking color",
        default=(1, 0, 0, 1),  #Default is red
        min=0.0, max=1.0,  #Colors range from 0 to 1
        size=4
        
    )
    bpy.types.Scene.marking_color = bpy.props.FloatVectorProperty(
        name="Marking color",
        subtype='COLOR',
        description="Select Marking color",
        default=(1, 0, 0, 1),  #Default is red
        min=0.0, max=1.0,  #Colors range from 0 to 1
        size=4
    )
    bpy.types.Scene.marking_transparency = bpy.props.FloatProperty(
        name="Transparency",
        description="Set the transparency for the marking (0.0 fully transparent, 1.0 fully opaque)",
        default=1,  #Default transparency is 100%
        min=0.0, max=1.0  #Transparency can range from 0.0 to 1.0
    )
    bpy.types.Scene.user_input_result = bpy.props.StringProperty(
    name="User Input Result",
    description="Stores the result from the user input pop-up",
)
    bpy.types.Scene.save_shape = bpy.props.BoolProperty(
        name="Save shape images",
        description="Saves an image after marking a shape",
        default=False,
        subtype='UNSIGNED'  
    )
    bpy.types.Scene.save_obj = bpy.props.BoolProperty(
        name="Save shape objects",
        description="Saves an OBJ file of each shape",
        default=False,
        subtype='UNSIGNED'  
    )
    bpy.types.Scene.auto_load = bpy.props.BoolProperty(
        name="Auto load auto.laz",
        description="Auto loads auto.laz on every exectuion",
        default=False,
        subtype='UNSIGNED'  
    )
    bpy.types.Scene.show_dots = bpy.props.BoolProperty(
        name="Show dots",
        description="Toggle showing feedback dots",
        default=True,
        subtype='UNSIGNED'  
    )
    bpy.types.Scene.ground_only = bpy.props.BoolProperty(
        name="Ground only",
        description="Toggle loading points from ground classification only",
        default=True,
        subtype='UNSIGNED'  
    )
    bpy.types.Scene.z_height_cut_off = bpy.props.FloatProperty(
        name="Max height",
        description="Height to cut off from ground level, 0 to not cut",
        default=0,
        min=0.0, max=100,  
        subtype='UNSIGNED'  
    )
    bpy.types.Scene.extra_z_height = bpy.props.FloatProperty(
        name="Marking height",
        description="Extra height of all markings compared to the ground level",
        default=0.05,
        min=-100, max=100, 
        subtype='UNSIGNED'  
    )
    bpy.types.Scene.filter_distance = bpy.props.FloatProperty(
        name="Filter distance",
        description="Max distance between points for filtering",
        default=0.2,
        min=0.001, max=1.0,
        subtype='UNSIGNED'  
    )
    bpy.types.Scene.filter_neighbors = bpy.props.IntProperty(
        name="Filter neighbors",
        description="Min amount of required neighbors for filtering",
        default=20,
        min=0, max=1000,  
        subtype='UNSIGNED'  
    )
    bpy.types.Scene.snap_to_road_mark= bpy.props.BoolProperty(
        name="Snap line",
        description="Snaps the line to nearby roadmark",
        default=True,
        subtype='UNSIGNED'  
    )
    bpy.types.Scene.overwrite_existing_data= bpy.props.BoolProperty(
        name="Overwrite existing point cloud data",
        description="Overwrite existing point cloud data with the same name",
        default=False,
        subtype='UNSIGNED'  
    )
    bpy.types.Scene.adjust_intensity_popup= bpy.props.BoolProperty(
        name="Intensity suggestion pop-up",
        description="Shows a pop-up to adjust intensity threshold",
        default=True,
        subtype='UNSIGNED'  
    )

#Unregister the operators and panel                                    
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
    bpy.utils.unregister_class(PopUpOperator)
    bpy.utils.unregister_class(CenterPointCloudOperator)
    bpy.utils.unregister_class(ExportToShapeFileOperator)
    bpy.utils.unregister_class(DashedLineMarkingOperator)
    
    del bpy.types.Scene.marking_transparency
    del bpy.types.Scene.marking_color
    del bpy.types.Scene.intensity_threshold
    del bpy.types.Scene.markings_threshold
    del bpy.types.Scene.fatline_width
    del bpy.types.Scene.user_input_result
    del bpy.types.Scene.save_shape
    del bpy.types.Scene.save_obj
    del bpy.types.Scene.auto_load
    del bpy.types.Scene.show_dots
    del bpy.types.Scene.snap_to_road_mark
    del bpy.types.Scene.z_height_cut_off
    del bpy.types.Scene.extra_z_height
    del bpy.types.Scene.points_percentage
    del bpy.types.Scene.sparsity_value
    del bpy.types.Scene.ground_only
    del bpy.types.Scene.overwrite_existing_data
    del bpy.types.Scene.filter_distance
    del bpy.types.Scene.filter_neighbors
    del bpy.types.Scene.adjust_intensity_popup
                   
if __name__ == "__main__":
    register()
    
    if(context.scene.auto_load):
        bpy.ops.wm.las_auto_open()
        
 
