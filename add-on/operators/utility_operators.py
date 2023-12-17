#library imports
import bpy
from bpy import context
from bpy_extras.view3d_utils import region_2d_to_location_3d
import bmesh
import numpy as np
import open3d as o3d
import time
import math
import laspy as lp
import scipy
from scipy.spatial import KDTree, cKDTree, ConvexHull
from scipy.spatial.distance import cdist
from mathutils import Vector, Matrix
import mathutils
import pandas as pd
import geopandas as gpd

#global variables
collection_name = "Collection" #the collection name in blender
point_cloud_point_size =  1 #The size of the points in the point cloud
undo_stack = [] #Keeps track of all objects created/removed for undo functions
redo_stack = []#Keeps track of all objects created/removed for redo functions

#Operator to import las/laz files                         
class LAS_OT_OpenOperator(bpy.types.Operator):
    
    bl_idname = "wm.las_open"
    bl_label = "Open LAS/LAZ File"
    bl_description = "Import a LAS/LAZ file"
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    
    def execute(self, context):
        start_time = time.time()
      
        bpy.context.scene["Filepath to the loaded pointcloud"] = self.filepath
        sparsity_value = bpy.context.scene.sparsity_value
        point_size = bpy.context.scene.point_size
        points_percentage=bpy.context.scene.points_percentage
        z_height_cut_off=bpy.context.scene.z_height_cut_off
        pointcloud_data = GetPointCloudData()
        pointcloud_data.pointcloud_load_optimized(self.filepath, point_size, sparsity_value,points_percentage,z_height_cut_off)
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
                view3d = context.space_data
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

    
#module imports
from ..utils.blender_utils import GetPointCloudData, is_mouse_in_3d_view, redraw_viewport, export_as_shapefile, is_click_on_white,set_view_to_top
from ..utils.math_utils import get_average_intensity