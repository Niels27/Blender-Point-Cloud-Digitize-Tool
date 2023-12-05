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


#module imports
from ..utils.math_utils import *
from ..utils.blender_utils import *

#global variables
collection_name = "Collection" #the collection name in blender
point_cloud_point_size =  1 #The size of the points in the point cloud
pointcloud_data = GetPointCloudData()
point_coords = pointcloud_data.point_coords
point_colors = pointcloud_data.point_colors
points_kdtree = pointcloud_data.points_kdtree

class LAS_OT_OpenOperator(bpy.types.Operator):
    
    bl_idname = "wm.las_open"
    bl_label = "Open LAS/LAZ File"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    
    
    def execute(self, context):
        start_time = time.time()
      
        bpy.context.scene["Filepath to the loaded pointcloud"] = self.filepath
        sparsity_value = bpy.context.scene.sparsity_value
        point_size = bpy.context.scene.point_size
        pointcloud_data.pointcloud_load_optimized(self.filepath, point_size, sparsity_value)
        print("Opened LAS/LAZ file: ", self.filepath,"in %s seconds" % (time.time() - start_time))
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
#Operator to remove all drawn markings from the scene collection
class RemoveAllObjectsOperator(bpy.types.Operator):
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
 

#Prints the point cloud coordinates and the average color & intensity around mouse click        
class GetPointsInfoOperator(bpy.types.Operator):
    bl_idname = "view3d.select_points"
    bl_label = "Get Points information"

    def modal(self, context, event):
        
        global point_coords, point_colors, points_kdtree
        
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

                average_intensity = get_average_intensity(nearest_indices[0],point_colors)
                #Calculate the average color
                average_color = np.mean(nearest_colors, axis=0)
                
                clicked_on_white = "Clicked on roadmark" if is_click_on_white(self, context, location,points_kdtree,point_colors) else "No roadmark detected"
                    
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

#Custom operator for the pop-up dialog
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
    #Define the custom draw method
    def draw(self, context):
        layout = self.layout
        col = layout.column()
        
        #Add custom buttons to the UI
        col.label(text=f"{self.click_to_correct} Click(s) might be incorrect")
        col.label(text="Choose an action:")
        col.separator()
        
        #Use 'props_enum' to create buttons for each enum property
        layout.props_enum(self, "action")
        
    def execute(self, context):
        #Access the stored data to perform the correction
        coord_3d_start = Vector(self.start_point)
        coord_3d_end = Vector(self.end_point)
        click_to_correct = self.click_to_correct
        
        #print("User chose to", "draw" if self.action == 'DRAW' else "correct")
        #Based on the user's choice, either draw or initiate a correction process
        context.scene.user_input_result = self.action
       
        if self.action == 'CORRECT':
            '''coord_3d_start, coord_3d_end = snap_to_road_mark(self,context, coord_3d_start, coord_3d_end, click_to_correct)
            create_rectangle_line_object(coord_3d_start, coord_3d_end)'''
            print("Corrected line drawing")
        
        elif self.action == ('CANCEL'):
            print("Canceled line drawing")
            return {'CANCELLED'}
        
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)
            