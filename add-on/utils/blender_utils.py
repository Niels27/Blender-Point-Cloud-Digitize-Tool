# library imports
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

# global variables
save_json = False
point_cloud_name = "Point cloud"
point_cloud_point_size = 1
collection_name = "Collection" #the default collection name in blender

# Blender utility functions


# Function to Check whether the mouseclick happened in the viewport or elsewhere
def is_mouse_in_3d_view(context, event):
    # Identify the 3D Viewport area and its regions
    view_3d_area = next(
        (area for area in context.screen.areas if area.type == "VIEW_3D"), None
    )
    if view_3d_area is not None:
        toolbar_region = next(
            (region for region in view_3d_area.regions if region.type == "TOOLS"), None
        )
        ui_region = next(
            (region for region in view_3d_area.regions if region.type == "UI"), None
        )
        view_3d_window_region = next(
            (region for region in view_3d_area.regions if region.type == "WINDOW"), None
        )

        # Check if the mouse is inside the 3D Viewport's window region
        if view_3d_window_region is not None:
            mouse_inside_view3d = (
                view_3d_window_region.x
                < event.mouse_x
                < view_3d_window_region.x + view_3d_window_region.width
                and view_3d_window_region.y
                < event.mouse_y
                < view_3d_window_region.y + view_3d_window_region.height
            )

            # Exclude areas occupied by the toolbar or UI regions
            if toolbar_region is not None:
                mouse_inside_view3d &= not (
                    toolbar_region.x
                    < event.mouse_x
                    < toolbar_region.x + toolbar_region.width
                    and toolbar_region.y
                    < event.mouse_y
                    < toolbar_region.y + toolbar_region.height
                )
            if ui_region is not None:
                mouse_inside_view3d &= not (
                    ui_region.x < event.mouse_x < ui_region.x + ui_region.width
                    and ui_region.y < event.mouse_y < ui_region.y + ui_region.height
                )

            return mouse_inside_view3d

    return False  # Default to False if checks fail.


# function to get 3D click point
def get_click_point_in_3d(context, event):
    # Convert the mouse position to 3D space
    coord_3d = view3d_utils.region_2d_to_location_3d(
        context.region,
        context.space_data.region_3d,
        (event.mouse_region_x, event.mouse_region_y),
        Vector((0, 0, 0)),
    )
    return coord_3d



#Function to store obj state
def prepare_object_for_export(obj):
    
    global collection_name
    #Get the path of the current Blender file
    blend_file_path = bpy.data.filepath
    directory = os.path.dirname(blend_file_path)
    shapes_dir = os.path.join(directory, "webbased poc/static/objects")
    
    #check to ensure the object is valid
    if obj is None:
        print("Invalid object for export.")
        return None

    #Check if the necessary context attributes are available
    if 'selected_objects' not in dir(bpy.context) or 'view_layer' not in dir(bpy.context):
        print("Required context attributes are not available.")
        #When functions get called outside of the UI, export without using the context but using the collection
        export_objects_from_collection(collection_name, shapes_dir)
    #when functions get called from the UI, export using the context
    else:
        try:
            bpy.context.view_layer.objects.active = obj  #Set as active object
            bpy.ops.object.mode_set(mode='OBJECT') #force object mode
            bpy.ops.object.select_all(action='DESELECT')  #Deselect all objects
            obj.select_set(True)  #Select the current object
            obj=set_origin_to_geometry_center(obj)
            save_shape_checkbox = bpy.context.scene.save_shape
            if(save_shape_checkbox):
                save_shape_as_image(obj)
                
            save_obj_checkbox = bpy.context.scene.save_obj
            if(save_obj_checkbox):
                export_shape_as_obj(obj, obj.name,directory)
                
        except Exception as e:
            print(f"Error during preparation for export: {e}")
            return None
            
#Function to export objects as OBJ files 
def export_shape_as_obj(obj, name, directory):
    
    #global properties from context
    global collection_name
    marking_color = bpy.context.scene.marking_color 
    transparency = bpy.context.scene.marking_transparency
    shape_material = bpy.data.materials.new(name="shape_material")
    shape_material.diffuse_color = (marking_color[0], marking_color[1], marking_color[2], transparency)

    #Assign the material to the object
    if len(obj.data.materials) > 0:
        obj.data.materials[0] = shape_material
    else:
        obj.data.materials.append(shape_material)
        
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
    obj.select_set(False)
 
#Function to set origin to geometry center based on an object
def set_origin_to_geometry_center(obj, return_obj=True):
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
                
    if return_obj:
        return obj

def export_mesh_to_obj(mesh, file_path, mtl_name):
    #Open the file for writing the OBJ file
    with open(file_path, 'w') as file:
        file.write(f"mtllib {mtl_name}.mtl\n")  # MTL file reference

        #Write vertices
        for vertex in mesh.vertices:
            file.write(f"v {vertex.co.x} {vertex.co.y} {vertex.co.z}\n")

        #Write faces (assuming the mesh is using the same material)
        file.write(f"usemtl {mtl_name}\n")  #Use the material
        for face in mesh.polygons:
            vertices = face.vertices[:]
            face_line = "f"
            for vert in vertices:
                face_line += f" {vert + 1}"  # OBJ files are 1-indexed
            file.write(f"{face_line}\n")

def export_material_to_mtl(mtl_path, mtl_name):
    #Open the file for writing the MTL file
    with open(mtl_path, 'w') as file:
        #Define a simple red material
        file.write(f"newmtl {mtl_name}\n")
        file.write("Ns 10.0000\n")  #Specular exponent
        file.write("Ka 1.0000 0.0000 0.0000\n")  #Ambient color (red)
        file.write("Kd 1.0000 0.0000 0.0000\n")  #Diffuse color (red)
        file.write("Ks 0.5000 0.5000 0.5000\n")  #Specular color 
        file.write("Ni 1.4500\n")  # Optical density 
        file.write("d 1.0000\n")  # transparency
        file.write("illum 2\n")  # Illumination 

def export_objects_from_collection(collection_name, export_directory):
    # Ensure the export directory exists
    os.makedirs(export_directory, exist_ok=True)

    # Retrieve the collection
    collection = bpy.data.collections.get(collection_name)
    if not collection:
        print(f"Collection '{collection_name}' not found")
        return

    #Iterate over objects in the collection
    for obj in collection.objects:
        #Check if object is a mesh
        if obj.type == 'MESH':
            obj=set_origin_to_geometry_center(obj)
            #Define file paths for the OBJ and MTL files
            obj_file_path = os.path.join(export_directory, f"{obj.name}.obj")
            mtl_file_path = os.path.join(export_directory, f"{obj.name}.mtl")
            print(f"Exporting {obj.name} to {obj_file_path} and {mtl_file_path}")

            #Apply transformations and get mesh data
            mesh = obj.to_mesh()
            export_mesh_to_obj(mesh, obj_file_path, obj.name)
            obj.to_mesh_clear()  #Clear the temporary mesh data

            #Export the material
            export_material_to_mtl(mtl_file_path, obj.name)


# Function to force top view in the viewport
def set_view_to_top(context):
    # Find the first 3D View
    for area in context.screen.areas:
        if area.type == "VIEW_3D":
            # Get the 3D View
            space_data = area.spaces.active

            # Get the current rotation in Euler angles
            current_euler = space_data.region_3d.view_rotation.to_euler()

            # Set the Z-axis rotation to 0, retaining X and Y rotations
            new_euler = mathutils.Euler((math.radians(0), 0, current_euler.z), "XYZ")
            space_data.region_3d.view_rotation = new_euler.to_quaternion()

            # Update the view
            area.tag_redraw()
            break

#Function to save an image of a shape
def save_shape_as_image(obj):
    
    obj_name=obj.name
    save_shape_checkbox = bpy.context.scene.save_shape
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

# Function to clears the viewport and delete the draw handler
def redraw_viewport():
    # global draw_handler  #Reference the global variable
    draw_handler = bpy.app.driver_namespace.get("my_draw_handler")

    if draw_handler is not None:
        # Remove the handler reference, stopping the draw calls
        bpy.types.SpaceView3D.draw_handler_remove(draw_handler, "WINDOW")
        # draw_handler = None
        del bpy.app.driver_namespace["my_draw_handler"]

        print("Draw handler removed successfully.")
        print("Stopped drawing the point cloud.")

    # Redraw the 3D view to reflect the removal of the point cloud
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()

    print("viewport redrawn")

# Function to install libraries from a list using pip
def install_libraries(library_list):
    for library in library_list:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", library])
            print(f"Successfully installed {library}")
        except subprocess.CalledProcessError as e:
            print(f"Error installing {library}: {e}")

# Function to update libraries from a list using pip
def update_libraries(library_list):
    for library in library_list:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--upgrade", library]
            )
            print(f"Successfully updated {library}")
        except subprocess.CalledProcessError as e:
            print(f"Error updating {library}: {e}")

# Function to uninstall libraries from a list using pip
def uninstall_libraries(library_list):
    for library in library_list:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", library])
            print(f"Successfully uninstall {library}")
        except subprocess.CalledProcessError as e:
            print(f"Error uninstall {library}: {e}")



