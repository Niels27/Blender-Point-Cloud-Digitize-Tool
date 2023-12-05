#library imports
import bpy
from bpy import context
from bpy_extras.view3d_utils import region_2d_to_location_3d
import bmesh
from bpy_extras import view3d_utils
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
from ..utils.blender_utils import *
from ..utils.digitizing_utils import *
from ..utils.math_utils import *

#global variables
last_processed_index = 0 #Global variable to keep track of the last processed index, for numbering road marks
pointcloud_data = GetPointCloudData()
point_coords = pointcloud_data.point_coords
point_colors = pointcloud_data.point_colors
points_kdtree=  pointcloud_data.points_kdtree

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
            average_color = get_average_color(nearest_indices[0], point_colors)
             
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
                average_color = get_average_color(nearest_indices[0],point_colors)
                    
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
                    create_dots_shape(rectangle_coords)
                      
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
            average_color = get_average_color(nearest_indices[0],point_colors)
             
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
        average_color = get_average_color(nearest_indices[0],point_colors)

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
        average_color = get_average_color(nearest_indices[0],point_colors)
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
            average_color = get_average_color(nearest_indices[0],point_colors)
             
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
            average_color = get_average_color(nearest_indices[0],point_colors)
             
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
        average_color = get_average_color(nearest_indices[0],point_colors)

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
        
        if context.object:
            bpy.ops.object.select_all(action='DESELECT')
            context.view_layer.objects.active = context.object
            context.object.select_set(True)
            bpy.ops.object.delete()  
            
        CurvedLineMarkOperator._is_running = False
        print("Operator was properly cancelled")
        return {'CANCELLED'}
    
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
            average_color = get_average_color(nearest_indices[0],point_colors)
             
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
        