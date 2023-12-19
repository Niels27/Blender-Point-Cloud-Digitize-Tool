#library imports
import bpy
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

#global variables
last_processed_index = 0 #Global variable to keep track of the last processed index, for numbering road marks

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

        self.prev_end_point = coord_3d_end

        #Create a rectangle object on top of the line
        create_rectangle_line_object(coord_3d_start, coord_3d_end)

    def cancel(self, context):
        DrawStraightFatLineOperator._is_running = False  #Reset the flag when the operator is cancelled
        print("Operator was properly cancelled")  #Debug message
        return {'CANCELLED'}
    
#Operator to draw simple shapes in the viewport using mouseclicks   
class SimpleMarkOperator(bpy.types.Operator):
    bl_idname = "view3d.mark_fast"
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
                if(average_intensity-intensity_threshold>intensity_difference_threshold):
                    bpy.ops.wm.correction_pop_up('INVOKE_DEFAULT', average_intensity=average_intensity, adjust_action='HIGHER')
                    return {'PASS_THROUGH'}
                #Region growing algorithm
                region_growth_coords,checked_indices=region_growing(point_coords, point_colors, points_kdtree, nearest_indices, radius, intensity_threshold, region_growth_coords)
            else:
                print("no road markings found")
                #if the average intensity is way lower than the threshold, give a warning
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
    bl_idname = "view3d.mark_complex"
    bl_label = "Mark complex Road Markings"
    bl_description= "Mark shapes using multiple points"
    
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
                if(average_intensity-intensity_threshold>intensity_difference_threshold):
                    bpy.ops.wm.correction_pop_up('INVOKE_DEFAULT', average_intensity=average_intensity, adjust_action='HIGHER') 
                    return {'PASS_THROUGH'}
                #Region growing algorithm
                region_growth_coords,checked_indices=region_growing(point_coords, point_colors, points_kdtree, nearest_indices, radius, intensity_threshold, region_growth_coords)
            else:
                print("no road markings found")
                #if the average intensity is way lower than the threshold, give a warning
                bpy.ops.wm.correction_pop_up('INVOKE_DEFAULT', average_intensity=average_intensity, adjust_action='LOWER')
                return {'PASS_THROUGH'}
            
            if region_growth_coords:
                #Create a single mesh for the combined rectangles
                create_dots_shape(region_growth_coords)
        #If escape is pressed, stop the operator 
        elif event.type == 'ESC':
            return self.cancel(context)  

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
                
                filtered_triangle_coords=filter_noise_with_dbscan(region_growth_coords)
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
                filtered_points = filter_noise_with_dbscan(points)
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
                filtered_current_triangle_coords=filter_noise_with_dbscan(region_growth_coords)
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
        mark_point(median_point, "dash line center")

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
            
            mark_point(new_cluster_center, "dash line center")

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



#module imports
from ..utils.blender_utils import GetPointCloudData, is_mouse_in_3d_view, set_view_to_top, get_click_point_in_3d
from ..utils.math_utils import calculate_adjusted_extreme_points
from ..utils.digitizing_utils import mark_point, create_shape, create_rectangle_line_object, create_polyline,create_flexible_triangle, create_dots_shape, draw_line, create_flexible_rectangle,create_fixed_square,draw_fixed_triangle
from ..utils.math_utils import get_average_color, get_average_intensity, filter_noise_with_dbscan, move_triangle_to_line, region_growing