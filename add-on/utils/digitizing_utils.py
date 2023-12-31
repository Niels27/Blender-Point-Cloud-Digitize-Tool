#library imports
import bpy
from bpy_extras.view3d_utils import region_2d_to_location_3d
import gpu
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
shape_counter=1 #Keeps track of amount of shapes drawn, used to number them

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


#module imports
from ..utils.blender_utils import is_click_on_white, prepare_object_for_export
from ..utils.math_utils import create_middle_points, filter_noise_with_dbscan, move_triangle_to_line, region_growing