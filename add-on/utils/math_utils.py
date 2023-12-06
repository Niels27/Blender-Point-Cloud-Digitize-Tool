#library imports
import bpy
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
from functools import partial
from sklearn.cluster import DBSCAN 
from scipy.interpolate import UnivariateSpline, make_interp_spline,CubicSpline

def get_average_intensity(indices,point_colors):
    #If indices is a NumPy array with more than one dimension, flatten it
    if isinstance(indices, np.ndarray) and indices.ndim > 1:
        indices = indices.flatten()

    #If indices is a scalar, convert it to a list with a single element
    if np.isscalar(indices):
        indices = [indices]

    total_intensity = 0.0
    point_amount = len(indices)
    for index in indices:
        
        intensity = np.average(point_colors[index])
        total_intensity += intensity

    return total_intensity / point_amount

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

def get_average_color(indices, point_colors):
    point_amount=len(indices)
    average_color = np.zeros(3, dtype=float)
    for index in indices:
        color = point_colors[index] 
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

def region_growing(point_coords, point_colors, points_kdtree, nearest_indices, radius, intensity_threshold, region_growth_coords):
#Region growing algorithm
    start_time = time.time()
    checked_indices = set()
    indices_to_check = list(nearest_indices[0])
    print("Region growing started")
    while indices_to_check:   
        current_index = indices_to_check.pop()
        if current_index not in checked_indices:
            checked_indices.add(current_index)
            intensity = np.average(point_colors[current_index]) #* 255  #grayscale
            if intensity>intensity_threshold:
                region_growth_coords.append(point_coords[current_index])
                _, neighbor_indices = points_kdtree.query([point_coords[current_index]], k=radius)
                indices_to_check.extend(neighbor_index for neighbor_index in neighbor_indices[0] if neighbor_index not in checked_indices)
    print("Region growing completed in: ", time.time()-start_time)
    return region_growth_coords
    
#module imports
from ..utils.digitizing_utils import mark_point