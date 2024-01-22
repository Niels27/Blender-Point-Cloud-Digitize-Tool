# library imports
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
from scipy.interpolate import UnivariateSpline, make_interp_spline, CubicSpline


#Math functions
#function to filter bad points
def filter_noise_with_dbscan(coords_list, eps=0.15, min_samples=20):
    #DBSCAN clustering
    eps=float(eps)
    min_samples=int(min_samples)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords_list)

    #Create a mask for the points belonging to clusters (excluding noise labeled as -1)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    #Filter the coordinates: keep only those points that are part of a cluster
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
    #Convert inputs to numpy arrays for easier calculations
    triangle_np = np.array(triangle)
    line_start_np = np.array(line_start)
    line_end_np = np.array(line_end)

    #Identify the base vertices (the two closest to the line)
    base_vertex_indices = find_base_vertices(triangle_np, line_start_np, line_end_np)
    base_vertices = triangle_np[base_vertex_indices]

    #Find the closest points on the line for the base vertices
    closest_points = [
        closest_point(vertex, line_start_np, line_end_np) for vertex in base_vertices
    ]

    #Move the base vertices to the closest points on the line
    triangle_np[base_vertex_indices] = closest_points

    #Calculate the height of the triangle to reposition the third vertex
    third_vertex_index = 3 - sum(base_vertex_indices)  #indices should be 0, 1, 2
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
    return sorted_indices[:2]  #Indices of the two closest vertices

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
        segment_points = coords_np[
            (coords_np[:, 0] >= x_min) & (coords_np[:, 0] < x_max)
        ]

        if len(segment_points) > 0:
            #Find the top and bottom points in this segment
            top_point = segment_points[segment_points[:, 1].argmax()]
            bottom_point = segment_points[segment_points[:, 1].argmin()]

            #Calculate the middle point
            middle_point = (top_point + bottom_point) / 2
            middle_points.append(middle_point)
            mark_point(middle_point, "middle_point")

    #Add the rightmost middle point at the end
    middle_points.append((top_right + bottom_right) / 2)

    mark_point(top_left, "top_left")
    mark_point(top_right, "top_right")
    mark_point(bottom_left, "bottom_left")
    mark_point(bottom_right, "bottom_right")

    return middle_points

#Function to Find the four corner points of the rectangle formed by the given points.
def find_rectangle_corners(points):
    #Extract X and Y coordinates
    x_coords = points[:, 0]
    y_coords = points[:, 1]

    #Find extremal values for X and Y coordinates
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)

    #Define corners based on extremal points
    bottom_left = np.array([min_x, min_y])
    bottom_right = np.array([max_x, min_y])
    top_right = np.array([max_x, max_y])
    top_left = np.array([min_x, max_y])

    #Combine corners into a single array
    corners = np.array([bottom_left, bottom_right, top_right, top_left])
    for corner in corners:
        mark_point(corner, "corner")
    return corners

#Function to calculate the middle line of the rectangle formed by the corners.
def calculate_middle_line(corners):
    #Calculate the midpoints of opposite sides
    midpoint_left = (
        corners[0] + corners[3]
    ) / 2  #Midpoint of bottom_left and top_left
    midpoint_right = (
        corners[1] + corners[2]
    ) / 2  #Midpoint of bottom_right and top_right
    mark_point(midpoint_left, "midpoint1")
    mark_point(midpoint_right, "midpoint2")
    return midpoint_left, midpoint_right

#function to snap the drawn line to the center line of the rectangle formed by the cluster.
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
def region_growing(point_coords,point_colors,points_kdtree,nearest_indices,radius,intensity_threshold, time_limit=10):
    #Region growing algorithm
    start_time = time.time()
    checked_indices = set()
    indices_to_check = list(nearest_indices[0])
    region_growth_coords=[]
    print("Region growing started")
    while indices_to_check:
        current_time = time.time()
        #Check if 30 seconds have passed
        if current_time - start_time > time_limit:
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



# module imports
from ..utils.digitizing_utils import mark_point

