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
# Singleton class to save point cloud data
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

    # Function to load the point cloud, store it's data and draw it using openGL, optimized version
    def pointcloud_load_optimized(self, path, point_size, sparsity_value, z_height_cut_off):
        start_time = time.time()
        print("Started loading point cloud.."),
        global use_pickled_kdtree, point_cloud_name, point_cloud_point_size, save_json
        overwrite_data = bpy.context.scene.overwrite_existing_data

        base_file_name = os.path.basename(path)
        point_cloud_name = base_file_name
        directory_path = os.path.dirname(path)
        blend_file_path = bpy.data.filepath
        if blend_file_path:
            directory = os.path.dirname(blend_file_path)
        else:
            # Prompt the user to save the file first
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

        if (
            not os.path.exists(os.path.join(stored_data_path, file_name_points))
            or overwrite_data
        ):
            point_cloud = lp.read(path)
            ground_code = 2  # the ground is usually 2

            print(f"Using classification code for GROUND: {ground_code}")
            use_ground_points_only = bpy.context.scene.ground_only
            if use_ground_points_only:
                # Filter points based on classification
                ground_points_mask = point_cloud.classification == ground_code
                if ground_points_mask.any():
                    # Applying the ground points mask
                    points_a = np.vstack(
                        (
                            point_cloud.x[ground_points_mask],
                            point_cloud.y[ground_points_mask],
                            point_cloud.z[ground_points_mask],
                        )
                    ).transpose()
                    colors_a = (
                        np.vstack(
                            (
                                point_cloud.red[ground_points_mask],
                                point_cloud.green[ground_points_mask],
                                point_cloud.blue[ground_points_mask],
                            )
                        ).transpose()
                        / 65535
                    )
                else:
                    print("classification ", ground_code, " not found")
            else:
                points_a = np.vstack(
                    (point_cloud.x, point_cloud.y, point_cloud.z)
                ).transpose()
                colors_a = (
                    np.vstack(
                        (point_cloud.red, point_cloud.green, point_cloud.blue)
                    ).transpose()
                    / 65535
                )

            # Convert points to float32
            points_a = points_a.astype(np.float32)
            # Convert colors to uint8
            colors_a = (colors_a * 255).astype(np.uint8)

            # Sort the points based on the Z coordinate
            sorted_points = points_a[points_a[:, 2].argsort()]

            # Determine the cutoff index for the lowest %
            cutoff_index = int(len(sorted_points) * 0.1)

            # Calculate the average Z value of the lowest % of points
            road_base_level = np.mean(sorted_points[:cutoff_index, 2])

            print("Estimated road base level:", road_base_level)

            # Store the original coordinates
            self.original_coords = np.copy(points_a)

            if z_height_cut_off > 0:
                # Filter points with Z coordinate > 0.5
                print("Number of points before filtering:", len(points_a))
                mask = points_a[:, 2] <= (road_base_level + z_height_cut_off)
                points_a = points_a[mask]
                colors_a = colors_a[mask]
                print("Number of points after filtering:", len(points_a))

            # Shifting coords
            points_a_avg = np.mean(points_a, axis=0)
            points_a = points_a - points_a_avg

            # Storing Shifting coords
            np.save(os.path.join(stored_data_path, file_name_avg_coords), points_a_avg)

            # Storing the centered coordinate arrays as npy file
            np.save(os.path.join(stored_data_path, file_name_points), points_a)
            np.save(os.path.join(stored_data_path, file_name_colors), colors_a)

        else:
            points_a = np.load(os.path.join(stored_data_path, file_name_points))
            colors_a = np.load(os.path.join(stored_data_path, file_name_colors))
            self.original_coords = np.load(
                os.path.join(stored_data_path, file_name_avg_coords)
            )

        # Store point data and colors globally
        self.point_coords = points_a
        self.point_colors = colors_a
        point_cloud_point_size = point_size

        print("point cloud loaded in: ", time.time() - start_time)

        step = int(1 / sparsity_value)

        # Evenly sample points using the provided sparsity value
        reduced_points = points_a[::step]
        reduced_colors = colors_a[::step]

        # Save json file of point cloud data
        if save_json:
            export_as_json(
                reduced_points, reduced_colors, JSON_data_path, point_cloud_name
            )

        # Function to save KD-tree with pickle and gzip
        def save_kdtree_pickle_gzip(file_path, kdtree):
            with gzip.open(
                file_path, "wb", compresslevel=1
            ) as f:  # compresslevel from 1-9, low-high compression
                pickle.dump(kdtree, f)

        # Function to load KD-tree with pickle and gzip
        def load_kdtree_pickle_gzip(file_path):
            with gzip.open(file_path, "rb") as f:
                return pickle.load(f)

        use_pickled_kdtree = True
        if use_pickled_kdtree:
            # KDTree handling
            kdtree_pickle_path = os.path.join(stored_data_path, file_name_kdtree_pickle)
            if not os.path.exists(kdtree_pickle_path) or overwrite_data:
                # Create the kdtree if it doesn't exist
                print("creating cKDTree..")
                start_time = time.time()
                self.points_kdtree = cKDTree(np.array(self.point_coords))
                save_kdtree_pickle_gzip(kdtree_pickle_path, self.points_kdtree)
                print(
                    "Compressed cKD-tree created at:", kdtree_pickle_path, " in:", time
                )
            else:
                print("kdtree found at: ", kdtree_pickle_path, "loading..")
                self.points_kdtree = load_kdtree_pickle_gzip(kdtree_pickle_path)
                print(
                    "Compressed cKD-tree loaded from gzip file in:",
                    time.time() - start_time,
                )
        else:
            # KDTree handling
            kdtree_path = os.path.join(stored_data_path, file_name_kdtree)
            self.points_kdtree = load_kdtree_from_file(kdtree_path)
            if (
                not os.path.exists(kdtree_pickle_path)
                or bpy.types.Scene.overwrite_existing_data == False
            ):
                # create the kdtree if it doesn't exist
                self.points_kdtree = cKDTree(np.array(self.point_coords))
                print("kdtree created in: ", time.time() - start_time)
                # Save the kdtree to a file
                save_kdtree_to_file(kdtree_path, self.points_kdtree)
                print("kdtree saved in: ", time.time() - start_time, "at", kdtree_path)

        try:
            redraw_viewport()
            draw_handler = bpy.app.driver_namespace.get("my_draw_handler")

            if draw_handler is None:
                # colors_ar should be in uint8 format
                reduced_colors = reduced_colors / 255.0  # Normalize to 0-1 range
                # Converting to tuple
                coords = tuple(map(tuple, reduced_points))
                colors = tuple(map(tuple, reduced_colors))

                shader = gpu.shader.from_builtin("3D_FLAT_COLOR")
                batch = batch_for_shader(
                    shader, "POINTS", {"pos": coords, "color": colors}
                )

                # the draw function
                def draw():
                    gpu.state.point_size_set(point_size)
                    bgl.glEnable(bgl.GL_DEPTH_TEST)
                    batch.draw(shader)
                    bgl.glDisable(bgl.GL_DEPTH_TEST)

                # Define draw handler to acces the drawn point cloud later on
                draw_handler = bpy.types.SpaceView3D.draw_handler_add(
                    draw, (), "WINDOW", "POST_VIEW"
                )
                # Store the draw handler reference in the driver namespace
                bpy.app.driver_namespace["my_draw_handler"] = draw_handler

                # Calculate the bounding box of the point cloud
                min_coords = np.min(self.point_coords, axis=0)
                max_coords = np.max(self.point_coords, axis=0)
                bbox_center = (min_coords + max_coords) / 2

                # Get the active 3D view
                for area in bpy.context.screen.areas:
                    if area.type == "VIEW_3D":
                        break

                # Set the view to look at the bounding box center from above at a certain height
                view3d = area.spaces[0]
                camera_height = 50
                view3d.region_3d.view_location = (
                    bbox_center[0],
                    bbox_center[1],
                    camera_height,
                )  # X, Y, z meters height
                # view3d.region_3d.view_rotation = bpy.context.scene.camera.rotation_euler  #Maintaining the current rotation
                view3d.region_3d.view_distance = (
                    camera_height  # Distance from the view point
                )
                print(
                    "openGL point cloud drawn in:",
                    time.time() - start_time,
                    "using ",
                    sparsity_value * 100,
                    " percent of points (",
                    len(reduced_points),
                    ") points",
                )

            else:
                print("Draw handler already exists, skipping drawing")
        except Exception as e:
            # Handle any other exceptions that might occur
            print(f"An error occurred: {e}")

# Function to load KDTree from a file
def load_kdtree_from_file(file_path):
    if os.path.exists(file_path):
        print("Existing kdtree found. Loading...")
        start_time = time.time()
        with open(file_path, "r") as file:
            kdtree_data = json.load(file)
        # Convert the loaded points back to a Numpy array
        points = np.array(kdtree_data["points"])
        print(
            "Loaded kdtree in: %s seconds" % (time.time() - start_time),
            "from: ",
            file_path,
        )
        return cKDTree(points)
    else:
        return None

# Function to save KDTree to a file
def save_kdtree_to_file(file_path, kdtree):
    kdtree_data = {"points": kdtree.data.tolist()}  # Convert Numpy array to Python list
    with open(file_path, "w") as file:
        json.dump(kdtree_data, file)

        if bpy.context.object:
            bpy.ops.object.select_all(action="DESELECT")
            bpy.context.view_layer.objects.active = bpy.context.object
            bpy.context.object.select_set(True)
            bpy.ops.object.delete()
    print("saved kdtree to", file_path)

# Function to export the point cloud as a shapefile
def export_as_shapefile(points, points_percentage=100, epsg_value=28992):
    global point_cloud_name
    start_time = time.time()
    num_points = len(points)
    num_points_to_keep = math.ceil(num_points * (points_percentage / 100))
    step = math.ceil(num_points / num_points_to_keep)
    points = points[::step]

    print(
        "exporting as shapefile using ",
        points_percentage,
        " percent of points: ",
        "(",
        len(points),
        " points)",
    )
    point_geometries = [Point(x, y, z) for x, y, z in points]
    crs = "EPSG:" + str(epsg_value)
    gdf = gpd.GeoDataFrame(geometry=point_geometries, crs=crs)
    print(
        "exported as a shapefile in: ", time.time() - start_time, " Saving to file..."
    )

    # Get the directory of the current Blender file
    blend_file_path = bpy.data.filepath
    directory = os.path.dirname(blend_file_path)

    # Create a folder 'road_mark_images' if it doesn't exist
    shapefile_dir = os.path.join(directory, "shapefiles")
    if not os.path.exists(shapefile_dir):
        os.makedirs(shapefile_dir)
    # Define the path for the output shapefile
    output_shapefile_path = os.path.join(shapefile_dir, f"{point_cloud_name}_shapefile")
    gdf.to_file(output_shapefile_path)
    print("saved shapefile to: ", shapefile_dir, " in: ", time.time() - start_time)

# Function to export point cloud as JSON
def export_as_json(point_coords, point_colors, JSON_data_path, point_cloud_name):
    start_time = time.time()
    print("exporting point cloud data as JSON")
    # Adjusting the structure to match the expected format
    point_cloud_data = [
        {
            "x": round(float(point[0]), 2),
            "y": round(float(point[1]), 2),
            "z": round(float(point[2]), 2),
            "color": {"r": int(color[0]), "g": int(color[1]), "b": int(color[2])},
        }
        for point, color in zip(point_coords, point_colors)
    ]

    # Save as compact JSON to reduce file size
    json_data = json.dumps(point_cloud_data, separators=(",", ":")).encode("utf-8")

    # Defines file paths
    json_file_path = os.path.join(
        JSON_data_path, f"{point_cloud_name}_points_colors.json.gz"
    )

    # Write to JSON file
    print("Compressing JSON...")
    with gzip.open(json_file_path, "wb") as f:
        f.write(json_data)

    print(
        "Combined JSON file compressed and saved at: ",
        JSON_data_path,
        "in: ",
        time.time() - start_time,
        "seconds",
    )



# module imports
from .blender_utils import redraw_viewport

