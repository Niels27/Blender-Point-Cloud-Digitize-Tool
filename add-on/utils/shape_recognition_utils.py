#library imports
import bpy
import cv2
import os
import bpy
import numpy as np
import math 
from mathutils import Vector, Matrix

#opencv functions
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

#Opencv shape detection from points    
def detect_shape_from_points(points, from_bmesh=False, scale_factor=100):

    if from_bmesh:
        #Convert bmesh vertices to numpy array
        coords_list = np.array([(point.x, point.y, point.z) for point in points])
    else:
        coords_list = np.array(points)
    
    #coords_list = filter_noise_with_dbscan(coords_list)
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
