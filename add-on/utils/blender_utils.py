
import bpy
from .math_utils import get_average_intensity

#Checks whether the mouseclick happened in the viewport or elsewhere    
def is_mouse_in_3d_view(context, event):
    
    #Identify the 3D Viewport area and its regions
    view_3d_area = next((area for area in context.screen.areas if area.type == 'VIEW_3D'), None)
    if view_3d_area is not None:
        toolbar_region = next((region for region in view_3d_area.regions if region.type == 'TOOLS'), None)
        ui_region = next((region for region in view_3d_area.regions if region.type == 'UI'), None)
        view_3d_window_region = next((region for region in view_3d_area.regions if region.type == 'WINDOW'), None)

        #Check if the mouse is inside the 3D Viewport's window region
        if view_3d_window_region is not None:
            mouse_inside_view3d = (
                view_3d_window_region.x < event.mouse_x < view_3d_window_region.x + view_3d_window_region.width and 
                view_3d_window_region.y < event.mouse_y < view_3d_window_region.y + view_3d_window_region.height
            )
            
            #Exclude areas occupied by the toolbar or UI regions
            if toolbar_region is not None:
                mouse_inside_view3d &= not (
                    toolbar_region.x < event.mouse_x < toolbar_region.x + toolbar_region.width and 
                    toolbar_region.y < event.mouse_y < toolbar_region.y + toolbar_region.height
                )
            if ui_region is not None:
                mouse_inside_view3d &= not (
                    ui_region.x < event.mouse_x < ui_region.x + ui_region.width and 
                    ui_region.y < event.mouse_y < ui_region.y + ui_region.height
                )
            
            return mouse_inside_view3d

    return False  #Default to False if checks fail.        

#function to check if a mouseclick is on a white object
def is_click_on_white(self, context, location, neighbors=5):
    global points_kdtree, point_colors
    intensity_threshold = context.scene.intensity_threshold

    #Define the number of nearest neighbors to search for
    num_neighbors = neighbors
    
    #Use the k-d tree to find the nearest points to the click location
    _, nearest_indices = points_kdtree.query([location], k=num_neighbors)
    
    average_intensity=get_average_intensity(nearest_indices)

    print(average_intensity)

    #If the average intensity is above the threshold, return True (click is on a "white" object)
    if average_intensity > intensity_threshold:
        return True
    else:
        print("Intensity threshold not met")
        return False
    