# add-on/__init__.py

bl_info = {
    "name": "Point Cloud Digitizing Tool",
    "description": "A custom add-on for digitizing road marks in point clouds",
    "author": "Niels van der Wal",
    "version": (1, 1, 4),
    "blender": (3, 4, 0),  
    "location": "View3D > Tools > Digitize Tool",
    "category": "3D View"
}

#list of libraries to install
library_list = [
    'numpy',
    'open3d',
    'laspy[laszip]',
    'scipy',
    'mathutils',
    'pandas',
    'geopandas',
    'shapely',
    'scikit-learn',
    'joblib',
    'opencv-python',
    'websockets',
    'asyncio'
]

#library imports
import sys
import bpy
from bpy import context
from bpy.props import StringProperty, IntProperty, FloatProperty, BoolProperty
print(sys.executable)
print(sys.version)

#install libraries
#from .utils.blender_utils import install_libraries, update_libraries, uninstall_libraries
#install_libraries('library_list')
#update_libraries()
#uninstall_libraries()

#module imports
from .utils.websocket_utils import *
from .operators.utility_operators import *
from .operators.digitizing_operators import *
from . UI.panel import AddOnPanel, PopUpOperator



            
#Register the operators and panel
def register():
    bpy.utils.register_class(LAS_OT_OpenOperator)
    bpy.utils.register_class(CreatePointCloudObjectOperator)
    bpy.utils.register_class(DrawStraightFatLineOperator)
    bpy.utils.register_class(RemoveAllMarkingsOperator)
    bpy.utils.register_class(AddOnPanel)
    bpy.utils.register_class(RemovePointCloudOperator)
    bpy.utils.register_class(GetPointsInfoOperator)
    bpy.utils.register_class(SimpleMarkOperator)
    bpy.utils.register_class(ComplexMarkOperator)
    bpy.utils.register_class(SelectionDetectionOpterator)
    bpy.utils.register_class(AutoTriangleMarkOperator) 
    bpy.utils.register_class(TriangleMarkOperator) 
    bpy.utils.register_class(FixedTriangleMarkOperator) 
    bpy.utils.register_class(FixedRectangleMarkOperator)
    bpy.utils.register_class(RectangleMarkOperator)
    bpy.utils.register_class(AutoRectangleMarkOperator)
    bpy.utils.register_class(AutoCurvedLineOperator)
    bpy.utils.register_class(SnappingLineMarkOperator)
    bpy.utils.register_class(CurbDetectionOperator)
    bpy.utils.register_class(PopUpOperator)
    bpy.utils.register_class(CenterPointCloudOperator)
    bpy.utils.register_class(ExportToShapeFileOperator)
    bpy.utils.register_class(FindAllRoadMarkingsOperator)
    bpy.utils.register_class(DashedLineMarkingOperator)

    
    bpy.types.Scene.point_size = IntProperty(name="POINT SIZE", default=1)
    bpy.types.Scene.intensity_threshold = bpy.props.FloatProperty(
        name="Intensity threshold",
        description="Minimum intensity threshold",
        default=160, 
        min=0,
        max=255,
        subtype='UNSIGNED'  
    )
    bpy.types.Scene.markings_threshold = bpy.props.IntProperty(
        name="Max:",
        description="Maximum markings amount for auto marker",
        default=5,  
        min=1, 
        max=100, 
        subtype='UNSIGNED' 
    )
    bpy.types.Scene.points_percentage = bpy.props.IntProperty(
        name="with point %:",
        description="Percentage of points to export",
        default=2,  
        min=1, 
        max=100, 
        subtype='UNSIGNED' 
    )
    bpy.types.Scene.sparsity_value = bpy.props.FloatProperty(
        name="Sparsity:",
        description="sparsity of points rendered",
        default=0.2,  
        min=0.01, 
        max=1, 
        subtype='UNSIGNED' 
    )
    bpy.types.Scene.line_width = bpy.props.FloatProperty(
        name="Line width",
        description="Fat Line Width",
        default=0.10,
        min=0.01, max=10,  #min and max width
        subtype='NONE'     
    )
    bpy.types.Scene.marking_color = bpy.props.FloatVectorProperty(
        name="Marking color",
        subtype='COLOR',
        description="Select Marking color",
        default=(1, 0, 0, 1),  #Default is red
        min=0.0, max=1.0,  #Colors range from 0 to 1
        size=4
        
    )
    bpy.types.Scene.marking_color = bpy.props.FloatVectorProperty(
        name="Marking color",
        subtype='COLOR',
        description="Select Marking color",
        default=(1, 0, 0, 1),  #Default is red
        min=0.0, max=1.0,  #Colors range from 0 to 1
        size=4
    )
    bpy.types.Scene.marking_transparency = bpy.props.FloatProperty(
        name="Transparency",
        description="Set the transparency for the marking (0.0 fully transparent, 1.0 fully opaque)",
        default=1,  #Default transparency is 100%
        min=0.0, max=1.0  #Transparency can range from 0.0 to 1.0
    )
    bpy.types.Scene.user_input_result = bpy.props.StringProperty(
    name="User Input Result",
    description="Stores the result from the user input pop-up",
)
    bpy.types.Scene.save_shape = bpy.props.BoolProperty(
        name="Save shape images",
        description="Saves an image after marking a shape",
        default=False,
        subtype='UNSIGNED'  
    )
    bpy.types.Scene.save_obj = bpy.props.BoolProperty(
        name="Save shape objects",
        description="Saves an OBJ file of each shape",
        default=False,
        subtype='UNSIGNED'  
    )
    bpy.types.Scene.show_dots = bpy.props.BoolProperty(
        name="Show dots",
        description="Toggle showing feedback dots",
        default=True,
        subtype='UNSIGNED'  
    )
    bpy.types.Scene.ground_only = bpy.props.BoolProperty(
        name="Ground only",
        description="Toggle loading points from ground classification only",
        default=True,
        subtype='UNSIGNED'  
    )
    bpy.types.Scene.z_height_cut_off = bpy.props.FloatProperty(
        name="Max height",
        description="Height to cut off from ground level, 0 to not cut",
        default=0,
        min=0.0, max=100,  
        subtype='UNSIGNED'  
    )
    bpy.types.Scene.extra_z_height = bpy.props.FloatProperty(
        name="Marking height",
        description="Extra height of all markings compared to the ground level",
        default=0.05,
        min=-100, max=100, 
        subtype='UNSIGNED'  
    )
    bpy.types.Scene.filter_distance = bpy.props.FloatProperty(
        name="Filter distance",
        description="Max distance between points for filtering",
        default=0.2,
        min=0.001, max=1.0,
        subtype='UNSIGNED'  
    )
    bpy.types.Scene.filter_neighbors = bpy.props.IntProperty(
        name="Filter neighbors",
        description="Min amount of required neighbors for filtering",
        default=20,
        min=0, max=1000,  
        subtype='UNSIGNED'  
    )
    bpy.types.Scene.snap_to_road_mark= bpy.props.BoolProperty(
        name="Snap line",
        description="Snaps the line to nearby roadmark",
        default=True,
        subtype='UNSIGNED'  
    )
    bpy.types.Scene.overwrite_existing_data= bpy.props.BoolProperty(
        name="Overwrite existing point cloud data",
        description="Overwrite existing point cloud data with the same name",
        default=False,
        subtype='UNSIGNED'  
    )
    bpy.types.Scene.adjust_intensity_popup= bpy.props.BoolProperty(
        name="Intensity suggestion pop-up",
        description="Shows a pop-up to adjust intensity threshold",
        default=True,
        subtype='UNSIGNED'  
    )   
    bpy.types.Scene.connect_socket= bpy.props.BoolProperty(
        name="Enable websocket",
        description="Open a websocket connection to the webviewer",
        default=False,
        update=update_connect_socket
    )
    

#Unregister the operators and panel                                    
def unregister():
    
    bpy.utils.unregister_class(LAS_OT_OpenOperator) 
    bpy.utils.unregister_class(DrawStraightFatLineOperator)
    bpy.utils.unregister_class(RemoveAllMarkingsOperator)
    bpy.utils.unregister_class(AddOnPanel)
    bpy.utils.unregister_class(RemovePointCloudOperator)
    bpy.utils.unregister_class(GetPointsInfoOperator)
    
    bpy.utils.unregister_class(SimpleMarkOperator)
    bpy.utils.unregister_class(ComplexMarkOperator)
    bpy.utils.unregister_class(SelectionDetectionOpterator)
    bpy.utils.unregister_class(FindAllRoadMarkingsOperator)  
    bpy.utils.unregister_class(FixedTriangleMarkOperator)
    bpy.utils.unregister_class(FixedRectangleMarkOperator) 
    bpy.utils.unregister_class(TriangleMarkOperator)
    bpy.utils.unregister_class(AutoTriangleMarkOperator)
    bpy.utils.unregister_class(RectangleMarkOperator) 
    bpy.utils.unregister_class(AutoRectangleMarkOperator) 
    bpy.utils.unregister_class(AutoCurvedLineOperator)
    bpy.utils.unregister_class(SnappingLineMarkOperator)
    bpy.utils.unregister_class(CurbDetectionOperator)
    
    bpy.utils.unregister_class(CreatePointCloudObjectOperator)
    bpy.utils.unregister_class(PopUpOperator)
    bpy.utils.unregister_class(CenterPointCloudOperator)
    bpy.utils.unregister_class(ExportToShapeFileOperator)
    bpy.utils.unregister_class(DashedLineMarkingOperator)
    
    del bpy.types.Scene.marking_transparency
    del bpy.types.Scene.marking_color
    del bpy.types.Scene.intensity_threshold
    del bpy.types.Scene.markings_threshold
    del bpy.types.Scene.line_width
    del bpy.types.Scene.user_input_result
    del bpy.types.Scene.save_shape
    del bpy.types.Scene.save_obj
    del bpy.types.Scene.show_dots
    del bpy.types.Scene.snap_to_road_mark
    del bpy.types.Scene.z_height_cut_off
    del bpy.types.Scene.extra_z_height
    del bpy.types.Scene.points_percentage
    del bpy.types.Scene.sparsity_value
    del bpy.types.Scene.ground_only
    del bpy.types.Scene.overwrite_existing_data
    del bpy.types.Scene.connect_socket
    del bpy.types.Scene.filter_distance
    del bpy.types.Scene.filter_neighbors
    del bpy.types.Scene.adjust_intensity_popup
                 
if __name__ == "__main__":
    register()

 

        
