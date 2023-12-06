# add-on/__init__.py

bl_info = {
    "name": "Point Cloud Digitizing Tool",
    "description": "A custom add-on for digitizing road marks in point clouds",
    "author": "Niels van der Wal",
    "version": (1, 0, 0),
    "blender": (3, 4, 0),  # Replace with the minimum Blender version your script is compatible with
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
    'opencv-python'
]

#library imports
import sys
import bpy
from bpy.props import StringProperty, IntProperty, FloatProperty, BoolProperty
print(sys.executable)
print(sys.version)

#install libraries
#from .utils.blender_utils import install_libraries, update_libraries, uninstall_libraries
#install_libraries('library_list')
#update_libraries()
#uninstall_libraries()

#module imports
from .operators.utility_operators import *
from .operators.drawing_operators import *
from . UI.panel import DIGITIZE_PT_Panel

#Register the operators and panel
def register():
    bpy.utils.register_class(LAS_OT_OpenOperator)
    bpy.utils.register_class(CreatePointCloudObjectOperator)
    bpy.utils.register_class(DrawStraightFatLineOperator)
    bpy.utils.register_class(RemoveAllObjectsOperator)
    bpy.utils.register_class(DIGITIZE_PT_Panel)
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
    bpy.utils.register_class(CurvedLineMarkOperator)
    bpy.utils.register_class(CorrectionPopUpOperator)
    bpy.utils.register_class(CenterPointCloudOperator)
    bpy.utils.register_class(ExportToShapeFileOperator)
    bpy.utils.register_class(FindALlRoadMarkingsOperator)

    bpy.types.Scene.point_size = IntProperty(name="POINT SIZE",
                                      default=1)
    bpy.types.Scene.sparsity_value = IntProperty(name="SPARSITY VALUE",
                                      default=1)
    
    bpy.types.Scene.intensity_threshold = bpy.props.FloatProperty(
        name="Intensity Threshold",
        description="Minimum intensity threshold",
        default=160,  #Default value
        min=0,#Minimum value
        max=255,#Max value
        subtype='UNSIGNED'  
    )
    bpy.types.Scene.markings_threshold = bpy.props.IntProperty(
        name="Max:",
        description="Maximum markings amount for auto marker",
        default=30,  #Default value
        min=1, #Minimum value
        max=100, #Max value  
        subtype='UNSIGNED' 
    )
    bpy.types.Scene.points_percentage = bpy.props.IntProperty(
        name="Points percentage:",
        description="Percentage of points rendered",
        default=50,  #Default value
        min=1, #Minimum value
        max=100, #Max value  
        subtype='UNSIGNED' 
    )
    bpy.types.Scene.fatline_width = bpy.props.FloatProperty(
        name="Width",
        description="Fat Line Width",
        default=0.10,
        min=0.01, max=10,  #min and max width
        subtype='NONE'  
        
    )
    bpy.types.Scene.marking_color = bpy.props.FloatVectorProperty(
        name="Marking Color",
        subtype='COLOR',
        description="Select Marking color",
        default=(1, 0, 0, 1),  #Default is red
        min=0.0, max=1.0,  #Colors range from 0 to 1
        size=4
        
    )
    bpy.types.Scene.marking_color = bpy.props.FloatVectorProperty(
        name="Marking Color",
        subtype='COLOR',
        description="Select Marking color",
        default=(1, 0, 0, 1),  #Default is red
        min=0.0, max=1.0,  #Colors range from 0 to 1
        size=4
    )
    bpy.types.Scene.marking_transparency = bpy.props.FloatProperty(
        name="Marking Transparency",
        description="Set the transparency for the marking (0.0 fully transparent, 1.0 fully opaque)",
        default=1,  #Default transparency is 100%
        min=0.0, max=1.0  #Transparency can range from 0.0 to 1.0
    )
    bpy.types.Scene.user_input_result = bpy.props.StringProperty(
    name="User Input Result",
    description="Stores the result from the user input pop-up",
)
    bpy.types.Scene.save_shape = bpy.props.BoolProperty(
        name="Save Shapes",
        description="Toggle saving shapes",
        default=False,
        subtype='UNSIGNED'  
    )
    bpy.types.Scene.auto_load = bpy.props.BoolProperty(
        name="Auto load auto.laz",
        description="Toggle auto loading auto.laz",
        default=False,
        subtype='UNSIGNED'  
    )
    bpy.types.Scene.show_dots = bpy.props.BoolProperty(
        name="Show Dots",
        description="Toggle showing dots",
        default=False,
        subtype='UNSIGNED'  
    )
    bpy.types.Scene.z_height_cut_off = bpy.props.FloatProperty(
        name="max z",
        description="height to cut off z",
        default=0.5,
        subtype='UNSIGNED'  
    )
    bpy.types.Scene.extra_z_height = bpy.props.FloatProperty(
        name="marking z",
        description="extra height of all objects",
        default=0.1,
        subtype='UNSIGNED'  
    )
    bpy.types.Scene.snap_to_road_mark= bpy.props.BoolProperty(
        name="auto snap",
        description="Snaps user drawn shape to roadmark",
        default=True,
        subtype='UNSIGNED'  
    )
                                    
def unregister():
    
    bpy.utils.unregister_class(LAS_OT_OpenOperator) 
    bpy.utils.unregister_class(DrawStraightFatLineOperator)
    bpy.utils.unregister_class(RemoveAllObjectsOperator)
    bpy.utils.unregister_class(DIGITIZE_PT_Panel)
    bpy.utils.unregister_class(RemovePointCloudOperator)
    bpy.utils.unregister_class(GetPointsInfoOperator)
    
    bpy.utils.unregister_class(SimpleMarkOperator)
    bpy.utils.unregister_class(ComplexMarkOperator)
    bpy.utils.unregister_class(SelectionDetectionOpterator)
    bpy.utils.unregister_class(FindALlRoadMarkingsOperator)  
    bpy.utils.unregister_class(FixedTriangleMarkOperator)
    bpy.utils.unregister_class(FixedRectangleMarkOperator) 
    bpy.utils.unregister_class(TriangleMarkOperator)
    bpy.utils.unregister_class(AutoTriangleMarkOperator)
    bpy.utils.unregister_class(RectangleMarkOperator) 
    bpy.utils.unregister_class(AutoRectangleMarkOperator) 
    bpy.utils.unregister_class(AutoCurvedLineOperator)
    bpy.utils.unregister_class(CurvedLineMarkOperator)
    
    bpy.utils.unregister_class(CreatePointCloudObjectOperator)
    bpy.utils.unregister_class(CorrectionPopUpOperator)
    bpy.utils.unregister_class(CenterPointCloudOperator)
    bpy.utils.unregister_class(ExportToShapeFileOperator)
    
    del bpy.types.Scene.marking_transparency
    del bpy.types.Scene.marking_color
    del bpy.types.Scene.intensity_threshold
    del bpy.types.Scene.markings_threshold
    del bpy.types.Scene.fatline_width
    del bpy.types.Scene.user_input_result
    del bpy.types.Scene.save_shape
    del bpy.types.Scene.auto_load
    del bpy.types.Scene.show_dots
    del bpy.types.Scene.snap_to_road_mark
    del bpy.types.Scene.z_height_cut_off
    del bpy.types.Scene.extra_z_height
    del bpy.types.Scene.points_percentage
    
    bpy.utils.unregister_class(OBJECT_OT_simple_undo)
    bpy.utils.unregister_class(OBJECT_OT_simple_redo)
                 
if __name__ == "__main__":
    register()

        
