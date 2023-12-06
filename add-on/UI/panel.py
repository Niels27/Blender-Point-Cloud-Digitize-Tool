# UI/panel
import bpy

#Panel for the Road Marking Digitizer
class DIGITIZE_PT_Panel(bpy.types.Panel):
    bl_label = "Road Marking Digitizer"
    bl_idname = "DIGITIZE_PT_Panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Digitizing Tool'

    def draw(self, context):
        
        layout = self.layout
        scene = context.scene
        
        layout.operator("wm.las_open", text="Import Point Cloud")
        layout.operator("custom.export_to_shapefile", text="export to shapefile")  
        layout.prop(scene, "points_percentage")
        layout.prop(scene, "z_height_cut_off")
        layout.operator("custom.center_pointcloud", text="Center Point Cloud")
        layout.operator("custom.create_point_cloud_object", text="Create Point Cloud object")
        layout.operator("custom.remove_point_cloud", text="Remove point cloud")
        layout.operator("custom.remove_all_markings", text="Remove All markings")
        
        row = layout.row(align=True)
        layout.operator("view3d.select_points", text="Get point color & intensity")
        layout.prop(scene, "intensity_threshold")
        
        row = layout.row()
        layout.prop(scene, "marking_color")
        layout.prop(scene, "marking_transparency")
        layout.prop(scene, "extra_z_height")
        
        row = layout.row(align=True)
        row.operator("view3d.line_drawer", text="Draw Line")
        row.prop(scene, "fatline_width")
 
        row = layout.row(align=True)
        layout.operator("view3d.mark_fast", text="Simple fill marker")
        layout.operator("view3d.mark_complex", text="Complex fill marker")
        layout.operator("view3d.selection_detection", text="Selection fill Marker")
        row = layout.row()
        row.operator("custom.find_all_road_marks", text="Auto Mark")
        row.prop(scene, "markings_threshold")
        
        layout.operator("custom.mark_fixed_triangle", text="fixed triangle marker")
        layout.operator("custom.mark_fixed_rectangle", text="fixed rectangle marker")
       
        layout.operator("custom.mark_triangle", text="triangle marker")
        layout.operator("custom.auto_mark_triangle", text="auto triangle marker")
        layout.operator("custom.mark_rectangle", text="rectangle marker")
        layout.operator("custom.auto_mark_rectangle", text="auto rectangle marker")
        layout.operator("custom.mark_snapping_line", text="snapping line marker") 
        row = layout.row()
        row.prop(scene, "snap_to_road_mark")
        layout.operator("custom.auto_curved_line", text="auto curved line")  
        row = layout.row()
        row.prop(scene, "save_shape") 
        row = layout.row()
        row.prop(scene, "show_dots")
        row = layout.row()
        
         #Dummy space
        for _ in range(5): 
            layout.label(text="")
            