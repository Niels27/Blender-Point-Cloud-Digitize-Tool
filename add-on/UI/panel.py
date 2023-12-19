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
        
        row = layout.row(align=True)
        row.operator("wm.las_open", text="Import point cloud")
        row.prop(scene, "ground_only")

        row = layout.row(align=True)
        row.prop(scene, "sparsity_value")
        row.prop(scene, "z_height_cut_off")
        
        row = layout.row(align=True)
        row.operator("custom.export_to_shapefile", text="Export shapefile")  
        row.prop(scene, "points_percentage")
        
        row_text = self.layout.row(align=True)
        row_text.label(text="                                     Tools")
        
        layout.operator("custom.create_point_cloud_object", text="Create point cloud object")
        row = layout.row(align=True)
        row.operator("custom.center_pointcloud", text="Center point cloud")
        row.operator("custom.remove_point_cloud", text="Remove point cloud")
        
        row = layout.row(align=True)
        row.operator("view3d.select_points", text="Click info")
        row.operator("custom.remove_all_markings", text="Remove all markings")
        layout.prop(scene, "intensity_threshold")
        
        layout.prop(scene, "marking_color")
        row = layout.row(align=True)
        row.prop(scene, "extra_z_height")
        row.prop(scene, "marking_transparency")
       
        row_text = self.layout.row(align=True)
        row_text.label(text="                                     Markers")
        
        row = layout.row(align=True)
        row.operator("view3d.mark_fast", text="Simple marker")
        row.operator("view3d.mark_complex", text="Complex marker")
        layout.operator("view3d.selection_detection", text="Selection marker")
        row = layout.row()
        row.operator("custom.find_all_road_marks", text="Auto mark")
        row.prop(scene, "markings_threshold")
        
        row = layout.row(align=True)
        row.operator("custom.mark_fixed_triangle", text="Fixed triangle")
        row.operator("custom.mark_fixed_rectangle", text="Fixed rectangle")
        row = layout.row(align=True)
        row.operator("custom.mark_triangle", text="Triangles marker")
        row.operator("custom.auto_mark_triangle", text="Auto triangles marker")
        row = layout.row(align=True)
        row.operator("custom.mark_rectangle", text="Rectangles marker")
        row.operator("custom.auto_mark_rectangle", text="Auto rectangles marker")
        
        row = layout.row()
        row.prop(scene, "snap_to_road_mark")
        row.prop(scene, "fatline_width")
        row = layout.row()
        row.operator("custom.mark_snapping_line", text="Line marker") 
        row.operator("custom.auto_curved_line", text="Auto line marker") 
       
        layout.operator("custom.dashed_line_marking_operator", text="Dash marker") 
        layout.operator("custom.curb_detection_operator", text="Curb marker") 
        
        row_text = self.layout.row(align=True)
        row_text.label(text="                                     Options")
        
        row = layout.row()
        row.prop(scene, "save_shape") 
        row = layout.row()
        row.prop(scene, "show_dots")

         #Dummy space
        for _ in range(10): 
            layout.label(text="")   
            
            
             
            