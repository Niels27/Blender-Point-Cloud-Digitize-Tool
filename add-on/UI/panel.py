# UI/panel
import bpy

#Panel for the Road Marking Digitizer
class AddOnPanel(bpy.types.Panel):
    bl_label = "Road Marking Digitizer"
    bl_idname = "DIGITIZE_PT_Panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Digitizing Tool'

    def draw(self, context):
        
        layout = self.layout
        scene = context.scene
        
        row_text = self.layout.row(align=True)
        row_text.label(text="                                  Point Cloud Tools")
        
        row = layout.row(align=True)
        row.operator("wm.las_open", text="Import point cloud")
        row.prop(scene, "sparsity_value")
        

        row = layout.row(align=True)
        row.prop(scene, "ground_only")
        row.prop(scene, "z_height_cut_off")
        
        row = layout.row(align=True)
        row.operator("custom.export_to_shapefile", text="Export shapefile")  
        row.prop(scene, "points_percentage")
        
        layout.operator("custom.create_point_cloud_object", text="Create point cloud object")
        row = layout.row(align=True)
        row.operator("custom.center_pointcloud", text="Center point cloud")
        row.operator("custom.remove_point_cloud", text="Remove point cloud")
        
        row = layout.row(align=True)
        row.operator("view3d.select_points", text="Get click info")
        
        row_text = self.layout.row(align=True)
        row_text.label(text="                                      Markers")
        
        row = layout.row(align=True)
        row.operator("custom.mark_fast", text="Simple marker")
        row.operator("custom.mark_complex", text="Complex marker")
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
        row.operator("custom.mark_snapping_line", text="Line marker") 
        row.operator("custom.auto_curved_line", text="Auto line marker") 
        row = layout.row()
        row.prop(scene, "snap_to_road_mark")
        row.prop(scene, "line_width")
       
        layout.operator("custom.dashed_line_marking_operator", text="Dash line marker") 
        layout.operator("custom.curb_detection_operator", text="Curb marker") 
        
        row_text = self.layout.row(align=True)
        row_text.label(text="                                     Marker Settings")
        
        layout.operator("custom.remove_all_markings", text="Remove all markings")
        row = layout.row(align=True)
        row.prop(scene, "extra_z_height")
        row.prop(scene, "marking_transparency")
        layout.prop(scene, "marking_color")
        row = layout.row(align=True)
        row.prop(scene, "filter_distance")
        row.prop(scene, "filter_neighbors")
        
        layout.prop(scene, "intensity_threshold")
        
        row_text = self.layout.row(align=True)
        row_text.label(text="                                   Extra Options")
        
        row = layout.row()
        row.prop(scene, "overwrite_existing_data")
        row = layout.row()
        row.prop(scene, "connect_socket") 
        row = layout.row()
        row.prop(scene, "show_dots")
        row = layout.row()
        row.prop(scene, "adjust_intensity_popup")
        row = layout.row()
        row.prop(scene, "save_shape") 
        row = layout.row()
        row.prop(scene, "save_obj") 

         #Dummy space
        for _ in range(10): 
            layout.label(text="")   

#Operator for the pop-up dialog
class PopUpOperator(bpy.types.Operator):
    bl_idname = "wm.correction_pop_up"
    bl_label = "Confirm correction pop up"
    bl_description = "Pop up to confirm"
    
    average_intensity: bpy.props.FloatProperty()
    adjust_action: bpy.props.StringProperty()
    
    action: bpy.props.EnumProperty(
        items=[
            ('CONTINUE', "Yes", "Yes"),
            ('STOP', "No", "No"),
        ],
        default='CONTINUE',
    )
    #Define the custom draw method
    def draw(self, context):
        layout = self.layout
        col = layout.column()
        
        #Add custom buttons to the UI
        if self.adjust_action=='LOWER':
            col.label(text="No road marks found. Try with lower threshold?")
        elif self.adjust_action=='HIGHER':
            col.label(text="Threshold might be too low, Try with higher threshold?")
        col.label(text="Choose an action:")
        col.separator()
        
        #Use 'props_enum' to create buttons for each enum property
        layout.props_enum(self, "action")
        
    def execute(self, context):

        #Based on the user's choice, perform the action
        context.scene.user_input_result = self.action
       
        if self.action == 'CONTINUE':
           if self.adjust_action=='LOWER':
               old_threshold=context.scene.intensity_threshold
               context.scene.intensity_threshold=self.average_intensity-20 #lower the threshold to the average intensity around the mouseclick
               print("changed intensity threshold from: ",old_threshold,"to: ",self.average_intensity," please try again")
           elif self.adjust_action=='HIGHER':
               old_threshold=context.scene.intensity_threshold
               context.scene.intensity_threshold=self.average_intensity-30 #higher the threshold to the average intensity around the mouseclick 
               print("changed intensity threshold from: ",old_threshold,"to: ",self.average_intensity," please try again")
        elif self.action == 'STOP':
            return {'CANCELLED'}
        
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)
