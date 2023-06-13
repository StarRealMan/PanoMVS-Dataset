import numpy as np
import bpy
from mathutils import Matrix

def set_scene():
    bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
    bpy.context.scene.render.engine = 'CYCLES'

def set_camera_pose(T):
    camera = bpy.data.objects["Camera"]
    camera.matrix_world = Matrix(T)

def set_render(color_path, depth_path, view_num):
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    render_layers = tree.nodes["Render Layers"]
    composite_node = tree.nodes["Composite"]
    
    color_output_node = tree.nodes.new(type="CompositorNodeOutputFile")
    color_output_node.format.file_format = 'PNG'
    color_output_node.base_path = color_path
    color_output_node.file_slots[0].path = str(view_num) + "_#.png"
    links.new(render_layers.outputs['Image'], color_output_node.inputs[0])
    links.new(render_layers.outputs['Image'], composite_node.inputs[0])

    depth_output_node = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_output_node.format.file_format = 'OPEN_EXR'
    depth_output_node.base_path = depth_path
    depth_output_node.file_slots[0].path = str(view_num) + "_#.exr"
    links.new(render_layers.outputs['Depth'], depth_output_node.inputs[0])
    links.new(render_layers.outputs['Depth'], composite_node.inputs[0])

def render_scene():
    bpy.ops.render.render(write_still=True)

T_path = "/home/star/T"
param_path = "/home/star/param"

pose_T = np.loadtxt(T_path)
pose_T = list(pose_T)
with open(param_path, "r") as param:
    lines = param.readlines()
    view_num = int(lines[0])

color_path = "/home/star/color"
depth_path = "/home/star/depth"

# set_scene()
set_camera_pose(pose_T)
set_render(color_path, depth_path, view_num)
render_scene()
