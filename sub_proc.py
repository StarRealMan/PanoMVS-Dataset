import numpy as np
import subprocess

def render_with_blender(blender_path, script_path, scene_path):
    command = [
        blender_path,
        "--background",
        scene_path,
        "-f", 
        "1",
        "--python",
        script_path,
    ]

    subprocess.run(command)

T_path = "/home/star/T"
param_path = "/home/star/param"
blender_path = "/home/star/Source/blender-3.5.1-linux-x64/blender"
scene_path = "/home/star/Dataset/nerf_data/blend_files/drums.blend"
# scene_path = "/home/star/Dataset/nerf_data/blend_files/lego.blend"
# scene_path = "/home/star/Dataset/nerf_data/blend_files/hotdog.blend"
script_path = "/home/star/render_blender.py"

pose = np.array([[1.0, 0.0, 0.0, 0.0], 
                 [0.0, 1.0, 0.0, 0.0], 
                 [0.0, 0.0, 1.0, 2.0], 
                 [0.0, 0.0, 0.0, 1.0]])
view_num = 5

np.savetxt(T_path, pose)
with open(param_path, "w") as param:
    param.writelines(str(view_num))

render_with_blender(blender_path, script_path, scene_path)