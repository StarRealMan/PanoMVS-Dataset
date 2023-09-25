import os
from random import *
import numpy as np
import nvisii
import colorsys
from scipy.spatial.transform import Rotation as R
import configargparse


# camera to world transform
def load_poses(poses_path):
    fisheye_path = os.path.join(poses_path, "fisheye_poses")
    fisheye_poses = []
    for fisheye_file in os.listdir(fisheye_path):
        fisheye_pose = np.loadtxt(fisheye_file)
        fisheye_poses.append(fisheye_pose)
        
    panorama_path = os.path.join(poses_path, "fisheye_poses")
    panorama_poses = []
    for panorama_file in os.listdir(panorama_path):
        panorama_pose = np.loadtxt(panorama_file)
        panorama_poses.append(panorama_pose)
    
    return fisheye_poses, panorama_poses

np2tuple = lambda x: tuple(x.flatten())
tuple2np = lambda x: np.array(list(x))

# transform between OpenGL and CV coordinate
T_cv2cg = np.array([[1.0, 0.0, 0.0], 
                    [0.0, -1.0, 0.0], 
                    [0.0, 0.0, -1.0]])
T_cg2cv = np.linalg.inv(T_cv2cg)

def quat_pos_to_matrix(quat, pos):
    rot = R.from_quat(tuple2np(quat))
    pos = tuple2np(pos)
    T = np.eye(4)
    T[:3, :3] = T_cv2cg.dot(rot)
    T[:3, 3] = pos
    
    return T

def matrix_to_quat_pos(T):
    rot = T_cg2cv.dot(T[:3, :3])
    rotR = R.from_matrix(rot)
    pos = T[:3, 3]
    quat = rotR.as_quat()
    
    return np2tuple(quat), np2tuple(pos)

def matrix_to_quat_pos(T):
    rot = R.from_matrix(T[:3, :3])
    pos = T[:3, 3]
    quat = rot.as_quat()
    
    return np2tuple(quat), np2tuple(pos)

def load_dome_text(text_path):
    dome_list = []
    for path in os.listdir(text_path):
        path = os.path.join(text_path, path)
        dome_list.append(path)

def load_shapnet_scenes(shapenet_path):
    scene_list = []
    for path in os.listdir(shapenet_path):
        path = os.path.join(shapenet_path, path)
        if os.path.isdir(path):
            for subpath in os.listdir(path):
                subpathpath = os.path.join(path, subpath)
                if os.path.isdir(subpathpath):
                    modelpath = os.path.join(subpathpath, "models", "model_normalized.obj")
                    scene_list.append(modelpath)
    return scene_list

def add_rand_shapenet_scene(file = "file"):
    position = (
        uniform(-5,5), 
        uniform(-5,5), 
        uniform(-1,3)
    )
    
    rotation = (
        uniform(0,1), 
        uniform(0,1), 
        uniform(0,1), 
        uniform(0,1)
    )
    
    s = uniform(0.1, 1.0)
    scale = (s, s, s)
    
    sdb = nvisii.import_scene(
        filepath = file, 
        position = position, 
        scale = scale, 
        rotation = rotation, 
        args = ["verbose"]
    )

def add_rand_nvisii_obj(name = "name"):
    obj = nvisii.entity.create(
        name = name, 
        transform = nvisii.transform.create(name), 
        material = nvisii.material.create(name)
    )

    mesh_id = randint(0,15)

    mesh = nvisii.mesh.get(f'm_{mesh_id}')
    obj.set_mesh(mesh)

    obj.get_transform().set_position((
        uniform(-5,5), 
        uniform(-5,5), 
        uniform(-1,3)
    ))

    obj.get_transform().set_rotation((
        uniform(0,1), 
        uniform(0,1), 
        uniform(0,1), 
        uniform(0,1)
    ))

    s = uniform(0.05,0.15)
    obj.get_transform().set_scale((
        s,s,s
    ))  

    rgb = colorsys.hsv_to_rgb(
        uniform(0,1), 
        uniform(0.7,1), 
        uniform(0.7,1)
    )

    obj.get_material().set_base_color(rgb)

    mat = obj.get_material()
    
    material_type = randint(0,2)
    
    if material_type == 0:  
        if randint(0,2): mat.set_roughness(uniform(.9, 1))
        else           : mat.set_roughness(uniform(.0,.1))
    
    if material_type == 1:  
        mat.set_metallic(uniform(0.9,1))
        if randint(0,2): mat.set_roughness(uniform(.9, 1))
        else           : mat.set_roughness(uniform(.0,.1))
    
    if material_type == 2:  
        mat.set_transmission(uniform(0.9,1))
        
        if randint(0,2): mat.set_roughness(uniform(.9, 1))
        else           : mat.set_roughness(uniform(.0,.1))
        
        if randint(0,2): mat.set_transmission_roughness(uniform(.9, 1))
        else           : mat.set_transmission_roughness(uniform(.0,.1))

    mat.set_sheen(uniform(0,1))
    mat.set_clearcoat(uniform(0,1))
    if randint(0,1): mat.set_anisotropic(uniform(0.9,1))

def set_scene(dome_path, shapenet_path, obj_size, scene_size):
    nvisii.mesh.create_sphere('m_0')
    nvisii.mesh.create_torus_knot('m_1')
    nvisii.mesh.create_teapotahedron('m_2')
    nvisii.mesh.create_box('m_3')
    nvisii.mesh.create_capped_cone('m_4')
    nvisii.mesh.create_capped_cylinder('m_5')
    nvisii.mesh.create_capsule('m_6')
    nvisii.mesh.create_cylinder('m_7')
    nvisii.mesh.create_disk('m_8')
    nvisii.mesh.create_dodecahedron('m_9')
    nvisii.mesh.create_icosahedron('m_10')
    nvisii.mesh.create_icosphere('m_11')
    nvisii.mesh.create_rounded_box('m_12')
    nvisii.mesh.create_spring('m_13')
    nvisii.mesh.create_torus('m_14')
    nvisii.mesh.create_tube('m_15')

    dome_list = load_dome_text(dome_path)
    dome_file = choices(dome_list)
    dome = nvisii.texture.create_from_file("dome", dome_file)
    nvisii.set_dome_light_intensity(.8)
    nvisii.set_dome_light_texture(dome)
    
    for obj_num in range(obj_size):
        add_rand_nvisii_obj(str(obj_num))
    
    scene_list = load_shapnet_scenes(shapenet_path)
    rand_scenes = choices(scene_list, k=scene_size)
    for scene_num, scene_file in enumerate(rand_scenes):
        add_rand_shapenet_scene(str(scene_num), scene_file)

def set_camera():
    camera = nvisii.entity.create(name = "camera")
    camera.set_transform(nvisii.transform.create(name = "camera_transform"))

    camera.set_camera(
        nvisii.camera.create_from_fov(
            name = "camera_camera", 
            field_of_view = np.deg2rad(90), 
            aspect = 1.0
        )
    )
    nvisii.set_camera_entity(camera)
    
    return camera

# cam to world transform
def set_cam_pose(camera, T):
    quat, pose = matrix_to_quat_pos(T)
    camera.get_transform().set_position(pose)
    camera.get_transform().set_rotation(quat)

# psuedo cam to cam
transform_dict = {
    "front": np.array([[1.0, 0.0, 0.0, 0.0], 
                       [0.0, 1.0, 0.0, 0.0], 
                       [0.0, 0.0, 1.0, 0.0], 
                       [0.0, 0.0, 0.0, 1.0]]), 
    "left": np.array([[1.0, 0.0, 0.0, 0.0], 
                      [0.0, 1.0, 0.0, 0.0], 
                      [0.0, 0.0, 1.0, 0.0], 
                      [0.0, 0.0, 0.0, 1.0]]), 
    "right": np.array([[1.0, 0.0, 0.0, 0.0], 
                       [0.0, 1.0, 0.0, 0.0], 
                       [0.0, 0.0, 1.0, 0.0], 
                       [0.0, 0.0, 0.0, 1.0]]), 
    "up": np.array([[1.0, 0.0, 0.0, 0.0], 
                    [0.0, 1.0, 0.0, 0.0], 
                    [0.0, 0.0, 1.0, 0.0], 
                    [0.0, 0.0, 0.0, 1.0]]), 
    "down": np.array([[1.0, 0.0, 0.0, 0.0], 
                      [0.0, 1.0, 0.0, 0.0], 
                      [0.0, 0.0, 1.0, 0.0], 
                      [0.0, 0.0, 0.0, 1.0]]), 
    "back": np.array([[1.0, 0.0, 0.0, 0.0], 
                      [0.0, 1.0, 0.0, 0.0], 
                      [0.0, 0.0, 1.0, 0.0], 
                      [0.0, 0.0, 0.0, 1.0]])
}

def render_cubemap(path, camera, T, reso, spp):
    for transform, matrix in transform_dict.items():
        set_cam_pose(camera, matrix.dot(T))
        if not os.path.exists(path):
            os.makedirs(path)
        nvisii.render_to_file(
            width = reso, 
            height = reso, 
            samples_per_pixel = spp, 
            file_path = os.path.join(path, transform+".png")
        )

def main(args):
    fisheye_poses, panorama_poses = load_poses()
    
    for num in range(args.size):
        nvisii.initialize(headless = True, verbose = True)
        nvisii.enable_denoiser()

        camera = set_camera()
        set_scene(args.dome_path, args.shapenet_path, 10, 10)

        panorama_path = os.path.join(args.workspace_path, str(num), "panorama")
        fisheye_path = os.path.join(args.workspace_path, str(num), "fisheye")
        
        if os.path.exists(panorama_path):
            os.makedirs(panorama_path)
        if os.path.exists(fisheye_path):
            os.makedirs(fisheye_path)
        
        for panorama_num in range(args.panorama_size):
            path = os.path.join(panorama_path, str(panorama_num))
            if os.path.exists(path):
                os.makedirs(path)
            T = panorama_poses[panorama_num]
            render_cubemap(path, camera, T, args.reso, args.spp)
            
        for fisheye_num in range(4):
            path = os.path.join(panorama_path, str(fisheye_num))
            if os.path.exists(path):
                os.makedirs(path)
            T = fisheye_poses[fisheye_num]
            render_cubemap(path, camera, T, args.reso, args.spp)

        nvisii.deinitialize()

if __name__ == "__main__":
    parser = configargparse.ArgumentParser()
    parser.add_argument('--poses_path', type = str, default = "/home/star/Develop/PanoMVS-Dataset/poses")
    parser.add_argument('--workspace_path', type = str, default = "/home/star/Develop/PanoMVS-Dataset/test")
    parser.add_argument('--dome_path', type = str, default = "/home/star/Dataset/sky_dome")
    parser.add_argument('--shapenet_path', type = str, default = "/home/star/Dataset/ShapeNetCore.v2")
    parser.add_argument('--size', type=int, default=1)
    parser.add_argument('--panorama_size', type=int, default=48)
    parser.add_argument('--reso', type = int, default = 800)
    parser.add_argument('--spp', type = int, default = 64)
    
    args = parser.parse_args()
    
    main(args)