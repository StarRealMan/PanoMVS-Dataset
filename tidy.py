import os
import shutil
from tqdm import tqdm

import numpy as np

base_path = "/home/star/Dataset/Replica_360_simplified"
target_path = "/home/star/Dataset/Replica_360_Tidy"
index = 0
c2b_trans = np.array([[1, 0, 0, 0], 
                      [0, -1, 0, 0], 
                      [0, 0, -1, 0.2], 
                      [0, 0, 0, 1]])

e2ee_trans = np.array([[1, 0, 0, 0], 
                      [0, -1, 0, 0], 
                      [0, 0, -1, 0], 
                      [0, 0, 0, 1]])

for scene_name in ["apartment_0", "apartment_1", "apartment_2", "hotel_0", 
                    "office_0", "office_1", "office_2", "office_3", "office_4", 
                    "room_0", "room_1", "room_2"]:
    
    fisheye_path = os.path.join(base_path, scene_name, "fisheye")
    eqr_image_path = os.path.join(base_path, scene_name, "eqr", "image")
    eqr_pose_path = os.path.join(base_path, scene_name, "eqr", "pose")
    
    target_eqr_path = os.path.join(target_path, "eqr")
    target_pose_path = os.path.join(target_path, "pose")
    
    if not os.path.exists(target_eqr_path):
        os.makedirs(target_eqr_path)
    if not os.path.exists(target_pose_path):
        os.makedirs(target_pose_path)
    
    saved_index = index
    
    for cam in ["cam1", "cam2", "cam3", "cam4"]:
        fisheye_cam_image_path = os.path.join(fisheye_path, cam, "image")
        target_camera_path = os.path.join(target_path, cam)
        
        if not os.path.exists(target_camera_path):
            os.makedirs(target_camera_path)
        
        index = saved_index
        
        for image in tqdm(sorted(os.listdir(fisheye_cam_image_path), key=lambda s: int(s.split('.')[0]))):
            image_path = os.path.join(fisheye_cam_image_path, image)
            shutil.copyfile(image_path, os.path.join(target_camera_path, str(index) + ".png"))
        
            if cam == "cam1":
                sub_id = image.split('.')[0]
                fisheye_cam_pose_path = os.path.join(fisheye_path, cam, "pose")
                pose_path = os.path.join(fisheye_cam_pose_path, sub_id + ".txt")
                cam_1_pose = np.loadtxt(pose_path)

                eqr_image_image_path = os.path.join(eqr_image_path, image)
                eqr_pose_pose_path = os.path.join(eqr_pose_path, sub_id + ".txt")
                eqr_pose_pose = np.loadtxt(eqr_pose_pose_path)
                
                target_image_path = os.path.join(target_eqr_path, str(index) + ".png")
                shutil.copyfile(eqr_image_image_path, target_image_path)
                
                target_eqr_pose_path = os.path.join(target_pose_path, str(index) + ".txt")
                
                eqr_pose = c2b_trans.dot(np.linalg.inv(cam_1_pose)).dot(eqr_pose_pose).dot(e2ee_trans)
                np.savetxt(target_eqr_pose_path, eqr_pose)
        
            index += 1
        