import os
import shutil
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm
from ocam import OcamCamera

def GenFishRays(W, H, ocam):
    u = torch.arange(W) + 0.5
    v = torch.arange(H) + 0.5
    u_grid, v_grid = torch.meshgrid(u, v, indexing = "xy")
    ratio = ocam.height / H
    u_grid = u_grid * ratio
    v_grid = v_grid * ratio
    uv = torch.stack([u_grid, v_grid], dim = -1)
    uv = uv.view(-1 ,2)
    
    rays_d = ocam.cam2world(uv)

    return rays_d

def EqualRec2Fisheye(er_image, image_H, image_W, ocam, R):
    '''
    input: 
    er_image: (B, C, H, W)
    image_H, image_W: pinhole image size
    ocam: fisheye camera  model
    R: pinhole to panorama rotation matrix (B, 3, 3)
    output:
    fisheye image: (B, C, image_H, image_W)
    '''
    
    B, _, H, W = er_image.shape
    device = er_image.device
    
    rays_d = GenFishRays(image_W, image_H, ocam)
    rays_d = rays_d.to(device).unsqueeze(0).view(1, -1, 3).repeat(B, 1, 1)
    rays_d = torch.bmm(rays_d, R.permute(0, 2, 1))
    rays_d = rays_d.view(B, image_H, image_W, 3)
    
    grid_phi = torch.atan2(rays_d[..., 2], rays_d[..., 0])
    grid_phi[grid_phi < -0.5 * torch.pi] += 2 * torch.pi
    grid_phi = 1.5 * torch.pi - grid_phi
    grid_phi = grid_phi / torch.pi - 1.0
    
    grid_theta = torch.pi - torch.acos(rays_d[..., 1])
    grid_theta = (grid_theta - 0.5 * torch.pi) / (H * torch.pi / W)
    
    grid = torch.stack([grid_phi, grid_theta], -1)
    
    fe_image = F.grid_sample(er_image, grid, align_corners = True, padding_mode = "border")
    
    return fe_image

def load_cam_intrinsic(param_path, fov):
    ocams = []
    for i in range(1, 5):
        key = f'cam{i}'
        ocam_file = os.path.join(param_path, f'o{key}.txt')
        ocams.append(OcamCamera(ocam_file, fov))

    return ocams

def gen_fisheye(scene_name, fe_base_path, res_path):
    eqr_path = fe_base_path + scene_name
    fisheye_path = res_path + scene_name + '/fisheye'
    
    views = ["cam1", "cam2", "cam3", "cam4"]
    for view in views:
        view_image_path = os.path.join(fisheye_path, view, "image")
        view_pose_path = os.path.join(fisheye_path, view, "pose")
        if not os.path.exists(view_image_path):
            os.makedirs(view_image_path)
        if not os.path.exists(view_pose_path):
            os.makedirs(view_pose_path)
    
    ocams = load_cam_intrinsic(param_path, fov)
    
    for view_num, view in enumerate(views):
        image_path = os.path.join(eqr_path, view, "image")
        for num, num_path in enumerate(tqdm(sorted(os.listdir(image_path), key=lambda s: int(s.split('.')[0])))):
            num_path = os.path.join(image_path, num_path)
            
            eqr = cv2.imread(num_path)
            eqr = torch.from_numpy(eqr).to(device)
            eqr = eqr.permute(2, 0, 1).unsqueeze(0).float()
            
            ocam = ocams[view_num]
            fisheye_h = ocam.height
            fisheye_w = ocam.width
            
            fisheye = EqualRec2Fisheye(eqr, fisheye_h, fisheye_w, ocam, 
                                                torch.eye(3, device = device).unsqueeze(0))
            
            fisheye = fisheye.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            
            
            view_image_path = os.path.join(fisheye_path, view, "image")
            image_name = os.path.join(view_image_path, str(num) + '.png')
            cv2.imwrite(image_name, fisheye)
            
            view_pose_path = os.path.join(fisheye_path, view, "pose")
            pose_name = os.path.join(view_pose_path, str(num) + '.txt')
            shutil.copyfile(os.path.join(os.path.join(eqr_path, view, "pose"), str(num) + '.txt'), pose_name)            

if __name__ == '__main__':
    fov = 220
    device = 'cuda:0'
    param_path = './ocam'
    fe_base_path = '/home/star/Dataset/Replica/replica_generated_fe_simplified/'
    res_path = '/home/star/Dataset/Replica_360_simplified/'
    
    for scene_name in ["apartment_0", "apartment_1", "apartment_2", "hotel_0", 
                       "office_0", "office_1", "office_2", "office_3", "office_4", 
                       "room_0", "room_1", "room_2"]:
        gen_fisheye(scene_name, fe_base_path, res_path)
        