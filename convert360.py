import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
import c2e

def cube_h2list(cube_h):
    assert cube_h.shape[0] * 6 == cube_h.shape[1]
    return np.split(cube_h, 6, axis=1)

def cube_list2h(cube_list):
    assert len(cube_list) == 6
    assert sum(face.shape == cube_list[0].shape for face in cube_list) == 6
    return np.concatenate(cube_list, axis=1)

def gen_360_fe(scene_name, msp_base_path, fe_pase_path):
    if not os.path.exists(msp_base_path + scene_name):
        return
    
    cube_path = msp_base_path + scene_name + '/fisheye'
    eqr_path = fe_pase_path + scene_name
    views = ["cam1", "cam2", "cam3", "cam4"]
    for view in views:
        view_image_path = os.path.join(eqr_path, view, "image")
        view_pose_path = os.path.join(eqr_path, view, "pose")
        if not os.path.exists(view_image_path):
            os.makedirs(view_image_path)
        if not os.path.exists(view_pose_path):
            os.makedirs(view_pose_path)

    for num, num_path in enumerate(tqdm(sorted(os.listdir(cube_path), key=lambda s: int(s.split('.')[0])))):
        num_path = os.path.join(cube_path, num_path)
        
        for rand_num, rand_num_path in enumerate(sorted(os.listdir(num_path), key=lambda s: int(s.split('.')[0]))):
            rand_num_path = os.path.join(num_path, rand_num_path)

            # Load cubemap
            front = cv2.imread(os.path.join(rand_num_path, 'front.png'))
            left = cv2.imread(os.path.join(rand_num_path, 'left.png'))
            right = cv2.imread(os.path.join(rand_num_path, 'right.png'))
            up = cv2.imread(os.path.join(rand_num_path, 'up.png'))
            down = cv2.imread(os.path.join(rand_num_path, 'down.png'))
            back = cv2.imread(os.path.join(rand_num_path, 'back.png'))

            cube_size = front.shape[0]
            cubemap = np.zeros((3 * cube_size, 4 * cube_size, 3), dtype=np.uint8)
            
            cubemap[cube_size:2*cube_size, 0:cube_size] = left
            cubemap[cube_size:2*cube_size, cube_size:2*cube_size] = front
            cubemap[cube_size:2*cube_size, 2*cube_size:3*cube_size] = right
            cubemap[0:cube_size, cube_size:2*cube_size] = up
            cubemap[2*cube_size:3*cube_size, cube_size:2*cube_size] = down
            cubemap[cube_size:2*cube_size, 3*cube_size:4*cube_size] = back
            
            # cv2.imwrite("test.png", cubemap)
            
            cubemap = cubemap.astype(np.float32) / 255.0
            
            # Convert to equirectangular
            equirec = c2e.c2e(cubemap, eqr_size, eqr_size * 2, eqr_mode, cube_format)
            
            # Save equirectangular
            view_image_path = os.path.join(eqr_path, views[rand_num], "image")
            image_name = os.path.join(view_image_path, str(num) + '.png')
            cv2.imwrite(image_name, equirec * 255.0)
            
            view_pose_path = os.path.join(eqr_path, views[rand_num], "pose")
            pose_name = os.path.join(view_pose_path, str(num) + '.txt')
            shutil.copyfile(os.path.join(rand_num_path, 'xyz.txt'), pose_name)

def gen_360_eqr(scene_name, msp_base_path, res_path):
    if not os.path.exists(msp_base_path + scene_name):
        return
    
    cube_path = msp_base_path + scene_name + '/eqr'
    eqr_path = res_path + scene_name + '/eqr'

    image_path = os.path.join(eqr_path, "image")
    pose_path = os.path.join(eqr_path, "pose")
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    if not os.path.exists(pose_path):
        os.makedirs(pose_path)

    for num, num_path in enumerate(tqdm(sorted(os.listdir(cube_path), key=lambda s: int(s.split('.')[0])))):
        num_path = os.path.join(cube_path, num_path)

        # Load cubemap
        front = cv2.imread(os.path.join(num_path, 'front.png'))
        left = cv2.imread(os.path.join(num_path, 'left.png'))
        right = cv2.imread(os.path.join(num_path, 'right.png'))
        up = cv2.imread(os.path.join(num_path, 'up.png'))
        down = cv2.imread(os.path.join(num_path, 'down.png'))
        back = cv2.imread(os.path.join(num_path, 'back.png'))

        cube_size = front.shape[0]
        cubemap = np.zeros((3 * cube_size, 4 * cube_size, 3), dtype=np.uint8)
        
        cubemap[cube_size:2*cube_size, 0:cube_size] = left
        cubemap[cube_size:2*cube_size, cube_size:2*cube_size] = front
        cubemap[cube_size:2*cube_size, 2*cube_size:3*cube_size] = right
        cubemap[0:cube_size, cube_size:2*cube_size] = up
        cubemap[2*cube_size:3*cube_size, cube_size:2*cube_size] = down
        cubemap[cube_size:2*cube_size, 3*cube_size:4*cube_size] = back
        
        # cv2.imwrite("test.png", cubemap)
        
        cubemap = cubemap.astype(np.float32) / 255.0
        
        # Convert to equirectangular
        equirec = c2e.c2e(cubemap, eqr_size, eqr_size * 2, eqr_mode, cube_format)
        
        # Save equirectangular
        view_image_path = os.path.join(eqr_path, "image")
        image_name = os.path.join(view_image_path, str(num) + '.png')
        cv2.imwrite(image_name, equirec * 255.0)
        
        view_pose_path = os.path.join(eqr_path, "pose")
        pose_name = os.path.join(view_pose_path, str(num) + '.txt')
        shutil.copyfile(os.path.join(num_path, 'xyz.txt'), pose_name)
        
if __name__ == '__main__':
    eqr_size = 512
    eqr_mode = 'bilinear'
    cube_format = 'dice'
    msp_base_path = '/home/star/Dataset/Replica/replica_generated_msp_simplified/'
    fe_base_path = '/home/star/Dataset/Replica/replica_generated_fe_simplified/'
    res_path = '/home/star/Dataset/Replica_360_simplified/'

    for scene_name in ["apartment_0", "apartment_1", "apartment_2", "hotel_0", 
                       "office_0", "office_1", "office_2", "office_3", "office_4", 
                       "room_0", "room_1", "room_2"]:
        for data_type in ["fisheye", "eqr"]:
            gen_360_fe(scene_name, msp_base_path, fe_base_path)
            gen_360_eqr(scene_name, msp_base_path, res_path)
        