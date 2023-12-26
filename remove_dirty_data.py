import os
import shutil
from tqdm import tqdm

dirty_data = {
    "apartment_0": [],
    "apartment_1": [373, 391, 393, 396, 398],
    "apartment_2": list(range(1226, 1228)) + list(range(1256, 1269)) + list(range(1272, 1305)) + list(range(1324, 1347)) + list(range(1348, 1352)) + list(range(1365, 1376)) + list(range(1332, 1344)) + list(range(1435, 1461)) + list(range(1545, 1557)) + list(range(1563, 1592)) + [1516, 1529, 1533, 1551, 1581, 1293, 1294, 1295, 1298, 1262, 1312, 1319, 1314, 977],
    "frl_apartment_0": [0, 1],
    "frl_apartment_1": [],
    "frl_apartment_2": [],
    "frl_apartment_3": [],
    "frl_apartment_4": [],
    "frl_apartment_5": [],
    "hotel_0": list(range(614, 618)) + list(range(864, 869)) + list(range(1004, 1030)) + list(range(1051, 1057)) + [1128, 999, 1000],
    "office_0": list(range(26, 30)) + [0, ],
    "office_1": list(range(505, 511)) + [497, 499, 503, ],
    "office_2": [0, 420, 540, 541, 542, 545, 548, 549, 550, 551, 556, 557, 558, 560, 566, 567, 570, 571, 573, 574, 577],
    "office_3": list(range(321, 324)) + list(range(328, 331)) + list(range(856, 862)) + list(range(1047, 1051)) + [2, 4, 6, 7, 334, 340],
    "office_3": [],
    "room_0": [24],
    "room_1": list(range(574, 583)) + [573, 576, 580, 582],
    "room_2": list(range(625, 628)) + [259, ]
}

def remove_dirty_data(res_path, res_res_path):
    for scene in dirty_data.keys():
        scene_path = os.path.join(res_path, scene)
        res_scene_path = os.path.join(res_res_path, scene)
        
        if not os.path.exists(res_scene_path):
            os.makedirs(res_scene_path)
        
        eqr_path = os.path.join(scene_path, 'eqr')
        res_eqr_path = os.path.join(res_scene_path, 'eqr')
        if not os.path.exists(res_eqr_path):
            os.makedirs(res_eqr_path)

        fisheye_path = os.path.join(scene_path, 'fisheye')
        res_fisheye_path = os.path.join(res_scene_path, 'fisheye')
        if not os.path.exists(res_fisheye_path):
            os.makedirs(res_fisheye_path)
        
        path_list = []
        res_path_list = []
        for path in os.listdir(eqr_path):
            path_list.append(os.path.join(eqr_path, path))
            res_path_list.append(os.path.join(res_eqr_path, path))
        
        for path in os.listdir(fisheye_path):
            path_list.append(os.path.join(fisheye_path, path))
            res_path_list.append(os.path.join(res_fisheye_path, path))
        
        clean_num = 0
        for file_name in tqdm(sorted(os.listdir(os.path.join(path_list[0], "image")), key=lambda s: int(s.split('.')[0]))):
            file_num = file_name.split('.')[0]
            
            if int(file_num) not in dirty_data[scene]:    
                for path_num, path in enumerate(path_list):
                    image = os.path.join(path, "image", file_num + '.png')
                    pose = os.path.join(path, "pose", file_num + '.txt')
                    
                    res_image_path = os.path.join(res_path_list[path_num], "image")
                    res_pose_path = os.path.join(res_path_list[path_num], "pose")
                    
                    if not os.path.exists(res_image_path):
                        os.makedirs(res_image_path)
                    if not os.path.exists(res_pose_path):
                        os.makedirs(res_pose_path)
                    
                    res_image = os.path.join(res_image_path, str(clean_num) + '.png')
                    res_pose = os.path.join(res_pose_path, str(clean_num) + '.txt')
                    
                    shutil.copy(image, res_image)
                    shutil.copy(pose, res_pose)
            
                clean_num += 1


if __name__ == "__main__":
    res_path = '/home/star/Dataset/Replica_360/'
    res_res_path = '/home/star/Dataset/Replica_360_clean/'
    
    if not os.path.exists(res_res_path):
        os.makedirs(res_res_path)
    
    remove_dirty_data(res_path, res_res_path)