import os
import numpy as np
from scipy.spatial.transform import Rotation as Rot

# load fisheye camera info from OmniMVS dataset
data_dir = "/home/star/Dataset/OmniMVS_dataset/OmniThings/omnithings"

poses = []
pose_file = os.path.join(data_dir, "poses.txt")
with open(pose_file) as f:
    data = f.readlines()
    
for pose in data:
    pose = list(map(float, pose.split()))
    T = np.eye(4)
    angle = pose[:3]
    trans = pose[3:]
    trans[2] += 20.0
    R = Rot.from_rotvec(angle).as_matrix()
    T[:3,:3] = R
    T[:3, 3] = np.array(trans) / 100.0
    
    poses.append(T)
    
for i in range(4):
    np.savetxt("./poses/fisheye_poses/"+str(i)+".txt", poses[i])

# # generate random poses for panorama camera
# panorama_size = 48

# for i in range(panorama_size):
#     np.savetxt("./poses/panorama_poses/"+str(i)+".txt", poses[i])