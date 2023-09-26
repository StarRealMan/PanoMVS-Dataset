import os
import numpy as np
from scipy.spatial.transform import Rotation as R

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
    pos = pose[3:]
    pos[2] += 20.0
    rotR = R.from_rotvec(angle).as_matrix()
    T[:3,:3] = rotR
    T[:3, 3] = np.array(pos) / 100.0
    
    poses.append(T)
    
for i in range(4):
    np.savetxt("./poses/fisheye_poses/"+str(i)+".txt", poses[i])

# generate random poses for panorama camera
panorama_size = 48
radius = 0.5

theta = 2 * np.pi * np.random.rand(panorama_size)
phi = np.arccos(2 * np.random.rand(panorama_size) - 1)
r = radius * np.cbrt(np.random.rand(panorama_size))

x = r * np.sin(phi) * np.cos(theta)
y = r * np.sin(phi) * np.sin(theta)
z = r * np.cos(phi)
rot = [R.from_euler('xyz', 360 * np.random.rand(3), degrees=True) 
        for _ in range(panorama_size)]

for i in range(panorama_size):
    # Create a 4x4 transformation matrix for the pose
    pose = np.eye(4)
    pose[:3, :3] = rot[i].as_matrix()
    pose[:3, 3] = [x[i], y[i], z[i]]

    np.savetxt("./poses/panorama_poses/"+str(i)+".txt", pose)
