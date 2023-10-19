import os
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


if __name__ == '__main__':
    eqr_size = 512
    eqr_mode = 'bilinear'
    cube_format = 'dice'
    cube_path = './cubemap'
    eqr_path = './eqr'
    
    if not os.path.exists(eqr_path):
        os.makedirs(eqr_path)
    
    for num, num_path in enumerate(tqdm(os.listdir(cube_path))):
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
        
        cubemap = cubemap.astype(np.float32) / 255.0
        
        # Convert to equirectangular
        equirec = c2e.c2e(cubemap, eqr_size, eqr_size * 2, eqr_mode, cube_format)
        
        # Save equirectangular
        cv2.imwrite(os.path.join(eqr_path, str(num) + '.png'), equirec * 255.0)