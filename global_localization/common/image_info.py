import numpy as np
from scipy.spatial.transform import Rotation as R


def make_images_info(struct_filename):
    """
    Load database SPI info from structure file
    Structure file format:
    line 1: head info
    line 2 - end:
    :param struct_filename: xxxx.txt, str
    :return:
    """
    images_info = []
    with open(struct_filename, "r") as struct_file:
        # skip the first line
        struct_file.readline()
        while True:
            line = struct_file.readline()
            if not line:
                break
            split = [i for i in line.split(",")]
            # split: image_file, timestamp,
            #        translation x, translation y, translation z,
            #        quaternion w, quaternion x, quaternion y, quaternion z
            position = np.array([float(split[2]), float(split[3]), float(split[4])])
            quaternion = np.array([float(split[6]), float(split[7]), float(split[8]), float(split[5])]) # x, y, z, w
            rotation = R.from_quat(quaternion).as_matrix()
            pose = np.hstack([rotation, position[...,None]]) # 3 * 4
            pose = np.vstack([pose, np.array([[0,0,0,1]])])

            images_info.append({
                'image_file': split[0],
                'timestamp': float(split[1]),
                'pose': pose,
                'vlad': None,
                'features': None,
            })
    return images_info