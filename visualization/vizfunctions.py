import numpy as np
from pyquaternion import Quaternion


BUILDER = np.array([[1, 1, 1,],
                    [1, 1, -1,],
                    [1, -1, 1,],
                    [1, -1, -1,],
                    [-1, 1, 1,],
                    [-1, 1, -1,],
                    [-1, -1, 1,],
                    [-1, -1, -1,],
                    [0.5, 0, 1],
                    [1, 0, 1]]).T

boxEdges = np.array([[1,2],
                     [1,3],
                     [1,10],
                     [2,4],
                     [2,6],
                     [3,4],
                     [3,7],
                     [4,8],
                     [5,1],
                     [5,6],
                     [5,7],
                     [6,8],
                     [7,8],
                     [9,10],])

def get_pred_box(pred_object,car_from_global,ref_from_car):

    pred_translation = np.matmul(np.matmul(ref_from_car,car_from_global),np.append(np.array(pred_object['translation']),1))
    rotation_quaternion = Quaternion(pred_object['rotation'])*Quaternion(matrix=car_from_global)*Quaternion(matrix=ref_from_car)
    rot_z = rotation_quaternion.rotation_matrix

    return np.matmul(rot_z,np.reshape(np.array(pred_object['size'])[[1,0,2]]/2,(3,1))*BUILDER).T + pred_translation[0:3]


def get_gt_boxes(gt_vector):

    theta = gt_vector[6] # yaw

    rotz = np.array([[np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]])

    return np.matmul(rotz,np.reshape(gt_vector[3:6]/2,(3,1))*BUILDER).T + gt_vector[0:3]

def get_class_matrix(gt_name):

    if gt_name == 'car':                    # Red
        name_vec = np.array([255, 0, 0])
    elif gt_name == 'pedestrian':           # Green
        name_vec = np.array([0, 255, 0])
    elif gt_name == 'traffic cone':         # Orange
        name_vec = np.array([255, 173, 0])
    elif gt_name == 'bus':                  # White
        name_vec = np.array([255, 255, 255])
    elif gt_name == 'truck':                # Black
        name_vec = np.array([0,0,0])
    elif gt_name == 'bicycle':              # Purple
        name_vec = np.array([255,0,255])
    elif gt_name == 'barrier':              # Pink
        name_vec = np.array([255, 0, 232])
    elif gt_name == 'construction vehicle': # Yellow
        name_vec = np.array([255, 255, 0])
    elif gt_name == 'motorcycle':           # Blue
        name_vec = np.array([0, 0, 255])
    elif gt_name == 'trailer':              # Brown
        name_vec = np.array([198, 122, 55])
    else:                                   # Gray
        name_vec = np.array([128, 128, 128])

    return np.zeros((10,3)) + name_vec
