import numpy as np
import pandas as pd

from pyquaternion import Quaternion
from vizfunctions import get_pred_box, get_gt_boxes, get_class_matrix, boxEdges
from math import sqrt

## testing
import json, time, os
from nuscenes.nuscenes import NuScenes
# from modfunctions import evaluate,config_factory, filter_eval_boxes
from nuscenes.eval.common.loaders import load_prediction, load_gt,add_center_dist
from nuscenes.eval.detection.data_classes import DetectionBox
## END testing

classList = ['car','truck','bus','trailer','construction_vehicle','pedestrian','motorcycle','bicycle','traffic_cone','barrier']
modelListAll = ['no_gt_sampler', 
                'baseline_no_fade',
                'baseline_no_fade_2x', 
                'baseline_with_fade',
                'baseline_with_fade_2x', 
                'v3_fps_ds',
                'v3_fps_ds_fade', 
                'v3_random_ds', 
                'v3_random_ds_fade',
                'v4_random_ds', 
                'v4_random_ds_fade', 
                'v4_longrange_bias', 
                'v4_longrange_bias_with_fade', 
                'v5_no_fade', 
                'v5_with_fade', 
                'v6_no_fade', 
                'v6_with_fade']

class_range = {
            'car': 50,
            'truck': 50,
            'bus': 50,
            'trailer': 50,
            'construction_vehicle': 50,
            'pedestrian': 40,
            'motorcycle': 40,
            'bicycle': 40,
            'traffic_cone': 30,
            'barrier': 30
          }

### Settings ###

# modelversion = 'baseline_with_fade'

for modelversion in modelListAll:
    datasetPath = 'data/nuscenes/nuscenes_infos_val.pkl'
    resultPath = 'results/HPC/' + modelversion + '/pts_bbox/results_nusc.json'
    savepath = '/home/simulation-desktop/Downloads/vis/db/' + modelversion + '/' # savefolder

    if not os.path.isdir(savepath):
        os.mkdir(savepath)

    detection_limit = 0.1

    clas = 'bicycle'

    # frame_hash = "f2efcbf8dc2f4cf4a4d96a9fc6ec0d84" # 0
    # frame_hash = "8ffab42b57d14c6cb0486620d8675a74" # 0
    # frame_hash = "b010e143510d4325b27cb65616411838" # 0
    # frame_hash = "8135e2527da0405d9887a6a9b456586f" # 0
    # frame_hash = "096112c80e6845a68ec19680d987a24c" # 0
    # frame_hash = "e31dad83431d497eb58c285b85453a40" # 0
    frame_hash = "b2024b635e624a719d16d85b395447d3"
    # frame_hash = "942605eb508545f5a45c12c55c76832b"
    # frame_hash = "51bbacdcf4c94b8c977ab36f1f0c9e1b"
    # frame_hash = "c735e5f28290490b9fe269fcc6f1f895"

    # frame_hash '9cb4d97844624443b1f7c3333bb13c5c' # Singapore pretty bad
    # frame_hash = '8d966583bc7c4627a2437be13761529c' # used in presentation
    # frame_hash = 'f75968e911a94a3e8eea092422614231' # flying motorcycle
    # frame_hash = '01a7d01c014f406f87b8fe11976c3d0a'
    # frame_hash = '94e396a143d64345a09f671443f942bc'
    # frame_hash = '41ce99072e8242c0b4fd7840f772f9fd' # Pretty good
    # frame_hash = '942c0bdb5e1b48628fb9d4bf3435511d' # Really bad


    ### Load data and prepare indexing ###

    datasetData = pd.read_pickle(datasetPath)
    datasetData = datasetData['infos']
    resultData = pd.read_json(resultPath)
    resultData = resultData['results'][5:]

    hash2ind = {}
    for d in range(len(datasetData)):
        hash2ind[datasetData[d]['token']] = d

    frame_index = hash2ind[frame_hash]


    ## testing

    # verbose = False
    # cfg = config_factory("cvpr_2019")
    # nusc = NuScenes("v1.0-trainval","data/nuscenes",verbose=verbose)

    # print('Loading boxes:')
    # start = time.time()
    # pred_boxes, meta = load_prediction(resultPath,cfg.max_boxes_per_sample,DetectionBox, verbose=verbose)
    # gt_boxes = load_gt(nusc,'val',DetectionBox,True)

    # print('Adding center distances: ')
    # pred_boxes = add_center_dist(nusc, pred_boxes)
    # gt_boxes = add_center_dist(nusc, gt_boxes)

    # print('Filtering boxes: ')
    # pred_boxes = filter_eval_boxes(nusc, pred_boxes, cfg.class_range, longrange=False, verbose=verbose)
    # gt_boxes = filter_eval_boxes(nusc, gt_boxes, cfg.class_range, longrange=False, verbose=verbose)

    # gtboxlist = [box for box in gt_boxes[frame_hash] if box.ego_dist < cfg.class_range[box.__getattribute__('detection_name')]]

    ## END testing
    ### Create homogenous transforms for box data ###

    ego2lidar_translation = -np.array(datasetData[frame_index]['lidar2ego_translation'])
    ego2lidar_rotation = Quaternion(np.array(datasetData[frame_index]['lidar2ego_rotation'])).rotation_matrix.T
    global2ego_translation = -np.array(datasetData[frame_index]['ego2global_translation'])
    global2ego_rotation = Quaternion(np.array(datasetData[frame_index]['ego2global_rotation'])).rotation_matrix.T

    ego2lidar_hom_trans = np.append(np.append(ego2lidar_rotation,ego2lidar_rotation@np.reshape(ego2lidar_translation,(3,1)),1),np.zeros((1,4)),0)
    ego2lidar_hom_trans[3,3] = 1
    global2ego_hom_trans = np.append(np.append(global2ego_rotation,global2ego_rotation@np.reshape(global2ego_translation,(3,1)),1),np.zeros((1,4)),0)
    global2ego_hom_trans[3,3] = 1

    ### Create prediction boxes ###
    for clas in classList:
        printfile = open(savepath + modelversion + "_pred_" + clas + ".obj", "w")
        counter = 0

        for i in range(len(resultData[frame_hash])):
            if resultData[frame_hash][i]['detection_score'] > detection_limit:
                if resultData[frame_hash][i]['detection_name'] == clas:
                    predVert = get_pred_box(resultData[frame_hash][i],global2ego_hom_trans,ego2lidar_hom_trans)
                    predVert = np.append(predVert,get_class_matrix(resultData[frame_hash][i]['detection_name']),1)
                    np.savetxt(printfile,predVert,"v %.6f %.6f %.6f %.6f %.6f %.6f","","\n")
                    counter = counter+1
        for i in range(counter):
            np.savetxt(printfile,boxEdges+i*np.array([10,10]),"l %i %i","","\n")
        printfile.close()

        ### Create ground truth boxes ###

        num_gt = len(datasetData[frame_index]['gt_boxes'])
        printfile = open(savepath+ modelversion + "_gt_" + clas + ".obj", "w")

        for i in range(num_gt):
            # print(datasetData[frame_index]['gt_boxes'][i])
            # print('#################################################################################################################')
            # # print('gt_box: ', gtboxlist[i])
            # print('box: ', datasetData[frame_index]['gt_boxes'][i])
            # print('class: ', datasetData[frame_index]['gt_names'][i])
            # # lägg in check för class range
            # distance = sqrt(datasetData[frame_index]['gt_boxes'][i][0]**2 + datasetData[frame_index]['gt_boxes'][i][1]**2)
            # print('distance: ', distance)
            # print('num_lidarpts: ', datasetData[frame_index]['num_lidar_pts'][i]+datasetData[frame_index]['num_radar_pts'][i])

            # if datasetData[frame_index]['num_lidar_pts'][i]+datasetData[frame_index]['num_radar_pts'][i] != 0 and distance < class_range[datasetData[frame_index]['gt_names'][i]]:
            if datasetData[frame_index]['gt_names'][i] == clas:
                gtVert = np.append(get_gt_boxes(datasetData[frame_index]['gt_boxes'][i]),get_class_matrix(datasetData[frame_index]['gt_names'][i]),1)
                np.savetxt(printfile,gtVert,"v %.6f %.6f %.6f %.6f %.6f %.6f","","\n")
            else:
                num_gt = num_gt - 1
            # else:
                # print('filtered')
            # print('#################################################################################################################')

        for i in range(num_gt):
            np.savetxt(printfile,boxEdges+i*np.array([10,10]),"l %i %i","","\n")
        printfile.close()
        print('class: ', clas,' num_gt: ', num_gt)

    ### Create frame pointcloud object ###

    file_path = datasetData[frame_index]['lidar_path']
    pcData = np.fromfile(file_path,dtype=np.float32)
    printfile = open(savepath+"pointcloud.obj", "w")
    np.savetxt(printfile,np.reshape(pcData,(int(len(pcData)/5),5))[:,0:3],"v %.6f %.6f %.6f","","\n")

    ### Update frame object with sweeps ###

    for i in range(len(datasetData[frame_index]['sweeps'])):              
        file_path = datasetData[frame_index]['sweeps'][i]['data_path']

        sensor2lidar_translation = np.array(datasetData[frame_index]['sweeps'][i]['sensor2lidar_translation'])
        sensor2lidar_rotation = np.array(datasetData[frame_index]['sweeps'][i]['sensor2lidar_rotation'])
        transformationMatrix = np.append(np.append(sensor2lidar_rotation,np.reshape(sensor2lidar_translation,(3,1)),1),np.zeros((1,4)),0)

        pcData = np.fromfile(file_path,dtype=np.float32)
        pcData = np.reshape(pcData,(int(len(pcData)/5),5))[:,0:4]
        pcData[:,3] = 1
        np.savetxt(printfile,np.matmul(transformationMatrix,pcData.T)[0:3,:].T,"v %.6f %.6f %.6f","","\n")

    printfile.close()

    print('Pointcloud objects succesfully saved to: ', savepath)

