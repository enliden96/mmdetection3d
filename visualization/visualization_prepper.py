import numpy as np
import pandas as pd

from pyquaternion import Quaternion
from vizfunctions import get_pred_box, get_gt_boxes, get_class_matrix, boxEdges


### Settings ###

datasetPath = 'data/nuscenes/nuscenes_infos_val.pkl'
resultPath = 'results/pts_bbox/results_nusc.json'
savepath = '/home/simulation-desktop/Downloads/vis/db/' # savefolder

detection_limit = 0.1

# frame_hash '9cb4d97844624443b1f7c3333bb13c5c' # Singapore pretty bad
# frame_hash = '8d966583bc7c4627a2437be13761529c' # used in presentation
# frame_hash = 'f75968e911a94a3e8eea092422614231' # flying motorcycle
# frame_hash = '01a7d01c014f406f87b8fe11976c3d0a'
# frame_hash = '94e396a143d64345a09f671443f942bc'
frame_hash = '41ce99072e8242c0b4fd7840f772f9fd' # Pretty good
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

printfile = open(savepath+"frame_pred.obj", "w")
counter = 0

for i in range(len(resultData[frame_hash])):
    if resultData[frame_hash][i]['detection_score'] > detection_limit:
        predVert = get_pred_box(resultData[frame_hash][i],global2ego_hom_trans,ego2lidar_hom_trans)
        predVert = np.append(predVert,get_class_matrix(resultData[frame_hash][i]['detection_name']),1)
        np.savetxt(printfile,predVert,"v %.6f %.6f %.6f %.6f %.6f %.6f","","\n")
        counter = counter+1
for i in range(counter):
    np.savetxt(printfile,boxEdges+i*np.array([10,10]),"l %i %i","","\n")
printfile.close()

### Create ground truth boxes ###

num_gt = len(datasetData[frame_index]['gt_boxes'])
printfile = open(savepath+"frame_gt.obj", "w")

for i in range(num_gt):
    gtVert = np.append(get_gt_boxes(datasetData[frame_index]['gt_boxes'][i]),get_class_matrix(datasetData[frame_index]['gt_names'][i]),1)
    np.savetxt(printfile,gtVert,"v %.6f %.6f %.6f %.6f %.6f %.6f","","\n")

for i in range(num_gt):
    np.savetxt(printfile,boxEdges+i*np.array([10,10]),"l %i %i","","\n")
printfile.close()


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

