import random
from mmdet3d.datasets.pipelines.dbsampler import DataBaseSampler
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation   
from tqdm import tqdm
import math

def get_num_pts(sampled):
    return sampled["points"].tensor.shape[0]

prep = {'filter_by_difficulty': [-1], "filter_by_min_points": dict(
                    car=5,
                    truck=5,
                    bus=5,
                    trailer=5,
                    construction_vehicle=5,
                    traffic_cone=5,
                    barrier=5,
                    motorcycle=5,
                    bicycle=5,
                    pedestrian=5)}
sample_grps = {'car': 0, 'truck': 0, 'construction_vehicle': 0, 'bus': 0, 'trailer': 0, 'barrier': 0, 'motorcycle': 0, 'bicycle': 0, 'pedestrian': 1, 'traffic_cone': 0}

DSR = {c: 1.0 for c in sample_grps.keys()}
# DSS = 1.5
DSS = {c: [1.0,1.0] for c in sample_grps.keys()} # set all class DSS to 1 by default


max_range_class=dict(
                    car= 50, 
                    truck= 50, 
                    bus= 50, 
                    trailer= 50, 
                    construction_vehicle= 50, 
                    traffic_cone= 30, 
                    barrier= 30, 
                    motorcycle= 40, 
                    bicycle= 40, 
                    pedestrian= 40)

# Set DSR and DSS for specific classes
# DSR["car"] = 1
# DSS["car"] = [1.7, 2.2]
# DSR["construction_vehicle"] = 1
# DSS["construction_vehicle"] = [1.3,1.8]

# flip_xy = False
# meth = "FPS"
dbs_v4 = DataBaseSampler("./data/nuscenes/nuscenes_dbinfos_train.pkl", "./data/nuscenes/", 1, prep, sample_grps, sample_grps.keys(), points_loader=dict(type='LoadPointsFromFile', load_dim=5, use_dim=[0,1,2,3], coord_type='LIDAR'), ds_rate=DSR, long_range_fraction=1/3)

morethanptslimit = 0
lessthanptslimit = 0
outsiderange = 0
insiderange = 0
closeTo50 = 0

withinlongrange = 0
withinshortrange = 0

gt_names = ["car", "truck", "construction_vehicle", "bus", "trailer", "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"]

# gt_bboxes: x,y,z,w,l,h,theta
for i in tqdm(range(100000)):
    sampled = dbs_v4.sample_all(np.array([[200,200,200,.1,.1,.1,0,0,0]]),np.array([2]))
    dist = math.sqrt(sampled["gt_bboxes_3d"][0][0]**2+sampled["gt_bboxes_3d"][0][1]**2)
    if dist < max_range_class[gt_names[sampled["gt_labels_3d"][0]]]*(1-1/3):
        withinshortrange += 1
    elif dist >= max_range_class[gt_names[sampled["gt_labels_3d"][0]]]*(1-1/3) and dist < max_range_class[gt_names[sampled["gt_labels_3d"][0]]]:
        withinlongrange += 1
    else: 
        outsiderange += 1
    # if dist > 50:
    #     if dist < 55:
    #         closeTo50 += 1
    #     else:
    #         outsiderange += 1
    if get_num_pts(sampled) >= 5:
        insiderange+=1
        morethanptslimit += 1
    else:
        insiderange+=1
        lessthanptslimit += 1
        
print("more than pts limit:", morethanptslimit)
print("less than pts limit:", lessthanptslimit)
print("percentage less than spec num points:", lessthanptslimit/(morethanptslimit+lessthanptslimit))

print("percentage close to 50 (<55):", closeTo50/(insiderange+outsiderange))
print("percentage outside range 55:", outsiderange/(insiderange+outsiderange))

print("percentage within long range:", withinlongrange/(withinlongrange+withinshortrange+outsiderange))
print("percentage within short range:", withinshortrange/(withinlongrange+withinshortrange+outsiderange))
print("percentage outside range:", outsiderange/(withinlongrange+withinshortrange+outsiderange))

# # print("sampled objs:", sampled)


# sampledPts = sampled["points"]
# # sampledGtBboxes = sampled["gt_bboxes"]
# nppcd = sampledPts[:].tensor.numpy()
# # print(nppcd[0])
# # nppcd = downsample_and_move_away(nppcd)
# # print(nppcd.shape)
# # print(nppcd)
# nppcd = np.vstack((nppcd, np.array([0,0,0,1])))
# nppcd = np.vstack((nppcd, np.array([1,0,0,1])))
# # print(nppcd[0])
# # print("mean intensity:", np.mean(nppcd[:, 3]))
# # print("std intensity: ", np.sqrt(np.var(nppcd[:, 3])))


# print(sampled["gt_bboxes_3d"])

# print(type(sampled["gt_bboxes_3d"][0]))

# bbox = get_open3d_bbox(sampled["gt_bboxes_3d"][0])


# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(nppcd[:, :3])
# pcd.colors = o3d.utility.Vector3dVector(np.hstack((nppcd[:, 3:4]/np.max(nppcd[:, 3]),nppcd[:, 3:4]/np.max(nppcd[:, 3]),nppcd[:, 3:4]/np.max(nppcd[:, 3]))))
# plotdata = []
# plotdata.append(pcd)
# for b in sampled["gt_bboxes_3d"]:
#     bbox, dir_line = get_open3d_bbox(b)
#     plotdata.append(bbox)
#     plotdata.append(dir_line)
# o3d.visualization.draw_geometries(plotdata)

