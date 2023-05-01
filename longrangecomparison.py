import random
from mmdet3d.datasets.pipelines.dbsampler import DataBaseSampler
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation   
from tqdm import tqdm

def get_open3d_bbox(bbox, color=(1,0,0)):
    bbox = bbox.copy()
    # bbox[2] = bbox[2] - bbox[5]/2
    # bbox[3] = bbox[3] - bbox[4]/2
    # bbox[4] = bbox[4] - bbox[3]/2
    # bbox[5] = bbox[5] - bbox[2]/2
    # bbox[6] = bbox[6] - np.pi/2
    bbox[2] += .9
    bbox = bbox.tolist()
    rot = Rotation.from_euler('xyz', [0, 0, bbox[6]], degrees=False).as_matrix()
    cent = bbox[:3]
    bbox = o3d.geometry.OrientedBoundingBox(bbox[:3], rot, bbox[3:6])
    bbox.color = color
    lineset = o3d.geometry.LineSet()
    points = [[1, 0, 0], [4, 0, 0]]
    line_ind = [[0,1]]
    lineset.points = o3d.utility.Vector3dVector(points)
    lineset.translate(cent)
    lineset.rotate(bbox.R, center = cent)  
    lineset.lines = o3d.utility.Vector2iVector(line_ind)
    lineset.colors = o3d.utility.Vector3dVector([[0,1,.5]])
    return bbox, lineset


def get_range_2d(x, y):
    return np.sqrt(x**2 + y**2)
    


################## OLD DS FUNC ##################

# def downsample_and_move_away(sample_pcd):
#     ds_ratio = 1
#     print("...................................................")
#     print(sample_pcd)
#     #save intensity and z columns
#     zsaved = sample_pcd[:,2]
#     intsaved = sample_pcd[:,3]
#     print(intsaved)
#     sample_pcd = sample_pcd[:,:2]+ds_ratio
#     print(sample_pcd)
#     sample_pcd = np.hstack((sample_pcd, zsaved.reshape(-1,1), intsaved.reshape(-1,1)))
#     # return sample_pcd
#     tmppcd = np.ndarray((0,4))
#     print(tmppcd)
#     for r in sample_pcd:
#         if random.random() < 1/ds_ratio:
#             tmppcd = np.vstack((tmppcd, r))
#     print("returned downsampled pcd: ", tmppcd)
#     return tmppcd
        
###############################################

prep = {'filter_by_difficulty': [-1], 'filter_by_min_points': {'car': 5, 'truck': 5, 'bus': 5, 'trailer': 5, 'construction_vehicle': 5, 'traffic_cone': 5, 'barrier': 5, 'motorcycle': 5, 'bicycle': 5, 'pedestrian': 5}}
sample_grps = {'car': 1, 'truck': 0, 'construction_vehicle': 0, 'bus': 0, 'trailer': 0, 'barrier': 0, 'motorcycle': 0, 'bicycle': 0, 'pedestrian': 0, 'traffic_cone': 0}

DSR = {c: 0.0 for c in sample_grps.keys()}
# DSS = 1.5
DSS = {c: [1.5,1.5] for c in sample_grps.keys()} # set all class DSS to 1 by default

# Set DSR and DSS for specific classes
# DSR["car"] = 0
# DSS["car"] = [1.7, 2.2]
# DSR["barrier"] = 1
# DSR["construction_vehicle"] = 1
# DSS["construction_vehicle"] = [1.3,1.8]

flip_xy = False
meth = "Random"
dbs_v4 = DataBaseSampler("./data/nuscenes/nuscenes_dbinfos_train.pkl", "./data/nuscenes/", 1, prep, sample_grps, sample_grps.keys(), points_loader=dict(type='LoadPointsFromFile', load_dim=5, use_dim=[0,1,2,3], coord_type='LIDAR'), ds_rate=DSR, long_range_fraction=1/3, ds_flip_xy=flip_xy, ds_method=meth)



# gt_bboxes: x,y,z,w,l,h,theta
for i in tqdm(range(100000)):
    sampled = dbs_v4.sample_all(np.array([[200,200,200,.1,.1,.1,0,0,0]]),np.array([2]))
    # print(sampled)
    # print(sampled["gt_bboxes_3d"])
    r = get_range_2d(sampled["gt_bboxes_3d"][0][0], sampled["gt_bboxes_3d"][0][1]) 
    npoints = sampled["points"].shape[0]
    if r > 40 and r < 41 and npoints > 50 and npoints < 75:
        sampled_no_ds = sampled
        break

DSR = {c: 1.0 for c in sample_grps.keys()}
    
dbs_v4 = DataBaseSampler("./data/nuscenes/nuscenes_dbinfos_train.pkl", "./data/nuscenes/", 1, prep, sample_grps, sample_grps.keys(), points_loader=dict(type='LoadPointsFromFile', load_dim=5, use_dim=[0,1,2,3], coord_type='LIDAR'), ds_rate=DSR, long_range_fraction=1/3, ds_flip_xy=flip_xy, ds_method=meth)

for i in tqdm(range(100000)):
    sampled = dbs_v4.sample_all(np.array([[200,200,200,.1,.1,.1,0,0,0]]),np.array([2]))

    r = get_range_2d(sampled["gt_bboxes_3d"][0][0], sampled["gt_bboxes_3d"][0][1])
    npoints = sampled["points"].shape[0]
    if r > 40 and r < 41 and npoints > 50 and npoints < 75:
        sampled_ds = sampled
        break

# print("-----------------------")
# print(sampled_ds)
# print(sampled_no_ds)
# print("sampled objs:", sampled)


############ no ds ############

sampledPts = sampled_no_ds["points"]
# sampledGtBboxes = sampled["gt_bboxes"]
nppcd = sampledPts[:].tensor.numpy()
# print(nppcd[0])
# nppcd = downsample_and_move_away(nppcd)
# print(nppcd.shape)
# print(nppcd)
# nppcd = np.vstack((nppcd, np.array([0,0,0,1])))
# nppcd = np.vstack((nppcd, np.array([1,0,0,1])))
# print(nppcd[0])
# print("mean intensity:", np.mean(nppcd[:, 3]))
# print("std intensity: ", np.sqrt(np.var(nppcd[:, 3])))


# print(sampled["gt_bboxes_3d"])

# print(type(sampled["gt_bboxes_3d"][0]))

bbox = get_open3d_bbox(sampled_no_ds["gt_bboxes_3d"][0])


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(nppcd[:, :3])
pcd.colors = o3d.utility.Vector3dVector(np.hstack((nppcd[:, 3:4]/np.max(nppcd[:, 3]),nppcd[:, 3:4]/np.max(nppcd[:, 3]),nppcd[:, 3:4]/np.max(nppcd[:, 3]))))
plotdata = []
plotdata.append(pcd)
for b in sampled_no_ds["gt_bboxes_3d"]:
    bbox, dir_line = get_open3d_bbox(b)
    plotdata.append(bbox)
    plotdata.append(dir_line)
o3d.visualization.draw_geometries(plotdata)

############## with ds ###############

sampledPts = sampled_ds["points"]
# sampledGtBboxes = sampled["gt_bboxes"]
nppcd = sampledPts[:].tensor.numpy()
# print(nppcd[0])
# nppcd = downsample_and_move_away(nppcd)
# print(nppcd.shape)
# print(nppcd)
# nppcd = np.vstack((nppcd, np.array([0,0,0,1])))
# nppcd = np.vstack((nppcd, np.array([1,0,0,1])))
# print(nppcd[0])
# print("mean intensity:", np.mean(nppcd[:, 3]))
# print("std intensity: ", np.sqrt(np.var(nppcd[:, 3])))


# print(sampled["gt_bboxes_3d"])

# print(type(sampled["gt_bboxes_3d"][0]))

bbox = get_open3d_bbox(sampled_ds["gt_bboxes_3d"][0])


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(nppcd[:, :3])
pcd.colors = o3d.utility.Vector3dVector(np.hstack((nppcd[:, 3:4]/np.max(nppcd[:, 3]),nppcd[:, 3:4]/np.max(nppcd[:, 3]),nppcd[:, 3:4]/np.max(nppcd[:, 3]))))
plotdata = []
plotdata.append(pcd)
for b in sampled_ds["gt_bboxes_3d"]:
    bbox, dir_line = get_open3d_bbox(b)
    plotdata.append(bbox)
    plotdata.append(dir_line)
o3d.visualization.draw_geometries(plotdata)
