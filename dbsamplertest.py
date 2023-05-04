import random
from mmdet3d.datasets.pipelines.dbsampler import DataBaseSampler
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

# def split_points_into_bboxes(pcd, bboxes):
#     pts_split = 
#     for pts in pcd:
#         if 
def get_open3d_bbox(bbox, color=(1,0,0)):
    print(bbox)
    bbox = bbox.copy()
    # bbox[2] = bbox[2] - bbox[5]/2
    # bbox[3] = bbox[3] - bbox[4]/2
    # bbox[4] = bbox[4] - bbox[3]/2
    # bbox[5] = bbox[5] - bbox[2]/2
    # bbox[6] = bbox[6] - np.pi/2
    bbox[2] += .9
    bbox = bbox.tolist()
    rot = Rotation.from_euler('xyz', [0, 0, bbox[6]], degrees=False).as_matrix()
    print(rot)
    print(bbox[6])
    cent = bbox[:3]
    bbox = o3d.geometry.OrientedBoundingBox(bbox[:3], rot, bbox[3:6])
    bbox.color = color
    print(bbox.R)
    print(bbox)
    lineset = o3d.geometry.LineSet()
    print("--------------------------------------")
    points = [[1, 0, 0], [4, 0, 0]]
    line_ind = [[0,1]]
    lineset.points = o3d.utility.Vector3dVector(points)
    lineset.translate(cent)
    lineset.rotate(bbox.R, center = cent)  
    lineset.lines = o3d.utility.Vector2iVector(line_ind)
    lineset.colors = o3d.utility.Vector3dVector([[0,1,.5]])
    return bbox, lineset
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

prep = {'filter_by_difficulty': [-1],'filter_by_range': {'car': (0,20), 'truck': (33,50), 'bus': (33,50), 'trailer': (33,50), 'construction_vehicle': (33,50), 'traffic_cone': (20,30), 'barrier': (20,30), 'motorcycle': (27,40), 'bicycle': (27,40), 'pedestrian': (27,40)} ,'filter_by_min_points': {'car': 5, 'truck': 5, 'bus': 5, 'trailer': 5, 'construction_vehicle': 5, 'traffic_cone': 5, 'barrier': 5, 'motorcycle': 5, 'bicycle': 5, 'pedestrian': 5}}
sample_grps = {'car': 20, 'truck': 0, 'construction_vehicle': 0, 'bus': 0, 'trailer': 0, 'barrier': 0, 'motorcycle': 0, 'bicycle': 0, 'pedestrian': 0, 'traffic_cone': 0}

DSR = {c: 0.5 for c in sample_grps.keys()}
# DSS = 1.5
DSS = {c: [1.5,1.5] for c in sample_grps.keys()} # set all class DSS to 1 by default

# Set DSR and DSS for specific classes
DSR["car"] = 1
DSS["car"] = [1,5]

flip_xy = True
dbs_v2 = DataBaseSampler("./data/nuscenes/nuscenes_dbinfos_train.pkl", "./data/nuscenes/", 1, prep, sample_grps, sample_grps.keys(), points_loader=dict(type='LoadPointsFromFile', load_dim=5, use_dim=[0,1,2,3], coord_type='LIDAR'),ds_flip_xy=flip_xy)



# gt_bboxes: x,y,z,w,l,h,theta
sampled = dbs_v2.sample_all(np.array([[5,5,5,.1,.1,.1,0,0,0],[-5,-5,-5,.1,.1,.1,0,0,0]]),np.array([2,2]))

# print("sampled objs:", sampled)



sampledPts = sampled["points"]
nppcd = sampledPts[:].tensor.numpy()
# print(nppcd[0])
# nppcd = downsample_and_move_away(nppcd)
# print(nppcd.shape)
# print(nppcd)
nppcd = np.vstack((nppcd, np.array([0,0,0,1])))
nppcd = np.vstack((nppcd, np.array([1,0,0,1])))
# print(nppcd[0])
# print("mean intensity:", np.mean(nppcd[:, 3]))
# print("std intensity: ", np.sqrt(np.var(nppcd[:, 3])))
print(sampled["gt_bboxes_3d"])

print(type(sampled["gt_bboxes_3d"][0]))

bbox = get_open3d_bbox(sampled["gt_bboxes_3d"][0])


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(nppcd[:, :3])
pcd.colors = o3d.utility.Vector3dVector(np.hstack((nppcd[:, 3:4]/np.max(nppcd[:, 3]),nppcd[:, 3:4]/np.max(nppcd[:, 3]),nppcd[:, 3:4]/np.max(nppcd[:, 3]))))
plotdata = []
plotdata.append(pcd)
for b in sampled["gt_bboxes_3d"]:
    bbox, dir_line = get_open3d_bbox(b)
    plotdata.append(bbox)
    plotdata.append(dir_line)
o3d.visualization.draw_geometries(plotdata)