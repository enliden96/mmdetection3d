from mmdet3d.datasets.pipelines.dbsampler_v2 import DataBaseSampler_v2
from mmdet3d.datasets.pipelines.dbsampler import DataBaseSampler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

        
def plot_num_points_vs_range(location_info_per_class_v1, location_info_per_class_v2, m1mult):
    ''' NOTE: Use exactly one sample of one class at a time when using this function
    
    Plots the number of points in each object vs the range from the sensor'''
    
    radial_dists = {c: 50 for c in location_info_per_class_v1.keys()}
    radial_dists["pedestrian"] = 40
    radial_dists["bicycle"] = 40
    radial_dists["motorcycle"] = 40
    radial_dists["traffic_cone"] = 30
    radial_dists["barrier"] = 30
    
    for cl in location_info_per_class_v1:
        if location_info_per_class_v1[cl]["range"] == []:
            continue
        print(len(location_info_per_class_v2[cl]["range"]))
        print(len(location_info_per_class_v2[cl]["num_points"]))
        plt.figure(figsize=(16,10))
        plt.subplot(1,3,1)
        plt.scatter(location_info_per_class_v1[cl]["range"], location_info_per_class_v1[cl]["num_points"], c='b', alpha=.1)
        plt.gca().set_xlabel("range from sensor[m]")
        plt.gca().set_ylabel("number of points (log scale)")
        plt.gca().set_yscale("log")
        plt.ylim(1, 1e5)
        plt.xlim(0, radial_dists[cl]+10)
        plt.title("DBSampler v1 ")
        plt.subplot(1,3,2)
        plt.scatter(location_info_per_class_v2[cl]["range"], location_info_per_class_v2[cl]["num_points"], c='r', alpha=.1)
        plt.gca().set_xlabel("range from sensor[m]")
        plt.gca().set_ylabel("number of points (log scale)")
        plt.gca().set_yscale("log")
        plt.xlim(0, radial_dists[cl]+10)
        plt.ylim(1, 1e5)
        plt.title("DBSampler v2")
        
        plt.subplot(1,3,3)
        plt.scatter(location_info_per_class_v1[cl]["range"], location_info_per_class_v1[cl]["num_points"], c='b', alpha=.1)
        plt.scatter(location_info_per_class_v2[cl]["range"], location_info_per_class_v2[cl]["num_points"], c='r', alpha=.1)
        plt.gca().set_xlabel("range from sensor[m]")
        plt.gca().set_ylabel("number of points (log scale)")
        plt.gca().set_yscale("log")
        plt.xlim(0, radial_dists[cl]+10)
        plt.ylim(1, 1e5)
        plt.title("Overlayed")
        if m1mult:
            plt.savefig("plots/num_points_vs_range/num_points_vs_range_%s_DSR%.2f_DSS%.2f-%.2f_flipxy_classSpecificRates.png" % (cl, DSR[cl], DSS[cl][0], DSS[cl][1]))
        else:
            plt.savefig("plots/num_points_vs_range/num_points_vs_range_%s_DSR%.2f_DSS%.2f-%.2f.png" % (cl, DSR[cl], DSS[cl][0], DSS[cl][1]))
        plt.clf()

prep_v1 = {'filter_by_difficulty': [-1], 'filter_by_min_points': {'car': 5, 'truck': 5, 'bus': 5, 'trailer': 5, 'construction_vehicle': 5, 'traffic_cone': 5, 'barrier': 5, 'motorcycle': 5, 'bicycle': 5, 'pedestrian': 5}}
prep_v2 = {'filter_by_difficulty': [-1], 'filter_by_min_points': {'car': 50, 'truck': 200, 'bus': 200, 'trailer': 400, 'construction_vehicle': 100, 'traffic_cone': 5, 'barrier': 15, 'motorcycle': 15, 'bicycle': 15, 'pedestrian': 15}}
# sample_grps = {'car': 2, 'truck': 2, 'construction_vehicle': 2, 'bus': 2, 'trailer': 2, 'barrier': 2, 'motorcycle': 2, 'bicycle': 2, 'pedestrian': 2, 'traffic_cone': 2}
sample_grps = {'car': 0, 'truck': 0, 'construction_vehicle': 0, 'bus': 0, 'trailer': 0, 'barrier': 0, 'motorcycle': 0, 'bicycle': 0, 'pedestrian': 1, 'traffic_cone': 0}
# sample_grps=dict(
        # car=2,
        # truck=3,
        # construction_vehicle=7,
        # bus=4,
        # trailer=6,
        # barrier=2,
        # motorcycle=6,
        # bicycle=6,
        # pedestrian=2,
        # traffic_cone=2)



# DSR = .5
DSR = {c: 0.5 for c in sample_grps.keys()}
# DSS = 1.5
DSS = {c: [1.02, 2.5] for c in sample_grps.keys()} # set all class DSS to 1 by default

### Set DSR and DSS for specific classes
DSR["pedestrian"] = 1.0
DSS["pedestrian"] = [1.0, 2.0]

m1mult = True
dbs_v1 = DataBaseSampler("./data/nuscenes/nuscenes_dbinfos_train.pkl", "./data/nuscenes/", 1, prep_v1, sample_grps, sample_grps.keys(), points_loader=dict(type='LoadPointsFromFile', load_dim=5, use_dim=[0,1,2,3], coord_type='LIDAR'))
dbs_v2 = DataBaseSampler_v2("./data/nuscenes/nuscenes_dbinfos_train.pkl", "./data/nuscenes/", 1, prep_v2, sample_grps, sample_grps.keys(), points_loader=dict(type='LoadPointsFromFile', load_dim=5, use_dim=[0,1,2,3], coord_type='LIDAR'), ds_rate=DSR, ds_scale=DSS, ds_flip_xy=m1mult)

# gt_bboxes: x,y,z,w,l,h,theta

class_count_v1 = []
class_count_v2 = []

location_info_per_class_v1 = {c: {"range": [], "x": [], "y": [], "z": [], "num_points": []} for c in sample_grps.keys()}
location_info_per_class_v2 = {c: {"range": [], "x": [], "y": [], "z": [], "num_points": []} for c in sample_grps.keys()}

class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

num_sample_iters = 10000

for i in tqdm(range(num_sample_iters)):
    sampled = dbs_v1.sample_all(np.array([[200,200,200,.1,.1,.1,0,0,0]]),np.array([0]))
    for c, gtbb in zip(sampled["gt_labels_3d"], sampled["gt_bboxes_3d"]):
        class_count_v1.append(c)
        location_info_per_class_v1[class_names[c]]["range"].append(np.sqrt(gtbb[0]**2 + gtbb[1]**2 + gtbb[2]**2))
        location_info_per_class_v1[class_names[c]]["x"].append(gtbb[0])
        location_info_per_class_v1[class_names[c]]["y"].append(gtbb[1])
        location_info_per_class_v1[class_names[c]]["z"].append(gtbb[2])
        location_info_per_class_v1[class_names[c]]["num_points"].append(len(sampled["points"]))

for i in tqdm(range(num_sample_iters)):
    sampled = dbs_v2.sample_all(np.array([[200,200,200,.1,.1,.1,0,0,0]]),np.array([0]))
    # Remove samples that contain less than 5 points after downsampling
    if len(sampled["points"]) < 5:
        continue
    for c, gtbb in zip(sampled["gt_labels_3d"], sampled["gt_bboxes_3d"]):
        class_count_v2.append(c)
        location_info_per_class_v2[class_names[c]]["range"].append(np.sqrt(gtbb[0]**2 + gtbb[1]**2 + gtbb[2]**2))
        location_info_per_class_v2[class_names[c]]["x"].append(gtbb[0])
        location_info_per_class_v2[class_names[c]]["y"].append(gtbb[1])
        location_info_per_class_v2[class_names[c]]["z"].append(gtbb[2])
        location_info_per_class_v2[class_names[c]]["num_points"].append(len(sampled["points"]))


# # plot_class_count_bar(class_names, class_count, num_sample_iters)

v1polyfitconst, covv1 = np.polyfit(location_info_per_class_v1["pedestrian"]["range"], location_info_per_class_v1["pedestrian"]["num_points"] , 2, cov=True)
v2polyfitconst, covv2 = np.polyfit(location_info_per_class_v2["pedestrian"]["range"], location_info_per_class_v2["pedestrian"]["num_points"] , 2, cov=True)

print(v1polyfitconst)
print(v2polyfitconst)


x = np.linspace(0, 100, 100)
plt.plot(x, np.polyval(v1polyfitconst, x), label="v1")
plt.plot(x, np.polyval(v2polyfitconst, x), label="v2")
plt.legend()
plt.show()

plot_num_points_vs_range(location_info_per_class_v1, location_info_per_class_v2, m1mult=m1mult) # m1mult same thing as ds_flip_xy

