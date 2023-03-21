from mmdet3d.datasets.pipelines.dbsampler_v2 import DataBaseSampler_v2
from mmdet3d.datasets.pipelines.dbsampler import DataBaseSampler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

def plot_range_distribution(location_info_per_class_v1, location_info_per_class_v2, m1mult):
    
    for cl in location_info_per_class_v1:
        plt.subplot(1,2,1)
        plt.hist(location_info_per_class_v1[cl]["range"], bins=range(0,100,5), alpha=.5, label=cl)
        plt.gca().set_xlabel("range from sensor[m]")
        plt.gca().set_ylabel("number of samples")
        plt.title("DBSampler v1 (min pts all classes 5)")
        plt.subplot(1,2,2)
        plt.hist(location_info_per_class_v2[cl]["range"], bins=range(0,100,5), alpha=.5, label=cl)
        plt.gca().set_xlabel("range from sensor[m]")
        plt.gca().set_ylabel("number of samples")
        plt.title("DBSampler v2")
        if m1mult:
            plt.savefig("plots/range1e2/range_dist_%s_DSR%.2f_DSS%.2f-%.2f_flipxy_classSpecificRates.png" % (cl, DSR[cl], DSS[cl][0], DSS[cl][1]))
        else:
            plt.savefig("plots/range/range_dist_%s_DSR%.2f_DSS%.2f-%.2f.png" % (cl, DSR[cl], DSS[cl][0], DSS[cl][1]))
        plt.clf()
        
        
        
def plot_class_count_bar(class_names, class_count, num_sample_iters):
    class_count = np.array(class_count)

    _, counts = np.unique(class_count, return_counts=True)

    #Normalize by num sampling iterations
    counts = counts/num_sample_iters

    plt.bar(class_names, counts, align='center')
    plt.xticks(rotation=90)
    plt.grid(alpha=.5)
    plt.tight_layout()
    plt.title("Number of Ground Truth Objects per Class, normalized by number of sampling iterations")
    plt.show()


def scatter_object_coords(location_info_per_class_v1, location_info_per_class_v2, m1mult):

    radial_dists = {c: 50 for c in location_info_per_class_v1.keys()}
    radial_dists["pedestrian"] = 40
    radial_dists["bicycle"] = 40
    radial_dists["motorcycle"] = 40
    radial_dists["traffic_cone"] = 30
    radial_dists["barrier"] = 30
    
    # for i in tqdm(range(len(location_info["x"]))):
    #     plt.scatter(location_info["x"][i], location_info["y"][i], c='b', alpha=.1)
    
    # for x, y in tqdm(zip(location_info["x"], location_info["y"])):
    #     plt.scatter(x, y, c='b', alpha=.1)
    for cl in location_info_per_class_v1:
        
        #DB sampler v1
        plt.subplot(1,2,1)
        plt.scatter(location_info_per_class_v1[cl]["x"], location_info_per_class_v1[cl]["y"], c='b', alpha=.1)
        plt.gca().add_patch(patches.Rectangle((-50, -50), 100, 100, linewidth=1, edgecolor='r', facecolor='none'))
        plt.gca().add_patch(patches.Circle((0, 0), radial_dists[cl], linewidth=1, edgecolor='g', facecolor='none'))
        plt.xlim(-100, 100)
        plt.ylim(-100, 100)
        plt.gca().set_xlabel("x[m]")
        plt.gca().set_ylabel("y[m]")
        plt.tight_layout()
        plt.title("db sampler v1 (min pts all classes 5)")
        plt.gca().set_aspect('equal', adjustable='box')
        #DB sampler v2
        plt.subplot(1,2,2)
        plt.scatter(location_info_per_class_v2[cl]["x"], location_info_per_class_v2[cl]["y"], c='b', alpha=.1)
        plt.gca().add_patch(patches.Rectangle((-50, -50), 100, 100, linewidth=1, edgecolor='r', facecolor='none'))
        plt.gca().add_patch(patches.Circle((0, 0), radial_dists[cl], linewidth=1, edgecolor='g', facecolor='none'))
        plt.xlim(-100, 100)
        plt.ylim(-100, 100)
        plt.gca().set_xlabel("x[m]")
        plt.gca().set_ylabel("y[m]")
        plt.tight_layout()
        plt.title("db sampler v2")
        
        # set size and save
        plt.gcf().set_size_inches(20,10)
        plt.gca().set_aspect('equal', adjustable='box')
        if m1mult:
            plt.savefig("./plots/scatter1e2/scatter_%s_DSR%.2f_DSS%.2f-%.2f_flipxy_classSpecificRates.png" % (cl, DSR[cl], DSS[cl][0], DSS[cl][1]))
        else:
            plt.savefig("./plots/scatter/scatter_%s_DSR%.2f_DSS%.2f-%.2f.png" % (cl, DSR[cl], DSS[cl][0], DSS[cl][1]))
        plt.clf()
        
def hist2d_object_coords(location_info_per_class_v1, location_info_per_class_v2, m1mult):

    radial_dists = {c: 50 for c in location_info_per_class_v1.keys()}
    radial_dists["pedestrian"] = 40
    radial_dists["bicycle"] = 40
    radial_dists["motorcycle"] = 40
    radial_dists["traffic_cone"] = 30
    radial_dists["barrier"] = 30
    
    # for i in tqdm(range(len(location_info["x"]))):
    #     plt.scatter(location_info["x"][i], location_info["y"][i], c='b', alpha=.1)
    
    # for x, y in tqdm(zip(location_info["x"], location_info["y"])):
    #     plt.scatter(x, y, c='b', alpha=.1)
    for cl in location_info_per_class_v1:
        
        # DB sampler v1
        plt.subplot(1,2,1)
        plt.hist2d(location_info_per_class_v1[cl]["x"], location_info_per_class_v1[cl]["y"], bins=50, range=[[-100, 100], [-100, 100]])
        plt.gca().add_patch(patches.Rectangle((-50, -50), 100, 100, linewidth=1, edgecolor='c', facecolor='none'))
        plt.gca().add_patch(patches.Circle((0, 0), radial_dists[cl], linewidth=1, edgecolor='m', facecolor='none'))
        plt.xlim(-100, 100)
        plt.ylim(-100, 100)
        plt.gca().set_xlabel("x[m]")
        plt.gca().set_ylabel("y[m]")
        plt.tight_layout()
        plt.title("db sampler v1 (min pts all classes 5)")
        plt.gca().set_aspect('equal', adjustable='box')
        #DB sampler v2
        plt.subplot(1,2,2)
        plt.hist2d(location_info_per_class_v2[cl]["x"], location_info_per_class_v2[cl]["y"], bins=50, range=[[-100, 100], [-100, 100]])
        plt.gca().add_patch(patches.Rectangle((-50, -50), 100, 100, linewidth=1, edgecolor='c', facecolor='none'))
        plt.gca().add_patch(patches.Circle((0, 0), radial_dists[cl], linewidth=1, edgecolor='m', facecolor='none'))
        plt.xlim(-100, 100)
        plt.ylim(-100, 100)
        plt.gca().set_xlabel("x[m]")
        plt.gca().set_ylabel("y[m]")
        plt.tight_layout()
        plt.title("db sampler v2")
        # fig, (ax1,ax2) = plt.subplots(1,2, sharex=True, sharey=True)
        # hh1 = ax1.hist2d(location_info_per_class_v1[cl]["x"], location_info_per_class_v1[cl]["y"], bins=50, range=[[-100, 100], [-100, 100]])
        # print(cl, hh1[3])
        # plt.colorbar(hh1[3], ax = ax1)
        # ax1.add_patch(patches.Rectangle((-50, -50), 100, 100, linewidth=1, edgecolor='c', facecolor='none'))
        # ax1.add_patch(patches.Circle((0, 0), radial_dists[cl], linewidth=1, edgecolor='m', facecolor='none'))
        # plt.xlim(-100, 100)
        # plt.ylim(-100, 100)
        # plt.gca().set_xlabel("x[m]")
        # plt.gca().set_ylabel("y[m]")
        # plt.tight_layout()
        # plt.title("db sampler v1 (min pts all classes 5)")
        # plt.gca().set_aspect('equal', adjustable='box')
        # #DB sampler v2
        # plt.subplot(1,2,2)
        # hh2 = plt.hist2d(location_info_per_class_v2[cl]["x"], location_info_per_class_v2[cl]["y"], bins=50, range=[[-100, 100], [-100, 100]])
        # print("---------------------")
        # print(cl, hh2[3])
        # print("************************")
        # plt.colorbar(hh2[3])
        # plt.gca().add_patch(patches.Rectangle((-50, -50), 100, 100, linewidth=1, edgecolor='c', facecolor='none'))
        # plt.gca().add_patch(patches.Circle((0, 0), radial_dists[cl], linewidth=1, edgecolor='m', facecolor='none'))
        # plt.xlim(-100, 100)
        # plt.ylim(-100, 100)
        # plt.gca().set_xlabel("x[m]")
        # plt.gca().set_ylabel("y[m]")
        # plt.tight_layout()
        # plt.title("db sampler v2")
        
        # set size and save
        plt.gcf().set_size_inches(20,10)
        plt.gca().set_aspect('equal', adjustable='box')
        if m1mult:
            plt.savefig("./plots/hist2d1e2/%s_DSR%.2f_DSS%.2f-%.2f_flipxy_classSpecificRates.png" % (cl, DSR[cl], DSS[cl][0], DSS[cl][1]))
        else:
            plt.savefig("./plots/hist2d/%s_DSR%.2f_DSS%.2f-%.2f.png" % (cl, DSR[cl], DSS[cl][0], DSS[cl][1]))
        plt.clf()



prep_v1 = {'filter_by_difficulty': [-1], 'filter_by_min_points': {'car': 5, 'truck': 5, 'bus': 5, 'trailer': 5, 'construction_vehicle': 5, 'traffic_cone': 5, 'barrier': 5, 'motorcycle': 5, 'bicycle': 5, 'pedestrian': 5}}
prep_v2 = {'filter_by_difficulty': [-1], 'filter_by_min_points': {'car': 100, 'truck': 200, 'bus': 200, 'trailer': 400, 'construction_vehicle': 100, 'traffic_cone': 5, 'barrier': 15, 'motorcycle': 15, 'bicycle': 15, 'pedestrian': 15}}
sample_grps = {'car': 2, 'truck': 2, 'construction_vehicle': 2, 'bus': 2, 'trailer': 2, 'barrier': 2, 'motorcycle': 2, 'bicycle': 2, 'pedestrian': 2, 'traffic_cone': 2}
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
DSS = {c: [1.5, 1.5] for c in sample_grps.keys()} # set all class DSS to 1 by default

# Set DSR and DSS for specific classes
DSR["car"] = .9
DSS["car"] = [1.7, 2.2]
DSR["pedestrian"] = .75
DSS["pedestrian"] = [1.5, 2.0]
DSR["bicycle"] = .75
DSS["bicycle"] = [1.4, 1.7]
DSR["construction_vehicle"] = .5
DSS["construction_vehicle"] = [1.3, 1.8]
m1mult = True

num_sample_iters = 100

dbs_v1 = DataBaseSampler("./data/nuscenes/nuscenes_dbinfos_train.pkl", "./data/nuscenes/", 1, prep_v1, sample_grps, sample_grps.keys(), points_loader=dict(type='LoadPointsFromFile', load_dim=5, use_dim=[0,1,2,3], coord_type='LIDAR'))
dbs_v2 = DataBaseSampler_v2("./data/nuscenes/nuscenes_dbinfos_train.pkl", "./data/nuscenes/", 1, prep_v2, sample_grps, sample_grps.keys(), points_loader=dict(type='LoadPointsFromFile', load_dim=5, use_dim=[0,1,2,3], coord_type='LIDAR'), ds_rate=DSR, ds_scale=DSS, ds_flip_xy=m1mult)

# gt_bboxes: x,y,z,w,l,h,theta

class_count_v1 = []
class_count_v2 = []

location_info_per_class_v1 = {c: {"range": [], "x": [], "y": [], "z": [], "num_points": []} for c in sample_grps.keys()}
location_info_per_class_v2 = {c: {"range": [], "x": [], "y": [], "z": [], "num_points": []} for c in sample_grps.keys()}

class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

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

    for c, gtbb in zip(sampled["gt_labels_3d"], sampled["gt_bboxes_3d"]):
        class_count_v2.append(c)
        location_info_per_class_v2[class_names[c]]["range"].append(np.sqrt(gtbb[0]**2 + gtbb[1]**2 + gtbb[2]**2))
        location_info_per_class_v2[class_names[c]]["x"].append(gtbb[0])
        location_info_per_class_v2[class_names[c]]["y"].append(gtbb[1])
        location_info_per_class_v2[class_names[c]]["z"].append(gtbb[2])
        location_info_per_class_v2[class_names[c]]["num_points"].append(len(sampled["points"]))


# # plot_class_count_bar(class_names, class_count, num_sample_iters)

scatter_object_coords(location_info_per_class_v1, location_info_per_class_v2, m1mult=m1mult) # m1mult same thing as ds_flip_xy

plot_range_distribution(location_info_per_class_v1, location_info_per_class_v2, m1mult=m1mult) # m1mult same thing as ds_flip_xy

hist2d_object_coords(location_info_per_class_v1, location_info_per_class_v2, m1mult=m1mult) # m1mult same thing as ds_flip_xy
