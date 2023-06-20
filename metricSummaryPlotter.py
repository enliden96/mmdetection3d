import numpy as np
import matplotlib.pyplot as plt

# set width of bar

import json

classList = ['car','truck','bus','trailer','construction_vehicle','pedestrian','motorcycle','bicycle','traffic_cone','barrier']

modelListGT = ['no_gt_sampler', 
                'baseline_no_fade', 
                'baseline_with_fade']

modelListAll = ['no_gt_sampler', 
                'baseline_no_fade', 
                'baseline_with_fade', 
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

modelListFade = [   'no_gt_sampler',  
                    'baseline_with_fade',
                    'v3_fps_ds_fade',  
                    'v3_random_ds_fade',
                    'v4_random_ds_fade', 
                    'v4_longrange_bias_with_fade', 
                    'v5_with_fade',  
                    'v6_with_fade']

modelListNoFade = [ 'no_gt_sampler',  
                    'baseline_no_fade',
                    'v3_fps_ds',  
                    'v3_random_ds',
                    'v4_random_ds', 
                    'v4_longrange_bias', 
                    'v5_no_fade',  
                    'v6_no_fade']

modelList = modelListGT
clas = 'car'
# clas = 'truck'
# clas = 'bus'
# clas = 'trailer'
# clas = 'construction_vehicle'
# clas = 'pedestrian'
# clas = 'motorcycle'
# clas = 'bicycle'
# clas = 'traffic_cone'
# clas = 'barrier'
# dist = '0.5'
# metric = 'label_aps'
# metric = 'mean_dist_aps'
metric = 'nd_score'
# metric = 'mean_ap'
barWidth = 0.1
lim_y = False
lim_weights = [0.6, 1.1]
pp = 'pp/'

modelversion = 'v5_with_fade'
datapath = 'results/filtered/' + pp + modelversion + '/' + modelversion + '_metrics_summary_filtered.json'

datapaths = ['results/filtered/' + pp + vers + '/' + vers + '_metrics_summary_filtered.json' for vers in modelList]

data = {}
for i in range(len(modelList)):
    f = open(datapaths[i])
    data[modelList[i]] = json.load(f)
print(data['no_gt_sampler'].keys())
print(data['no_gt_sampler']['mean_dist_aps'].keys())
print(data['no_gt_sampler']['mean_dist_aps']['car'])

# for model in modelList:
#     d = [data[model]['label_aps'][k][]]


# print(data['no_gt_sampler'][rang][clas].keys())
# print(data['no_gt_sampler'][rang][clas][dist].keys())


fig, ax = plt.subplots(figsize =(16, 8))
# maxVal = 0
# vals = np.arange(len(modelList)).tolist()
# for i in range(len(modelList)):
#     valueList = [data[modelList[i]][rang][clas][d][val] for d in data[modelList[i]][rang][clas].keys()]
#     tmp = max(valueList)
#     if tmp > maxVal:
#         maxVal = tmp
#     vals[i] = valueList

# Set position of bar on X axis
# br = np.arange(len(modelList)).tolist()
# for i in range(len(modelList)):
#     br[i] = np.arange(len(vals[i])) + i*barWidth

# Make the plot
if metric == 'label_aps':
    vals = [data[d][metric][clas][dist] for d in modelList]
elif metric == 'mean_dist_aps':
    vals = [data[d][metric][clas] for d in modelList]
    if modelList == modelListGT:
        print('%s & %.3f \t & %.3f \t & %.3f' % (clas,vals[0], vals[1], vals[2]))
else:
    vals = [data[d][metric] for d in modelList]
    if modelList == modelListGT:
        print('%s & %.3f \t & %.3f \t & %.3f' % (clas,vals[0], vals[1], vals[2]))
plt.bar(np.arange(len(vals)), vals, width = barWidth, edgecolor ='grey', label =modelList[i])    

maxVal = max(vals)
# Adding Xticks
plt.xlabel('Model', fontweight ='bold', fontsize = 15)
plt.axhline(y=maxVal,linewidth=1, color='k')
if lim_y:
    ax.set_ylim([lim_weights[0]*maxVal,lim_weights[1]*maxVal])
plt.ylabel(metric, fontweight ='bold', fontsize = 15)
plt.xticks([r for r in range(len(vals))],
		modelList)
ax.set_xticklabels(modelList,rotation='vertical')

plt.legend()
# plt.show()
