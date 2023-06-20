import numpy as np
import matplotlib.pyplot as plt

# set width of bar

import json

classList = ['car','truck','bus','trailer','construction_vehicle','pedestrian','motorcycle','bicycle','traffic_cone','barrier']
mod2name = {'no_gt_sampler': 'No GTDS', 
            'baseline_no_fade': 'Baseline GTDS', 
            'baseline_with_fade': 'Baseline GTDS + fade',  
            'v3_random_ds': 'RFS Random', 
            'v3_random_ds_fade': 'RFS Random + fade',
            'v3_fps_ds': 'RFS FPS',
            'v3_fps_ds_fade': 'RFS FPS + fade',
            'v4_random_ds': 'RRSv1', 
            'v4_random_ds_fade': 'RRSv1 + fade', 
            'v4_longrange_bias': 'RRSv2', 
            'v4_longrange_bias_with_fade': 'RRSv2 + fade', 
            'v5_no_fade': 'HRFiSv1', 
            'v5_with_fade': 'HRFiSv1 + fade', 
            'v6_no_fade': 'HRFiSv2', 
            'v6_with_fade': 'HRFiSv2 + fade',
            'baseline_no_fade_2x': 'Baseline GTDS 2x',
            'baseline_with_fade_2x': 'Baseline GTDS 2x + fade'}

modelListGT = ['no_gt_sampler', 
                'baseline_no_fade', 
                'baseline_with_fade']

modelListTune = ['no_gt_sampler', 
                'baseline_no_fade',
                'baseline_no_fade_2x', 
                'baseline_with_fade',
                'baseline_with_fade_2x']

modelListFpsRandom = [  'no_gt_sampler', 
                        'baseline_no_fade', 
                        'baseline_with_fade', 
                        'v3_fps_ds',
                        'v3_fps_ds_fade', 
                        'v3_random_ds', 
                        'v3_random_ds_fade'
                        ]

modelListAll = ['no_gt_sampler', 
                'baseline_no_fade', 
                'baseline_with_fade', 
                'v3_random_ds', 
                'v3_random_ds_fade',
                'v3_fps_ds',
                'v3_fps_ds_fade', 
                'v4_random_ds', 
                'v4_random_ds_fade', 
                'v4_longrange_bias', 
                'v4_longrange_bias_with_fade', 
                'v5_no_fade', 
                'v5_with_fade', 
                'v6_no_fade', 
                'v6_with_fade'] 

modelListPP = ['no_gt_sampler', 
                'baseline_no_fade', 
                'baseline_with_fade', 
                'v3_random_ds', 
                'v3_random_ds_fade', 
                'v6_no_fade', 
                'v6_with_fade']

modelListFade = [   'no_gt_sampler',  
                    'baseline_with_fade',
                    'v3_random_ds_fade',
                    'v3_fps_ds_fade',  
                    'v4_random_ds_fade', 
                    'v4_longrange_bias_with_fade', 
                    'v5_with_fade',  
                    'v6_with_fade']

modelListNoFade = [ 'no_gt_sampler',  
                    'baseline_no_fade',
                    'v3_random_ds',
                    'v3_fps_ds',  
                    'v4_random_ds', 
                    'v4_longrange_bias', 
                    'v5_no_fade',  
                    'v6_no_fade']


pp = ''
# pp = 'pp/'

modelList = modelListTune
# rang = 'short'
# rang = 'medium'
rang = 'all'
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
val = 'mean_ap'
# val = 'nd_score'
# val = 'mean_dist_aps'
# val = 'num_gt_objects'
barWidth = 0.05
lim_y = False
lim_weights = [0.8, 1.1]


datapaths = ['results/filtered/' + pp + vers + '/' + vers + '_statistics.json' for vers in modelList]

data = {}
for i in range(len(modelList)):
    f = open(datapaths[i])
    data[modelList[i]] = json.load(f)

print(data['no_gt_sampler'].keys())
print(data['no_gt_sampler']['short'].keys())
print(data['no_gt_sampler']['short']['metrics_summary'].keys())

print('mean dist AP')
print()
for model in modelList:
    d = [data[model][rang]['metrics_summary']['mean_dist_aps'][k] for k in classList]
    print('%s & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f \\\\' % (mod2name[model] ,d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9]))

print('mAP ranges')
print()
print(' & short & medium & long & all')
for model in modelList:
    d = [data[model][k]['metrics_summary']['mean_ap'] for k in data[model].keys()]
    print('%s & %.3f & %.3f & %.3f & %.3f \\\\' % (mod2name[model] ,d[0], d[1], d[2], d[3]))

fig, ax = plt.subplots(figsize =(12, 8))

vals = np.arange(len(modelList)).tolist()
if val == 'nd_score' or val == 'mean_ap':
    for i in range(len(modelList)):
        valueList = [data[modelList[i]][d]['metrics_summary'][val] for d in data[modelList[i]].keys()]
        tmp = max(valueList)
        # if tmp > maxVal:
        #     maxVal = tmp
        vals[i] = valueList
elif val == 'num_gt_objects':
    for i in range(len(modelList)):
        valueList = [data[modelList[i]][d][val][clas] for d in data[modelList[i]].keys()]
        print(valueList)
        # print(valueList)
        # tmp = max(valueList)
        # if tmp > maxVal:
        #     maxVal = tmp
        vals[i] = valueList

else:
    for i in range(len(modelList)):
        valueList = [data[modelList[i]][d]['metrics_summary'][val][clas] for d in data[modelList[i]].keys()]
        tmp = max(valueList)
        
        # if tmp > maxVal:
        #     maxVal = tmp
        vals[i] = valueList

# Set position of bar on X axis
br = np.arange(len(modelList)).tolist()
for i in range(len(modelList)):
    br[i] = np.arange(len(vals[i])) + i*barWidth

# Make the plot
for i in range(len(modelList)):
    plt.bar(br[i], vals[i], width = barWidth, edgecolor ='grey', label =mod2name[modelList[i]])    

# Adding Xticks
plt.xlabel('range', fontweight ='bold', fontsize = 15)
# plt.axhline(y=maxVal,linewidth=1, color='k')
# if lim_y:
#     ax.set_ylim([lim_weights[0]*maxVal,lim_weights[1]*maxVal])
plt.ylabel(val, fontweight ='bold', fontsize = 15)
plt.xticks([r + 3*barWidth for r in range(len(vals[1]))],
		data[modelList[0]].keys())
plt.title(clas)
plt.legend()
plt.show()