import numpy as np
import json
import matplotlib.pyplot as plt

datapath = 'results/HPC/baseline_with_fade/pts_bbox/metrics_details.json'
datapath2 = 'results/HPC/baseline_no_fade/pts_bbox/metrics_details.json'
classnRange = 'bicycle:0.5'
# classnRange = 'traffic_cone:0.5'
f = open(datapath)
f2 = open(datapath2)
data = json.load(f)
data2 = json.load(f2)
print(data.keys())
print(data[classnRange].keys())
precision = data[classnRange]['precision']
recall = data[classnRange]['recall']
precision2 = data2[classnRange]['precision']
recall2 = data2[classnRange]['recall']

plt.plot(recall,precision)
plt.plot(recall2,precision2)
plt.plot([0,1],[0.1,0.1],'r--')
plt.vlines(0.1,0,1,'r','dashed')
plt.axis([0,1,0,1])
plt.legend(['1','2'])

plt.axis('equal')

plt.ylabel('Precision',fontsize=20)
plt.xlabel('Recall',fontsize=20)
plt.tight_layout()
plt.show()


## annat värde



print('########################################################################################')
datapath = 'results/filtered/baseline_with_fade/baseline_with_fade_statistics.json'
f = open(datapath)
data = json.load(f)
print(data.keys())
print(data['all'].keys())
print(data['all']['car'].keys())
print(data['all']['car']['0.5'].keys())

data = data['all']['car']['0.5']

print(data.keys())
tp_list = np.array(data['tp_list'])
fp_list = np.array(data['fp_list'])
conf_list = np.array(data['conf_list'])
gt = data['npos']
print('type: ', type(gt))
arange = np.linspace(0,1,11)
# print(arange)
# a = np.array([0,1,2,3,4,5,6,7,8,9,10])
# print('a.asbool: ', a.astype(bool))
# print(np.sum(a))
print('lentplist: ', len(tp_list))
print('lenconflist: ', len(conf_list))
for con in arange:
    picker = np.array(conf_list[conf_list > con].astype(bool))
    print('lenpicker: ', len(picker))
    # print(picker[0:11])
    # print('sel: ', a[picker[0:11]])
    tp = np.sum(tp_list[picker])
    print('lentp: ', tp)
    fp = np.sum(fp_list[picker])

    print(con)

# coord_range = np.array([100,100,10]).reshape(3,1)
# print(coord_range)
# print('####################################################################################')
# pts = np.random.rand(3,20)
# print(pts)
# print('####################################################################################')
# print(coord_range*pts)
# print('####################################################################################')
# print(coord_range*pts - coord_range/2)
# print('####################################################################################')