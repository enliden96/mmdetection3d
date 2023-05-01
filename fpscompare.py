from dgl.geometry import farthest_point_sampler
import torch
import pointnet2_ops
import time
from tqdm import tqdm
from torch_cluster import fps
import math
import matplotlib.pyplot as plt

def calc_std(arr: list):
    mean = sum(arr)/len(arr)
    return math.sqrt(sum([(x - mean)**2 for x in arr]) / len(arr))

npointslist = [100, 1000, 2000, 5000, 10000]
results = {}
methods = ["Random", "DGL", "PointNet2", "TorchCluster"]
for m in methods:
    results[m] = {}
# test_tensor_3d = torch.rand((1000, npoints, 3))

# print(test_tensor_3d[0])
for npoints in npointslist:
    randomtimes = []
    pn2times =  []
    dgltimes =  []
    tctimes =   []

    warmuptensor = torch.rand((1, 1, 3))
    warmuptensor = warmuptensor.to("cuda")

    ptskept = npoints//2**3
    rat = (npoints//2**3)/npoints
    batchsize = 10

    for i in tqdm(range(100)):
        test_tensor_3d = torch.rand((batchsize, npoints, 3))
        
        test_tensor_3d = test_tensor_3d.to("cpu")
        torch.cuda.synchronize()
        randomstart = time.time()
        for t in test_tensor_3d:
            ind = torch.randperm(t.shape[0])[:ptskept]
            t = t[ind]
        randomend = time.time()
        
        
        test_tensor_3d = test_tensor_3d.to("cpu")
        dglstart = time.time()
        for t in test_tensor_3d:
            ind = farthest_point_sampler(t.reshape(1,-1,3), ptskept)
            t = t[ind]

        dglend = time.time()
        
        test_tensor_3d = test_tensor_3d.to("cpu")
        
        torch.cuda.synchronize()
        pn2start = time.time()
        # test_tensor_3d = test_tensor_3d.to("cuda")
        for t in test_tensor_3d:
            t = t.to("cuda")
            # print(pointnet2_ops.pointnet2_utils.furthest_point_sample(t.reshape(1,-1,3), 1000//2**3))
            ind = pointnet2_ops.pointnet2_utils.furthest_point_sample(t.reshape(1,-1,3), ptskept)
            # print(ind)
            t = t[ind.long()]
            # farthest_point_sampler(t.reshape(1,-1,3), 1000//2**3)
        torch.cuda.synchronize()
        pn2end = time.time()

        # print("pointnet2 time:", pn2end-pn2start)

        # print(test_tensor_3d.device)
        # test_tensor_3d = test_tensor_3d.to("cpu")
        
        test_tensor_3d = test_tensor_3d.to("cpu")
        torch.cuda.synchronize()
        tcstart = time.time()
        # test_tensor_3d = test_tensor_3d.to("cuda")
        for t in test_tensor_3d:
            t = t.to("cuda")
            ind = fps(t, ratio=rat, random_start=False)
            t = t[ind]
        torch.cuda.synchronize()
        tcend = time.time()    

        randomtimes.append(randomend-randomstart)
        dgltimes.append(dglend-dglstart)
        pn2times.append(pn2end-pn2start)
        tctimes.append(tcend-tcstart)
        
        # test_tensor_3d = test_tensor_3d.to("cpu")




    mean_random = sum(randomtimes)/len(randomtimes)/batchsize
    mean_dgl = sum(dgltimes)/len(dgltimes)/batchsize
    mean_pn2 = sum(pn2times)/len(pn2times)/batchsize
    mean_tc = sum(tctimes)/len(tctimes)/batchsize

    std_random = calc_std(randomtimes)/batchsize
    std_dgl = calc_std(dgltimes)/batchsize
    std_pn2 = calc_std(pn2times)/batchsize
    std_tc = calc_std(tctimes)/batchsize


    print("npoints:", npoints)
    print("random avg time:", mean_random)
    print("dgl avg time:", mean_dgl)
    print("pn2 avg time:", mean_pn2)
    print("tc avg time:", mean_tc)

    # print("pn2times", pn2times)
    # # print(std_pn2)

    # print("tctimes", tctimes)
    # print(std_tc)

    results["Random"][npoints] = [mean_random, std_random]
    results["DGL"][npoints] = [mean_dgl, std_dgl]
    results["PointNet2"][npoints] = [mean_pn2, std_pn2]
    results["TorchCluster"][npoints] = [mean_tc, std_tc]






plt.figure(figsize=(10,10))

for m in methods:
    plt.plot(npointslist, [results[m][100][0], results[m][1000][0], results[m][2000][0], results[m][5000][0], results[m][10000][0]])
    plt.errorbar(npointslist, [results[m][100][0], results[m][1000][0], results[m][2000][0], results[m][5000][0], results[m][10000][0]], yerr=[results[m][100][1], results[m][1000][1], results[m][2000][1], results[m][5000][1], results[m][10000][1]], capsize=5, label=m)
plt.xticks(npointslist, ["100", "1000", "2000", "5000", "10000"])
plt.ylabel("Time per sample (s)")
plt.xlabel("Number of points")
plt.legend()

plt.figure(figsize=(10,10))
i = 0
for m in methods:
    plt.scatter(i, results[m][100][0], s=50)
    plt.errorbar(i, results[m][100][0], yerr=[results[m][100][1]], capsize=5, label=m)
    i += 1
plt.title("Number of points: 100")
plt.legend()
plt.xticks([0,1,2,3], ["Random", "DGL", "PointNet2", "PyTorch Cluster"])

plt.figure(figsize=(10,10))
i=0
for m in methods:
    plt.scatter(i, results[m][1000][0], s=50)
    plt.errorbar(i, results[m][1000][0], yerr=[results[m][1000][1]], capsize=5, label=m)
    i += 1
plt.title("Number of points: 1000")
plt.legend()
plt.xticks([0,1,2,3], ["Random", "DGL", "PointNet2", "PyTorch Cluster"])

plt.show()


# plt.figure(figsize=(10,10))
# plt.bar([1,2,3,4], [mean_random, mean_dgl, mean_pn2, mean_tc], yerr=[std_random, std_dgl, std_pn2, std_tc], capsize=10)
# # plt.scatter([1,2,3], [mean_dgl, mean_pn2, mean_tc], color='red', facecolor='red', s=100, facecolors='none')
# # plt.errorbar([1,2,3], [mean_dgl, mean_pn2, mean_tc], yerr=[std_dgl, std_pn2, std_tc], fmt='o')
# plt.title("Number of points: " + str(npoints))

# plt.xticks([1,2,3,4], ["Random", "DGL", "PointNet2", "PyTorch Cluster"])
# plt.show()