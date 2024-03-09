import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import coo_matrix
from Graph import MotifSnapShot,AnomalyMotifSnapShot,SnapShot,CircleMotifSnapShot
from utils import load_data, generate_snapshots
import time

data = 'dnc'
root_path = '/home/yzr/Graph/DynamicAGD/MotifyAD/motif_list/'+data+'/'
#uci, digg, dnc, alpha, otc, ast
start = time.time()
edges,times,n = load_data(data)
size = 20
snaps = generate_snapshots(edges, 10)

SnapShots = [CircleMotifSnapShot(snaps[i],size=size) for i in range(len(snaps))]
otc_circle = []
for sp in SnapShots:
    otc_circle.append(sp.motif_list)
np.save(root_path+data+"_"+str(size)+".npy", otc_circle)
end = time.time()
print("time cost: {:.4f} mins".format((end-start)/60))