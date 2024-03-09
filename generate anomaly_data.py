import numpy as np
import torch
import torch.nn as nn
from Graph import AnomalyMotifySnapShot
import pickle
import os
from utils import load_data, generate_snapshots

def ensure_dir_exit(path):
    if not os.path.exists(path):
        os.mkdir(path)

p = 0.05
root_path = 'anomaly{}_data'.format(p)
ensure_dir_exit(root_path)
data_list = ['uci', 'digg', 'dnc', 'alpha', 'otc']
for data in data_list:
    print("generate {} data....".format(data))
    edges,times,n = load_data(data)
    snaps = generate_snapshots(edges, 10)
    SnapShots = [AnomalyMotifySnapShot(snaps[i],p=p) for i in range(len(snaps))]
    if data == 'digg':
        digg_path = os.path.join(root_path, data)
        ensure_dir_exit(digg_path)
        for i in range(10):
            with open('{}/digg_{}.pkl'.format(digg_path, i), 'wb') as file:
                pickle.dump(SnapShots[i], file)
    else:
        with open('{}/{}.pkl'.format(root_path, data), 'wb') as file:
            pickle.dump(SnapShots, file)

