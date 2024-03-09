
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
import argparse
import datetime as dt
import random
from model import MADG
from utils import load_data, time_diff, generate_snapshots
import time
from Graph import AnomalyMotifSnapShot, AnomalyCircleSnapShot

def set_seed_everywhere(seed, cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

def loss_func(adj, A_hat):
    # structure reconstruction loss
    diff_structure = torch.pow(A_hat - adj, 2)
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
    return structure_reconstruction_errors


def train(args):
    # data_list = ['uci', 'dnc', 'alpha', 'otc']
    start_time = time.time()
    set_seed_everywhere(666, cuda=True)
    data = args.dataset
    p = args.p
    start_t = dt.datetime.now()
    print('Start data building at: {}-{} {}:{}:{}'.format(start_t.month,
                                                       start_t.day,
                                                       start_t.hour,
                                                       start_t.minute,
                                                        start_t.second))
    edges,times,n = load_data(data)
    snap_num = 10
    snapshots = generate_snapshots(edges, snap_num)
    if args.motif == 'triangle':     
        SnapShots = [AnomalyMotifSnapShot(snapshots[i],p=p) for i in range(len(snapshots))]
    elif args.motif == 'circle':
        SnapShots = [AnomalyCircleSnapShot(snapshots[i],p=p) for i in range(len(snapshots))]

    e_t1 = dt.datetime.now()
    h, m, s = time_diff(e_t1, start_t)
    print('dataset built used: {:02d}h {:02d}m {:02}s'.format(h, m, s))

    snap_len = len(SnapShots)  
    model = MADG(nodes_num=n, snap_len=snap_len,feat_size=args.hidden_dim,hidden_size=args.hidden_dim, num_heads=args.num_heads,dropout=args.dropout,n_layers = args.n_layers, device=args.device)
    
    if args.device == 'cuda':
        device = torch.device(args.device)
        for snap in SnapShots:
            snap.norm_adj = snap.norm_adj.to(device)
            snap.adj = snap.adj.to(device)
            snap.label_adj = snap.label_adj.to(device)
            snap.motif_adj = snap.motif_adj.to(device)
            snap.motif_norm_adj = snap.motif_norm_adj.to(device)
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    scores = []
    labels = []
    for epoch in range(1,args.epoch+1):
        model.train()
        optimizer.zero_grad()
        A_hat = model(SnapShots)

        loss = torch.cat([loss_func(snap.motif_adj, A_hat[i]) for i,snap in enumerate(SnapShots)])
        l = torch.mean(loss)
        l.backward()
        nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=4)
        optimizer.step()        
        print("Epoch:", '%04d' % (epoch), "train_loss=", "{}".format(l.item()))

        if epoch%10 == 0 or epoch == args.epoch:
            model.eval()
            A_hat = model(SnapShots)
            result = []
            prek_result = []
            reck_result = []
            for i,snap in enumerate(SnapShots):
                
                if snap.motif_num == 0:
                    continue

                loss = loss_func(snap.motif_adj, A_hat[i])
                score = loss.detach().cpu().numpy()
                motif_score = score[snap.motif_id_list]
     
                scores.append(motif_score)
                labels.append(snap.motif_labels)

                auc = roc_auc_score(snap.motif_labels, motif_score)
                auc = round(auc,4)
                result.append(auc)
                print("Epoch:", '%04d' % (epoch), " Snap:",'%d' % (i+1),'Auc', auc)
                idx = motif_score.argsort()
                pred_label = snap.motif_labels[idx]
                topk = [50]
                for k in topk:
                    predk = pred_label[-k:]
                    precisionk = sum(predk)/len(predk)
                    recallk = sum(predk)/sum(pred_label)
                    # print('precision@{}:{:.4f},recall@{}:{:.4f}'.format(k,precisionk,k,recallk))
                    prek_result.append(round(precisionk,4))
                    reck_result.append(round(recallk,4))
            print("p=",p,result)
            print("Pre@k:", prek_result)
            print("Rec@k:", reck_result)
            result = np.array(result)
            mean_auc = round(np.mean(result),4)
            print("mean auc = ", mean_auc)
    print("time cost: {}".format(time.time() - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='uci', help='dataset name: uci/dnc/alpha/otc')
    parser.add_argument('--hidden_dim', type=int, default=64, help='dimension of hidden embedding (default: 64)')
    parser.add_argument('--epoch', type=int, default=10, help='Training epoch')
    parser.add_argument('--num_heads', type=int, default=2, help='Number of the heads of attention module')
    parser.add_argument('--n_layers', type=int, default=2, help='layers for motif-augmented gcn')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--p', type=float, default=0.1, help='anomalous rate')
    parser.add_argument('--motif', default='triangle', help='triangle/circle')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')
    args = parser.parse_args()
    train(args)

