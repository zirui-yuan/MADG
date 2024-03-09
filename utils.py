import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp

def load_data1(file_path):
    # load the data and sort by the time
    data = np.loadtxt(file_path, dtype=[('src', int), ('dst', int), ('time', int)], comments='%', usecols=(0, 1, 3))
    data.sort(order = 'time')
    src_node = data['src']
    dst_node = data['dst']
    times = data['time']
    edges = np.vstack((src_node, dst_node)).T
    orgin_node_ids = np.unique(edges) # re tag the node_id
    n = len(orgin_node_ids)
    node2id = {orgin_node_ids[i]:i for i in range(n)}
    src_node = [node2id[node] for node in src_node]
    dst_node = [node2id[node] for node in dst_node]
    edges = np.vstack((src_node, dst_node)).T
    return edges, times, n

def load_dnc_data(file_path):
    # load the data and sort by the time
    with open(file_path) as f:
        data = []
        for line in f.readlines():
            data.append(tuple(line.strip('\n').strip('\ufeff').split(',')))
    data = np.array(data,dtype = [('src', int), ('dst', int), ('time', int)])
    data.sort(order = 'time')
    src_node = data['src']
    dst_node = data['dst']
    times = data['time']
    edges = np.vstack((src_node, dst_node)).T
    orgin_node_ids = np.unique(edges)
    n = len(orgin_node_ids)
    node2id = {orgin_node_ids[i]:i for i in range(n)}
    src_node = [node2id[node] for node in src_node]
    dst_node = [node2id[node] for node in dst_node]
    edges = np.vstack((src_node, dst_node)).T
    return edges, times, n

def load_csv_data(file_path):
    data = pd.read_csv(file_path, header=None, usecols=[0,1,3])
    data = data.to_numpy()
    data = data[np.argsort(data[:,2])] #sort on time
    src_node = data[:,0]
    dst_node = data[:,1]
    times = data[:,2]
    edges = np.vstack((src_node, dst_node)).T
    orgin_node_ids = np.unique(edges)
    n = len(orgin_node_ids)
    node2id = {orgin_node_ids[i]:i for i in range(n)}
    src_node = [node2id[node] for node in src_node]
    dst_node = [node2id[node] for node in dst_node]
    edges = np.vstack((src_node, dst_node)).T
    return edges, times, n

def load_data(file_name):
    if file_name == 'uci':
        return load_data1('raw_data/out.opsahl-ucsocial')
    elif file_name == 'digg':
        return load_data1('raw_data/out.munmun_digg_reply')
    elif file_name == 'dnc':
        return load_dnc_data('raw_data/email-dnc.edges')
    elif file_name == 'alpha':
        return load_csv_data('raw_data/soc-sign-bitcoinalpha.csv')
    elif file_name == 'otc':
        return load_csv_data('raw_data/soc-sign-bitcoinotc.csv')
    elif file_name == 'ast': 
        return load_data1('raw_data/tech-as-topology.edges')  
    else:
        raise NotImplementedError('So far, only support six dataset: uci, digg, dnc, alpha, otc, ast') 




def get_snapshots(edges, times, n):
    # split the edges to n snapshots
    snapshot_list = []
    total_time_scope = times[-1] - times[0]
    snapshot_scope = int(total_time_scope/n)
    time_points = [times[0] + i*snapshot_scope for i in range(1, n)]
    time_points.append(times[-1])

    snapshot_list.append(edges[np.where(times <= time_points[0])])
    for i in range(len(time_points)-1):
        scope = np.where((times > time_points[i]) & (times <= time_points[i+1]))
        snapshot_list.append(edges[scope])
    
    return snapshot_list

def split_snapshots_by_edge_num(edges, n):
    # split the edges to n snapshots
    snapshot_list = []
    edges_num = len(edges)
    current = 0
    while (edges_num - current) >= n:
        snapshot_list.append(edges[current:current + n])
        current += n
    snapshot_list.append(edges[current:])
    
    return snapshot_list

def generate_snapshots(edges, n):
    snap_list = []
    edges_num = len(edges)
    snap_edges_num = int(edges_num/n)
    for i in range(n-1):
        snap_list.append(edges[i*snap_edges_num:(i+1)*snap_edges_num])
    snap_list.append(edges[(i+1)*snap_edges_num:])
    return snap_list

def split_nodes(n, batch_size):
    nodes_list = list(range(n))
    batch_list = []
    current = 0
    while (n - current) >= batch_size:
        batch_list.append(nodes_list[current:current + batch_size])
        current += batch_size
    batch_list.append(nodes_list[current:])
    return batch_list





def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_adj(adj, is_sparse=False):
    """Preprocessing of adjacency matrix for simple pygGCN model and conversion to
    tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    if is_sparse:
        adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized)
        return adj_normalized
    else:
        return torch.from_numpy(adj_normalized.A).float()

def adj2tensor(adj, is_sparse=False):
    """Preprocessing of adjacency matrix for simple gat model and conversion to
    tuple representation."""
    #adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    if is_sparse:
        adj_normalized = sparse_mx_to_torch_sparse_tensor(adj)
        return adj_normalized
    else:
        return torch.from_numpy(adj).float()

def time_diff(t_end, t_start):
    """
    计算时间差。t_end, t_start are datetime format, so use deltatime
    Parameters
    ----------
    t_end
    t_start

    Returns
    -------
    """
    diff_sec = (t_end - t_start).seconds
    diff_min, rest_sec = divmod(diff_sec, 60)
    diff_hrs, rest_min = divmod(diff_min, 60)
    return (diff_hrs, rest_min, rest_sec)



        
    


