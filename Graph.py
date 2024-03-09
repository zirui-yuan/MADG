import numpy as np
import torch
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from utils import preprocess_adj, adj2tensor
from tqdm import tqdm
import time
import random

class SnapShot:
    def __init__(self, edges) -> None:
        self.edges = edges
        self.edges_num = len(self.edges)
        self.nodes = np.unique(edges).tolist()
        self.nodes.sort()
        self.nodes_num = len(self.nodes)
        self.node_ids = list(range(self.nodes_num))
        self.id2node = {i:self.nodes[i] for i in range(self.nodes_num)}
        self.node2id = {self.nodes[i]:i for i in range(self.nodes_num)}

        self.adj = self.init_adj() 
        self.norm_adj = preprocess_adj(self.adj, is_sparse=True).to_dense()
        self.adj = adj2tensor(self.adj, is_sparse=True)
        self.label_adj = self.adj.to_dense()
    

    def convert_symmetric(self, X, sparse=True):
        # add symmetric edges
        if sparse:
            X += X.T - sp.diags(X.diagonal())
        else: 
            X += X.T - np.diag(X.diagonal())
        return X

    def init_adj(self):
        values = torch.ones(len(self.edges), dtype=torch.float32)
        indices = self.edges.T
        src_node = indices[0].tolist()
        dst_node = indices[1].tolist()
        src_node = [self.node2id[i] for i in src_node]
        dst_node = [self.node2id[i] for i in dst_node]
        adj = coo_matrix((values, (src_node, dst_node)),
                         shape=(self.nodes_num, self.nodes_num))
        adj = self.convert_symmetric(adj, sparse=True)
        return adj
        

class CircleMotifSnapShot(SnapShot):
    def __init__(self, edges, motif_list=None, size = 4) -> None:
        super().__init__(edges)
        self.size = size
        self.motif_list = []
        if motif_list:
            self.motif_list = motif_list
        else:
            self.search_motif_from_adj()

        self.motif_num = len(self.motif_list)
        if self.motif_num > 500:
            self.motif_list = random.sample(self.motif_list, 500)
            self.motif_num = 500

    def search_motif_from_adj(self):
        start_nodes = random.sample(self.node_ids, len(self.node_ids))
        with tqdm(total=len(start_nodes)) as t:
            for node in start_nodes:
                s = time.time()
                candi_motif = [node]
                self.dfs(candi_motif, self.size-1, t, s)
                
                t.set_postfix(motif_num = len(self.motif_list))
                t.update(1)
                if len(self.motif_list) >= 500:
                    break

        if len(self.motif_list) != 0: 
            self.motif_list = np.unique(np.sort(self.motif_list,axis=1), axis=0).tolist()


    def dfs(self, candi_motif, size, t, s):
        node0 = candi_motif[0]
        node1 = candi_motif[-1]
        max_node = max(candi_motif)
        for node_next in self.adj[node1]._indices()[0]:
            if int(node_next) > max_node:
                candi_motif.append(int(node_next))
                
                if size == 1:
                    if node_next in self.adj[node0]._indices()[0]:
                        self.motif_list.append(candi_motif[:])
                        t.set_postfix(motif_num = len(self.motif_list))
                                              
                else:
                    self.dfs(candi_motif,size-1, t,s)
                candi_motif.pop(-1)
            if len(self.motif_list) >= 500:
                break
            elif time.time()-s>100:
                break

                
        
class MotifSnapShot(SnapShot):
    def __init__(self, edges) -> None:
        super().__init__(edges)
        self.motif_list = self.search_motif_from_adj()
        self.motif_num = len(self.motif_list)
    
    def search_motif_from_adj(self):
        motif_list = []
        for node0 in self.node_ids:
            for node1 in self.adj[node0]._indices()[0]:
                if node1 <= node0:
                    continue
                for node2 in self.adj[node1]._indices()[0]:
                    if node2 <= node0 or node2 <= node1:
                        continue
                    if node2 in self.adj[node0]._indices()[0]:
                        motif_list.append([node0, int(node1), int(node2)])
        return motif_list

class AnomalyCircleSnapShot(CircleMotifSnapShot):
    def __init__(self, edges, motif_list=None, size = 4, p=0.02) -> None:
        super().__init__(edges, motif_list, size)
        
        self.anomaly_motifs = self.generate_anomalys(p)
        self.adj,self.norm_adj, self.new_edges = self.rebuild_anomaly_adj()
        self.label_adj = self.adj.to_dense()
        self.anomaly_nodes = np.unique(np.array(self.anomaly_motifs)).tolist()
        self.anomaly_motifs_num = len(self.anomaly_motifs)

        self.motifs = self.motif_list + self.anomaly_motifs
        self.augmented_nodes_num = self.nodes_num + len(self.motifs)
        self.augmented_nodes_list = list(range(self.augmented_nodes_num))
        self.motif_start_id = self.nodes_num
        self.motif_id_list = self.augmented_nodes_list[self.motif_start_id:]

        self.motif_labels = np.array([0]*self.motif_num +[1]*self.anomaly_motifs_num)
        self.nodes_labels = np.zeros(self.nodes_num, dtype=np.int)
        self.nodes_labels[self.anomaly_nodes] = 1
        self.motif_adj, self.motif_norm_adj = self.build_motif_adj()

        
    def generate_anomalys(self, p):
        anomaly_motifs = []
        anomaly_motifs_num = max(1, int(self.motif_num*p))
        for _ in range(anomaly_motifs_num):
            candi_motif = random.sample(self.node_ids, self.size) #list
            candi_motif.sort()
            while candi_motif in self.motif_list or candi_motif in anomaly_motifs:
                candi_motif = random.sample(self.node_ids, self.size)
                candi_motif.sort()
            
            anomaly_motifs.append(candi_motif)
        return anomaly_motifs   


    def rebuild_anomaly_adj(self):
        indices = self.edges.T
        src_nodes = indices[0].tolist()
        dst_nodes = indices[1].tolist()
        src_nodes = [self.node2id[i] for i in src_nodes]
        dst_nodes = [self.node2id[i] for i in dst_nodes]
        anomaly_motifs_src = []
        anomaly_motifs_dst = []
        for motif in self.anomaly_motifs:
            anomaly_motifs_src += [motif[i] for i in range(self.size)]
            anomaly_motifs_dst += [motif[self.size-i-1] for i in range(self.size)]
        src_nodes += anomaly_motifs_src
        dst_nodes += anomaly_motifs_dst
        edges = np.vstack((src_nodes, dst_nodes)).T
        values = torch.ones(len(src_nodes), dtype=torch.float32)
        adj = coo_matrix((values, (src_nodes, dst_nodes)),
                         shape=(self.nodes_num, self.nodes_num))
        adj = self.convert_symmetric(adj, sparse=True)
        norm_adj = preprocess_adj(adj, is_sparse=True).to_dense()
        adj = adj2tensor(adj, is_sparse=True)
        return adj, norm_adj, edges
        
    
    def build_motif_adj(self):
        indices = self.edges.T
        src_nodes = indices[0].tolist()
        dst_nodes = indices[1].tolist()
        src_nodes = [self.node2id[i] for i in src_nodes]
        dst_nodes = [self.node2id[i] for i in dst_nodes]
        motif_src_nodes = []
        motif_dst_nodes = []
        for motif_id, motif in zip(self.motif_id_list, self.motifs):
            neighbor_set = set()
            for node in motif:
                for neighbor in self.adj[node]._indices()[0]:
                    neighbor_set.add(int(neighbor))
                neighbor_set.add(node)
            motif_dst_nodes+=list(neighbor_set)
            motif_src_nodes+=[motif_id]*len(neighbor_set)
        
        src_nodes += motif_src_nodes
        dst_nodes += motif_dst_nodes
            
        values = torch.ones(len(src_nodes), dtype=torch.float32)
        adj = coo_matrix((values, (src_nodes, dst_nodes)),
                         shape=(self.augmented_nodes_num, self.augmented_nodes_num))
        adj = self.convert_symmetric(adj, sparse=True)
        norm_adj = preprocess_adj(adj, is_sparse=True).to_dense()
        adj = adj2tensor(adj, is_sparse=True).to_dense()
        adj = torch.sqrt(adj/adj.sum(-1,keepdim=True))
        return adj, norm_adj

class AnomalyMotifSnapShot(MotifSnapShot):
    def __init__(self, edges, p=0.02) -> None:
        super().__init__(edges)
        
        self.anomaly_motifs = self.generate_anomalys(p)
        self.adj,self.norm_adj, self.new_edges = self.rebuild_anomaly_adj()
        self.label_adj = self.adj.to_dense()
        self.anomaly_nodes = np.unique(np.array(self.anomaly_motifs)).tolist()
        self.anomaly_motifs_num = len(self.anomaly_motifs)

        self.motifs = self.motif_list + self.anomaly_motifs
        self.augmented_nodes_num = self.nodes_num + len(self.motifs)
        self.augmented_nodes_list = list(range(self.augmented_nodes_num))
        self.motif_start_id = self.nodes_num
        self.motif_id_list = self.augmented_nodes_list[self.motif_start_id:]

        self.motif_labels = np.array([0]*self.motif_num +[1]*self.anomaly_motifs_num)
        self.nodes_labels = np.zeros(self.nodes_num, dtype=np.int)
        self.nodes_labels[self.anomaly_nodes] = 1
        self.motif_adj, self.motif_norm_adj = self.build_motif_adj()

        
    def generate_anomalys(self, p):
        anomaly_motifs = []
        anomaly_motifs_num = max(1, int(self.motif_num*p))
        for _ in range(anomaly_motifs_num):
            candi_motif = random.sample(self.node_ids, 3) #list
            candi_motif.sort()
            while candi_motif in self.motif_list or candi_motif in anomaly_motifs:
                candi_motif = random.sample(self.node_ids, 3)
                candi_motif.sort()
            
            anomaly_motifs.append(candi_motif)
        return anomaly_motifs

    def generate_anomalys2(self, p):
        anomaly_motifs = []
        anomaly_motifs_num = max(1, int(self.motif_num*p))
        select_zone = list(np.unique(self.motif_list))

        for _ in range(anomaly_motifs_num):
            candi_motif = random.sample(select_zone, 3) #list
            candi_motif.sort()
            while candi_motif in self.motif_list or candi_motif in anomaly_motifs:
                candi_motif = random.sample(select_zone, 3)
                candi_motif.sort()
            
            anomaly_motifs.append(candi_motif)
        return anomaly_motifs     


    def rebuild_anomaly_adj(self):
        indices = self.edges.T
        src_nodes = indices[0].tolist()
        dst_nodes = indices[1].tolist()
        src_nodes = [self.node2id[i] for i in src_nodes]
        dst_nodes = [self.node2id[i] for i in dst_nodes]
        anomaly_motifs_src = []
        anomaly_motifs_dst = []
        for motif in self.anomaly_motifs:
            anomaly_motifs_src += [motif[0],motif[1],motif[2]]
            anomaly_motifs_dst += [motif[1],motif[2],motif[0]]
        src_nodes += anomaly_motifs_src
        dst_nodes += anomaly_motifs_dst
        edges = np.vstack((src_nodes, dst_nodes)).T
        values = torch.ones(len(src_nodes), dtype=torch.float32)
        adj = coo_matrix((values, (src_nodes, dst_nodes)),
                         shape=(self.nodes_num, self.nodes_num))
        adj = self.convert_symmetric(adj, sparse=True)
        norm_adj = preprocess_adj(adj, is_sparse=True).to_dense()
        adj = adj2tensor(adj, is_sparse=True)
        return adj, norm_adj, edges
        
    
    def build_motif_adj(self):
        indices = self.edges.T
        src_nodes = indices[0].tolist()
        dst_nodes = indices[1].tolist()
        src_nodes = [self.node2id[i] for i in src_nodes]
        dst_nodes = [self.node2id[i] for i in dst_nodes]
        motif_src_nodes = []
        motif_dst_nodes = []
        for motif_id, motif in zip(self.motif_id_list, self.motifs):
            neighbor_set = set()
            for node in motif:
                for neighbor in self.adj[node]._indices()[0]:
                    neighbor_set.add(int(neighbor))
                neighbor_set.add(node)
            motif_dst_nodes+=list(neighbor_set)
            motif_src_nodes+=[motif_id]*len(neighbor_set)
        
        src_nodes += motif_src_nodes
        dst_nodes += motif_dst_nodes
            
        values = torch.ones(len(src_nodes), dtype=torch.float32)
        adj = coo_matrix((values, (src_nodes, dst_nodes)),
                         shape=(self.augmented_nodes_num, self.augmented_nodes_num))
        adj = self.convert_symmetric(adj, sparse=True)
        norm_adj = preprocess_adj(adj, is_sparse=True).to_dense()
        adj = adj2tensor(adj, is_sparse=True).to_dense()
        adj = torch.sqrt(adj/adj.sum(-1,keepdim=True))
        # adj = adj/adj.sum(-1,keepdim=True)
        return adj, norm_adj


class AnomalyEdgeSnapShot(SnapShot):
    def __init__(self, edges, p=0.02) -> None:
        super().__init__(edges)
        self.edges_list = self.rebuild_edges()
        self.motif_num = self.edges_num
        self.anomaly_edges = self.generate_anomalys(p)
        self.adj,self.norm_adj = self.rebuild_anomaly_adj()
        self.label_adj = self.adj.to_dense()
        self.anomaly_nodes = np.unique(np.array(self.anomaly_edges)).tolist()
        self.anomaly_motifs_num = len(self.anomaly_edges)

        self.motifs = self.edges_list + self.anomaly_edges
        self.augmented_nodes_num = self.nodes_num + len(self.motifs)
        self.augmented_nodes_list = list(range(self.augmented_nodes_num))
        self.motif_start_id = self.nodes_num
        self.motif_id_list = self.augmented_nodes_list[self.motif_start_id:]

        self.motif_labels = np.array([0]*self.edges_num +[1]*self.anomaly_motifs_num)
        self.nodes_labels = np.zeros(self.nodes_num, dtype=np.int)
        self.nodes_labels[self.anomaly_nodes] = 1
        self.motif_adj, self.motif_norm_adj = self.build_motif_adj()

    def rebuild_edges(self):
        edges = self.edges.tolist()
        edges = [[self.node2id[i] for i in edge] for edge in edges]
        return edges
        
    def generate_anomalys(self, p):
        anomaly_edges = []
        anomaly_edges_num = max(1, round(self.edges_num*p))
        for _ in range(anomaly_edges_num):
            candi_edges = random.sample(self.node_ids, 2) #list
            candi_edges.sort()
            while candi_edges in self.edges_list or candi_edges in anomaly_edges:
                candi_edges = random.sample(self.node_ids, 2)
                candi_edges.sort()
            
            anomaly_edges.append(candi_edges)
        return anomaly_edges
    
    def rebuild_anomaly_adj(self):
        indices = self.edges.T
        src_nodes = indices[0].tolist()
        dst_nodes = indices[1].tolist()
        src_nodes = [self.node2id[i] for i in src_nodes]
        dst_nodes = [self.node2id[i] for i in dst_nodes]

        anomaly_edges_src = []
        anomaly_edges_dst = []
        for edge in self.anomaly_edges:
            anomaly_edges_src += [edge[0]]
            anomaly_edges_dst += [edge[1]]
        src_nodes += anomaly_edges_src
        dst_nodes += anomaly_edges_dst
        values = torch.ones(len(src_nodes), dtype=torch.float32)
        adj = coo_matrix((values, (src_nodes, dst_nodes)),
                         shape=(self.nodes_num, self.nodes_num))
        adj = self.convert_symmetric(adj, sparse=True)
        norm_adj = preprocess_adj(adj, is_sparse=True).to_dense()

        adj = adj2tensor(adj, is_sparse=True)
        return adj, norm_adj
    
    def build_motif_adj(self):
        indices = self.edges.T
        src_nodes = indices[0].tolist()
        dst_nodes = indices[1].tolist()
        src_nodes = [self.node2id[i] for i in src_nodes]
        dst_nodes = [self.node2id[i] for i in dst_nodes]
        motif_src_nodes = []
        motif_dst_nodes = []
        for motif_id, motif in zip(self.motif_id_list, self.motifs):
            neighbor_set = set()
            for node in motif:
                for neighbor in self.adj[node]._indices()[0]:
                    neighbor_set.add(int(neighbor))
                neighbor_set.add(node)
            motif_dst_nodes+=list(neighbor_set)
            motif_src_nodes+=[motif_id]*len(neighbor_set)
        
        src_nodes += motif_src_nodes
        dst_nodes += motif_dst_nodes

        anomaly_edges_src = []
        anomaly_edges_dst = []
        for edge in self.anomaly_edges:
            anomaly_edges_src += [edge[0]]
            anomaly_edges_dst += [edge[1]]
        src_nodes += anomaly_edges_src
        dst_nodes += anomaly_edges_dst
            
        values = torch.ones(len(src_nodes), dtype=torch.float32)
        adj = coo_matrix((values, (src_nodes, dst_nodes)),
                         shape=(self.augmented_nodes_num, self.augmented_nodes_num))
        adj = self.convert_symmetric(adj, sparse=True)
        norm_adj = preprocess_adj(adj, is_sparse=True).to_dense()

        adj = adj2tensor(adj, is_sparse=True).to_dense()
        adj = torch.sqrt(adj/adj.sum(-1,keepdim=True))
        # adj = adj/adj.sum(-1,keepdim=True)
        return adj, norm_adj