import torch.nn as nn
import torch.nn.functional as F
import torch
from layers import GraphConvolution, MultiHeadAttention, SpGraphAttentionLayer
from torch_geometric.nn import SAGEConv

def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        



class Encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout, n_layers):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.gc1 = GraphConvolution(nfeat, nhid)
        if n_layers > 1:
            self.stack_layers = [GraphConvolution(nhid, nhid) for _ in range(n_layers-1)]
            self.stack_layers = nn.ModuleList(self.stack_layers)
        # self.gc2 = GraphConvolution(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, motif_emb, adj, pad_n, pos_idx):
        x = torch.cat((x,motif_emb))
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        if self.n_layers > 1:
            for enc_layer in self.stack_layers:
                x = F.relu(enc_layer(x, adj))        
        # motif_emb = x[-len(motif_emb):]
        x = x[:-len(motif_emb)]
        hid = x.size(1)
        device = x.device
        output = torch.zeros(pad_n, hid).to(device)
        output[pos_idx] = x
        return output

class GCNEncoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCNEncoder, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj, pad_n, pos_idx):
        # x = torch.cat((x,motif_emb))
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        # motif_emb = x[-len(motif_emb):]
        hid = x.size(1)
        device = x.device
        output = torch.zeros(pad_n, hid).to(device)
        output[pos_idx] = x
        return output

class GATEncoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GATEncoder, self).__init__()

        self.gat1 = SpGraphAttentionLayer(nfeat, 
                                                 nhid*2, 
                                                 dropout=dropout, 
                                                 alpha=0.1, 
                                                 concat=True)
        self.gat2 = SpGraphAttentionLayer(nhid*2, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=0.1, 
                                                 concat=True)
        self.dropout = dropout

    def forward(self, x, adj, pad_n, pos_idx):
        # x = torch.cat((x,motif_emb))
        adj= adj.to_dense()
        x = F.relu(self.gat1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gat2(x, adj))
        # motif_emb = x[-len(motif_emb):]
        hid = x.size(1)
        device = x.device
        output = torch.zeros(pad_n, hid).to(device)
        output[pos_idx] = x
        return output


class SAGEEncoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(SAGEEncoder, self).__init__()

        self.gc1 = SAGEConv(nfeat, nhid)
        self.gc2 = SAGEConv(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj, pad_n, pos_idx):
        # x = torch.cat((x,motif_emb))
        device = x.device
        adj = adj.to_dense().type(torch.LongTensor).to(device)
        adj = adj.to_sparse().indices()
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        # motif_emb = x[-len(motif_emb):]
        hid = x.size(1)
        
        output = torch.zeros(pad_n, hid).to(device)
        output[pos_idx] = x
        return output

class Attribute_Decoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(Attribute_Decoder, self).__init__()
        self.gc1 = GraphConvolution(nhid, nhid)
        self.gc2 = GraphConvolution(nhid, nfeat)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))

        return x


class Structure_Decoder(nn.Module):
    def __init__(self, nhid, dropout, n_layers):
        super(Structure_Decoder, self).__init__()
        self.n_layers = n_layers
        # self.gc_layers = [GraphConvolution(nhid, nhid) for _ in range(n_layers)]
        # self.gc_layers = nn.ModuleList(self.gc_layers)
        self.dropout = dropout

    def forward(self, x, adj):
        # for gc in self.gc_layers:
            # x = F.relu(gc(x, adj))
            # 
        # x = F.dropout(x, self.dropout, training=self.training)
        x = x @ x.T

        return x

class MotifFeatExtract(nn.Module):
    def __init__(self, hid, num_heads, dropout):
        super().__init__()
        self.motif_token = nn.Embedding(1, hid)
        self.attn = MultiHeadAttention(hid, num_heads, dropout)
    
    def forward(self, x):
        #[motif_batch, 3, dim]
        motif_token_feat = self.motif_token.weight.repeat(x.size(0), 1, 1)
        x = torch.cat((motif_token_feat, x),dim=1)
        #[motif_batch, 4, dim]
        x = self.attn(x)
        x = x[:,0,:]
        # [motif_batch, dim]
        return x

class MyModelwoM(nn.Module):
    def __init__(self, nodes_num, snap_len, feat_size, hidden_size, num_heads, dropout, encoder='gcn', device ="cuda:0"):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.nodes_num = nodes_num
        self.node = torch.arange(nodes_num, device=self.device)
        self.node_embedding = nn.Embedding(nodes_num, feat_size)
        self.encoder = encoder
        if encoder == 'gcn':
            self.shared_encoder = GCNEncoder(feat_size, hidden_size, dropout)
        elif encoder == 'gat':
            self.shared_encoder = GATEncoder(feat_size, hidden_size, dropout)
        elif encoder == 'graphsage':
            self.shared_encoder = SAGEEncoder(feat_size, hidden_size, dropout)
            
        self.struct_decoder = Structure_Decoder(hidden_size, dropout,n_layers=1)

        self.time_encoder = nn.Embedding(2*snap_len-1, num_heads)
        self.relative_matrix = self.build_matrix(snap_len)

        self.attn = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.apply(lambda module: init_params(module))
    
    def build_matrix(self, snap_len):
        pos_matrix_1 = torch.stack([idx*torch.ones(snap_len, device="cuda:0",dtype=torch.int64) for idx in range(snap_len)])
        pos_matrix_2 = torch.stack([torch.arange(snap_len, device=self.device) for _ in range(snap_len)])
        matrix = (pos_matrix_2 - pos_matrix_1) + snap_len - 1
        return matrix
    
    def forward(self, snapshots):

        x = [self.node_embedding.weight[snap.nodes] for snap in snapshots]

        if self.encoder == 'gcn':
            x = torch.stack([self.shared_encoder(x[i], snap.norm_adj, self.nodes_num, snap.nodes) for i,snap in enumerate(snapshots)]) 
        else:
            x = torch.stack([self.shared_encoder(x[i], snap.adj, self.nodes_num, snap.nodes) for i,snap in enumerate(snapshots)]) 

        x = x.transpose(0, 1) # [node_num, snap_len, hid]
        attn_bias = self.time_encoder(self.relative_matrix)        # self.edges = np.unique(edges, axis=0)
        x = self.attn(x, attn_bias) #[node_num, snap_len, hid]
        x = x.transpose(0, 1) #[snap_len, node_num, hid]
        x = [x[i][snapshots[i].nodes] for i in range(len(snapshots))]
        # motif_node_embs = [[x[i][motif] for motif in snap.motifs] for i,snap in enumerate(snapshots)]
        # snap_motif_node_embs = [torch.stack([x[i][motif] for motif in snapshots[i].motifs]) for i in range(len(snapshots))] #[snap_len, motif_num, 3, hid]
        
        output = [self.struct_decoder(x[i], snapshots[i].norm_adj) for i in range(len(snapshots))]
        return output

class MADG(nn.Module):
    def __init__(self, nodes_num, snap_len, feat_size, hidden_size, num_heads, dropout, n_layers, device ="cuda:0"):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.nodes_num = nodes_num
        self.node = torch.arange(nodes_num, device=self.device)
        self.node_embedding = nn.Embedding(nodes_num, feat_size)
        self.shared_encoder = Encoder(feat_size, hidden_size, dropout,n_layers=n_layers)
        self.struct_decoder = Structure_Decoder(hidden_size, dropout,n_layers=1)

        self.time_encoder = nn.Embedding(2*snap_len-1, num_heads)
        self.relative_matrix = self.build_matrix(snap_len)

        self.attn = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.MotifFE1 = MotifFeatExtract(feat_size, num_heads, dropout)
        self.MotifFE2 = MotifFeatExtract(hidden_size, num_heads, dropout)
        self.apply(lambda module: init_params(module))
    
    def build_matrix(self, snap_len):
        pos_matrix_1 = torch.stack([idx*torch.ones(snap_len, device="cuda:0",dtype=torch.int64) for idx in range(snap_len)])
        pos_matrix_2 = torch.stack([torch.arange(snap_len, device=self.device) for _ in range(snap_len)])
        matrix = (pos_matrix_2 - pos_matrix_1) + snap_len - 1
        return matrix
    
    def forward(self, snapshots):
        x = [self.node_embedding.weight[snap.nodes] for snap in snapshots]
        motif_embs = [self.MotifFE1(torch.stack([x[i][motif] for motif in snapshots[i].motifs])) for i in range(len(x))]
        x = torch.stack([self.shared_encoder(x[i], motif_embs[i], snap.motif_norm_adj, self.nodes_num, snap.nodes) for i,snap in enumerate(snapshots)]) 
        x = x.transpose(0, 1) # [node_num, snap_len, hid]
        attn_bias = self.time_encoder(self.relative_matrix)
        attn_bias = attn_bias.transpose(1,2).transpose(0,1)
        x = self.attn(x, attn_bias) #[node_num, snap_len, hid]
        x = x.transpose(0, 1) #[snap_len, node_num, hid]
        x = [x[i][snapshots[i].nodes] for i in range(len(snapshots))]
        x = [torch.cat((x[i], self.MotifFE2(torch.stack([x[i][motif] for motif in snapshots[i].motifs])))) for i in range(len(x))] 
        output = [self.struct_decoder(x[i], snapshots[i].motif_norm_adj) for i in range(len(snapshots))]
        return output

class MyModelLocalAttention(nn.Module):
    def __init__(self, nodes_num, snap_len, feat_size, hidden_size, num_heads, dropout, w, device ="cuda:0"):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.nodes_num = nodes_num
        self.w = w
        self.node = torch.arange(nodes_num, device=self.device)
        self.node_embedding = nn.Embedding(nodes_num, feat_size)
        self.shared_encoder = Encoder(feat_size, hidden_size, dropout)
        self.struct_decoder = Structure_Decoder(hidden_size, dropout)

        self.time_encoder = nn.Embedding(2*snap_len-1, num_heads)
        self.relative_matrix = self.build_matrix(snap_len)

        self.attn = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.MotifFE1 = MotifFeatExtract(hidden_size, num_heads, dropout)
        self.MotifFE2 = MotifFeatExtract(hidden_size, num_heads, dropout)
        self.apply(lambda module: init_params(module))
    
    def build_matrix(self, snap_len):
        pos_matrix_1 = torch.stack([idx*torch.ones(snap_len, device="cuda:0",dtype=torch.int64) for idx in range(snap_len)])
        pos_matrix_2 = torch.stack([torch.arange(snap_len, device=self.device) for _ in range(snap_len)])
        matrix = (pos_matrix_2 - pos_matrix_1) + snap_len - 1
        return matrix
    
    def forward(self, snapshots):

        x = [self.node_embedding.weight[snap.nodes] for snap in snapshots]

        motif_embs = [self.MotifFE1(torch.stack([x[i][motif] for motif in snapshots[i].motifs])) for i in range(len(x))]
        x = torch.stack([self.shared_encoder(x[i], motif_embs[i], snap.motif_norm_adj, self.nodes_num, snap.nodes) for i,snap in enumerate(snapshots)]) 
        x = x.transpose(0, 1) # [node_num, snap_len, hid]
        attn_bias = self.time_encoder(self.relative_matrix)
        attn_bias = attn_bias.transpose(1,2).transpose(0,1)
        x = self.attn(x, attn_bias) #[node_num, snap_len, hid]
        x = x.transpose(0, 1) #[snap_len, node_num, hid]
        x = [x[i][snapshots[i].nodes] for i in range(len(snapshots))]
        x = [torch.cat((x[i], self.MotifFE2(torch.stack([x[i][motif] for motif in snapshots[i].motifs])))) for i in range(len(x))] 
        output = [self.struct_decoder(x[i], snapshots[i].motif_norm_adj) for i in range(len(snapshots))]
        return output

class MiniBatchModel(nn.Module):
    def __init__(self, nodes_num, snap_len, feat_size, hidden_size, num_heads, dropout, device ="cuda:0"):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.nodes_num = nodes_num
        self.node = torch.arange(nodes_num, device=self.device)
        self.node_embedding = nn.Embedding(nodes_num, feat_size)
        self.shared_encoder = Encoder(feat_size, hidden_size, dropout)
        self.struct_decoder = Structure_Decoder(hidden_size, dropout)
        
        self.time_encoder = nn.Embedding(2*snap_len-1, num_heads)
        self.relative_matrix = self.build_matrix(snap_len)

        self.attn = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.MotifFE1 = MotifFeatExtract(hidden_size, num_heads, dropout)
        self.MotifFE2 = MotifFeatExtract(hidden_size, num_heads, dropout)
        self.apply(lambda module: init_params(module))
    
    def build_matrix(self, snap_len):
        pos_matrix_1 = torch.stack([idx*torch.ones(snap_len, device="cuda:0",dtype=torch.int64) for idx in range(snap_len)])
        pos_matrix_2 = torch.stack([torch.arange(snap_len, device=self.device) for _ in range(snap_len)])
        matrix = (pos_matrix_2 - pos_matrix_1) + snap_len - 1
        return matrix
    
    def forward(self, snapshots, batch_size, snap_batch_nodes, snap_motifs_id, pos_id, adj_sample):
        # snap_batch_nodes: [snap, batch_nodes]
        snap_len = len(snapshots)        
        x = [self.node_embedding.weight[batch_nodes] for batch_nodes in snap_batch_nodes] #[snap_len, node_num, hid]
        motif_embs = [self.MotifFE1(torch.stack([x[i][motif] for motif in motifs])) for i, motifs in enumerate(snap_motifs_id)]
        x = torch.stack([self.shared_encoder(x[i], motif_embs[i], snap.motif_norm_adj[adj_sample[i]][:,adj_sample[i]], batch_size, pos_id[i]) for i,snap in enumerate(snapshots)]) 
        x = x.transpose(0, 1) # [node_num, snap_len, hid]
        attn_bias = self.time_encoder(self.relative_matrix)
        attn_bias = attn_bias.transpose(1,2).transpose(0,1)
        x = self.attn(x, attn_bias) #[node_num, snap_len, hid]
        x = x.transpose(0, 1) #[snap_len, node_num, hid]
        x = [x[i][pos_id[i]] for i in range(snap_len)]
        x = [torch.cat((x[i], self.MotifFE2(torch.stack([x[i][motif] for motif in motifs])))) for i, motifs in enumerate(snap_motifs_id)] 
        output = [self.struct_decoder(x[i], snapshots[i].motif_norm_adj[adj_sample[i]][:,adj_sample[i]]) for i in range(len(snapshots))]
        return output