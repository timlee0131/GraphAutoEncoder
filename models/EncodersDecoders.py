import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv

# GCN
class GCN_Encoder(nn.Module):
    def __init__(self, n_dim, n_latent) -> None:
        super(GCN_Encoder, self).__init__()
        self.conv1 = GCNConv(n_dim, n_latent * 2)
        self.conv2 = GCNConv(n_latent * 2, n_latent)
    
    def forward(self, x, edge_index):
        h1 = F.relu(self.conv1(x, edge_index))
        h2 = self.conv2(h1, edge_index)
        return h2
    
class GCN_Decoder(nn.Module):
    def __init__(self, n_latent, n_dim) -> None:
        super(GCN_Decoder, self).__init__()
        self.conv1 = GCNConv(n_latent, n_latent * 2)
        self.conv2 = GCNConv(n_latent * 2, n_dim)
    
    def forward(self, x, edge_index):
        z1 = F.relu(self.conv1(x, edge_index))
        z2 = self.conv2(z1, edge_index)
        return z2

# GAT
class GAT_Encoder(nn.Module):
    def __init__(self, n_dim, n_latent, heads=1, dropout=0) -> None:
        super(GAT_Encoder, self).__init__()
        self.conv1 = GATv2Conv(n_dim, n_latent * 2, heads=heads, concat=True, dropout=dropout)
        self.conv2 = GATv2Conv(n_latent * 2 * heads, n_latent, concat=False, dropout=dropout)
    
    def forward(self, x, edge_index):
        h1 = F.elu(self.conv1(x, edge_index))
        h2 = self.conv2(h1, edge_index)
        return h2
    
class GAT_Decoder(nn.Module):
    def __init__(self, n_latent, n_dim, heads=1, dropout=0) -> None:
        super(GAT_Decoder, self).__init__()
        self.conv1 = GATv2Conv(n_latent, n_latent * 2 * heads, heads=heads, concat=True, dropout=dropout)
        self.conv2 - GATv2Conv(n_latent * 2 * heads, n_dim, heads=heads, concat=False, dropout=dropout)
    
    def forward(self, x, edge_index):
        z1 = F.elu(self.conv1(x, edge_index))
        z2 = self.conv2(z1, edge_index)
        
        return z2