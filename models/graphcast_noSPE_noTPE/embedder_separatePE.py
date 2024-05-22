import torch
from torch import nn
from .mlp import MLP

class Embedder(nn.Module):
    def __init__(self, grid_node_ndwi_dim, grid_node_tpe_dim, grid_node_spe_dim, mesh_node_dim, edge_dim, hid_dim):
        super().__init__()
        self.ndwi_ts_dim=grid_node_ndwi_dim
        self.grid_node_x_mlp = MLP(self.ndwi_ts_dim, hid_dim, hid_dim)
        self.grid_node_mlp = MLP(hid_dim, hid_dim, hid_dim)

    def forward(self, data):
        x = self.grid_node_x_mlp(data['grid'].x[:,:self.ndwi_ts_dim])
        data['grid'].x = self.grid_node_mlp(x)
        return data