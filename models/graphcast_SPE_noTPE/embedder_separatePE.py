import torch
from torch import nn
from .mlp import MLP

class Embedder(nn.Module):
    def __init__(self, grid_node_ndwi_dim, grid_node_tpe_dim, grid_node_spe_dim, mesh_node_dim, edge_dim, hid_dim):
        super().__init__()
        self.ndwi_ts_dim=grid_node_ndwi_dim
        self.tpe_dim=grid_node_tpe_dim
        self.spe_dim=grid_node_spe_dim
        self.grid_node_x_mlp = MLP(self.ndwi_ts_dim, hid_dim, hid_dim)
        self.grid_node_pe_mlp = MLP(self.tpe_dim+self.spe_dim, hid_dim, hid_dim)
        self.grid_node_mlp = MLP(hid_dim*2, hid_dim, hid_dim)
        self.mesh_node_mlp = MLP(mesh_node_dim, hid_dim, hid_dim)
        self.grid2mesh_edge_mlp = MLP(edge_dim, hid_dim, hid_dim)
        self.mesh2mesh_edge_mlp = MLP(edge_dim, hid_dim, hid_dim)
        self.mesh2grid_edge_mlp = MLP(edge_dim, hid_dim, hid_dim)

    def forward(self, data):
        x = self.grid_node_x_mlp(data['grid'].x[:,:self.ndwi_ts_dim])
        pe = self.grid_node_pe_mlp(data['grid'].x[:,self.ndwi_ts_dim:])
        data['grid'].x = self.grid_node_mlp(torch.cat((x,pe),dim=1))
        data['mesh'].x = self.mesh_node_mlp(data['mesh'].x)
        data['grid', 'to', 'mesh'].edge_attr = self.grid2mesh_edge_mlp(data['grid', 'to', 'mesh'].edge_attr)
        data['mesh', 'to', 'mesh'].edge_attr = self.mesh2mesh_edge_mlp(data['mesh', 'to', 'mesh'].edge_attr)
        data['mesh', 'to', 'grid'].edge_attr = self.mesh2grid_edge_mlp(data['mesh', 'to', 'grid'].edge_attr)
        return data