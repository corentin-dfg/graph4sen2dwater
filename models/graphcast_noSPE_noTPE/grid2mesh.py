import torch
from torch import nn
import torch_geometric as pyg
from .mlp import MLP
       
class Grid2MeshGNN(nn.Module):
    def __init__(self, grid_node_dim, mesh_node_dim, edge_dim, hid_dim):
        super().__init__()
        self.edge_mlp = MLP(edge_dim+grid_node_dim+mesh_node_dim, hid_dim, hid_dim)
        self.mesh_mlp = MLP(mesh_node_dim+hid_dim, hid_dim, hid_dim)
        self.grid_mlp = MLP(grid_node_dim, hid_dim, hid_dim)

    def forward(self, data):
        edge_new = self.edge_mlp(data['grid'].x[data['grid','to','mesh'].edge_index[0]])
        mesh_new = self.mesh_mlp(pyg.utils.scatter(edge_new, data['grid', 'to', 'mesh'].edge_index[1], dim=0, reduce='sum', dim_size=data['mesh'].x.shape[0]))
        grid_new = self.grid_mlp(data['grid'].x)

        data['grid'].x = data['grid'].x + grid_new
        data['mesh'].x = mesh_new
        return data