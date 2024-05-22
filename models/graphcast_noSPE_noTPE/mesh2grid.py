import torch
from torch import nn
import torch_geometric as pyg
from .mlp import MLP
    
class Mesh2GridGNN(nn.Module):
    def __init__(self, grid_node_dim, mesh_node_dim, edge_dim, hid_dim):
        super().__init__()
        self.edge_mlp = MLP(edge_dim+mesh_node_dim+grid_node_dim, hid_dim, hid_dim)
        self.grid_mlp = MLP(grid_node_dim+hid_dim, hid_dim, hid_dim)

    def forward(self, data):
        edge_new = self.edge_mlp(torch.cat((data['mesh'].x[data['mesh','to','grid'].edge_index[0]],data['grid'].x[data['mesh','to','grid'].edge_index[1]]), dim=-1))
        grid_new = self.grid_mlp(torch.cat((data['grid'].x, pyg.utils.scatter(edge_new, data['mesh', 'to', 'grid'].edge_index[1], dim=0, reduce='sum', dim_size=data['grid'].x.shape[0])), dim=-1))

        data['grid'].x = data['grid'].x + grid_new
        return data
    
