import torch
from torch import nn
import torch_geometric as pyg
from .mlp import MLP

class Mesh2MeshGNN(nn.Module):
    def __init__(self, mesh_node_dim, edge_dim, hid_dim):
        super().__init__()
        self.edge_mlp = MLP(edge_dim+mesh_node_dim+mesh_node_dim, hid_dim, hid_dim)
        self.mesh_mlp = MLP(mesh_node_dim+hid_dim, hid_dim, hid_dim)

    def forward(self, data):
        edge_new = self.edge_mlp(torch.cat((data['mesh'].x[data['mesh','to','mesh'].edge_index[0]],data['mesh'].x[data['mesh','to','mesh'].edge_index[1]]), dim=-1))
        mesh_new = self.mesh_mlp(torch.cat((data['mesh'].x, pyg.utils.scatter(edge_new, data['mesh', 'to', 'mesh'].edge_index[1], dim=0, reduce='sum', dim_size=data['mesh'].x.shape[0])), dim=-1))

        data['mesh'].x = data['mesh'].x + mesh_new
        return data