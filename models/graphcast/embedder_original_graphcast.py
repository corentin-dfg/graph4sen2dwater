from torch import nn
from .mlp import MLP

class Embedder(nn.Module):
    def __init__(self, grid_node_dim, mesh_node_dim, edge_dim, hid_dim):
        super().__init__()
        self.grid_node_mlp = MLP(grid_node_dim, hid_dim, hid_dim)
        self.mesh_node_mlp = MLP(mesh_node_dim, hid_dim, hid_dim)
        self.grid2mesh_edge_mlp = MLP(edge_dim, hid_dim, hid_dim)
        self.mesh2mesh_edge_mlp = MLP(edge_dim, hid_dim, hid_dim)
        self.mesh2grid_edge_mlp = MLP(edge_dim, hid_dim, hid_dim)

    def forward(self, data):
        data['grid'].x = self.grid_node_mlp(data['grid'].x)
        data['mesh'].x = self.mesh_node_mlp(data['mesh'].x)
        data['grid', 'to', 'mesh'].edge_attr = self.grid2mesh_edge_mlp(data['grid', 'to', 'mesh'].edge_attr)
        data['mesh', 'to', 'mesh'].edge_attr = self.mesh2mesh_edge_mlp(data['mesh', 'to', 'mesh'].edge_attr)
        data['mesh', 'to', 'grid'].edge_attr = self.mesh2grid_edge_mlp(data['mesh', 'to', 'grid'].edge_attr)
        return data