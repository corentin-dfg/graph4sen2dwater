from torch import nn
from .mlp import MLP

class Outputter(nn.Module):
    def __init__(self, grid_node_dim, hid_dim, out_dim):
        super().__init__()
        self.grid_node_mlp = MLP(grid_node_dim, hid_dim, out_dim, last=True)

    def forward(self, data):
        return self.grid_node_mlp(data['grid'].x)