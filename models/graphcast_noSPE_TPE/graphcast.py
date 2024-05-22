from torch import nn

from data.graphdata_noSPE import Graph

from .embedder_separatePE import Embedder
from .grid2mesh import Grid2MeshGNN
from .mesh2mesh import Mesh2MeshGNN
from .mesh2grid import Mesh2GridGNN
from .output import Outputter

class GraphCast(nn.Module):
    def __init__(self, n_features, t_len, hid_dim, n_processor_layers):
        super().__init__()
        self.n_features = n_features
        self.t_len = t_len

        self.n_processor_layers = n_processor_layers

        self.embedder = Embedder(Graph.grid_node_ndwi_dim(self.n_features, self.t_len), Graph.grid_node_tpe_dim(self.n_features, self.t_len), Graph.grid_node_spe_dim(self.n_features, self.t_len), Graph.mesh_node_dim(), Graph.edge_dim(), hid_dim)
        self.grid2mesh = Grid2MeshGNN(hid_dim, 0, 0, hid_dim)
        processor_layers = []
        for i in range(self.n_processor_layers):
            processor_layers.append(Mesh2MeshGNN(hid_dim, 0, hid_dim))
        self.mesh2mesh = nn.Sequential(*processor_layers)
        self.mesh2grid = Mesh2GridGNN(hid_dim, hid_dim, 0, hid_dim)
        self.outputter = Outputter(hid_dim, hid_dim, n_features)

    def forward(self, data):
        data = data.clone()

        current_state_x = data['grid'].x[:,(self.t_len-1)*self.n_features:self.t_len*self.n_features]
        
        # Encoder
        data = self.embedder(data)
        data = self.grid2mesh(data)

        # Processor
        data = self.mesh2mesh(data)

        # Decoder
        data = self.mesh2grid(data)

        pred_diff_state_x = self.outputter(data)
        pred_diff_state_x = pred_diff_state_x * pred_diff_state_x.std(0) # Output normalization
        pred_next_state_x = current_state_x + pred_diff_state_x

        return pred_next_state_x
