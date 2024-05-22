import torch
import torch_geometric as pyg

import rasterio
import numpy as np
from skimage.measure import regionprops_table
from skimage.morphology import dilation, erosion
from skimage.segmentation import slic

class Graph(pyg.data.HeteroData):

    def __init__(self, imgs=None, dates=None, crs=None, coord_transform=None, path=None, nb_spxl=0, compactness=0, slic_multitemporal=False, k_mesh2grid=3):
        super().__init__()

        if not imgs is None and not dates is None and not crs is None and not coord_transform is None and not path is None and nb_spxl>=0:
            self.imgs = imgs
            self.dates = dates
            self.path = path
            self.img_height = self.imgs.shape[-2]
            self.img_width = self.imgs.shape[-1]

            # SLIC region adjacency mesh
            if slic_multitemporal:
                ## multitemporal:
                segmented = torch.as_tensor(slic(imgs[:-1,0].numpy(), n_segments=nb_spxl, compactness=.1, convert2lab=False, start_label=0, channel_axis=0), dtype=int)
            else:
                ## last date:
                segmented = torch.as_tensor(slic(self.imgs[-2,0], n_segments=nb_spxl, compactness=compactness, convert2lab=False, start_label=0), dtype=int)
            
            self['grid'].node_index = torch.arange(self.img_height*self.img_width)
            self['mesh'].node_index = torch.unique(segmented)

            self['grid'].x = torch.zeros((self['grid'].node_index.shape[0],0), dtype=torch.float32)
            self['mesh'].x = torch.zeros((self['mesh'].node_index.shape[0],0), dtype=torch.float32)

            self['grid'].pos = Graph.lat_lon_from_pxl_coord(
                                    torch.stack([torch.arange(self.img_height).unsqueeze(-1).repeat(1, self.img_height).view(-1), torch.arange(self.img_width).repeat(self.img_width)], dim=-1).float()
                                    , crs, coord_transform)
            mesh_centroids = regionprops_table(segmented.numpy()+1, properties=('label','centroid'))
            self['mesh'].pos = Graph.lat_lon_from_pxl_coord(
                                    torch.vstack((torch.as_tensor(mesh_centroids['centroid-0']), torch.as_tensor(mesh_centroids['centroid-1']))).swapaxes(0,1).float()
                                    , crs, coord_transform)

            self['grid'].y = torch.zeros((self['grid'].node_index.shape[0],0), dtype=torch.float32)

            self['grid', 'to', 'mesh'].edge_index = torch.vstack((self['grid'].node_index, segmented.flatten()))
            self['mesh', 'to', 'mesh'].edge_index = Graph.rag(segmented)
            self['mesh', 'to', 'grid'].edge_index = pyg.nn.knn(self['mesh'].pos, self['grid'].pos, k_mesh2grid).flip(0)
            
            self['grid', 'to', 'mesh'].edge_attr = torch.zeros((self['grid', 'to', 'mesh'].edge_index.shape[1],0), dtype=torch.float32)
            self['mesh', 'to', 'mesh'].edge_attr = torch.zeros((self['grid', 'to', 'mesh'].edge_index.shape[1],0), dtype=torch.float32)
            self['mesh', 'to', 'grid'].edge_attr = torch.zeros((self['grid', 'to', 'mesh'].edge_index.shape[1],0), dtype=torch.float32)

            self.__init_grid_node_features()
            self.__init_mesh_node_features()
            self.__init_grid2mesh_edge_features()
            self.__init_mesh2mesh_edge_features()
            self.__init_mesh2grid_edge_features()

            self.__init_grid_node_y()

    def __init_grid_node_features(self):
        ndwi = self.imgs.flatten(start_dim=1, end_dim=-1)[:-1].swapaxes(0,1)
        self['grid'].x = ndwi

    def __init_mesh_node_features(self):
        None

    def __init_grid2mesh_edge_features(self):
        None

    def __init_mesh2mesh_edge_features(self):
        None

    def __init_mesh2grid_edge_features(self):
        None

    def __init_grid_node_y(self):
        self['grid'].y = self.imgs.flatten(start_dim=1, end_dim=-1)[-1:].swapaxes(0,1)

    def normalize(self, mean, std):
        self['grid'].x = (self['grid'].x - mean)/std
        self['grid'].x = torch.where(torch.isfinite(self['grid'].x), self['grid'].x, 0)
        self['grid'].y = (self['grid'].y - mean)/std
        self['grid'].y = torch.where(torch.isfinite(self['grid'].y), self['grid'].y, 0)        

    def grid_node_ndwi_dim(n_features, t_len):
        return t_len*n_features
    
    def grid_node_tpe_dim(n_features, t_len):
        return 0
    
    def grid_node_spe_dim(n_features, t_len):
        return 0

    def grid_node_dim(n_features, t_len):
        return Graph.grid_node_ndwi_dim(n_features, t_len) + Graph.grid_node_tpe_dim(n_features, t_len) + Graph.grid_node_spe_dim(n_features, t_len)
    
    def mesh_node_dim():
        return 0
    
    def edge_dim():
        return 0
    
    def rag(segmented):
        # Processing the edge indices with a RAG (Region Adjacency Graph)
        segmented_bordered = np.full((segmented.shape[0]+2,segmented.shape[1]+2),-1)
        segmented_bordered[1:-1,1:-1] = segmented
        eroded = erosion(segmented_bordered)
        dilated = dilation(segmented_bordered)
        boundaries0 = (eroded != segmented_bordered)
        boundaries1 = (dilated != segmented_bordered)
        labels_small = torch.as_tensor(np.concatenate((eroded[boundaries0], segmented_bordered[boundaries1])))
        labels_large = torch.as_tensor(np.concatenate((segmented_bordered[boundaries0], dilated[boundaries1])))

        edge_index, inter = torch.unique(torch.stack((labels_small,labels_large)),dim=1, return_counts=True)
        inter = inter[edge_index[0,:]!=-1]
        edge_index = edge_index[:,edge_index[0,:]!=-1]
        edge_index = pyg.utils.to_undirected(edge_index)
        return edge_index

    def lat_lon_from_pxl_coord(pos, src_crs, src_transform, dst_crs=rasterio.crs.CRS.from_epsg(4326)):
        # param & return: torch tensor [num_nodes, num_dimensions]
        xs_src_crs, ys_src_crs = rasterio.transform.xy(src_transform, pos[:,1].numpy(), pos[:,0].numpy())
        xs_dst_crs, ys_dst_crs = rasterio.warp.transform(src_crs, dst_crs, np.array(xs_src_crs), np.array(ys_src_crs))
        return torch.tensor([xs_dst_crs, ys_dst_crs], dtype=torch.float32).T #lon, lat
