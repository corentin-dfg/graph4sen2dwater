from data.graphdata_noPE import Graph

class SitsToGraph(object):
    def __init__(self, nb_spxl, compactness, slic_multitemporal, k_mesh2grid):
        self.nb_spxl = nb_spxl
        self.compactness = compactness
        self.slic_multitemporal = slic_multitemporal
        self.k_mesh2grid = k_mesh2grid

    def __call__(self, sits):
        return Graph(sits['imgs'], sits['dates'], sits['crs'], sits['coord_transform'], sits['path'], self.nb_spxl, self.compactness, self.slic_multitemporal, self.k_mesh2grid)