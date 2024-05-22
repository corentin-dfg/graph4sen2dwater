import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import random
import numpy as np
import glob
import os

class datahandler:
    '''
        Data Handler, it handle the dataset
    '''

    def __init__(self, dataset_root):
        '''
            It creates a Data Handler object.
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            The dataset must be organized in this format:

            RootFolder
            │   README.md
            │   requirements.txt    
            │
            └───dataset_root
                └───basin_1
                    └───zone_1
                        └───t0.tif
                        └───t1.tif
                        └─── ...
                        └───t30.tif
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            Input
                - root: root path of the dataset
        '''

        self.dataset_root = dataset_root
        self.paths = self.__load_paths()
        self.paths_keys = sorted(list(self.paths.keys()))
    
    def __load_paths(self, verbose=False):
        '''
            Loads the images path

            Output:
                - paths: python dictionary, keys==locations, values==paths of timeseries
        '''

        # Dict keys correspond to the last level of the folder hierarchy
        geo_locations = glob.glob(os.path.join(self.dataset_root, '**/'), recursive=True)
        paths = {}
        for i, c in enumerate(geo_locations):
            imgs_c = glob.glob(os.path.join(c, '*.tif'))
            if len(imgs_c)>0:
                imgs_c.sort()
                paths[c.split(os.sep)[-2]] = imgs_c

        return paths
    