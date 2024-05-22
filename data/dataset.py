import torch
import torchvision
import rasterio
import numpy as np

from glob import glob
import os

class SitsDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_root, img_shape, latlon_from_crs=True, transform=None):
        super().__init__()
        self.dataset_root = dataset_root
        self.paths = self.__load_paths()
        self.paths_keys = sorted(list(self.paths.keys()))
        self.img_shape = img_shape # (C, H, W)
        self.latlon_from_crs = latlon_from_crs
        self.transform = transform

    def __load_paths(self):
        # Dict keys correspond to the last level of the folder hierarchy
        geo_locations = glob(os.path.join(self.dataset_root, '**/'), recursive=True)
        paths = {}
        for i, c in enumerate(geo_locations):
            imgs_c_tif = glob(os.path.join(c, '*.tif'))
            imgs_c_tiff = glob(os.path.join(c, '*.tiff'))
            imgs_c = imgs_c_tif + imgs_c_tiff
            if len(imgs_c)>0:
                imgs_c.sort()
                paths[c.split(os.sep)[-2]] = imgs_c

        return paths
    
    def __load_data(self, path):        
        RIO_FORMAT = ['.tif', '.tiff']

        date = np.array(os.path.splitext(os.path.basename(path))[0].split('-'), dtype=int)
        
        if any(frmt in path for frmt in RIO_FORMAT):
            with rasterio.open(path) as src:
                data     = src.read()
                data = np.moveaxis(data, 0, -1)
                metadata = src.profile
                return data, metadata, date, src.crs, src.transform
            
        else:
            raise Exception('File can not be opened, format not supported.')

    def __len__(self):
        return len(self.paths_keys)
    
    def __getitem__(self, idx):
        dir = self.paths_keys[idx]
        t_len = len(self.paths[dir])
                    
        imgs = torch.zeros((t_len,) + self.img_shape) # (T, C, W, H)
        dates = torch.zeros((t_len, 3), dtype=int) # (T, [Y, M, D])

        for i, file in enumerate(self.paths[dir]):
            img, _, date, crs, coord_transform = self.__load_data(file) #NOTE NaN data must be filtered beforehand
            imgs[i] = torchvision.transforms.ToTensor()(img)
            dates[i] = torch.as_tensor(date, dtype=int)
        
        if self.latlon_from_crs:
            sits = {'imgs': imgs, 'dates': dates, 'crs': crs, 'coord_transform': coord_transform, 'path': dir}
        else:
            sits = {'imgs': imgs, 'dates': dates, 'path': dir}

        if self.transform:
            sits = self.transform(sits)

        return sits
    