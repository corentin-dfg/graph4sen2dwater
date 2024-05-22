from .normalizer import normalizer
from .spectral_indices import spectral_indices
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
import rasterio
import random
import cv2
import os

class datareader:
    '''
        Class containing static methods for reading images
    '''

    @staticmethod
    def load(path):
        '''
            Load an image and its metadata given its path.
            
            The following image format are supported:
                - .png
                - .jpg
                - .jpeg
                - .tif
                - .npy
            please adapt your format accordingly. 
            
            Inputs:
                - path: position of the image, if None the function will ask for the image path using a menu
                - info (optional): allows to print process informations
            Outputs:
                - data: WxHxB image, with W width, H height and B bands
                - metadata: dictionary containing image metadata
        '''
        
        MPL_FORMAT = ['.png', '.jpg', '.jpeg']
        RIO_FORMAT = ['.tif', '.tiff']
        NP_FORMAT  = ['.npy']
        
        if any(frmt in path for frmt in RIO_FORMAT):
            with rasterio.open(path) as src:
                data     = src.read()
                metadata = src.profile
            data = np.moveaxis(data, 0, -1)
            
        elif any(frmt in path for frmt in MPL_FORMAT):
            data = plt.imread(path)
            metadata = None

        elif any(frmt in path for frmt in NP_FORMAT):
            data     = np.load(path)
            metadata = None
            
        else:
            data     = None
            metadata = None
            print('!!! File can not be opened, format not supported !!!')

        date = np.array(os.path.splitext(os.path.basename(path))[0].split('-'), dtype=int)
        lat = float(os.path.split(path)[-2].split('_')[-3])
        lon = float(os.path.split(path)[-2].split('_')[-1])
            
        return data, metadata, date, lat, lon
   
    @staticmethod
    def save(image, path, meta):
        '''
            Save an image and its metadata given its path
            Inputs:
                - image: the image to be saved
                - path: position of the image
                - meta: metadata for the image to be saved
        '''

        RASTERIO_EXTENSIONS   = ['.tif', '.tiff']
        MATPLOTLIB_EXTENSIONS = ['.png', '.jpg', 'jpeg']

        if any(frmt in path for frmt in RASTERIO_EXTENSIONS):

            if meta!=None:
                meta.update({'driver':'GTiff',
                            'width':image.shape[0],
                            'height':image.shape[1],
                            'count':image.shape[2],
                            'dtype':'float64'})

            with rasterio.open(fp=path, mode='w',**meta) as dst:
                for count in range(image.shape[2]):
                    dst.write(image[:,:,count], count+1)

        elif any(frmt in path for frmt in MATPLOTLIB_EXTENSIONS):
            plt.imsave(path, image)

        else:
            print('[!] File cannot be saved, format not supported!')
