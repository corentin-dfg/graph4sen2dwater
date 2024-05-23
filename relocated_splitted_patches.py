import os
from glob import glob
import rasterio
from tqdm import tqdm

dataset_root_path = "dataset/SEN2DWATER_patched_NDWI_tlen7_relocated_splitted/"

NB_PATCH_PER_LINE = 4
IMG_SIZE = 64
SPATIAL_RES = 10 # 10m/pxl for Sentinel-2 images

def load_paths():
    # Dict keys correspond to the last level of the folder hierarchy
    geo_locations = glob(os.path.join(dataset_root_path, '**/'), recursive=True)
    paths = {}
    for i, c in enumerate(geo_locations):
        imgs_c_tif = glob(os.path.join(c, '*.tif'))
        imgs_c_tiff = glob(os.path.join(c, '*.tiff'))
        imgs_c = imgs_c_tif + imgs_c_tiff
        if len(imgs_c)>0:
            imgs_c.sort()
            paths[c.split(os.sep)[-2]] = imgs_c

    return paths

paths = load_paths()
paths_keys = sorted(list(paths.keys()))
print(paths_keys)

for dir in tqdm(paths_keys):
    for path in paths[dir]:
        with rasterio.open(path) as src:
            data     = src.read()
            metadata = src.profile

            counter = int(str.split(dir, '-')[1])
            metadata['transform'] = rasterio.Affine.translation(counter%NB_PATCH_PER_LINE*IMG_SIZE*SPATIAL_RES,-(counter//NB_PATCH_PER_LINE)*IMG_SIZE*SPATIAL_RES) * metadata['transform']
            
            with rasterio.open(path+"f", 'w', **metadata) as dst:    
                dst.write(data)
