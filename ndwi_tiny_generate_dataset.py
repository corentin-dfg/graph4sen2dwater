import os
from tqdm import tqdm

from dataio.datahandler import datahandler
from dataio.datareader import datareader
from dataio.spectral_indices import *
from dataio.normalizer import *

DATASET_ROOT_PATH = "../../data/SEN2DWATER_patched"

NORMALIZE = True
NORMALIZE_MAX_SCALE = 10000

NORMALIZED_DIFFERENCE = True
ND_BANDS = [2, 7] # NDWI [2, 7] ; NDVI [3, 7]

NAN_FILTER_ONLY_SUB_SITS = False
SUB_SITS_RANGE = slice(0, 7) # slice(inclusive, exclusive) or None
OUT_SUFFIX = "_NDWI_tlen7"

dh       = datahandler(DATASET_ROOT_PATH)
keys     = list(dh.paths.keys())

for key in tqdm(dh.paths.keys()):
    paths_filter = dh.paths[key] if SUB_SITS_RANGE==None or NAN_FILTER_ONLY_SUB_SITS==False else dh.paths[key][SUB_SITS_RANGE]
    
    contains_nan = False
    for img_path in paths_filter:
        data, _, _, _, _ = datareader.load(img_path)
        if NORMALIZED_DIFFERENCE:
            data_for_filter = data[:,:,ND_BANDS] # Needed to filter NaN value only on NDWI bands
        if np.any(np.isnan(data_for_filter)):
            contains_nan = True
            break
    
    if not contains_nan:
        paths = dh.paths[key] if SUB_SITS_RANGE==None else dh.paths[key][SUB_SITS_RANGE]
        for img_path in paths:
            data, metadata, _, _, _ = datareader.load(img_path)
            
            if NORMALIZE==True:
                data = normalizer.max_scaler(data, NORMALIZE_MAX_SCALE)
            
            if NORMALIZED_DIFFERENCE==True:
                data = spectral_indices.normalized_difference(data, ND_BANDS)[:,:,np.newaxis]

            out_path = dh.dataset_root+OUT_SUFFIX+img_path[len(dh.dataset_root):]
            os.makedirs(os.path.split(out_path)[0], exist_ok=True)
            datareader.save(data, out_path, metadata)

