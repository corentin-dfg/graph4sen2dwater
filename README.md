# Forecasting water resources from satellite image time series using a graph-based learning strategy: GraphCast for Satellite Image Time Series

Source code for the paper "**[Forecasting water resources from satellite image time series using a graph-based learning strategy]([https://github.com/corentin-dfg/graph4sen2dwater](https://isprs-archives.copernicus.org/articles/XLVIII-2-2024/81/2024/))**" by _[Corentin Dufourg](https://www.linkedin.com/in/corentin-dufourg/), [Charlotte Pelletier](https://sites.google.com/site/charpelletier), Stéphane May and [Sébastien Lefèvre](http://people.irisa.fr/Sebastien.Lefevre/)_, at **[ISPRS Technical Commission II Symposium 2024](https://www.isprs.org/tc2-symposium2024/)**.

We propose to adapt a recent weather forecasting architecture, called [GraphCast](https://www.science.org/doi/abs/10.1126/science.adi2336), to a water resources forecasting task using high-resolution satellite image time series (SITS). Based on an intermediate mesh, the data geometry used within the network is adapted to match high spatial resolution data acquired in two-dimensional space. In particular, we introduce a predefined irregular mesh based on a segmentation map to guide the network’s predictions and bring more detail to specific areas.


|       | # Params  | RMSE (&darr;) | PSNR (&uarr;) | SSIM (&uarr;) | Runtime (min) |
| ---: | :---: | :---: | :---: | :---: | :---: |
| **Input average** | - | 0.1550 | 23.32 | 0.7465 | - |
| **Persistence**   | - | 0.1332 | 25.03 | 0.7897 | - | 
| **LSTM**   | 17,345 | 0.1162<sub>&plusmn;0.0005</sub> | 25.53<sub>&plusmn;0.05</sub> | **0.8282**<sub>&plusmn;0.0005</sub> | 22 | 
| **ConvLSTM**   | 150,721 | 0.1197<sub>&plusmn;0.0029</sub> | 25.28<sub>&plusmn;0.19</sub> | 0.8113<sub>&plusmn;0.0030</sub> | 26 | 
| **TDCNN-ConvLSTM**   | 407,681 | 0.1111<sub>&plusmn;0.0008</sub> | 25.68<sub>&plusmn;0.08</sub> | 0.8083<sub>&plusmn;0.0008</sub> | 46 | 
| **Ours**   | 228,673 | **0.1097**<sub>&plusmn;0.0035</sub> | **26.42**<sub>&plusmn;0.27</sub> | 0.8170<sub>&plusmn;0.0070</sub> | 41 | 


**Table:** *Number of parameters, RMSE, PSNR, SSIM and runtimefor baseline models and our GraphCast adaptation. The results are provided with average and standard deviation on three random initializations (__best__).*


## Overview

* ```ndwi_tiny_generate_dataset.py``` can be used to generate a reduced version of the [SEN2DWATER](https://ieeexplore.ieee.org/abstract/document/10282352) dataset, retaining only the NDWI channel over a limited number of dates.
* ```relocated_splitted_patches.py``` fixes the geolocation of the splitted patches from the [SEN2DWATER](https://ieeexplore.ieee.org/abstract/document/10282352) dataset.
* ```train_baseline_train-val-test.py``` is the method to train the baseline models (LSTM, ConvLSTM, TDCNN-ConvLSTM).
* ```train_graphcast_train-val-test.py``` is the method to train our adaptation of GraphCast. The ```NB_SPXL```, ```T_LEN``` and ```SLIC_MULTITEMPORAL``` constants can be set to reproduce auxiliary results.
* ```train_graphcast_train-val-test_noSPE-noTPE.py```, ```train_graphcast_train-val-test_SPE-noTPE.py``` and ```train_graphcast_train-val-test_noSPE-TPE.py``` are used to perform the ablation study of the temporal and spatial encodings.
* ```dataset/``` should contain the data from SEN2DWATER to train and evaluate the models.
* ```data/``` contains data structure and dataset classes.
* ```dataio/``` contains the reading functions of the SEN2DWATER dataset, adapted from its [GitHub repository](https://github.com/francescomauro1998/SEN2DWATER).
* ```models/``` contains the architectures implemented with PyTorch and PyG.


### License Information
The data derived from [SEN2DWATER](https://ieeexplore.ieee.org/abstract/document/10282352) dataset are governed by the [Legal Notice on the use of Copernicus Sentinel Data and Service Information](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice/).


## Reproducibility 

The split used for the data is given in folder ```datasets/```. The seeds used for the 3 random initializations are ```0,1,2```. Note that the ```pyg.utils.scatter``` operation use in Graph Neural Networks may behave nondeterministically when given tensors on a CUDA device.


## Citation

If you use this work, consider citing our [paper](https://isprs-archives.copernicus.org/articles/XLVIII-2-2024/81/2024/):

```latex
@article{dufourg2024forecasting,
  title={Forecasting water resources from satellite image time series using a graph-based learning strategy},
  author={Dufourg, Corentin and Pelletier, Charlotte and May, St{\'e}phane and Lef{\`e}vre, S{\'e}bastien},
  journal={The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
  volume={48},
  pages={81--88},
  year={2024},
  publisher={Copernicus GmbH}
}
```
