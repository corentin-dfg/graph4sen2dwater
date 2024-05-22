import argparse

import os
import numpy as np
import torch
import torch_geometric as pyg
from torchvision import transforms
from torchmetrics.regression import  MeanSquaredError
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

from data.sits_to_graph_SPE_noTPE import SitsToGraph
from data import SitsDataset, DeterminedSubSitsFromEnd
from models.graphcast_SPE_noTPE import GraphCast

from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int) #0, 1, 2
parser.add_argument('--spxl', default=256, type=int)
parser.add_argument('--compactness', default=0.1, type=float)
parser.add_argument('--tlen', default=7, type=int)
args = parser.parse_args()

SEED = args.seed

dataset_root_path = "./SEN2DWATER_patched_NDWI_tlen7_relocated_splitted/"

T_LEN = args.tlen
IMG_WIDTH = 64
IMG_HEIGHT = 64
BANDS = 1

NB_SPXL = args.spxl
COMPACTNESS = args.compactness
SLIC_MULTITEMPORAL = False
K_MESH2GRID = 3

HIDDEN_DIM = 64
LR = 0.0001
BATCH_SIZE = 5
EPOCHS = 50

def make_deterministic(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
make_deterministic(SEED)

subsits = DeterminedSubSitsFromEnd(T_LEN)
to_graph = SitsToGraph(NB_SPXL, COMPACTNESS, SLIC_MULTITEMPORAL, K_MESH2GRID)
transform = transforms.Compose([subsits, to_graph])
train_ds = SitsDataset(dataset_root_path+"train/", (BANDS, IMG_HEIGHT, IMG_WIDTH), transform=transform)
val_ds = SitsDataset(dataset_root_path+"val/", (BANDS, IMG_HEIGHT, IMG_WIDTH), transform=transform)
test_ds = SitsDataset(dataset_root_path+"test/", (BANDS, IMG_HEIGHT, IMG_WIDTH), transform=transform)

train_loader_generator = torch.Generator().manual_seed(SEED)
train_dl = pyg.loader.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=train_loader_generator, num_workers=4, pin_memory=True)
val_loader_generator = torch.Generator().manual_seed(SEED)
val_dl = pyg.loader.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, generator=val_loader_generator, num_workers=4, pin_memory=True)
test_loader_generator = torch.Generator().manual_seed(SEED)
test_dl = pyg.loader.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, generator=test_loader_generator, num_workers=4, pin_memory=True)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = GraphCast(n_features=BANDS,
                        t_len=T_LEN-1,
                        hid_dim=HIDDEN_DIM,
                        n_processor_layers=4
                    )
print("# Params: ", count_parameters(model))

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
model = model.to(device)

print("LR: ", LR)
opt = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)
criterion = torch.nn.HuberLoss()
rmse = MeanSquaredError(squared=False).to(device)
ssim = StructuralSimilarityIndexMeasure(data_range=(-1,1)).to(device)
psnr = PeakSignalNoiseRatio(data_range=(-1,1)).to(device)


writer = SummaryWriter(comment=f"_{model._get_name()}_SEED-{SEED}_SPLX-{NB_SPXL}_COMPACTNESS-{COMPACTNESS}_TLEN-{T_LEN}_SPE-noTPE")

best_test_loss = float('inf')

for epoch in range(EPOCHS):
    #TRAIN
    model.train()

    train_loss = 0.0
    train_rmse = 0.0
    train_ssim = 0.0
    train_psnr = 0.0

    for data in tqdm(train_dl, desc=f"[TRAIN {epoch+1}/{EPOCHS}]"):
        data = data.to(device)
        
        next_ndwi = model(data)
        loss = criterion(next_ndwi, data['grid'].y)

        train_loss += loss.item()
        train_rmse += rmse(next_ndwi.reshape((-1, BANDS, IMG_HEIGHT, IMG_WIDTH)), data['grid'].y.reshape((-1, BANDS, IMG_HEIGHT, IMG_WIDTH))).item()
        train_ssim += ssim(next_ndwi.reshape((-1, BANDS, IMG_HEIGHT, IMG_WIDTH)), data['grid'].y.reshape((-1, BANDS, IMG_HEIGHT, IMG_WIDTH))).item()
        train_psnr += psnr(next_ndwi.reshape((-1, BANDS, IMG_HEIGHT, IMG_WIDTH)), data['grid'].y.reshape((-1, BANDS, IMG_HEIGHT, IMG_WIDTH))).item()

        loss.backward()
        opt.step()
        opt.zero_grad()

    print(f"[TRAIN {epoch+1}/{EPOCHS}] Loss: {train_loss/len(train_dl)}")
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'scheduler': scheduler.state_dict(),
        }, os.path.join(writer.log_dir,"last.pt"))

    writer.add_scalar('LR', opt.param_groups[0]['lr'], epoch+1)
    writer.add_scalar('Loss/train', train_loss/len(train_dl), epoch+1)
    writer.add_scalar('RMSE/train', train_rmse/len(train_dl), epoch+1)
    writer.add_scalar('SSIM/train', train_ssim/len(train_dl), epoch+1)
    writer.add_scalar('PSNR/train', train_psnr/len(train_dl), epoch+1)

    #VAL
    model.eval()

    val_loss = 0.0
    val_rmse = 0.0
    val_ssim = 0.0
    val_psnr = 0.0

    with torch.no_grad():
        for i, data in enumerate(tqdm(val_dl, desc=f"[VAL {epoch+1}/{EPOCHS}]")):
            data = data.to(device)
            
            current_state_x = data['grid'].x[:,(T_LEN-2)*BANDS:(T_LEN-1)*BANDS]
            next_ndwi = model(data)
            loss = criterion(next_ndwi, data['grid'].y)

            val_loss += loss.item()
            val_rmse += rmse(next_ndwi.reshape((-1, BANDS, IMG_HEIGHT, IMG_WIDTH)), data['grid'].y.reshape((-1, BANDS, IMG_HEIGHT, IMG_WIDTH))).item()
            val_ssim += ssim(next_ndwi.reshape((-1, BANDS, IMG_HEIGHT, IMG_WIDTH)), data['grid'].y.reshape((-1, BANDS, IMG_HEIGHT, IMG_WIDTH))).item()
            val_psnr += psnr(next_ndwi.reshape((-1, BANDS, IMG_HEIGHT, IMG_WIDTH)), data['grid'].y.reshape((-1, BANDS, IMG_HEIGHT, IMG_WIDTH))).item()
    
    scheduler.step(val_loss)

    writer.add_scalar('Loss/val', val_loss/len(val_dl), epoch+1)
    writer.add_scalar('RMSE/val', val_rmse/len(val_dl), epoch+1)
    writer.add_scalar('SSIM/val', val_ssim/len(val_dl), epoch+1)
    writer.add_scalar('PSNR/val', val_psnr/len(val_dl), epoch+1)

    print(f"[VAL {epoch+1}/{EPOCHS}] Loss: {val_loss/len(val_dl)}")

    #TEST
    test_loss = 0.0
    test_rmse = 0.0
    test_ssim = 0.0
    test_psnr = 0.0

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_dl, desc=f"[TEST {epoch+1}/{EPOCHS}]")):
            data = data.to(device)
            
            current_state_x = data['grid'].x[:,(T_LEN-2)*BANDS:(T_LEN-1)*BANDS]
            next_ndwi = model(data)
            loss = criterion(next_ndwi, data['grid'].y)

            test_loss += loss.item()
            test_rmse += rmse(next_ndwi.reshape((-1, BANDS, IMG_HEIGHT, IMG_WIDTH)), data['grid'].y.reshape((-1, BANDS, IMG_HEIGHT, IMG_WIDTH))).item()
            test_ssim += ssim(next_ndwi.reshape((-1, BANDS, IMG_HEIGHT, IMG_WIDTH)), data['grid'].y.reshape((-1, BANDS, IMG_HEIGHT, IMG_WIDTH))).item()
            test_psnr += psnr(next_ndwi.reshape((-1, BANDS, IMG_HEIGHT, IMG_WIDTH)), data['grid'].y.reshape((-1, BANDS, IMG_HEIGHT, IMG_WIDTH))).item()

    writer.add_scalar('Loss/test', test_loss/len(test_dl), epoch+1)
    writer.add_scalar('RMSE/test', test_rmse/len(test_dl), epoch+1)
    writer.add_scalar('SSIM/test', test_ssim/len(test_dl), epoch+1)
    writer.add_scalar('PSNR/test', test_psnr/len(test_dl), epoch+1)

    print(f"[TEST {epoch+1}/{EPOCHS}] Loss: {test_loss/len(test_dl)}")

writer.close()
