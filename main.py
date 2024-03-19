import os
import argparse
import torch
#from monai.apps import DecathlonDataset
"""
import tempfile
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from monai.config import print_config
from monai.data import DataLoader
from monai.transforms import (
    EnsureChannelFirstd,
    CenterSpatialCropd,
    Compose,
    Lambdad,
    LoadImaged,
    Resized,
    ScaleIntensityd,
)
from monai.utils import set_determinism
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from generative.inferers import DiffusionInferer
"""
from models.nets import DiffusionModelUNet
from models.schedulers import DDPMScheduler, DDIMScheduler

def main(args) :

    print(f'step 1. data loader')
    """
    train_ds = DecathlonDataset(root_dir=root_dir,
                                task="Task01_BrainTumour",
                                transform=data_transform,
                                section="training", download=True)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=8, persistent_workers=True)
    val_ds = DecathlonDataset(root_dir=root_dir, task="Task01_BrainTumour", transform=data_transform, section="validation",
                              download=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False,
                            num_workers=8, persistent_workers=True)
    """

    print(f'step 2. model')
    device = "cuda"
    model = DiffusionModelUNet(spatial_dims=3,
                               in_channels=3,
                               out_channels=1,
                               num_channels=[256, 256, 512],
                               attention_levels=[False, False, True],
                               num_head_channels=[0, 0, 512],
                               num_res_blocks=2,
                               cross_attention_dim = 768)
    model.to(device)
    temp_images = torch.randn(1,3,3)
    timestep = 0
    context = torch.randn(1,4,768)
    noise_pred = model(x=temp_images,
                       timesteps = timestep,
                       context = context)




if __name__ == '__main__' :

    parser = argparse.ArgumentParser()
    args = parser.parse_argument()
    main(args)
