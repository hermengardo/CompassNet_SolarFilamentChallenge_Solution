import os
import random

import cv2
import pandas as pd
import torch
from torch.utils.data import DataLoader

from CompassNet import CompassNet
from Config import CFG
from Loss import SegSpineLoss
from SunDataset import SunDataset
from utils import (
    collate_fn,
    load_train_val,
    preprocess,
    random_gamma,
    random_rotation,
    train_loop,
)

random.seed(CFG.seed)
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def augmentation(image, mask):
    if random.random() < 0.9:
        image = torch.flip(image, [-1])
        mask = torch.flip(mask, [-1])

    if random.random() < 0.9:
        image = torch.flip(image, [-2])
        mask = torch.flip(mask, [-2])

    if random.random() < 0.9:
        image, mask = random_rotation(image, mask)
        
    if random.random() < 0.25:
        image = random_gamma(image, min_gamma=0.5, max_gamma=1.5)
        
    return image, mask

train, val = load_train_val()

train_dataset = SunDataset(
    train, 
    preprocess=preprocess, 
    augmentation=augmentation,
    spine=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=CFG.batch_size,
    num_workers=CFG.num_workers,
    pin_memory=True,
    collate_fn=collate_fn,
    generator=torch.Generator().manual_seed(CFG.seed),
    shuffle=True,
    persistent_workers=True,
    multiprocessing_context="fork"
)

val_dataset = SunDataset(
    val, 
    preprocess=preprocess,
    spine=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CFG.batch_size,
    num_workers=CFG.num_workers,
    pin_memory=True,
    collate_fn=collate_fn,
    shuffle=False,
    persistent_workers=True,
    multiprocessing_context="fork"
)

cnet = CompassNet(
    dropout=0.1, 
    weights_path="native",
    freeze_bn=True, 
    num_heads=4,
    dilations=(2, 4, 8),
    spine=True
)

history = train_loop(
    cnet, 
    train_loader, 
    val_loader, 
    epochs=100,
    path=f"cnet@{CFG.resolution}x{CFG.seed}.pt",
    loss_fn=SegSpineLoss(spine_weight=0.05),
    val_loss_fn=SegSpineLoss(seg_only=True),
    patience=100
)

pd.DataFrame(history).to_csv(f"cnet@{CFG.resolution}x{CFG.seed}.csv", index=False)
