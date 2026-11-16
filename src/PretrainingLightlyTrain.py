import timm
import numpy as np
import random
import datetime as dt
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2
import math
import json
import optuna
import pandas as pd
import os
import lightly_train

from sklearn.metrics import jaccard_score
from functools import wraps
from sklearn.model_selection import train_test_split
from datetime import timedelta
from collections import defaultdict
from typing import Callable
from torchvision import transforms
from pathlib import Path
from torch.utils.data import IterableDataset, get_worker_info, DataLoader, Dataset
from tqdm import tqdm
from pycocotools import mask as coco_mask
from scipy.ndimage import binary_fill_holes

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

class Config:
    seed: int = 112
    pin_memory: bool = True
    batch_size: int = 128
    image_resolution: int = 1024
    local_resolution: int = 518
    num_workers: int = 4
    lr: float = 5e-5
    weight_decay: float = 4e-2
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    output = "/home/jaoliveira/not_related/out/"
    gong = "/home/jaoliveira/not_related/data/ssl_post/gong/"


CFG = Config()

random.seed(CFG.seed)

if __name__ == "__main__":
    lightly_train.train(
        out=CFG.output,
        data=CFG.gong,
        overwrite=True,
        model="dinov2_vit/vits14_pretrained",
        method="dinov2",
        epochs=100,
        batch_size=64,
        seed=CFG.seed,
        num_workers=CFG.num_workers,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
    )