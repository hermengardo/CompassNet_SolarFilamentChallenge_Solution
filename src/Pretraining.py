import os
import random
import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from CompassNet import CompassNet
from utils import set_optimizer, validate, collate_fn, random_rotation, random_gamma

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

class Config:
    seed: int = 112
    pin_memory: bool = True
    batch_size: int = 4
    image_resolution: int = 1024
    num_workers: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    patience: int = 50
    val_size: float = 0.1
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    n_blocks: tuple = (4, 9)
    block_frac: tuple = (0.05, 0.16)
    max_hidden: float = 0.25
    warmup: int = 500

    output = "./pretraining/"
    gong = "./data/ssl_post/gong/"


CFG = Config()

random.seed(CFG.seed)


def block_mask(size, rng):
    mask = torch.zeros(1, size, size)

    for _ in range(rng.randint(*CFG.n_blocks)):
        height = int(size * rng.uniform(*CFG.block_frac))
        width = int(height * rng.uniform(0.5, 2.0))
        y = rng.randint(0, size - height)
        x = rng.randint(0, size - width)
        mask[:, y:y + height, x:x + width] = 1.0

        if mask.mean() > CFG.max_hidden:
            break

    return mask


def augment(img):
    if random.random() < 0.5:
        img = torch.flip(img, dims=(-1,))
    if random.random() < 0.5:
        img = torch.flip(img, dims=(-2,))
    if random.random() < 0.9:
        img, _ = random_rotation(img, img)
    if random.random() < 0.9:
        img = random_gamma(img)

    return img


class MaskedDataset(Dataset):
    def __init__(self, files, train=True):
        self.files = files
        self.train = train

    def __getitem__(self, idx):
        img = cv2.imread(str(self.files[idx]), cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0
        img = cv2.resize(
            img,
            (CFG.image_resolution, CFG.image_resolution),
            interpolation=cv2.INTER_AREA,
        )
        img = torch.from_numpy(img).unsqueeze(0)

        if self.train:
            img = augment(img)
            rng = random
            noise = torch.randn_like(img)
        else:
            rng = random.Random(CFG.seed + idx)
            gen = torch.Generator().manual_seed(CFG.seed + idx)
            noise = torch.randn(img.shape, generator=gen)

        disk = img > 0
        hide = block_mask(CFG.image_resolution, rng) * disk
        masked = torch.where(hide > 0, img[disk].mean() + 0.03 * noise, img)

        return masked, torch.cat([img, hide])

    def __len__(self):
        return len(self.files)


class MaskedCompletionLoss(nn.Module):
    def __init__(self, kirsch):
        super().__init__()
        self.kirsch = kirsch
        self.scales = len(kirsch.filters.dilations)

    def forward(self, output, target):
        img = target[:, :1]
        hide = target[:, 1:]
        batch, _, height, width = img.shape

        with torch.no_grad(), torch.autocast(device_type=img.device.type, enabled=False):
            ori = self.kirsch(img.float()).reshape(batch, self.scales, 3, height, width)
            direction = ori[:, :, :2].flatten(1, 2)
            magnitude = ori[:, :, 2:3]
            weight = magnitude / (magnitude.mean((3, 4), keepdim=True) + 1e-6)
            weight = weight.clamp(max=3.0)
            weight = weight.expand(-1, -1, 2, -1, -1).flatten(1, 2)

        loss_img = (
            hide * (output[:, :1] - F.avg_pool2d(img, 9, 1, 4)).abs()
        ).sum() / hide.sum().clamp_min(1.0)

        weight = hide * weight
        loss_ori = (
            weight * (torch.tanh(output[:, 1:]) - direction).abs()
        ).sum() / weight.sum().clamp_min(1e-6)

        return loss_img + loss_ori

if __name__ == "__main__":
    files = []
    for path in sorted(Path(CFG.gong).rglob("*")):
        if path.is_file() and path.suffix.lower() == ".png":
            files.append(path)

    train_files, val_files = train_test_split(
        files, test_size=CFG.val_size, random_state=CFG.seed
    )

    train_loader = DataLoader(
        MaskedDataset(train_files, train=True),
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=CFG.pin_memory,
        drop_last=True,
        collate_fn=collate_fn,
    )
    
    val_loader = DataLoader(
        MaskedDataset(val_files, train=False),
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        pin_memory=CFG.pin_memory,
        collate_fn=collate_fn,
    )

    model = CompassNet(weights_path="native", spine=False, num_heads=4, freeze_bn=True)
    model.head = nn.Conv2d(
        model.head.in_channels,
        1 + 2 * len(model.kirsch.filters.dilations),
        kernel_size=1,
    )
    model.to(CFG.device)

    loss_fn = MaskedCompletionLoss(model.kirsch).to(CFG.device)

    optimizer = set_optimizer(
        model=model,
        optimizer_fn=torch.optim.AdamW,
        lr=CFG.lr,
        weight_decay=CFG.weight_decay,
    )

    total_steps = CFG.epochs * len(train_loader)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: min(
            (s + 1) / CFG.warmup,
            0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)),
        ),
    )

    best_val = float("inf")
    val_loss = 0.0
    count = 0

    os.makedirs(CFG.output, exist_ok=True)
    for epoch in range(CFG.epochs):
        model.train()
        loss_sum = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CFG.epochs}")

        for batch, (img, target) in enumerate(pbar):
            img = img.to(CFG.device, non_blocking=True)
            target = target.to(CFG.device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(img)
                loss = loss_fn(output.float(), target.float())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            loss_sum += loss.item()
            pbar.set_postfix(
                val=f"{val_loss:.4f}",
                loss=f"{loss_sum / (batch + 1):.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.6f}",
            )

        val_loss = validate(model, val_loader, loss_fn, CFG.device)

        if val_loss < best_val:
            count = 0
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(CFG.output, "checkpoint.pt"))
        else:
            count += 1

        if count > CFG.patience:
            print(f"Stopped early on epoch {epoch}")
            break