import json
import random

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from typing import Callable
from pycocotools import mask as coco_mask
from IPython.display import display

from torchmetrics.classification import (
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
)

from Loss import BCEDiceLoss, SegSpineLoss
from Config import CFG

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

def solar_disk_detection(img):
    height, width = img.shape

    small = cv2.resize(
        (np.clip(img, 0, 1) * 255).astype(np.uint8),
        (256, 256),
        interpolation=cv2.INTER_AREA
    )

    blur = cv2.GaussianBlur(small, (9, 9), 2.0)
    edges = cv2.Canny(blur, 30, 100)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return width / 2, height / 2, min(width, height) * 0.45

    image_center = np.array([127.5, 127.5])
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1000:
            continue

        (x, y), r = cv2.minEnclosingCircle(contour)
        center_distance = np.linalg.norm(
            np.array([x, y]) - image_center
        )

        if r > 40 and center_distance < 50:
            candidates.append((area, x, y, r, contour))

    if not candidates:
        return width/2, height/2, min(width, height)*0.45
        
    _, x, y, r, _ = max(candidates, key=lambda c: c[0])

    scale_x = width/256
    scale_y = height/256

    x *= scale_x
    y *= scale_y
    r *= (scale_x + scale_y)/2

    return x, y, r

def disk_masking(height, width, center_x, center_y, radius):
    Y, X = np.ogrid[:height, :width]
    mask = np.hypot(X - center_x, Y - center_y) <= radius 
    
    return mask

def radial_flattening(img, center_x, center_y, radius, mask, window_size=21):
    height, width = img.shape
    
    Y, X = np.ogrid[:height, :width]
    dist = np.hypot(X - center_x, Y - center_y)
    dist = dist.astype(np.int32)

    radius = int(radius)
    ring_index = np.clip(dist, 0, radius)

    masked_ring_index = ring_index[mask]
    masked_values = img[mask].astype(np.float64)

    ring_sums = np.bincount(
        masked_ring_index, 
        weights=masked_values, 
        minlength=radius + 1
    )
    
    ring_counts = np.bincount(
        masked_ring_index, 
        minlength=radius + 1
    ).astype(np.float64)
    
    radial_profile = (
        ring_sums/np.maximum(ring_counts, 1)
    ).astype(np.float32)

    has_pixels = ring_counts > 0

    if not has_pixels.all():
        radial_profile[~has_pixels] = np.interp(
            np.flatnonzero(~has_pixels),
            np.flatnonzero(has_pixels),
            radial_profile[has_pixels],
        )

    padded_profile = np.pad(radial_profile, window_size//2, mode="edge")
    smoothing_kernel = np.ones(window_size) / window_size
    radial_profile = np.convolve(padded_profile, smoothing_kernel, mode="valid").astype(np.float32)

    background = radial_profile[ring_index]
    reference = float(np.mean(radial_profile))
    
    flattened = img * reference / np.maximum(background, 0.05)
    flattened = np.clip(flattened, 0, 1).astype(np.float32)
    
    return flattened

def clahe(img, mask, tileGridSize=(8, 8), clipLimit=2.0):
    img = np.where(mask, img, img[mask].mean())
    img = (np.clip(img, 0, 1)*255.0 + 0.5).astype(np.uint8)
    
    clahe = cv2.createCLAHE(
        clipLimit=clipLimit, 
        tileGridSize=tileGridSize
    )
    
    img = clahe.apply(img).astype(np.float32) / 255.0
    
    return img

def off_disk_zeroing(img, mask):
    img[~mask] = 0.0
    return img
def preprocess(img):
    height, width = img.shape

    center_x, center_y, radius = solar_disk_detection(img)
    mask = disk_masking(height, width, center_x, center_y, radius)

    img = radial_flattening(
        img, center_x, center_y, radius, mask,
        window_size=21
    )

    img = clahe(img, mask)
    img = off_disk_zeroing(img, mask)

    return img

def poly_to_mask(seg):
    rles = coco_mask.frPyObjects(
        seg, CFG.resolution, CFG.resolution
    )
    return coco_mask.decode(coco_mask.merge(rles)).astype(bool)

def read_image_metadata(path=CFG.metadata):
    with open(path) as f:
        coco = json.load(f)

    image_info = pd.DataFrame(coco["images"])
    annotations = pd.DataFrame(coco["annotations"])

    metadata = image_info.merge(annotations, left_on="id", right_on="image_id")[[
        "file_name", 
        "url", 
        "date_captured", 
        "segmentation", 
        "area", 
        "iscrowd", 
        "spine", 
        "image_id", 
        "bbox", 
        "category_id"
    ]]
    
    metadata["date_captured"] = pd.to_datetime(metadata["date_captured"])
    metadata["month"] = metadata["date_captured"].dt.month
    metadata["year"] = metadata["date_captured"].dt.year

    return metadata

def min_dist_days(days, ref_days):
    ref = np.unique(ref_days)
    idx = np.searchsorted(ref, days)
    left = np.abs(days - ref[np.clip(idx - 1, 0, len(ref) - 1)])
    right = np.abs(ref[np.clip(idx, 0, len(ref) - 1)] - days)
    return np.minimum(left, right)


def split(metadata, val_size=CFG.val_size, block_days=90, gap_days=30, seed=CFG.seed):
    rng = np.random.default_rng(seed)

    df = (
        metadata.drop_duplicates("file_name")[["file_name", "date_captured", "phase"]]
        .copy()
    )
    df["date"] = df["date_captured"].dt.normalize()
    df["day"] = (df["date"] - df["date"].min()).dt.days
    df["block"] = df["day"] // block_days

    blocks = (
        df.groupby("block")
        .agg(n=("file_name", "size"),
             phase=("phase", lambda s: s.mode().iloc[0]))
        .reset_index()
    )

    val_blocks = []
    for phase, g in blocks.groupby("phase", observed=True):
        target = val_size * g["n"].sum()
        acc = 0
        for i in rng.permutation(len(g)):
            if acc >= target:
                break
            row = g.iloc[i]
            val_blocks.append(row["block"])
            acc += row["n"]

    is_val = df["block"].isin(val_blocks).to_numpy()

    dist = min_dist_days(df["day"].to_numpy(), df.loc[is_val, "day"].to_numpy())
    is_train = (~is_val) & (dist >= gap_days)

    train_ids = df.loc[is_train, "file_name"].tolist()
    val_ids = df.loc[is_val, "file_name"].tolist()

    print("Train Set Distribution")
    display(df[is_train]["phase"].value_counts(normalize=True))

    print()
    print("Validation Set Distribution")
    display(df[is_val]["phase"].value_counts(normalize=True))

    return train_ids, val_ids


def audit_split(metadata, train_ids, val_ids, gap_days=30):
    df = (
        metadata.drop_duplicates("file_name")[["file_name", "date_captured"]]
        .copy()
    )
    df["day"] = (
        df["date_captured"].dt.normalize() - df["date_captured"].min().normalize()
    ).dt.days

    train_days = df.loc[df["file_name"].isin(train_ids), "day"].to_numpy()
    val_days = df.loc[df["file_name"].isin(val_ids), "day"].to_numpy()

    assert set(train_ids).isdisjoint(val_ids)

    dist = min_dist_days(train_days, val_days)
    assert dist.min() >= gap_days

    return dist


def load_train_val(
    silso_path="./data/SILSO.csv",
    val_size=CFG.val_size,
    seed=CFG.seed,
    block_days=90,
    gap_days=30,
):
    metadata = read_image_metadata()
    SILSO = pd.read_csv(
        silso_path,
        sep=";",
        header=None,
        names=["year", "month", "day", "decimal_year", "sn", "sn_std", "n_obs", "definitive"],
    )[["year", "month", "day", "sn"]]
    SILSO = SILSO[SILSO["sn"] != -1]
    SILSO = SILSO[SILSO.year.between(metadata.year.min(), metadata.year.max())]

    sunspots = (
        SILSO.groupby(["year", "month"], as_index=False)["sn"].mean()
        .rename(columns={"sn": "monthly_mean_sunspot_number"})
    )
    sunspots["phase"] = pd.qcut(
        sunspots["monthly_mean_sunspot_number"], 3, labels=["low", "moderate", "high"]
    )

    metadata = metadata.merge(sunspots, how="left", on=["year", "month"]).copy()

    train_ids, val_ids = split(
        metadata, val_size=val_size, block_days=block_days, gap_days=gap_days, seed=seed
    )
    audit_split(metadata, train_ids, val_ids, gap_days=gap_days)

    train = metadata[metadata["file_name"].isin(train_ids)]
    val = metadata[metadata["file_name"].isin(val_ids)]

    return train, val

def collate_fn(batch):
    images, masks = zip(*batch)
    return torch.stack(images), torch.stack(masks)

def set_optimizer(model, optimizer_fn, lr, weight_decay):
    decay_params = []
    no_decay_params = []
    bias_scale_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("bias_scale"):
            bias_scale_params.append(param)
        elif "bias" in name or param.ndim == 1:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return optimizer_fn(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
            {"params": bias_scale_params, "weight_decay": 0.0, "eps": 1e-16},
        ],
        lr=lr,
    )

@torch.no_grad()
def validate(model, val_dataset, loss_fn=BCEDiceLoss(), device=CFG.device):
    model.eval()
    total_loss, n = 0.0, 0

    for img, mask in tqdm(val_dataset, desc="Validate"):
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(img)

        loss = loss_fn(output.float(), mask.float())
        total_loss += loss.item()
        n += 1

    return total_loss / n

def evaluate_metrics(model, loader, device=CFG.device, threshold=0.5):
    model.eval()
    iou = BinaryJaccardIndex().to(device)
    precision = BinaryPrecision().to(device)
    recall = BinaryRecall().to(device)
    f1 = BinaryF1Score().to(device)

    for img, mask in tqdm(loader, desc="Metrics"):
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(img)

        pred = (torch.sigmoid(logits.float()) > threshold).long()
        target = (mask > 0.5).long()

        iou.update(pred, target)
        precision.update(pred, target)
        recall.update(pred, target)
        f1.update(pred, target)

    return {
        "iou": iou.compute().item(),
        "precision": precision.compute().item(),
        "recall": recall.compute().item(),
        "f1": f1.compute().item(),
    }

def train_loop(
    model: nn.Module,
    train_dataset: torch.utils.data.DataLoader,
    val_dataset: torch.utils.data.DataLoader,
    epochs: int=5,
    lr: float=CFG.lr,
    weight_decay:float=1e-4,
    device: torch.device=CFG.device,
    optimizer_fn:Callable=torch.optim.AdamW,
    loss_fn=SegSpineLoss(spine_weight=0.1),
    val_loss_fn=SegSpineLoss(seg_only=True),
    patience:int=5,
    path: str = "best.pt",
):
    if val_loss_fn is None:
        val_loss_fn = loss_fn
        
    optimizer = set_optimizer(
        model=model,
        lr=lr,
        optimizer_fn=optimizer_fn,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    model.to(device)

    history = defaultdict(list)
    best_val = float("inf")
    val_loss = 0.0
    count = 0

    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        batches = 0
        pbar = tqdm(train_dataset, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch, (img, mask) in enumerate(pbar):
            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                output = model(img)
                loss = loss_fn(output.float(), mask.float())

            loss_sum += loss.item()
            batches += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            pbar.set_postfix(
                val=f"{val_loss:.4f}",
                loss=f"{loss_sum / batches:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.6f}",
            )

        scheduler.step()
        val_loss = validate(model, val_dataset, val_loss_fn, device)

        history["val_loss"].append(val_loss)
        history["epoch"].append(epoch)
        history["train_loss"].append(loss_sum / batches)

        if val_loss < best_val:
            count = 0
            best_val = val_loss
            torch.save(model.state_dict(), path)
        else:
            count += 1

        if count > patience:
            print(f"Stopped early on epoch {epoch}")
            return history

    return history

def random_rotation(image, mask, max_angle=45.0):
    h, w = image.shape[-2:]
    rad = math.radians(random.uniform(-max_angle, max_angle))
    cos, sin = math.cos(rad), math.sin(rad)
    theta = torch.tensor(
        [[cos, -sin * h / w, 0.0], [sin * w / h, cos, 0.0]],
        dtype=image.dtype,
    ).unsqueeze(0)

    grid = F.affine_grid(theta, (1, 1, h, w), align_corners=False)
    image = F.grid_sample(image.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
    mask = F.grid_sample(mask.unsqueeze(0), grid, mode="nearest", padding_mode="zeros", align_corners=False)[0]
    return image, mask
    
def random_gamma(image, min_gamma=0.5, max_gamma=1.5):
    gamma = random.uniform(min_gamma, max_gamma)
    return torch.clamp(image, 0.0, 1.0) ** gamma

def apply(x, k, flip):
    if flip:
        x = torch.flip(x, dims=(-1,))
    x = torch.rot90(x, k, dims=(-2, -1))
    return x

def invert(y, k, flip):
    y = torch.rot90(y, -k, dims=(-2, -1))
    if flip:
        y = torch.flip(y, dims=(-1,))
    return y

@torch.no_grad()
def TTA_predict(model, x):
    total = 0
    for flip in (False, True):
        for k in range(4):
            augmented = apply(x, k, flip)
            logits = model(augmented)
            prob = torch.sigmoid(logits)
            prob = invert(prob, k, flip)
            total = total + prob
            
    return total/8

@torch.no_grad()
def predict(model, x):
    logits = model(x)
    prob = torch.sigmoid(logits)

    return prob

@torch.no_grad()
def probabilities(
    model,
    loader,
    predict_fn=TTA_predict,
    device=CFG.device,
):
    model.to(device)
    model.eval()
    probs = []
    targets = []
    for img, mask in tqdm(loader, desc=predict_fn.__name__):
        img = img.to(device)
        prob = predict_fn(model, img)
        prob = prob.cpu()
        probs.append(prob)

        target = mask > 0.5
        target = target.long()
        targets.append(target)

    probs = torch.cat(probs)
    targets = torch.cat(targets)
    return probs, targets

def clean_mask(prob, threshold=0.5, min_area=5):
    mask = (prob > threshold).astype(np.uint8)
    mask = fill_hole_and_denoise(mask, min_area)
    return mask
    
def score(
    probs,
    targets,
    threshold=0.5,
    **params,
):
    iou = BinaryJaccardIndex()
    precision = BinaryPrecision()
    recall = BinaryRecall()
    f1 = BinaryF1Score()

    for prob, target in zip(probs, targets):
        prob = prob[0]
        prob = prob.numpy()
        mask = clean_mask(
            prob,
            threshold,
            **params,
        )
        mask = torch.from_numpy(mask).long()
        target = target[0].long()

        iou.update(mask, target)
        precision.update(mask, target)
        recall.update(mask, target)
        f1.update(mask, target)

    return {
        "iou": iou.compute().item(),
        "precision": precision.compute().item(),
        "recall": recall.compute().item(),
        "f1": f1.compute().item(),
    }
    
# From https://github.com/irisaltHu/MORDEN/blob/main/segmentation/utils/post_processing/tools/hole_filling.py
def fill_hole_and_denoise(mask, threshold=5):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    canvas = np.zeros_like(mask, np.uint8)
    for i in range(len(contours)):
        tmp_canvas = np.zeros_like(mask, np.uint8)
        tmp_canvas = cv2.drawContours(tmp_canvas, contours, i, 255, -1)
        if np.count_nonzero(tmp_canvas) >= threshold:
            canvas = np.logical_or(canvas, tmp_canvas)

    return canvas

def load_image(path, resolution, preprocess):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    img = cv2.resize(preprocess(img), (resolution, resolution), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(img)[None, None]


def predict_probability(model, x, predict_fn, out_size):
    with torch.inference_mode():
        prob = predict_fn(model, x)[0, 0].cpu().numpy()
    return cv2.resize(prob, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


def encode_components(binary):
    n_labels, labels = cv2.connectedComponents(binary, connectivity=8)
    for k in range(1, n_labels):
        component = np.asfortranarray(labels == k, dtype=np.uint8)
        yield coco_mask.encode(component)["counts"].decode("utf-8")


def make_submission(
    model,
    test_dir,
    resolution,
    device,
    predict_fn,
    preprocess,
    clean_mask,
    threshold=0.5,
    min_area=0,
    out_size=2048,
    output_path="submission.csv",
):
    model.eval().to(device)
    min_area = int(min_area * (out_size / resolution) ** 2)

    rows = []
    for path in tqdm(sorted(Path(test_dir).glob("*"))):
        x = load_image(path, resolution, preprocess).to(device)
        prob = predict_probability(model, x, predict_fn, out_size)
        binary = clean_mask(prob, threshold=threshold, min_area=min_area).astype(np.uint8)
        rows.extend(
            {"filament_id": f"{path.stem}_{k}", "segmentation_rle": counts}
            for k, counts in enumerate(encode_components(binary), start=1)
        )

    submission = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    submission.to_csv(output_path, index=False)
    return submission