import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")

import cv2
import numpy as np
import multiprocessing as mp

SRC = Path("data/ssl")
DST = Path("data/ssl_post")
WORKERS = 6


def solar_disk_detection(gray):
    h, w = gray.shape
    small = cv2.resize((gray * 255).astype(np.uint8), (256, 256), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(small, (9, 9), 2.0)
    threshval, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return w / 2.0, h / 2.0, max(w, h) / 2.0
    biggest = max(cnts, key=cv2.contourArea)
    center, r0 = cv2.minEnclosingCircle(biggest)
    cx = center[0] * (w / 256.0)
    cy = center[1] * (h / 256.0)
    rad = r0 * ((w / 256.0 + h / 256.0) / 2.0)
    return cx, cy, rad


def disk_masking(h, w, cx, cy, rad):
    ys, xs = np.ogrid[:h, :w]
    return (xs - cx) ** 2 + (ys - cy) ** 2 <= rad * rad


def radial_flattening(img, cx, cy, rad, mask):
    h, w = img.shape
    ys, xs = np.ogrid[:h, :w]
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    maxr = int(rad)
    di = np.clip(dist.astype(np.int32), 0, maxr)
    ridx = di[mask]
    vals = img[mask].astype(np.float64)
    sums = np.bincount(ridx, weights=vals, minlength=maxr + 1)
    counts = np.bincount(ridx, minlength=maxr + 1).astype(np.float64)
    profile = (sums / np.maximum(counts, 1)).astype(np.float32)
    valid = counts > 0
    if not valid.all():
        profile[~valid] = np.interp(np.flatnonzero(~valid), np.flatnonzero(valid), profile[valid])
    padded = np.pad(profile, 10, mode="edge")
    profile = np.convolve(padded, np.ones(21) / 21, mode="valid").astype(np.float32)
    bg = profile[di]
    flat = img / (bg + 1e-6) * float(profile.mean() + 1e-6)
    return np.clip(flat, 0, 1).astype(np.float32)


def clahe(img, mask):
    filled = np.where(mask, img, img[mask].mean())
    u8 = (np.clip(filled, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return c.apply(u8).astype(np.float32) / 255.0


def off_disk_zeroing(img, mask):
    img[~mask] = 0.0
    return img


def process_one(rel):
    src = SRC / rel
    dst = DST / rel.with_suffix(".png")
    if dst.exists():
        return None
    img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return ("readfail", rel)
    gray = img.astype(np.float32) / 255.0
    h, w = gray.shape
    cx, cy, rad = solar_disk_detection(gray)
    mask = disk_masking(h, w, cx, cy, rad)
    if mask.sum() < 0.04 * h * w or rad <= 8:
        return ("smallmask", rel)
    flat = radial_flattening(gray, cx, cy, rad, mask)
    enhanced = clahe(flat, mask)
    result = off_disk_zeroing(enhanced, mask)
    out = (np.clip(result, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    dst.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(dst), out)
    return None if ok else ("writefail", rel)


def worker(files):
    fails = []
    for i, rel in enumerate(files):
        r = process_one(rel)
        if r is not None:
            fails.append(r)
        if i % 200 == 0:
            print(f"{Path(files[0]).parts[0]} worker {i}/{len(files)}", flush=True)
    return fails


def main():
    rels = []
    for ds in ("bbso", "gong", "kso"):
        rels.extend(sorted(p.relative_to(SRC) for p in (SRC / ds).rglob("*.jpg")))
    print(f"{len(rels)} images -> {DST}", flush=True)
    W = WORKERS
    chunks = [rels[i::W] for i in range(W)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(W) as pool:
        all_fails = pool.map(worker, chunks)
    fails = [f for fr in all_fails for f in fr]
    if fails:
        for tag, rel in fails:
            print(tag, rel)
    n_out = sum(1 for _ in DST.rglob("*.png"))
    print(f"done: {n_out}/{len(rels)} written, {len(fails)} fails", flush=True)


if __name__ == "__main__":
    main()
