import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from Config import CFG

random.seed(CFG.seed)

class SunDataset(Dataset):
    def __init__(self,
             metadata: pd.DataFrame,
             preprocess: Optional[Callable] = None,
             augmentation: Optional[Callable] = None,
             image_dir: str = CFG.train,
             resolution: int = CFG.resolution,
             spine: bool = False,
             spine_thickness: int = 5) -> None:

        self.image_dir = Path(image_dir)
        self.preprocess = preprocess
        self.augmentation = augmentation
        self.resolution = resolution
        self.spine = spine
        self.spine_thickness = spine_thickness
        self.files = list(metadata["file_name"].unique())
    
        self.spines = (
            metadata.groupby("file_name")["spine"].agg(list).to_dict()
            if spine else {}
        )
    
        self.masks = defaultdict(list)
        annotations = (
            metadata
            .groupby(["file_name", "image_id"])["segmentation"]
            .agg(list)
        )
        for (filename, _), segmentations in annotations.items():
            self.masks[filename].append(segmentations)
    
    def __getitem__(self, idx):
        filename = self.files[idx]
        path = self.image_dir / filename
    
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0
    
        height, width = img.shape
    
        if self.preprocess is not None:
            img = self.preprocess(img)
    
        size = (self.resolution, self.resolution)
        if (width, height) != size:
            img = cv2.resize(
                img,
                size,
                interpolation=cv2.INTER_AREA
            )
            
        scale_x = self.resolution / width
        scale_y = self.resolution / height
    
        annotators = self.masks[filename]
        mask = np.zeros(size, dtype=np.float32)
    
        for segmentations in annotators:
            annot_mask = np.zeros(size, dtype=np.float32)
    
            for segmentation in segmentations:
                for poly in segmentation:
                    pts = (
                        np.array(poly, dtype=np.float32).reshape(-1, 2)
                        * (scale_x, scale_y)
                    )
                    cv2.fillPoly(
                        annot_mask,
                        [np.round(pts).astype(np.int32)],
                        1
                    )
    
            mask += annot_mask
    
        mask /= len(annotators)
    
        img = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        if self.spine:
            spine = self.spine_mask(filename, size, scale_x, scale_y)
            mask = torch.cat([mask, torch.from_numpy(spine).unsqueeze(0)])
        
        if self.augmentation is not None:
            img, mask = self.augmentation(img, mask)
        
        return img, mask

    def __len__(self):
        return len(self.files)

    def spine_mask(self, filename, size, scale_x, scale_y):
        canvas = np.zeros(size, dtype=np.float32)
    
        for spine in self.spines[filename]:
            pts = np.array(spine, dtype=np.float32).reshape(-1, 2) * (scale_x, scale_y)
            cv2.polylines(
                canvas,
                [np.round(pts).astype(np.int32)],
                False,
                1.0,
                self.spine_thickness,
            )
    
        return canvas