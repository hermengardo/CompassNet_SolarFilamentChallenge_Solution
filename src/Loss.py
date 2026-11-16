import torch
import torch.nn as nn

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = (1 - bce_weight)
        self.smooth = smooth

    def forward(self, logits, target):
        bce = self.bce(logits, target)

        probs = torch.sigmoid(logits)
        dims = tuple(range(1, probs.ndim))
        intersection = (probs * target).sum(dims)
        cardinality = probs.sum(dims) + target.sum(dims)
        dice = 1 - (2 * intersection + self.smooth) / (cardinality + self.smooth)

        return self.bce_weight * bce + self.dice_weight * dice.mean()

class SegSpineLoss(nn.Module):
    def __init__(self, spine_weight=0.1, seg_only=False):
        super().__init__()
        self.seg_loss = BCEDiceLoss()
        self.spine_loss = BCEDiceLoss()
        self.spine_weight = spine_weight
        self.seg_only = seg_only

    def forward(self, output, target):
        seg = self.seg_loss(output[:, :1], target[:, :1])
        
        if self.seg_only:
            return seg
            
        spine = self.spine_loss(output[:, 1:], target[:, 1:])
        return seg + self.spine_weight * spine