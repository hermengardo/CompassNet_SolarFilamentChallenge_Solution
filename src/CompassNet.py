import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torchvision.models import ResNet34_Weights

class KirschFilters(nn.Module):
    """Compute positive and negative Kirsch responses at each configured scale.

    The fixed eight-direction filter bank provides the directional responses
    later aggregated into the multi-scale orientation field.

    Args:
        dilations: Spatial dilation rates applied to the filter bank.
    """
    def __init__(self, dilations=(1, 2, 4)):
        super().__init__()

        kernels = torch.tensor(
            [
                [
                    [5, 5, 5],
                    [-3, 0, -3],
                    [-3, -3, -3],
                ],
                [
                    [-3, 5, 5],
                    [-3, 0, 5],
                    [-3, -3, -3],
                ],
                [
                    [-3, -3, 5],
                    [-3, 0, 5],
                    [-3, -3, 5],
                ],
                [
                    [-3, -3, -3],
                    [-3, 0, 5],
                    [-3, 5, 5],
                ],
                [
                    [-3, -3, -3],
                    [-3, 0, -3],
                    [5, 5, 5],
                ],
                [
                    [-3, -3, -3],
                    [5, 0, -3],
                    [5, 5, -3],
                ],
                [
                    [5, -3, -3],
                    [5, 0, -3],
                    [5, -3, -3],
                ],
                [
                    [5, 5, -3],
                    [5, 0, -3],
                    [-3, -3, -3],
                ],
            ],
            dtype=torch.float32,
        ).unsqueeze(1) / 15

        self.register_buffer("kernels", kernels)
        self.dilations = dilations

    def forward(self, image):
        responses = []

        for dilation in self.dilations:
            padded = F.pad(
                image,
                (dilation,) * 4,
                mode="replicate"
            )

            response = F.conv2d(
                padded,
                self.kernels,
                dilation=dilation
            )

            positive = F.relu(response)
            negative = F.relu(-response)

            responses.append(
                torch.cat([positive, negative], dim=1)
            )

        return torch.cat(responses, dim=1)


class OrientationFieldExtractor(nn.Module):
    """Aggregate Kirsch responses into per-scale orientation moments.

    For each scale, combines the eight directions into two doubled-angle
    components and a response-energy map. ``normalize`` converts these
    moments into direction components and log-magnitude.

    Args:
        num_scales: Number of scales represented in the response tensor.
        tau: Stabilizer used when normalizing the direction components.
    """
    def __init__(self,
                 num_scales=3,
                 tau=0.1):
        super().__init__()

        k = torch.arange(8, dtype=torch.float32) * (torch.pi / 4)

        self.register_buffer(
            "cosines",
            torch.cos(2 * k).view(1, 8, 1, 1)
        )

        self.register_buffer(
            "sines",
            torch.sin(2 * k).view(1, 8, 1, 1)
        )

        self.num_scales = num_scales
        self.tau = tau

    def forward(self, responses):
        B, C, H, W = responses.shape

        responses = responses.reshape(B, self.num_scales, 16, H, W)

        positive, negative = responses.split(8, dim=2)
        weights = positive + negative

        vx = (weights * self.cosines.unsqueeze(1)).sum(dim=2)
        vy = (weights * self.sines.unsqueeze(1)).sum(dim=2)
        energy = weights.sum(dim=2)

        out = (
            torch.stack(
                [vx, vy, energy],
                dim=2
            ).reshape(B, self.num_scales * 3, H, W)
        )

        return out

    def normalize(self, moments):
        B, C, H, W = moments.shape

        moments = moments.reshape(
            B, self.num_scales, 3, H, W
        )

        vx, vy, energy = moments.unbind(dim=2)

        direction_x = vx / (energy + self.tau)
        direction_y = vy / (energy + self.tau)
        magnitude = torch.log1p(energy)

        orientation = torch.stack(
            [direction_x,
             direction_y,
             magnitude],
            dim=2
        )

        return orientation.reshape(B, self.num_scales * 3, H, W)


class CompassAttention(nn.Module):
    """Apply windowed self-attention with an orientation-dependent bias.

    Args:
        dim: Number of input and output feature channels.
        num_heads: Number of attention heads.
        window_size: Side length of each non-overlapping attention window.
        orientation_dim: Number of orientation input channels.
        num_scales: Number of orientation scales.
    """

    def __init__(self,
                 dim,
                 num_heads=8,
                 window_size=8,
                 orientation_dim=9,
                 num_scales=3):
        super().__init__()

        self.window_size = window_size
        self.num_heads = num_heads
        self.num_scales = num_scales
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**(-0.5)

        self.extractor = OrientationFieldExtractor(num_scales=num_scales)

        self.pos = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.orientation_proj = nn.Conv2d(orientation_dim, dim, 1)

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.attention_proj = nn.Linear(dim, dim)

        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        self.bias_scale = nn.Parameter(torch.ones(num_heads))

        coords = torch.stack(
            torch.meshgrid(
                torch.arange(window_size),
                torch.arange(window_size),
                indexing="ij",
            ),
            dim=-1,
        ).float().reshape(-1, 2).flip(-1)

        delta = coords.unsqueeze(0) - coords.unsqueeze(1)
        delta = delta / torch.linalg.vector_norm(
            delta, dim=-1, keepdim=True
        ).clamp_min(1e-6)

        self.register_buffer("relative_direction", delta)

    def directional_bias(self, orientation_moments):
        B, C, H, W = orientation_moments.shape

        moments = orientation_moments.reshape(
            B, self.num_scales, 3, H, W
        )

        vx, vy, energy = moments.unbind(dim=2)

        vx = vx.sum(dim=1)
        vy = vy.sum(dim=1)
        energy = energy.sum(dim=1)

        double = F.normalize(
            torch.stack([vx, vy], dim=1), dim=1, eps=1e-6
        )

        confidence = (
            torch.sqrt(vx.square() + vy.square()) / (energy + 1e-6)
        ).clamp(0, 1).unsqueeze(1)

        double_windows = partition_windows(double, self.window_size)
        confidence_windows = partition_windows(
            confidence, self.window_size
        ).squeeze(-1)

        pair_double = F.normalize(
            double_windows.unsqueeze(2) + double_windows.unsqueeze(1),
            dim=-1, eps=1e-6,
        )

        rel = self.relative_direction.to(
            device=double.device, dtype=double.dtype
        )
        dx, dy = rel[..., 0], rel[..., 1]
        rel_double = torch.stack([dx * dx - dy * dy, 2 * dx * dy], dim=-1)

        alignment_sq = 0.5 * (
            1 + (pair_double * rel_double.unsqueeze(0)).sum(dim=-1)
        )

        pair_confidence = (
            confidence_windows.unsqueeze(2)
            + confidence_windows.unsqueeze(1)
        ) * 0.5

        bias = alignment_sq * pair_confidence

        return (
            bias.unsqueeze(1)
            * self.bias_scale.view(1, self.num_heads, 1, 1)
        )

    def forward(self, features, orientation_moments):
        identity = features
        B, C, H, W = features.shape
        window_size = self.window_size

        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size

        features = F.pad(features, (0, pad_w, 0, pad_h))
        Hp, Wp = features.shape[-2:]

        orientation_moments = ADPM(
            orientation_moments, (H, W), self.num_scales
        )
        orientation_moments = F.pad(orientation_moments, (0, pad_w, 0, pad_h))

        orientation = self.extractor.normalize(orientation_moments)

        features = features + self.pos(features)
        Z = features + self.orientation_proj(orientation)

        tokens = partition_windows(Z, window_size)
        normalized = self.norm1(tokens)

        Bn, N, _ = normalized.shape

        qkv = self.qkv(normalized).reshape(
            Bn, N, 3,
            self.num_heads,
            self.head_dim
        ).permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(dim=0)

        logits = (q @ k.transpose(-2, -1)) * self.scale
        logits = logits + self.directional_bias(orientation_moments)

        attention = logits.softmax(dim=-1)
        attended = attention @ v

        attended = attended.transpose(1, 2).reshape(Bn, N, C)

        tokens = tokens + self.attention_proj(attended)
        tokens = tokens + self.mlp(self.norm2(tokens))

        output = reverse_windows(tokens, B, Hp, Wp, window_size)
        output = self.proj(output)
        output = output[:, :, :H, :W]

        return identity + output


class OrientationConditioning(nn.Module):
    """Condition decoder features on the multi-scale orientation field.

    Args:
        orientation_dim: Number of input orientation channels.
        channels: Number of feature channels to produce.
        num_scales: Number of orientation scales.
    """
    def __init__(self, orientation_dim, channels, num_scales=3):
        super().__init__()

        self.num_scales = num_scales
        self.proj = nn.Sequential(
            nn.Conv2d(orientation_dim, channels, 1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
        )

        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, features, orientation_moments):
        moments = ADPM(
            orientation_moments,
            features.shape[-2:],
            self.num_scales,
        )

        B, C, H, W = moments.shape
        moments = moments.reshape(B, self.num_scales, 3, H, W)

        vx, vy, energy = moments.unbind(dim=2)

        orientation = torch.stack(
            [
                vx / (energy + 0.1),
                vy / (energy + 0.1),
                torch.log1p(energy),
            ],
            dim=2,
        ).reshape(B, self.num_scales * 3, H, W)

        out = features + self.alpha * self.proj(orientation)

        return out


class KirschCompass(nn.Module):
    """Convert an input image into raw and normalized orientation fields.

    Args:
        dilations: Spatial dilation rates used by the Kirsch filters.
    """
    def __init__(self, dilations=(1, 2, 4)):
        super().__init__()
        D = len(dilations)
        self.filters = KirschFilters(dilations)
        self.extractor = OrientationFieldExtractor(num_scales=D)
        self.out_channels = 3 * len(dilations)

    def moments(self, image):
        with torch.autocast(device_type=image.device.type, enabled=False):
            image = image.float()

            mean = image.mean(dim=(-2, -1), keepdim=True)
            std = image.std(
                dim=(-2, -1), keepdim=True, unbiased=False
            ).clamp_min(1e-6)

            image = (image - mean) / std
            responses = self.filters(image)

            return self.extractor(responses)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        moments = self.moments(image)
        return self.extractor.normalize(moments)


class CompassNet(nn.Module):
    """Orientation-Aware U-Net with ResNet34 backbone.

    Args:
        dropout: Dropout probability applied to the deepest encoder features.
        weights_path: ``"native"`` for ImageNet weights, a checkpoint path,
            or ``None`` for randomly initialized backbone weights.
        num_heads: Number of heads in the orientation-guided attention.
        dilations: Spatial scales used by the Kirsch filter bank.
        spine: Whether to return an auxiliary spine prediction.
        freeze_bn: Whether to freeze the encoder's BatchNorm layers (for small batch sizes)

    Returns:
        Segmentation logits with one channel, or two when ``spine=True``.
    """
    def __init__(
        self,
        dropout: float = 0.05,
        weights_path: str | None = None,
        num_heads: int = 8,
        dilations=(1, 2, 4),
        spine: bool = False,
        freeze_bn: bool = True
    ):
        super().__init__()

        self.spine = spine
        self.freeze_bn = freeze_bn

        if weights_path == "native":
            weights = ResNet34_Weights.IMAGENET1K_V1
        else:
            weights = None

        backbone = torchvision.models.resnet34(weights=weights)

        self.kirsch = KirschCompass(dilations)
        D = self.kirsch.out_channels
        num_scales = len(dilations)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.compass_attention = CompassAttention(
            256,
            num_heads=num_heads,
            window_size=8,
            orientation_dim=D,
            num_scales=num_scales,
        )

        self.dropout = nn.Dropout2d(dropout)

        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 64)
        self.dec0 = DecoderBlock(64, 1 + D, 64)

        self.orient4 = OrientationConditioning(D, 256, num_scales)
        self.orient3 = OrientationConditioning(D, 128, num_scales)
        self.orient2 = OrientationConditioning(D, 64, num_scales)
        self.orient1 = OrientationConditioning(D, 64, num_scales)

        self.head = nn.Conv2d(64, 1, 1)

        if self.spine:
            self.spine_head = (
                nn.Sequential(CNR_block(64, 64), nn.Conv2d(64, 1, 1))
            )

        if weights_path is not None and weights_path != "native":
            self.load(weights_path)

        if self.freeze_bn:
            self.freeze_backbone_bn()

    def freeze_backbone_bn(self):
        for m in [self.stem,
                  self.layer1,
                  self.layer2,
                  self.layer3,
                  self.layer4]:
            for child in m.modules():
                if isinstance(child, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    child.eval()
                    for p in child.parameters():
                        p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_bn:
            self.freeze_backbone_bn()
        return self

    def load(self, weights_path: str):
        state_dict = torch.load(weights_path, map_location="cpu")
        model_state = self.state_dict()
    
        state_dict = {
            k: v for k, v in state_dict.items()
            if k in model_state and v.shape == model_state[k].shape
        }
    
        self.load_state_dict(state_dict, strict=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        orientation_moments = self.kirsch.moments(image)
        orientation = self.kirsch.extractor.normalize(orientation_moments)

        backbone_input = (image - 0.449) / 0.226
        backbone_input = backbone_input.repeat(1, 3, 1, 1)

        stem_features = self.stem(backbone_input)
        layer1_features = self.layer1(self.pool(stem_features))
        layer2_features = self.layer2(layer1_features)
        layer3_features = self.layer3(layer2_features)

        layer3_features = self.compass_attention(
            layer3_features, orientation_moments
        )

        deep_features = self.layer4(layer3_features)
        features = self.dropout(deep_features)

        features = self.dec4(features, layer3_features)
        features = self.orient4(features, orientation_moments)

        features = self.dec3(features, layer2_features)
        features = self.orient3(features, orientation_moments)

        features = self.dec2(features, layer1_features)
        features = self.orient2(features, orientation_moments)

        features = self.dec1(features, stem_features)
        features = self.orient1(features, orientation_moments)

        full_resolution_input = torch.cat([image, orientation], dim=1)
        features = self.dec0(features, full_resolution_input)

        mask = self.head(features)

        if self.spine:
            spine = self.spine_head(features)
            return torch.cat([mask, spine], dim=1)

        return mask


def CNR_block(in_channels: int, out_channels: int, dilation: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
        nn.GroupNorm(8, out_channels),
        nn.ReLU(inplace=True),
    )

class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            CNR_block(in_channels + skip_channels, out_channels),
            CNR_block(out_channels, out_channels),
        )

    def forward(self,
                features: torch.Tensor,
                skip: torch.Tensor) -> torch.Tensor:
        upsampled = F.interpolate(
            features, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )

        return self.conv(torch.cat([upsampled, skip], dim=1))

def ADPM(moments, size, num_scales):
    B, C, H, W = moments.shape
    moments = moments.reshape(B, num_scales * 3, H, W)
    return F.adaptive_avg_pool2d(moments, size)


def partition_windows(x, window_size):
    B, C, H, W = x.shape

    return (
        x.reshape(B, C, H // window_size, window_size, W // window_size, window_size)
        .permute(0, 2, 4, 3, 5, 1)
        .reshape(-1, window_size * window_size, C)
    )


def reverse_windows(windows, batch_size, height, width, window_size):
    C = windows.shape[-1]

    return (
        windows.reshape(
            batch_size, height // window_size, width // window_size, window_size, window_size, C
        )
        .permute(0, 5, 1, 3, 2, 4)
        .reshape(batch_size, C, height, width)
    )
    
