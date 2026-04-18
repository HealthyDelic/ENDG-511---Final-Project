import torch
from torch import nn


class ConvBlock(nn.Module):
    """A single conv -> BN -> ReLU block.

    Keeping this separate makes the encoder definition easier to read
    and lets us reuse the same pattern for every layer without repeating the three-line stack.
    """

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.block = nn.Sequential(
            # bias=False because BatchNorm already handles the bias shift
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Pass x through the conv-BN-ReLU block."""
        return self.block(x)


class SmallCNNEncoder(nn.Module):
    """Lightweight CNN encoder shared across all experiments.

    Four strided ConvBlocks progressively downsample the spatial dimensions
    while expanding the channel count. The final adaptive pool squashes whatever
    spatial size comes out into a fixed 1x1, so the encoder works regardless of
    input resolution -- useful when changing image_size without rewriting the model.

    At image_size=64, the feature map before pooling is 8x8x256.
    After pooling and flatten: a 256-d vector per image.
    """

    def __init__(self, in_channels: int = 3, feature_dim: int = 256) -> None:
        super().__init__()
        self.feature_dim = feature_dim  # stored so downstream code (ConvMAE decoder) can read it
        self.backbone = nn.Sequential(
            ConvBlock(in_channels, 32, stride=1),   # 64x64 -> 64x64
            ConvBlock(32, 64, stride=2),             # 64x64 -> 32x32
            ConvBlock(64, 128, stride=2),            # 32x32 -> 16x16
            ConvBlock(128, feature_dim, stride=2),   # 16x16 -> 8x8
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))    # 8x8 -> 1x1

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the spatial feature map (before global pooling) for use by the ConvMAE decoder."""
        # Returns the spatial feature map (before pooling).
        # ConvMAE's decoder needs this to reconstruct the image.
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the full forward pass and return a flat feature vector per image."""
        # Full forward pass: backbone -> global pool -> flatten to a feature vector.
        # This is what the classifier and SimCLR projector receive.
        feature_map = self.forward_features(x)
        pooled = self.pool(feature_map)
        return pooled.flatten(1)
