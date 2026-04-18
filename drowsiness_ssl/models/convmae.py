import torch
from torch import nn
from torch.nn import functional as F


def apply_patch_mask(images: torch.Tensor, patch_size: int, mask_ratio: float):
    """Zero out a random subset of patches in each image.

    Divides the image into a grid of non-overlapping patches and randomly
    selects mask_ratio of them to blank out (set to zero). Returns both
    the masked images and a boolean patch-level mask that the loss function
    uses to ignore the unmasked patches.
    """
    batch_size, channels, height, width = images.shape
    grid_h = height // patch_size
    grid_w = width // patch_size
    total_patches = grid_h * grid_w
    num_mask = max(1, int(total_patches * mask_ratio))  # mask at least one patch

    # Build a boolean mask at patch resolution, then expand to pixel resolution
    patch_mask = torch.zeros(batch_size, total_patches, device=images.device, dtype=torch.bool)
    for batch_idx in range(batch_size):
        # Pick which patches to mask for this image -- random and independent per image
        indices = torch.randperm(total_patches, device=images.device)[:num_mask]
        patch_mask[batch_idx, indices] = True

    # Reshape from flat patch list back to the 2D grid
    patch_mask = patch_mask.view(batch_size, grid_h, grid_w)
    # Expand from patch-level to pixel-level by repeating each patch's flag across its pixels
    pixel_mask = patch_mask.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
    pixel_mask = pixel_mask.unsqueeze(1).expand(-1, channels, -1, -1)

    masked_images = images.clone()
    masked_images[pixel_mask] = 0.0  # zero-fill the masked regions
    return masked_images, patch_mask


class ConvMAEModel(nn.Module):
    """Convolutional Masked Autoencoder (ConvMAE).

    The model works by:
      1. Randomly masking a fraction of patches in the input image
      2. Passing the masked image through the CNN encoder
      3. Decoding the encoder features back to the original image resolution
      4. Computing reconstruction loss only on the masked patches

    The intuition is that the encoder is forced to learn rich visual representations
    in order to fill in the missing regions from context alone.

    The decoder is a simple stack of conv + upsample blocks -- it doesn't need to be
    sophisticated because its only job during pretraining is to provide a training signal.
    It gets thrown away after pretraining.
    """

    def __init__(self, encoder: nn.Module, input_channels: int = 3, patch_size: int = 8, mask_ratio: float = 0.75) -> None:
        super().__init__()
        self.encoder = encoder
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        feature_dim = getattr(encoder, "feature_dim", 256)
        # The decoder progressively upsamples back to the input resolution.
        # Three 2x upsample stages recover the 8x spatial downsampling from the encoder.
        self.decoder = nn.Sequential(
            nn.Conv2d(feature_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, input_channels, kernel_size=3, padding=1),  # output: (B, 3, H, W)
        )

    def forward(self, images: torch.Tensor):
        """Mask patches, encode, decode, and return a dict with reconstruction, target, and mask."""
        masked_images, patch_mask = apply_patch_mask(images, self.patch_size, self.mask_ratio)
        # Encoder sees only the masked (partially zeroed) image
        features = self.encoder.forward_features(masked_images)
        reconstruction = self.decoder(features)
        # Final interpolation handles any rounding mismatch between encoder output and input size
        reconstruction = F.interpolate(reconstruction, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return {
            "reconstruction": reconstruction,  # what the model thinks the original looked like
            "target": images,                  # the actual original (ground truth)
            "masked_images": masked_images,    # what was fed to the encoder (useful for debugging)
            "patch_mask": patch_mask,          # which patches were masked (used in loss)
        }
