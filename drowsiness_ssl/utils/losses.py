import torch
from torch.nn import functional as F


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """Normalized Temperature-scaled Cross-Entropy loss (NT-Xent) used by SimCLR.

    For a batch of N images, we have 2N embeddings (two augmented views each).
    The goal: for each embedding, the other view of the same image should score
    highest in a softmax over all the other 2N-1 embeddings.

    Implementation follows the SimCLR paper:
      - Concatenate both views into one 2N batch
      - Compute the full NxN cosine similarity matrix (divided by temperature)
      - Mask out the self-similarity diagonal (a sample shouldn't match itself)
      - The positive pair for sample i is sample i+N (and vice versa)
      - Use cross-entropy to push the model toward those positive pairs
    """
    batch_size = z1.shape[0]
    # Stack both views into a single 2N tensor so we can compute all pairwise similarities at once
    z = torch.cat([z1, z2], dim=0)
    similarity = torch.matmul(z, z.T) / temperature

    # Mask out diagonal (self-similarity) -- a sample should not be its own positive
    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    similarity = similarity.masked_fill(mask, -1e9)

    # Targets: sample i (from z1) pairs with sample i+batch_size (from z2), and vice versa
    targets = torch.arange(batch_size, device=z.device)
    targets = torch.cat([targets + batch_size, targets], dim=0)
    return F.cross_entropy(similarity, targets)


def masked_reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    patch_mask: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    """MSE loss computed only over the masked (missing) patches.

    We don't penalize reconstruction of the visible patches -- the model has
    all the information it needs to reconstruct those, so including them in the
    loss would just add noise. Only the masked patches require genuine prediction.
    """
    # Expand the patch-level mask to pixel level so we can apply it to the image tensor
    pixel_mask = patch_mask.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
    pixel_mask = pixel_mask.unsqueeze(1).expand_as(target).float()
    squared_error = (reconstruction - target) ** 2
    masked_error = squared_error * pixel_mask
    # Normalize by the number of masked pixels (not total pixels) so the scale
    # stays consistent regardless of the mask ratio
    return masked_error.sum() / pixel_mask.sum().clamp_min(1.0)
