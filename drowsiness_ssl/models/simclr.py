import torch
from torch import nn
from torch.nn import functional as F


class ProjectionHead(nn.Module):
    """Two-layer MLP that projects encoder features into the contrastive loss space.

    SimCLR keeps the projection head separate from the encoder so that
    after pretraining we can discard the head and use the encoder features directly.
    The head learns a representation tuned for the NT-Xent loss, which may not
    generalize as well as the raw encoder features for downstream classification.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project encoder features into the contrastive loss space."""
        return self.layers(x)


class SimCLRModel(nn.Module):
    """SimCLR wrapper: encoder + projection head.

    Forward returns both the raw encoder features and the L2-normalized projections.
    Training uses the projections for the NT-Xent loss.
    After pretraining, the encoder features are what gets saved to the checkpoint.
    """

    def __init__(self, encoder: nn.Module, feature_dim: int, projection_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.projector = ProjectionHead(feature_dim, feature_dim, projection_dim)

    def forward(self, x: torch.Tensor):
        """Encode the input and return (raw features, L2-normalized projections)."""
        features = self.encoder(x)
        projections = self.projector(features)
        # L2 normalize so the dot product in NT-Xent is equivalent to cosine similarity
        projections = F.normalize(projections, dim=1)
        return features, projections
