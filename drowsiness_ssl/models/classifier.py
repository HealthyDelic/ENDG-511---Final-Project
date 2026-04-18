from torch import nn


class EncoderClassifier(nn.Module):
    """Encoder + linear classification head.

    The design is intentionally minimal: the encoder is plugged in as-is, then
    a small head (dropout + linear) maps the feature vector to class logits.

    Two modes:
      freeze_encoder=True  -- only the head trains (linear probe)
      freeze_encoder=False -- the entire network trains end-to-end (fine-tuning)
    """

    def __init__(self, encoder, feature_dim, num_classes, freeze_encoder):
        super().__init__()
        self.encoder = encoder

        if freeze_encoder:
            # Freeze all encoder parameters so gradients don't flow back into them.
            # This lets us test what the pretrained representations are worth on their own.
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        # Dropout before the linear layer helps prevent the small head from overfitting,
        # especially at low label fractions where there aren't many examples to learn from.
        self.head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x):
        """Encode the input and return raw class logits."""
        features = self.encoder(x)
        return self.head(features)
