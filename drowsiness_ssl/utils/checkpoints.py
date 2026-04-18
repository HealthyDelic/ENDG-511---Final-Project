from pathlib import Path

import torch


def save_pretrain_checkpoint(
    checkpoint_dir: Path,
    method: str,
    tag: str,
    encoder,
    model,
    history,
    metadata,
) -> None:
    """Save the pretrained encoder in two forms: tagged and generic alias.

    We save two copies of the encoder weights:
      - {method}_{tag}_encoder.pt  e.g. simclr_full_encoder.pt    (the specific variant)
      - {method}_encoder.pt        e.g. simclr_encoder.pt          (generic alias for convenience)

    The alias makes it easy to point `--pretrained_path` at a model without knowing
    the exact variant name. The full model checkpoint (encoder + pretraining head + history)
    is also saved separately for debugging or restarting pretraining.
    """
    encoder_path = checkpoint_dir / f"{method}_{tag}_encoder.pt"
    alias_path = checkpoint_dir / f"{method}_encoder.pt"
    full_model_path = checkpoint_dir / f"{method}_{tag}_full.pt"

    # Encoder-only payload -- this is what the downstream classifier loads
    encoder_payload = {
        "encoder_state_dict": encoder.state_dict(),
        "method": method,
        "tag": tag,
        "metadata": metadata,
    }
    # Full payload -- encoder + pretraining head + per-epoch history
    full_payload = {
        "model_state_dict": model.state_dict(),
        "encoder_state_dict": encoder.state_dict(),
        "method": method,
        "tag": tag,
        "metadata": metadata,
        "history": history,
    }

    torch.save(encoder_payload, encoder_path)
    torch.save(encoder_payload, alias_path)   # overwrite alias with the latest run's weights
    torch.save(full_payload, full_model_path)
    print(f"saved_encoder={encoder_path}")
    print(f"saved_encoder_alias={alias_path}")
    print(f"saved_full_model={full_model_path}")


def load_encoder_checkpoint(path: Path, encoder, device: str = "cpu") -> None:
    """Load encoder weights from a checkpoint file into an existing encoder module."""
    # Handles both the encoder-only format (has 'encoder_state_dict' key)
    # and raw state dicts (in case someone saved the encoder weights directly).
    payload = torch.load(path, map_location=device)
    state_dict = payload.get("encoder_state_dict", payload)
    encoder.load_state_dict(state_dict)
