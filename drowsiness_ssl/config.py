import random
from pathlib import Path

import numpy as np
import torch

# Image resolution used everywhere -- small enough for fast training on a laptop GPU,
# large enough to capture facial features (yawning, eye closure, nodding).
DEFAULT_IMAGE_SIZE = 64

# ImageNet mean/std -- reasonable starting point since our encoder uses a standard CNN
# architecture and the images are natural-ish (face frames from a car cabin camera).
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


def set_seed(seed):
    """Seed all random number generators so experiment results are reproducible."""
    # Lock down all the RNG sources that could affect experiment reproducibility.
    # Python's random module, NumPy, and both CPU/GPU PyTorch all need to be seeded
    # separately because they each have their own independent state.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_project_dirs(root):
    """Create the standard project folder layout (data/, checkpoints/, results/) if missing."""
    # Create the standard project folder structure if it doesn't exist yet.
    # Running this at startup means scripts never fail on a missing directory.
    for folder in ("data", "checkpoints", "results"):
        (root / folder).mkdir(parents=True, exist_ok=True)


def infer_method_and_tag(run_name: str) -> tuple[str, str]:
    """Parse a run name string into (method, tag) so the right checkpoint can be auto-located."""
    # Parse the run name to figure out which SSL method and variant we're working with.
    # This lets us auto-locate the right encoder checkpoint without needing the user
    # to pass the full path every time.
    #
    # Convention:
    #   simclr_no_color  -> (simclr, no_color)
    #   simclr_minimal   -> (simclr, minimal)
    #   simclr / simclr_full -> (simclr, full)   -- default augmentation
    #   mae / convmae    -> (convmae, mask75)     -- default mask ratio
    #   anything else    -> (cnn_scratch, scratch)
    lower = run_name.lower()
    if "simclr" in lower:
        if "no_color" in lower:
            return "simclr", "no_color"
        if "minimal" in lower:
            return "simclr", "minimal"
        if "full" in lower:
            return "simclr", "full"
        return "simclr", "full"  # default SimCLR variant
    if "convmae" in lower or "mae" in lower:
        if "mask25" in lower:
            return "convmae", "mask25"
        if "mask50" in lower:
            return "convmae", "mask50"
        if "mask75" in lower:
            return "convmae", "mask75"
        return "convmae", "mask75"  # default mask ratio
    return "cnn_scratch", "scratch"
