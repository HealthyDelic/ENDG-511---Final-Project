import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

from drowsiness_ssl.config import DEFAULT_MEAN, DEFAULT_STD


class RecursiveImageDataset(Dataset):
    """Unlabeled image dataset that walks a directory tree recursively.

    Used for SSL pretraining where we don't need class labels -- just raw images.
    Supports jpg, jpeg, png, and bmp.
    """

    def __init__(self, root: str | Path, transform=None) -> None:
        self.root = Path(root)
        self.transform = transform
        suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
        # Walk the whole tree so we pick up images nested inside subtype folders
        self.files = sorted([path for path in self.root.rglob("*") if path.suffix.lower() in suffixes])
        if not self.files:
            raise FileNotFoundError(f"No image files found under {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        image = Image.open(self.files[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image


class BinaryNestedFolderDataset(Dataset):
    """Binary image dataset with recursive subtype folders under each class folder.

    Expected layouts:

    1. root/
       |-- notdrowsy/
       |   `-- *.jpg
       `-- drowsy/
           |-- yawning/
           |-- slowBlinkWithNodding/
           `-- sleepyCombination/

    2. root/train, root/val, root/test with the same class structure inside each split.
    """

    def __init__(self, root: str | Path, transform=None) -> None:
        self.root = Path(root)
        self.transform = transform
        self.class_names = ["notdrowsy", "drowsy"]
        self.class_to_idx = {"notdrowsy": 0, "drowsy": 1}
        suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
        self.samples: list[tuple[Path, int]] = []

        for class_name in self.class_names:
            class_dir = self.root / class_name
            if not class_dir.exists():
                continue
            for path in sorted(class_dir.rglob("*")):
                if path.suffix.lower() in suffixes:
                    self.samples.append((path, self.class_to_idx[class_name]))

        if not self.samples:
            raise FileNotFoundError(f"No labeled image files found under {self.root}")

        self.targets = [label for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class ContrastivePairDataset(Dataset):
    """Wraps an unlabeled dataset and returns two differently-augmented views of the same image.

    This is exactly what SimCLR needs: each call to __getitem__ applies the stochastic
    augmentation pipeline twice independently, giving the model two views to contrast.
    """

    def __init__(self, base_dataset: Dataset, transform) -> None:
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        # Load once, augment twice -- the two views should look quite different
        image_path = self.base_dataset.files[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), self.transform(image)


@dataclass
class LabeledSplits:
    # Convenient container so callers get named access to each split
    # rather than having to unpack a bare tuple.
    train: Dataset
    val: Dataset
    test: Dataset
    class_names: list[str]


def _normalize_transform() -> transforms.Compose:
    """Return a ToTensor + ImageNet normalization transform used across all pipelines."""
    # Standard ImageNet normalization applied after converting to a tensor.
    # Reused across all the transform pipelines below.
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD),
        ]
    )


def _simclr_augment(image_size: int, variant: str):
    """Return a stochastic augmentation pipeline for SimCLR contrastive pretraining.

    Three variants let us ablate the contribution of each augmentation component:
    full, no_color (geometry only + blur), and minimal (crop + flip only).
    """
    # Three augmentation levels for the ablation study:
    #
    #   full     -- the full SimCLR recipe: crop, flip, color jitter, grayscale, blur
    #   no_color -- same geometry, but no color jitter or grayscale (tests color augmentation value)
    #   minimal  -- just crop and flip (geometric-only baseline)
    if variant == "full":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
                *_normalize_transform().transforms,
            ]
        )
    if variant == "no_color":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
                *_normalize_transform().transforms,
            ]
        )
    # minimal variant
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            *_normalize_transform().transforms,
        ]
    )


def _pretrain_transform(image_size: int):
    """Return a deterministic resize + normalize transform for ConvMAE pretraining."""
    # ConvMAE sees clean, unaugmented images -- the masking is applied inside the model.
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            *_normalize_transform().transforms,
        ]
    )


def _train_transform(image_size: int):
    """Return the supervised training transform: resize, random horizontal flip, normalize."""
    # Light augmentation for supervised training -- just a horizontal flip.
    # We don't do anything aggressive here because the label fraction is often small.
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            *_normalize_transform().transforms,
        ]
    )


def _eval_transform(image_size: int):
    """Return the evaluation transform: deterministic resize + normalize, no randomness."""
    # Deterministic preprocessing for val and test -- no randomness here.
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            *_normalize_transform().transforms,
        ]
    )


def build_pretrain_dataset(method: str, root: str | Path, image_size: int, simclr_variant: str) -> Dataset:
    """Build an unlabeled dataset ready for SSL pretraining.

    Returns a ContrastivePairDataset for SimCLR (two augmented views per image)
    or a plain RecursiveImageDataset for ConvMAE (clean images; masking is done inside the model).
    """
    # For SimCLR we need paired views, so wrap the dataset in ContrastivePairDataset.
    # For ConvMAE we just need clean images -- masking happens inside the model forward pass.
    if method == "simclr":
        base_dataset = RecursiveImageDataset(root=root, transform=None)
        return ContrastivePairDataset(base_dataset=base_dataset, transform=_simclr_augment(image_size, simclr_variant))
    return RecursiveImageDataset(root=root, transform=_pretrain_transform(image_size))


def build_labeled_splits(
    root: str | Path,
    image_size: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> LabeledSplits:
    """Load the labeled dataset and split it into train/val/test with subject-grouped partitioning.

    Handles three directory layouts automatically: pre-split (train/val/test subfolders),
    a single train/ folder (we split it), or a flat root directory.
    """
    # Support three dataset layouts:
    #   1. root/train/, root/val/, root/test/  -- pre-split by the dataset creator
    #   2. root/train/ only                   -- we'll do a subject-grouped split
    #   3. root/ directly                     -- flat structure, we split from here
    root = Path(root)
    train_dir = root / "train"
    val_dir = root / "val"
    test_dir = root / "test"

    if train_dir.exists() and val_dir.exists() and test_dir.exists():
        # Pre-split dataset -- just load each split with the right transform
        train = BinaryNestedFolderDataset(train_dir, transform=_train_transform(image_size))
        val = BinaryNestedFolderDataset(val_dir, transform=_eval_transform(image_size))
        test = BinaryNestedFolderDataset(test_dir, transform=_eval_transform(image_size))
        return LabeledSplits(train=train, val=val, test=test, class_names=train.class_names)

    if train_dir.exists():
        # Only a train folder -- perform subject-grouped split internally
        train, val, test = _split_imagefolder(train_dir, image_size, val_ratio, test_ratio, seed)
        class_names = train.dataset.class_names if isinstance(train, Subset) else train.class_names
        return LabeledSplits(train=train, val=val, test=test, class_names=class_names)

    # Fall back to treating the root itself as the dataset directory
    train, val, test = _split_imagefolder(root, image_size, val_ratio, test_ratio, seed)
    class_names = train.dataset.class_names if isinstance(train, Subset) else train.class_names
    return LabeledSplits(train=train, val=val, test=test, class_names=class_names)


def _extract_subject_id(path: Path) -> str:
    """Return the subject ID from a filename formatted as {subject}_{...}.ext."""
    return path.stem.split("_")[0]


def _compute_split_indices(
    root: Path, val_ratio: float, test_ratio: float, seed: int
) -> tuple[list[int], list[int], list[int]]:
    """Subject-grouped split into train/val/test index lists.

    All frames from a given subject stay in the same split, preventing
    any subject-level data leakage between train, val, and test.
    """
    dataset = BinaryNestedFolderDataset(root, transform=None)

    # Group sample indices by subject ID so we can split at the subject level
    subject_to_indices: dict[str, list[int]] = {}
    for idx, (path, _) in enumerate(dataset.samples):
        subject_id = _extract_subject_id(path)
        subject_to_indices.setdefault(subject_id, []).append(idx)

    subjects = np.array(sorted(subject_to_indices.keys()))
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)  # shuffle before splitting so the assignment is random but reproducible

    n_subjects = len(subjects)
    n_test = max(1, int(round(n_subjects * test_ratio)))
    n_val = max(1, int(round(n_subjects * val_ratio)))
    n_train = max(1, n_subjects - n_val - n_test)

    train_subjects = set(subjects[:n_train])
    val_subjects = set(subjects[n_train:n_train + n_val])
    test_subjects = set(subjects[n_train + n_val:])

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for subject, indices in subject_to_indices.items():
        if subject in train_subjects:
            train_indices.extend(indices)
        elif subject in val_subjects:
            val_indices.extend(indices)
        else:
            test_indices.extend(indices)

    return sorted(train_indices), sorted(val_indices), sorted(test_indices)


def _split_imagefolder(root: Path, image_size: int, val_ratio: float, test_ratio: float, seed: int):
    """Perform a subject-grouped train/val/test split and cache the result to disk.

    Reads or writes a .split_indices.json cache file so all experiments
    at the same parameters operate on identical splits.
    """
    # Cache the split indices to disk so all experiments use identical splits.
    # If the split parameters change (ratio, seed), we recompute and overwrite.
    split_cache = root / ".split_indices.json"

    if split_cache.exists():
        with split_cache.open() as f:
            cache = json.load(f)
        if (
            cache.get("method") == "subject_grouped"
            and cache.get("val_ratio") == val_ratio
            and cache.get("test_ratio") == test_ratio
            and cache.get("seed") == seed
        ):
            # Cached split matches our parameters -- reuse it
            train_indices = cache["train"]
            val_indices = cache["val"]
            test_indices = cache["test"]
        else:
            # Parameters changed -- recompute and update the cache
            train_indices, val_indices, test_indices = _compute_split_indices(
                root, val_ratio, test_ratio, seed
            )
            split_cache.write_text(json.dumps({
                "method": "subject_grouped",
                "val_ratio": val_ratio, "test_ratio": test_ratio, "seed": seed,
                "train": train_indices, "val": val_indices, "test": test_indices,
            }))
    else:
        # No cache yet -- compute and save
        train_indices, val_indices, test_indices = _compute_split_indices(
            root, val_ratio, test_ratio, seed
        )
        split_cache.write_text(json.dumps({
            "method": "subject_grouped",
            "val_ratio": val_ratio, "test_ratio": test_ratio, "seed": seed,
            "train": train_indices, "val": val_indices, "test": test_indices,
        }))

    # Instantiate three separate dataset objects so each split has its own transform
    train_dataset = BinaryNestedFolderDataset(root, transform=_train_transform(image_size))
    val_dataset = BinaryNestedFolderDataset(root, transform=_eval_transform(image_size))
    test_dataset = BinaryNestedFolderDataset(root, transform=_eval_transform(image_size))
    return Subset(train_dataset, train_indices), Subset(val_dataset, val_indices), Subset(test_dataset, test_indices)


def _get_dataset_root(dataset: Dataset) -> Path | None:
    """Walk Subset wrappers to find the root of the underlying BinaryNestedFolderDataset."""
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return getattr(dataset, "root", None)


def subset_dataset_by_fraction(dataset: Dataset, fraction: float, seed: int) -> Dataset:
    """Return a stratified random subset containing `fraction` of each class."""
    if fraction >= 1.0:
        return dataset

    # Stratified sampling: keep the same class ratio at any fraction size.
    # This matters most at very low fractions (e.g. 5%) where one class could
    # otherwise dominate just by chance.
    labels = np.array(_extract_labels(dataset))
    indices = np.arange(len(dataset))
    rng = np.random.default_rng(seed)
    keep_indices: list[int] = []

    for cls in np.unique(labels):
        cls_indices = indices[labels == cls].copy()
        rng.shuffle(cls_indices)
        keep = max(1, int(round(len(cls_indices) * fraction)))  # keep at least 1 sample per class
        keep_indices.extend(cls_indices[:keep].tolist())

    keep_indices.sort()
    return Subset(dataset, keep_indices)


def build_shared_train_subset(train_dataset: Dataset, label_frac: float, seed: int) -> Dataset:
    """Stratified fraction subset with a persistent cache.

    All models that call this with the same dataset root, label_frac, and seed
    are guaranteed to train on the exact same labeled examples.
    The cache is stored as ``<data_root>/.fraction_subsets.json``.
    """
    if label_frac >= 1.0:
        return train_dataset

    # Locate the data root so we know where to write the cache file
    root = _get_dataset_root(train_dataset)
    cache_path = (root / ".fraction_subsets.json") if root is not None else None
    cache_key = f"{label_frac:.4f}_{seed}"

    # Return the cached indices if they exist for this (fraction, seed) combination
    if cache_path is not None and cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        if cache_key in cache:
            return Subset(train_dataset, cache[cache_key])
    else:
        cache = {}

    # No cache hit -- compute a fresh stratified subset
    labels = np.array(_extract_labels(train_dataset))
    indices = np.arange(len(train_dataset))
    rng = np.random.default_rng(seed)
    keep_indices: list[int] = []

    for cls in np.unique(labels):
        cls_indices = indices[labels == cls].copy()
        rng.shuffle(cls_indices)
        keep = max(1, int(round(len(cls_indices) * label_frac)))
        keep_indices.extend(cls_indices[:keep].tolist())

    keep_indices.sort()

    # Save to cache so subsequent experiments at the same fraction see the same images
    if cache_path is not None:
        if cache_path.exists():
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        cache[cache_key] = keep_indices
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    return Subset(train_dataset, keep_indices)


def _extract_labels(dataset: Dataset) -> Sequence[int]:
    """Recursively extract integer class labels from a dataset, unwrapping any Subset wrappers."""
    # Recursively unwrap Subset wrappers to pull out the underlying target list.
    # This lets us do stratified sampling even when the dataset is already a Subset.
    if isinstance(dataset, Subset):
        base_labels = _extract_labels(dataset.dataset)
        return [base_labels[index] for index in dataset.indices]
    if hasattr(dataset, "targets"):
        return list(dataset.targets)
    raise ValueError("Dataset does not expose targets for stratified subsampling.")
