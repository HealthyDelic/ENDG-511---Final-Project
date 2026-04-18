import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from drowsiness_ssl.config import DEFAULT_IMAGE_SIZE, ensure_project_dirs, set_seed
from drowsiness_ssl.data import build_pretrain_dataset
from drowsiness_ssl.models.convmae import ConvMAEModel
from drowsiness_ssl.models.encoder import SmallCNNEncoder
from drowsiness_ssl.models.simclr import SimCLRModel
from drowsiness_ssl.utils.checkpoints import save_pretrain_checkpoint
from drowsiness_ssl.utils.train import run_convmae_epoch, run_simclr_epoch


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the SSL pretraining script."""
    parser = argparse.ArgumentParser(description="Self-supervised pretraining for driver drowsiness detection.")

    # Which SSL method to run
    parser.add_argument("--method", choices=["simclr", "convmae"], required=True)

    # Paths
    parser.add_argument("--data_dir", type=str, default="data/Labeled/train")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")

    # Image and model settings
    parser.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--feature_dim", type=int, default=256)         # encoder output size
    parser.add_argument("--projection_dim", type=int, default=128)      # SimCLR projection head output

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)

    # SimCLR-specific
    parser.add_argument("--temperature", type=float, default=0.5)       # NT-Xent temperature
    parser.add_argument("--simclr_variant", choices=["full", "no_color", "minimal"], default="full")

    # ConvMAE-specific
    parser.add_argument("--mask_ratio", type=float, default=0.75)       # fraction of patches to mask
    parser.add_argument("--patch_size", type=int, default=8)

    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Entry point: build the dataset, run SSL pretraining, and save the encoder checkpoint."""
    args = parse_args()
    set_seed(args.seed)
    ensure_project_dirs(Path.cwd())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the pretraining dataset -- unlabeled, no class information needed here
    dataset = build_pretrain_dataset(
        method=args.method,
        root=args.data_dir,
        image_size=args.image_size,
        simclr_variant=args.simclr_variant,
    )

    # drop_last=True avoids incomplete batches which would mess up the NT-Xent batch pairing
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    # The encoder is shared between both SSL methods -- only the pretraining wrapper differs
    encoder = SmallCNNEncoder(in_channels=3, feature_dim=args.feature_dim).to(device)

    if args.method == "simclr":
        model = SimCLRModel(
            encoder=encoder,
            feature_dim=args.feature_dim,
            projection_dim=args.projection_dim,
        ).to(device)

    else:
        model = ConvMAEModel(
            encoder=encoder,
            input_channels=3,
            patch_size=args.patch_size,
            mask_ratio=args.mask_ratio,
        ).to(device)

    # AdamW on all model parameters -- both encoder and the pretraining head/decoder
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: list[dict[str, float]] = []
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        if args.method == "simclr":
            train_loss = run_simclr_epoch(
                model=model,
                loader=loader,
                optimizer=optimizer,
                device=device,
                temperature=args.temperature,
            )

        else:
            train_loss = run_convmae_epoch(
                model=model,
                loader=loader,
                optimizer=optimizer,
                device=device,
            )

        history.append({"epoch": epoch, "train_loss": train_loss})
        print(f"epoch={epoch:03d} method={args.method} train_loss={train_loss:.4f}", flush=True)
        best_loss = min(best_loss, train_loss)

    # Save encoder weights after all epochs complete.
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # The tag identifies the specific variant so we can have multiple checkpoints side-by-side
    if args.method == "simclr":
        tag = args.simclr_variant

    else:
        tag = f"mask{int(args.mask_ratio * 100):02d}"  # e.g. mask75

    save_pretrain_checkpoint(
        checkpoint_dir=checkpoint_dir,
        method=args.method,
        tag=tag,
        encoder=model.encoder,
        model=model,
        history=history,
        metadata={
            "args": vars(args),
            "best_loss": best_loss,
            "device": str(device),
        },
    )

    summary_path = checkpoint_dir / f"{args.method}_{tag}_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "method": args.method,
                "tag": tag,
                "best_loss": best_loss,
                "history": history,
            },
            handle,
            indent=2,
        )

    print(f"saved_summary={summary_path}")


if __name__ == "__main__":
    main()
