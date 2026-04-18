# Standard library imports
import argparse
import json
from pathlib import Path
from time import perf_counter

# PyTorch
import torch
from torch.utils.data import DataLoader

# Project-level imports -- config, data pipeline, model definitions, and utilities
from drowsiness_ssl.config import DEFAULT_IMAGE_SIZE, ensure_project_dirs, infer_method_and_tag, set_seed
from drowsiness_ssl.data import build_labeled_splits, build_shared_train_subset
from drowsiness_ssl.models.classifier import EncoderClassifier
from drowsiness_ssl.models.encoder import SmallCNNEncoder
from drowsiness_ssl.utils.checkpoints import load_encoder_checkpoint
from drowsiness_ssl.utils.metrics import plot_confusion_matrix, plot_roc_curve, summarize_classification
from drowsiness_ssl.utils.train import evaluate_classifier, train_classifier_epoch


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the classifier training and evaluation script."""
    # All the knobs you'd realistically want to tune from the command line.
    # Defaults are set to match the experimental setup used in the report.
    parser = argparse.ArgumentParser(description="Train/evaluate drowsiness classifier with optional SSL encoder.")

    # Paths
    parser.add_argument("--data_dir", type=str, default="data/Labeled/train")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--results_dir", type=str, default="results")

    # What experiment are we running.
    parser.add_argument("--label_frac", type=float, required=True)   # fraction of training labels to use (e.g. 0.05 = 5%)
    parser.add_argument("--run_name", type=str, required=True)        # e.g. "SimCLR_finetune", "CNN_scratch"
    parser.add_argument("--pretrained_path", type=str, default=None)  # override auto-detected checkpoint path
    parser.add_argument("--freeze_encoder", action="store_true")      # linear probe mode: only the head trains

    # Model and image settings
    parser.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--feature_dim", type=int, default=256)  # output dim of the CNN encoder

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # Data loading
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    """Entry point: train the classifier, evaluate on val/test, and save results to disk."""
    args = parse_args()

    # Fix random seeds so results are reproducible across runs
    set_seed(args.seed)
    ensure_project_dirs(Path.cwd())

    # Use GPU if available, otherwise fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Start the wall-clock timer -- we'll record total runtime at the end
    overall_start_time = perf_counter()

    # Build the subject-grouped train/val/test splits.
    # Subjects are kept together so the model never sees test-subject images during training.
    splits = build_labeled_splits(
        root=args.data_dir,
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # Grab only the requested fraction of labeled training data.
    # A shared cache is used so all experiments at the same fraction see the same images.
    train_dataset = build_shared_train_subset(splits.train, args.label_frac, seed=args.seed)

    # Wrap each split in a DataLoader.
    # pin_memory speeds up CPU->GPU transfers when a GPU is available.
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,                                  # shuffle training data each epoch
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    val_loader = DataLoader(
        splits.val,
        batch_size=args.batch_size,
        shuffle=False,                                 # no need to shuffle val/test
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    test_loader = DataLoader(
        splits.test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    # Figure out which pretraining method and variant to use based on the run name.
    # For example, "SimCLR_finetune" -> method="simclr", tag="full".
    # "CNN_scratch" stays as-is and skips the checkpoint loading step below.
    method, tag = infer_method_and_tag(args.run_name)
    encoder = SmallCNNEncoder(in_channels=3, feature_dim=args.feature_dim)

    if method != "cnn_scratch":
        # Try to find the right encoder checkpoint automatically.
        # First look for the specific variant (e.g. simclr_full_encoder.pt),
        # then fall back to the generic alias (e.g. simclr_encoder.pt).
        checkpoint_path = Path(args.pretrained_path) if args.pretrained_path else Path(args.checkpoint_dir) / f"{method}_{tag}_encoder.pt"

        if not checkpoint_path.exists():
            fallback = Path(args.checkpoint_dir) / f"{method}_encoder.pt"
            checkpoint_path = fallback if fallback.exists() else checkpoint_path
        load_encoder_checkpoint(checkpoint_path, encoder, device="cpu")

    # Attach the classifier head on top of the encoder.
    # If freeze_encoder=True, only the head's 514 parameters will be trained (linear probe).
    # If False, the entire network is fine-tuned end-to-end.
    model = EncoderClassifier(
        encoder=encoder,
        feature_dim=args.feature_dim,
        num_classes=len(splits.class_names),
        freeze_encoder=args.freeze_encoder,
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()

    # Only pass parameters that actually need gradients to the optimizer.
    # When the encoder is frozen, this is just the head weights.
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Track the best checkpoint based on validation accuracy.
    # This will be restored at the end before running the final test evaluation.
    best_state = None
    best_val_acc = -1.0
    history: list[dict[str, float]] = []  # per-epoch log for saving to history.json
    total_train_time = 0.0

    for epoch in range(1, args.epochs + 1):
        # One full pass through the training data
        train_stats = train_classifier_epoch(model, train_loader, optimizer, criterion, device)

        val_stats = evaluate_classifier(model, val_loader, criterion, device)

        total_train_time += train_stats["epoch_time_sec"]

        # Log everything for later analysis
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_acc": train_stats["accuracy"],
                "train_epoch_time_sec": train_stats["epoch_time_sec"],
                "val_loss": val_stats["loss"],
                "val_acc": val_stats["accuracy"],
            }
        )

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_stats['loss']:.4f} train_acc={train_stats['accuracy']:.4f} "
            f"val_loss={val_stats['loss']:.4f} val_acc={val_stats['accuracy']:.4f} "
            f"epoch_time_sec={train_stats['epoch_time_sec']:.2f}",
            flush=True,
        )

        # Save a snapshot of the weights whenever val accuracy improves.
        # This is the standard "best model" checkpoint strategy.
        if val_stats["accuracy"] > best_val_acc:
            best_val_acc = val_stats["accuracy"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    # Restore the best checkpoint before running final evaluation on the test set
    if best_state is not None:
        model.load_state_dict(best_state)

    # Run inference on the held-out test set.
    # return_predictions = True gives us the raw labels and softmax probabilities needed for the confusion matrix and ROC curve. measure_runtime=True records per-sample latency.
    test_stats = evaluate_classifier(
        model,
        test_loader,
        criterion,
        device,
        return_predictions=True,
        measure_runtime=True,
    )

    # Compute all the classification metrics: accuracy, precision, recall, F1, confusion matrix, and if we have class probabilities, the ROC curve and AUC as well.
    summary = summarize_classification(
        y_true=test_stats["targets"],
        y_pred=test_stats["predictions"],
        y_prob=test_stats.get("probabilities"),
        class_names=splits.class_names,
        loss=test_stats["loss"],
    )

    # Count parameters -- useful for reporting model complexity
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    # Attach runtime and model info to the summary so everything is in one place
    summary["runtime"] = {
        "device": str(device),
        "train_total_sec": total_train_time,
        "train_avg_epoch_sec": total_train_time / max(args.epochs, 1),
        "test_total_sec": test_stats["runtime_sec"],
        "test_per_sample_sec": test_stats["runtime_per_sample_sec"],
        "test_fps": 1.0 / test_stats["runtime_per_sample_sec"] if test_stats["runtime_per_sample_sec"] > 0 else 0.0,
        "overall_wall_clock_sec": perf_counter() - overall_start_time,
    }

    summary["model"] = {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "freeze_encoder": bool(args.freeze_encoder),
    }

    # Results go into results/<run_name>/label_frac_X.XX/
    # This makes it easy to compare runs side by side in the file system.
    output_dir = Path(args.results_dir) / args.run_name / f"label_frac_{args.label_frac:.2f}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the full model checkpoint so we can reload it for inference later if needed
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "history": history,
            "summary": summary,
            "args": vars(args),
        },

        output_dir / "best_model.pt",
    )

    # Also write the training history and final metrics as plain JSON --
    # easier to read without loading the full PyTorch checkpoint
    with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    # Generate and save the confusion matrix plot
    plot_confusion_matrix(
        confusion_matrix=summary["confusion_matrix"],
        class_names=splits.class_names,
        output_path=output_dir / "confusion_matrix.png",
    )

    # ROC curve is only meaningful for binary classification, which this is,
    # but we guard anyway in case summarize_classification didn't produce one
    if "roc_curve" in summary:
        plot_roc_curve(
            fpr=summary["roc_curve"]["fpr"],
            tpr=summary["roc_curve"]["tpr"],
            auc=summary["roc_auc"],
            output_path=output_dir / "roc_curve.png",
        )

    print(f"saved_results={output_dir}")

    # Print the summary to the terminal, but strip out the raw ROC curve arrays --
    print_summary = {k: v for k, v in summary.items() if k != "roc_curve"}
    print(json.dumps(print_summary, indent=2))


if __name__ == "__main__":
    main()
