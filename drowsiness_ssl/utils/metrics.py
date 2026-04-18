import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _compute_roc_auc(y_true, y_scores):
    """Compute ROC curve (fpr, tpr) and AUC for binary classification.

    Implemented from scratch rather than using sklearn so the project has no
    extra runtime dependency. The algorithm sorts predictions by descending score
    and accumulates true/false positives along the threshold sweep.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_scores = np.asarray(y_scores, dtype=float)
    # Sort by descending score -- we sweep thresholds from high to low
    desc_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[desc_indices]
    tps = np.cumsum(y_true_sorted)          # true positives at each threshold
    fps = np.cumsum(1.0 - y_true_sorted)    # false positives at each threshold
    n_pos = float(y_true.sum())
    n_neg = float(len(y_true) - n_pos)
    tpr = tps / max(n_pos, 1)   # recall = TP / (TP + FN)
    fpr = fps / max(n_neg, 1)   # fall-out = FP / (FP + TN)
    # Prepend the origin (0, 0) so the curve starts from the corner
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    # AUC via the trapezoidal rule
    auc = float(np.trapezoid(tpr, fpr))
    return fpr.tolist(), tpr.tolist(), auc


def summarize_classification(y_true, y_pred, class_names, loss: float, y_prob=None) -> dict:
    """Compute the full set of classification metrics from raw prediction arrays.

    Returns a dict with accuracy, macro precision/recall/F1, per-class accuracy,
    the confusion matrix, and (if y_prob is provided) ROC AUC.
    Everything is in a JSON-serializable format so it can go straight to summary.json.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    num_classes = len(class_names)

    # Build the confusion matrix manually -- rows = true class, columns = predicted class
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        confusion[int(true_label), int(pred_label)] += 1

    accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    per_class_acc = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for cls_idx in range(num_classes):
        tp = confusion[cls_idx, cls_idx]
        fp = confusion[:, cls_idx].sum() - tp  # other classes predicted as this class
        fn = confusion[cls_idx, :].sum() - tp  # this class predicted as something else
        support = confusion[cls_idx, :].sum()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        precision_scores.append(float(precision))
        recall_scores.append(float(recall))
        f1_scores.append(float(f1))
        per_class_acc.append(float(tp / max(support, 1)))

    result = {
        "loss": float(loss),
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precision_scores)),
        "macro_recall": float(np.mean(recall_scores)),
        "macro_f1": float(np.mean(f1_scores)),
        "per_class_accuracy": per_class_acc,
        "class_names": list(class_names),
        "confusion_matrix": confusion.tolist(),
    }

    # ROC AUC only makes sense for binary classification with probability outputs
    if y_prob is not None and len(class_names) == 2:
        y_prob_arr = np.asarray(y_prob)
        pos_scores = y_prob_arr[:, 1]  # probability of the positive (drowsy) class
        fpr, tpr, auc = _compute_roc_auc(y_true, pos_scores)
        result["roc_auc"] = auc
        result["roc_curve"] = {"fpr": fpr, "tpr": tpr}
    return result


def plot_confusion_matrix(confusion_matrix, class_names, output_path):
    """Save a colour-coded confusion matrix image to output_path."""
    confusion = np.asarray(confusion_matrix)
    fig, ax = plt.subplots(figsize=(6, 6))
    image = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    # Annotate each cell with the raw count
    for row in range(confusion.shape[0]):
        for col in range(confusion.shape[1]):
            ax.text(col, row, int(confusion[row, col]), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)  # free memory -- matplotlib holds figures open by default


def plot_roc_curve(fpr, tpr, auc, output_path):
    """Save an ROC curve plot (with the random-classifier baseline) to output_path."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC curve (AUC = {auc:.4f})")
    # Diagonal reference line for a random classifier (AUC = 0.5)
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_json(path, payload):
    """Write a dict to a JSON file with human-readable indentation."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
