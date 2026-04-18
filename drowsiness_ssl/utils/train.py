from time import perf_counter

import torch

from drowsiness_ssl.utils.losses import masked_reconstruction_loss, nt_xent_loss


def run_simclr_epoch(model, loader, optimizer, device, temperature):
    """Run one epoch of SimCLR contrastive pretraining.

    Each batch from the loader contains two views (view_one, view_two) of the same images.
    We pass each through the encoder+projector, then compute the NT-Xent loss to push
    the two views of each image closer together while pushing other images apart.
    """
    model.train()
    total_loss = 0.0
    total_batches = 0
    for view_one, view_two in loader:
        view_one = view_one.to(device)
        view_two = view_two.to(device)
        optimizer.zero_grad(set_to_none=True)  # set_to_none is slightly faster than zeroing
        _, projection_one = model(view_one)
        _, projection_two = model(view_two)
        loss = nt_xent_loss(projection_one, projection_two, temperature)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        total_batches += 1
    return total_loss / max(total_batches, 1)


def run_convmae_epoch(model, loader, optimizer, device):
    """Run one epoch of ConvMAE reconstruction pretraining.

    The model applies a random patch mask internally, then tries to reconstruct
    the original image. Loss is only computed over the masked patches.
    """
    model.train()
    total_loss = 0.0
    total_batches = 0
    for images in loader:
        images = images.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = masked_reconstruction_loss(
            reconstruction=outputs["reconstruction"],
            target=outputs["target"],
            patch_mask=outputs["patch_mask"],
            patch_size=model.patch_size,
        )
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        total_batches += 1
    return total_loss / max(total_batches, 1)


def train_classifier_epoch(model, loader, optimizer, criterion, device) -> dict[str, float]:
    """Train the classifier for one epoch and return loss, accuracy, and wall-clock time."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    start_time = perf_counter()
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        predictions = logits.argmax(dim=1)
        # Accumulate weighted loss (loss * batch_size) so the final average is sample-weighted,
        # not batch-weighted -- matters when the last batch is smaller than the others.
        total_loss += float(loss.item()) * images.size(0)
        total_correct += int((predictions == labels).sum().item())
        total_examples += int(images.size(0))
    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
        "epoch_time_sec": perf_counter() - start_time,
    }


@torch.no_grad()
def evaluate_classifier(
    model,
    loader,
    criterion,
    device,
    return_predictions: bool = False,
    measure_runtime: bool = False,
) -> dict:
    """Evaluate the classifier on a loader and return metrics.

    With return_predictions=True, also returns the raw targets, predictions,
    and per-class softmax probabilities -- needed for confusion matrix and ROC.
    With measure_runtime=True, records total and per-sample inference time.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    all_targets = []
    all_predictions = []
    all_probabilities = []
    start_time = perf_counter() if measure_runtime else None
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)   # @torch.no_grad() above handles disabling the grad tape
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)
        total_loss += float(loss.item()) * images.size(0)
        total_correct += int((predictions == labels).sum().item())
        total_examples += int(images.size(0))
        if return_predictions:
            probs = torch.softmax(logits, dim=1)
            all_targets.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_probabilities.extend(probs.cpu().tolist())
    stats = {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
    }
    if measure_runtime and start_time is not None:
        total_runtime = perf_counter() - start_time
        stats["runtime_sec"] = total_runtime
        stats["runtime_per_sample_sec"] = total_runtime / max(total_examples, 1)
    if return_predictions:
        stats["targets"] = all_targets
        stats["predictions"] = all_predictions
        stats["probabilities"] = all_probabilities
    return stats
