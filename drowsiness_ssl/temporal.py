from dataclasses import dataclass

import numpy as np


@dataclass
class TemporalDecision:
    """Result of aggregating predictions over a sliding window of frames."""
    frame_count: int
    drowsy_frames: int
    drowsy_ratio: float
    is_drowsy: bool  # True when drowsy_ratio >= threshold_ratio


def aggregate_frame_predictions(
    frame_predictions: list[int] | np.ndarray,
    positive_class: int = 1,
    threshold_ratio: float = 0.70,
) -> TemporalDecision:
    """Majority-vote over a window of per-frame predictions.

    Single-frame predictions can be noisy -- a brief glance away or a blink
    can flip the label for one frame. This function smooths that out by
    looking at a rolling window and only flagging DROWSY when the majority
    of frames in the window are classified as drowsy.

    threshold_ratio=0.70 means at least 7 out of 10 frames must be drowsy
    before the final decision is DROWSY.
    """
    predictions = np.asarray(frame_predictions, dtype=int)
    if predictions.size == 0:
        raise ValueError("frame_predictions must contain at least one prediction.")
    drowsy_frames = int((predictions == positive_class).sum())
    ratio = drowsy_frames / int(predictions.size)
    return TemporalDecision(
        frame_count=int(predictions.size),
        drowsy_frames=drowsy_frames,
        drowsy_ratio=ratio,
        is_drowsy=ratio >= threshold_ratio,
    )
