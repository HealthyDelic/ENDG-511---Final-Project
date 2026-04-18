from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from drowsiness_ssl.config import DEFAULT_MEAN, DEFAULT_STD
from drowsiness_ssl.models.classifier import EncoderClassifier
from drowsiness_ssl.models.encoder import SmallCNNEncoder


@dataclass
class FrameInference:
    """Holds the per-frame prediction output so we can pass it around cleanly.

    drowsy_probability is the raw softmax score for class 1 (drowsy).
    confidence is the score for whichever class was predicted (i.e. max probability).
    face_box is None when face detection is disabled or no face was found.
    """
    predicted_label: int
    predicted_name: str
    confidence: float
    drowsy_probability: float
    face_box: tuple[int, int, int, int] | None = None


def resolve_device(requested: str) -> torch.device:
    """Resolve a device string ('auto', 'cuda', or 'cpu') to a torch.device."""
    # "auto" picks CUDA if available, CPU otherwise.
    # Explicit "cuda" or "cpu" are passed through directly.
    if requested == "cuda":
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_eval_transform(image_size: int):
    """Build the evaluation preprocessing pipeline: resize, to-tensor, and ImageNet normalize."""
    # The same normalization used during training -- must match or predictions will be off.
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD),
        ]
    )


def load_trained_classifier(checkpoint_path: Path, device: torch.device):
    """Load a trained classifier from a best_model.pt checkpoint.

    The checkpoint stores the original training args alongside the weights,
    so we can reconstruct the model architecture without the user having to
    specify feature_dim and image_size manually.
    """
    payload = torch.load(checkpoint_path, map_location=device)
    args = payload.get("args", {})
    feature_dim = int(args.get("feature_dim", 256))
    encoder = SmallCNNEncoder(in_channels=3, feature_dim=feature_dim)
    model = EncoderClassifier(
        encoder=encoder,
        feature_dim=feature_dim,
        num_classes=2,
        freeze_encoder=False,  # we're loading a fully trained model, not a frozen probe
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()

    preprocess = build_eval_transform(int(args.get("image_size", 64)))
    metadata = {
        "run_name": args.get("run_name", checkpoint_path.parent.name),
        "label_frac": args.get("label_frac"),
        "image_size": args.get("image_size", 64),
    }
    return model, preprocess, metadata


class WebcamDrowsinessInference:
    """Handles per-frame inference from a webcam stream.

    Optionally runs a Haar cascade face detector to crop to the face region
    before passing the image to the model. If no face is found (or detection
    is disabled), the entire frame is used instead.
    """

    def __init__(self, model, preprocess, device: torch.device, detect_face: bool = False) -> None:
        self.model = model
        self.preprocess = preprocess
        self.device = device
        self.detect_face = detect_face
        self.class_names = ["notdrowsy", "drowsy"]
        self.last_result: FrameInference | None = None

        if detect_face:
            # Load OpenCV's built-in frontal face detector
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            self.face_detector = cv2.CascadeClassifier(str(cascade_path))
            if self.face_detector.empty():
                # Cascade file not found or failed to load -- fall back to full frame
                self.face_detector = None
                self.detect_face = False
        else:
            self.face_detector = None

    def predict(self, frame_bgr: np.ndarray) -> FrameInference:
        """Run the classifier on one BGR frame and return a FrameInference result."""
        crop, box = self._extract_region(frame_bgr)
        # OpenCV uses BGR, PIL uses RGB -- convert before preprocessing
        image_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_rgb)
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)  # add batch dimension

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

        predicted_label = int(np.argmax(probabilities))
        result = FrameInference(
            predicted_label=predicted_label,
            predicted_name=self.class_names[predicted_label],
            confidence=float(probabilities[predicted_label]),
            drowsy_probability=float(probabilities[1]),
            face_box=box,
        )
        self.last_result = result  # cache so the overlay can use it on skipped frames
        return result

    def _extract_region(self, frame_bgr: np.ndarray):
        """Return the region of interest (face crop or full frame) and its bounding box."""
        if self.detect_face and self.face_detector is not None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
            if len(faces) > 0:
                # If multiple faces are detected, use the largest one
                x, y, w, h = max(faces, key=lambda candidate: candidate[2] * candidate[3])
                # Add a margin so we don't clip the forehead or chin
                margin_x = int(0.15 * w)
                margin_y = int(0.2 * h)
                x0 = max(0, x - margin_x)
                y0 = max(0, y - margin_y)
                x1 = min(frame_bgr.shape[1], x + w + margin_x)
                y1 = min(frame_bgr.shape[0], y + h + margin_y)
                return frame_bgr[y0:y1, x0:x1], (x0, y0, x1, y1)
        # No face found (or detection disabled) -- use the whole frame
        return frame_bgr, None


def draw_status_overlay(
    frame: np.ndarray,
    inference: FrameInference | None,
    temporal_decision,
    fps: float,
    model_label: str,
) -> np.ndarray:
    """Draw prediction info and alerts on the frame for the live display window."""
    if inference is None:
        return frame

    # Apply a subtle red tint when the temporal vote says DROWSY
    if temporal_decision is not None and temporal_decision.is_drowsy:
        tint = np.zeros_like(frame)
        tint[:, :, 2] = 120  # red channel
        frame = cv2.addWeighted(frame, 1.0, tint, 0.18, 0)

    # Draw the face bounding box if we have one
    if inference.face_box is not None:
        x0, y0, x1, y1 = inference.face_box
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 2)

    # Build the lines of text to render
    status_text = f"frame: {inference.predicted_name} ({inference.confidence:.2f})"
    vote_text = "window: warming up"
    if temporal_decision is not None:
        vote_text = (
            f"10-frame vote: {temporal_decision.drowsy_frames}/{temporal_decision.frame_count} "
            f"= {temporal_decision.drowsy_ratio:.2f}"
        )
    decision_text = (
        "decision: DROWSY" if temporal_decision is not None and temporal_decision.is_drowsy else "decision: ALERT"
    )
    prob_text = f"drowsy_prob: {inference.drowsy_probability:.2f}"
    fps_text = f"fps: {fps:.1f}"
    model_text = f"model: {model_label}"

    lines = [status_text, prob_text, vote_text, decision_text, fps_text, model_text]
    y = 30
    for line in lines:
        # Draw white text with a thin dark outline for readability on any background
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 1, cv2.LINE_AA)
        y += 30
    return frame
