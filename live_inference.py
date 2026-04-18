import argparse
from collections import deque
from pathlib import Path
from time import perf_counter

import cv2
import torch

from drowsiness_ssl.temporal import aggregate_frame_predictions
from drowsiness_ssl.utils.inference import (
    WebcamDrowsinessInference,
    draw_status_overlay,
    load_trained_classifier,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the live webcam inference script."""
    parser = argparse.ArgumentParser(description="Live webcam drowsiness detection with 10-frame voting.")

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to a trained classifier checkpoint, usually results/.../best_model.pt",
    )

    parser.add_argument("--camera_index", type=int, default=0)              # 0 = default webcam
    parser.add_argument("--window_size", type=int, default=10)              # number of frames in the rolling vote
    parser.add_argument("--threshold_ratio", type=float, default=0.70)     # fraction of frames that must be drowsy
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--detect_face", action="store_true")               # crop to face region before inference
    parser.add_argument("--flip_horizontal", action="store_true")           # mirror image (front-facing webcam)
    parser.add_argument("--display_scale", type=float, default=1.0)         # resize the output window
    parser.add_argument("--frame_skip", type=int, default=0, help="Skip N frames between predictions.")
    return parser.parse_args()


def main() -> None:
    """Entry point: open the webcam, run per-frame inference, and display the live overlay window."""
    args = parse_args()
    device = resolve_device(args.device)

    # Load the trained model from the checkpoint file
    model, preprocess, metadata = load_trained_classifier(Path(args.weights), device)
    pipeline = WebcamDrowsinessInference(
        model=model,
        preprocess=preprocess,
        device=device,
        detect_face=args.detect_face,
    )

    capture = cv2.VideoCapture(args.camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open webcam at index {args.camera_index}")

    # Rolling window of per-frame predictions for the temporal vote
    prediction_window: deque[int] = deque(maxlen=args.window_size)
    frame_counter = 0
    prev_time = perf_counter()
    fps = 0.0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if args.flip_horizontal:
                frame = cv2.flip(frame, 1)

            frame_counter += 1
            # frame_skip lets us reduce GPU load by only running inference every N frames
            run_prediction = args.frame_skip <= 0 or ((frame_counter - 1) % (args.frame_skip + 1) == 0)

            if run_prediction:
                inference = pipeline.predict(frame)
                prediction_window.append(inference.predicted_label)

            else:
                # Reuse the last result so the overlay stays up-to-date on skipped frames
                inference = pipeline.last_result

            # Aggregate the sliding window into a final DROWSY/ALERT decision
            if len(prediction_window) > 0:
                decision = aggregate_frame_predictions(
                    list(prediction_window),
                    positive_class=1,
                    threshold_ratio=args.threshold_ratio,
                )

            else:
                decision = None  # window not yet warmed up

            # Exponential moving average for a smooth FPS display
            now = perf_counter()
            dt = now - prev_time
            prev_time = now

            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)

            output = draw_status_overlay(
                frame=frame.copy(),
                inference=inference,
                temporal_decision=decision,
                fps=fps,
                model_label=metadata.get("run_name", "classifier"),
            )

            if args.display_scale != 1.0:
                output = cv2.resize(
                    output,
                    None,
                    fx=args.display_scale,
                    fy=args.display_scale,
                    interpolation=cv2.INTER_LINEAR,
                )

            cv2.imshow("Driver Drowsiness Detection", output)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):  # ESC or 'q' to quit
                break
            
    finally:
        # Always release the camera and close windows, even if we crash
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
