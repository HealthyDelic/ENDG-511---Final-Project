# Driver Drowsiness SSL Project

This project trains and evaluates three approaches for driver drowsiness detection:

- CNN_scratch: supervised CNN trained from scratch
- SimCLR: self-supervised contrastive pretraining + downstream classifier
- ConvMAE: masked autoencoding pretraining + downstream classifier

"matrics.py" creates the required test and validation datasets.

The three drowsy subfolders are treated as one downstream label for binary drowsiness detection.

# Notes
- The code uses a shared small CNN encoder across all methods.
- SimCLR and ConvMAE codes save encoder checkpoints into directly into checkpoints.
- metrics.py writes confusion matrices, and summaries into results folder.
- For video-level decisions, aggregate 10 frame predictions and mark the clip as drowsy if at least 70% of those frames are predicted as drowsy.
- `ive_inference.py opens a webcam feed, runs per-frame classification, and applies the 70% rule in real time.

# Pretrain SimCLR
python pretrain.py --method simclr --data_dir data/unlabeled --simclr_variant full

# Pretrain ConvMAE
python pretrain.py --method convmae --data_dir data/unlabeled --mask_ratio 0.75

# Train/evaluate scratch CNN
python metrics.py --data_dir data/labeled --label_frac 0.10 --run_name CNN_scratch

# Train/evaluate pretrained SimCLR encoder
python metrics.py --data_dir data/labeled --label_frac 0.10 --run_name SimCLR --freeze_encoder

# Train/evaluate pretrained ConvMAE encoder
python metrics.py --data_dir data/labeled --label_frac 0.10 --run_name ConvMAE --freeze_encoder

# Run live webcam inference
python live_inference.py --weights results/CNN_scratch/label_frac_1.00/best_model.pt --camera_index 0 --detect_face --flip_horizontal

# Dataset Link
https://www.kaggle.com/datasets/samymesbah/nthu-dataset-ddd-multi-class