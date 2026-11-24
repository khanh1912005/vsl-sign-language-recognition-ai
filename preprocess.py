"""
=============================================================
TIEN XU LY DU LIEU - HE THONG NHAN DIEN NGON NGU KY HIEU
=============================================================
Trich xuat landmarks tu video bang MediaPipe Tasks API (v1.1)
Fix: Unicode normalization for Vietnamese folder names

Author: Le Gia Khanh - A48572
=============================================================
"""

import os
import sys
import cv2
import numpy as np
import mediapipe as mp
import pickle
import json
import unicodedata
from tqdm import tqdm
import warnings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_DIR = rf"{BASE_DIR}\dataset"
OUTPUT_DIR = rf"{BASE_DIR}\processed_data"
MODEL_PATH = rf"{BASE_DIR}\holistic_landmarker.task"
MAX_FRAMES = 30

BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

POSE_FEATURES = 33 * 4
HAND_FEATURES = 21 * 3
TOTAL_FEATURES = POSE_FEATURES + HAND_FEATURES * 2  # 258


def normalize_str(s):
    """Chuan hoa Unicode ve dang NFC."""
    return unicodedata.normalize("NFC", s)


def extract_landmarks(result):
    """Trich xuat landmarks tu ket qua HolisticLandmarker."""
    if result.pose_landmarks and len(result.pose_landmarks) > 0:
        pose_lm = result.pose_landmarks[0]
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility if hasattr(lm, 'visibility') else 0.0]
                         for lm in pose_lm]).flatten()
    else:
        pose = np.zeros(POSE_FEATURES)

    if result.left_hand_landmarks and len(result.left_hand_landmarks) > 0:
        lh_lm = result.left_hand_landmarks[0]
        lh = np.array([[lm.x, lm.y, lm.z] for lm in lh_lm]).flatten()
    else:
        lh = np.zeros(HAND_FEATURES)

    if result.right_hand_landmarks and len(result.right_hand_landmarks) > 0:
        rh_lm = result.right_hand_landmarks[0]
        rh = np.array([[lm.x, lm.y, lm.z] for lm in rh_lm]).flatten()
    else:
        rh = np.zeros(HAND_FEATURES)

    return np.concatenate([pose, lh, rh])


def process_video(video_path, landmarker, max_frames=MAX_FRAMES):
    """Xu ly 1 video: trich xuat landmarks cho tung frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return None

    if total_frames >= max_frames:
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    else:
        indices = list(range(total_frames))

    landmarks_sequence = []
    frame_idx = 0
    sample_idx = 0

    while cap.isOpened() and sample_idx < len(indices):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx == indices[sample_idx]:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            try:
                result = landmarker.detect(mp_image)
                landmarks = extract_landmarks(result)
                landmarks_sequence.append(landmarks)
            except Exception:
                landmarks_sequence.append(np.zeros(TOTAL_FEATURES))
            sample_idx += 1

        frame_idx += 1

    cap.release()

    if len(landmarks_sequence) == 0:
        return None

    sequence = np.array(landmarks_sequence)
    if len(sequence) < max_frames:
        pad_length = max_frames - len(sequence)
        padding = np.tile(sequence[-1], (pad_length, 1))
        sequence = np.vstack([sequence, padding])
    elif len(sequence) > max_frames:
        sequence = sequence[:max_frames]

    return sequence


def main():
    print("=" * 60)
    print("TIEN XU LY DU LIEU - TRICH XUAT LANDMARKS TU VIDEO")
    print(f"MediaPipe version: {mp.__version__}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load label mapping va normalize Unicode
    with open(os.path.join(DATASET_DIR, "label_mapping.pkl"), "rb") as f:
        label_mapping_raw = pickle.load(f)

    # Normalize all keys to NFC
    label_mapping = {normalize_str(k): v for k, v in label_mapping_raw.items()}
    idx_to_label = {v: k for k, v in label_mapping.items()}

    print(f"So luong classes: {len(label_mapping)}")

    # Save normalized label mapping
    with open(os.path.join(OUTPUT_DIR, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUTPUT_DIR, "idx_to_label.json"), "w", encoding="utf-8") as f:
        json.dump({str(v): k for k, v in label_mapping.items()}, f, ensure_ascii=False, indent=2)

    # Setup HolisticLandmarker
    with open(MODEL_PATH, "rb") as _f:
        options = HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=_f.read()),
            running_mode=VisionRunningMode.IMAGE,
        )

    # ========================================
    # Process TRAIN set (has subdirectories)
    # ========================================
    split = "train"
    split_dir = os.path.join(DATASET_DIR, split)
    print(f"\n{'=' * 40}")
    print(f"Dang xu ly: {split}")
    print(f"{'=' * 40}")

    all_sequences = []
    all_labels = []
    errors = 0
    skipped_labels = []

    label_dirs = sorted(os.listdir(split_dir))
    total_videos = 0
    for label_name in label_dirs:
        label_path = os.path.join(split_dir, label_name)
        if os.path.isdir(label_path):
            total_videos += len([f for f in os.listdir(label_path) if f.endswith(".mp4")])

    print(f"Tong so video: {total_videos}")
    pbar = tqdm(total=total_videos, desc=f"Processing {split}")

    with HolisticLandmarker.create_from_options(options) as landmarker:
        for label_name in label_dirs:
            label_path = os.path.join(split_dir, label_name)
            if not os.path.isdir(label_path):
                continue

            # Normalize dir name and look up
            norm_name = normalize_str(label_name)
            if norm_name in label_mapping:
                label_idx = label_mapping[norm_name]
            else:
                skipped_labels.append(label_name)
                videos = [f for f in os.listdir(label_path) if f.endswith(".mp4")]
                pbar.update(len(videos))
                continue

            videos = [f for f in os.listdir(label_path) if f.endswith(".mp4")]
            for video_file in videos:
                video_path = os.path.join(label_path, video_file)
                try:
                    sequence = process_video(video_path, landmarker)
                    if sequence is not None:
                        all_sequences.append(sequence)
                        all_labels.append(label_idx)
                    else:
                        errors += 1
                except Exception:
                    errors += 1
                pbar.update(1)

    pbar.close()

    if skipped_labels:
        print(f"  Skipped {len(skipped_labels)} labels (no match)")

    if len(all_sequences) > 0:
        X = np.array(all_sequences, dtype=np.float32)
        y = np.array(all_labels, dtype=np.int32)
        np.save(os.path.join(OUTPUT_DIR, f"X_{split}.npy"), X)
        np.save(os.path.join(OUTPUT_DIR, f"y_{split}.npy"), y)
        print(f"\nKet qua {split}:")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Unique classes: {len(np.unique(y))}")
        print(f"  Errors: {errors}")

    # ========================================
    # Process PUBLIC_TEST (flat directory, no labels for now)
    # We skip test set for now since it has no labels
    # ========================================
    print(f"\n{'=' * 40}")
    print(f"Note: public_test va private_test khong co label subdirs")
    print(f"Chi su dung train set de train va evaluate (split train/val)")
    print(f"{'=' * 40}")

    print("\n" + "=" * 60)
    print("TIEN XU LY HOAN TAT!")
    print("=" * 60)


if __name__ == "__main__":
    main()
