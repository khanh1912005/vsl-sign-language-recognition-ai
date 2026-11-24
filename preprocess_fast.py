"""
=============================================================
TIỀN XỬ LÝ NHANH - MULTIPROCESSING
Trích xuất landmarks MediaPipe Holistic với đa tiến trình (multi-worker)
=============================================================
"""

import os
import sys
import io
import time
import glob
import json
import pickle
import unicodedata
import cv2
import numpy as np
import mediapipe as mp
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import warnings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
warnings.filterwarnings("ignore")

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DATASET_DIR = rf"{BASE_DIR}\dataset"
OUTPUT_DIR = rf"{BASE_DIR}\processed_data"
MODEL_PATH = rf"{BASE_DIR}\holistic_landmarker.task"
MAX_FRAMES = 30

POSE_FEATURES = 33 * 4     # 132
HAND_FEATURES = 21 * 3     # 63
TOTAL_FEATURES = POSE_FEATURES + HAND_FEATURES * 2  # 258

_worker_landmarker = None


def normalize_str(s):
    return unicodedata.normalize("NFC", s)


def init_worker():
    global _worker_landmarker
    BaseOptions = mp.tasks.BaseOptions
    HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
    HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    with open(MODEL_PATH, "rb") as _f:
        options = HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=_f.read()),
            running_mode=VisionRunningMode.IMAGE,
        )
    _worker_landmarker = HolisticLandmarker.create_from_options(options)


def extract_landmarks(result):
    if result.pose_landmarks and len(result.pose_landmarks) > 0:
        pose = np.array([[lm.x, lm.y, lm.z, getattr(lm, 'visibility', 0.0)] for lm in result.pose_landmarks]).flatten()
    else:
        pose = np.zeros(POSE_FEATURES)

    if result.left_hand_landmarks and len(result.left_hand_landmarks) > 0:
        lh = np.array([[lm.x, lm.y, lm.z] for lm in result.left_hand_landmarks]).flatten()
    else:
        lh = np.zeros(HAND_FEATURES)

    if result.right_hand_landmarks and len(result.right_hand_landmarks) > 0:
        rh = np.array([[lm.x, lm.y, lm.z] for lm in result.right_hand_landmarks]).flatten()
    else:
        rh = np.zeros(HAND_FEATURES)

    return np.concatenate([pose, lh, rh])


def process_single_video(item):
    global _worker_landmarker
    video_path, label_idx = item

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, label_idx

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return None, label_idx

    if total_frames >= MAX_FRAMES:
        indices = np.linspace(0, total_frames - 1, MAX_FRAMES, dtype=int)
    else:
        indices = list(range(total_frames))

    landmarks_seq = []
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
                result = _worker_landmarker.detect(mp_image)
                landmarks = extract_landmarks(result)
                landmarks_seq.append(landmarks)
            except Exception:
                landmarks_seq.append(np.zeros(TOTAL_FEATURES))
            sample_idx += 1

        frame_idx += 1

    cap.release()

    if len(landmarks_seq) == 0:
        return None, label_idx

    seq = np.array(landmarks_seq, dtype=np.float32)
    if len(seq) < MAX_FRAMES:
        pad_len = MAX_FRAMES - len(seq)
        padding = np.tile(seq[-1], (pad_len, 1))
        seq = np.vstack([seq, padding])
    elif len(seq) > MAX_FRAMES:
        seq = seq[:MAX_FRAMES]

    return seq, label_idx


def main():
    print("=" * 60)
    print("TIỀN XỬ LÝ DỮ LIỆU ĐA TIẾN TRÌNH (MULTIPROCESSING)")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load and normalize label mapping
    with open(os.path.join(DATASET_DIR, "label_mapping.pkl"), "rb") as f:
        label_mapping_raw = pickle.load(f)
    label_mapping = {normalize_str(k): v for k, v in label_mapping_raw.items()}

    # 2. Collect all video paths and their labels
    train_dir = os.path.join(DATASET_DIR, "train")
    items = []
    for label_name in os.listdir(train_dir):
        label_path = os.path.join(train_dir, label_name)
        if not os.path.isdir(label_path):
            continue
        norm_name = normalize_str(label_name)
        if norm_name not in label_mapping:
            continue
        label_idx = label_mapping[norm_name]
        for f in os.listdir(label_path):
            if f.endswith(".mp4"):
                items.append((os.path.join(label_path, f), label_idx))

    print(f"Tổng số video tìm thấy: {len(items)}")

    # 3. Determine number of workers (use 4 workers for optimal CPU balance)
    num_workers = min(6, max(1, cpu_count() - 2))
    print(f"Sử dụng {num_workers} worker processes...")

    t0 = time.time()
    all_sequences = []
    all_labels = []
    errors = 0

    with Pool(processes=num_workers, initializer=init_worker) as pool:
        for seq, lbl in tqdm(pool.imap(process_single_video, items, chunksize=8), total=len(items), desc="Đang trích xuất landmarks"):
            if seq is not None:
                all_sequences.append(seq)
                all_labels.append(lbl)
            else:
                errors += 1

    elapsed = time.time() - t0
    print(f"\nThời gian hoàn thành: {elapsed/60:.1f} phút ({elapsed/len(items):.2f}s / video)")

    X = np.array(all_sequences, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)

    print(f"X shape: {X.shape}, min: {X.min():.3f}, max: {X.max():.3f}, non-zero: {(X != 0).mean():.1%}")
    print(f"y shape: {y.shape}, classes: {len(np.unique(y))}")
    print(f"Lỗi: {errors}")

    # Save
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y)
    print("Đã lưu X_train.npy và y_train.npy thành công!")


if __name__ == "__main__":
    main()
