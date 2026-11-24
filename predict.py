r"""
=============================================================
DỰ ĐOÁN CỬ CHỈ ĐƠN LẺ TỪ VIDEO (ISOLATED SIGN PREDICTION)
=============================================================
Sử dụng đặc trưng bất biến không gian - thời gian (458 chiều)
kết hợp mô hình Pure TFLite / Keras để nhận diện cử chỉ VSL.

Usage:
    python predict.py "D:\thực chiến\Nhận diện xử lý ngôn ngữ ký hiệu\dataset\public_test\100708.mp4"

Author: Le Gia Khanh - A48572
=============================================================
"""

import os
import sys
import time
import json
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from feature_engineering import extract_sequence_features
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.stdout.reconfigure(encoding='utf-8')

MODEL_TFLITE_PATH = rf"{BASE_DIR}\models\model.tflite"
MODEL_KERAS_PATH = rf"{BASE_DIR}\models\best_model.keras"
LABEL_MAPPING_PATH = rf"{BASE_DIR}\processed_data\label_mapping.json"
HOLISTIC_TASK_PATH = rf"{BASE_DIR}\holistic_landmarker.task"
MAX_FRAMES = 30

POSE_FEATURES = 33 * 4
HAND_FEATURES = 21 * 3
TOTAL_RAW_FEATURES = POSE_FEATURES + HAND_FEATURES * 2  # 258


def extract_landmarks(result):
    """Trích xuất 258 tọa độ thô từ kết quả HolisticLandmarker."""
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


def extract_video_raw_landmarks(video_path, landmarker):
    """Đọc video và trích xuất chuỗi 30 frame landmarks thô (30, 258)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Không thể mở video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        raise ValueError("Video không có frame nào")

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
                result = landmarker.detect(mp_image)
                landmarks = extract_landmarks(result)
                landmarks_seq.append(landmarks)
            except Exception:
                landmarks_seq.append(np.zeros(TOTAL_RAW_FEATURES))
            sample_idx += 1

        frame_idx += 1

    cap.release()

    seq = np.array(landmarks_seq, dtype=np.float32)
    if len(seq) < MAX_FRAMES:
        pad_len = MAX_FRAMES - len(seq)
        padding = np.tile(seq[-1], (pad_len, 1))
        seq = np.vstack([seq, padding])
    elif len(seq) > MAX_FRAMES:
        seq = seq[:MAX_FRAMES]

    return seq


def predict_video(video_path, top_k=5):
    """Phân tích video và dự đoán Top-K nhãn ngôn ngữ ký hiệu."""
    print(f"Đang phân tích video: {video_path}")

    with open(LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
        label_mapping = json.load(f)
    idx_to_label = {v: k for k, v in label_mapping.items()}

    BaseOptions = mp.tasks.BaseOptions
    HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
    HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    with open(HOLISTIC_TASK_PATH, "rb") as _f:
        options = HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=_f.read()),
            running_mode=VisionRunningMode.IMAGE,
        )

    t0 = time.time()
    with HolisticLandmarker.create_from_options(options) as landmarker:
        raw_seq = extract_video_raw_landmarks(video_path, landmarker)
    mp_time_ms = (time.time() - t0) * 1000.0

    t1 = time.time()
    feat_seq = extract_sequence_features(raw_seq)  # Shape: (30, 458)
    input_data = np.expand_dims(feat_seq, axis=0)   # Shape: (1, 30, 458)

    if os.path.exists(MODEL_TFLITE_PATH):
        with open(MODEL_TFLITE_PATH, "rb") as _f:
            interpreter = tf.lite.Interpreter(model_content=_f.read())
        interpreter.allocate_tensors()
        in_idx = interpreter.get_input_details()[0]['index']
        out_idx = interpreter.get_output_details()[0]['index']
        interpreter.set_tensor(in_idx, input_data)
        interpreter.invoke()
        probabilities = interpreter.get_tensor(out_idx)[0]
        engine_name = "Pure TFLite"
    else:
        model = tf.keras.models.load_model(MODEL_KERAS_PATH)
        probabilities = model.predict(input_data, verbose=0)[0]
        engine_name = "Keras"
    infer_time_ms = (time.time() - t1) * 1000.0

    top_indices = np.argsort(probabilities)[::-1][:top_k]

    print("\n" + "=" * 55)
    print(f"KẾT QUẢ DỰ ĐOÁN ({engine_name} - Suy luận: {infer_time_ms:.2f} ms):")
    print("=" * 55)
    for rank, idx in enumerate(top_indices, 1):
        label = idx_to_label.get(idx, f"Lớp {idx}")
        score = probabilities[idx] * 100
        bar = "█" * int(score / 5)
        print(f"  Top {rank}: {label:<20} {score:6.2f}%  {bar}")
    print(f"\nThời gian trích xuất MediaPipe (30 frames): {mp_time_ms:.1f} ms ({mp_time_ms/30:.1f} ms/frame)")
    return idx_to_label.get(top_indices[0], ""), float(probabilities[top_indices[0]])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        video = sys.argv[1]
    else:
        # Nếu không truyền tham số, tự động mở hộp thoại chọn tệp video
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            print("Đang mở hộp thoại chọn video từ máy tính...")
            video = filedialog.askopenfilename(
                title="Chọn video ngôn ngữ ký hiệu cần nhận diện",
                initialdir=os.path.join(os.path.dirname(__file__), "dataset", "public_test"),
                filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All Files", "*.*")]
            )
            root.destroy()
        except Exception:
            video = rf"{BASE_DIR}\dataset\public_test\100708.mp4"

    if video and os.path.exists(video):
        predict_video(video)
    else:
        print("Chưa chọn video hợp lệ hoặc tệp không tồn tại.")
