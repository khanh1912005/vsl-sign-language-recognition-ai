"""
=====================================================================
CHƯƠNG TRÌNH CHÍNH TÁC VỤ 2 - OLP'AI 2025 (main.py)
NHẬN DIỆN NGÔN NGỮ KÝ HIỆU TIẾNG VIỆT (100 LỚP)
---------------------------------------------------------------------
Sinh viên thực hiện:
  1. Lê Gia Khánh     - MSV: A48572 - Lớp: TA3601
  2. Bàng Nguyên Đăng - MSV: A48553 - Lớp: TA3601
---------------------------------------------------------------------
Chức năng:
  - Cố định toàn bộ Random Seed (SEED = 42) đảm bảo tái lập 100%.
  - Hỗ trợ dự đoán trên tập dữ liệu kiểm thử (public_test / private_test)
    hoặc một thư mục bất kỳ và xuất ra tệp submission.csv & output.csv.
=====================================================================
"""

import os
import sys
import csv
import json
import time
import random
import argparse
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1. Cố định Seed theo đúng quy chế Tác vụ 2 - OLP'AI 2025
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

import mediapipe as mp
import tensorflow as tf
tf.keras.utils.set_random_seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Hỗ trợ chạy từ thư mục gốc dự án hoặc từ thư mục con Final\Tac_vu_2\
PROJECT_ROOT = BASE_DIR if os.path.exists(os.path.join(BASE_DIR, "models")) else os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from predict import extract_video_raw_landmarks
from feature_engineering import extract_sequence_features

TFLITE_PATH = os.path.join(PROJECT_ROOT, "models", "model.tflite")
TASK_PATH = os.path.join(PROJECT_ROOT, "holistic_landmarker.task")
LABEL_MAPPING_PATH = os.path.join(PROJECT_ROOT, "processed_data", "label_mapping.json")


def load_idx_to_label():
    """Đọc bảng ánh xạ chỉ số lớp (0..99) sang nhãn ngôn ngữ ký hiệu tiếng Việt."""
    with open(LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
        label_mapping = json.load(f)
    return {int(v): k for k, v in label_mapping.items()}


def run_inference_on_folder(test_dir, output_csv_path, limit=None):
    """Chạy dự đoán cho các video trong thư mục test_dir và xuất ra file CSV."""
    idx_to_label = load_idx_to_label()
    with open(TFLITE_PATH, "rb") as _f:
        interpreter = tf.lite.Interpreter(model_content=_f.read())
    interpreter.allocate_tensors()
    in_idx = interpreter.get_input_details()[0]["index"]
    out_idx = interpreter.get_output_details()[0]["index"]

    videos = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".mp4")])
    if limit is not None and limit > 0:
        videos = videos[:limit]

    print("=" * 72)
    print("CHƯƠNG TRÌNH NHẬN DIỆN NGÔN NGỮ KÝ HIỆU - TÁC VỤ 2 (PURE TFLITE)")
    print("=" * 72)
    print(f"Thư mục đầu vào : {test_dir}")
    print(f"Số video xử lý  : {len(videos)} video")
    print(f"File kết quả    : {output_csv_path}")
    print("-" * 72)

    results = []
    t_start = time.time()

    BaseOptions = mp.tasks.BaseOptions
    HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
    HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    with open(TASK_PATH, "rb") as _f:
        options = HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=_f.read()),
            running_mode=VisionRunningMode.IMAGE,
        )

    with HolisticLandmarker.create_from_options(options) as landmarker:
        for idx, fname in enumerate(videos, 1):
            vpath = os.path.join(test_dir, fname)
            try:
                raw_seq = extract_video_raw_landmarks(vpath, landmarker)
                feat_seq = extract_sequence_features(raw_seq)
                x_in = np.expand_dims(feat_seq, axis=0).astype(np.float32)
                interpreter.set_tensor(in_idx, x_in)
                interpreter.invoke()
                probs = interpreter.get_tensor(out_idx)[0]
                pred_id = int(np.argmax(probs))
                pred_label = idx_to_label.get(pred_id, f"Lớp {pred_id}")
                conf = float(probs[pred_id])
            except Exception as e:
                pred_label = idx_to_label.get(0, "")
                conf = 0.0

            results.append((fname, pred_label, f"{conf*100:.2f}%"))
            print(f"  [{idx:03d}/{len(videos):03d}] {fname:<14} -> Nhãn dự đoán: {pred_label:<22} (Độ tin cậy: {conf*100:6.2f}%)")

    # Ghi ra submission.csv và output.csv chuẩn UTF-8
    with open(output_csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_name", "label"])
        for fname, lbl, _ in results:
            writer.writerow([fname, lbl])

    alt_csv = os.path.join(os.path.dirname(output_csv_path), "output.csv")
    if os.path.abspath(alt_csv) != os.path.abspath(output_csv_path):
        with open(alt_csv, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["video_name", "label"])
            for fname, lbl, _ in results:
                writer.writerow([fname, lbl])

    elapsed = time.time() - t_start
    print("-" * 72)
    print(f"Hoàn tất trong {elapsed:.2f}s! Đã xuất file kết quả tại:")
    print(f"  1. {output_csv_path}")
    print(f"  2. {alt_csv}")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy chương trình Tác vụ 2: Nhận diện Ngôn ngữ ký hiệu")
    parser.add_argument("--test_dir", type=str, default=os.path.join(PROJECT_ROOT, "dataset", "public_test"),
                        help="Đường dẫn thư mục chứa video kiểm thử (mặc định: dataset/public_test)")
    parser.add_argument("--out", type=str, default=os.path.join(PROJECT_ROOT, "submission.csv"),
                        help="Đường dẫn file CSV kết quả đầu ra (mặc định: submission.csv)")
    parser.add_argument("--limit", type=int, default=10,
                        help="Số lượng video chạy thử (nhập 0 để chạy toàn bộ thư mục)")
    args = parser.parse_args()

    limit_val = None if args.limit == 0 else args.limit
    run_inference_on_folder(args.test_dir, args.out, limit=limit_val)
