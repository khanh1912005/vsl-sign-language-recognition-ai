"""
=============================================================
NHẬN DIỆN CHUỖI HÀNH ĐỘNG LIÊN TỤC (CONTINUOUS SIGN SEQUENCE DECODER)
=============================================================
Giải quyết bài toán dịch một chuỗi nhiều cử chỉ liên tiếp:
1. Phân tích tín hiệu hoạt động bàn tay (Hand Activity) & năng lượng
   chuyển động (Motion Energy) trên toàn bộ luồng video dài.
2. Tự động phát hiện khoảng nghỉ / điểm chuyển tiếp (Boundary Detection)
   kết hợp Cửa sổ trượt (Sliding Window) và Khử trùng lặp kiểu CTC
   (Confidence-based Non-Maximum Suppression & Collapse).
3. Đánh giá định lượng bằng độ đo Word Error Rate (WER) và Sentence Accuracy
   trên 100 chuỗi câu ghép từ tập Held-out Test độc lập.

Author: Le Gia Khanh - A48572
=============================================================
"""

import os
import sys
import json
import time
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from feature_engineering import extract_sequence_features
from predict import extract_landmarks, TOTAL_RAW_FEATURES
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.stdout.reconfigure(encoding='utf-8')

PROCESSED_DIR = rf"{BASE_DIR}\processed_data"
MODEL_DIR = rf"{BASE_DIR}\models"
MODEL_TFLITE_PATH = os.path.join(MODEL_DIR, "model.tflite")
LABEL_MAPPING_PATH = os.path.join(PROCESSED_DIR, "label_mapping.json")
HOLISTIC_TASK_PATH = rf"{BASE_DIR}\holistic_landmarker.task"
WINDOW_SIZE = 30


def compute_frame_activity(raw_seq):
    """
    Tính toán trạng thái hoạt động cử chỉ trên từng frame của chuỗi dài (T, 258):
    - hand_active: Có ít nhất 1 bàn tay được MediaPipe phát hiện HOẶC cổ tay
      trong Pose đưa lên vùng ký hiệu (cao hơn ngực/bụng trên: y < 0.75).
    - motion_energy: Tốc độ di chuyển của 2 bàn tay / cổ tay giữa các frame.
    """
    T = len(raw_seq)
    pose = raw_seq[:, :132].reshape(T, 33, 4)
    lh = raw_seq[:, 132:195].reshape(T, 21, 3)
    rh = raw_seq[:, 195:258].reshape(T, 21, 3)

    lh_present = np.abs(lh).sum(axis=(1, 2)) > 1e-4
    rh_present = np.abs(rh).sum(axis=(1, 2)) > 1e-4

    # Tính năng lượng chuyển động của cổ tay, khuỷu tay và đầu ngón tay
    tracks = np.concatenate([
        pose[:, [13, 14, 15, 16], :2].reshape(T, -1),
        lh[:, [0, 8, 12], :2].reshape(T, -1),
        rh[:, [0, 8, 12], :2].reshape(T, -1)
    ], axis=-1)
    vel = np.zeros(T, dtype=np.float32)
    vel[1:] = np.linalg.norm(tracks[1:] - tracks[:-1], axis=-1)
    if T > 1:
        vel[0] = vel[1]

    # Frame đang thực hiện ký hiệu khi có bàn tay HOẶC tay đang chuyển động
    active_mask = lh_present | rh_present | (vel > 0.003)

    return active_mask, vel


def segment_active_intervals(raw_seq, min_active_frames=12, min_idle_gap=5):
    """
    Tách chuỗi video dài thành các phân đoạn cử chỉ (active segments) dựa trên
    khoảng nghỉ (idle frames) hoặc điểm cực tiểu năng lượng chuyển động.
    """
    T = len(raw_seq)
    active_mask, vel = compute_frame_activity(raw_seq)

    # Làm mượt active_mask để lấp các khoảng rớt tay ngắn (< min_idle_gap)
    mask = active_mask.copy()
    idle_run = 0
    for t in range(T):
        if not mask[t]:
            idle_run += 1
        else:
            if 0 < idle_run < min_idle_gap:
                mask[t - idle_run:t] = True
            idle_run = 0

    # Gom các đoạn active liên tục
    raw_intervals = []
    in_seg = False
    start_t = 0
    for t in range(T):
        if mask[t] and not in_seg:
            in_seg = True
            start_t = t
        elif not mask[t] and in_seg:
            in_seg = False
            if t - start_t >= min_active_frames:
                raw_intervals.append((start_t, t))
    if in_seg and (T - start_t >= min_active_frames):
        raw_intervals.append((start_t, T))

    # Nếu một đoạn active quá dài (ví dụ người dùng làm nhiều cử chỉ liền nhau không hạ tay),
    # chia nhỏ tại các điểm cực tiểu vận tốc (velocity valleys) cách nhau ~25-35 frames
    final_intervals = []
    for (s, e) in raw_intervals:
        length = e - s
        if length <= 45:
            final_intervals.append((s, e))
        else:
            # Ước lượng số cử chỉ trong đoạn dài (mỗi cử chỉ chuẩn ~30 frames)
            n_sub = max(2, int(round(length / 30.0)))
            step = length / float(n_sub)
            cut_points = [s]
            for k in range(1, n_sub):
                expected_cut = int(round(s + k * step))
                search_l = max(cut_points[-1] + 18, expected_cut - 6)
                search_r = min(e - 18, expected_cut + 6)
                if search_l < search_r:
                    local_min = search_l + int(np.argmin(vel[search_l:search_r + 1]))
                    cut_points.append(local_min)
                else:
                    cut_points.append(expected_cut)
            cut_points.append(e)
            for k in range(len(cut_points) - 1):
                if cut_points[k + 1] - cut_points[k] >= min_active_frames:
                    final_intervals.append((cut_points[k], cut_points[k + 1]))

    return final_intervals


def resample_segment_to_30(seg_raw):
    """Lấy mẫu đều một phân đoạn bất kỳ về đúng 30 frames."""
    L = len(seg_raw)
    if L == WINDOW_SIZE:
        return seg_raw
    indices = np.linspace(0, L - 1, WINDOW_SIZE, dtype=int)
    return seg_raw[indices]


class TFLiteSignRecognizer:
    """Bộ suy luận nhanh sử dụng Pure TFLite Interpreter."""
    def __init__(self, tflite_path=MODEL_TFLITE_PATH, label_path=LABEL_MAPPING_PATH):
        with open(label_path, "r", encoding="utf-8") as f:
            label_mapping = json.load(f)
        self.idx_to_label = {v: k for k, v in label_mapping.items()}

        with open(tflite_path, "rb") as _f:
            self.interpreter = tf.lite.Interpreter(model_content=_f.read())
        self.interpreter.allocate_tensors()
        self.in_idx = self.interpreter.get_input_details()[0]['index']
        self.out_idx = self.interpreter.get_output_details()[0]['index']

    def predict_30_raw(self, raw_30):
        feat_30 = extract_sequence_features(raw_30)
        inp = np.expand_dims(feat_30, axis=0)
        self.interpreter.set_tensor(self.in_idx, inp)
        self.interpreter.invoke()
        probs = self.interpreter.get_tensor(self.out_idx)[0]
        best_idx = int(np.argmax(probs))
        return best_idx, self.idx_to_label[best_idx], float(probs[best_idx]), probs


def decode_continuous_landmarks(raw_seq, recognizer, conf_threshold=0.40, dedup_adjacent=True):
    """
    Giải mã chuỗi landmarks liên tục (T, 258) thành danh sách các từ tiếng Việt:
    1. Phân đoạn theo khoảng nghỉ & cực tiểu vận tốc.
    2. Dự đoán từng phân đoạn với ngưỡng tin cậy.
    3. Khử trùng lặp liên tiếp nếu cùng một cử chỉ kéo dài.
    """
    intervals = segment_active_intervals(raw_seq)
    decoded = []

    for (s, e) in intervals:
        # Căn tâm đoạn cử chỉ về cửa sổ 30 frames để giữ nguyên tỷ lệ thời gian như lúc huấn luyện
        length = e - s
        if length < WINDOW_SIZE:
            center = (s + e) // 2
            s_pad = max(0, center - WINDOW_SIZE // 2)
            e_pad = min( len(raw_seq), s_pad + WINDOW_SIZE)
            s_pad = max(0, e_pad - WINDOW_SIZE)
            seg_30 = resample_segment_to_30(raw_seq[s_pad:e_pad])
        else:
            seg_30 = resample_segment_to_30(raw_seq[s:e])
        cls_idx, word, conf, _ = recognizer.predict_30_raw(seg_30)
        if conf >= conf_threshold:
            if dedup_adjacent and len(decoded) > 0 and decoded[-1]["label_idx"] == cls_idx:
                # Nếu trùng từ ngay trước đó và khoảng cách rất sát nhau (< 6 frames), gộp lại
                if s - decoded[-1]["end_frame"] < 6:
                    decoded[-1]["end_frame"] = e
                    decoded[-1]["confidence"] = max(decoded[-1]["confidence"], conf)
                    continue
            decoded.append({
                "label_idx": cls_idx,
                "word": word,
                "confidence": round(conf, 4),
                "start_frame": int(s),
                "end_frame": int(e)
            })

    return decoded


def levenshtein_distance(seq1, seq2):
    """Tính khoảng cách chỉnh sửa Levenshtein để đo Word Error Rate (WER)."""
    m, n = len(seq1), len(seq2)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    for i in range(m + 1):
        dp[i, 0] = i
    for j in range(n + 1):
        dp[0, j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return int(dp[m, n])


def benchmark_continuous_sequences(num_sequences=100, seed=42):
    """
    Đánh giá định lượng khả năng dịch chuỗi hành động trên 100 câu ghép
    (mỗi câu gồm 3 đến 4 cử chỉ lấy từ tập Held-out Test độc lập).
    Kiểm thử cả 2 kịch bản:
      - Kịch bản A: Có khoảng nghỉ ngắn (8 frames hạ tay) giữa các cử chỉ
      - Kịch bản B: Thực hiện liên tiếp (chuyển tiếp mượt 3 frames giữa 2 cử chỉ)
    """
    with open(os.path.join(PROCESSED_DIR, "split_info.json"), "r", encoding="utf-8") as f:
        split = json.load(f)
    X_raw_all = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    y_all = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))

    test_idx = split["test_indices_orig"]
    X_test_raw = X_raw_all[test_idx]
    y_test = y_all[test_idx]

    recognizer = TFLiteSignRecognizer()
    rng = np.random.default_rng(seed)

    results = {}
    for mode_name, pause_len in [("with_pause", 8), ("continuous_no_pause", 0)]:
        total_words = 0
        total_edits = 0
        exact_sentence_matches = 0
        sample_examples = []

        t0 = time.time()
        for s_i in range(num_sequences):
            seq_len = int(rng.integers(3, 5))  # 3 hoặc 4 từ mỗi câu
            chosen = rng.choice(len(X_test_raw), size=seq_len, replace=False)
            gt_labels = [int(y_test[c]) for c in chosen]
            gt_words = [recognizer.idx_to_label[l] for l in gt_labels]

            # Ghép chuỗi landmarks
            chunks = []
            for k, c_idx in enumerate(chosen):
                chunks.append(X_test_raw[c_idx])
                if k < len(chosen) - 1 and pause_len > 0:
                    # Tạo khoảng nghỉ tự nhiên (pose giữ nguyên, 2 tay hạ xuống = 0)
                    idle_chunk = np.zeros((pause_len, TOTAL_RAW_FEATURES), dtype=np.float32)
                    idle_chunk[:, :132] = X_test_raw[c_idx][-1:, :132]
                    # Đặt cổ tay pose thấp xuống vùng hông
                    idle_chunk[:, 15*4 + 1] = 0.85
                    idle_chunk[:, 16*4 + 1] = 0.85
                    chunks.append(idle_chunk)

            long_raw = np.concatenate(chunks, axis=0)
            decoded = decode_continuous_landmarks(long_raw, recognizer, conf_threshold=0.25)
            pred_labels = [d["label_idx"] for d in decoded]
            pred_words = [d["word"] for d in decoded]

            edits = levenshtein_distance(gt_labels, pred_labels)
            total_words += len(gt_labels)
            total_edits += edits
            if gt_labels == pred_labels:
                exact_sentence_matches += 1

            if s_i < 5:
                sample_examples.append({
                    "ground_truth": " -> ".join(gt_words),
                    "predicted": " -> ".join(pred_words),
                    "word_edits": edits,
                    "exact_match": (gt_labels == pred_labels)
                })

        elapsed = time.time() - t0
        wer = total_edits / float(total_words)
        word_acc = max(0.0, 1.0 - wer)
        sent_acc = exact_sentence_matches / float(num_sequences)

        results[mode_name] = {
            "num_sequences": num_sequences,
            "total_words": total_words,
            "word_error_rate": round(wer, 4),
            "word_recognition_accuracy": round(word_acc, 4),
            "exact_sentence_accuracy": round(sent_acc, 4),
            "avg_decode_time_ms": round(elapsed / num_sequences * 1000.0, 2),
            "examples": sample_examples
        }

    out_file = os.path.join(MODEL_DIR, "sequence_evaluation.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("=" * 65)
    print("KẾT QUẢ KIỂM THỬ DỊCH CHUỖI HÀNH ĐỘNG LIÊN TỤC (100 CÂU TEST)")
    print("=" * 65)
    for mode_name, r in results.items():
        title = "Có khoảng nghỉ ngắn giữa các từ" if mode_name == "with_pause" else "Thực hiện liên tiếp không nghỉ"
        print(f"\n[{title}] ({r['total_words']} từ trên {r['num_sequences']} chuỗi):")
        print(f"  - Độ chính xác cấp từ (1 - WER):    {r['word_recognition_accuracy']*100:.2f}% (WER: {r['word_error_rate']*100:.2f}%)")
        print(f"  - Tỷ lệ khớp chính xác cả câu:      {r['exact_sentence_accuracy']*100:.2f}%")
        print(f"  - Thời gian giải mã trung bình:     {r['avg_decode_time_ms']:.2f} ms / câu")
        print("  - Ví dụ thực tế:")
        for ex in r["examples"][:3]:
            status = "ĐÚNG CẢ CÂU" if ex["exact_match"] else f"Sai {ex['word_edits']} từ"
            print(f"      Gốc:     {ex['ground_truth']}")
            print(f"      Dự đoán: {ex['predicted']}  [{status}]")
    print("=" * 65)
    return results


if __name__ == "__main__":
    benchmark_continuous_sequences()
