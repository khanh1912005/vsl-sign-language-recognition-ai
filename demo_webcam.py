"""
=============================================================
DEMO REAL-TIME WEBCAM - NHẬN DIỆN CHUỖI NGÔN NGỮ KÝ HIỆU (VSL)
=============================================================
Nhận diện cử chỉ đơn lẻ và chuỗi hành động liên tục từ Webcam:
- Sử dụng đặc trưng bất biến không gian - thời gian (458 chiều)
- Động cơ suy luận Pure TFLite siêu tốc (~1-2 ms)
- Tự động phát hiện trạng thái Nghỉ (Idle Detection) khi hạ tay,
  tránh đoán nhầm khi người dùng không thực hiện cử chỉ.
- Lọc mượt xác suất (Temporal Smoothing) & ghép câu tự động.

Phím tắt:
- 'q': Thoát chương trình
- 'c': Xóa lịch sử câu

Author: Le Gia Khanh - A48572
=============================================================
"""

import os
import sys
import time
import json
from collections import deque
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from feature_engineering import extract_sequence_features
from predict import extract_landmarks, TOTAL_RAW_FEATURES
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.stdout.reconfigure(encoding='utf-8')

MODEL_TFLITE_PATH = rf"{BASE_DIR}\models\model.tflite"
LABEL_MAPPING_PATH = rf"{BASE_DIR}\processed_data\label_mapping.json"
HOLISTIC_TASK_PATH = rf"{BASE_DIR}\holistic_landmarker.task"
MAX_FRAMES = 30
CONFIDENCE_THRESHOLD = 0.55


def put_vietnamese_text(img, text, position, font_size=24, color=(255, 255, 255)):
    """Vẽ chữ tiếng Việt có dấu lên khung hình OpenCV."""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def is_frame_active(raw_lm):
    """Kiểm tra người dùng có đang giơ tay thực hiện ký hiệu hay đang nghỉ."""
    lh = raw_lm[132:195]
    rh = raw_lm[195:258]
    if np.abs(lh).sum() > 1e-4 or np.abs(rh).sum() > 1e-4:
        return True
    pose = raw_lm[:132].reshape(33, 4)
    hip_y = 0.5 * (pose[23, 1] + pose[24, 1])
    if pose[15, 3] > 0.3 and (pose[15, 1] < hip_y - 0.05 or pose[16, 1] < hip_y - 0.05):
        return True
    return False


def main():
    print("=" * 60)
    print("KHỞI ĐỘNG HỆ THỐNG NHẬN DIỆN WEBCAM THỜI GIAN THỰC (PURE TFLITE)")
    print("=" * 60)

    with open(LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
        label_mapping = json.load(f)
    idx_to_label = {v: k for k, v in label_mapping.items()}

    with open(MODEL_TFLITE_PATH, "rb") as _f:
        interpreter = tf.lite.Interpreter(model_content=_f.read())
    interpreter.allocate_tensors()
    in_idx = interpreter.get_input_details()[0]['index']
    out_idx = interpreter.get_output_details()[0]['index']
    print(f"Đã nạp Pure TFLite model: {MODEL_TFLITE_PATH}")

    BaseOptions = mp.tasks.BaseOptions
    HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
    HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    with open(HOLISTIC_TASK_PATH, "rb") as _f:
        options = HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=_f.read()),
            running_mode=VisionRunningMode.IMAGE,
        )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("LỖI: Không tìm thấy thiết bị Webcam!")
        return

    frames_buffer = deque(maxlen=MAX_FRAMES)
    prob_history = deque(maxlen=5)
    sentence = []
    current_prediction = ""
    current_confidence = 0.0
    is_idle = True

    with HolisticLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            t_frame = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            try:
                res = landmarker.detect(mp_image)
                lm = extract_landmarks(res)
            except Exception:
                lm = np.zeros(TOTAL_RAW_FEATURES, dtype=np.float32)

            if is_frame_active(lm):
                is_idle = False
                frames_buffer.append(lm)
            else:
                # Nếu hạ tay nghỉ, xóa dần buffer để sẵn sàng cho cử chỉ mới
                if len(frames_buffer) > 0:
                    frames_buffer.clear()
                    prob_history.clear()
                is_idle = True
                current_prediction = ""
                current_confidence = 0.0

            if not is_idle and len(frames_buffer) >= 18:
                raw_arr = np.array(frames_buffer, dtype=np.float32)
                if len(raw_arr) < MAX_FRAMES:
                    idx_resample = np.linspace(0, len(raw_arr) - 1, MAX_FRAMES, dtype=int)
                    raw_arr = raw_arr[idx_resample]

                feat_seq = extract_sequence_features(raw_arr)
                inp = np.expand_dims(feat_seq, axis=0)
                interpreter.set_tensor(in_idx, inp)
                interpreter.invoke()
                probs = interpreter.get_tensor(out_idx)[0]
                prob_history.append(probs)

                avg_probs = np.mean(prob_history, axis=0)
                best_idx = int(np.argmax(avg_probs))
                current_confidence = float(avg_probs[best_idx])

                if current_confidence >= CONFIDENCE_THRESHOLD:
                    pred_label = idx_to_label.get(best_idx, "")
                    current_prediction = pred_label
                    if len(sentence) == 0 or sentence[-1] != pred_label:
                        sentence.append(pred_label)
                        if len(sentence) > 7:
                            sentence.pop(0)

            fps = 1.0 / max(1e-3, time.time() - t_frame)

            # Thanh trạng thái trên cùng
            cv2.rectangle(frame, (0, 0), (w, 82), (25, 35, 45), -1)
            if is_idle:
                status_txt = f"Trạng thái: Nghỉ / Chờ cử chỉ...  |  {fps:.1f} FPS"
                frame = put_vietnamese_text(frame, status_txt, (15, 14), font_size=24, color=(180, 200, 220))
            elif current_prediction:
                status_txt = f"Nhận diện: {current_prediction} ({current_confidence*100:.1f}%)  |  {fps:.1f} FPS"
                frame = put_vietnamese_text(frame, status_txt, (15, 14), font_size=26, color=(0, 255, 128))
            else:
                status_txt = f"Đang thu nhận cử chỉ ({len(frames_buffer)}/30)...  |  {fps:.1f} FPS"
                frame = put_vietnamese_text(frame, status_txt, (15, 14), font_size=24, color=(255, 215, 0))

            # Thanh hiển thị chuỗi câu phía dưới
            cv2.rectangle(frame, (0, h - 52), (w, h), (20, 20, 20), -1)
            sentence_text = "Chuỗi từ: " + (" -> ".join(sentence) if sentence else "(Chưa có từ)")
            frame = put_vietnamese_text(frame, sentence_text, (15, h - 42), font_size=22, color=(255, 255, 255))

            cv2.imshow("He thong Nhan dien Ngon ngu Ky hieu Tieng Viet (VSL)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                sentence.clear()
                frames_buffer.clear()
                prob_history.clear()
                current_prediction = ""

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
