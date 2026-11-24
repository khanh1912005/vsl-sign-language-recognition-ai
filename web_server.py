r"""
=====================================================================
MÁY CHỦ WEB ĐA NGƯỜI DÙNG THỜI GIAN THỰC - VSL AI (web_server.py)
=====================================================================
Kiến trúc tối ưu chịu tải cao & Độ chính xác cao:
  1. Đẩy việc nặng về trình duyệt (Edge Computing):
     Trình duyệt tự bật Webcam (chuẩn hóa khung vuông 1:1 480x480 như tập
     huấn luyện 224x224) và chạy MediaPipe JS lấy tọa độ khung xương.
  2. Nạp mô hình Pure TFLite (model.tflite) ĐÚNG 1 LẦN DUY NHẤT vào RAM
     khi khởi động máy chủ (Singleton Interpreter).
  3. Xử lý xoay vòng theo mã phiên (session_id):
     - Luồng Webcam: tự động khớp tay trái/phải theo cổ tay Pose, nội suy
       khung hình thiếu, hỗ trợ nhận diện đối xứng gương (Mirror-Aware)
       và lọc mượt xác suất theo từng session_id.
     - Luồng Tải Video (Upload): tách biệt hoàn toàn (không trộn xác suất
       với video trước), đồng thời giải mã khung hình video + vẽ khung xương
       MediaPipe trả về trình duyệt để phát mượt mọi định dạng (kể cả FMP4).

Cách chạy:
    cd /d "D:\thực chiến\Nhận diện xử lý ngôn ngữ ký hiệu"
    python web_server.py
Sau đó mở trình duyệt tại: http://localhost:8000
=====================================================================
"""

import os
import sys
import time
import json
import base64
import hashlib
import secrets
import tempfile
import threading
from collections import deque
from typing import List, Optional, Dict, Any

sys.stdout.reconfigure(encoding='utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Header, Query, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from feature_engineering import extract_sequence_features
from predict import extract_landmarks, TOTAL_RAW_FEATURES
import mediapipe as mp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

TFLITE_PATH = os.path.join(BASE_DIR, "models", "model.tflite")
TASK_PATH = os.path.join(BASE_DIR, "holistic_landmarker.task")
LABEL_MAPPING_PATH = os.path.join(BASE_DIR, "processed_data", "label_mapping.json")

# Giới hạn tài nguyên an toàn cho xử lý Video tải lên (Chống cạn kiệt RAM / CPU)
MAX_UPLOAD_BYTES = 25 * 1024 * 1024        # Tối đa 25 MB cho mỗi tệp video tải lên
UPLOAD_CHUNK_SIZE = 64 * 1024              # Đọc theo từng khối 64 KB
MAX_VIDEO_DURATION_SEC = 25.0              # Tối đa 25 giây cho video cử chỉ đơn
MAX_VIDEO_SCAN_FRAMES = 600                # Quét tối đa 600 khung hình
MAX_DECODED_BUFFER_FRAMES = 60             # Giữ tối đa 60 khung hình đã thu nhỏ trong RAM
MAX_FRAME_DIMENSION = 480                  # Thu nhỏ khung hình nếu cạnh lớn nhất > 480px
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".webm", ".mkv"}

# =====================================================================
# 1. NẠP MÔ HÌNH TFLITE & BẢNG NHÃN 1 LẦN DUY NHẤT VÀO RAM KHI KHỞI ĐỘNG
# =====================================================================
print("=" * 70)
print("KHỞI ĐỘNG MÁY CHỦ WEB ĐA NGƯỜI DÙNG VSL AI (SINGLETON TFLITE ENGINE)")
print("=" * 70)

if not os.path.exists(LABEL_MAPPING_PATH):
    raise RuntimeError(f"Thiếu tệp bảng nhãn bắt buộc: {LABEL_MAPPING_PATH}")
with open(LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
    label_mapping = json.load(f)
IDX_TO_LABEL: Dict[int, str] = {int(v): k for k, v in label_mapping.items()}
NUM_CLASSES = len(IDX_TO_LABEL)

if not os.path.exists(TFLITE_PATH):
    raise RuntimeError(f"Thiếu tệp mô hình TFLite bắt buộc: {TFLITE_PATH}")
t_load = time.perf_counter()
with open(TFLITE_PATH, "rb") as _f:
    TFLITE_INTERPRETER = tf.lite.Interpreter(model_content=_f.read(), num_threads=4)
TFLITE_INTERPRETER.allocate_tensors()
IN_IDX = TFLITE_INTERPRETER.get_input_details()[0]["index"]
OUT_IDX = TFLITE_INTERPRETER.get_output_details()[0]["index"]
TFLITE_LOCK = threading.Lock()
print(f"[OK] Đã nạp mô hình TFLite 1 lần duy nhất vào RAM trong {(time.perf_counter() - t_load)*1000:.2f} ms")
print(f"[OK] Đường dẫn model: {TFLITE_PATH} ({NUM_CLASSES} lớp từ vựng)")

MP_LOCK = threading.Lock()
if not os.path.exists(TASK_PATH):
    raise RuntimeError(f"Thiếu tệp mô hình MediaPipe Holistic bắt buộc: {TASK_PATH}")
with open(TASK_PATH, "rb") as _f:
    _mp_options = mp.tasks.vision.HolisticLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=_f.read()),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
    )
SERVER_LANDMARKER = mp.tasks.vision.HolisticLandmarker.create_from_options(_mp_options)

# Nạp mmap tập đặc trưng kiểm thử để phục vụ kiểm thử chịu tải đa người dùng & demo mẫu
X_FEATURES_PATH = os.path.join(BASE_DIR, "processed_data", "X_features.npy")
SPLIT_INFO_PATH = os.path.join(BASE_DIR, "processed_data", "split_info.json")
METADATA_PATH = os.path.join(BASE_DIR, "processed_data", "metadata.json")

SAMPLE_POOL = []
SAMPLE_VIDEOS_LIST = []
if os.path.exists(X_FEATURES_PATH) and os.path.exists(SPLIT_INFO_PATH) and os.path.exists(METADATA_PATH):
    try:
        _X_mmap = np.load(X_FEATURES_PATH, mmap_mode="r")
        with open(SPLIT_INFO_PATH, "r", encoding="utf-8") as f:
            _split = json.load(f)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            _meta = json.load(f)
        seen_labels = set()
        for idx in _split.get("test_indices_orig", []):
            lbl = _meta[idx]["label_name"]
            vpath = _meta[idx]["path"]
            if not os.path.isabs(vpath):
                # metadata.json lưu đường dẫn tương đối theo thư mục dự án (dataset\train\...)
                vpath = os.path.join(BASE_DIR, vpath)
            if lbl not in seen_labels:
                seen_labels.add(lbl)
                SAMPLE_POOL.append({
                    "true_label": lbl,
                    "features": np.round(_X_mmap[idx].astype(np.float32), 4).tolist(),
                })
                if os.path.exists(vpath) and len(SAMPLE_VIDEOS_LIST) < 15:
                    SAMPLE_VIDEOS_LIST.append({
                        "id": len(SAMPLE_VIDEOS_LIST),
                        "filename": _meta[idx]["filename"],
                        "true_label": lbl,
                        "path": vpath,
                    })
            if len(SAMPLE_POOL) >= 24:
                break
        # Thêm các video từ public_test vào danh sách chọn nhanh
        pub_dir = os.path.join(BASE_DIR, "dataset", "public_test")
        if os.path.exists(pub_dir):
            for fn in sorted(os.listdir(pub_dir))[:6]:
                if fn.lower().endswith(".mp4"):
                    SAMPLE_VIDEOS_LIST.insert(0, {
                        "id": len(SAMPLE_VIDEOS_LIST) + 100,
                        "filename": f"public_test/{fn}",
                        "true_label": "Tập Public Test",
                        "path": os.path.join(pub_dir, fn),
                    })
        print(f"[OK] Đã chuẩn bị {len(SAMPLE_POOL)} mẫu từ vựng đa dạng (30x458) từ tập Test.")
    except Exception as e:
        print(f"[WARN] Không thể nạp SAMPLE_POOL: {e}")


# =====================================================================
# 2. CÁC HÀM CHUẨN HÓA HÌNH HỌC CHO WEBCAM & GIẢI MÃ KHUNG HÌNH VIDEO
# =====================================================================
POSE_SYMMETRIC_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20),
    (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)
]

HAND_CONNECTIONS_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

UPPER_POSE_PAIRS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24)
]


def align_hands_to_pose_wrists(raw_seq: np.ndarray) -> np.ndarray:
    """
    Đảm bảo bàn tay trong vùng nhớ LH (132:195) luôn gắn với cổ tay trái Pose[15]
    và bàn tay trong vùng nhớ RH (195:258) luôn gắn với cổ tay phải Pose[16],
    khắc phục hiện tượng MediaPipe JS trên trình duyệt đảo nhãn Left/Right Hand.
    """
    out = raw_seq.copy()
    T = out.shape[0]
    for t in range(T):
        pose = out[t, :132].reshape(33, 4)
        if np.abs(pose[15, :2]).sum() < 1e-4 or np.abs(pose[16, :2]).sum() < 1e-4:
            continue
        p_lw = pose[15, :2]
        p_rw = pose[16, :2]

        lh = out[t, 132:195].reshape(21, 3).copy()
        rh = out[t, 195:258].reshape(21, 3).copy()
        lh_ok = np.abs(lh).sum() > 1e-4
        rh_ok = np.abs(rh).sum() > 1e-4

        if lh_ok and rh_ok:
            # Nếu LH đang gần cổ tay phải Pose[16] hơn và RH đang gần cổ tay trái Pose[15] hơn -> Hoán đổi
            d_normal = np.linalg.norm(lh[0, :2] - p_lw) + np.linalg.norm(rh[0, :2] - p_rw)
            d_swapped = np.linalg.norm(lh[0, :2] - p_rw) + np.linalg.norm(rh[0, :2] - p_lw)
            if d_swapped + 0.03 < d_normal:
                out[t, 132:195] = rh.flatten()
                out[t, 195:258] = lh.flatten()
        elif lh_ok and not rh_ok:
            if np.linalg.norm(lh[0, :2] - p_rw) + 0.04 < np.linalg.norm(lh[0, :2] - p_lw):
                out[t, 195:258] = lh.flatten()
                out[t, 132:195] = 0.0
        elif rh_ok and not lh_ok:
            if np.linalg.norm(rh[0, :2] - p_lw) + 0.04 < np.linalg.norm(rh[0, :2] - p_rw):
                out[t, 132:195] = rh.flatten()
                out[t, 195:258] = 0.0
    return out


def flip_horizontal_raw_seq(raw_seq: np.ndarray) -> np.ndarray:
    """Tạo chuỗi tọa độ đối xứng gương (x -> 1 - x, hoán đổi khớp trái/phải) cho Webcam."""
    T = raw_seq.shape[0]
    pose = raw_seq[:, :132].reshape(T, 33, 4).copy()
    lh = raw_seq[:, 132:195].reshape(T, 21, 3).copy()
    rh = raw_seq[:, 195:258].reshape(T, 21, 3).copy()

    p_mask = np.abs(pose[:, :, :3]).sum(axis=-1) > 1e-5
    pose[:, :, 0] = np.where(p_mask, 1.0 - pose[:, :, 0], 0.0)
    for a, b in POSE_SYMMETRIC_PAIRS:
        pose[:, [a, b], :] = pose[:, [b, a], :]

    lh_mask = np.abs(lh).sum(axis=-1) > 1e-5
    rh_mask = np.abs(rh).sum(axis=-1) > 1e-5
    lh[:, :, 0] = np.where(lh_mask, 1.0 - lh[:, :, 0], 0.0)
    rh[:, :, 0] = np.where(rh_mask, 1.0 - rh[:, :, 0], 0.0)

    # Khi lật gương: bàn tay trái trở thành bàn tay phải và ngược lại
    return np.concatenate([
        pose.reshape(T, 132),
        rh.reshape(T, 63),
        lh.reshape(T, 63),
    ], axis=-1).astype(np.float32)


def draw_skeleton_on_frame(frame_bgr: np.ndarray, raw_258: np.ndarray) -> np.ndarray:
    """Vẽ khung xương Pose + 2 bàn tay lên khung hình video để hiển thị trực quan trên Web."""
    canvas = frame_bgr.copy()
    h, w = canvas.shape[:2]
    pose = raw_258[:132].reshape(33, 4)
    lh = raw_258[132:195].reshape(21, 3)
    rh = raw_258[195:258].reshape(21, 3)

    if np.abs(pose[:, :3]).sum() > 1e-4:
        for a, b in UPPER_POSE_PAIRS:
            if pose[a, 3] > 0.2 and pose[b, 3] > 0.2:
                pt1 = (int(pose[a, 0] * w), int(pose[a, 1] * h))
                pt2 = (int(pose[b, 0] * w), int(pose[b, 1] * h))
                cv2.line(canvas, pt1, pt2, (248, 189, 56), 2, cv2.LINE_AA)

    for hand_pts, color in [(lh, (129, 185, 16)), (rh, (11, 158, 245))]:
        if np.abs(hand_pts).sum() > 1e-4:
            for a, b in HAND_CONNECTIONS_PAIRS:
                pt1 = (int(hand_pts[a, 0] * w), int(hand_pts[a, 1] * h))
                pt2 = (int(hand_pts[b, 0] * w), int(hand_pts[b, 1] * h))
                cv2.line(canvas, pt1, pt2, color, 2, cv2.LINE_AA)
            for j in range(21):
                pt = (int(hand_pts[j, 0] * w), int(hand_pts[j, 1] * h))
                cv2.circle(canvas, pt, 3, (255, 255, 255), -1, cv2.LINE_AA)

    return canvas


def extract_video_landmarks_and_preview(video_path: str):
    """
    Đọc video (.mp4 FMP4, H264, .avi, .mov, .webm) với cơ chế bảo vệ RAM & CPU:
    - Kiểm tra thời lượng và số khung hình tối đa.
    - Không lưu toàn bộ khung hình gốc vào RAM: thu nhỏ khung hình quá khổ ngay khi đọc
      và dùng bộ đệm lấy mẫu phân tầng tối đa 60 khung hình (kết hợp cap.grab() bỏ qua
      giải mã các khung hình trung gian khi video dài), sau đó trích xuất đúng 30 frame
      cho MediaPipe Holistic và ảnh preview JPEG Base64.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Không thể mở video: {video_path}")

    header_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1 or fps > 120:
        fps = 14.0

    if header_frames > 0 and (header_frames / fps) > MAX_VIDEO_DURATION_SEC and header_frames > MAX_VIDEO_SCAN_FRAMES:
        cap.release()
        raise ValueError(
            f"Video quá dài ({header_frames / fps:.1f}s). Vui lòng tải lên video cử chỉ có thời lượng tối đa {int(MAX_VIDEO_DURATION_SEC)} giây."
        )

    sampled_frames = []
    stride = 1
    n_actual = 0

    try:
        while cap.isOpened():
            if n_actual >= MAX_VIDEO_SCAN_FRAMES:
                raise ValueError(
                    f"Video vượt quá giới hạn {MAX_VIDEO_SCAN_FRAMES} khung hình ({int(MAX_VIDEO_DURATION_SEC)} giây)."
                )

            if n_actual % stride == 0:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                h, w = frame.shape[:2]
                if max(h, w) > MAX_FRAME_DIMENSION:
                    scale = MAX_FRAME_DIMENSION / float(max(h, w))
                    frame = cv2.resize(
                        frame,
                        (max(1, int(w * scale)), max(1, int(h * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                sampled_frames.append(frame)
                if len(sampled_frames) >= MAX_DECODED_BUFFER_FRAMES:
                    sampled_frames = sampled_frames[::2]
                    stride *= 2
            else:
                ret = cap.grab()
                if not ret:
                    break

            n_actual += 1
    finally:
        cap.release()

    if len(sampled_frames) == 0:
        raise ValueError("Video không có khung hình nào hợp lệ")

    n_buf = len(sampled_frames)
    if n_buf >= 30:
        indices = np.linspace(0, n_buf - 1, 30, dtype=int)
    else:
        indices = list(range(n_buf))

    landmarks_seq = []
    preview_raw_b64 = []
    preview_skel_b64 = []

    with MP_LOCK:
        for idx in indices:
            frame = sampled_frames[idx]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            try:
                res = SERVER_LANDMARKER.detect(mp_image)
                lm = extract_landmarks(res)
            except Exception:
                lm = np.zeros(TOTAL_RAW_FEATURES, dtype=np.float32)
            landmarks_seq.append(lm)

            # Phóng lên 320x320 sắc nét để hiển thị trên trình duyệt
            disp_clean = cv2.resize(frame, (320, 320), interpolation=cv2.INTER_LINEAR)
            disp_skel = draw_skeleton_on_frame(disp_clean, lm)

            _, buf_clean = cv2.imencode(".jpg", disp_clean, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            _, buf_skel = cv2.imencode(".jpg", disp_skel, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            preview_raw_b64.append(base64.b64encode(buf_clean).decode("ascii"))
            preview_skel_b64.append(base64.b64encode(buf_skel).decode("ascii"))

    seq = np.array(landmarks_seq, dtype=np.float32)
    if len(seq) < 30:
        pad_len = 30 - len(seq)
        padding = np.tile(seq[-1], (pad_len, 1))
        seq = np.vstack([seq, padding])
    elif len(seq) > 30:
        seq = seq[:30]

    return seq, preview_raw_b64, preview_skel_b64, float(fps), n_actual


# =====================================================================
# 3. QUẢN LÝ TRẠNG THÁI XOAY VÒNG THEO MÃ PHIÊN (SESSION_ID) TRONG RAM
# =====================================================================
def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _anonymize_session_id(session_id: str) -> str:
    """Tạo mã hiển thị ẩn danh một chiều cho /api/stats, không bao giờ lộ session_id thật."""
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:8]
    return f"anon_{digest}"


class UserSessionState:
    """Lưu trạng thái riêng biệt của từng người dùng theo session_id kèm xác thực token."""
    def __init__(self, session_id: str, session_token: Optional[str] = None):
        self.session_id = session_id
        self.public_tag = _anonymize_session_id(session_id)
        self.token_hash: Optional[str] = _hash_token(session_token) if session_token else None
        self.created_at = time.time()
        self.last_active = time.time()
        self.prob_history = deque(maxlen=2)  # Làm mượt nhẹ 2 cửa sổ gần nhất cho Webcam
        self.draft_word: str = ""
        self.draft_streak: int = 0
        self.final_words: List[str] = []
        self.total_requests: int = 0
        self.last_infer_ms: float = 0.0


SESSIONS: Dict[str, UserSessionState] = {}
SESSIONS_LOCK = threading.Lock()
SESSION_TTL_SEC = 1800

SERVER_METRICS = {
    "started_at": time.time(),
    "total_inferences": 0,
    "total_infer_time_ms": 0.0,
    "last_infer_ms": 0.0,
    "peak_concurrent_sessions": 0,
}


def get_or_create_session(session_id: str, session_token: Optional[str] = None) -> UserSessionState:
    if not session_id or len(session_id) > 96:
        raise ValueError("Mã phiên (session_id) không hợp lệ.")
    now = time.time()
    with SESSIONS_LOCK:
        if len(SESSIONS) > 80:
            expired = [sid for sid, st in SESSIONS.items() if now - st.last_active > SESSION_TTL_SEC]
            for sid in expired:
                del SESSIONS[sid]

        if session_id not in SESSIONS:
            SESSIONS[session_id] = UserSessionState(session_id, session_token=session_token)
            st = SESSIONS[session_id]
        else:
            st = SESSIONS[session_id]
            if st.token_hash is not None:
                candidate_hash = _hash_token(session_token or "")
                if not secrets.compare_digest(st.token_hash, candidate_hash):
                    raise PermissionError("Từ chối truy cập: session_token không hợp lệ cho phiên này.")
            elif session_token:
                st.token_hash = _hash_token(session_token)

        st.last_active = now
        if len(SESSIONS) > SERVER_METRICS["peak_concurrent_sessions"]:
            SERVER_METRICS["peak_concurrent_sessions"] = len(SESSIONS)
        return st


def smooth_vietnamese_sentence(words: List[str]) -> str:
    if not words:
        return ""
    raw = " ".join(words)
    text = raw.capitalize()
    if not text.endswith("."):
        text += "."
    return text


def _invoke_tflite_single(feat_seq_30x458: np.ndarray) -> np.ndarray:
    x_in = np.expand_dims(feat_seq_30x458, axis=0).astype(np.float32)
    TFLITE_INTERPRETER.set_tensor(IN_IDX, x_in)
    TFLITE_INTERPRETER.invoke()
    return TFLITE_INTERPRETER.get_tensor(OUT_IDX)[0].copy()


def run_tflite_inference_for_session(
    session_id: str,
    matrix: np.ndarray,
    confidence_threshold: float = 0.55,
    is_isolated_video: bool = False,
    mirror_assist: bool = True,
    session_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Hàm suy luận trung tâm xoay vòng theo session_id:
    - Kiểm tra quyền truy cập phiên (session_token) trước khi thực thi.
    - Nếu là Video tải lên (is_isolated_video=True): đánh giá độc lập 100% trên
      video đó (tuyệt đối không trộn xác suất với video trước).
    - Nếu là Webcam gửi chuỗi 258 chiều: tự động khớp tay trái/phải theo cổ tay
      Pose, trích xuất đặc trưng 458 chiều chuẩn Python và kiểm tra đối xứng gương
      (Mirror-Aware) để đạt độ chính xác cao nhất.
    """
    st = get_or_create_session(session_id, session_token=session_token)

    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Ma trận đầu vào phải có 2 chiều (T, D), nhận được shape={arr.shape}")

    if arr.shape[0] != 30:
        idx = np.linspace(0, arr.shape[0] - 1, 30, dtype=int)
        arr = arr[idx]

    is_raw_258 = (arr.shape[1] == 258)
    if is_raw_258:
        if not is_isolated_video:
            arr = align_hands_to_pose_wrists(arr)
        feat_seq = extract_sequence_features(arr)
        feat_seq_mirror = extract_sequence_features(flip_horizontal_raw_seq(arr)) if (mirror_assist and not is_isolated_video) else None
    elif arr.shape[1] == 458:
        feat_seq = np.clip(arr, -10.0, 10.0)
        feat_seq_mirror = None
    else:
        raise ValueError(f"Số chiều đặc trưng phải là 458 hoặc 258, nhận được {arr.shape[1]}")

    with TFLITE_LOCK:
        t_infer = time.perf_counter()
        probs = _invoke_tflite_single(feat_seq)
        if feat_seq_mirror is not None:
            probs_mirror = _invoke_tflite_single(feat_seq_mirror)
            # Nếu người dùng trên Webcam đang làm động tác theo chiều gương có độ tự tin vượt trội -> dùng bản khớp gương
            if float(np.max(probs_mirror)) > float(np.max(probs)) + 0.08:
                probs = probs_mirror
        infer_ms = (time.perf_counter() - t_infer) * 1000.0

    st.total_requests += 1
    st.last_infer_ms = infer_ms

    if is_isolated_video:
        # Video tải lên hoặc mẫu thử độc lập: KHÔNG trộn xác suất với cửa sổ cũ
        eval_probs = probs
    else:
        st.prob_history.append(probs)
        eval_probs = np.mean(st.prob_history, axis=0)

    top_indices = np.argsort(eval_probs)[::-1][:5]
    top5 = [
        {
            "rank": i + 1,
            "class_id": int(idx),
            "label": IDX_TO_LABEL.get(int(idx), f"Lớp {idx}"),
            "confidence": round(float(eval_probs[idx]) * 100.0, 2),
        }
        for i, idx in enumerate(top_indices)
    ]

    best_label = top5[0]["label"]
    best_conf = float(eval_probs[top_indices[0]])

    committed_new_word = False
    if not is_isolated_video:
        if best_conf >= confidence_threshold:
            if best_label == st.draft_word:
                st.draft_streak += 1
            else:
                st.draft_word = best_label
                st.draft_streak = 1

            # Nếu độ tự tin rất cao (>= 78%) hoặc lặp lại >= 2 cửa sổ -> chốt vào câu
            if st.draft_streak >= 2 or best_conf >= 0.78:
                if best_label.lower() in ["xóa", "xoa", "hủy"]:
                    if st.final_words:
                        st.final_words.pop()
                else:
                    if not st.final_words or st.final_words[-1] != best_label:
                        st.final_words.append(best_label)
                        committed_new_word = True
                        if len(st.final_words) > 25:
                            st.final_words.pop(0)
        else:
            st.draft_word = ""
            st.draft_streak = 0

    SERVER_METRICS["total_inferences"] += 1
    SERVER_METRICS["total_infer_time_ms"] += infer_ms
    SERVER_METRICS["last_infer_ms"] = infer_ms

    return {
        "session_id": session_id,
        "infer_ms": round(infer_ms, 2),
        "top1_label": best_label,
        "top1_confidence": round(best_conf * 100.0, 2),
        "above_threshold": best_conf >= confidence_threshold,
        "draft_word": st.draft_word,
        "draft_streak": st.draft_streak,
        "committed_new_word": committed_new_word,
        "final_words": st.final_words,
        "nlp_sentence": smooth_vietnamese_sentence(st.final_words),
        "top5": top5,
        "session_requests": st.total_requests,
        "active_sessions": len(SESSIONS),
    }


# =====================================================================
# 4. KHỞI TẠO FASTAPI SERVER & CÁC ENDPOINT REST / WEBSOCKET
# =====================================================================
app = FastAPI(
    title="VSL AI - Hệ thống Web Nhận diện Ngôn ngữ Ký hiệu Đa người dùng",
    description="Singleton TFLite + Browser MediaPipe JS + Session-based Round-Robin Inference",
    version="2.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictWindowRequest(BaseModel):
    session_id: str
    features: List[List[float]]  # Ma trận 30 x 458 hoặc 30 x 258
    confidence_threshold: Optional[float] = 0.55
    is_isolated_video: Optional[bool] = False
    mirror_assist: Optional[bool] = True
    session_token: Optional[str] = None


class SessionActionRequest(BaseModel):
    session_id: str
    action: str
    session_token: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def index_page():
    index_path = os.path.join(STATIC_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/predict_window")
def api_predict_window(
    req: PredictWindowRequest,
    x_session_token: Optional[str] = Header(default=None),
):
    token = req.session_token or x_session_token
    try:
        res = run_tflite_inference_for_session(
            session_id=req.session_id,
            matrix=np.array(req.features, dtype=np.float32),
            confidence_threshold=req.confidence_threshold if req.confidence_threshold is not None else 0.55,
            is_isolated_video=bool(req.is_isolated_video),
            mirror_assist=True if req.mirror_assist is None else bool(req.mirror_assist),
            session_token=token,
        )
        return res
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/session/action")
def api_session_action(
    req: SessionActionRequest,
    x_session_token: Optional[str] = Header(default=None),
):
    token = req.session_token or x_session_token
    try:
        st = get_or_create_session(req.session_id, session_token=token)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if req.action == "clear":
        st.final_words.clear()
        st.draft_word = ""
        st.draft_streak = 0
        st.prob_history.clear()
    elif req.action == "backspace":
        if st.final_words:
            st.final_words.pop()
    elif req.action == "commit_draft":
        if st.draft_word and (not st.final_words or st.final_words[-1] != st.draft_word):
            st.final_words.append(st.draft_word)
            st.draft_word = ""
            st.draft_streak = 0
    elif req.action == "reset_buffer":
        st.prob_history.clear()
        st.draft_word = ""
        st.draft_streak = 0

    return {
        "session_id": req.session_id,
        "draft_word": st.draft_word,
        "final_words": st.final_words,
        "nlp_sentence": smooth_vietnamese_sentence(st.final_words),
    }


def _process_video_file_sync(
    video_path: str,
    session_id: str,
    session_token: Optional[str],
    display_filename: str,
    true_label: Optional[str] = None,
) -> Dict[str, Any]:
    """Thực thi giải mã OpenCV + MediaPipe + TFLite trong worker threadpool để không chặn event loop."""
    t0 = time.perf_counter()
    raw_seq, frames_b64, skel_b64, fps, total_frames = extract_video_landmarks_and_preview(video_path)
    mp_ms = (time.perf_counter() - t0) * 1000.0

    res = run_tflite_inference_for_session(
        session_id=session_id,
        matrix=raw_seq,
        confidence_threshold=0.50,
        is_isolated_video=True,  # Độc lập 100%, không trộn với video cũ!
        mirror_assist=False,
        session_token=session_token,
    )
    res["filename"] = display_filename
    if true_label is not None:
        res["true_label"] = true_label
    res["mediapipe_extract_ms"] = round(mp_ms, 1)
    res["frames_b64"] = frames_b64
    res["skel_b64"] = skel_b64
    res["fps"] = round(fps, 1)
    res["total_frames"] = total_frames
    return res


@app.post("/api/upload_video")
async def api_upload_video(
    file: UploadFile = File(...),
    session_id: str = Form("default_upload"),
    session_token: Optional[str] = Form(None),
    x_session_token: Optional[str] = Header(default=None),
):
    """
    Nhận tệp video (.mp4, .avi, .mov, .webm, .mkv) tải lên:
    - Kiểm tra phần mở rộng và đọc theo từng chunk 64 KB (giới hạn tối đa 25 MB).
    - Đẩy tác vụ nặng OpenCV + MediaPipe + TFLite sang threadpool (run_in_threadpool)
      để không bao giờ chặn event loop của FastAPI và các kết nối WebSocket khác.
    """
    orig_name = file.filename or "video.mp4"
    suffix = (os.path.splitext(orig_name)[1] or ".mp4").lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng tệp '{suffix}' không được hỗ trợ. Chỉ chấp nhận: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}",
        )

    token = session_token or x_session_token
    total_bytes = 0
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Kích thước tệp vượt quá giới hạn an toàn ({MAX_UPLOAD_BYTES // (1024 * 1024)} MB).",
                    )
                tmp.write(chunk)

        if total_bytes == 0:
            raise HTTPException(status_code=400, detail="Tệp video tải lên bị rỗng (0 bytes).")

        return await run_in_threadpool(
            _process_video_file_sync,
            tmp_path,
            session_id,
            token,
            orig_name,
            None,
        )
    except HTTPException:
        raise
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Không thể xử lý video: {e}")
    finally:
        await file.close()
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@app.get("/api/sample_videos")
def api_list_sample_videos():
    """Trả về danh sách các video mẫu có sẵn trong dataset để người dùng bấm chọn thử ngay trên Web."""
    return {
        "videos": [
            {"id": v["id"], "filename": v["filename"], "true_label": v["true_label"]}
            for v in SAMPLE_VIDEOS_LIST
        ]
    }


@app.get("/api/predict_sample_video")
async def api_predict_sample_video(
    video_id: int = 0,
    session_id: str = "sample_vid",
    session_token: Optional[str] = Query(default=None),
    x_session_token: Optional[str] = Header(default=None),
):
    """Đọc video mẫu theo video_id trên threadpool, trả về luồng khung hình phát trên Web + kết quả dự đoán."""
    target = next((v for v in SAMPLE_VIDEOS_LIST if v["id"] == video_id), None)
    if not target and SAMPLE_VIDEOS_LIST:
        target = SAMPLE_VIDEOS_LIST[0]
    if not target or not os.path.exists(target["path"]):
        raise HTTPException(status_code=404, detail="Không tìm thấy video mẫu")

    token = session_token or x_session_token
    display_name = f"{target['filename']} (Nhãn gốc: {target['true_label']})"
    try:
        return await run_in_threadpool(
            _process_video_file_sync,
            target["path"],
            session_id,
            token,
            display_name,
            target["true_label"],
        )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Không thể xử lý video mẫu: {e}")


@app.get("/api/sample_matrix")
def api_sample_matrix(index: int = 0):
    if not SAMPLE_POOL:
        raise HTTPException(status_code=404, detail="Không có mẫu thử")
    return SAMPLE_POOL[index % len(SAMPLE_POOL)]


@app.get("/api/sample_pool")
def api_sample_pool(count: int = 24):
    if not SAMPLE_POOL:
        raise HTTPException(status_code=404, detail="Không có mẫu thử")
    return {"samples": SAMPLE_POOL[:max(1, min(count, len(SAMPLE_POOL)))]}


@app.get("/api/stats")
def api_server_stats():
    """
    Trả về số liệu thống kê toàn cục của máy chủ.
    Bảo mật: Chỉ trả về mã định danh ẩn danh một chiều (st.public_tag = 'anon_xxxxxxxx'),
    tuyệt đối không trả về session_id thực của người dùng để ngăn chặn đọc/sửa trái phép.
    """
    now = time.time()
    with SESSIONS_LOCK:
        active_list = [
            {
                "session_id": st.public_tag,
                "requests": st.total_requests,
                "last_infer_ms": round(st.last_infer_ms, 2),
                "words_count": len(st.final_words),
                "idle_sec": round(now - st.last_active, 1),
            }
            for _, st in list(SESSIONS.items())[-15:]
        ]
    total_inf = SERVER_METRICS["total_inferences"]
    avg_ms = (SERVER_METRICS["total_infer_time_ms"] / total_inf) if total_inf > 0 else 0.0
    return {
        "uptime_sec": round(now - SERVER_METRICS["started_at"], 1),
        "active_sessions": len(SESSIONS),
        "peak_concurrent_sessions": SERVER_METRICS["peak_concurrent_sessions"],
        "total_inferences": total_inf,
        "avg_infer_ms": round(avg_ms, 2),
        "last_infer_ms": round(SERVER_METRICS["last_infer_ms"], 2),
        "theoretical_max_rps": round(1000.0 / max(0.5, avg_ms), 0) if avg_ms > 0 else 350,
        "num_classes": NUM_CLASSES,
        "recent_sessions": active_list,
    }


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    token: Optional[str] = Query(default=None),
):
    await websocket.accept()
    try:
        get_or_create_session(session_id, session_token=token)
    except (PermissionError, ValueError) as e:
        await websocket.send_json({"type": "error", "message": str(e)})
        await websocket.close(code=1008)
        return

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "predict")
            msg_token = data.get("session_token") or token
            if msg_type == "predict":
                features = data.get("features")
                threshold = float(data.get("confidence_threshold", 0.55))
                mirror_assist = bool(data.get("mirror_assist", True))
                res = await run_in_threadpool(
                    run_tflite_inference_for_session,
                    session_id,
                    np.array(features, dtype=np.float32),
                    threshold,
                    False,
                    mirror_assist,
                    msg_token,
                )
                await websocket.send_json({"type": "prediction", "data": res})
            elif msg_type == "action":
                act = data.get("action", "clear")
                res = await run_in_threadpool(
                    api_session_action,
                    SessionActionRequest(session_id=session_id, action=act, session_token=msg_token),
                    msg_token,
                )
                await websocket.send_json({"type": "action_ack", "data": res})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    print("\n>>> Mở trình duyệt truy cập địa chỉ: http://localhost:8000 <<<\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
