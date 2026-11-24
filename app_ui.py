"""
=====================================================================
GIAO DIỆN ĐỒ HỌA TẢI VIDEO VÀ NHẬN DIỆN NGÔN NGỮ KÝ HIỆU (app_ui.py)
=====================================================================
Chức năng:
  - Người dùng nhấn nút "Tải video lên" để chọn file .mp4 từ máy tính.
  - Tự động phát video và trích xuất đặc trưng với MediaPipe Holistic.
  - Suy luận siêu tốc bằng mô hình Pure TFLite (25-30 ms).
  - Hiển thị nhãn dự đoán lớn, bảng Top 5 kèm thanh phần trăm trực quan.
=====================================================================
"""

import os
import sys
import time
import json
import threading
import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

sys.stdout.reconfigure(encoding='utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from predict import extract_video_raw_landmarks
from feature_engineering import extract_sequence_features
import mediapipe as mp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TFLITE_PATH = os.path.join(BASE_DIR, "models", "model.tflite")
TASK_PATH = os.path.join(BASE_DIR, "holistic_landmarker.task")
LABEL_MAPPING_PATH = os.path.join(BASE_DIR, "processed_data", "label_mapping.json")

# Load Label Mapping
with open(LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
    label_mapping = json.load(f)
idx_to_label = {int(v): k for k, v in label_mapping.items()}

# Load TFLite Model
with open(TFLITE_PATH, "rb") as _f:
    interpreter = tf.lite.Interpreter(model_content=_f.read())
interpreter.allocate_tensors()
in_idx = interpreter.get_input_details()[0]["index"]
out_idx = interpreter.get_output_details()[0]["index"]

# Prepare MediaPipe Task Options
BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

with open(TASK_PATH, "rb") as _f:
    options = HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=_f.read()),
        running_mode=VisionRunningMode.IMAGE,
    )
landmarker_engine = HolisticLandmarker.create_from_options(options)


class SignLanguageApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Nhận Diện Ngôn Ngữ Ký Hiệu Tiếng Việt - VSL AI")
        self.geometry("980x680")
        self.minsize(920, 640)
        self.configure(bg="#F4F6F9")

        self.current_video_path = None
        self.is_processing = False
        self.is_playing = False
        self.cap = None

        self._build_ui()

    def _build_ui(self):
        # Header Banner
        header = tk.Frame(self, bg="#1E3A8A", height=75)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        title_lbl = tk.Label(
            header,
            text="HỆ THỐNG NHẬN DIỆN NGÔN NGỮ KÝ HIỆU TIẾNG VIỆT",
            font=("Segoe UI", 16, "bold"),
            fg="white",
            bg="#1E3A8A"
        )
        title_lbl.pack(pady=(12, 2))

        subtitle_lbl = tk.Label(
            header,
            text="Mô hình Conv1D-BiGRU (TFLite) • 100 Cử chỉ • Đặc trưng bất biến 458 chiều",
            font=("Segoe UI", 10),
            fg="#BFDBFE",
            bg="#1E3A8A"
        )
        subtitle_lbl.pack()

        # Main Content Container
        main_frame = tk.Frame(self, bg="#F4F6F9", padx=20, pady=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left Column: Video Preview & Controls
        left_col = tk.Frame(main_frame, bg="white", relief=tk.RIDGE, bd=1, padx=15, pady=15)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        video_title = tk.Label(left_col, text="Xem Trước Video", font=("Segoe UI", 12, "bold"), bg="white", fg="#1F2937")
        video_title.pack(anchor="w", pady=(0, 10))

        self.canvas_w = 440
        self.canvas_h = 330
        self.video_canvas = tk.Canvas(left_col, width=self.canvas_w, height=self.canvas_h, bg="#111827", highlightthickness=0)
        self.video_canvas.pack(pady=5)
        self.video_canvas.create_text(
            self.canvas_w // 2, self.canvas_h // 2,
            text="Chưa có video được chọn\n\nNhấn 'Tải video lên' bên dưới",
            fill="#9CA3AF", font=("Segoe UI", 11), justify="center", tags="placeholder"
        )

        self.lbl_video_info = tk.Label(left_col, text="Tệp: (Chưa chọn)", font=("Segoe UI", 9), fg="#6B7280", bg="white", wraplength=420)
        self.lbl_video_info.pack(anchor="w", pady=(8, 12))

        # Buttons
        btn_frame = tk.Frame(left_col, bg="white")
        btn_frame.pack(fill=tk.X, pady=(5, 0))

        self.btn_upload = tk.Button(
            btn_frame,
            text="📂 Tải video lên (Upload)",
            font=("Segoe UI", 11, "bold"),
            bg="#2563EB",
            fg="white",
            activebackground="#1D4ED8",
            activeforeground="white",
            relief=tk.FLAT,
            padx=15,
            pady=8,
            cursor="hand2",
            command=self.choose_video
        )
        self.btn_upload.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        self.btn_replay = tk.Button(
            btn_frame,
            text="▶ Xem lại",
            font=("Segoe UI", 11),
            bg="#E5E7EB",
            fg="#374151",
            activebackground="#D1D5DB",
            relief=tk.FLAT,
            padx=12,
            pady=8,
            cursor="hand2",
            state=tk.DISABLED,
            command=self.replay_video
        )
        self.btn_replay.pack(side=tk.RIGHT, padx=(5, 0))

        # Status indicator
        self.lbl_status = tk.Label(left_col, text="Sẵn sàng tiếp nhận video.", font=("Segoe UI", 9, "italic"), fg="#059669", bg="white")
        self.lbl_status.pack(anchor="w", pady=(10, 0))

        # Right Column: Predictions & Details
        right_col = tk.Frame(main_frame, bg="white", relief=tk.RIDGE, bd=1, padx=15, pady=15)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        res_title = tk.Label(right_col, text="Kết Quả Nhận Diện", font=("Segoe UI", 12, "bold"), bg="white", fg="#1F2937")
        res_title.pack(anchor="w", pady=(0, 10))

        # Big Result Card
        pred_card = tk.Frame(right_col, bg="#EFF6FF", relief=tk.GROOVE, bd=1, padx=12, pady=12)
        pred_card.pack(fill=tk.X, pady=(0, 15))

        card_sub = tk.Label(pred_card, text="CỬ CHỈ DỰ ĐOÁN (TOP 1)", font=("Segoe UI", 9, "bold"), fg="#1E40AF", bg="#EFF6FF")
        card_sub.pack(anchor="w")

        self.lbl_top1 = tk.Label(pred_card, text="---", font=("Segoe UI", 24, "bold"), fg="#1D4ED8", bg="#EFF6FF")
        self.lbl_top1.pack(pady=4)

        self.lbl_conf = tk.Label(pred_card, text="Độ tin cậy: -- %", font=("Segoe UI", 11), fg="#374151", bg="#EFF6FF")
        self.lbl_conf.pack()

        # Top 5 Table
        top5_title = tk.Label(right_col, text="Xác Suất 5 Cử Chỉ Cao Nhất (Top 5)", font=("Segoe UI", 10, "bold"), bg="white", fg="#374151")
        top5_title.pack(anchor="w", pady=(5, 6))

        self.top5_bars = []
        for i in range(5):
            row = tk.Frame(right_col, bg="white", pady=3)
            row.pack(fill=tk.X)

            lbl_rank = tk.Label(row, text=f"#{i+1}", font=("Segoe UI", 9, "bold"), width=3, fg="#6B7280", bg="white", anchor="w")
            lbl_rank.pack(side=tk.LEFT)

            lbl_name = tk.Label(row, text="---", font=("Segoe UI", 9), width=16, anchor="w", bg="white", fg="#111827")
            lbl_name.pack(side=tk.LEFT, padx=(2, 8))

            progress = ttk.Progressbar(row, orient="horizontal", length=130, mode="determinate")
            progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

            lbl_val = tk.Label(row, text="0.0%", font=("Segoe UI", 9, "bold"), width=7, anchor="e", bg="white", fg="#2563EB")
            lbl_val.pack(side=tk.RIGHT)

            self.top5_bars.append((lbl_name, progress, lbl_val))

        # Benchmark / Timing Info
        timing_frame = tk.Frame(right_col, bg="#F9FAFB", relief=tk.GROOVE, bd=1, padx=10, pady=8)
        timing_frame.pack(fill=tk.X, pady=(18, 0))

        self.lbl_time_mp = tk.Label(timing_frame, text="Trích xuất khung xương (MediaPipe): -- ms", font=("Segoe UI", 8), fg="#4B5563", bg="#F9FAFB")
        self.lbl_time_mp.pack(anchor="w")

        self.lbl_time_infer = tk.Label(timing_frame, text="Thời gian suy luận mô hình (TFLite): -- ms", font=("Segoe UI", 8), fg="#4B5563", bg="#F9FAFB")
        self.lbl_time_infer.pack(anchor="w")

    def choose_video(self):
        if self.is_processing:
            messagebox.showwarning("Đang xử lý", "Hệ thống đang phân tích video trước đó, vui lòng đợi trong giây lát!")
            return

        filetypes = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
            ("Tất cả tệp", "*.*")
        ]
        chosen = filedialog.askopenfilename(
            title="Chọn video ngôn ngữ ký hiệu cần nhận diện",
            initialdir=os.path.join(BASE_DIR, "dataset", "public_test") if os.path.exists(os.path.join(BASE_DIR, "dataset", "public_test")) else BASE_DIR,
            filetypes=filetypes
        )
        if not chosen:
            return

        self.current_video_path = chosen
        self.lbl_video_info.config(text=f"Tệp: {os.path.basename(chosen)}\n({chosen})")
        self.btn_replay.config(state=tk.NORMAL)

        # Bắt đầu phát xem trước và nhận diện trong luồng phụ (thread)
        self.start_processing_thread(chosen)

    def start_processing_thread(self, video_path):
        self.is_processing = True
        self.lbl_status.config(text="⏳ Đang đọc video và trích xuất đặc trưng...", fg="#D97706")
        self.lbl_top1.config(text="Đang xử lý...", fg="#9CA3AF")
        self.lbl_conf.config(text="Vui lòng đợi ~1 giây")

        # Start playback in UI thread
        self.play_video(video_path)

        # Run AI processing in background
        t = threading.Thread(target=self._process_video_ai, args=(video_path,), daemon=True)
        t.start()

    def _process_video_ai(self, video_path):
        try:
            t0 = time.time()
            raw_seq = extract_video_raw_landmarks(video_path, landmarker_engine)
            time_mp = (time.time() - t0) * 1000.0

            t1 = time.time()
            feat_seq = extract_sequence_features(raw_seq)
            x_in = np.expand_dims(feat_seq, axis=0).astype(np.float32)

            interpreter.set_tensor(in_idx, x_in)
            interpreter.invoke()
            probs = interpreter.get_tensor(out_idx)[0]
            time_infer = (time.time() - t1) * 1000.0

            top_indices = np.argsort(probs)[::-1][:5]
            top_results = [(idx_to_label.get(idx, f"Lớp {idx}"), float(probs[idx])) for idx in top_indices]

            # Gửi kết quả về giao diện chính
            self.after(0, self._update_results_ui, top_results, time_mp, time_infer)

        except Exception as e:
            self.after(0, self._show_error, str(e))
        finally:
            self.is_processing = False

    def _update_results_ui(self, top_results, time_mp, time_infer):
        top1_name, top1_score = top_results[0]
        self.lbl_top1.config(text=top1_name, fg="#1D4ED8")
        self.lbl_conf.config(text=f"Độ tin cậy: {top1_score*100:.2f}%")

        for i, (name, score) in enumerate(top_results):
            lbl_name, bar, lbl_val = self.top5_bars[i]
            lbl_name.config(text=name)
            bar["value"] = score * 100
            lbl_val.config(text=f"{score*100:5.2f}%")

        self.lbl_time_mp.config(text=f"Trích xuất khung xương (MediaPipe 30 frames): {time_mp:.1f} ms ({time_mp/30:.1f} ms/frame)")
        self.lbl_time_infer.config(text=f"Thời gian suy luận mô hình (Pure TFLite): {time_infer:.2f} ms")
        self.lbl_status.config(text="✅ Nhận diện thành công!", fg="#059669")

    def _show_error(self, err_msg):
        self.lbl_status.config(text="❌ Có lỗi xảy ra khi xử lý video.", fg="#DC2626")
        messagebox.showerror("Lỗi nhận diện", f"Không thể xử lý video:\n{err_msg}")

    def replay_video(self):
        if self.current_video_path and os.path.exists(self.current_video_path):
            self.play_video(self.current_video_path)

    def play_video(self, video_path):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(video_path)
        self._next_frame()

    def _next_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame_rgb.shape
            scale = min(self.canvas_w / w, self.canvas_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(frame_rgb, (new_w, new_h))

            img_pil = Image.fromarray(resized)
            self.photo = ImageTk.PhotoImage(image=img_pil)

            self.video_canvas.delete("all")
            pad_x = (self.canvas_w - new_w) // 2
            pad_y = (self.canvas_h - new_h) // 2
            self.video_canvas.create_image(pad_x, pad_y, anchor=tk.NW, image=self.photo)

            fps = self.cap.get(cv2.CAP_PROP_FPS)
            delay = int(1000 / fps) if fps > 0 else 33
            self.after(delay, self._next_frame)
        else:
            self.cap.release()
            self.cap = None

    def on_close(self):
        if self.cap is not None:
            self.cap.release()
        self.destroy()


if __name__ == "__main__":
    app = SignLanguageApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
