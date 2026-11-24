# -*- coding: utf-8 -*-
"""Vẽ sơ đồ kiến trúc hệ thống Web đa người dùng (dùng cho báo cáo)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUT = rf"{BASE_DIR}\models\web_architecture.png"

fig, ax = plt.subplots(figsize=(13, 7.2))
ax.set_xlim(0, 13)
ax.set_ylim(0, 7.2)
ax.axis("off")


def box(x, y, w, h, title, lines, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.15",
                                fc=fc, ec=ec, lw=2))
    ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="center", fontsize=11, fontweight="bold", color="#0b2545")
    for i, ln in enumerate(lines):
        ax.text(x + 0.15, y + h - 0.75 - i * 0.36, "• " + ln, ha="left", va="center", fontsize=8.8, color="#1b1b1b")


def arrow(x1, y1, x2, y2, text="", color="#34495e", dy=0.18):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, lw=1.8, color=color))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, text, ha="center", va="center", fontsize=8.5,
                color=color, fontweight="bold", bbox=dict(fc="white", ec="none", alpha=0.85))


# Tầng 1: Trình duyệt người dùng
ax.text(2.1, 6.95, "TẦNG 1 – MÁY KHÁCH (TRÌNH DUYỆT)", ha="center", fontsize=10.5, fontweight="bold", color="#1f4e79")
for i, uname in enumerate(["Người dùng A", "Người dùng B", "Người dùng N"]):
    y = 4.75 - i * 2.2
    box(0.2, y, 3.8, 2.0, f"{uname} (session_id riêng)",
        ["Webcam → cắt khung vuông 1:1 (480×480)",
         "MediaPipe Holistic JS → 258 tọa độ/khung",
         "Lấy mẫu ~14 FPS, gom 16–30 khung",
         "Gửi ma trận 30×258 qua WebSocket"],
        "#eaf2fb", "#2e86c1")

# Tầng 2: Mạng / Cloudflare
box(4.55, 2.6, 2.3, 2.3, "Cloudflare Tunnel",
    ["HTTPS / WSS", "trycloudflare.com", "→ 127.0.0.1:8000"], "#fdf2e9", "#e67e22")
ax.text(5.7, 6.95, "TẦNG 2 – MẠNG", ha="center", fontsize=10.5, fontweight="bold", color="#e67e22")

# Tầng 3: Máy chủ FastAPI
ax.text(10.0, 6.95, "TẦNG 3 – MÁY CHỦ FASTAPI (web_server.py)", ha="center", fontsize=10.5, fontweight="bold", color="#1e8449")
box(7.4, 4.55, 5.4, 2.15, "Lớp giao tiếp (REST + WebSocket)",
    ["/ws/{session_id}, /api/predict_window",
     "/api/upload_video, /api/predict_sample_video",
     "/api/session/action (commit / backspace / clear)",
     "/api/stats, /api/sample_pool"],
    "#e9f7ef", "#1e8449")
box(7.4, 2.25, 5.4, 2.05, "Tiền xử lý & Suy luận",
    ["align_hands_to_pose_wrists + mirror assist",
     "extract_sequence_features: 30×258 → 30×458",
     "Singleton model.tflite nạp 1 lần + TFLITE_LOCK",
     "Suy luận ~3–5 ms / cửa sổ trên CPU"],
    "#e9f7ef", "#1e8449")
box(7.4, 0.15, 5.4, 1.85, "Quản lý phiên (SESSIONS[session_id])",
    ["prob_history, draft_word, draft_streak",
     "final_words → smooth_vietnamese_sentence",
     "Cách ly dữ liệu hoàn toàn giữa người dùng"],
    "#e9f7ef", "#1e8449")

# Mũi tên
for i in range(3):
    y = 5.75 - i * 2.2
    arrow(4.0, y, 4.55, 3.75 + (1 - i) * 0.5, color="#2e86c1")
arrow(6.85, 4.1, 7.4, 5.4, "JSON 30×258", color="#e67e22", dy=0.25)
arrow(10.1, 4.55, 10.1, 4.3, color="#1e8449")
arrow(10.1, 2.25, 10.1, 2.0, color="#1e8449")
arrow(7.4, 1.1, 6.6, 2.6, "Top-5 + câu tiếng Việt", color="#8e44ad", dy=-0.05)

plt.title("Hình: Kiến trúc hệ thống Web nhận diện VSL đa người dùng", fontsize=1, color="white")
plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches="tight")
print("Saved", OUT)
