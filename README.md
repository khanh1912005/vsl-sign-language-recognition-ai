# Hệ Thống Nhận Diện & Xử Lý Ngôn Ngữ Ký Hiệu Việt Nam (VSL Recognition AI)

Hệ thống nhận diện **100 từ vựng Ngôn ngữ ký hiệu Việt Nam (VSL)** theo thời gian thực dựa trên khung xương (**MediaPipe Holistic**), kỹ thuật trích xuất **458 đặc trưng hình học - động học bất biến (Invariant Spatial-Temporal Features)** và mạng học sâu **Spatial-Temporal Conv1D + Unrolled Bidirectional GRU** tối ưu hóa trên **Pure TFLite (`TFLITE_BUILTINS`)**.

---

## Điểm nổi bật kỹ thuật

- **Trích xuất 458 đặc trưng bất biến (`feature_engineering.py`):**
  - Chuẩn hóa 25 khớp thân trên theo tâm vai và độ rộng vai (`75` chiều).
  - Chuẩn hóa hình dáng từng bàn tay (Local Handshape) theo cổ tay và kích thước lòng bàn tay (`120` chiều).
  - Vị trí tương đối cổ tay & 5 đầu ngón tay so với tâm vai và mũi (`72` chiều).
  - Tương tác 2 bàn tay (Bimanual Interaction, `18` chiều) + cờ trạng thái tay (`2` chiều).
  - Vận tốc chuyển động bậc 1 theo thời gian (Temporal Velocity, `171` chiều) + nội suy tuyến tính khi khuất tay ngắn hạn.
- **Chia tập chuẩn khoa học (Zero Data Leakage - `split_dataset.py`):**
  - Phân chia **Group-Stratified 3-Way Split** theo `base_id` trên 3.875 video (2.806 nhóm gốc): **Train** (1.962 nhóm / 2.739 mẫu), **Validation** (422 nhóm gốc), **Held-out Test** (422 nhóm gốc độc lập).
- **Mô hình nhẹ, suy luận siêu nhanh (`train.py`, `models/model.tflite`):**
  - Kiến trúc `Conv1D + Unrolled Bi-GRU` (`607.844` tham số).
  - Độ chính xác trên **Held-out Test**: **Top-1 = 86.26%**, **Top-5 = 94.79%**, **Weighted F1 = 85.15%** (tăng **+16.35%** so với Baseline 258 chiều thô).
  - Xuất **Pure TFLite** dung lượng **2.76 MB**, tốc độ suy luận CPU **~2.89 ms/mẫu**.

---

## Cấu trúc thư mục

- `web_server.py`, `static/index.html`: Máy chủ FastAPI đa người dùng (Live Webcam, Phân tích Video, Kiểm thử tải đồng thời).
- `feature_engineering.py`: Biến đổi chuỗi tọa độ thô `(30, 258)` sang đặc trưng bất biến `(30, 458)`.
- `preprocess.py`, `preprocess_fast.py`: Trích xuất landmarks MediaPipe Holistic từ video.
- `split_dataset.py`: Chia tập Group-Stratified chống rò rỉ dữ liệu.
- `train.py`: Huấn luyện, đánh giá Held-out Test, Ablation Study và xuất Pure TFLite.
- `predict.py`, `main.py`: Suy luận 1 video hoặc chạy batch xuất `submission.csv` / `output.csv`.
- `predict_sequence.py`: Giải mã chuỗi cử chỉ liên tục và đánh giá Word Error Rate (WER).
- `app_ui.py`, `demo_webcam.py`: Giao diện Desktop (Tkinter) và OpenCV Webcam.
- `models/`: Trọng số mô hình (`model.tflite`, `best_model.keras`), báo cáo đánh giá và biểu đồ thực nghiệm.

---

## Hướng dẫn cài đặt & chạy nhanh

```bash
pip install -r requirements_web.txt
```

1. **Chạy giao diện Web (FastAPI):**
   ```bash
   python web_server.py
   ```
   Mở trình duyệt tại `http://127.0.0.1:8000`.

2. **Dự đoán 1 video mẫu:**
   ```bash
   python predict.py "dataset/public_test/100708.mp4"
   ```

3. **Dự đoán hàng loạt (Tác vụ 2):**
   ```bash
   python main.py --limit 10
   ```
