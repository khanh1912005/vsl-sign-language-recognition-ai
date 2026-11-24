FROM python:3.11-slim

WORKDIR /app

# Cài đặt thư viện hệ thống cần thiết cho OpenCV & MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_web.txt .
RUN pip install --no-cache-dir -r requirements_web.txt

COPY models/ ./models/
COPY static/ ./static/
COPY processed_data/label_mapping.json ./processed_data/label_mapping.json
COPY holistic_landmarker.task ./holistic_landmarker.task
COPY predict.py preprocess.py feature_engineering.py train.py web_server.py ./

EXPOSE 7860

CMD ["sh", "-c", "uvicorn web_server:app --host 0.0.0.0 --port ${PORT:-7860}"]
