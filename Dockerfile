#Dockerfile
FROM python:3.10-slim

# OpenCVを動かすための必須ライブラリ
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# もし手元に yolov8n-pose.pt があればコピーしておくと、
# 毎回ダウンロードしなくて済みます（無くても動きます）
# COPY yolov8n-pose.pt .

# コードのコピー
COPY app.py .

# ライブラリインストール
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]