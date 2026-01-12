FROM python:3.10-slim

# システムライブラリを最小限にする
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# pip自体のアップグレードと、CPU版のインストール
# 1. まず pip 自体を最新にする
RUN pip install --no-cache-dir --upgrade pip

# 2. 最初にかつ強力に「CPU版のtorch」だけをインストールする
# これで、後から入るライブラリがGPU版を連れてくるのを防ぎます
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. その後に、残りのライブラリを入れる
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
# モデルファイルがあればコピー（なければ実行時にダウンロードされる）
# COPY yolov8n-pose.pt .

EXPOSE 8501
CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0"]
