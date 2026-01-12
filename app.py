
#main.py
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# --- 1. モデルの準備 ---
# キャッシュして再読み込みを防ぐ
@st.cache_resource
def load_model():
    # 初回は自動ダウンロードされますが、
    # Dockerビルド時にファイルを含めておくと高速です
    return YOLO('yolov8n-pose.pt')

def is_fallen(results):
  if len(results[0]) > 0:
    data = results[0].keypoints.data[0].cpu().numpy()
    pos = data[:,:2][data[:,2]>0.4]
    #print(len(pos))
    # 8点以上検出できなければ判定しない
    if len(pos) < 8:
      return False
    var_x = np.var(pos[:,0])
    var_y = np.var(pos[:,1])
    #print(var_x,var_y)
    if var_x > 2*var_y:
      return True
  return False



model = load_model()

# --- 2. 画面レイアウト ---
st.title("Docker YOLO Pose App")
st.write("YOLOv8-Poseを使ったリアルタイム（静止画）推論")

# カメラ入力
img_file_buffer = st.camera_input("写真を撮る")

if img_file_buffer is not None:
    # 画像の読み込み (PIL形式)
    image = Image.open(img_file_buffer)
    image = image.resize((512,384))

    # --- 3. 推論 & 描画 ---
    # YOLOはPIL画像をそのまま受け取れます
    results = model(image)
    if is_fallen(results):
      message = 'fall!'
      color = (255,0,0)
    else:
      message = 'normal'
      color = (0,255,0)

    # 結果画像を生成 (BGR形式のnumpy配列が返る)
    res_plotted = results[0].plot()

    org = (50, 50)          # 文字の左下の座標 (x, y)
    font = cv2.FONT_HERSHEY_SIMPLEX # フォントの種類
    fontScale = 10.0         # 文字の大きさ
    #color = (0, 255, 0)     # 文字の色 (B, G, R) -> ここでは緑
    thickness = 2           # 線の太さ
    lineType = cv2.LINE_AA  # アンチエイリアス（滑らかにする）

    cv2.putText(res_plotted, message, org, font, fontScale, color, thickness, lineType)
    # ========

    # 色変換: OpenCV(BGR) -> Streamlit(RGB)
    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

    # --- 4. 表示 ---
    st.image(res_rgb, caption="推論結果")

    # おまけ: 検出された人数を表示
    num_persons = len(results[0].keypoints)
    st.success(f"{num_persons} 人を検出しました！")