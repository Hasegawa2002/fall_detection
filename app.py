import streamlit as st
import cv2
import numpy as np
import tempfile
import time

# 作成した別ファイルからクラスと関数をインポート
from model import YOLOv8_Pose_ONNX
from visualize import draw_skeleton

# --- 設定値 ---
MODEL_PATH = 'yolov8n-pose.onnx' # モデルファイルのパス

st.set_page_config(page_title="Fall Detection App", layout="wide")
st.title("転倒検知システム (YOLOv8-Pose ONNX)")

# --- サイドバー設定 ---
st.sidebar.header("設定")
skip_frame = st.sidebar.slider("フレームスキップ数", min_value=1, max_value=30, value=10)
conf_thr = st.sidebar.slider("検出信頼度閾値", min_value=0.1, max_value=1.0, value=0.5)

# --- 1. 判定ロジック (ご提示のコード) ---
def judge(data, prev=None, img_height=480):
    if data is not None and prev is not None:
        mask = prev['mask'] & data['mask']
        if mask.sum() < 3: return False 

        # 移動ベクトル計算
        diff = data['pos'][mask] - prev['pos'][mask]
        mean_diff = diff.mean(axis=0)
        vel = np.linalg.norm(mean_diff) 

        # 方向計算
        direction = mean_diff / vel if vel > 1e-5 else np.array([0, 0])

        # 分散 (体の広がり)
        var_x = np.var(data['pos'][:,0][mask])
        var_y = np.var(data['pos'][:,1][mask])

        # 判定
        speed_cond = (vel / img_height) > 0.03
        dir_cond = direction[1] > 0.5
        pose_cond = (var_x * 2) > var_y

        if speed_cond and dir_cond and pose_cond:
            return True
    return False

def data_preprocess(results, thr=0.5):
    try:
        # 結果が空でないか確認
        if len(results) == 0:
            return None
        
        # 1人目のデータを取得 (results[0] は (17, 3) のndarray想定)
        kpts = results[0][:,:2]
        confs = results[0][:,2]

        mask = confs > thr
        if mask.sum() == 0: return None

        return {'pos': kpts, 'mask': mask}
    except Exception as e:
        # エラー時はNoneを返す
        return None

# --- モデルのキャッシュ化 (読み込み高速化) ---
@st.cache_resource
def load_model(path):
    return YOLOv8_Pose_ONNX(path)

# --- メイン処理 ---
uploaded_file = st.file_uploader("動画ファイルをアップロード", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # 一時ファイルとして保存
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # モデルロード
    try:
        model = load_model(MODEL_PATH)
        st.sidebar.success("モデル読み込み完了")
    except Exception as e:
        st.error(f"モデルの読み込みに失敗しました: {e}")
        st.stop()

    if st.button("解析開始"):
        cap = cv2.VideoCapture(video_path)
        
        # 動画情報の取得
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 表示用のプレースホルダー
        st_image = st.empty()
        st_status = st.empty()
        progress_bar = st.progress(0)

        prev_data = None
        frame_count = 0
        
        start_time = time.perf_counter()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # フレームスキップ
            if frame_count % skip_frame != 0:
                continue

            # --- 推論 ---
            results = model(frame) # model.pyのクラスを使用

            # --- データ処理 & 判定 ---
            current_data = data_preprocess(np.array(results), thr=conf_thr)

            is_fall = False
            if current_data is not None:
                is_fall = judge(current_data, prev=prev_data, img_height=height)
                prev_data = current_data
            else:
                prev_data = None

            # --- 描画 ---
            # visualize.py の関数を使用
            annotated_frame = draw_skeleton(frame, results, conf_threshold=conf_thr)

            # 判定結果の描画
            if is_fall:
                cv2.putText(annotated_frame, "!!! FALL DETECTED !!!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.rectangle(annotated_frame, (0, 0), (width, height), (0, 0, 255), 10)
                st_status.error(f"⚠️ {frame_count}フレーム目: 転倒検知！")
            else:
                cv2.putText(annotated_frame, "Normal", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                st_status.info("解析中: Normal")

            # Streamlitで表示するために BGR -> RGB 変換
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st_image.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # 進捗バー更新
            if total_frames > 0:
                progress_bar.progress(min(frame_count / total_frames, 1.0))

        cap.release()
        st.success("解析が終了しました")