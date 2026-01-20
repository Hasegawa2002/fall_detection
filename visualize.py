import cv2
import numpy as np

# COCOフォーマットの接続定義（どの点とどの点を線で結ぶか）
# (始点のインデックス, 終点のインデックス)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),    # 鼻 - 目
    (1, 3), (2, 4),    # 目 - 耳
    (5, 6),            # 肩 - 肩
    (5, 7), (7, 9),    # 左肩 - 左肘 - 左手首
    (6, 8), (8, 10),   # 右肩 - 右肘 - 右手首
    (5, 11), (6, 12),  # 肩 - 腰 (体幹)
    (11, 12),          # 腰 - 腰
    (11, 13), (13, 15),# 左腰 - 左膝 - 左足首
    (12, 14), (14, 16) # 右腰 - 右膝 - 右足首
]

# 色の定義 (B, G, R)
COLOR_POINT = (0, 0, 255)  # 赤 (関節点)
COLOR_LINE  = (0, 255, 0)  # 緑 (骨格の線)

def draw_skeleton(img, keypoints_list, conf_threshold=0.5):
    """
    画像に関節点と骨格を描画する関数
    img: 元画像 (OpenCV形式)
    keypoints_list: ONNXモデルから取得したリスト [(17,3), (17,3), ...]
    conf_threshold: 描画するかどうかの信頼度しきい値
    """
    # 元画像を変更しないようにコピー
    vis_img = img.copy()

    for kpts in keypoints_list:
        # kpts は (17, 3) のnumpy配列 [x, y, conf]

        # --- 1. 骨格（線）を描画 ---
        for p1_idx, p2_idx in SKELETON_CONNECTIONS:
            # 始点と終点の座標と信頼度を取得
            x1, y1, conf1 = kpts[p1_idx]
            x2, y2, conf2 = kpts[p2_idx]

            # 両方の点の信頼度が高い場合のみ線を引く
            if conf1 > conf_threshold and conf2 > conf_threshold:
                pt1 = (int(x1), int(y1))
                pt2 = (int(x2), int(y2))
                cv2.line(vis_img, pt1, pt2, COLOR_LINE, thickness=2, lineType=cv2.LINE_AA)

        # --- 2. 関節（点）を描画 ---
        for i in range(len(kpts)):
            x, y, conf = kpts[i]
            if conf > conf_threshold:
                # 半径3の塗りつぶし円を描画
                center = (int(x), int(y))
                cv2.circle(vis_img, center, 4, COLOR_POINT, thickness=-1, lineType=cv2.LINE_AA)

    return vis_img