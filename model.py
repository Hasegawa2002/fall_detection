import cv2
import numpy as np
import onnxruntime as ort

class YOLOv8_Pose_ONNX:
    def __init__(self, model_path, confidence_thres=0.25, iou_thres=0.45):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.conf_threshold = confidence_thres
        self.iou_threshold = iou_thres

        self.input_width = 640
        self.input_height = 640

    def preprocess(self, img):
        """
        画像を歪めずにリサイズして、あまりを埋める。
        """
        self.img_height, self.img_width = img.shape[:2]

        # 1. 拡大縮小率を計算（縦と横、縮尺があう方に合わせる）
        scale = min(self.input_width / self.img_width, self.input_height / self.img_height)

        # 2. リサイズ後の新しいサイズ
        new_w = int(self.img_width * scale)
        new_h = int(self.img_height * scale)
        img_resized = cv2.resize(img, (new_w, new_h))

        # 3. パディング（余白）の計算
        dw = (self.input_width - new_w) / 2
        dh = (self.input_height - new_h) / 2

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        # 灰色(114, 114, 114)で余白を埋めるのがYOLO流儀
        img_padded = cv2.copyMakeBorder(
            img_resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        # 4. 正規化などの処理
        img_input = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        img_input = img_input.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_input = np.expand_dims(img_input, axis=0)

        # 後処理で座標を戻すために、スケールとパディング量を保存しておく
        self.scale = scale
        self.pad_w = left
        self.pad_h = top

        return img_input

    def postprocess(self, output):
        preds = np.squeeze(output[0]).T

        scores = preds[:, 4]
        keep_idxs = scores > self.conf_threshold
        preds = preds[keep_idxs]
        scores = scores[keep_idxs]

        if len(preds) == 0:
            return []

        # ボックス座標の取得
        boxes = preds[:, :4]
        boxes_xywh = boxes.copy()
        boxes_xywh[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_xywh[:, 1] = boxes[:, 1] - boxes[:, 3] / 2

        indices = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(), scores.tolist(),
            self.conf_threshold, self.iou_threshold
        )

        results = []
        for i in indices:
            idx = i if isinstance(i, (int, np.integer)) else i[0]
            row = preds[idx]

            # --- 関節点 (Keypoints) の取得と座標補正 ---
            kpts_raw = row[5:].reshape(17, 3)

            # X座標: (x - パディング) / スケール
            kpts_raw[:, 0] = (kpts_raw[:, 0] - self.pad_w) / self.scale
            # Y座標: (y - パディング) / スケール
            kpts_raw[:, 1] = (kpts_raw[:, 1] - self.pad_h) / self.scale

            # 画像外にはみ出した座標をクリップ（念のため）
            kpts_raw[:, 0] = np.clip(kpts_raw[:, 0], 0, self.img_width)
            kpts_raw[:, 1] = np.clip(kpts_raw[:, 1], 0, self.img_height)

            results.append(kpts_raw)

        return results

    def __call__(self, img):
        input_tensor = self.preprocess(img)
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        return self.postprocess(outputs)