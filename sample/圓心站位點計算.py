import cv2
import numpy as np
import math
from ultralytics import YOLO
from pathlib import Path


class VideoPositionDetector:
    def __init__(self, model_path='yolo11n-pose.pt'):
        """
        初始化 YOLO11 Pose 模型
        model_path: 預設使用 nano 版本 (最快)，若需要更高精度可改用 'yolo11s-pose.pt' 或 'yolo11m-pose.pt'
        """
        print(f"正在載入 YOLO11 模型: {model_path} ...")
        self.model = YOLO(model_path)
        print("模型載入完成。")

    def get_contour_center(self, frame: np.ndarray):
        """
        動態偵測畫面中「亮部開口」的中心點
        """
        # 轉灰階
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 二值化：亮度大於 50 視為開口 (可根據現場光線調整 threshold)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        # 去雜訊
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # 找輪廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # 若找不到亮部，回傳畫面正中心作為備案
            h, w = frame.shape[:2]
            return (w // 2, h // 2), None

        # 取面積最大的輪廓作為開口
        largest_contour = max(contours, key=cv2.contourArea)

        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy), largest_contour

        h, w = frame.shape[:2]
        return (w // 2, h // 2), largest_contour

    def get_nose_from_yolo(self, frame: np.ndarray):
        """
        使用 YOLO11-Pose 找出鼻子座標
        回傳: (x, y) 或 None
        """
        # verbose=False 關閉 YOLO 的大量 log 輸出
        results = self.model(frame, verbose=False)

        for result in results:
            # 修正點：先檢查 keypoints 是否存在，且 keypoints.xy 的長度大於 0
            if result.keypoints is not None and len(result.keypoints.xy) > 0:

                # 取得所有偵測到的人的 keypoints (轉為 numpy)
                all_keypoints = result.keypoints.xy.cpu().numpy()

                # 再次確認轉出來的陣列不是空的 (雙重保險)
                if len(all_keypoints) == 0:
                    continue

                # 取第一個人 [0]
                keypoints = all_keypoints[0]

                # YOLO Pose Keypoint index 0 是鼻子 (Nose)
                nose_x, nose_y = keypoints[0]

                # 檢查座標是否為 0 (YOLO 有時偵測不到會填 0，或者人不在畫面內)
                if nose_x == 0 and nose_y == 0:
                    continue

                return (int(nose_x), int(nose_y))

        return None

    def calculate_angle(self, center: tuple, target: tuple) -> float:
        """
        計算角度：0度在正上方(12點鐘)，順時鐘遞增
        """
        cx, cy = center
        tx, ty = target

        dx = tx - cx
        dy = ty - cy

        # atan2 回傳的是弧度，範圍 -pi 到 pi
        # 在圖像座標中(y向下): 右(0), 下(pi/2), 左(pi), 上(-pi/2)
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)

        # 座標變換:
        # 我們希望 上(-90原值) -> 0
        # 右(0原值) -> 90
        # 下(90原值) -> 180
        # 左(180原值) -> 270
        # 公式: (degree + 90) % 360
        final_angle = (angle_deg + 90) % 360
        return final_angle

    def process_video(self, source_path):
        """
        主迴圈：處理影片
        """
        cap = cv2.VideoCapture(source_path)

        if not cap.isOpened():
            print(f"錯誤: 無法開啟影片 {source_path}")
            return

        # 取得影片資訊 (用來設定輸出影片或是視窗大小)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"開始處理影片: {width}x{height} @ {fps}fps")
        print("按 'q' 鍵停止處理...")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("影片結束")
                break

            # 複製一份用來畫圖
            annotated_frame = frame.copy()

            # 1. 偵測圓心 (每幀重新計算以適應晃動)
            center_point, contour = self.get_contour_center(frame)

            # 視覺化圓心
            cv2.circle(annotated_frame, center_point, 8, (0, 255, 0), -1)  # 綠點
            if contour is not None:
                cv2.drawContours(annotated_frame, [contour], -1, (0, 255, 0), 2)

            # 2. YOLO 偵測鼻子
            nose_point = self.get_nose_from_yolo(frame)

            if nose_point:
                # 3. 計算角度
                angle = self.calculate_angle(center_point, nose_point)

                # --- 視覺化 ---
                # 畫鼻子點
                cv2.circle(annotated_frame, nose_point, 8, (0, 0, 255), -1)  # 紅點
                # 畫連接線
                cv2.line(annotated_frame, center_point, nose_point, (255, 255, 0), 2)  # 青色線

                # 顯示角度資訊
                text_info = f"Angle: {int(angle)} deg"

                # 加上文字背景讓字更清楚
                (text_w, text_h), _ = cv2.getTextSize(text_info, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                cv2.rectangle(annotated_frame, (20, 20), (20 + text_w + 10, 20 + text_h + 20), (0, 0, 0), -1)
                cv2.putText(annotated_frame, text_info, (25, 25 + text_h),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                # 畫一個儀表板風格的弧線 (選用)
                radius = 100
                axes = (radius, radius)
                angle_corrected = angle - 90  # ellipse 的 0度是3點鐘
                # cv2.ellipse(annotated_frame, center_point, axes, 0, -90, angle_corrected, (0, 255, 255), 2)

            # 顯示結果
            # 如果圖片太大，縮小一點顯示
            display_frame = cv2.resize(annotated_frame, (0, 0), fx=0.7, fy=0.7)
            cv2.imshow('YOLO11 Pose Tracking', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # 將此處換成您的影片路徑，若要使用 Webcam 請輸入 0
    video_path = 'rec_20251117_172908.mp4'

    # 初始化並執行
    detector = VideoPositionDetector(model_path='yolo11n-pose.pt')

    # 由於我沒有您的影片檔，這裡您可以換成 0 (Webcam) 測試，或換成實際檔案路徑
    # detector.process_video(0)
    detector.process_video(video_path)