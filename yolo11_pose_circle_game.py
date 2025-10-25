"""
YOLO11 Pose 圓形視窗互動遊戲
Python 3.11

功能：
- 以 USB 攝影機擷取即時畫面
- 使用 Ultralytics YOLO11-pose 偵測人體骨架
- 將相機畫面與骨架一併繪製於圓形視窗中（外圈加邊框）
- 自動偵測是否有 CUDA，GPU 則啟用半精度以加速
- 顯示 FPS

依賴：
  pip install ultralytics opencv-python
  # 若需 GPU，請依環境安裝對應的 torch CUDA 版本（可至 pytorch.org 選擇指令）

快捷鍵：
  q  退出程式

執行：
python yolo11_pose_ripple_bg.py --weights yolo11n-pose.pt --imgsz 640 --width 1280 --height 720 --circle_ratio 0.9 --line_thickness 2
"""
from __future__ import annotations
import argparse
import time
from typing import Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO11 Pose 圓形視窗互動遊戲")
    parser.add_argument("--camera", type=int, default=0, help="攝影機索引（預設 0）")
    parser.add_argument("--weights", type=str, default="yolo11n-pose.pt", help="YOLO11 pose 權重檔")
    parser.add_argument("--imgsz", type=int, default=640, help="推論輸入大小（短邊）")
    parser.add_argument("--width", type=int, default=1280, help="相機寬度")
    parser.add_argument("--height", type=int, default=720, help="相機高度")
    parser.add_argument("--circle_ratio", type=float, default=0.9, help="圓形視窗直徑相對於短邊比例 (0~1)")
    parser.add_argument("--show_boxes", action="store_true", help="同時繪出偵測框（預設只看骨架）")
    parser.add_argument("--line_thickness", type=int, default=2, help="繪圖線寬")
    parser.add_argument("--conf", type=float, default=0.25, help="信心門檻")
    return parser.parse_args()


def create_circular_mask(h: int, w: int, center: Tuple[int, int] | None = None, radius: int | None = None) -> np.ndarray:
    """建立單通道 8-bit 圓形遮罩，圓內為 255、圓外為 0。"""
    if center is None:
        center = (w // 2, h // 2)
    if radius is None:
        radius = min(h, w) // 2
    Y, X = np.ogrid[:h, :w]
    dist_from_center = (X - center[0]) ** 2 + (Y - center[1]) ** 2
    mask = (dist_from_center <= radius ** 2).astype(np.uint8) * 255
    return mask


def main():
    args = parse_args()

    # 初始化攝影機
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if cv2.getBuildInformation().find("MSVC") != -1 else cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError("無法開啟攝影機。請確認 USB 攝影機連線或索引是否正確。")

    # 裝置與半精度設定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_half = device.startswith("cuda")

    # 載入 YOLO11 pose
    model = YOLO(args.weights)
    model.to(device)

    # FPS 計算
    prev_t = time.time()
    fps = 0.0

    win_name = "YOLO11 Pose Circle Game"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # 主迴圈
    while True:
        ok, frame = cap.read()
        if not ok:
            print("讀取影像失敗，嘗試繼續……")
            continue

        h, w = frame.shape[:2]
        # 計算圓形參數
        diameter = int(min(h, w) * args.circle_ratio)
        radius = diameter // 2
        center = (w // 2, h // 2)

        # 推論（以原始畫面推論，再做圓形顯示遮罩）
        # Ultralytics 會自動 letterbox；指定 imgsz 與 conf，可選擇半精度
        results = model(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            verbose=False,
            device=device,
            half=use_half,
        )

        # 取得繪製後的影像（含骨架連線）。若不想要框，可先畫圖後再把框擦掉或關閉顯示。
        annotated = results[0].plot(
            boxes=args.show_boxes,
            kpt_line=True,  # 連結關鍵點骨架
            line_width=args.line_thickness,
        )

        # 製作圓形遮罩並套用
        mask = create_circular_mask(h, w, center=center, radius=radius)
        masked = cv2.bitwise_and(annotated, annotated, mask=mask)

        # 將圓外區域設為純黑（已由 bitwise_and 達成）並畫上圓邊框
        border_color = (255, 255, 255)
        cv2.circle(masked, center, radius, border_color, 3)

        # FPS 計算與顯示
        now = time.time()
        dt = now - prev_t
        prev_t = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt
        cv2.putText(
            masked,
            f"FPS: {fps:.1f} | Device: {device}{' (half)' if use_half else ''}",
            (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # 顯示
        cv2.imshow(win_name, masked)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
