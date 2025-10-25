"""
YOLO11 Pose 圓形視窗互動遊戲（以上傳圖片為背景，偵測到人體骨架時產生水波紋）
Python 3.11

功能：
- 以 USB 攝影機進行骨架偵測（畫面不顯示攝影機）
- 顯示為「上傳圖片」的背景（縮放鋪滿），並套用圓形視窗遮罩
- 當偵測到人體骨架時，於背景上產生水波紋動畫（可依手/腳/鼻子位置觸發）
- 可選擇是否同時在背景上繪製骨架
- 自動偵測是否有 CUDA，GPU 則啟用半精度以加速
- 顯示 FPS

安裝：
  pip install ultralytics opencv-python
  # 如需 GPU，請至 pytorch.org 依環境安裝對應的 torch CUDA 版本

執行：
  python yolo11_pose_ripple_bg.py --bg ./calm_water.jpg --camera 0 --weights yolo11n-pose.pt --draw_skeleton
  # 參數說明：見 argparse 區塊

快捷鍵：
  q  退出程式
"""
from __future__ import annotations
import argparse
import time
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# COCO 17 keypoints 的連線（骨架）
COCO_EDGES = [
    (5, 7), (7, 9),    # 左臂
    (6, 8), (8, 10),   # 右臂
    (5, 6),            # 肩連線
    (5, 11), (6, 12),  # 肩到髖
    (11, 12),          # 髖連線
    (11, 13), (13, 15),# 左腿
    (12, 14), (14, 16),# 右腿
    (0, 1), (0, 2),    # 鼻-眼
    (1, 3), (2, 4)     # 眼-耳
]

@dataclass
class Ripple:
    x: float
    y: float
    r: float = 8.0
    max_r: float = 1200.0
    growth: float = 6.0
    alpha: float = 0.7
    fade: float = 0.985  # 每幀透明度乘以係數

    def step(self):
        self.r += self.growth
        self.alpha *= self.fade
        return (self.alpha > 0.02) and (self.r < self.max_r)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO11 Pose 圓形視窗互動遊戲（圖片背景＋水波紋）")
    parser.add_argument("--camera", type=int, default=0, help="攝影機索引（預設 0）")
    parser.add_argument("--weights", type=str, default="yolo11n-pose.pt", help="YOLO11 pose 權重檔")
    parser.add_argument("--imgsz", type=int, default=640, help="推論輸入大小（短邊）")
    parser.add_argument("--width", type=int, default=1280, help="視窗寬度")
    parser.add_argument("--height", type=int, default=720, help="視窗高度")
    parser.add_argument("--circle_ratio", type=float, default=0.95, help="圓形視窗直徑相對於短邊比例 (0~1)")
    parser.add_argument("--conf", type=float, default=0.25, help="偵測信心門檻")
    parser.add_argument("--line_width", type=int, default=2, help="骨架線寬（配合 --draw_skeleton 使用）")
    parser.add_argument("--draw_skeleton", action="store_true", help="於背景上繪製骨架")
    parser.add_argument("--bg", type=str, default="calm_water.jpg", help="背景圖片路徑（上傳圖片）")
    parser.add_argument("--ripple_from", type=str, default="wrists", choices=["wrists", "ankles", "nose", "center"], help="水波觸發位置來源")
    parser.add_argument("--ripple_interval", type=float, default=0.35, help="最小觸發間隔（秒）")
    return parser.parse_args()


def create_circular_mask(h: int, w: int, center: Tuple[int, int] | None = None, radius: int | None = None) -> np.ndarray:
    if center is None:
        center = (w // 2, h // 2)
    if radius is None:
        radius = min(h, w) // 2
    Y, X = np.ogrid[:h, :w]
    mask = ((X - center[0]) ** 2 + (Y - center[1]) ** 2 <= radius ** 2).astype(np.uint8) * 255
    return mask


def fit_bg_to_canvas(bg: np.ndarray, width: int, height: int) -> np.ndarray:
    """等比縮放並置中裁切，使背景鋪滿指定寬高。"""
    h, w = bg.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("背景圖片讀取失敗，尺寸為 0。")
    scale = max(width / w, height / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(bg, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x0 = (nw - width) // 2
    y0 = (nh - height) // 2
    return resized[y0:y0+height, x0:x0+width]


def extract_kpts(result) -> Tuple[np.ndarray, np.ndarray]:
    """回傳 (kpts_xy, kpts_conf)，維度 (N, K, 2) 與 (N, K)。若無則回傳空陣列。"""
    if not hasattr(result, "keypoints") or result.keypoints is None:
        return np.zeros((0, 17, 2), dtype=np.float32), np.zeros((0, 17), dtype=np.float32)
    xy = result.keypoints.xy  # tensor (N, K, 2)
    conf = getattr(result.keypoints, "conf", None)  # tensor (N, K) 或 None
    xy = xy.cpu().numpy() if hasattr(xy, "cpu") else np.asarray(xy)
    if conf is None:
        kconf = np.ones((xy.shape[0], xy.shape[1]), dtype=np.float32)
    else:
        kconf = conf.cpu().numpy() if hasattr(conf, "cpu") else np.asarray(conf)
    return xy.astype(np.float32), kconf.astype(np.float32)


def pick_ripple_points(kpts_xy: np.ndarray, kpts_conf: np.ndarray, width: int, height: int, mode: str) -> List[Tuple[int, int]]:
    """根據 keypoint 產生水波中心座標（多個）。為了視覺一致，y 固定在水面帶（畫面 2/3 高處）。"""
    points: List[Tuple[int, int]] = []
    water_y = int(height * 0.66)
    if kpts_xy.size == 0:
        return points

    idx_map = {"nose": 0, "l_wrist": 9, "r_wrist": 10, "l_ankle": 15, "r_ankle": 16}

    for i in range(kpts_xy.shape[0]):
        xy = kpts_xy[i]
        cf = kpts_conf[i]
        if mode == "center":
            points.append((width // 2, water_y))
            continue
        if mode == "nose":
            if cf[idx_map["nose"]] > 0.2:
                cx = int(np.clip(xy[idx_map["nose"], 0], 0, width - 1))
                points.append((cx, water_y))
            continue
        if mode == "wrists":
            for j in (idx_map["l_wrist"], idx_map["r_wrist"]):
                if cf[j] > 0.2:
                    cx = int(np.clip(xy[j, 0], 0, width - 1))
                    points.append((cx, water_y))
            continue
        if mode == "ankles":
            for j in (idx_map["l_ankle"], idx_map["r_ankle"]):
                if cf[j] > 0.2:
                    cx = int(np.clip(xy[j, 0], 0, width - 1))
                    points.append((cx, water_y))
            continue
    return points


def draw_skeletons(canvas: np.ndarray, kpts_xy: np.ndarray, kpts_conf: np.ndarray, lw: int = 2):
    if kpts_xy.size == 0:
        return
    h, w = canvas.shape[:2]
    overlay = canvas.copy()
    for i in range(kpts_xy.shape[0]):
        xy = kpts_xy[i]
        cf = kpts_conf[i]
        # 畫線
        for a, b in COCO_EDGES:
            if cf[a] > 0.2 and cf[b] > 0.2:
                pa = (int(np.clip(xy[a, 0], 0, w - 1)), int(np.clip(xy[a, 1], 0, h - 1)))
                pb = (int(np.clip(xy[b, 0], 0, w - 1)), int(np.clip(xy[b, 1], 0, h - 1)))
                cv2.line(overlay, pa, pb, (255, 255, 255), lw, cv2.LINE_AA)
        # 關節點
        for k in range(xy.shape[0]):
            if cf[k] > 0.2:
                p = (int(np.clip(xy[k, 0], 0, w - 1)), int(np.clip(xy[k, 1], 0, h - 1)))
                cv2.circle(overlay, p, max(1, lw), (255, 255, 255), -1, cv2.LINE_AA)
    # 半透明疊加
    cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, dst=canvas)


def draw_ripples(canvas: np.ndarray, ripples: List[Ripple]):
    if not ripples:
        return
    base = canvas.copy()
    for rp in ripples:
        # 繪製 3 條相隔的波紋線
        for k in range(3):
            rr = int(rp.r + k * 10)
            cv2.circle(base, (int(rp.x), int(rp.y)), rr, (255, 255, 255), 1, cv2.LINE_AA)
    # 使用平均 alpha 疊加（視覺上已足夠）
    mean_alpha = float(np.mean([rp.alpha for rp in ripples]))
    mean_alpha = float(np.clip(mean_alpha, 0.02, 0.8))
    cv2.addWeighted(base, mean_alpha, canvas, 1.0 - mean_alpha, 0, dst=canvas)


def main():
    args = parse_args()

    # 載入背景
    bg_img = cv2.imread(args.bg, cv2.IMREAD_COLOR)
    if bg_img is None:
        raise FileNotFoundError(f"找不到背景圖片：{args.bg}")

    # 初始化攝影機（僅用於偵測，不顯示）
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if cv2.getBuildInformation().find("MSVC") != -1 else cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError("無法開啟攝影機。請確認 USB 攝影機連線或索引是否正確。")

    # 裝置與半精度
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_half = device.startswith("cuda")

    # 載入 YOLO11 pose
    model = YOLO(args.weights)
    model.to(device)

    prev_t = time.time()
    fps = 0.0

    win_name = "YOLO11 Pose Ripple on Image"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    ripples: List[Ripple] = []
    last_spawn = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("讀取影像失敗，嘗試繼續……")
            continue

        # 產生顯示用畫布（背景圖片鋪滿）
        canvas = fit_bg_to_canvas(bg_img, args.width, args.height)
        h, w = canvas.shape[:2]

        # 推論（用攝影機畫面）
        results = model(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            verbose=False,
            device=device,
            half=use_half,
        )
        res = results[0]
        kxy, kcf = extract_kpts(res)

        # 選擇水波觸發點
        now = time.time()
        if kxy.size > 0 and (now - last_spawn) >= args.ripple_interval:
            centers = pick_ripple_points(kxy, kcf, w, h, args.ripple_from)
            for cx, cy in centers:
                ripples.append(Ripple(x=cx, y=cy))
            last_spawn = now

        # 更新水波
        ripples = [rp for rp in ripples if rp.step()]
        draw_ripples(canvas, ripples)

        # （可選）把骨架也畫在背景上
        if args.draw_skeleton:
            draw_skeletons(canvas, kxy, kcf, lw=args.line_width)

        # 應用圓形遮罩與邊框
        diameter = int(min(h, w) * args.circle_ratio)
        radius = diameter // 2
        center = (w // 2, h // 2)
        mask = create_circular_mask(h, w, center=center, radius=radius)
        masked = cv2.bitwise_and(canvas, canvas, mask=mask)
        cv2.circle(masked, center, radius, (255, 255, 255), 3)

        # FPS 顯示
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
