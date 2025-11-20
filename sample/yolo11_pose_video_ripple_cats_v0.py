"""
YOLO11 Pose 視訊背景＋條件觸發水波＆貓咪淡入
Python 3.11

需求實作：
- 背景改為 *上傳的 mp4 影片*（循環播放）。
- 以 USB 攝影機做人體骨架偵測（YOLOv11-pose）。
- 觸發條件：同一位人的 5 個臉部關鍵點「鼻子、左眼、右眼、左耳、右耳」**同時存在**且**連續維持 3 秒**。
- 符合條件後：在「鼻子」座標位置產生水波紋，並從三張上傳的貓圖中**隨機選一張**，以 **3 秒淡入**方式浮現於畫面。
- 保留圓形視窗遮罩與（可選）骨架疊圖。

安裝：
  pip install ultralytics opencv-python
  # 如需 GPU，請依環境安裝對應 torch CUDA 版本

執行範例：
  python yolo11_pose_video_ripple_cats_v0.py \
    --bg_video /mnt/data/calm_water.mp4 \
    --cats /mnt/data/cat1.png,/mnt/data/cat2.png,/mnt/data/cat3.png \
    --camera 0 --weights yolo11n-pose.pt --draw_skeleton
  # 按 q 離開
"""
from __future__ import annotations
import argparse
import time
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# COCO 17 關鍵點：0鼻,1左眼,2右眼,3左耳,4右耳
FACE_IDXS = [0, 1, 2, 3, 4]

# 連線（骨架）
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
    max_r: float = 1600.0
    growth: float = 6.0
    alpha: float = 0.75
    fade: float = 0.985

    def step(self) -> bool:
        self.r += self.growth
        self.alpha *= self.fade
        return (self.alpha > 0.02) and (self.r < self.max_r)

@dataclass
class CatOverlay:
    img: np.ndarray
    start: float
    duration: float = 3.0  # 秒
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0

    def alpha(self, now: float) -> float:
        a = (now - self.start) / max(1e-6, self.duration)
        return float(np.clip(a, 0.0, 1.0))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO11 Pose｜影片背景＋條件觸發水波＆貓咪淡入")
    p.add_argument("--camera", type=int, default=0, help="攝影機索引（預設 0）")
    p.add_argument("--weights", type=str, default="yolo11n-pose.pt", help="YOLO11 pose 權重檔")
    p.add_argument("--imgsz", type=int, default=640, help="推論輸入大小")
    p.add_argument("--width", type=int, default=1280, help="視窗寬度")
    p.add_argument("--height", type=int, default=720, help="視窗高度")
    p.add_argument("--circle_ratio", type=float, default=0.95, help="圓形視窗直徑相對於短邊比例 (0~1)")
    p.add_argument("--conf", type=float, default=0.25, help="偵測信心門檻")
    p.add_argument("--draw_skeleton", action="store_true", help="於背景上繪製骨架")
    p.add_argument("--line_width", type=int, default=2, help="骨架線寬")

    # 新增：背景影片與貓咪設定
    p.add_argument("--bg_video", type=str, default="./mnt/data/calm_water.mp4", help="背景 mp4 路徑")
    p.add_argument("--cats", type=str, default="./mnt/data/cat1.png,./mnt/data/cat2.png,./mnt/data/cat3.png", help="以逗號分隔的貓圖路徑")
    p.add_argument("--cat_fade", type=float, default=3.0, help="貓咪淡入秒數")

    # 觸發條件設定
    p.add_argument("--face_hold_sec", type=float, default=3.0, help="需連續滿足的秒數")
    p.add_argument("--trigger_cooldown", type=float, default=3.0, help="觸發後的冷卻秒數")
    return p.parse_args()


def create_circular_mask(h: int, w: int, center: Tuple[int, int] | None = None, radius: int | None = None) -> np.ndarray:
    if center is None:
        center = (w // 2, h // 2)
    if radius is None:
        radius = min(h, w) // 2
    Y, X = np.ogrid[:h, :w]
    return (((X - center[0]) ** 2 + (Y - center[1]) ** 2) <= radius ** 2).astype(np.uint8) * 255


def resize_fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """等比縮放並中心裁切，鋪滿 width x height。"""
    ih, iw = image.shape[:2]
    scale = max(width / iw, height / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x0 = (nw - width) // 2
    y0 = (nh - height) // 2
    return resized[y0:y0 + height, x0:x0 + width]


def extract_kpts(result) -> Tuple[np.ndarray, np.ndarray]:
    if not hasattr(result, "keypoints") or result.keypoints is None:
        return np.zeros((0, 17, 2), dtype=np.float32), np.zeros((0, 17), dtype=np.float32)
    xy = result.keypoints.xy  # (N,K,2)
    conf = getattr(result.keypoints, "conf", None)  # (N,K) or None
    xy = xy.cpu().numpy() if hasattr(xy, "cpu") else np.asarray(xy)
    if conf is None:
        kconf = np.ones((xy.shape[0], xy.shape[1]), dtype=np.float32)
    else:
        kconf = conf.cpu().numpy() if hasattr(conf, "cpu") else np.asarray(conf)
    return xy.astype(np.float32), kconf.astype(np.float32)


def choose_locked_person(prev_center: Optional[Tuple[float, float]], centers: List[Tuple[float, float]]) -> Optional[int]:
    if not centers:
        return None
    if prev_center is None:
        # 直接選第 1 個（或可用其他策略）
        return 0
    px, py = prev_center
    dists = [((cx - px) ** 2 + (cy - py) ** 2, i) for i, (cx, cy) in enumerate(centers)]
    dists.sort()
    return dists[0][1]


def draw_skeletons(canvas: np.ndarray, kpts_xy: np.ndarray, kpts_conf: np.ndarray, lw: int = 2):
    if kpts_xy.size == 0:
        return
    h, w = canvas.shape[:2]
    overlay = canvas.copy()
    for i in range(kpts_xy.shape[0]):
        xy = kpts_xy[i]
        cf = kpts_conf[i]
        for a, b in COCO_EDGES:
            if cf[a] > 0.2 and cf[b] > 0.2:
                pa = (int(np.clip(xy[a, 0], 0, w - 1)), int(np.clip(xy[a, 1], 0, h - 1)))
                pb = (int(np.clip(xy[b, 0], 0, w - 1)), int(np.clip(xy[b, 1], 0, h - 1)))
                cv2.line(overlay, pa, pb, (255, 255, 255), lw, cv2.LINE_AA)
        for k in range(xy.shape[0]):
            if cf[k] > 0.2:
                p = (int(np.clip(xy[k, 0], 0, w - 1)), int(np.clip(xy[k, 1], 0, h - 1)))
                cv2.circle(overlay, p, max(1, lw), (255, 255, 255), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, dst=canvas)


def draw_ripples(canvas: np.ndarray, ripples: List[Ripple]):
    if not ripples:
        return
    base = canvas.copy()
    for rp in ripples:
        for k in range(3):
            rr = int(rp.r + k * 10)
            cv2.circle(base, (int(rp.x), int(rp.y)), rr, (255, 255, 255), 1, cv2.LINE_AA)
    mean_alpha = float(np.mean([rp.alpha for rp in ripples]))
    mean_alpha = float(np.clip(mean_alpha, 0.02, 0.8))
    cv2.addWeighted(base, mean_alpha, canvas, 1.0 - mean_alpha, 0, dst=canvas)


def overlay_cat(canvas: np.ndarray, cat: CatOverlay, now: float):
    a = cat.alpha(now)
    if a <= 0.0:
        return
    x, y, w, h = cat.x, cat.y, cat.w, cat.h
    if w <= 0 or h <= 0:
        return
    roi = canvas[y:y + h, x:x + w]
    if roi.shape[0] != h or roi.shape[1] != w:
        return
    img = cv2.resize(cat.img, (w, h), interpolation=cv2.INTER_LINEAR)
    # 淡入合成：dst = a*cat + (1-a)*roi
    blended = cv2.addWeighted(img, a, roi, 1.0 - a, 0)
    canvas[y:y + h, x:x + w] = blended

    print("貓有出現!!")


def main():
    args = parse_args()

    # 背景影片
    bgcap = cv2.VideoCapture(args.bg_video)
    if not bgcap.isOpened():
        raise FileNotFoundError(f"無法開啟背景影片：{args.bg_video}")

    # 攝影機（用於偵測）
    cam = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if cv2.getBuildInformation().find("MSVC") != -1 else cv2.VideoCapture(args.camera)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cam.isOpened():
        raise RuntimeError("無法開啟攝影機。請確認連線或索引是否正確。")

    # 模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_half = device.startswith("cuda")
    model = YOLO(args.weights)
    model.to(device)

    # 貓圖
    cat_paths = [s.strip() for s in args.cats.split(",") if s.strip()]
    cat_imgs: List[np.ndarray] = []
    for pth in cat_paths:
        img = cv2.imread(pth, cv2.IMREAD_COLOR)
        if img is not None:
            cat_imgs.append(img)
    if not cat_imgs:
        print("警告：找不到可用的貓咪圖片，將不顯示貓咪淡入效果。")

    # 狀態
    prev_t = time.time()
    fps = 0.0
    ripples: List[Ripple] = []
    locked_center: Optional[Tuple[float, float]] = None
    hold_start: Optional[float] = None
    last_trigger: float = 0.0
    active_cat: Optional[CatOverlay] = None

    win_name = "Video BG + Pose Ripples + Cats"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    while True:
        # 讀背景影格；若結束則循環
        ok_bg, bg_frame = bgcap.read()
        if not ok_bg or bg_frame is None:
            bgcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok_bg, bg_frame = bgcap.read()
            if not ok_bg or bg_frame is None:
                raise RuntimeError("背景影片讀取失敗。")
        canvas = resize_fit(bg_frame, args.width, args.height)
        h, w = canvas.shape[:2]

        # 讀取攝影機影格
        ok_cam, cam_frame = cam.read()
        if not ok_cam:
            print("讀取攝影機影像失敗，嘗試繼續……")
            continue

        # 推論（僅用於偵測）
        results = model(
            cam_frame,
            imgsz=args.imgsz,
            conf=args.conf,
            verbose=False,
            half=use_half,
        )
        res = results[0]
        kxy, kcf = extract_kpts(res)

        # 找出同時擁有 5 臉部點的候選人與其 nose 中心
        candidates: List[int] = []
        centers: List[Tuple[float, float]] = []
        for i in range(kxy.shape[0]):
            cf = kcf[i]
            if all(cf[idx] > 0.25 for idx in FACE_IDXS):
                candidates.append(i)
                centers.append((float(kxy[i, 0, 0]), float(kxy[i, 0, 1])))  # 以鼻子為中心

        # 選擇/追蹤鎖定對象
        locked_idx: Optional[int] = None
        if candidates:
            sel = choose_locked_person(locked_center, [centers[candidates.index(i)] for i in candidates])
            if sel is not None:
                locked_idx = candidates[sel]
                locked_center = centers[sel]
        else:
            locked_center = None

        now = time.time()

        # 檢查是否持續滿足條件
        face_ready = locked_idx is not None
        if face_ready:
            if hold_start is None:
                hold_start = now
        else:
            hold_start = None

        # 滿足 3 秒且過了冷卻 → 觸發
        if hold_start is not None and (now - hold_start) >= args.face_hold_sec and (now - last_trigger) >= args.trigger_cooldown:
            nose_x, nose_y = locked_center if locked_center is not None else (w // 2, h // 2)
            ripples.append(Ripple(x=nose_x, y=nose_y))
            last_trigger = now
            hold_start = None  # 重置避免連續觸發

            # 啟動隨機貓圖淡入
            if cat_imgs:
                cat_img = random.choice(cat_imgs)
                # 尺寸：寬度為畫面 40%，等比
                target_w = int(w * 0.4)
                scale = target_w / cat_img.shape[1]
                target_h = int(cat_img.shape[0] * scale)
                # 放置位置：置中靠下
                x = (w - target_w) // 2
                y = max(0, h - target_h - int(h * 0.05))
                active_cat = CatOverlay(img=cat_img, start=now, duration=max(0.5, args.cat_fade), x=x, y=y, w=target_w, h=target_h)

        # 更新水波並繪製
        ripples = [rp for rp in ripples if rp.step()]
        draw_ripples(canvas, ripples)

        # （可選）骨架疊圖
        if args.draw_skeleton:
            draw_skeletons(canvas, kxy, kcf, lw=args.line_width)

        # 繪製貓咪淡入
        if active_cat is not None:
            overlay_cat(canvas, active_cat, now)

        # 圓形遮罩＋邊框
        diameter = int(min(h, w) * args.circle_ratio)
        radius = diameter // 2
        center = (w // 2, h // 2)
        mask = create_circular_mask(h, w, center=center, radius=radius)
        masked = cv2.bitwise_and(canvas, canvas, mask=mask)
        cv2.circle(masked, center, radius, (255, 255, 255), 3)

        # FPS 顯示
        dt = now - prev_t
        prev_t = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt
        cv2.putText(masked, f"FPS: {fps:.1f} | Device: {device}{' (half)' if use_half else ''}", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        # 顯示
        cv2.imshow(win_name, masked)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cam.release()
    bgcap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
