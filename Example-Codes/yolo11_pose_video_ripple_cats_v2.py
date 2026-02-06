"""
YOLO11 Pose 視訊背景＋條件觸發水波＆貓咪淡入（多人追蹤＆邊框貼合版）
Python 3.11

新增／修正（對應需求 1–5）：
1) 新增支援 cat0.png、cat4.png（亦可自動搜尋 cat*.png|jpg）。
2) 新增 `--cat_size_ratio`（預設 0.25），以「原圖大小」百分比縮放貓圖顯示。
3) 貓圖顯示位置＝圓形視窗邊框上「距鼻子最近」的點；貼邊顯示（向內略微偏移避免被遮）。
4) 加入簡易多人追蹤（centroid matching）。同一 Track 只會觸發一次水波＋貓淡入。
5) 若該人離開偵測範圍（逾 `--miss_timeout`），移除其 Track 與對應貓圖（貓立即消失）。

仍保留：背景 mp4 循環、圓形遮罩、（可選）骨架疊圖、PNG Alpha 正確淡入。

執行示例：
  python yolo11_pose_video_ripple_cats_v2.py \
    --bg_video ./calm_water.mp4 \
    --cats ./cat0.png,./cat1.png,./cat2.png,./cat3.png,./cat4.png \
    --cat_size_ratio 0.25 \
    --camera 0 --weights yolo11n-pose.pt --draw_skeleton
"""
from __future__ import annotations
import argparse
import time
import random
import glob
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

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
    img: np.ndarray  # BGR 或 BGRA
    start: float
    duration: float = 3.0  # 秒（淡入時間）
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0
    def alpha(self, now: float) -> float:
        a = (now - self.start) / max(1e-6, self.duration)
        return float(np.clip(a, 0.0, 1.0))

@dataclass
class Track:
    id: int
    center: Tuple[float, float]
    last_seen: float
    hold_start: Optional[float] = None
    triggered: bool = False
    cat: Optional[CatOverlay] = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO11 Pose｜影片背景＋條件觸發水波＆貓咪淡入（多人追蹤版）")
    p.add_argument("--camera", type=int, default=0, help="攝影機索引（預設 0）")
    p.add_argument("--weights", type=str, default="yolo11n-pose.pt", help="YOLO11 pose 權重檔")
    p.add_argument("--imgsz", type=int, default=640, help="推論輸入大小")
    p.add_argument("--width", type=int, default=1280, help="視窗寬度")
    p.add_argument("--height", type=int, default=720, help="視窗高度")
    p.add_argument("--circle_ratio", type=float, default=0.95, help="圓形視窗直徑相對於短邊比例 (0~1)")
    p.add_argument("--conf", type=float, default=0.25, help="偵測信心門檻")
    p.add_argument("--draw_skeleton", action="store_true", help="於背景上繪製骨架")
    p.add_argument("--line_width", type=int, default=2, help="骨架線寬")

    # 背景影片與貓咪設定
    p.add_argument("--bg_video", type=str, default="./calm_water.mp4", help="背景 mp4 路徑")
    p.add_argument("--cats", type=str, default="./cat0.png,./cat1.png,./cat2.png,./cat3.png,./cat4.png", help="以逗號分隔的貓圖路徑；可留空自動搜尋")
    p.add_argument("--cat_fade", type=float, default=3.0, help="貓咪淡入秒數")
    p.add_argument("--cat_size_ratio", type=float, default=0.25, help="貓圖顯示尺寸＝原圖的比例（0~1）")

    # 觸發條件／追蹤
    p.add_argument("--face_hold_sec", type=float, default=3.0, help="需連續滿足的秒數")
    p.add_argument("--trigger_cooldown", type=float, default=3.0, help="全域冷卻秒數（避免短時間多次觸發）")
    p.add_argument("--match_thresh", type=float, default=100.0, help="追蹤匹配距離閾值（像素）")
    p.add_argument("--miss_timeout", type=float, default=1.5, help="追蹤遺失多久視為離開（秒）")
    return p.parse_args()


def create_circular_mask(h: int, w: int, center: Tuple[int, int] | None = None, radius: int | None = None) -> np.ndarray:
    if center is None:
        center = (w // 2, h // 2)
    if radius is None:
        radius = min(h, w) // 2
    Y, X = np.ogrid[:h, :w]
    return (((X - center[0]) ** 2 + (Y - center[1]) ** 2) <= radius ** 2).astype(np.uint8) * 255


def resize_fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
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
    t = cat.alpha(now)
    if t <= 0.0:
        return
    x, y, w, h = cat.x, cat.y, cat.w, cat.h
    h = min(h, canvas.shape[0] - y)
    w = min(w, canvas.shape[1] - x)
    if w <= 0 or h <= 0:
        return
    roi = canvas[y:y + h, x:x + w]
    img = cv2.resize(cat.img, (w, h), interpolation=cv2.INTER_LINEAR)
    if img.ndim == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3].astype(np.float32)
        alpha = (img[:, :, 3].astype(np.float32) / 255.0) * t
        alpha = alpha[..., None]
        out = bgr * alpha + roi.astype(np.float32) * (1.0 - alpha)
        roi[:] = np.clip(out, 0, 255).astype(np.uint8)
    else:
        blended = cv2.addWeighted(img, t, roi, 1.0 - t, 0)
        roi[:] = blended


def discover_cat_paths(cats_arg: str) -> List[str]:
    paths: List[str] = []
    if cats_arg:
        paths.extend([s.strip() for s in cats_arg.split(',') if s.strip()])
    patterns = ["./cats/*.png", "./cats/*.jpg", "./cat*.png", "./cat*.jpg"]
    for pat in patterns:
        for p in glob.glob(pat):
            if p not in paths:
                paths.append(p)
    return paths


def nearest_point_on_circle(center: Tuple[int, int], radius: int, p: Tuple[float, float]) -> Tuple[int, int]:
    cx, cy = center
    vx, vy = p[0] - cx, p[1] - cy
    norm = (vx * vx + vy * vy) ** 0.5
    if norm < 1e-3:
        return (cx + radius, cy)
    ux, uy = vx / norm, vy / norm
    px = int(round(cx + ux * radius))
    py = int(round(cy + uy * radius))
    return px, py


def place_cat_along_circle(center: Tuple[int, int], radius: int, nose: Tuple[float, float], cat_w: int, cat_h: int, inset: int = 6) -> Tuple[int, int]:
    # 邊框上最近點，向內微移避免被圓邊遮擋
    px, py = nearest_point_on_circle(center, radius - inset, nose)
    # 以該點為中心放置
    x = int(px - cat_w // 2)
    y = int(py - cat_h // 2)
    return x, y


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

    # 讀取貓圖（支援 Alpha）
    cat_paths = discover_cat_paths(args.cats)
    cat_imgs: List[np.ndarray] = []
    for pth in cat_paths:
        img = cv2.imread(pth, cv2.IMREAD_UNCHANGED)
        if img is not None:
            cat_imgs.append(img)
    cats_loaded = len(cat_imgs)

    # 狀態
    prev_t = time.time()
    fps = 0.0
    ripples: List[Ripple] = []
    next_track_id = 1
    tracks: Dict[int, Track] = {}
    last_global_trigger: float = 0.0

    win_name = "Video BG + Pose Ripples + Cats (Tracks)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    while True:
        # 背景影格循環
        ok_bg, bg_frame = bgcap.read()
        if not ok_bg or bg_frame is None:
            bgcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok_bg, bg_frame = bgcap.read()
            if not ok_bg or bg_frame is None:
                raise RuntimeError("背景影片讀取失敗。")
        canvas = resize_fit(bg_frame, args.width, args.height)
        h, w = canvas.shape[:2]

        # 圓形視窗參數
        diameter = int(min(h, w) * args.circle_ratio)
        radius = diameter // 2
        center = (w // 2, h // 2)

        # 攝影機影格
        ok_cam, cam_frame = cam.read()
        if not ok_cam:
            print("讀取攝影機影像失敗，嘗試繼續……")
            continue

        # 推論（僅用於偵測）
        results = model(cam_frame, imgsz=args.imgsz, conf=args.conf, verbose=False, half=use_half)
        res = results[0]
        kxy, kcf = extract_kpts(res)

        # 收集候選（5 臉部點全有）
        centers: List[Tuple[float, float]] = []
        cand_idx: List[int] = []
        for i in range(kxy.shape[0]):
            cf = kcf[i]
            if all(cf[idx] > 0.25 for idx in FACE_IDXS):
                centers.append((float(kxy[i, 0, 0]), float(kxy[i, 0, 1])))  # 鼻子
                cand_idx.append(i)

        now = time.time()

        # 建立所有配對距離
        pair_dists: List[Tuple[float, int, int]] = []  # (dist, track_id, cand_j)
        for tid, tr in tracks.items():
            for j, c in enumerate(centers):
                dx = tr.center[0] - c[0]
                dy = tr.center[1] - c[1]
                d2 = (dx * dx + dy * dy) ** 0.5
                if d2 <= args.match_thresh:
                    pair_dists.append((d2, tid, j))
        pair_dists.sort(key=lambda x: x[0])

        assigned_tracks: set[int] = set()
        assigned_cands: set[int] = set()

        # 先配對既有 tracks
        for _, tid, j in pair_dists:
            if tid in assigned_tracks or j in assigned_cands:
                continue
            tr = tracks[tid]
            tr.center = centers[j]
            tr.last_seen = now
            assigned_tracks.add(tid)
            assigned_cands.add(j)

            # 條件累積（該 cand 已滿足 5 臉點）
            if tr.hold_start is None:
                tr.hold_start = now
            if (not tr.triggered) and (now - tr.hold_start >= args.face_hold_sec) and (now - last_global_trigger >= args.trigger_cooldown):
                # 觸發：水波 at 鼻子、貓淡入 at 邊框最近點
                nx, ny = tr.center
                nx = int(np.clip(nx, 0, w - 1))
                ny = int(np.clip(ny, 0, h - 1))
                ripples.append(Ripple(x=nx, y=ny))
                last_global_trigger = now
                tr.triggered = True

                if cats_loaded > 0:
                    cat_img = random.choice(cat_imgs)
                    # 尺寸以原圖比例縮放
                    target_w = max(8, int(cat_img.shape[1] * args.cat_size_ratio))
                    target_h = max(8, int(cat_img.shape[0] * args.cat_size_ratio))
                    # 邊框貼近：以鼻子→圓心向量找到邊界點
                    x, y = place_cat_along_circle(center, radius, (nx, ny), target_w, target_h, inset=6)
                    tr.cat = CatOverlay(img=cat_img, start=now, duration=max(0.5, args.cat_fade), x=x, y=y, w=target_w, h=target_h)

        # 為未配對的候選新增 tracks
        for j, c in enumerate(centers):
            if j in assigned_cands:
                continue
            tracks[next_track_id] = Track(id=next_track_id, center=c, last_seen=now, hold_start=now)
            next_track_id += 1

        # 移除長時間未見的 tracks（並讓貓消失）
        to_delete = []
        for tid, tr in tracks.items():
            if tid in assigned_tracks:
                continue
            if now - tr.last_seen > args.miss_timeout:
                to_delete.append(tid)
        for tid in to_delete:
            tracks.pop(tid, None)

        # 更新水波
        ripples = [rp for rp in ripples if rp.step()]
        draw_ripples(canvas, ripples)

        # （可選）骨架疊圖
        if args.draw_skeleton:
            draw_skeletons(canvas, kxy, kcf, lw=args.line_width)

        # 繪製所有存活 track 的貓圖（淡入完成後持續顯示，直到此人離開被移除）
        for tr in tracks.values():
            if tr.cat is not None:
                overlay_cat(canvas, tr.cat, now)

        # 圓形遮罩＋邊框
        mask = create_circular_mask(h, w, center=center, radius=radius)
        masked = cv2.bitwise_and(canvas, canvas, mask=mask)
        cv2.circle(masked, center, radius, (255, 255, 255), 3)

        # FPS 與狀態
        dt = now - prev_t
        prev_t = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt
        status = f"FPS:{fps:.1f}  Cats:{cats_loaded}  Tracks:{len(tracks)}"
        cv2.putText(masked, status, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

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
