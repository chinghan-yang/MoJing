"""
YOLO11 Pose 視訊背景＋影片素材水波＋貓咪淡入（繼承機制版）
Python 3.11

修改重點：
1. 水波播放結束時，若原始觸發者已離開，會自動尋找場上其他目標（模擬「人回到畫面」的情況）。
2. 若水波結束時場上無人，則不生成貓咪（避免虛空貓）。
3. 繼承機制：新的目標會直接獲得貓咪，無需重新等待觸發。

執行示例：
  python yolo11_pose_video_ripple_movie_handover.py \
    --bg_video ./calm_water.mp4 \
    --ripple_video ./Ripple.mp4 \
    --cats ./cat0.png,./cat1.png,./cat2.png,./cat3.png,./cat4.png \
    --cat_size_ratio 0.25 --cat_fade 3.0 --follow_smooth 0.18 \
    --camera 0 --weights yolo11n-pose.pt --draw_skeleton
"""
from __future__ import annotations
import argparse
import time
import random
import glob
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# COCO 17 關鍵點：0鼻, 1左眼, 2右眼, 3左耳, 4右耳, 5左肩, 6右肩
FACE_IDXS = [0, 1, 2, 3, 4, 5, 6]

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

# ----------------- 獨立水波管理 -----------------
@dataclass
class ActiveRipple:
    """獨立管理水波播放"""
    frame_idx: int = 0
    owner_track_id: int = -1  # 紀錄原始觸發者 ID

# ----------------- 貓圖與追蹤 -----------------
@dataclass
class CatOverlay:
    base: np.ndarray
    start: float
    duration: float = 3.0
    cx: float = 0.0
    cy: float = 0.0
    rot_deg: float = 0.0

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

# ----------------- 參數 -----------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO11 Pose｜影片水波繼承版")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--weights", type=str, default="yolo11n-pose.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--width", type=int, default=1080)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--circle_ratio", type=float, default=0.95)
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--draw_skeleton", action="store_true")
    p.add_argument("--line_width", type=int, default=2)

    # 背景、水波影片與貓
    p.add_argument("--bg_video", type=str, default="./calm_water.mp4")
    p.add_argument("--ripple_video", type=str, default="./Ripple.mp4", help="水波特效影片路徑")
    p.add_argument("--cats", type=str, default="./cat0.png,./cat1.png,./cat2.png,./cat3.png,./cat4.png")
    p.add_argument("--cat_fade", type=float, default=3.0)
    p.add_argument("--cat_size_ratio", type=float, default=0.25)
    p.add_argument("--follow_smooth", type=float, default=0.18)
    p.add_argument("--rot_offset", type=float, default=0.0)

    # 觸發／追蹤
    p.add_argument("--face_hold_sec", type=float, default=3.0)
    p.add_argument("--trigger_cooldown", type=float, default=3.0)
    p.add_argument("--match_thresh", type=float, default=100.0)
    p.add_argument("--miss_timeout", type=float, default=1.5)

    return p.parse_args()

# ----------------- 工具函式 -----------------

def load_video_frames(path: str, width: int, height: int) -> List[np.ndarray]:
    frames = []
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Warning: 無法開啟水波影片 {path}，將不會顯示水波特效。")
        return []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        frames.append(resized)

    cap.release()
    print(f"已載入水波影片: {len(frames)} 幀")
    return frames

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
    xy = result.keypoints.xy
    conf = getattr(result.keypoints, "conf", None)
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
        xy = kpts_xy[i]; cf = kpts_conf[i]
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

def ensure_bgra(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 4:
        return img
    if img.ndim == 3 and img.shape[2] == 3:
        a = np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.uint8)
        return np.concatenate([img, a], axis=2)
    raise ValueError("Invalid image shape for BGRA conversion")

def rotate_with_alpha(bgra: np.ndarray, angle_deg: float, scale: float = 1.0) -> np.ndarray:
    bgra = ensure_bgra(bgra)
    (h, w) = bgra.shape[:2]
    c = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(c, angle_deg, scale)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    nw = int((h * sin) + (w * cos))
    nh = int((h * cos) + (w * sin))
    M[0, 2] += (nw / 2) - c[0]
    M[1, 2] += (nh / 2) - c[1]
    rotated = cv2.warpAffine(bgra, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return rotated

def overlay_bgra_center(canvas: np.ndarray, bgra: np.ndarray, cx: int, cy: int, alpha_scale: float):
    if alpha_scale <= 0.0:
        return
    bgra = ensure_bgra(bgra)
    h, w = bgra.shape[:2]
    x0 = int(cx - w // 2); y0 = int(cy - h // 2)
    x1 = max(0, x0); y1 = max(0, y0)
    x2 = min(canvas.shape[1], x0 + w); y2 = min(canvas.shape[0], y0 + h)
    if x1 >= x2 or y1 >= y2: return
    cut = bgra[y1 - y0:y2 - y0, x1 - x0:x2 - x0]
    roi = canvas[y1:y2, x1:x2]
    bgr = cut[:, :, :3].astype(np.float32)
    a = (cut[:, :, 3].astype(np.float32) / 255.0) * alpha_scale
    a = a[..., None]
    out = bgr * a + roi.astype(np.float32) * (1.0 - a)
    canvas[y1:y2, x1:x2] = np.clip(out, 0, 255).astype(np.uint8)

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

def tangent_angle_deg(center: Tuple[int, int], p: Tuple[int, int], clockwise: bool = True, offset_deg: float = 0.0) -> float:
    cx, cy = center
    rx, ry = p[0] - cx, p[1] - cy
    if clockwise:
        tx, ty = (ry, -rx)
    else:
        tx, ty = (-ry, rx)
    angle = np.degrees(np.arctan2(ty, tx))
    return float(angle + offset_deg)

# ----------------- 主流程 -----------------

def main():
    args = parse_args()

    bgcap = cv2.VideoCapture(args.bg_video)
    if not bgcap.isOpened():
        raise FileNotFoundError(f"無法開啟背景影片：{args.bg_video}")

    ripple_frames = load_video_frames(args.ripple_video, args.width, args.height)

    cam = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if cv2.getBuildInformation().find("MSVC") != -1 else cv2.VideoCapture(args.camera)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cam.isOpened():
        raise RuntimeError("無法開啟攝影機。")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_half = device.startswith("cuda")
    model = YOLO(args.weights)
    model.to(device)

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

    cat_paths = discover_cat_paths(args.cats)
    raw_cats: List[np.ndarray] = []
    for pth in cat_paths:
        img = cv2.imread(pth, cv2.IMREAD_UNCHANGED)
        if img is not None:
            if img.ndim == 3 and img.shape[2] == 3:
                a = np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.uint8)
                img = np.concatenate([img, a], axis=2)
            tw = max(8, int(img.shape[1] * args.cat_size_ratio))
            th = max(8, int(img.shape[0] * args.cat_size_ratio))
            raw_cats.append(cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR))
    cats_loaded = len(raw_cats)

    prev_t = time.time(); fps = 0.0
    next_track_id = 1
    tracks: Dict[int, Track] = {}

    running_ripples: List[ActiveRipple] = []
    last_global_trigger: float = 0.0

    win_name = "Video BG + Ripple Movie (Handover) + Cats"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ok_bg, bg_frame = bgcap.read()
        if not ok_bg or bg_frame is None:
            bgcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok_bg, bg_frame = bgcap.read()
            if not ok_bg or bg_frame is None:
                raise RuntimeError("背景影片讀取失敗。")
        canvas = resize_fit(bg_frame, args.width, args.height)
        h, w = canvas.shape[:2]

        diameter = int(min(h, w) * args.circle_ratio)
        radius = diameter // 2
        center = (w // 2, h // 2)

        ok_cam, cam_frame = cam.read()
        if not ok_cam:
            print("讀取攝影機影像失敗，嘗試繼續……")
            continue

        results = model(cam_frame, imgsz=args.imgsz, conf=args.conf, verbose=False, half=use_half)
        res = results[0]
        kxy, kcf = extract_kpts(res)

        centers: List[Tuple[float, float]] = []
        for i in range(kxy.shape[0]):
            cf = kcf[i]
            if all(cf[idx] > 0.25 for idx in FACE_IDXS):
                centers.append((float(kxy[i, 0, 0]), float(kxy[i, 0, 1])))

        now = time.time()

        pair: List[Tuple[float, int, int]] = []
        for tid, tr in tracks.items():
            for j, c in enumerate(centers):
                d = ((tr.center[0]-c[0])**2 + (tr.center[1]-c[1])**2) ** 0.5
                if d <= args.match_thresh:
                    pair.append((d, tid, j))
        pair.sort(key=lambda x: x[0])
        assigned_t, assigned_c = set(), set()
        for _, tid, j in pair:
            if tid in assigned_t or j in assigned_c:
                continue
            tr = tracks[tid]
            tr.center = centers[j]
            tr.last_seen = now
            assigned_t.add(tid); assigned_c.add(j)
            if tr.hold_start is None:
                tr.hold_start = now

            # ----------------- 邏輯核心 -----------------
            cats_on_screen = any(t.cat is not None for t in tracks.values())
            ripples_active = len(running_ripples) > 0

            # 觸發條件
            if (not tr.triggered) and \
               (now - tr.hold_start >= args.face_hold_sec) and \
               (now - last_global_trigger >= args.trigger_cooldown) and \
               (not cats_on_screen) and (not ripples_active):

                print(f"Track {tr.id} Triggered! Spawning Independent Ripple...")
                tr.triggered = True
                last_global_trigger = now

                if len(ripple_frames) > 0:
                    running_ripples.append(ActiveRipple(frame_idx=0, owner_track_id=tr.id))
                else:
                    if cats_loaded > 0:
                        nx, ny = tr.center
                        px, py = nearest_point_on_circle(center, radius - 6, (nx, ny))
                        rot = tangent_angle_deg(center, (px, py), clockwise=True, offset_deg=args.rot_offset)
                        tr.cat = CatOverlay(base=random.choice(raw_cats), start=now, duration=args.cat_fade, cx=float(px), cy=float(py), rot_deg=rot)

        # ----------------- 繪製水波 (獨立於 Track 迴圈) -----------------
        finished_ripples = []
        for rp in running_ripples:
            if rp.frame_idx < len(ripple_frames):
                cv2.add(canvas, ripple_frames[rp.frame_idx], dst=canvas)
                rp.frame_idx += 1
            else:
                finished_ripples.append(rp)

        # ----------------- [核心修改] 處理水波結束 -> 繼承與生成 -----------------
        for rp in finished_ripples:
            running_ripples.remove(rp)
            print(f"Ripple finished. Checking for target...")

            target_track: Optional[Track] = None

            # 1. 優先檢查原始觸發者
            if rp.owner_track_id in tracks:
                target_track = tracks[rp.owner_track_id]
                print(f" -> Original owner (ID {rp.owner_track_id}) found.")

            # 2. 若原始觸發者消失，檢查場上是否有其他無貓的目標 (繼承機制)
            else:
                # 尋找還沒有貓咪的 Track
                candidates = [t for t in tracks.values() if t.cat is None]
                if candidates:
                    target_track = candidates[0] # 取第一個找到的
                    # 將觸發狀態同步給這位幸運兒，避免他立刻又觸發一次水波
                    target_track.triggered = True
                    print(f" -> Owner lost. Handing over cat to new Track ID {target_track.id}.")
                else:
                    print(" -> No one on screen. No cat spawned.")

            # 3. 若有目標，執行生成
            if target_track is not None and cats_loaded > 0:
                nx, ny = target_track.center
                px, py = nearest_point_on_circle(center, radius - 6, (nx, ny))
                rot = tangent_angle_deg(center, (px, py), clockwise=True, offset_deg=args.rot_offset)
                target_track.cat = CatOverlay(base=random.choice(raw_cats), start=now, duration=args.cat_fade, cx=float(px), cy=float(py), rot_deg=rot)

        # 新增與刪除 Track
        for j, c in enumerate(centers):
            if j in assigned_c:
                continue
            tracks[next_track_id] = Track(id=next_track_id, center=c, last_seen=now, hold_start=now)
            next_track_id += 1

        to_del = []
        for tid, tr in tracks.items():
            if tid in assigned_t:
                continue
            if now - tr.last_seen > args.miss_timeout:
                to_del.append(tid)
        for tid in to_del:
            tracks.pop(tid, None)

        if args.draw_skeleton:
            draw_skeletons(canvas, kxy, kcf, lw=args.line_width)

        for tid, tr in tracks.items():
            if tr.cat is None:
                continue
            nx, ny = tr.center
            px, py = nearest_point_on_circle(center, radius - 6, (nx, ny))
            a = float(np.clip(args.follow_smooth, 0.0, 1.0))
            tr.cat.cx = tr.cat.cx * (1 - a) + px * a
            tr.cat.cy = tr.cat.cy * (1 - a) + py * a
            tr.cat.rot_deg = tangent_angle_deg(center, (int(tr.cat.cx), int(tr.cat.cy)), clockwise=True, offset_deg=args.rot_offset)
            rotated = rotate_with_alpha(tr.cat.base, tr.cat.rot_deg)
            overlay_bgra_center(canvas, rotated, int(round(tr.cat.cx)), int(round(tr.cat.cy)), tr.cat.alpha(now))

        mask = create_circular_mask(h, w, center=center, radius=radius)
        masked = cv2.bitwise_and(canvas, canvas, mask=mask)
        cv2.circle(masked, center, radius, (255, 255, 255), 3)

        dt = now - prev_t; prev_t = now
        if dt > 0: fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt
        cv2.putText(masked, f"FPS:{fps:.1f} Ripples:{len(running_ripples)} Tracks:{len(tracks)}", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(win_name, masked)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cam.release(); bgcap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()