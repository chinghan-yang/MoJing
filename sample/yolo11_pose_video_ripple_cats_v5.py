"""
YOLO11 Pose 視訊背景＋條件觸發水波＆貓咪淡入（沿圓邊旋轉＆跟隨 + 影片風格水波）
Python 3.11

新增（對齊上傳 Ripple.mp4 的視覺風格）：
- 將水波改為 **折射變形（refraction displacement）**，以徑向正弦波 + 時間/半徑衰減模擬真實水面。
- 以 `cv2.remap` 做區域像素位移，支援多個同時波紋，並可調波長、速度、幅度、衰減與高光強度。
- 保留先前功能：貓圖沿圓邊切線旋轉＆跟隨、同人只觸發一次、離開則貓消失、PNG Alpha 淡入、圓形遮罩、骨架疊圖。

執行示例：
  python yolo11_pose_video_ripple_cats_v5.py \
    --bg_video ./calm_water.mp4 \
    --cats ./cat0.png,./cat1.png,./cat2.png,./cat3.png,./cat4.png \
    --cat_size_ratio 0.25 --cat_fade 3.0 --follow_smooth 0.18 \
    --ripple_lambda 24 --ripple_speed 180 --ripple_amp 6 --ripple_radial_decay 0.015 --ripple_time_tau 1.6 --ripple_highlight 0.22 \
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

# ----------------- 水波（影片風格）資料結構 -----------------
@dataclass
class Ripple:
    x: float
    y: float
    start: float
    # 視覺參數（可由 CLI 指定預設）
    wavelength: float  # 像素，波長
    speed: float       # 像素/秒，相速度
    amp: float         # 像素，位移幅度
    radial_decay: float  # 每像素的徑向衰減係數（越大越快消失）
    time_tau: float      # 秒，時間衰減常數（e^-t/tau）
    highlight: float     # 0~1，高光強度

    def alive(self, now: float) -> bool:
        # 生命期：當時間衰減很小或擴散過大就視為結束
        t = now - self.start
        return t < (self.time_tau * 5.0)

# ----------------- 貓圖與追蹤 -----------------
@dataclass
class CatOverlay:
    base: np.ndarray  # BGRA（已縮放）
    start: float
    duration: float = 3.0  # 淡入秒數
    cx: float = 0.0  # 以圖中心對齊的擺放坐標
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
    p = argparse.ArgumentParser(description="YOLO11 Pose｜影片背景＋水波＆貓淡入（旋轉跟隨 + 影片風格水波）")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--weights", type=str, default="yolo11n-pose.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--circle_ratio", type=float, default=0.95)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--draw_skeleton", action="store_true")
    p.add_argument("--line_width", type=int, default=2)

    # 背景與貓
    p.add_argument("--bg_video", type=str, default="./calm_water.mp4")
    p.add_argument("--cats", type=str, default="./cat0.png,./cat1.png,./cat2.png,./cat3.png,./cat4.png")
    p.add_argument("--cat_fade", type=float, default=3.0)
    p.add_argument("--cat_size_ratio", type=float, default=0.25)
    p.add_argument("--follow_smooth", type=float, default=0.18)
    p.add_argument("--rot_offset", type=float, default=30.0)

    # 觸發／追蹤
    p.add_argument("--face_hold_sec", type=float, default=3.0)
    p.add_argument("--trigger_cooldown", type=float, default=3.0)
    p.add_argument("--match_thresh", type=float, default=100.0)
    p.add_argument("--miss_timeout", type=float, default=1.5)

    # 水波（影片風格）
    p.add_argument("--ripple_lambda", type=float, default=24.0)
    p.add_argument("--ripple_speed", type=float, default=180.0)
    p.add_argument("--ripple_amp", type=float, default=6.0)
    p.add_argument("--ripple_radial_decay", type=float, default=0.015)
    p.add_argument("--ripple_time_tau", type=float, default=1.6)
    p.add_argument("--ripple_highlight", type=float, default=0.22)
    return p.parse_args()

# ----------------- 共用工具 -----------------
COCO_EDGES = COCO_EDGES

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

# ---------- 旋轉與貼圖（支援 BGRA Alpha） ----------

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

# ---------- 幾何工具 ----------

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

# ---------- 水波（影片風格）核心：區域折射位移 ----------

def apply_ripples_refraction(canvas: np.ndarray, ripples: List[Ripple], now: float):
    if not ripples:
        return
    H, W = canvas.shape[:2]
    src = canvas.copy()

    for rp in ripples:
        if not rp.alive(now):
            continue
        t = now - rp.start
        # 影響半徑（隨時間擴散，外加 3 個波長裕度）
        R = int(min(max(rp.speed * t + 3 * rp.wavelength, rp.wavelength * 2), max(H, W)))
        x0 = max(0, int(rp.x - R)); y0 = max(0, int(rp.y - R))
        x1 = min(W, int(rp.x + R)); y1 = min(H, int(rp.y + R))
        if x1 - x0 <= 2 or y1 - y0 <= 2:
            continue

        roi = src[y0:y1, x0:x1]
        h, w = roi.shape[:2]
        # 座標網格（以波中心為原點）
        xs = np.arange(w, dtype=np.float32); ys = np.arange(h, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys)
        dx = X + x0 - rp.x
        dy = Y + y0 - rp.y
        r = np.sqrt(dx * dx + dy * dy) + 1e-6

        k = 2.0 * np.pi / rp.wavelength
        phase = k * (r - rp.speed * t)  # sin(k r - ω t)，其中 ω = k * v
        # 位移（徑向）：幅度 × 正弦 × 衰減（半徑與時間）
        decay = np.exp(-rp.radial_decay * r) * np.exp(-t / rp.time_tau)
        disp = (rp.amp * np.sin(phase) * decay).astype(np.float32)
        ux = (dx / r).astype(np.float32); uy = (dy / r).astype(np.float32)

        map_x = (X + disp * ux).astype(np.float32)
        map_y = (Y + disp * uy).astype(np.float32)

        distorted = cv2.remap(roi, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # 高光（沿波峰），模擬反光閃爍
        if rp.highlight > 1e-3:
            crest = (np.cos(phase) * 0.5 + 0.5) ** 6  # 越接近波峰越亮
            crest *= decay
            crest = np.clip(crest * rp.highlight, 0.0, 1.0).astype(np.float32)
            hl = (crest[..., None] * 255.0).astype(np.uint8)
            # 疊加到亮度：轉成 YUV 做亮度加成較自然
            yuv = cv2.cvtColor(distorted, cv2.COLOR_BGR2YUV)
            yuv[:, :, 0] = np.clip(yuv[:, :, 0].astype(np.int32) + (hl[:, :, 0] // 6), 0, 255).astype(np.uint8)
            distorted = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

        canvas[y0:y1, x0:x1] = distorted

# ----------------- 主流程 -----------------

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
        raise RuntimeError("無法開啟攝影機。")

    # 模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_half = device.startswith("cuda")
    model = YOLO(args.weights)
    model.to(device)

    # 讀取貓圖（保留 Alpha，並先縮放到目標比例）
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
                # 把沒有透明度通道的貓圖（BGR，3 通道）轉成含透明度的 4 通道（BGRA）
                a = np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.uint8)
                img = np.concatenate([img, a], axis=2)
            # 預縮放到比例;把原圖寬度乘上縮放比例，並且並設下最小 8 像素，避免比例太小造成 0 或過小尺寸
            tw = max(8, int(img.shape[1] * args.cat_size_ratio))
            th = max(8, int(img.shape[0] * args.cat_size_ratio))
            raw_cats.append(cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR))
    cats_loaded = len(raw_cats)

    # 狀態
    prev_t = time.time(); fps = 0.0
    ripples: List[Ripple] = []
    next_track_id = 1
    tracks: Dict[int, Track] = {}
    last_global_trigger: float = 0.0

    win_name = "Video BG + Pose Ripples + Cats (Film-like ripples)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        # 背景影格循環
        # 嘗試讀取下一張背景影格；ok_bg 表示是否成功、bg_frame 是影格資料。
        ok_bg, bg_frame = bgcap.read()
        if not ok_bg or bg_frame is None: # 若讀不到（例如影片播到尾端或解碼失敗），把播放位置設回第 0 張影格再讀一次，達到循環播放的效果；若重讀仍失敗就拋出錯誤。
            bgcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok_bg, bg_frame = bgcap.read()
            if not ok_bg or bg_frame is None:
                raise RuntimeError("背景影片讀取失敗。")
        canvas = resize_fit(bg_frame, args.width, args.height)
        h, w = canvas.shape[:2]

        # 圓形參數
        diameter = int(min(h, w) * args.circle_ratio)
        radius = diameter // 2
        center = (w // 2, h // 2)

        # 攝影機影格
        ok_cam, cam_frame = cam.read()
        if not ok_cam:
            print("讀取攝影機影像失敗，嘗試繼續……")
            continue

        # 推論
        results = model(cam_frame, imgsz=args.imgsz, conf=args.conf, verbose=False, half=use_half)
        res = results[0]
        kxy, kcf = extract_kpts(res)

        # 候選（五臉點）
        centers: List[Tuple[float, float]] = [] # 準備一List放每個人的代表座標（x, y）。型別註記表示元素是兩個浮點數的 tuple
        for i in range(kxy.shape[0]): # 逐一走訪這一幀偵測到的每個人
            cf = kcf[i] # 取出第 i 個人的各關鍵點信心值（與 kxy 對應的 confidence）
            if all(cf[idx] > 0.25 for idx in FACE_IDXS): # 只有當此人五個臉部關鍵點（鼻、左眼、右眼、左耳、右耳）都存在且信心值 > 0.25，才視為有效臉部
                centers.append((float(kxy[i, 0, 0]), float(kxy[i, 0, 1])))  # 鼻

        now = time.time()

        # 追蹤匹配
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
            if (not tr.triggered) and (now - tr.hold_start >= args.face_hold_sec) and (now - last_global_trigger >= args.trigger_cooldown):
                nx, ny = tr.center
                nx = int(np.clip(nx, 0, w - 1)); ny = int(np.clip(ny, 0, h - 1))
                # 建立「影片風格」水波
                ripples.append(Ripple(x=nx, y=ny, start=now,
                                      wavelength=args.ripple_lambda,
                                      speed=args.ripple_speed,
                                      amp=args.ripple_amp,
                                      radial_decay=args.ripple_radial_decay,
                                      time_tau=args.ripple_time_tau,
                                      highlight=args.ripple_highlight))
                last_global_trigger = now
                tr.triggered = True
                if cats_loaded > 0:
                    cat_raw = random.choice(raw_cats)
                    px, py = nearest_point_on_circle(center, radius - 6, (nx, ny))
                    rot = tangent_angle_deg(center, (px, py), clockwise=True, offset_deg=args.rot_offset)
                    tr.cat = CatOverlay(base=cat_raw, start=now, duration=max(0.5, args.cat_fade), cx=float(px), cy=float(py), rot_deg=rot)

        # 新增未配對 tracks
        for j, c in enumerate(centers):
            if j in assigned_c:
                continue
            tracks[next_track_id] = Track(id=next_track_id, center=c, last_seen=now, hold_start=now)
            next_track_id += 1

        # 移除離開的人（貓隨之消失）
        to_del = []
        for tid, tr in tracks.items():
            if tid in assigned_t:
                continue
            if now - tr.last_seen > args.miss_timeout:
                to_del.append(tid)
        for tid in to_del:
            tracks.pop(tid, None)

        # 套用「影片風格水波」
        ripples = [rp for rp in ripples if rp.alive(now)]
        apply_ripples_refraction(canvas, ripples, now)

        # 骨架（可選）
        if args.draw_skeleton:
            draw_skeletons(canvas, kxy, kcf, lw=args.line_width)

        # 更新與繪製貓圖（旋轉 + 沿圓邊跟隨）
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

        # 圓形遮罩 + 邊框
        mask = create_circular_mask(h, w, center=center, radius=radius)
        masked = cv2.bitwise_and(canvas, canvas, mask=mask)
        cv2.circle(masked, center, radius, (255, 255, 255), 3)

        # FPS
        dt = now - prev_t; prev_t = now
        if dt > 0: fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt
        cv2.putText(masked, f"FPS:{fps:.1f} Ripples:{len(ripples)} Tracks:{len(tracks)}", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(win_name, masked)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cam.release(); bgcap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
