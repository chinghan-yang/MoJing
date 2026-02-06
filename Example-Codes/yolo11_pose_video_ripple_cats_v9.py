"""
YOLO11 Pose 視訊背景＋影片素材水波＋動態貓咪序列圖（PNG Sequence 版）
Python 3.11

需求調整：
1. [修改] 貓咪素材讀取：讀取資料夾內的 PNG 序列 (00.png, 01.png...) 實現動態效果。
2. [新增] --anim_fps: 控制貓咪動作的播放速度 (預設 12 FPS)。
3. [保留] 所有先前功能：水波中途淡入、徑向偏移、鏡像翻轉、多人防衝突等。

執行示例：
  python yolo11_pose_anim_cats.py \
    --bg_video ./calm_water.mp4 \
    --ripple_video ./Ripple.mp4 \
    --cats ./cat0,./cat1,./cat2,./cat3,./cat4 \
    --cat_size_ratio 0.25 --cat_fade 3.0 --follow_smooth 0.15 \
    --cat_appear_phase 0.7 --cat_offset_radial 30 \
    --anim_fps 10 \
    --camera 0 --weights yolo11n-pose.pt --debug_angle
"""
from __future__ import annotations
import argparse
import time
import random
import glob
import math
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# COCO 17 關鍵點
FACE_IDXS = [0, 1, 2, 3, 4, 5, 6]
COCO_EDGES = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)
]


# ----------------- 資料結構 -----------------
@dataclass
class ActiveRipple:
    frame_idx: int = 0
    owner_track_id: int = -1
    cat_spawned: bool = False


@dataclass
class CatOverlay:
    frames: List[np.ndarray]  # [修改] 改為儲存一系列圖片
    start: float
    duration: float = 3.0
    cx: float = 0.0
    cy: float = 0.0
    rot_deg: float = 0.0

    # [新增] 動畫控制參數
    frame_idx: int = 0
    last_anim_time: float = 0.0
    anim_fps: float = 12.0

    def alpha(self, now: float) -> float:
        a = (now - self.start) / max(1e-6, self.duration)
        return float(np.clip(a, 0.0, 1.0))

    def get_current_frame(self, now: float) -> np.ndarray:
        """根據時間決定當前要顯示哪一張圖"""
        if len(self.frames) <= 1:
            return self.frames[0]

        # 檢查是否該換幀了
        if now - self.last_anim_time > (1.0 / self.anim_fps):
            self.frame_idx = (self.frame_idx + 1) % len(self.frames)
            self.last_anim_time = now

        return self.frames[self.frame_idx]


@dataclass
class PoseData:
    nose: Tuple[float, float]
    shoulder_center: Tuple[float, float]


@dataclass
class Track:
    id: int
    pose: PoseData
    last_seen: float
    hold_start: Optional[float] = None
    triggered: bool = False
    cat: Optional[CatOverlay] = None


# ----------------- 參數 -----------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO11 Pose｜動態貓咪序列圖版")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--weights", type=str, default="yolo11n-pose.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--width", type=int, default=1080)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--circle_ratio", type=float, default=0.95)
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--draw_skeleton", action="store_true")
    p.add_argument("--line_width", type=int, default=2)

    p.add_argument("--debug_angle", action="store_true", help="顯示角度數值")

    p.add_argument("--bg_video", type=str, default="./calm_water.mp4")
    p.add_argument("--ripple_video", type=str, default="./Ripple.mp4")

    # [修改] cats 參數現在建議傳入資料夾路徑，例如 "./cat0,./cat1"
    p.add_argument("--cats", type=str, default="./cat0,./cat1,./cat2,./cat3,./cat4",
                   help="貓咪素材資料夾路徑，逗號分隔")

    p.add_argument("--cat_fade", type=float, default=3.0)
    p.add_argument("--cat_size_ratio", type=float, default=0.25)
    p.add_argument("--follow_smooth", type=float, default=0.15)
    p.add_argument("--rot_offset", type=float, default=0.0)

    p.add_argument("--cat_appear_phase", type=float, default=0.7, help="水波播放進度 (0.0~1.0)")
    p.add_argument("--cat_offset_radial", type=float, default=0.0, help="貓咪徑向偏移")

    # [新增] 動畫速度
    p.add_argument("--anim_fps", type=float, default=12.0, help="貓咪動畫播放速度 (FPS)")

    p.add_argument("--face_hold_sec", type=float, default=3.0)
    p.add_argument("--trigger_cooldown", type=float, default=3.0)
    p.add_argument("--match_thresh", type=float, default=100.0)
    p.add_argument("--miss_timeout", type=float, default=1.5)
    return p.parse_args()


# ----------------- 幾何與工具函式 -----------------

def load_video_frames(path: str, width: int, height: int) -> List[np.ndarray]:
    frames = []
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Warning: 無法開啟影片 {path}")
        return []
    while True:
        ret, frame = cap.read()
        if not ret: break
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        frames.append(resized)
    cap.release()
    return frames


def create_circular_mask(h: int, w: int, center: Tuple[int, int] | None = None,
                         radius: int | None = None) -> np.ndarray:
    if center is None: center = (w // 2, h // 2)
    if radius is None: radius = min(h, w) // 2
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
    xy = result.keypoints.xy.cpu().numpy()
    conf = result.keypoints.conf
    kconf = conf.cpu().numpy() if conf is not None else np.ones((xy.shape[0], xy.shape[1]))
    return xy.astype(np.float32), kconf.astype(np.float32)


def draw_skeletons(canvas: np.ndarray, kpts_xy: np.ndarray, kpts_conf: np.ndarray, lw: int = 2):
    if kpts_xy.size == 0: return
    overlay = canvas.copy()
    for i in range(kpts_xy.shape[0]):
        xy = kpts_xy[i];
        cf = kpts_conf[i]
        for a, b in COCO_EDGES:
            if cf[a] > 0.2 and cf[b] > 0.2:
                cv2.line(overlay, (int(xy[a, 0]), int(xy[a, 1])), (int(xy[b, 0]), int(xy[b, 1])), (255, 255, 255), lw,
                         cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, dst=canvas)


def ensure_bgra(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 4: return img
    if img.ndim == 3 and img.shape[2] == 3:
        return np.concatenate([img, np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.uint8)], axis=2)
    raise ValueError("Invalid image")


def rotate_with_alpha(bgra: np.ndarray, angle_deg: float) -> np.ndarray:
    bgra = ensure_bgra(bgra)
    (h, w) = bgra.shape[:2]
    c = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(c, angle_deg, 1.0)
    cos = abs(M[0, 0]);
    sin = abs(M[0, 1])
    nw = int((h * sin) + (w * cos))
    nh = int((h * cos) + (w * sin))
    M[0, 2] += (nw / 2) - c[0]
    M[1, 2] += (nh / 2) - c[1]
    return cv2.warpAffine(bgra, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0, 0))


def overlay_bgra_center(canvas: np.ndarray, bgra: np.ndarray, cx: int, cy: int, alpha_scale: float):
    if alpha_scale <= 0.0: return
    bgra = ensure_bgra(bgra)
    h, w = bgra.shape[:2]
    x0 = int(cx - w // 2);
    y0 = int(cy - h // 2)
    x1 = max(0, x0);
    y1 = max(0, y0)
    x2 = min(canvas.shape[1], x0 + w);
    y2 = min(canvas.shape[0], y0 + h)
    if x1 >= x2 or y1 >= y2: return
    cut = bgra[y1 - y0:y2 - y0, x1 - x0:x2 - x0]
    roi = canvas[y1:y2, x1:x2]
    bgr = cut[:, :, :3].astype(np.float32)
    a = (cut[:, :, 3].astype(np.float32) / 255.0) * alpha_scale
    a = a[..., None]
    out = bgr * a + roi.astype(np.float32) * (1.0 - a)
    canvas[y1:y2, x1:x2] = np.clip(out, 0, 255).astype(np.uint8)


def get_ray_circle_intersection(center: Tuple[int, int], radius: int, origin: Tuple[float, float],
                                direction: Tuple[float, float]) -> Tuple[int, int]:
    cx, cy = center
    ox, oy = origin
    dx, dy = direction
    ex = ox - cx;
    ey = oy - cy
    a = dx * dx + dy * dy
    b = 2 * (ex * dx + ey * dy)
    c = ex * ex + ey * ey - radius * radius
    if a < 1e-6:
        dir_x, dir_y = ox - cx, oy - cy
        norm = (dir_x ** 2 + dir_y ** 2) ** 0.5
        if norm < 1e-3: return (cx, cy + radius)
        return (int(cx + dir_x / norm * radius), int(cy + dir_y / norm * radius))

    delta = b * b - 4 * a * c
    if delta < 0: return (int(ox), int(oy))
    t = (-b + np.sqrt(delta)) / (2 * a)
    return (int(round(ox + t * dx)), int(round(oy + t * dy)))


# [新增] 載入動畫序列的函式
def load_anim_sequences(cats_arg: str, size_ratio: float, max_h: float) -> List[List[np.ndarray]]:
    paths = []
    if cats_arg:
        paths.extend([s.strip() for s in cats_arg.split(',') if s.strip()])

    loaded_sequences = []

    for p in paths:
        frames = []
        # 判斷是資料夾還是檔案
        if os.path.isdir(p):
            # 讀取資料夾內的 png
            files = sorted(glob.glob(os.path.join(p, "*.png")))
        else:
            # 為了相容，如果指定的是檔案，就當作只有一幀的動畫
            files = [p] if os.path.exists(p) else []

        for f in files:
            img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            if img is not None:
                # 確保 BGRA
                if img.ndim == 3 and img.shape[2] == 3:
                    img = np.concatenate([img, np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.uint8)], axis=2)

                # 統一縮放
                tw = max(8, int(img.shape[1] * size_ratio))
                th = max(8, int(img.shape[0] * size_ratio))
                if th > max_h:
                    scale = max_h / float(th)
                    th = int(max_h)
                    tw = int(tw * scale)

                frames.append(cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR))

        if frames:
            loaded_sequences.append(frames)
            print(f"Loaded cat animation from {p}: {len(frames)} frames.")

    return loaded_sequences


# ----------------- 主流程 -----------------

def main():
    args = parse_args()

    bgcap = cv2.VideoCapture(args.bg_video)
    ripple_frames = load_video_frames(args.ripple_video, args.width, args.height)
    cam = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if cv2.getBuildInformation().find(
        "MSVC") != -1 else cv2.VideoCapture(args.camera)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    model = YOLO(args.weights)
    if torch.cuda.is_available(): model.to("cuda")

    # 幾何資訊
    diameter = int(min(args.height, args.width) * args.circle_ratio)
    radius = diameter // 2
    center = (args.width // 2, args.height // 2)

    # [修改] 載入貓咪動畫序列
    max_cat_h = float(radius)
    anim_sequences = load_anim_sequences(args.cats, args.cat_size_ratio, max_cat_h)
    cats_loaded = len(anim_sequences)

    if cats_loaded == 0:
        print("Warning: No cat images found! Check --cats path.")

    prev_t = time.time();
    fps = 0.0
    next_track_id = 1
    tracks: Dict[int, Track] = {}
    running_ripples: List[ActiveRipple] = []
    last_global_trigger = 0.0

    win_name = "Video BG + Animated Cats"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, bg_frame = bgcap.read()
        if not ret:
            bgcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, bg_frame = bgcap.read()

        canvas = resize_fit(bg_frame, args.width, args.height)
        h, w = canvas.shape[:2]

        ret, cam_frame = cam.read()
        if ret:
            cam_frame = cv2.flip(cam_frame, 1)
            results = model(cam_frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
            kxy, kcf = extract_kpts(results[0])
        else:
            kxy = np.zeros((0, 17, 2));
            kcf = np.zeros((0, 17))

        detected_poses = []
        for i in range(kxy.shape[0]):
            cf = kcf[i]
            has_basic = (cf[0] > 0.25) and (cf[5] > 0.25) and (cf[6] > 0.25)
            has_left_face = (cf[1] > 0.25) and (cf[3] > 0.25)
            has_right_face = (cf[2] > 0.25) and (cf[4] > 0.25)

            if has_basic and (has_left_face or has_right_face):
                nx, ny = kxy[i, 0]
                ls, rs = kxy[i, 5], kxy[i, 6]
                mx, my = (ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2
                detected_poses.append(PoseData(nose=(float(nx), float(ny)), shoulder_center=(float(mx), float(my))))

        now = time.time()

        pair = []
        for tid, tr in tracks.items():
            for j, p in enumerate(detected_poses):
                d = ((tr.pose.nose[0] - p.nose[0]) ** 2 + (tr.pose.nose[1] - p.nose[1]) ** 2) ** 0.5
                if d <= args.match_thresh: pair.append((d, tid, j))
        pair.sort(key=lambda x: x[0])

        assigned_t, assigned_c = set(), set()
        for _, tid, j in pair:
            if tid in assigned_t or j in assigned_c: continue
            tracks[tid].pose = detected_poses[j]
            tracks[tid].last_seen = now
            if tracks[tid].hold_start is None: tracks[tid].hold_start = now
            assigned_t.add(tid);
            assigned_c.add(j)

            tr = tracks[tid]

            if not tr.triggered and \
                    (now - tr.hold_start >= args.face_hold_sec) and \
                    (now - last_global_trigger >= args.trigger_cooldown):

                tr.triggered = True
                last_global_trigger = now

                cats_on_screen = any(t.cat is not None for t in tracks.values())

                if ripple_frames and not cats_on_screen:
                    running_ripples.append(ActiveRipple(0, tr.id))
                elif cats_loaded:
                    vec = (tr.pose.shoulder_center[0] - tr.pose.nose[0], tr.pose.shoulder_center[1] - tr.pose.nose[1])
                    px, py = get_ray_circle_intersection(center, radius, tr.pose.nose, vec)
                    # [修改] 隨機選擇一組動畫序列
                    selected_seq = random.choice(anim_sequences)
                    tr.cat = CatOverlay(frames=selected_seq, start=now, duration=args.cat_fade,
                                        cx=float(px), cy=float(py), anim_fps=args.anim_fps)

        for j, p in enumerate(detected_poses):
            if j not in assigned_c:
                tracks[next_track_id] = Track(next_track_id, p, now, now)
                next_track_id += 1

        tracks = {tid: tr for tid, tr in tracks.items() if now - tr.last_seen <= args.miss_timeout}

        # 繪製水波與中途淡入
        finished = []
        ripple_total_frames = len(ripple_frames)
        for rp in running_ripples:
            if rp.frame_idx < ripple_total_frames:
                cv2.add(canvas, ripple_frames[rp.frame_idx], dst=canvas)

                progress = rp.frame_idx / ripple_total_frames
                if progress >= args.cat_appear_phase and not rp.cat_spawned:
                    target = tracks.get(rp.owner_track_id)
                    if not target:
                        cands = [t for t in tracks.values() if t.cat is None]
                        if cands: target = cands[0]; target.triggered = True

                    if target and cats_loaded:
                        vec = (target.pose.shoulder_center[0] - target.pose.nose[0],
                               target.pose.shoulder_center[1] - target.pose.nose[1])
                        px, py = get_ray_circle_intersection(center, radius, target.pose.nose, vec)
                        # [修改] 隨機選擇一組動畫序列
                        selected_seq = random.choice(anim_sequences)
                        target.cat = CatOverlay(frames=selected_seq, start=now, duration=args.cat_fade,
                                                cx=float(px), cy=float(py), anim_fps=args.anim_fps)
                        rp.cat_spawned = True

                rp.frame_idx += 1
            else:
                finished.append(rp)

        for rp in finished:
            running_ripples.remove(rp)
            if not rp.cat_spawned:
                target = tracks.get(rp.owner_track_id)
                if not target:
                    cands = [t for t in tracks.values() if t.cat is None]
                    if cands: target = cands[0]; target.triggered = True
                if target and cats_loaded:
                    vec = (target.pose.shoulder_center[0] - target.pose.nose[0],
                           target.pose.shoulder_center[1] - target.pose.nose[1])
                    px, py = get_ray_circle_intersection(center, radius, target.pose.nose, vec)
                    selected_seq = random.choice(anim_sequences)
                    target.cat = CatOverlay(frames=selected_seq, start=now, duration=args.cat_fade,
                                            cx=float(px), cy=float(py), anim_fps=args.anim_fps)

        if args.draw_skeleton: draw_skeletons(canvas, kxy, kcf, args.line_width)

        # 貓咪繪製
        for tr in tracks.values():
            if not tr.cat: continue

            vec_x = tr.pose.shoulder_center[0] - tr.pose.nose[0]
            vec_y = tr.pose.shoulder_center[1] - tr.pose.nose[1]

            target_x, target_y = get_ray_circle_intersection(center, radius, tr.pose.nose, (vec_x, vec_y))

            dir_x = center[0] - target_x
            dir_y = center[1] - target_y
            norm = (dir_x ** 2 + dir_y ** 2) ** 0.5
            if norm > 0.001:
                ux, uy = dir_x / norm, dir_y / norm
                target_x += ux * args.cat_offset_radial
                target_y += uy * args.cat_offset_radial

            alpha = args.follow_smooth
            tr.cat.cx = tr.cat.cx * (1 - alpha) + target_x * alpha
            tr.cat.cy = tr.cat.cy * (1 - alpha) + target_y * alpha

            body_angle = math.degrees(math.atan2(vec_y, vec_x))
            tr.cat.rot_deg = 90 - body_angle + args.rot_offset

            # [修改] 取得當前動畫幀
            current_img = tr.cat.get_current_frame(now)

            rotated_cat = rotate_with_alpha(current_img, tr.cat.rot_deg)
            overlay_bgra_center(canvas, rotated_cat, int(tr.cat.cx), int(tr.cat.cy), tr.cat.alpha(now))

            if args.debug_angle:
                text = f"Ang:{tr.cat.rot_deg:.1f}"
                cv2.putText(canvas, text, (int(tr.cat.cx), int(tr.cat.cy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        mask = create_circular_mask(h, w, center, radius)
        masked = cv2.bitwise_and(canvas, canvas, mask=mask)
        cv2.circle(masked, center, radius, (255, 255, 255), 3)

        dt = now - prev_t;
        prev_t = now
        fps = 0.9 * fps + 0.1 * (1 / dt) if dt > 0 else 0
        cv2.putText(masked, f"FPS:{fps:.1f} Users:{len(tracks)}", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)

        cv2.imshow(win_name, masked)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cam.release();
    bgcap.release();
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()