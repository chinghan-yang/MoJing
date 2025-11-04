#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera preview at 1920x1080 with video recording toggle.
Python 3.11 + OpenCV

Hotkeys:
  r : start/stop recording (MP4)
  q : quit
"""

from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
import time

import cv2
import numpy as np


DISPLAY_W, DISPLAY_H = 1920, 1080  # 視窗顯示解析度（固定）
DEFAULT_FPS_FALLBACK = 30          # 取不到 FPS 時的預設
RECORD_DIR = Path("recordings")    # 錄影輸出資料夾


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OpenCV live preview (1920x1080) with recording")
    p.add_argument("--cam", "--camera-index", type=int, default=2, dest="cam_index",
                   help="Camera index (default: 2, same as original file)")
    p.add_argument("--width", type=int, default=1920, help="Try to set camera capture width (default: 1920)")
    p.add_argument("--height", type=int, default=1080, help="Try to set camera capture height (default: 1080)")
    p.add_argument("--fps", type=float, default=0.0,
                   help="Try to set camera FPS (default: 0 = use camera default)")
    p.add_argument("--out", type=Path, default=RECORD_DIR, help="Output directory for recordings")
    p.add_argument("--codec", type=str, default="mp4v",
                   help="FourCC for MP4 (default: mp4v). Common options: mp4v, avc1, H264")
    p.add_argument("--record-scale", choices=["capture", "display"], default="display",
                   help="Record at camera capture size, or the 1920x1080 display size (default: display)")
    return p


def create_video_writer(path: Path, fourcc_str: str, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    vw = cv2.VideoWriter(str(path), fourcc, fps, frame_size)
    if not vw.isOpened():
        raise RuntimeError(f"VideoWriter無法開啟：{path}（fourcc={fourcc_str}, fps={fps}, size={frame_size}）")
    return vw


def put_rec_overlay(frame: np.ndarray, rec_secs: float) -> np.ndarray:
    # 加上 REC 圖示與時間
    out = frame.copy()
    rec_text = f"REC  {int(rec_secs // 60):02d}:{int(rec_secs % 60):02d}"
    # 紅色圓點 + 文字（BGR）
    cv2.circle(out, (30, 40), 10, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
    cv2.putText(out, rec_text, (55, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
    return out


def main() -> None:
    args = build_argparser().parse_args()

    # 啟動攝影機
    cap = cv2.VideoCapture(args.cam_index)
    if not cap.isOpened():
        raise SystemExit(f"無法開啟攝影機（index={args.cam_index}）")

    # 嘗試設定攝影機原生解析度與 FPS（若硬體不支援，OpenCV 會忽略）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.fps > 0:
        cap.set(cv2.CAP_PROP_FPS, args.fps)

    # 讀取實際攝影機參數
    cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
    cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap_fps or cap_fps <= 1:
        cap_fps = DEFAULT_FPS_FALLBACK

    print(f"[Camera] size={cap_w}x{cap_h}, fps={cap_fps:.2f}")

    # 建立視窗並固定顯示解析度為 1920x1080
    window_name = "live"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, DISPLAY_W, DISPLAY_H)

    # 錄影狀態
    recording = False
    vw: cv2.VideoWriter | None = None
    rec_start_ts = 0.0
    args.out.mkdir(parents=True, exist_ok=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("讀取影格失敗（串流結束？），程式結束。")
                break

            # 顯示影像縮放至 1920x1080
            display_frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_AREA)

            # 處理錄影
            if recording:
                rec_secs = time.time() - rec_start_ts
                display_frame = put_rec_overlay(display_frame, rec_secs)

                # 寫入錄影：依設定寫入顯示尺寸或攝影機原生尺寸
                if args.record_scale == "display":
                    out_frame = display_frame
                    out_size = (DISPLAY_W, DISPLAY_H)
                else:
                    out_frame = frame
                    out_size = (cap_w, cap_h)

                if vw is None:
                    # 延遲到第一次要寫入時才建立 VideoWriter，避免尺寸不一致
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"rec_{timestamp}.mp4"
                    vw = create_video_writer(args.out / filename, args.codec, cap_fps, out_size)
                    print(f"[REC] 開始錄影 → {args.out / filename}")

                # 若 VideoWriter 與 out_size 不一致（理論上不會），則重建
                # 這裡保守檢查一次 frame size
                # （OpenCV 無法更改已開啟 VideoWriter 的尺寸）
                if vw is not None:
                    vw.write(out_frame)

            # 顯示
            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("收到 q，結束程式。")
                break
            elif key == ord('r'):
                # 切換錄影狀態
                if not recording:
                    recording = True
                    rec_start_ts = time.time()
                    vw = None  # 讓下一迴圈初始化 writer（取正確尺寸）
                else:
                    recording = False
                    if vw is not None:
                        vw.release()
                        vw = None
                    print("[REC] 錄影已停止。")

    finally:
        # 收尾
        if vw is not None:
            vw.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
