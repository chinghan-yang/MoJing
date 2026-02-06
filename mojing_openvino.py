import cv2
import json
import os
import asyncio
import threading
import websockets
import numpy as np
import math
import time  # 【新增】 用於計算 FPS
from ultralytics import YOLO
from datetime import datetime  # 用於生成錄影檔名

# --- 設定參數 (Configuration) ---
CONFIG = {
    "source": 0,  # 輸入來源: 0 代表 Webcam, 或者輸入影片路徑如 "video.mp4"
    "mirror": True,  # 鏡像翻轉 (僅對 Webcam 有效)
    "width": 1920,  # Webcam 寬度
    "height": 1080,  # Webcam 高度
    "draw_skeleton": True,  # 是否繪製骨架
    "save_json": False,  # 是否存檔
    "output_dir": "output",
    "ws_host": "localhost",
    "ws_port": 8765,
    "model_path": "yolo11m-pose_openvino_model/",

    # 骨架節點信心門檻
    "keypoint_threshold": 0.6,

    # 計算角度用的圓心座標 (預設為畫面中心，可自行調整)
    "center_x": 985,
    "center_y": 155,

    # 遮罩檔案路徑
    "mask_path": "well_mask_20251226.npy"
}

# --- 骨架連線定義 (COCO 格式: [起點, 終點, 顏色RGB]) ---
SKELETON_CONNECTIONS = [
    (5, 7, (0, 255, 255)), (7, 9, (0, 255, 255)),  # 左手 (黃)
    (6, 8, (0, 255, 0)), (8, 10, (0, 255, 0)),  # 右手 (綠)
    (11, 13, (0, 255, 255)), (13, 15, (0, 255, 255)),  # 左腿 (黃)
    (12, 14, (0, 255, 0)), (14, 16, (0, 255, 0)),  # 右腿 (綠)
    (5, 6, (255, 128, 0)), (11, 12, (255, 128, 0)),  # 肩膀與臀部 (藍)
    (5, 11, (255, 128, 0)), (6, 12, (255, 128, 0)),  # 軀幹 (藍)
    (0, 1, (255, 0, 255)), (0, 2, (255, 0, 255)),  # 臉部 (紫)
    (1, 3, (255, 0, 255)), (2, 4, (255, 0, 255))
]


# --- WebSocket 伺服器類別 ---
class WebSocketServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.clients = set()
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._start_loop, daemon=True)
        self.thread.start()

    def _start_loop(self):
        asyncio.set_event_loop(self.loop)

        async def run_server():
            async with websockets.serve(self.handler, self.host, self.port):
                print(f"[WS] Server started at ws://{self.host}:{self.port}")
                await asyncio.Future()

        try:
            self.loop.run_until_complete(run_server())
        except RuntimeError as e:
            print(f"[WS Error] Loop failed: {e}")

    async def handler(self, websocket):
        self.clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)

    def broadcast(self, message):
        if self.clients and self.loop.is_running():
            coroutine = self._send_all(message)
            asyncio.run_coroutine_threadsafe(coroutine, self.loop)

    async def _send_all(self, message):
        if self.clients:
            await asyncio.gather(*[client.send(message) for client in self.clients], return_exceptions=True)


# --- 輔助函式: 計算相對於圓心的方位角 ---
def calculate_position_angle(center_pt, target_pt, is_mirror=False):
    dx = target_pt[0] - center_pt[0]
    dy = target_pt[1] - center_pt[1]
    angle_deg = math.degrees(math.atan2(dy, dx))

    if is_mirror:
        final_angle = angle_deg + 90
    else:
        final_angle = -(angle_deg + 90)

    return int(final_angle % 360)


# --- 主程式邏輯 ---
def main():
    # 確保輸出資料夾存在
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    ws_server = WebSocketServer(CONFIG["ws_host"], CONFIG["ws_port"])

    print(f"[AI] Loading YOLOv11 model: {CONFIG['model_path']}...")
    model = YOLO(CONFIG["model_path"])

    # --- 載入遮罩檔案 ---
    well_mask = None
    if os.path.exists(CONFIG["mask_path"]):
        try:
            well_mask = np.load(CONFIG["mask_path"])
            print(f"[System] Loaded mask file: {CONFIG['mask_path']}, shape: {well_mask.shape}")
        except Exception as e:
            print(f"[Error] Failed to load mask file: {e}")
    else:
        print(f"[Warning] Mask file not found: {CONFIG['mask_path']}")

    source = CONFIG["source"]
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    if source == 0 or (isinstance(source, int)):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["height"])
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[Error] Cannot open source: {source}")
        return

    frame_count = 0
    kp_threshold = CONFIG["keypoint_threshold"]

    # 取得設定的圓心
    center_point = (CONFIG["center_x"], CONFIG["center_y"])

    # 判斷是否開啟了鏡像模式 (Webcam 且 Mirror=True)
    is_mirror_mode = (source == 0 and CONFIG["mirror"])

    # --- 錄影功能變數 ---
    is_recording = False
    video_writer = None

    # --- 【新增】 FPS 計算變數 ---
    prev_time = time.time()
    current_fps = 0.0

    print(f"[System] Mirror Mode: {is_mirror_mode}")
    print("[System] Starting loop. Press 'q' to exit, 'r' to toggle recording.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[System] End of stream.")
                break

            if is_mirror_mode:
                frame = cv2.flip(frame, 1)

            img_h, img_w = frame.shape[:2]

            # --- AI 推論 ---
            results = model.predict(frame, verbose=False, conf=0.25)
            result = results[0]

            keypoints_xy = result.keypoints.xy.cpu().numpy()
            confs_data = result.keypoints.conf.cpu().numpy()

            skeletons_list = []
            percentage_list = []
            angle_list = []
            people_to_draw = []

            if result.keypoints.has_visible:
                for i, person_kpts in enumerate(keypoints_xy):
                    # 取得原始信心值
                    raw_confs = confs_data[i] if confs_data is not None else np.zeros(17)

                    # --- 應用遮罩檢查 ---
                    masked_confs = raw_confs.copy()

                    if well_mask is not None:
                        for k_idx, (kx, ky) in enumerate(person_kpts):
                            ix, iy = int(kx), int(ky)
                            if 0 <= iy < well_mask.shape[0] and 0 <= ix < well_mask.shape[1]:
                                if well_mask[iy, ix] == 1:
                                    masked_confs[k_idx] = -1.0

                    person_confs = masked_confs

                    # --- 步驟 1: 擷取各部位信心狀態 (含肩膀) ---
                    has_nose = person_confs[0] >= kp_threshold
                    has_le = person_confs[1] >= kp_threshold
                    has_re = person_confs[2] >= kp_threshold
                    has_lear = person_confs[3] >= kp_threshold
                    has_rear = person_confs[4] >= kp_threshold
                    has_ls = person_confs[5] >= kp_threshold
                    has_rs = person_confs[6] >= kp_threshold

                    has_any_head = has_nose or has_le or has_re or has_lear or has_rear
                    if not has_any_head:
                        continue

                    target_point = None

                    # --- 步驟 2: 決定角度計算的基準點 (優先順序) ---
                    if has_ls and has_rs:
                        target_point = (person_kpts[5] + person_kpts[6]) / 2
                    elif has_nose:
                        target_point = person_kpts[0]
                    elif has_le and has_re:
                        target_point = (person_kpts[1] + person_kpts[2]) / 2
                    elif has_le:
                        target_point = person_kpts[1]
                    elif has_re:
                        target_point = person_kpts[2]
                    elif has_lear and has_rear:
                        target_point = (person_kpts[3] + person_kpts[4]) / 2
                    elif has_lear:
                        target_point = person_kpts[3]
                    elif has_rear:
                        target_point = person_kpts[4]

                    if target_point is None:
                        continue

                    # --- 步驟 3: 計算角度 ---
                    current_angle = calculate_position_angle(center_point, target_point, is_mirror=is_mirror_mode)

                    # --- 步驟 4: 處理骨架資料 ---
                    person_formatted = []
                    person_draw_data = []

                    for j, (x, y) in enumerate(person_kpts):
                        conf = person_confs[j]

                        if conf >= kp_threshold:
                            final_conf = round(float(conf), 3)
                            is_valid = True
                        else:
                            final_conf = -1.0
                            is_valid = False

                        person_formatted.append([float(x), float(y), final_conf])
                        person_draw_data.append((int(x), int(y), is_valid))

                    # --- 步驟 5: 計算比例 ---
                    has_shoulder = has_ls or has_rs
                    cond_2 = has_nose and has_le
                    cond_3 = has_nose and has_re
                    cond_4 = has_nose and has_le and has_re
                    cond_5 = has_nose and has_lear
                    cond_6 = has_nose and has_rear
                    cond_7 = has_nose and has_lear and has_rear
                    cond_8 = has_lear and has_le
                    cond_9 = has_rear and has_re
                    cond_10 = has_nose and has_le and has_lear
                    cond_11 = has_nose and has_re and has_rear

                    is_50_condition = (
                            cond_2 or cond_3 or cond_4 or cond_5 or cond_6 or
                            cond_7 or cond_8 or cond_9 or cond_10 or cond_11
                    )

                    pct = 0
                    if has_shoulder:
                        if has_nose or is_50_condition:
                            pct = 100
                    elif is_50_condition:
                        pct = 50
                    elif has_nose:
                        pct = 0

                    skeletons_list.append(person_formatted)
                    people_to_draw.append(person_draw_data)
                    percentage_list.append(pct)
                    angle_list.append(current_angle)

            # --- 排序邏輯 ---
            if angle_list:
                combined = list(zip(angle_list, skeletons_list, percentage_list, people_to_draw))
                combined.sort(key=lambda x: x[0], reverse=True)
                angle_list, skeletons_list, percentage_list, people_to_draw = map(list, zip(*combined))

            # --- 建構 JSON ---
            json_output = {
                "frame_index": frame_count,
                "skeletons": skeletons_list,
                "skeleton_percentage": percentage_list,
                "angle": angle_list
            }

            json_str = json.dumps(json_output, indent=1)

            if CONFIG["save_json"]:
                filename = f"{frame_count:04d}.json"
                with open(os.path.join(CONFIG["output_dir"], filename), 'w', newline='\n') as f:
                    f.write(json_str)

            ws_server.broadcast(json_str)

            # --- 【新增】 計算 FPS ---
            curr_time = time.time()
            time_diff = curr_time - prev_time
            if time_diff > 0:
                fps = 1.0 / time_diff
                # 使用簡單的移動平均讓 FPS 顯示更平滑
                current_fps = (current_fps * 0.9) + (fps * 0.1)
            prev_time = curr_time

            # --- 視覺化 ---
            display_frame = frame.copy()
            cv2.drawMarker(display_frame, center_point, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20,
                           thickness=2)

            if CONFIG["draw_skeleton"]:
                for idx, person_pts in enumerate(people_to_draw):
                    for idx_a, idx_b, color in SKELETON_CONNECTIONS:
                        pt_a = person_pts[idx_a]
                        pt_b = person_pts[idx_b]
                        if pt_a[2] and pt_b[2]:
                            cv2.line(display_frame, (pt_a[0], pt_a[1]), (pt_b[0], pt_b[1]), color, 2)
                    for x, y, is_valid in person_pts:
                        if is_valid:
                            cv2.circle(display_frame, (x, y), 4, (0, 0, 255), -1)

                    if idx < len(percentage_list) and idx < len(angle_list):
                        info_text = f"{percentage_list[idx]}% | {angle_list[idx]}deg"
                        text_pos = person_pts[0][:2]
                        cv2.putText(display_frame, info_text, (text_pos[0] - 20, text_pos[1] - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # --- 【新增】 在畫面上繪製 FPS ---
            fps_text = f"FPS: {int(current_fps)}"
            # (30, 60) 是文字位置,字體大小 1.0, 綠色 (0, 255, 0), 粗細 2
            cv2.putText(display_frame, fps_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # --- 錄影功能 ---
            if is_recording and video_writer is not None:
                video_writer.write(display_frame)

            if is_recording:
                cv2.circle(display_frame, (30, 100), 10, (0, 0, 255), -1)
                cv2.putText(display_frame, "REC", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("YOLOv11 Pose Estimation", display_frame)

            # --- 鍵盤控制 ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # 切換錄影狀態
                if not is_recording:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = os.path.join(CONFIG["output_dir"], f"recording_{timestamp}.mp4")

                    vid_fps = cap.get(cv2.CAP_PROP_FPS)
                    if vid_fps == 0 or math.isnan(vid_fps):
                        vid_fps = 30.0

                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(save_path, fourcc, vid_fps, (img_w, img_h))

                    is_recording = True
                    print(f"[System] Recording started: {save_path}")
                else:
                    if video_writer:
                        video_writer.release()
                    video_writer = None
                    is_recording = False
                    print("[System] Recording stopped.")

            frame_count += 1

    except Exception as e:
        print(f"[Error] {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        print("[System] Resources released. Exiting.")


if __name__ == "__main__":
    main()