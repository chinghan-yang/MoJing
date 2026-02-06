import cv2
import json
import os
import asyncio
import threading
import websockets
import numpy as np
import math
from ultralytics import YOLO

# --- 設定參數 (Configuration) ---
CONFIG = {
    "source": "rec_20251117_172908.mp4",  # 輸入來源
    "mirror": True,  # 鏡像翻轉 (僅對 Webcam 有效)
    "width": 1920,  # Webcam 寬度
    "height": 1080,  # Webcam 高度
    "draw_skeleton": True,  # 是否繪製骨架
    "save_json": True,  # 是否存檔
    "output_dir": "output",
    "ws_host": "localhost",
    "ws_port": 8765,
    "model_path": "yolo11n-pose.pt",

    # 骨架節點信心門檻
    "keypoint_threshold": 0.5
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


# --- 輔助函式: 計算角度 ---
def calculate_azimuth_angle(cx, cy, img_w, img_h):
    img_cx = img_w / 2
    img_cy = img_h / 2
    dx = cx - img_cx
    dy = cy - img_cy
    theta = math.degrees(math.atan2(dy, dx))
    angle = (theta + 90) % 360
    return int(angle)


# --- 主程式邏輯 ---
def main():
    if CONFIG["save_json"]:
        os.makedirs(CONFIG["output_dir"], exist_ok=True)

    ws_server = WebSocketServer(CONFIG["ws_host"], CONFIG["ws_port"])

    print(f"[AI] Loading YOLOv11 model: {CONFIG['model_path']}...")
    model = YOLO(CONFIG["model_path"])

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

    print("[System] Starting loop. Press 'q' to exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[System] End of stream.")
                break

            if source == 0 and CONFIG["mirror"]:
                frame = cv2.flip(frame, 1)

            img_h, img_w = frame.shape[:2]

            # --- AI 推論 (conf=0.25) ---
            results = model.predict(frame, verbose=False, conf=0.25)
            result = results[0]

            keypoints_xy = result.keypoints.xy.cpu().numpy()
            confs_data = result.keypoints.conf.cpu().numpy()
            boxes_xywh = result.boxes.xywh.cpu().numpy() if result.boxes is not None else []

            skeletons_list = []
            percentage_list = []
            angle_list = []
            people_to_draw = []

            if result.keypoints.has_visible:
                for i, person_kpts in enumerate(keypoints_xy):
                    person_formatted = []
                    person_draw_data = []

                    person_confs = confs_data[i] if confs_data is not None else np.zeros(17)

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

                    skeletons_list.append(person_formatted)
                    people_to_draw.append(person_draw_data)

                    # 計算比例
                    has_nose = person_confs[0] >= kp_threshold
                    has_eyes = person_confs[1] >= kp_threshold and person_confs[2] >= kp_threshold
                    has_shoulders = person_confs[5] >= kp_threshold and person_confs[6] >= kp_threshold

                    pct = 0
                    if has_nose and has_eyes and has_shoulders:
                        pct = 100
                    elif has_nose and has_eyes:
                        pct = 50
                    percentage_list.append(pct)

                    # 計算角度
                    current_angle = 0
                    if len(boxes_xywh) > i:
                        cx, cy = boxes_xywh[i][:2]
                        current_angle = calculate_azimuth_angle(cx, cy, img_w, img_h)
                    angle_list.append(current_angle)

            # --- 建構最終 JSON (修改處) ---
            json_output = {
                "frame_index": frame_count,
                "skeletons": skeletons_list,
                "skeleton_percentage": percentage_list,
                "angle": angle_list
            }

            # 【修改】 加入 indent=1 增加可讀性
            json_str = json.dumps(json_output, indent=1)

            if CONFIG["save_json"]:
                filename = f"{frame_count:04d}.json"
                # 【修改】 加入 newline='\n'
                with open(os.path.join(CONFIG["output_dir"], filename), 'w', newline='\n') as f:
                    f.write(json_str)

            ws_server.broadcast(json_str)

            # --- 視覺化 ---
            display_frame = frame.copy()
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

            cv2.imshow("YOLOv11 Pose Estimation", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

    except Exception as e:
        print(f"[Error] {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[System] Resources released. Exiting.")


if __name__ == "__main__":
    main()