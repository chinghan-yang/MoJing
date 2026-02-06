import cv2
import json
import os
import asyncio
import threading
import websockets
import numpy as np
from ultralytics import YOLO

# --- 設定參數 (Configuration) ---
CONFIG = {
    "source": "rec_20251117_172908.mp4",  # 輸入來源
    "mirror": True,  # 鏡像翻轉
    "width": 1920,  # Webcam 寬度
    "height": 1080,  # Webcam 高度
    "draw_skeleton": True,  # 是否繪製骨架
    "save_json": True,  # 是否存檔
    "output_dir": "output",
    "ws_host": "localhost",
    "ws_port": 8765,
    "model_path": "yolo11n-pose.pt",

    # 【新增】 骨架節點信心門檻
    "keypoint_threshold": 0.9
}

# --- 骨架連線定義 (COCO 格式: [起點, 終點, 顏色RGB]) ---
# 顏色格式為 BGR (OpenCV)
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


# --- WebSocket 伺服器類別 (維持不變) ---
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


# --- 主程式邏輯 ---
def main():
    if CONFIG["save_json"]:
        os.makedirs(CONFIG["output_dir"], exist_ok=True)

    ws_server = WebSocketServer(CONFIG["ws_host"], CONFIG["ws_port"])

    print(f"[AI] Loading YOLOv11 model: {CONFIG['model_path']}...")
    model = YOLO(CONFIG["model_path"])

    # 嘗試開啟來源
    source = CONFIG["source"]
    # 如果是數字字串 (e.g. "0") 轉為 int，否則視為檔案路徑
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    # 根據來源決定是否加入 CAP_DSHOW (Windows Webcam 修正)
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
    threshold = CONFIG["keypoint_threshold"]

    print("[System] Starting loop. Press 'q' to exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[System] End of stream.")
                break

            if source == 0 and CONFIG["mirror"]:
                frame = cv2.flip(frame, 1)

            # --- AI 推論 ---
            results = model.predict(frame, verbose=False, conf=0.5)
            result = results[0]

            # --- 資料解析與格式化 ---
            # 取得絕對座標 (Pixel Coordinates)
            keypoints_xy = result.keypoints.xy.cpu().numpy()  # (N, 17, 2)
            confs_data = result.keypoints.conf.cpu().numpy()  # (N, 17)

            frame_json_data = []

            # 用於繪圖的暫存列表 (只包含該 frame 的所有有效人體數據)
            people_to_draw = []

            if result.keypoints.has_visible:
                for i, person_kpts in enumerate(keypoints_xy):
                    person_formatted = []  # JSON 用
                    person_draw_data = []  # 繪圖用 (儲存 x, y, valid_flag)

                    for j, (x, y) in enumerate(person_kpts):
                        conf = confs_data[i][j] if confs_data is not None else 0.0

                        # 【邏輯判斷】 Confidence Threshold Check
                        if conf >= threshold:
                            # 數值有效：保留 conf，標記為可繪製
                            final_conf = round(float(conf), 3)
                            is_valid = True
                        else:
                            # 數值無效：conf 設為 -1，標記為不可繪製
                            final_conf = -1.0
                            is_valid = False

                        # 存入 JSON 結構
                        person_formatted.append([float(x), float(y), final_conf])

                        # 存入繪圖結構 (x, y, is_valid)
                        person_draw_data.append((int(x), int(y), is_valid))

                    frame_json_data.append(person_formatted)
                    people_to_draw.append(person_draw_data)

            # --- 建立與傳送 JSON ---
            json_output = {str(frame_count): frame_json_data}
            json_str = json.dumps(json_output, indent=1)

            if CONFIG["save_json"]:
                filename = f"{frame_count:04d}.json"
                with open(os.path.join(CONFIG["output_dir"], filename), 'w', newline='\n') as f:
                    f.write(json_str)

            ws_server.broadcast(json_str)

            # --- 動作 3: 視覺化 (手動繪製) ---
            # 我們不再使用 result.plot()，因為它無法依據單點 threshold 隱藏
            display_frame = frame.copy()  # 複製一份以免影響原始 frame

            if CONFIG["draw_skeleton"]:
                for person_pts in people_to_draw:
                    # person_pts 是一個 list: [(x, y, valid), (x, y, valid), ...]

                    # 1. 繪製骨架連線 (Limbs)
                    for idx_a, idx_b, color in SKELETON_CONNECTIONS:
                        pt_a = person_pts[idx_a]
                        pt_b = person_pts[idx_b]

                        # 只有當連線的「兩端點」都有效 (valid=True) 時才畫線
                        if pt_a[2] and pt_b[2]:
                            cv2.line(display_frame, (pt_a[0], pt_a[1]), (pt_b[0], pt_b[1]), color, 2)

                    # 2. 繪製關鍵點 (Joints)
                    for x, y, is_valid in person_pts:
                        if is_valid:
                            # 紅色圓點，半徑 4
                            cv2.circle(display_frame, (x, y), 4, (0, 0, 255), -1)

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