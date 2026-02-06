import cv2
import json
import os
import asyncio
import threading
import websockets
from ultralytics import YOLO
from datetime import datetime

# --- 設定參數 (Configuration) ---
CONFIG = {
    "source": "rec_20251117_172908.mp4",  # 輸入來源: 0 代表 Webcam, 或者輸入影片路徑如 "video.mp4"
    "mirror": True,  # 是否啟用水平翻轉 (僅對 Webcam 有效)
    "width": 1920,  # Webcam 寬度
    "height": 1080,  # Webcam 高度
    "draw_skeleton": True,  # 是否在畫面上繪製骨架
    "save_json": False,  # 是否儲存 JSON 檔案
    "output_dir": "output",  # JSON 輸出資料夾
    "ws_host": "localhost",  # WebSocket Host
    "ws_port": 8765,  # WebSocket Port
    "model_path": "yolo11n-pose.pt"  # YOLOv11 pose model (n, s, m, l, x)
}


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

        # 【修正重點】定義一個 async task 來啟動 Server
        async def run_server():
            # 使用 async with 語法，這是新版 websockets 的標準寫法
            async with websockets.serve(self.handler, self.host, self.port):
                print(f"[WS] Server started at ws://{self.host}:{self.port}")
                # 使用 Future 讓這個 Coroutine 永遠暫停，保持 Server 運行
                await asyncio.Future()

        # 透過 run_until_complete 啟動上述的 async task
        # 這樣 websockets.serve 執行時，Event Loop 已經是 running 狀態
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
        """將訊息廣播給所有連線的 Client"""
        # 檢查 Loop 是否還在運行
        if self.clients and self.loop.is_running():
            coroutine = self._send_all(message)
            asyncio.run_coroutine_threadsafe(coroutine, self.loop)

    async def _send_all(self, message):
        if self.clients:
            # 廣播訊息，並忽略連線已斷開的錯誤
            await asyncio.gather(*[client.send(message) for client in self.clients], return_exceptions=True)


# --- 主程式邏輯 ---
def main():
    # 1. 初始化環境
    if CONFIG["save_json"]:
        os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # 2. 啟動 WebSocket Server
    ws_server = WebSocketServer(CONFIG["ws_host"], CONFIG["ws_port"])

    # 3. 載入 YOLOv11 模型
    print(f"[AI] Loading YOLOv11 model: {CONFIG['model_path']}...")
    model = YOLO(CONFIG["model_path"])

    # 4. 開啟影像來源
    cap = cv2.VideoCapture(CONFIG["source"])

    if CONFIG["source"] == 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["height"])

    frame_count = 0

    print("[System] Starting loop. Press 'q' to exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[System] End of stream.")
                break

            # 鏡像翻轉 (如果是 Webcam)
            if CONFIG["source"] == 0 and CONFIG["mirror"]:
                frame = cv2.flip(frame, 1)

            # --- AI 推論 ---
            results = model.predict(frame, verbose=False, conf=0.5)
            result = results[0]

            # --- 資料解析與格式化 ---
            # 改用 .xy 獲取絕對像素座標 (原本是 .xyn)
            keypoints_data = result.keypoints.xy.cpu().numpy()  # Shape: (Num_People, 17, 2)
            confs_data = result.keypoints.conf.cpu().numpy()  # Shape: (Num_People, 17)

            frame_json_data = []

            if result.keypoints.has_visible:
                for i, person_kpts in enumerate(keypoints_data):
                    person_formatted = []

                    for j, (x, y) in enumerate(person_kpts):
                        # 取得對應的 confidence
                        conf = confs_data[i][j] if confs_data is not None else 0.0

                        # Confidence 保留小數點後 3 位
                        person_formatted.append([float(x), float(y), round(float(conf), 3)])

                    frame_json_data.append(person_formatted)

            # 建構最終 JSON 物件
            json_output = {
                str(frame_count): frame_json_data
            }
            json_str = json.dumps(json_output)

            # --- 動作 1: 存檔 ---
            if CONFIG["save_json"]:
                filename = f"{frame_count:04d}.json"
                filepath = os.path.join(CONFIG["output_dir"], filename)
                with open(filepath, 'w') as f:
                    f.write(json_str)

            # --- 動作 2: WebSocket 傳輸 ---
            ws_server.broadcast(json_str)

            # --- 動作 3: 視覺化與顯示 ---
            display_frame = frame
            if CONFIG["draw_skeleton"]:
                display_frame = result.plot()

            cv2.imshow("YOLOv11 Pose Estimation", display_frame)

            # 按 'q' 離開
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