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
    "keypoint_threshold": 0.9
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


# --- 輔助函式: 通用向量角度計算 ---
def calculate_vector_angle(start_pt, end_pt):
    """
    計算從 start_pt 指向 end_pt 的向量角度 (順時針)。
    影像座標系: X向右, Y向下
    atan2回傳: 右(0), 下(90), 左(180), 上(-90)

    目標定義:
    - 12點鐘 (上) = 0度
    - 3點鐘 (右) = 90度
    - 6點鐘 (下) = 180度
    - 9點鐘 (左) = 270度

    修正邏輯: Target = atan2_degree + 90
    """
    vec_x = end_pt[0] - start_pt[0]
    vec_y = end_pt[1] - start_pt[1]

    # 1. 計算標準 atan2 角度 (以向右 X 軸為 0)
    angle_rad = math.atan2(vec_y, vec_x)
    angle_deg = math.degrees(angle_rad)

    # 2. 轉換為時鐘座標 (以向上 -Y 軸為 0)
    # 下(90) + 90 = 180
    # 上(-90) + 90 = 0
    final_angle = angle_deg + 90

    # 3. 確保在 0-360 範圍內
    return int(final_angle % 360)


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
                    person_formatted = []
                    person_draw_data = []

                    person_confs = confs_data[i] if confs_data is not None else np.zeros(17)

                    # 1. 處理骨架資料
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

                    # --- 【修改】 判斷關鍵部位是否存在 (COCO Indices) ---
                    # 0:Nose, 1:LEye, 2:REye, 3:LEar, 4:REar, 5:LShoulder, 6:RShoulder
                    has_nose = person_confs[0] >= kp_threshold
                    has_le = person_confs[1] >= kp_threshold
                    has_re = person_confs[2] >= kp_threshold
                    has_lear = person_confs[3] >= kp_threshold
                    has_rear = person_confs[4] >= kp_threshold
                    has_ls = person_confs[5] >= kp_threshold
                    has_rs = person_confs[6] >= kp_threshold
                    has_shoulder = has_ls or has_rs

                    # --- 【修改】 計算比例 (skeleton_percentage) ---
                    # 定義符合 50% 的 11 種規則組合 (不含肩膀)
                    # 規則 2~11 (因為規則1是0%)
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

                    # 只要滿足上述任一條件
                    is_50_condition = (
                            cond_2 or cond_3 or cond_4 or cond_5 or cond_6 or
                            cond_7 or cond_8 or cond_9 or cond_10 or cond_11
                    )

                    pct = 0

                    if has_shoulder:
                        # 規則: 在1~11規則中，只要有額外偵測到左肩或右肩，skeleton_percentage = 100
                        # 規則 1 是 "僅偵測到鼻子" (has_nose)
                        # 規則 2~11 是 is_50_condition
                        if has_nose or is_50_condition:
                            pct = 100
                    elif is_50_condition:
                        # 滿足規則 2~11 且沒有肩膀 -> 50
                        pct = 50
                    elif has_nose:
                        # 滿足規則 1 (僅鼻子) 且沒有肩膀 -> 0
                        pct = 0
                    else:
                        # 都不滿足
                        pct = 0

                    percentage_list.append(pct)

                    # --- 計算角度 (Body Angle) ---
                    current_angle = 0

                    # 優先: 鼻子 -> 肩膀中心 (向量向下，應為180度)
                    if has_nose and (has_ls and has_rs):
                        nose_pt = person_kpts[0]
                        l_shoulder_pt = person_kpts[5]
                        r_shoulder_pt = person_kpts[6]
                        shoulder_center = (l_shoulder_pt + r_shoulder_pt) / 2
                        current_angle = calculate_vector_angle(start_pt=nose_pt, end_pt=shoulder_center)

                    # 次要: 雙眼中心 -> 鼻子 (向量向下，應為180度)
                    elif has_nose and (has_le and has_re):
                        nose_pt = person_kpts[0]
                        l_eye_pt = person_kpts[1]
                        r_eye_pt = person_kpts[2]
                        eye_center = (l_eye_pt + r_eye_pt) / 2
                        current_angle = calculate_vector_angle(start_pt=eye_center, end_pt=nose_pt)

                    angle_list.append(current_angle)

            # --- 排序邏輯: 依照 Angle 由大到小排序 ---
            if angle_list:
                combined = list(zip(angle_list, skeletons_list, percentage_list, people_to_draw))
                combined.sort(key=lambda x: x[0], reverse=True)
                angle_list, skeletons_list, percentage_list, people_to_draw = map(list, zip(*combined))

            # --- 建構最終 JSON ---
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