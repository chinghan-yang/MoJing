from ultralytics import YOLO

# Load a YOLO11n-pose PyTorch model
model = YOLO("sample/yolo11m-pose.pt")

# Export the model
model.export(format="openvino")  # creates 'yolo11n_openvino_model/'