from ultralytics import YOLO

# Load base model
model = YOLO("models/basev8n.pt")

# Train the dataset 30 epoches
model.train(data="data.yaml", epochs=30, imgsz=640, device=[0])