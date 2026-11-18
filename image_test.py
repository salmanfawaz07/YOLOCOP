
from ultralytics import YOLO

# Load our trained model
model = YOLO(
    "runs/detect/outputs/traffic_violation_v1/weights/best.pt"
)

# Detect violations in traffic photographs
results = model.predict(
    source="test_images",
    conf=0.25,
    imgsz=640,
    device="cpu",
    save=True
)

print("Traffic image detection completed!")
