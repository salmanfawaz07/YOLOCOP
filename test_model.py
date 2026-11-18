
from ultralytics import YOLO

# Load the best trained model
model = YOLO(
    "runs/detect/outputs/traffic_violation_v1/weights/best.pt"
)

# Evaluate on the separate TEST dataset
results = model.val(
    data="Traffic_violation/data.yaml",
    split="test",
    imgsz=640,
    batch=8,
    device="cpu",
    plots=True,
    name="traffic_violation_test"
)

print("\nTEST DATASET EVALUATION COMPLETED!")
