
from ultralytics import YOLO

# Load trained model
model = YOLO(
    "runs/detect/outputs/traffic_violation_v1/weights/best.pt"
)

# Input video
video_path = "test_videos/indian_traffic1.mp4"

# Detect and display video
results = model.predict(
    source=video_path,
    conf=0.25,
    imgsz=640,
    device="cpu",
    show=True,       # Display video while processing
    save=True,       # Save processed video
    verbose=False,   # Hide frame-by-frame terminal messages
    name="traffic_video_test"
)

print("Traffic video detection completed!")
