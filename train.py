
from pathlib import Path
from ultralytics import YOLO

# ---------------------------------------
# 1. PROJECT CONFIGURATION
# ---------------------------------------

DATA_YAML = "Traffic_violation/data.yaml"
MODEL_WEIGHTS = "yolov8n.pt"

OUTPUT_FOLDER = "outputs"
RUN_NAME = "traffic_violation_v1"

# ---------------------------------------
# 2. VERIFY CONFIGURATION
# ---------------------------------------

if not Path(DATA_YAML).exists():
    raise FileNotFoundError(
        f"Dataset configuration not found: {DATA_YAML}"
    )

print("=" * 50)
print("TRAFFIC VIOLATION DETECTION")
print("YOLOv8 TRAINING")
print("=" * 50)

# ---------------------------------------
# 3. LOAD PRETRAINED MODEL
# ---------------------------------------

print("\nLoading pretrained YOLOv8 Nano model...")

model = YOLO(MODEL_WEIGHTS)

print("Pretrained model loaded successfully.")

# ---------------------------------------
# 4. START TRAINING
# ---------------------------------------

print("\nStarting training...")

results = model.train(

    data=DATA_YAML,

    epochs=100,
    imgsz=640,
    batch=4,

    device="cpu",
    workers=2,

    patience=20,

    optimizer="auto",
    cos_lr=True,

    close_mosaic=10,

    cache=False,
    amp=False,

    seed=42,
    deterministic=True,

    project=OUTPUT_FOLDER,
    name=RUN_NAME,

    exist_ok=False,

    save=True,
    save_period=10,

    plots=True,
    val=True,

    verbose=True

)

# ---------------------------------------
# 5. TRAINING COMPLETED
# ---------------------------------------

print("\n" + "=" * 50)
print("TRAINING FINISHED")
print("=" * 50)

print(
    "Best weights:",
    f"{OUTPUT_FOLDER}/{RUN_NAME}/weights/best.pt"
)

print(
    "Last weights:",
    f"{OUTPUT_FOLDER}/{RUN_NAME}/weights/last.pt"
)

print("Training results:", results)
