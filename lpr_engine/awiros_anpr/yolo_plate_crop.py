
from pathlib import Path
import cv2
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parent.parent

MODEL_PATH = PROJECT / "runs/detect/outputs/traffic_violation_v1/weights/best.pt"
IMAGE_DIR = PROJECT / "test_images"

OUTPUT_DIR = BASE / "plate_test_results"
OUTPUT_DIR.mkdir(exist_ok=True)

CROP_DIR = OUTPUT_DIR / "plate_crops"
CROP_DIR.mkdir(exist_ok=True)

model = YOLO(str(MODEL_PATH))

image_files = [
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
]

print("Total test images:", len(image_files))

total_plates = 0

for image_path in image_files:

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    print("\nTesting:", image_path.name)

    results = model.predict(
        source=image,
        classes=[1],
        conf=0.05,
        imgsz=1280,
        device="cpu",
        verbose=False
    )

    annotated = image.copy()
    count = 0

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            x1, y1, x2, y2 = map(
                int, box.xyxy[0].cpu().numpy()
            )

            confidence = float(box.conf[0])

            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            count += 1
            total_plates += 1

            crop_name = f"{image_path.stem}_plate_{count}.jpg"
            cv2.imwrite(str(CROP_DIR / crop_name), crop)

            cv2.rectangle(
                annotated, (x1, y1), (x2, y2),
                (0, 255, 0), 2
            )

            print(
                f"  Plate {count}: confidence {confidence:.3f}"
            )

    cv2.imwrite(
        str(OUTPUT_DIR / f"{image_path.stem}_result.jpg"),
        annotated
    )

    print("  Number plates detected:", count)

print("\n================================")
print("TOTAL NUMBER PLATES:", total_plates)
print("Cropped images:", CROP_DIR)
print("Annotated images:", OUTPUT_DIR)
