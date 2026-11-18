
import sys
import copy
from pathlib import Path

import cv2
import numpy as np
import paddle
from ultralytics import YOLO

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PADDLEOCR_DIR = BASE_DIR / "PaddleOCR"

YOLO_MODEL_PATH = (
    BASE_DIR.parent.parent
    / "runs/detect/outputs/traffic_violation_v1/weights/best.pt"
)

IMAGE_PATH = (
    BASE_DIR.parent.parent
    / "test_images/b8f60a65-Deshpande_Petrol_Pump_1096_6_26_2019_12_37_51_jpg.rf.705def5d59a833af91f931e65e0fae50.jpg"
)

WEIGHTS_PATH = BASE_DIR / "model.safetensors"
DICT_PATH = BASE_DIR / "en_dict.txt"

OUTPUT_DIR = BASE_DIR / "yolo_ocr_results"
CROP_DIR = OUTPUT_DIR / "plate_crops"

OUTPUT_DIR.mkdir(exist_ok=True)
CROP_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# ADD PADDLEOCR SOURCE PATH
# --------------------------------------------------

sys.path.insert(0, str(PADDLEOCR_DIR))

from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from safetensors.numpy import load_file

# --------------------------------------------------
# AWIROS MODEL CONFIGURATION
# --------------------------------------------------

CTC_NUM_CLASSES = 64
NRTR_NUM_CLASSES = 67

MODEL_CONFIG = {
    "Architecture": {
        "model_type": "rec",
        "algorithm": "SVTR_HGNet",
        "Transform": None,
        "Backbone": {
            "name": "PPHGNetV2_B4",
            "text_rec": True
        },
        "Head": {
            "name": "MultiHead",
            "out_channels_list": {
                "CTCLabelDecode": CTC_NUM_CLASSES,
                "NRTRLabelDecode": NRTR_NUM_CLASSES
            },
            "head_list": [
                {
                    "CTCHead": {
                        "Neck": {
                            "name": "svtr",
                            "dims": 120,
                            "depth": 2,
                            "hidden_dims": 120,
                            "kernel_size": [1, 3],
                            "use_guide": True
                        },
                        "Head": {"fc_decay": 1e-05}
                    }
                },
                {
                    "NRTRHead": {
                        "nrtr_dim": 384,
                        "max_text_length": 25
                    }
                }
            ]
        }
    }
}

IMAGE_SHAPE = [3, 48, 320]


# --------------------------------------------------
# LOAD AWIROS OCR
# --------------------------------------------------

def load_ocr_model():

    print("Loading Awiros OCR model...")

    paddle.set_device("cpu")

    post_process = build_post_process({
        "name": "CTCLabelDecode",
        "character_dict_path": str(DICT_PATH),
        "use_space_char": True
    })

    config = copy.deepcopy(MODEL_CONFIG)

    model = build_model(config["Architecture"])
    model.eval()

    state = load_file(str(WEIGHTS_PATH))

    state = {
        key: paddle.to_tensor(value)
        for key, value in state.items()
    }

    model.set_state_dict(state)

    print("Awiros OCR loaded successfully.")

    return model, post_process


# --------------------------------------------------
# PREPROCESS PLATE CROP
# --------------------------------------------------

def preprocess_plate(img):

    height, width = img.shape[:2]

    target_h = 48
    target_w = 320

    ratio = target_h / height
    new_w = min(int(width * ratio), target_w)

    resized = cv2.resize(img, (new_w, target_h))

    padded = np.zeros(
        (target_h, target_w, 3),
        dtype=np.uint8
    )

    padded[:, :new_w] = resized

    image = padded.astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5

    image = image.transpose((2, 0, 1))

    return np.expand_dims(image, axis=0)


# --------------------------------------------------
# RECOGNIZE NUMBER PLATE
# --------------------------------------------------

def recognize_plate(crop, ocr_model, post_process):

    tensor = paddle.to_tensor(
        preprocess_plate(crop)
    )

    with paddle.no_grad():
        predictions = ocr_model(tensor)

    if isinstance(predictions, dict):
        prediction = predictions.get(
            "ctc",
            next(iter(predictions.values()))
        )

    elif isinstance(predictions, (list, tuple)):
        prediction = predictions[0]

    else:
        prediction = predictions

    result = post_process(prediction.numpy())

    if isinstance(result, (list, tuple)) and len(result) > 0:
        text, confidence = result[0]
        return text.strip().upper(), float(confidence)

    return "", 0.0


# --------------------------------------------------
# MAIN YOLO + OCR PIPELINE
# --------------------------------------------------

def main():

    print("\nLoading YOLO model...")

    detector = YOLO(str(YOLO_MODEL_PATH))

    print("YOLO model loaded successfully.")

    ocr_model, post_process = load_ocr_model()

    print("\nReading test image...")

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH}"
        )

    print("Image loaded:", image.shape)

    # Detect only number plates (class 1)
    results = detector.predict(
        source=image,
        classes=[1],
        conf=0.25,
        imgsz=1280,
        device="cpu",
        verbose=False
    )

    annotated = image.copy()

    plate_count = 0

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].cpu().numpy()
            )

            confidence = float(
                box.conf[0].cpu().numpy()
            )

            # Add a small margin around the plate
            margin_x = int((x2 - x1) * 0.08)
            margin_y = int((y2 - y1) * 0.12)

            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)

            x2 = min(image.shape[1], x2 + margin_x)
            y2 = min(image.shape[0], y2 + margin_y)

            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            plate_count += 1

            crop_path = (
                CROP_DIR /
                f"plate_{plate_count}.jpg"
            )

            cv2.imwrite(str(crop_path), crop)

            # Run Awiros OCR on the detected crop
            text, ocr_confidence = recognize_plate(
                crop,
                ocr_model,
                post_process
            )

            label = text if text else "Number plate unreadable"

            print("\n------------------------------")
            print("Plate:", plate_count)
            print("YOLO confidence:", round(confidence, 4))
            print("OCR result:", label)
            print("OCR confidence:", round(ocr_confidence, 4))
            print("------------------------------")

            display_label = (
                f"{text} ({ocr_confidence:.2f})"
                if text else "Unreadable"
            )

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            cv2.putText(
                annotated,
                display_label,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    if plate_count == 0:
        print("\nNo number plates detected by YOLO.")

    output_path = OUTPUT_DIR / "yolo_awiros_result.jpg"

    cv2.imwrite(str(output_path), annotated)

    print("\nTotal number plates detected:", plate_count)
    print("Saved result:", output_path)
    print("Saved crops:", CROP_DIR)


if __name__ == "__main__":
    main()
