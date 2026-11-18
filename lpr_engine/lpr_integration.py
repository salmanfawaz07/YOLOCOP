
from pathlib import Path
import subprocess
import shutil
import json
import sys

from ultralytics import YOLO


# ==========================================
# 1. PROJECT PATHS
# ==========================================

LPR_ENGINE = Path(__file__).resolve().parent
PROJECT_ROOT = LPR_ENGINE.parent

YOLO_MODEL = (
    PROJECT_ROOT
    / "runs/detect/outputs/traffic_violation_v1/weights/best.pt"
)

INPUT_IMAGES = PROJECT_ROOT / "test_images"

AWIROS_DIR = LPR_ENGINE / "awiros_anpr"
AWIROS_SCRIPT = AWIROS_DIR / "test.py"

LPR_PYTHON = (
    LPR_ENGINE / ".venv_lpr" / "Scripts" / "python.exe"
)

OUTPUT_DIR = LPR_ENGINE / "outputs" / "lpr_batch_test"
CROP_DIR = OUTPUT_DIR / "detection" / "crops" / "numberplate"
RESULT_JSON = OUTPUT_DIR / "ocr_results.json"


# ==========================================
# 2. CHECK REQUIRED FILES
# ==========================================

def check_files():

    required_files = [
        YOLO_MODEL,
        INPUT_IMAGES,
        AWIROS_SCRIPT,
        LPR_PYTHON,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file or folder missing: {path}"
            )

    print("All required files are available.")


# ==========================================
# 3. YOLO NUMBER PLATE DETECTION
# ==========================================

def detect_numberplates():

    print("\nSTEP 1: Loading YOLOv8 model...")

    model = YOLO(str(YOLO_MODEL))

    # Remove previous test output to avoid old crops
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("STEP 2: Detecting number plates...")

    model.predict(
        source=str(INPUT_IMAGES),
        classes=[1],
        conf=0.25,
        device="cpu",
        save=True,
        save_crop=True,
        project=str(OUTPUT_DIR),
        name="detection",
        exist_ok=True,
        verbose=False
    )

    if not CROP_DIR.exists():
        raise RuntimeError("YOLO crop folder was not created.")

    crops = list(CROP_DIR.glob("*"))

    print(f"\nNumber plate crops generated: {len(crops)}")

    return crops


# ==========================================
# 4. AWIROS NUMBER PLATE RECOGNITION
# ==========================================

def recognize_numberplates(crops):

    print("\nSTEP 3: Starting Awiros OCR...")

    if not crops:
        print("No number plates detected.")
        return []

    command = [
        str(LPR_PYTHON),
        str(AWIROS_SCRIPT),
        "--image_path", str(CROP_DIR),
        "--device", "cpu",
        "--output_json", str(RESULT_JSON)
    ]

    subprocess.run(
        command,
        cwd=str(AWIROS_DIR),
        check=True
    )

    if not RESULT_JSON.exists():
        raise RuntimeError("Awiros result JSON was not created.")

    with open(RESULT_JSON, "r", encoding="utf-8") as file:
        results = json.load(file)

    return results


# ==========================================
# 5. DISPLAY FINAL REPORT
# ==========================================

def display_report(results):

    print("\n" + "=" * 65)
    print("       TRAFFIC NUMBER PLATE RECOGNITION REPORT")
    print("=" * 65)

    if not results:
        print("No number plates were recognized.")
        return

    print(
        f"{'S.No':<6}"
        f"{'NUMBER PLATE':<25}"
        f"{'CONFIDENCE':<15}"
        f"{'STATUS'}"
    )

    print("-" * 65)

    for index, item in enumerate(results, start=1):

        plate = item["prediction"].strip().upper()
        confidence = item["confidence"]

        if plate:
            status = "Recognized"
        else:
            plate = "UNREADABLE"
            status = "Manual review"

        print(
            f"{index:<6}"
            f"{plate:<25}"
            f"{confidence:<15.2%}"
            f"{status}"
        )

    print("=" * 65)

    print(f"\nDetailed JSON report: {RESULT_JSON}")


# ==========================================
# 6. MAIN PROGRAM
# ==========================================

def main():

    print("\nTRAFFIC ANPR INTEGRATION SYSTEM")
    print("--------------------------------")

    check_files()

    crops = detect_numberplates()

    results = recognize_numberplates(crops)

    display_report(results)

    print("\nIntegration test completed successfully!")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print("\nERROR:", error)
        sys.exit(1)
