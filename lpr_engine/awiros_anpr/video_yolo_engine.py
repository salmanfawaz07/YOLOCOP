
from pathlib import Path
import cv2
import json
import math
from ultralytics import YOLO
import subprocess
import shutil
# ==========================================
# PROJECT PATHS
# ==========================================

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parent.parent

MODEL_PATH = (
    PROJECT / "runs/detect/outputs/traffic_violation_v1/weights/best.pt"
)

VIDEO_PATH = PROJECT / "test_videos/indian_traffic1.mp4"

OUTPUT = BASE / "video_engine_output"
CROPS = OUTPUT / "plate_crops"
EVIDENCE = OUTPUT / "evidence"

OUTPUT.mkdir(parents=True, exist_ok=True)
CROPS.mkdir(parents=True, exist_ok=True)
EVIDENCE.mkdir(parents=True, exist_ok=True)

# ==========================================
# SETTINGS
# ==========================================

CONFIDENCE = 0.08
IMAGE_SIZE = 960
OCR_CROP_INTERVAL = 15

CLASS_NAMES = {
    0: "bike",
    1: "numberplate",
    2: "triple riding",
    3: "using mobile",
    4: "with helmet",
    5: "without helmet"
}

VIOLATION_CLASSES = {
    2: "Triple Riding",
    3: "Using Mobile",
    5: "Without Helmet"
}

# ==========================================
# LOAD MODEL AND VIDEO
# ==========================================

print("Loading YOLO model...")
model = YOLO(str(MODEL_PATH))
print("YOLO model loaded.")

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 25
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

output_video = OUTPUT / "detected_traffic.mp4"
temp_video = OUTPUT / "detected_traffic_temp.avi"

# MJPG intermediate video
writer = cv2.VideoWriter(
    str(temp_video),
    cv2.VideoWriter_fourcc(*"MJPG"),
    fps,
    (width, height)
)

if not writer.isOpened():
    raise RuntimeError("Could not initialize video writer.")

print("Video writer initialized successfully.")

print("Video:", VIDEO_PATH.name)
print("Resolution:", width, "x", height)
print("Total frames:", total_frames)

# ==========================================
# RESULT STORAGE
# ==========================================

plate_records = []
violation_records = []
saved_evidence = set()
frame_number = 0

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def inside(inner_box, outer_box):
    cx, cy = center(inner_box)
    x1, y1, x2, y2 = outer_box
    return x1 <= cx <= x2 and y1 <= cy <= y2


def associate_with_bike(target_box, bikes):
    """Associate a plate or violation with a bike by geometry."""

    candidates = []

    for bike in bikes:
        bike_box = bike["box"]

        if inside(target_box, bike_box):
            candidates.append(bike)

    if not candidates:
        return None

    # Prefer the smallest containing bike box.
    return min(
        candidates,
        key=lambda b: (
            (b["box"][2] - b["box"][0]) *
            (b["box"][3] - b["box"][1])
        )
    )


# ==========================================
# PROCESS VIDEO
# ==========================================

print("\nStarting video processing...\n")

while True:

    success, frame = cap.read()

    if not success:
        break

    frame_number += 1

    # Track all six classes in one model pass.
    results = model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        device="cpu",
        verbose=False
    )

    result = results[0]
    bikes = []
    other_detections = []

    if result.boxes is not None:

        for box in result.boxes:

            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])

            coords = tuple(
                map(int, box.xyxy[0].cpu().numpy())
            )

            track_id = None

            if box.id is not None:
                track_id = int(box.id[0])

            detection = {
                "class_id": cls_id,
                "confidence": confidence,
                "box": coords,
                "track_id": track_id
            }

            if cls_id == 0 and track_id is not None:
                bikes.append(detection)

            elif cls_id in [1, 2, 3, 5]:
                other_detections.append(detection)

    # ======================================
    # DRAW BIKE TRACKS
    # ======================================

    annotated = frame.copy()

    for bike in bikes:

        x1, y1, x2, y2 = bike["box"]
        track_id = bike["track_id"]

        cv2.rectangle(
            annotated, (x1, y1), (x2, y2),
            (255, 180, 0), 2
        )

        cv2.putText(
            annotated,
            f"Bike ID: {track_id}",
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 180, 0),
            2
        )

    # ======================================
    # PROCESS PLATES AND VIOLATIONS
    # ======================================

    for detection in other_detections:

        cls_id = detection["class_id"]
        x1, y1, x2, y2 = detection["box"]
        confidence = detection["confidence"]

        bike = associate_with_bike(
            detection["box"], bikes
        )

        bike_id = bike["track_id"] if bike else None

        if cls_id == 1:

            label = "Number Plate"

            if bike_id is not None:
                label += f" | Bike {bike_id}"

                # Save occasional crops for OCR.
                if frame_number % OCR_CROP_INTERVAL == 0:

                    crop = frame[y1:y2, x1:x2]

                    if crop.size > 0:

                        crop_name = (
                            f"track_{bike_id}_frame_{frame_number}.jpg"
                        )

                        crop_path = CROPS / crop_name

                        if cv2.imwrite(str(crop_path), crop):

                            plate_records.append({
                                "image": crop_name,
                                "bike_id": bike_id,
                                "frame": frame_number,
                                "yolo_confidence": confidence
                            })

            color = (0, 255, 0)

        else:

            label = VIOLATION_CLASSES[cls_id]
            color = (0, 0, 255)

            incident_key = (bike_id, label)

            # Save one initial evidence image per bike and violation.
            if incident_key not in saved_evidence:

                evidence_name = (
                    f"frame_{frame_number}_"
                    f"bike_{bike_id if bike_id is not None else 'unknown'}_"
                    f"{cls_id}.jpg"
                )

                evidence_path = EVIDENCE / evidence_name

                cv2.imwrite(str(evidence_path), frame)

                violation_records.append({
                    "bike_id": bike_id,
                    "violation": label,
                    "frame": frame_number,
                    "evidence": evidence_name,
                    "yolo_confidence": confidence
                })

                saved_evidence.add(incident_key)

        # Draw plate and violation boxes.
        cv2.rectangle(
            annotated, (x1, y1), (x2, y2),
            color, 2
        )

        cv2.putText(
            annotated,
            f"{label} {confidence:.2f}",
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

    writer.write(annotated)

    if frame_number % 100 == 0:
        print(
            f"Processed {frame_number}/{total_frames} frames"
        )

# ==========================================
# SAVE OUTPUTS
# ==========================================

cap.release()
writer.release()

print("YOLO video processing completed.")

# Check intermediate video
if not temp_video.exists() or temp_video.stat().st_size == 0:
    raise RuntimeError("Output video is empty. No frames were saved.")

print("Intermediate video size:", temp_video.stat().st_size)

# Convert to browser-compatible H.264 MP4
ffmpeg = shutil.which("ffmpeg")
ffprobe = shutil.which("ffprobe")

if not ffmpeg:
    raise RuntimeError(
        "FFmpeg is not installed or not available in PATH. "
        "Install FFmpeg and restart VS Code."
    )
if not ffprobe:
    raise RuntimeError(
        "FFprobe is not installed or not available in PATH. "
        "Install the FFmpeg package that includes ffprobe."
    )

command = [
    ffmpeg,
    "-y",
    "-i", str(temp_video),
    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
    "-c:v", "libx264",
    "-preset", "fast",
    "-crf", "23",
    "-pix_fmt", "yuv420p",
    "-color_range", "tv",
    "-movflags", "+faststart",
    "-an",
    str(output_video)
]

process = subprocess.run(
    command,
    capture_output=True,
    text=True
)

if process.returncode != 0:
    raise RuntimeError(
        "H.264 video conversion failed:\n" + process.stderr[-3000:]
    )

if not output_video.exists() or output_video.stat().st_size == 0:
    raise RuntimeError("Final MP4 was not generated correctly.")

probe = subprocess.run(
    [
        ffprobe, "-v", "error",
        "-show_entries",
        "format=duration:stream=codec_type,codec_name,pix_fmt,width,height",
        "-of", "json", str(output_video),
    ],
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="replace",
)
if probe.returncode != 0:
    raise RuntimeError(
        "FFprobe could not validate the final MP4:\n" + probe.stderr[-3000:]
    )

try:
    probe_data = json.loads(probe.stdout)
    duration = float(probe_data["format"]["duration"])
    video_streams = [
        stream for stream in probe_data.get("streams", [])
        if stream.get("codec_type") == "video"
    ]
except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
    raise RuntimeError(f"FFprobe returned invalid MP4 metadata: {error}") from error

if not math.isfinite(duration) or duration <= 0 or not video_streams:
    raise RuntimeError("Final MP4 has no valid duration or video stream.")
video_stream = video_streams[0]
try:
    stream_width = int(video_stream.get("width", 0))
    stream_height = int(video_stream.get("height", 0))
except (TypeError, ValueError) as error:
    raise RuntimeError("FFprobe returned invalid video dimensions.") from error
if (
    video_stream.get("codec_name") != "h264"
    or video_stream.get("pix_fmt") != "yuv420p"
    or stream_width <= 0
    or stream_height <= 0
):
    raise RuntimeError(
        "Final MP4 is not a browser-compatible H.264/yuv420p video stream."
    )

# Keep the intermediate file if conversion or validation fails.
temp_video.unlink()

metadata = {
    "video": VIDEO_PATH.name,
    "total_frames_processed": frame_number,
    "plate_crops": plate_records,
    "violations": violation_records,
    "output_video": str(output_video),
    "video_size_bytes": output_video.stat().st_size
}

metadata_path = OUTPUT / "video_results.json"

metadata_path.write_text(
    json.dumps(metadata, indent=4),
    encoding="utf-8"
)

print("\n====================================")
print("VIDEO PROCESSING COMPLETED")
print("====================================")
print("Frames processed:", frame_number)
print("Plate crops saved:", len(plate_records))
print("Violations recorded:", len(violation_records))
print("Processed video:", output_video)
print("Final video size:", output_video.stat().st_size)
print("Results JSON:", metadata_path)
print("Evidence folder:", EVIDENCE)
print("Plate crops folder:", CROPS)
