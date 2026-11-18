
from pathlib import Path
from collections import defaultdict, Counter
import json
import re

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "video_engine_output"

VIDEO_JSON = OUTPUT / "video_results.json"
OCR_JSON = OUTPUT / "ocr_results.json"
EXCEL_FILE = OUTPUT / "traffic_violation_report.xlsx"

# Read processing results
video_data = json.loads(VIDEO_JSON.read_text(encoding="utf-8"))
ocr_data = json.loads(OCR_JSON.read_text(encoding="utf-8"))

# Match OCR readings to their original plate crop and bike ID
crop_to_bike = {
    item["image"]: item["bike_id"]
    for item in video_data["plate_crops"]
}

# Collect OCR readings per vehicle
readings = defaultdict(list)

for item in ocr_data:
    filename = item["image"]
    bike_id = crop_to_bike.get(filename)

    if bike_id is None:
        continue

    prediction = re.sub(
        r"[^A-Z0-9]", "", item["prediction"].upper()
    )

    confidence = float(item["confidence"])

    if prediction:
        readings[bike_id].append({
            "plate": prediction,
            "confidence": confidence
        })

# Select a plate only when at least two readings agree
vehicle_plates = {}

for bike_id, items in readings.items():

    reliable = [
        item for item in items
        if item["confidence"] >= 0.70
    ]

    votes = defaultdict(list)

    for item in reliable:
        votes[item["plate"]].append(item["confidence"])

    if not votes:
        vehicle_plates[bike_id] = {
            "plate": "Number plate unreadable",
            "status": "Needs review",
            "confidence": 0
        }
        continue

    best_plate = max(
        votes,
        key=lambda plate: (len(votes[plate]), sum(votes[plate]))
    )

    scores = votes[best_plate]

    if len(scores) >= 2:
        vehicle_plates[bike_id] = {
            "plate": best_plate,
            "status": "Repeated OCR match - verify",
            "confidence": round(sum(scores) / len(scores), 4)
        }
    else:
        vehicle_plates[bike_id] = {
            "plate": best_plate,
            "status": "Single OCR reading - verify",
            "confidence": round(scores[0], 4)
        }

# Create Excel workbook
wb = Workbook()

ws = wb.active
ws.title = "Violation Details"

headers = [
    "Vehicle ID",
    "Vehicle Number",
    "OCR Status",
    "OCR Confidence",
    "Violation",
    "Frame Number",
    "YOLO Confidence",
    "Evidence Image"
]

ws.append(headers)

for violation in video_data["violations"]:

    bike_id = violation["bike_id"]

    plate_info = vehicle_plates.get(
        bike_id,
        {
            "plate": "Number plate unreadable",
            "status": "Not associated",
            "confidence": 0
        }
    )

    ws.append([
        bike_id if bike_id is not None else "Unknown",
        plate_info["plate"],
        plate_info["status"],
        plate_info["confidence"],
        violation["violation"],
        violation["frame"],
        violation["yolo_confidence"],
        f"evidence/{violation['evidence']}"
    ])

# Second sheet: vehicle-wise OCR summary
summary = wb.create_sheet("Vehicle Summary")

summary.append([
    "Vehicle ID",
    "Recognised Number",
    "OCR Status",
    "OCR Confidence",
    "Number of OCR Readings"
])

for bike_id, info in sorted(
    vehicle_plates.items(),
    key=lambda x: str(x[0])
):
    summary.append([
        bike_id,
        info["plate"],
        info["status"],
        info["confidence"],
        len(readings[bike_id])
    ])

# Format worksheets
for sheet in wb.worksheets:

    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions

    for cell in sheet[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill(
            "solid", fgColor="203864"
        )
        cell.alignment = Alignment(
            horizontal="center"
        )

    for column in sheet.columns:
        max_length = max(
            len(str(cell.value or ""))
            for cell in column
        )
        width = min(max(max_length + 3, 14), 45)
        sheet.column_dimensions[
            get_column_letter(column[0].column)
        ].width = width

wb.save(EXCEL_FILE)

print("\nREPORT GENERATED SUCCESSFULLY")
print("Excel report:", EXCEL_FILE)
print("Vehicles with OCR readings:", len(vehicle_plates))
print("Violation records:", len(video_data["violations"]))
print("Evidence folder:", OUTPUT / "evidence")
