import streamlit as st
import cv2
import pandas as pd
import json
import uuid
import subprocess
import re
import shutil
import math
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from ultralytics import YOLO
import sys
import huggingface_hub
from supabase import create_client
from datetime import datetime
# ==========================================================
# YOLOCOP - AI Traffic Violation Detection System
# YOLOv8 + ByteTrack + Awiros ANPR
# ==========================================================
# Initialize Streamlit session state
SESSION_DEFAULTS = {
    "img_token": None,
    "vid_token": None,
    "active_kind": "image",
    "upload_token": None,
    "media_path": None,
    "last_run": None,
}

for key, default_value in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default_value
BASE = Path(__file__).resolve().parent
# Upload directory
WORK_DIR = BASE / "uploads"
WORK_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = (
    BASE / "runs" / "detect" / "outputs"
    / "traffic_violation_v1" / "weights" / "best.pt"
)
ENGINE_DIR = BASE / "lpr_engine" / "awiros_anpr"

OCR_SCRIPT = ENGINE_DIR / "test.py"
OCR_DICT = ENGINE_DIR / "en_dict.txt"
PADDLEOCR_DIR = ENGINE_DIR / "PaddleOCR"
HF_REPO_ID = "MohammedSalmanFawaz/yolocop-awiros-ocr"
PLATE_DETECTION_CONFIDENCE = 0.03
PLATE_SCAN_INTERVAL = 5

# Use the local OCR environment on Windows when available.
# Otherwise use the current application environment (Streamlit Cloud).
OCR_PYTHON_CANDIDATES = [
    BASE / "lpr_engine" / ".venv_lpr" / "Scripts" / "python.exe",
    ENGINE_DIR / ".venv_lpr" / "Scripts" / "python.exe",
    BASE / "lpr_engine" / ".venv_lpr" / "bin" / "python",
    ENGINE_DIR / ".venv_lpr" / "bin" / "python",
    Path(sys.executable),
]

OCR_PYTHON = next(
    (p for p in OCR_PYTHON_CANDIDATES if p.exists()),
    Path(sys.executable)
)

# ==========================================================
# DESIGN  (black + shadow burgundy, grid, 3D, motion)
# ==========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Space+Mono:wght@400;700&display=swap');

:root {
    --bg:#050505; --wine:#8E1B3A; --wine-bright:#D6264F; --wine-deep:#3B0B18;
    --rose:#FF6B88; --amber:#FFC531; --lime:#5BFF8A; --red:#FF3B4E;
    --text:#F4EAEC; --muted:#A28C92; --border:#2B1219; --card:#0C0608;
}

[data-testid="stSidebar"], [data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] {display:none!important;}
#MainMenu, footer {visibility:hidden;}
header[data-testid="stHeader"] {background:transparent;}

.stApp {
    background:
        linear-gradient(rgba(142,27,58,.13) 1px, transparent 1px),
        linear-gradient(90deg, rgba(142,27,58,.13) 1px, transparent 1px),
        radial-gradient(ellipse at 50% -10%, #3B0B18 0%, transparent 55%),
        #050505;
    background-size:44px 44px, 44px 44px, auto, auto;
    color:var(--text);
    font-family:Inter,sans-serif;
}
.stApp::before, .stApp::after {
    content:""; position:fixed; border-radius:50%; filter:blur(110px);
    pointer-events:none; z-index:0; opacity:.38;
}
.stApp::before {width:420px; height:420px; background:var(--wine);
    top:-140px; left:-120px; animation:drift 15s ease-in-out infinite;}
.stApp::after {width:460px; height:460px; background:#5a0e24;
    bottom:-170px; right:-130px; animation:drift 19s ease-in-out infinite reverse;}

/* the whole page is one dark card, like the reference layout */
.block-container {
    max-width:940px; margin-top:1.6rem; margin-bottom:2rem;
    padding:1.6rem 2rem 2.2rem; position:relative; z-index:1;
    background:rgba(10,5,7,.88); border:1px solid var(--border); border-radius:18px;
    box-shadow:0 30px 90px rgba(0,0,0,.75), 0 0 80px rgba(142,27,58,.22);
}
h1,h2,h3 {color:var(--text)!important;}
p {color:var(--muted);}

@keyframes drift {0%,100%{transform:translate(0,0) scale(1);} 50%{transform:translate(70px,50px) scale(1.15);}}
@keyframes fadeUp {from{opacity:0; transform:translateY(22px);} to{opacity:1; transform:none;}}
@keyframes float {0%,100%{transform:translateY(0);} 50%{transform:translateY(-5px);}}
@keyframes roadmove {from{background-position:0 0;} to{background-position:0 56px;}}
@keyframes ride {from{left:-6%;} to{left:104%;}}
@keyframes light-r {0%,30%{opacity:1; box-shadow:0 0 18px var(--red);} 34%,100%{opacity:.15; box-shadow:none;}}
@keyframes light-a {0%,32%{opacity:.15; box-shadow:none;} 36%,48%{opacity:1; box-shadow:0 0 18px var(--amber);} 52%,100%{opacity:.15; box-shadow:none;}}
@keyframes light-g {0%,50%{opacity:.15; box-shadow:none;} 54%,92%{opacity:1; box-shadow:0 0 18px var(--lime);} 96%,100%{opacity:.15; box-shadow:none;}}
@keyframes shine {to{background-position:200% center;}}
@keyframes sway {
    0%,100% {transform:perspective(1000px) rotateX(4deg) rotateY(-3deg);}
    50% {transform:perspective(1000px) rotateX(2deg) rotateY(3deg);}
}
@keyframes stripes {from{background-position:0 0;} to{background-position:104px 0;}}
@keyframes pulse {0%,100%{box-shadow:0 0 0 0 rgba(91,255,138,.55);} 50%{box-shadow:0 0 0 7px rgba(91,255,138,0);}}

/* ---------- header ---------- */
.hero {position:relative; overflow:hidden; border-radius:14px; padding:6px 4px 0;}
.hero-top {display:flex; justify-content:space-between; align-items:flex-start; gap:18px;}
.brand {font-size:2.5rem; font-weight:900; letter-spacing:5px;
    background:linear-gradient(90deg,var(--wine-bright),var(--rose),var(--wine),var(--wine-bright));
    background-size:200% auto; -webkit-background-clip:text; background-clip:text;
    -webkit-text-fill-color:transparent; animation:shine 6s linear infinite;
    filter:drop-shadow(0 0 16px rgba(214,38,79,.45));}
.sub {font-family:'Space Mono',monospace; color:var(--muted); font-size:.7rem;
    letter-spacing:2.5px; margin-top:6px;}
.hero-right {display:flex; align-items:flex-start; gap:16px;}
.status {font-family:'Space Mono',monospace; font-size:.64rem; font-weight:700;
    letter-spacing:1px; padding:6px 13px; border-radius:20px; white-space:nowrap;
    color:var(--lime); background:rgba(91,255,138,.08); border:1px solid rgba(91,255,138,.35);}
.status i {display:inline-block; width:8px; height:8px; border-radius:50%;
    margin-right:7px; background:var(--lime); animation:pulse 2s infinite;}
.status.off {color:var(--red); background:rgba(255,59,78,.08); border-color:rgba(255,59,78,.4);}
.status.off i {background:var(--red); animation:none;}

.signal {width:50px; padding:9px 0; border-radius:13px; background:#030303;
    border:2px solid #2a1219; display:flex; flex-direction:column; gap:8px;
    align-items:center; transform:perspective(500px) rotateY(-18deg);
    box-shadow:8px 8px 22px rgba(0,0,0,.7); flex-shrink:0;}
.signal i {width:23px; height:23px; border-radius:50%; display:block;}
.signal i:nth-child(1) {background:var(--red); animation:light-r 6s infinite;}
.signal i:nth-child(2) {background:var(--amber); animation:light-a 6s infinite;}
.signal i:nth-child(3) {background:var(--lime); animation:light-g 6s infinite;}

.road-wrap {position:relative; height:110px; margin:18px -4px 0; overflow:hidden;
    perspective:420px; border-radius:12px; background:linear-gradient(#080305,#12060b);}
.road {position:absolute; left:-25%; right:-25%; bottom:-70px; height:260px;
    transform:rotateX(64deg); transform-origin:50% 100%;
    background:linear-gradient(#150a0e,#240f17);
    border-left:6px solid var(--wine-bright); border-right:6px solid var(--wine-bright);
    box-shadow:0 0 40px rgba(214,38,79,.45);}
.road::before {content:""; position:absolute; top:0; bottom:0; left:calc(50% - 5px);
    width:10px; background:repeating-linear-gradient(180deg,var(--amber) 0 28px,transparent 28px 56px);
    animation:roadmove .7s linear infinite;}
.rider {position:absolute; bottom:18px; font-size:1.7rem; z-index:2;
    animation:ride 8s linear infinite; filter:drop-shadow(0 0 12px var(--wine-bright));}

.rule {border-bottom:1px solid var(--border); margin:24px 0 20px;}

/* ---------- headings ---------- */
.h-title {font-size:1.45rem; font-weight:800; color:var(--text); margin:2px 0 4px;
    animation:fadeUp .6s ease both;}
.h-sub {color:var(--muted); font-size:.95rem; margin-bottom:14px;}
.h-row {display:flex; justify-content:space-between; align-items:baseline;}
.h-tag {color:var(--muted); font-size:.8rem;}

/* ---------- upload cards ---------- */
.up-title {font-size:1.1rem; font-weight:800; margin-top:2px;}
.up-sub {font-family:'Space Mono',monospace; font-size:.66rem; letter-spacing:1px;
    color:var(--muted); margin-bottom:6px;}
[data-testid="stFileUploader"] {
    background:linear-gradient(160deg,#1a0a10,#0e0508); border:1px solid var(--border);
    border-radius:14px; padding:12px; transition:transform .2s, box-shadow .2s, border-color .2s;
}
[data-testid="stFileUploader"]:hover {
    transform:perspective(700px) rotateX(3deg) translateY(-4px);
    border-color:var(--wine-bright); box-shadow:0 16px 36px rgba(214,38,79,.25);
}
[data-testid="stFileUploader"] section {background:transparent!important;
    border:1px dashed #4a1a29!important; border-radius:10px;}

/* ---------- buttons ---------- */
div.stButton > button, div.stDownloadButton > button {
    background:linear-gradient(135deg,var(--wine-bright),var(--wine))!important;
    color:#FFF4F6!important; border:0!important; border-radius:11px!important;
    min-height:54px; font-weight:900!important; letter-spacing:1.3px;
    box-shadow:0 8px 0 #4a0c1d, 0 14px 30px rgba(214,38,79,.35);
    transition:transform .12s, box-shadow .12s;
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
    transform:translateY(-3px); color:#fff!important;
    box-shadow:0 11px 0 #4a0c1d, 0 20px 38px rgba(214,38,79,.5);
}
div.stButton > button:active, div.stDownloadButton > button:active {
    transform:translateY(6px); box-shadow:0 2px 0 #4a0c1d;
}
div.stButton > button:disabled, div.stDownloadButton > button:disabled {
    background:#1a1013!important; color:#6c5a60!important;
    box-shadow:0 6px 0 #0d0709; transform:none;
}

.stProgress > div > div > div > div {background:linear-gradient(90deg,var(--wine),var(--rose));}
[data-testid="stSlider"] [role="slider"] {background:var(--wine-bright)!important;
    box-shadow:0 0 14px var(--wine-bright);}
video, [data-testid="stImage"] img {border:1px solid var(--border); border-radius:10px;}
[data-testid="stDataFrame"] {border:1px solid var(--border); border-radius:10px;}
[data-testid="stExpander"] {border:1px solid var(--border)!important;
    border-radius:12px!important; background:rgba(20,8,12,.6);}

/* ---------- placeholders ---------- */
.ph {background:linear-gradient(160deg,#1a0a10,#0e0508); border:1px solid var(--border);
    border-radius:14px; padding:26px 20px; text-align:center; color:var(--muted);}
.ph .ico {font-size:1.6rem; display:block; margin-bottom:6px;}
.ph.slim {text-align:left; padding:16px 20px;}

/* ---------- 3D KPI ---------- */
.kpi-wrap {perspective:900px;}
.kpi {position:relative; background:linear-gradient(160deg,#1a0a10,#0c0507);
    border:1px solid var(--border); border-radius:14px; padding:14px 16px;
    transform-style:preserve-3d; animation:float 5s ease-in-out infinite;
    transition:transform .25s, border-color .25s, box-shadow .25s;
    box-shadow:0 12px 28px rgba(0,0,0,.55);}
.kpi:hover {transform:rotateX(8deg) rotateY(-10deg) translateY(-6px) scale(1.03);
    border-color:var(--wine-bright); box-shadow:0 22px 40px rgba(214,38,79,.28); animation:none;}
.kpi::before {content:""; position:absolute; left:0; top:12%; bottom:12%; width:4px;
    border-radius:4px; background:var(--wine-bright); box-shadow:0 0 14px var(--wine-bright);}
.kpi.amber::before {background:var(--amber); box-shadow:0 0 14px var(--amber);}
.kpi.lime::before {background:var(--lime); box-shadow:0 0 14px var(--lime);}
.kpi-l {font-family:'Space Mono',monospace; font-size:.58rem; letter-spacing:1.6px; color:var(--muted);}
.kpi-v {font-size:2rem; font-weight:900; color:var(--text); line-height:1.2; transform:translateZ(30px);}
.kpi-s {font-size:.68rem; color:var(--muted);}

/* ---------- vehicle cards ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    background:linear-gradient(160deg,#1a0a10,#0c0507);
    border:1px solid var(--border)!important; border-radius:14px!important;
    transition:transform .25s, box-shadow .25s, border-color .25s;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform:perspective(800px) rotateX(3deg) translateY(-6px);
    border-color:var(--wine-bright)!important; box-shadow:0 20px 40px rgba(214,38,79,.25);
}
.badge {display:inline-block; padding:3px 11px; border-radius:20px; font-size:.68rem;
    font-weight:800; margin:3px 5px 3px 0; background:rgba(214,38,79,.14); color:var(--rose);
    border:1px solid rgba(214,38,79,.55); box-shadow:0 0 14px rgba(214,38,79,.25);}
.plate {font-family:'Space Mono',monospace; font-size:1.2rem; font-weight:700;
    background:var(--amber); color:#101010; border-radius:7px; padding:4px 14px;
    display:inline-block; letter-spacing:3px; border:2px solid #101010;
    box-shadow:0 0 0 2px var(--amber), 0 6px 0 #a67c00, 0 10px 18px rgba(255,197,49,.3);}
.plate.unread {background:#1a1013; color:#7d6a70; font-size:.8rem; letter-spacing:0;
    box-shadow:none; border-color:#1a1013;}

.notice {background:rgba(255,197,49,.07); border:1px solid rgba(255,197,49,.35);
    color:#ffe08a; border-radius:10px; padding:10px 14px; font-size:.82rem; margin:14px 0;}

/* ---------- road safety pledge (centered, end of page) ---------- */
.pledge-stage {perspective:1000px; margin:14px 0 4px;}
.pledge {position:relative; text-align:center; max-width:760px; margin:0 auto;
    padding:40px 26px 30px; border-radius:18px; overflow:hidden;
    background:linear-gradient(160deg,#1c0a11,#080304);
    border:1px solid rgba(214,38,79,.4);
    box-shadow:0 30px 70px rgba(0,0,0,.7), 0 0 60px rgba(214,38,79,.2);
    animation:sway 9s ease-in-out infinite;}
.pledge::before, .pledge::after {content:""; position:absolute; left:0; right:0; height:8px;
    background:repeating-linear-gradient(90deg,var(--amber) 0 26px,#0a0305 26px 52px);
    animation:stripes 2.5s linear infinite;}
.pledge::before {top:0;} .pledge::after {bottom:0;}
.pledge-line {font-size:1.2rem; font-weight:800; color:var(--text); margin:10px 0;
    opacity:0; animation:fadeUp .8s ease forwards;}
.pledge-line:nth-child(1) {animation-delay:.2s;}
.pledge-line:nth-child(2) {animation-delay:.6s;}
.pledge-because {font-family:'Space Mono',monospace; font-size:1rem; font-weight:700;
    letter-spacing:8px; color:var(--rose); margin:22px 0 8px; opacity:0;
    animation:fadeUp .8s ease 1s forwards; text-shadow:0 0 18px var(--wine-bright);}
.pledge-big {font-size:2rem; font-weight:900; margin:8px 0 14px; opacity:0;
    background:linear-gradient(90deg,var(--amber),var(--rose),var(--amber));
    background-size:200% auto; -webkit-background-clip:text; background-clip:text;
    -webkit-text-fill-color:transparent;
    animation:fadeUp .9s ease 1.4s forwards, shine 4s linear 2s infinite;}
.pledge-by {text-align:right; color:var(--muted); font-style:italic; font-size:.92rem;
    padding-right:10px; opacity:0; animation:fadeUp .8s ease 1.9s forwards;}

.footer {text-align:center; color:#b8a4aa; font-family:'Space Mono',monospace;
    font-size:.8rem; letter-spacing:1.6px; padding-top:4px;}
.footer b {color:var(--rose); text-shadow:0 0 12px rgba(214,38,79,.7);}
</style>
""", unsafe_allow_html=True)


# ==========================================================
# MODEL
# ==========================================================

@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


# ==========================================================
# GENERAL HELPERS
# ==========================================================

def new_run_folder():
    folder = WORK_DIR / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + "_" + uuid.uuid4().hex[:5]
    )
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "evidence").mkdir()
    (folder / "plate_crops").mkdir()
    return folder


def get_class_name(model, class_id):
    names = model.names
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    return str(names[class_id])


def normalise_class(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def is_plate(name):
    value = normalise_class(name)
    return "numberplate" in value or value == "plate"


def is_bike(name):
    return normalise_class(name) in ("bike", "motorbike", "motorcycle")


def is_violation(name):
    value = normalise_class(name)
    return any(term in value for term in [
        "trippleriding", "tripleriding", "withouthelmet", "usingmobile"
    ])


def violation_label(name):
    value = normalise_class(name)
    if "trippleriding" in value or "tripleriding" in value:
        return "Triple Riding"
    if "withouthelmet" in value:
        return "Without Helmet"
    if "usingmobile" in value:
        return "Using Mobile"
    return str(name)


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def box_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def point_inside(point, box):
    x, y = point
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def associate_to_bike(target_box, bikes):
    """Match a plate/violation box to the most suitable bike box."""
    if not bikes:
        return None

    center = box_center(target_box)
    candidates = []

    for bike in bikes:
        inside = point_inside(center, bike["box"])
        overlap = box_iou(target_box, bike["box"])
        if inside or overlap > 0.05:
            candidates.append((overlap + (0.5 if inside else 0), bike))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def relative_center_distance(bike):
        bike_center = box_center(bike["box"])
        width = max(bike["box"][2] - bike["box"][0], 1)
        height = max(bike["box"][3] - bike["box"][1], 1)
        distance = (
            (center[0] - bike_center[0]) ** 2
            + (center[1] - bike_center[1]) ** 2
        ) ** 0.5
        return distance / (width ** 2 + height ** 2) ** 0.5

    nearest = min(bikes, key=relative_center_distance)
    if relative_center_distance(nearest) <= 1.25:
        return nearest
    return None


def crop_box(image, box):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def save_crop(image, path, verify=False):
    if image is None or image.size == 0:
        raise ValueError(f"Cannot save an empty image crop: {path}")
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not save image crop: {path}")
    if verify and cv2.imread(str(path)) is None:
        raise OSError(f"Saved image crop cannot be decoded: {path}")

def download_ocr_weights():
    try:
        model_path = huggingface_hub.hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="model.safetensors"
        )
        return Path(model_path)

    except Exception as e:
        raise RuntimeError(
            f"Unable to download Awiros OCR model from Hugging Face: {e}"
        )

def extract_detections(result, model):
    detections = []
    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        track_id = int(box.id[0].item()) if box.id is not None else None
        detections.append({
            "class": get_class_name(model, class_id),
            "confidence": float(box.conf[0].item()),
            "box": box.xyxy[0].tolist(),
            "track_id": track_id
        })
    return detections


def detection_passes_threshold(item, confidence, image_shape):
    threshold = (
        PLATE_DETECTION_CONFIDENCE
        if is_plate(item["class"])
        else confidence
    )
    if item["confidence"] < threshold:
        return False
    if not is_plate(item["class"]):
        return True

    image_height, image_width = image_shape[:2]
    x1, y1, x2, y2 = item["box"]
    box_width = x2 - x1
    box_height = y2 - y1
    if (
        x1 <= 0 or y1 <= 0 or x2 >= image_width or y2 >= image_height
        or box_width < 12 or box_height < 8
    ):
        return False
    aspect_ratio = box_width / box_height
    return 1.2 <= aspect_ratio <= 8.0


def annotate_detections(frame, detections):
    """Draw compact, readable labels without changing detection results."""
    annotated = frame.copy()
    colors = {
        "bike": (255, 180, 0),
        "plate": (0, 220, 100),
        "violation": (40, 70, 245),
    }
    height, width = frame.shape[:2]
    font_scale = max(0.38, min(0.52, width / 2200))
    thickness = 1 if width < 1100 else 2
    box_thickness = 2 if width >= 900 else 1

    for item in detections:
        x1, y1, x2, y2 = map(int, item["box"])
        if is_bike(item["class"]):
            color = colors["bike"]
        elif is_plate(item["class"]):
            color = colors["plate"]
        elif is_violation(item["class"]):
            color = colors["violation"]
        else:
            color = (235, 235, 235)
        label = f"{item['class']} {item['confidence']:.2f}"
        if is_bike(item["class"]) and item.get("track_id") is not None:
            label += f" ID:{item['track_id']}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        label_y = y1 - 5
        if label_y - text_h - baseline < 0:
            label_y = min(height - baseline - 2, y1 + text_h + baseline + 5)
        bg_top = max(0, label_y - text_h - baseline - 3)
        bg_bottom = min(height - 1, label_y + baseline + 2)
        bg_right = min(width - 1, x1 + text_w + 6)
        cv2.rectangle(annotated, (x1, bg_top), (bg_right, bg_bottom), color, -1)
        cv2.putText(
            annotated, label, (x1 + 3, label_y), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, (15, 15, 15), thickness, cv2.LINE_AA
        )
    return annotated


# ==========================================================
# IMAGE PROCESSING
# ==========================================================

def process_image(model, image_path, output_dir, confidence, progress):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("Could not read the uploaded image.")

    result = model.predict(
        source=image,
        conf=min(confidence, PLATE_DETECTION_CONFIDENCE),
        verbose=False
    )[0]
    all_detections = extract_detections(result, model)
    detections = [
        item for item in all_detections
        if detection_passes_threshold(item, confidence, image.shape)
    ]

    bikes = []
    for index, item in enumerate(all_detections):
        if is_bike(item["class"]):
            bikes.append({
                "box": item["box"],
                "vehicle_id": f"Image-Vehicle-{index + 1}"
            })

    records, plate_map = [], []
    evidence_dir = output_dir / "evidence"
    plate_dir = output_dir / "plate_crops"
    annotated = annotate_detections(image, detections)

    for number, item in enumerate(detections, start=1):
        name = item["class"]

        if is_bike(name):
            bike = next((b for b in bikes if b["box"] == item["box"]), None)
            vehicle_id = bike["vehicle_id"] if bike else "Image-Unmatched"
        else:
            matched = associate_to_bike(item["box"], bikes)
            vehicle_id = (
                matched["vehicle_id"] if matched else f"Image-Unmatched-{number}"
            )

        evidence_image = ""
        if is_violation(name):
            evidence_image = f"evidence/violation_{number}.jpg"

        records.append({
            "Frame": 1,
            "Vehicle ID": vehicle_id,
            "Class": name,
            "Confidence": round(item["confidence"], 4),
            "Evidence Image": evidence_image
        })

        if is_plate(name):
            crop = crop_box(image, item["box"])
            if crop is not None and crop.size > 0:
                filename = f"plate_image_{number}.jpg"
                save_crop(crop, plate_dir / filename, verify=True)
                plate_map.append({
                    "image": filename, "vehicle_id": vehicle_id, "frame": 1
                })

        if is_violation(name):
            crop = crop_box(image, item["box"])
            if crop is not None:
                save_crop(crop, output_dir / evidence_image)

    output_image = output_dir / "detected_image.jpg"
    if not cv2.imwrite(str(output_image), annotated):
        raise OSError(f"Could not save annotated image: {output_image}")
    progress.progress(1.0, text="Image detection completed.")

    return output_image, records, plate_map, 1


# ==========================================================
# VIDEO PROCESSING
# ==========================================================

def convert_annotated_video(temp_video, output_video):
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg:
        raise RuntimeError(
            "FFmpeg was not found on PATH. Install FFmpeg, restart VS Code, "
            "and confirm `ffmpeg -version` works."
        )
    if not ffprobe:
        raise RuntimeError(
            "FFprobe was not found on PATH. Install the FFmpeg package that "
            "includes ffprobe, restart VS Code, and confirm `ffprobe -version` works."
        )
    if not temp_video.exists() or temp_video.stat().st_size == 0:
        raise RuntimeError("The annotated AVI is missing or empty; MP4 conversion was not run.")
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(temp_video),
        "-vf", "scale='min(1280,iw)':-2",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "28",
        "-threads", "1",
        "-pix_fmt", "yuv420p",
        "-color_range", "tv",
        "-movflags", "+faststart",
        "-an",
        str(output_video),
    ]
    
    conversion = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if conversion.returncode != 0:
        raise RuntimeError(
            "FFmpeg could not convert the annotated AVI to H.264 MP4.\n"
            + conversion.stderr[-5000:]
        )
    if not output_video.exists() or output_video.stat().st_size == 0:
        raise RuntimeError("FFmpeg reported success but the final MP4 is missing or empty.")

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
            "FFprobe could not validate the final MP4.\n"
            + probe.stderr[-3000:]
        )
    try:
        metadata = json.loads(probe.stdout)
        duration = float(metadata["format"]["duration"])
        video_streams = [
            stream for stream in metadata.get("streams", [])
            if stream.get("codec_type") == "video"
        ]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"FFprobe returned invalid MP4 metadata: {error}") from error

    if not math.isfinite(duration) or duration <= 0 or not video_streams:
        raise RuntimeError("The final MP4 has no valid duration or video stream.")
    stream = video_streams[0]
    try:
        stream_width = int(stream.get("width", 0))
        stream_height = int(stream.get("height", 0))
    except (TypeError, ValueError) as error:
        raise RuntimeError("FFprobe returned invalid video dimensions.") from error
    if (
        stream.get("codec_name") != "h264"
        or stream.get("pix_fmt") != "yuv420p"
        or stream_width <= 0
        or stream_height <= 0
    ):
        raise RuntimeError(
            "The final MP4 did not validate as a browser-compatible "
            "H.264/yuv420p video stream."
        )

    temp_video.unlink()
    return duration


def process_video(model, video_path, output_dir, confidence, progress):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Could not open the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps) or fps <= 0:
        fps = 25
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("The uploaded video has invalid frame dimensions.")

    output_video = output_dir / "detected_traffic.mp4"
    temp_video = output_dir / "detected_traffic_temp.avi"
    writer = cv2.VideoWriter(
        str(temp_video), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise ValueError("Could not create the temporary MJPG AVI video.")

    records, plate_map = [], []
    frame_number = plate_number = evidence_number = 0
    evidence_dir = output_dir / "evidence"
    plate_dir = output_dir / "plate_crops"
    # Keep only the strongest plate crop per tracked vehicle; still scan every
    # configured interval so a poor frame does not decide the OCR result.
    best_plate_candidates = {}

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_number += 1

            result = model.track(
                source=frame, persist=True, tracker="bytetrack.yaml",
                conf=confidence,
                verbose=False
            )[0]

            all_detections = extract_detections(result, model)
            if frame_number % PLATE_SCAN_INTERVAL == 0:
                plate_result = model.predict(
                    source=frame,
                    classes=[1],
                    conf=PLATE_DETECTION_CONFIDENCE,
                    imgsz=960,
                    verbose=False,
                )[0]
                detected_plates = [
                    item for item in all_detections
                    if is_plate(item["class"])
                ]
                for plate in extract_detections(plate_result, model):
                    if not detection_passes_threshold(
                        plate, confidence, frame.shape
                    ):
                        continue
                    if not any(
                        box_iou(plate["box"], existing["box"]) > 0.5
                        for existing in detected_plates
                    ):
                        plate["track_id"] = None
                        all_detections.append(plate)
                        detected_plates.append(plate)
            detections = [
                item for item in all_detections
                if detection_passes_threshold(item, confidence, frame.shape)
            ]

            bikes = []
            for index, item in enumerate(all_detections, start=1):
                if is_bike(item["class"]):
                    vehicle_id = (
                        str(item["track_id"])
                        if item["track_id"] is not None
                        else f"Untracked-f{frame_number}-b{index}"
                    )
                    item["vehicle_id"] = vehicle_id
                    bikes.append({
                        "box": item["box"],
                        "vehicle_id": vehicle_id
                    })

            writer.write(annotate_detections(frame, detections))

            for detection_index, item in enumerate(detections, start=1):
                name = item["class"]

                if is_bike(name):
                    vehicle_id = item.get("vehicle_id", "Unmatched")
                else:
                    matched = associate_to_bike(item["box"], bikes)
                    vehicle_id = (
                        matched["vehicle_id"] if matched
                        else f"Unmatched-f{frame_number}-d{detection_index}"
                    )

                evidence_image = ""
                if is_violation(name):
                    evidence_number += 1
                    evidence_image = (
                        f"evidence/violation_f{frame_number}_{evidence_number}.jpg"
                    )

                records.append({
                    "Frame": frame_number,
                    "Vehicle ID": vehicle_id,
                    "Class": name,
                    "Confidence": round(item["confidence"], 4),
                    "Evidence Image": evidence_image
                })

                if is_plate(name):
                    crop = crop_box(frame, item["box"])
                    if crop is not None and crop.size > 0:
                        matched_bike = associate_to_bike(item["box"], bikes)
                        plate_vehicle_id = (
                            matched_bike["vehicle_id"] if matched_bike
                            else f"Unmatched-plate-f{frame_number}-d{detection_index}"
                        )
                        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                        area_ratio = crop.shape[0] * crop.shape[1] / max(1, height * width)
                        quality_score = (
                            0.55 * item["confidence"]
                            + 0.25 * min(sharpness / 500.0, 1.0)
                            + 0.20 * min(area_ratio / 0.015, 1.0)
                        )
                        current = best_plate_candidates.get(plate_vehicle_id)
                        if current is None or quality_score > current["score"]:
                            best_plate_candidates[plate_vehicle_id] = {
                                "score": quality_score,
                                "crop": crop.copy(),
                                "frame": frame_number,
                                "confidence": item["confidence"],
                            }

                if is_violation(name):
                    crop = crop_box(frame, item["box"])
                    if crop is not None and crop.size > 0:
                        save_crop(crop, output_dir / evidence_image)

            if total_frames > 0:
                progress.progress(
                    min(frame_number / total_frames, 1.0),
                    text=f"Processing frame {frame_number}/{total_frames}"
                )
    finally:
        cap.release()
        writer.release()

    if frame_number == 0:
        raise RuntimeError("The uploaded video contains no readable frames.")

    for index, (vehicle_id, candidate) in enumerate(
        best_plate_candidates.items(), start=1
    ):
        filename = f"best_plate_v{vehicle_id}_{index}.jpg"
        save_crop(candidate["crop"], plate_dir / filename, verify=True)
        plate_map.append({
            "image": filename,
            "vehicle_id": vehicle_id,
            "frame": candidate["frame"],
            "detection_confidence": round(candidate["confidence"], 4),
        })

    progress.progress(1.0, text="Converting and validating H.264 output...")
    convert_annotated_video(temp_video, output_video)
    return output_video, records, plate_map, frame_number

# ==========================================================
# AWIROS OCR + HUGGING FACE
# ==========================================================

def download_ocr_weights():
    try:
        return Path(
            huggingface_hub.hf_hub_download(
                repo_id=HF_REPO_ID,
                filename="model.safetensors"
            )
        )
    except Exception as e:
        raise RuntimeError(
            f"Hugging Face OCR model download failed: {e}"
        ) from e


def run_awiros_ocr(plate_directory, output_json):
    plate_directory = Path(plate_directory)
    output_json = Path(output_json)

    if not list(plate_directory.glob("*.jpg")):
        return []

    # Use the active Streamlit Python environment
    ocr_python = Path(sys.executable)

    if not OCR_SCRIPT.exists():
        raise FileNotFoundError(
            f"Awiros script not found: {OCR_SCRIPT}"
        )

    if not OCR_DICT.exists():
        raise FileNotFoundError(
            f"OCR dictionary not found: {OCR_DICT}"
        )
        
    "--paddleocr_dir", ""
    
    # Download model from Hugging Face
    ocr_weights = download_ocr_weights()

    command = [
        str(ocr_python),
        str(OCR_SCRIPT),
        "--image_path", str(plate_directory),
        "--weights", str(ocr_weights),
        "--dict_path", str(OCR_DICT),
        "--device", "cpu",
        "--output_json", str(output_json),
        "--paddleocr_dir", ""
    ]

    result = subprocess.run(
        command,
        cwd=str(ENGINE_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Awiros OCR failed.\n"
            + result.stdout[-3000:]
            + "\n"
            + result.stderr[-5000:]
        )

    if not output_json.exists():
        raise RuntimeError(
            "Awiros did not create its output JSON."
        )

    with open(output_json, "r", encoding="utf-8") as f:
        results = json.load(f)

    if not isinstance(results, list):
        raise ValueError("Awiros output JSON must contain a list.")

    for result in results:
        if not isinstance(result, dict):
            raise ValueError("Invalid OCR result.")

        if not result.get("image"):
            raise ValueError("OCR result has no image filename.")

        if "confidence" not in result or "prediction" not in result:
            raise ValueError("OCR result is incomplete.")

        confidence = float(result["confidence"])

        if not 0 <= confidence <= 1:
            raise ValueError("OCR confidence is outside [0, 1].")

    return results

# ==========================================================
# MATCH OCR READINGS TO VEHICLES
# ==========================================================

def build_ocr_lookup(ocr_results, plate_map):
    mapping = {Path(item["image"]).name.casefold(): item for item in plate_map}
    readings = defaultdict(list)

    for result in ocr_results:
        filename = Path(result["image"]).name
        association = mapping.get(filename.casefold())
        if not association:
            continue

        prediction = str(result.get("prediction", "")).strip().upper()
        if not prediction:
            continue

        confidence = float(result["confidence"])
        readings[association["vehicle_id"]].append({
            "plate": prediction,
            "confidence": confidence,
            "frame": association["frame"],
            "image": filename
        })

    return readings


def build_ocr_report_rows(ocr_results, plate_map, ocr_error=None):
    results_by_image = {
        Path(result["image"]).name.casefold(): result
        for result in ocr_results
        if result.get("image")
    }
    mapped_images = set()
    rows = []

    for association in plate_map:
        image = Path(association["image"]).name
        key = image.casefold()
        mapped_images.add(key)
        result = results_by_image.get(key)
        prediction = ""
        confidence = None
        status = "No OCR result returned"
        error = ""

        if ocr_error:
            prediction = "Unreadable - Verify"
            status = "OCR failed"
            error = ocr_error
        elif result is not None:
            prediction = str(result.get("prediction", "")).strip().upper()
            confidence = float(result["confidence"])
            status = "Recognized - verify" if prediction else "Unreadable - Verify"
        else:
            prediction = "Unreadable - Verify"

        rows.append({
            "Image": image,
            "Vehicle ID": association["vehicle_id"],
            "Frame": association["frame"],
            "Prediction": prediction or "Unreadable - Verify",
            "Confidence": confidence,
            "OCR Status": status,
            "Error": error,
        })

    for key, result in results_by_image.items():
        if key in mapped_images:
            continue
        rows.append({
            "Image": Path(result["image"]).name,
            "Vehicle ID": "Unmatched",
            "Frame": None,
            "Prediction": str(result.get("prediction", "")).strip().upper(),
            "Confidence": float(result["confidence"]),
            "OCR Status": "No matching YOLO crop association",
            "Error": "",
        })

    if not rows:
        rows.append({
            "Image": "",
            "Vehicle ID": "",
            "Frame": None,
            "Prediction": "",
            "Confidence": None,
            "OCR Status": (
                "OCR failed" if ocr_error
                else "No YOLO number-plate crops were created"
            ),
            "Error": ocr_error or "",
        })
    return rows


def choose_plate(readings):
    if not readings:
        return "Unreadable - Verify", None, "No OCR reading"

    votes = defaultdict(float)
    count = Counter()
    for item in readings:
        votes[item["plate"]] += item["confidence"]
        count[item["plate"]] += 1

    best_plate = max(votes, key=votes.get)
    matching = [i for i in readings if i["plate"] == best_plate]
    average = sum(i["confidence"] for i in matching) / len(matching)

    status = (
        "Repeated OCR match - verify" if len(matching) >= 2
        else "Single OCR reading - verify"
    )
    return best_plate, round(average, 4), status


# ==========================================================
# VEHICLE-WISE REPORT
# ==========================================================

def build_vehicle_report(records, ocr_lookup):
    """One summary row per vehicle, with all distinct detected violations."""
    grouped = defaultdict(list)
    for item in records:
        if is_violation(item["Class"]):
            grouped[str(item["Vehicle ID"])].append(item)

    report_rows = []
    for vehicle_id, detections in grouped.items():
        plate, ocr_conf, ocr_status = choose_plate(ocr_lookup.get(vehicle_id, []))
        violations = list(dict.fromkeys(
            violation_label(item["Class"]) for item in detections
        ))
        frames = [int(item["Frame"]) for item in detections]
        report_rows.append({
            "Vehicle ID": vehicle_id,
            "Number Plate": plate,
            "OCR Confidence": ocr_conf,
            "OCR Status": ocr_status,
            "Violations": "; ".join(violations),
            "Violation Count": len(violations),
            "First Frame": min(frames) if frames else None,
            "Last Frame": max(frames) if frames else None,
        })
    return report_rows


def build_evidence_rows(records, ocr_lookup):
    """One evidence row for each violation detection; no raw box coordinates."""
    rows = []
    for item in records:
        if not is_violation(item["Class"]):
            continue
        vehicle_id = str(item["Vehicle ID"])
        plate, ocr_conf, ocr_status = choose_plate(ocr_lookup.get(vehicle_id, []))
        rows.append({
            "Vehicle ID": vehicle_id,
            "Number Plate": plate,
            "Violation": violation_label(item["Class"]),
            "Frame": item["Frame"],
            "Detection Confidence": item["Confidence"],
            "OCR Confidence": ocr_conf,
            "OCR Status": ocr_status,
            "Evidence Image": item.get("Evidence Image", ""),
        })
    return rows


def enrich_detection_records(records, ocr_lookup):
    """Add the selected OCR result to the on-screen detection list."""
    enriched = []
    for item in records:
        row = dict(item)
        vehicle_id = str(row.get("Vehicle ID", "Unmatched"))
        plate, ocr_conf, ocr_status = choose_plate(ocr_lookup.get(vehicle_id, []))
        row["Number Plate"] = plate
        row["OCR Confidence"] = ocr_conf
        row["OCR Status"] = ocr_status
        enriched.append(row)
    return enriched


# ==========================================================
# SAVE REPORT
# ==========================================================

def save_reports(evidence_rows, vehicle_rows, output_dir):
    """Write the final two-sheet, vehicle-centred Excel report."""
    output_dir = Path(output_dir)
    summary_columns = [
        "Vehicle ID", "Number Plate", "OCR Confidence", "OCR Status",
        "Violations", "Violation Count", "First Frame", "Last Frame",
    ]
    evidence_columns = [
        "Vehicle ID", "Number Plate", "Violation", "Frame",
        "Detection Confidence", "OCR Confidence", "OCR Status", "Evidence Image",
    ]
    summary_df = pd.DataFrame(vehicle_rows, columns=summary_columns)
    evidence_df = pd.DataFrame(evidence_rows, columns=evidence_columns)
    xlsx_path = output_dir / "traffic_violation_report.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Vehicle Summary", index=False)
        evidence_df.to_excel(writer, sheet_name="Evidence Details", index=False)

    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    from openpyxl import load_workbook

    header_fill = PatternFill(fill_type="solid", fgColor="6B102B")
    header_font = Font(name="Aptos", size=11, bold=True, color="FFFFFF")
    body_font = Font(name="Aptos", size=10, color="24191C")
    alternate_fill = PatternFill(fill_type="solid", fgColor="F8F1F3")
    plain_fill = PatternFill(fill_type="solid", fgColor="FFFFFF")
    side = Side(style="thin", color="D8C6CC")
    border = Border(left=side, right=side, top=side, bottom=side)
    percent_headers = {"OCR Confidence", "Detection Confidence"}
    integer_headers = {"Frame", "First Frame", "Last Frame", "Violation Count"}

    workbook = load_workbook(xlsx_path)
    for worksheet in workbook.worksheets:
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        worksheet.row_dimensions[1].height = 28
        for cell in worksheet[1]:
            cell.fill, cell.font, cell.border = header_fill, header_font, border
            cell.alignment = Alignment(vertical="center", wrap_text=True)
        headers = [cell.value for cell in worksheet[1]]
        for row_index in range(2, worksheet.max_row + 1):
            fill = alternate_fill if row_index % 2 == 0 else plain_fill
            for cell in worksheet[row_index]:
                cell.fill, cell.font, cell.border = fill, body_font, border
                cell.alignment = Alignment(vertical="top", wrap_text=True)
        for col_index, header in enumerate(headers, start=1):
            letter = get_column_letter(col_index)
            if header in percent_headers:
                for cells in worksheet.iter_cols(min_col=col_index, max_col=col_index, min_row=2):
                    for cell in cells:
                        cell.number_format = "0.00%"
            elif header in integer_headers:
                for cells in worksheet.iter_cols(min_col=col_index, max_col=col_index, min_row=2):
                    for cell in cells:
                        cell.number_format = "0"
            lengths = [len(str(c.value)) if c.value is not None else 0 for c in worksheet[letter]]
            worksheet.column_dimensions[letter].width = min(max(max(lengths, default=0) + 2, 13), 48)
    workbook.save(xlsx_path)
    return xlsx_path


# ==========================================================
# DASHBOARD HELPERS
# ==========================================================

def kpi(label, value, sub="", tone=""):
    st.markdown(
        f'<div class="kpi-wrap"><div class="kpi {tone}">'
        f'<div class="kpi-l">{label}</div><div class="kpi-v">{value}</div>'
        f'<div class="kpi-s">{sub}</div></div></div>',
        unsafe_allow_html=True
    )


def best_plate_image(run, vehicle_id):
    owner = {p["image"]: p["vehicle_id"] for p in run["plate_map"]}
    best = None
    for r in run["ocr_results"]:
        name = Path(r.get("Image", "")).name
        if owner.get(name) == vehicle_id:
            confidence = r.get("Confidence")
            if confidence is None:
                continue
            conf = float(confidence)
            if best is None or conf > best[0]:
                best = (conf, name)
    if best:
        path = Path(run["folder"]) / "plate_crops" / best[1]
        return path if path.exists() else None
    return None


def run_pipeline(media_path, is_image, confidence, file_name, status_box):
    st.session_state.last_run = None
    run_dir = new_run_folder()
    model = load_model()
    progress = st.progress(0, text="Starting YOLO detection...")

    fn = process_image if is_image else process_video
    output, records, plate_map, frames = fn(
        model, media_path, run_dir, confidence, progress
    )

    with open(run_dir / "plate_map.json", "w", encoding="utf-8") as f:
        json.dump(plate_map, f, indent=2)

    raw_ocr_results, ocr_error = [], None

    if plate_map:
        status_box.info(
            f"YOLO complete. {len(plate_map)} plate crops found. "
            "Running Awiros OCR..."
        )
        try:
            raw_ocr_results = run_awiros_ocr(
                run_dir / "plate_crops",
                run_dir / "ocr_results.json"
            )
        except (
            OSError,
            RuntimeError,
            ValueError,
            subprocess.SubprocessError,
        ) as error:
            ocr_error = str(error)
            status_box.error(f"Awiros OCR error: {ocr_error}")

    else:
        status_box.warning(
            "No number-plate boxes were detected by YOLO, so OCR was skipped."
        )

    ocr_lookup = build_ocr_lookup(raw_ocr_results, plate_map)
    ocr_results = build_ocr_report_rows(
        raw_ocr_results, plate_map, ocr_error
    )
    vehicle_rows = build_vehicle_report(records, ocr_lookup)
    evidence_rows = build_evidence_rows(records, ocr_lookup)
    detection_rows = enrich_detection_records(records, ocr_lookup)
    report_path = save_reports(evidence_rows, vehicle_rows, run_dir)

    with open(run_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump({
            "input_file": file_name,
            "media_type": "image" if is_image else "video",
            "processed_frames": frames,
            "detection_count": len(records),
            "plate_crop_count": len(plate_map),
            "ocr_reading_count": len(raw_ocr_results),
            "completed_at": datetime.now().isoformat()
        }, f, indent=2)

    st.session_state.last_run = {
        "folder": str(run_dir),
        "output": str(output),
        "report": str(report_path),
        "media_type": "image" if is_image else "video",
        "records": detection_rows,
        "vehicle_rows": vehicle_rows,
        "evidence_rows": evidence_rows,
        "ocr_results": ocr_results,
        "plate_map": plate_map,
        "ocr_error": ocr_error,
        "processed_frames": frames,
        "file_name": file_name
    }

    progress.empty()
    status_box.success("Detection and report generation completed.")


# ==========================================================
# HEADER  (name + status, signal lights, riding motorcycle)
# ==========================================================

model_ok = MODEL_PATH.exists()

st.markdown(f"""
<div class="hero">
  <div class="hero-top">
    <div>
      <div class="brand">YOLOCOP</div>
      <div class="sub">AI TRAFFIC VIOLATION DETECTION SYSTEM</div>
    </div>
    <div class="hero-right">
      <span class="status {'' if model_ok else 'off'}"><i></i>{'AI ENGINE READY' if model_ok else 'MODEL MISSING'}</span>
      <div class="signal"><i></i><i></i><i></i></div>
    </div>
  </div>
  <div class="road-wrap">
    <div class="road"></div>
    <div class="rider">🏍️</div>
  </div>
</div>
<div class="rule"></div>
""", unsafe_allow_html=True)


# ==========================================================
# TRAFFIC VIOLATION DETECTION  (upload + start)
# ==========================================================
# Initialize Streamlit session state
SESSION_DEFAULTS = {
    "img_token": None,
    "vid_token": None,
    "active_kind": "image",
    "upload_token": None,
    "media_path": None,
    "last_run": None,
}

for key, default_value in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default_value
st.markdown(
    '<div class="h-title">Traffic Violation Detection</div>'
    '<div class="h-sub">Upload an image or video and start AI detection.</div>',
    unsafe_allow_html=True
)

col_img, col_vid = st.columns(2, gap="medium")

with col_img:
    st.markdown('<div class="up-title">🖼️ Upload Image</div>'
                '<div class="up-sub">JPG · PNG · JPEG</div>', unsafe_allow_html=True)
    img_up = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp"],
                              label_visibility="collapsed", key="img_uploader")

with col_vid:
    st.markdown('<div class="up-title">🎥 Upload Video</div>'
                '<div class="up-sub">MP4 · AVI · MOV</div>', unsafe_allow_html=True)
    vid_up = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"],
                              label_visibility="collapsed", key="vid_uploader")


def _tok(f):
    return f"{f.name}_{f.size}" if f is not None else None


if _tok(img_up) != st.session_state.img_token:
    st.session_state.img_token = _tok(img_up)
    if img_up is not None:
        st.session_state.active_kind = "image"

if _tok(vid_up) != st.session_state.vid_token:
    st.session_state.vid_token = _tok(vid_up)
    if vid_up is not None:
        st.session_state.active_kind = "video"

uploaded = img_up if st.session_state.active_kind == "image" else vid_up
if uploaded is None:
    uploaded = img_up or vid_up

media_path, is_image = None, False

if uploaded is not None:
    suffix = Path(uploaded.name).suffix.lower()
    is_image = suffix in [".jpg", ".jpeg", ".png", ".webp"]
    token = f"{uploaded.name}_{uploaded.size}"

    if st.session_state.upload_token != token:
        st.session_state.upload_token = token
        st.session_state.last_run = None
        saved = WORK_DIR / f"input_{uuid.uuid4().hex[:10]}{suffix}"
        with open(saved, "wb") as f:
            f.write(uploaded.getbuffer())
        st.session_state.media_path = str(saved)

    media_path = Path(st.session_state.media_path)
    st.caption(f"Selected: `{uploaded.name}` · {uploaded.size / (1024 * 1024):.2f} MB")

    if is_image:
        st.image(str(media_path), use_container_width=True)
    else:
        st.video(str(media_path))

confidence = st.slider("Detection confidence", 0.10, 0.90, 0.35, 0.05)

if not model_ok:
    st.error(f"YOLO model not found: {MODEL_PATH}")

start = st.button(
    "▶  START DETECTION",
    use_container_width=True,
    disabled=(media_path is None or not model_ok)
)

if start:
    status_box = st.empty()
    try:
        run_pipeline(media_path, is_image, confidence, uploaded.name, status_box)
    except Exception as error:
        st.error(f"Processing failed: {error}")
        st.exception(error)

run = st.session_state.last_run

st.markdown('<div class="rule"></div>', unsafe_allow_html=True)


# ==========================================================
# DETECTION RESULTS
# ==========================================================

st.markdown('<div class="h-title">Detection Results</div>', unsafe_allow_html=True)

if not run:
    st.markdown(
        '<div class="ph"><span class="ico">🔎</span>'
        'Processed image or video appears here</div>',
        unsafe_allow_html=True
    )
else:
    out = Path(run["output"])
    if run["media_type"] == "image":
        st.image(str(out), use_container_width=True)
    else:
        st.video(str(out))

    rows = run["vehicle_rows"]
    ocr = run["ocr_results"]
    records = run["records"]

    vehicles_flagged = len({r["Vehicle ID"] for r in rows})
    plates_read = len({
        str(r.get("Prediction", "")).strip().upper()
        for r in ocr
        if str(r.get("Prediction", "")).strip()
        and str(r.get("Prediction", "")).strip() != "Unreadable - Verify"
    })
    confs = [
        float(r["Confidence"]) for r in ocr
        if r.get("Confidence") is not None
    ]
    avg_ocr = f"{sum(confs) / len(confs):.0%}" if confs else "—"

    st.write("")
    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi("VIOLATIONS", sum(r["Violation Count"] for r in rows), "unique types per vehicle")
    with k2: kpi("VEHICLES", vehicles_flagged, "flagged")
    with k3: kpi("PLATES READ", plates_read, f"{len(run['plate_map'])} crops", "amber")
    with k4: kpi("AVG OCR", avg_ocr, f"{run['processed_frames']} frames", "lime")

    if run["ocr_error"]:
        st.warning("Awiros OCR could not complete. Detection results are still available.")
        with st.expander("OCR error details"):
            st.code(run["ocr_error"])

    st.markdown(
        '<div class="notice">⚠ Automated output. A human officer must verify '
        'every plate and violation before any enforcement action.</div>',
        unsafe_allow_html=True
    )


# ==========================================================
# VIOLATION EVIDENCE
# ==========================================================

st.write("")
st.markdown(
    '<div class="h-row"><div class="h-title">Violation evidence</div>'
    '<div class="h-tag">After processing</div></div>',
    unsafe_allow_html=True
)

if not run:
    st.markdown(
        '<div class="ph slim">Detection records and recognized number plates</div>',
        unsafe_allow_html=True
    )
else:
    rows = run["vehicle_rows"]

    if not rows:
        st.info("No violation detections were returned by YOLO.")
    else:
        df = pd.DataFrame(rows)
        all_types = sorted({
            violation.strip()
            for value in df["Violations"].fillna("")
            for violation in str(value).split(";") if violation.strip()
        })
        f1, f2 = st.columns([2, 1])
        chosen = f1.multiselect("Filter by violation", all_types, default=all_types)
        search = f2.text_input("Search plate / vehicle ID")

        cards = []
        for row in rows:
            row_types = [v.strip() for v in row["Violations"].split(";") if v.strip()]
            if chosen and not any(v in chosen for v in row_types):
                continue
            if search and search.lower() not in row["Vehicle ID"].lower() and search.lower() not in row["Number Plate"].lower():
                continue
            cards.append((row["Vehicle ID"], row))

        if not cards:
            st.info("No vehicles match the current filters.")

        cols = st.columns(2)
        for i, (vid, item) in enumerate(cards):
            unread = str(item["Number Plate"]).startswith("Unreadable")
            with cols[i % 2]:
                with st.container(border=True):
                    st.markdown(f"**Vehicle {vid}**")
                    crop = best_plate_image(run, vid)
                    if crop:
                        st.image(str(crop), use_container_width=True)
                    st.markdown(
                        f'<span class="plate {"unread" if unread else ""}">'
                        f'{item["Number Plate"]}</span>',
                        unsafe_allow_html=True
                    )
                    ocr_confidence = item["OCR Confidence"]
                    ocr_confidence_text = f"{ocr_confidence:.0%}" if ocr_confidence is not None else "—"
                    st.caption(f'OCR {ocr_confidence_text} · {item["OCR Status"]}')
                    st.markdown(
                        "".join(f'<span class="badge">{v.strip()}</span>'
                                for v in item["Violations"].split(";") if v.strip()),
                        unsafe_allow_html=True
                    )
                    st.caption(
                        f'Frames {item["First Frame"]}–{item["Last Frame"]}'
                    )

    folder = Path(run["folder"])
    ev_files = sorted((folder / "evidence").glob("*.jpg"))
    pl_files = sorted((folder / "plate_crops").glob("*.jpg"))

    if ev_files:
        with st.expander(f"Violation evidence images ({len(ev_files)})"):
            cols = st.columns(3)
            for i, p in enumerate(ev_files):
                with cols[i % 3]:
                    st.image(str(p), caption=p.name, use_container_width=True)

    if pl_files:
        with st.expander(f"Number-plate crops ({len(pl_files)})"):
            cols = st.columns(4)
            for i, p in enumerate(pl_files):
                with cols[i % 4]:
                    st.image(str(p), caption=p.name, use_container_width=True)

    with st.expander("OCR readings"):
        if run["ocr_results"]:
            st.dataframe(pd.DataFrame(run["ocr_results"]),
                         use_container_width=True, hide_index=True)
        else:
            st.info("No OCR predictions available.")

    with st.expander("Evidence details"):
        if run.get("evidence_rows"):
            st.dataframe(pd.DataFrame(run["evidence_rows"]),
                         use_container_width=True, hide_index=True)
        else:
            st.info("No violation evidence records were created.")

    with st.expander("All detections"):
        st.dataframe(pd.DataFrame(run["records"]),
                     use_container_width=True, hide_index=True)


# ==========================================================
# EXPORT
# ==========================================================

st.write("")

if run:
    report_path = Path(run["report"])
    with open(report_path, "rb") as f:
        report_bytes = f.read()

    is_xlsx = report_path.suffix == ".xlsx"
    st.download_button(
        "↓  EXPORT EXCEL REPORT" if is_xlsx else "↓  EXPORT CSV REPORT",
        data=report_bytes,
        file_name=report_path.name,
        mime=("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
              if is_xlsx else "text/csv"),
        use_container_width=True
    )
else:
    st.download_button(
        "↓  EXPORT EXCEL REPORT",
        data=b"",
        file_name="yolocop_report.xlsx",
        use_container_width=True,
        disabled=True
    )


# ==========================================================
# ROAD SAFETY MESSAGE  (centered, end of page)
# ==========================================================

st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="pledge-stage">
  <div class="pledge">
    <div class="pledge-line">Aapke aur sabki suraksha ke liye</div>
    <div class="pledge-line">Traffic niyamon ka kripya paalan kijiye</div>
    <div class="pledge-because">KYONKI</div>
    <div class="pledge-big">"Road kisi ke baap ka nahi hai"</div>
    <div class="pledge-by">— Akshay Kumar aka Khiladi</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ==========================================================
# FOOTER
# ==========================================================

st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

st.markdown(
    '<div class="footer">🚦 <b>Sadak Suraksha, Jeevan Raksha</b> · Made by Salman</div>',
    unsafe_allow_html=True
)

# ==========================================================
# YOLOCOP COMMUNITY COMMENTS
# GOOGLE LOGIN + SUPABASE
# ==========================================================

@st.cache_resource
def get_comments_db():
    return create_client(
        st.secrets["supabase"]["url"],
        st.secrets["supabase"]["secret_key"]
    )


def show_community_comments():

    st.markdown("---")
    st.markdown("## 💬 Community Comments")
    st.caption(
        "Share your feedback, suggestions and experience with YOLOCOP."
    )

    logged_in = st.user.is_logged_in

    # GOOGLE LOGIN
    if not logged_in:
        st.info("Sign in with Google to write a comment.")

        if st.button(
            "🔵 Continue with Google",
            key="yolocop_google_login"
        ):
            st.login()

    else:
        user_name = st.user.get("name", "Google User")
        user_email = st.user.get("email", "")
        user_id = st.user.get("sub", "")
        user_picture = st.user.get("picture", "")

        col1, col2 = st.columns([1, 8])

        with col1:
            if user_picture:
                st.image(user_picture, width=45)

        with col2:
            st.write(f"Commenting as **{user_name}**")

        # COMMENT INPUT FORM
        with st.form(
            "yolocop_comment_form",
            clear_on_submit=True
        ):
            comment_text = st.text_area(
                "Write your comment",
                placeholder="Share your thoughts about YOLOCOP...",
                max_chars=2000,
                height=120
            )

            submitted = st.form_submit_button(
                "Publish Comment",
                use_container_width=True
            )

            if submitted:
                comment_text = comment_text.strip()

                if not comment_text:
                    st.warning("Please write a comment.")

                elif not user_id:
                    st.error("Google account identity could not be verified.")

                else:
                    try:
                        db = get_comments_db()

                        db.table("yolocop_comments").insert({
                            "google_sub": user_id,
                            "author_name": user_name,
                            "author_email": user_email,
                            "avatar_url": user_picture,
                            "comment": comment_text
                        }).execute()

                        st.success("Comment published successfully!")
                        st.rerun()

                    except Exception as exc:
                        st.error(
                            "Unable to publish your comment. "
                            "Please check the database configuration."
                        )
                        print("COMMENT INSERT ERROR:", repr(exc))

        if st.button("Sign out", key="yolocop_google_logout"):
            st.logout()

    # DISPLAY EXISTING COMMENTS
    st.markdown("---")
    st.markdown("### Community Discussion")

    try:
        db = get_comments_db()

        response = (
            db.table("yolocop_comments")
            .select("id, author_name, avatar_url, comment, created_at")
            .order("created_at", desc=True)
            .limit(100)
            .execute()
        )

        comments = response.data or []

        st.markdown(f"### {len(comments)} COMMENTS")

        if not comments:
            st.write("Be the first to comment on YOLOCOP!")

        for item in comments:
            col1, col2 = st.columns([1, 12])

            with col1:
                if item.get("avatar_url"):
                    st.image(item["avatar_url"], width=42)
                else:
                    st.write("👤")

            with col2:
                st.markdown(f"**{item['author_name']}**")
                st.write(item["comment"])

                if item.get("created_at"):
                    st.caption(item["created_at"][:16])

            st.markdown("---")

    except Exception as exc:
        st.warning(
            "Comments database is not connected yet. "
            "Please check Supabase configuration."
        )
        print("COMMENT FETCH ERROR:", repr(exc))


show_community_comments()