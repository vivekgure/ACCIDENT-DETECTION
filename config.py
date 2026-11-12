"""
Configuration Settings for AURA-VISION AI Vehicle Accident Detection System.
Defines system paths, lightweight YOLO model parameters, road-user class mappings,
ByteTrack tracking heuristics, physics collision thresholds, hardware acceleration,
and Flask server settings.
"""

from pathlib import Path
from typing import Dict, Set, Optional, List

# ==============================================================================
# 1. Base Project Paths
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
UPLOADS_DIR = BASE_DIR / "uploads"
EVENTS_DIR = BASE_DIR / "accident_events"
SAMPLE_DIR = BASE_DIR / "sample_data"
SPRITES_DIR = SAMPLE_DIR / "sprites"
OUTPUT_DIR = BASE_DIR / "output_results"
DATASETS_DIR = BASE_DIR / "datasets"

for d in [MODELS_DIR, UPLOADS_DIR, EVENTS_DIR, SAMPLE_DIR, SPRITES_DIR, OUTPUT_DIR, DATASETS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. YOLO Model & Hardware Acceleration Settings
# ==============================================================================
# Primary lightweight real-time model (YOLOv8n)
YOLO_MODEL_NAME = "yolov8n.pt"
YOLO_MODEL_PATH = MODELS_DIR / YOLO_MODEL_NAME

# Optional lightweight alternative (only when selected or as fallback)
YOLO_FALLBACK_MODEL_NAME = "yolo26n.pt"
YOLO_FALLBACK_MODEL_PATH = MODELS_DIR / YOLO_FALLBACK_MODEL_NAME

# Supported lightweight models
AVAILABLE_MODELS: List[str] = [
    "yolov8n.pt",
    "yolo26n.pt",
]

# Hardware device preference: "auto", "cuda", "cpu"
DEVICE_PREFERENCE = "auto"

# Hardware constraints for RTX 4050 (6 GB VRAM)
GPU_VRAM_TARGET_MB = 6144
GPU_VRAM_WARNING_MB = 5200

# Half-precision FP16 for CUDA acceleration
USE_HALF_PRECISION = True

# Inference configuration
YOLO_INPUT_SIZE = 640                # Default resolution (640), with 480 / 384 performance modes
SUPPORTED_INPUT_SIZES = [384, 480, 640]
CONFIDENCE_THRESHOLD = 0.25          # Configurable detection confidence threshold
NMS_IOU_THRESHOLD = 0.45             # Non-Maximum Suppression IoU threshold
DUPLICATE_CLASS_IOU_THRESHOLD = 0.72 # Suppress near-duplicate same-class boxes

# ==============================================================================
# 3. Road User Classes & Mapping (Pretrained COCO + Custom-Model Ready)
# ==============================================================================
# Exact verified COCO class indices for road users:
# 0: person, 1: bicycle, 2: car, 3: motorcycle (includes motor scooters in COCO),
# 5: bus, 7: truck.
# Note: Pretrained COCO does not contain distinct 'scooter' or 'three_wheeler' /
# 'auto_rickshaw' classes. Custom-trained weights exposing those names are mapped
# automatically via ROAD_USER_ALIASES without misclassifying them as cars.
COCO_CLASS_MAP: Dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Canonical road-user aliases for dynamic model.names inspection (COCO + custom models)
ROAD_USER_ALIASES: Dict[str, str] = {
    "person": "person",
    "pedestrian": "person",
    "bicycle": "bicycle",
    "bike": "bicycle",
    "cycle": "bicycle",
    "car": "car",
    "sedan": "car",
    "suv": "car",
    "van": "car",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "scooter": "scooter",
    "moped": "scooter",
    "bus": "bus",
    "truck": "truck",
    "lorry": "truck",
    "auto_rickshaw": "auto_rickshaw",
    "autorickshaw": "auto_rickshaw",
    "auto-rickshaw": "auto_rickshaw",
    "rickshaw": "auto_rickshaw",
    "three_wheeler": "three_wheeler",
    "three-wheeler": "three_wheeler",
    "3wheeler": "three_wheeler",
    "tuk-tuk": "auto_rickshaw",
    "tuktuk": "auto_rickshaw",
}

VEHICLE_CLASSES: Set[str] = {
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "scooter",
    "auto_rickshaw",
    "three_wheeler",
}
ROAD_USER_CLASSES: Set[str] = VEHICLE_CLASSES | {"person"}
TRACKED_CLASSES: Set[str] = set(ROAD_USER_CLASSES)

# ==============================================================================
# 4. Multi-Object Tracking Configuration (ByteTrack / Internal Fallback)
# ==============================================================================
TRACKER_TYPE = "bytetrack"
TRACKER_CONFIG_PATH = "bytetrack.yaml"   # Ultralytics built-in ByteTrack config

MAX_TRACK_AGE = 30            # Frames to retain track during temporary occlusion
MIN_HITS = 2                  # Consecutive detections before confirming track
IOU_TRACKER_THRESHOLD = 0.25  # Spatial association IoU threshold
MAX_CENTROID_DISTANCE = 140.0 # Pixel radius for velocity-guided centroid association

# ==============================================================================
# 5. Speed & Perspective Calibration
# ==============================================================================
# Pixel displacement is NOT converted to km/h unless explicit camera calibration
# (PIXELS_PER_METER) is provided.
PIXELS_PER_METER: Optional[float] = None
DEFAULT_CAMERA_FPS = 30.0

def get_speed_unit_label() -> str:
    """Returns 'km/h' if calibration is active, otherwise 'px/s'."""
    return "km/h" if PIXELS_PER_METER is not None else "px/s"

# ==============================================================================
# 6. Accident Analysis & 4-Tier Temporal Classification
# ==============================================================================
TIER_NORMAL = "NORMAL_TRAFFIC"
TIER_RISK = "COLLISION_RISK"
TIER_SUSPECTED = "SUSPECTED_ACCIDENT"
TIER_CONFIRMED = "CONFIRMED_ACCIDENT"

# Risk score thresholds (0-100)
ACCIDENT_RISK_THRESHOLD = 55.0       # Base threshold for raising alert
RISK_THRESHOLD_CAUTION = 35.0        # Potential collision risk / caution
RISK_THRESHOLD_SUSPECTED = 60.0      # Suspected accident
RISK_THRESHOLD_CONFIRMED = 80.0      # Confirmed accident (requires temporal confirmation)

# Temporal persistence: An event is confirmed only if sustained over multiple frames
MIN_TEMPORAL_CONFIRM_FRAMES = 3      # Consecutive anomaly frames required for confirmed status
ALERT_COOLDOWN_FRAMES = 90           # Minimum frames between separate incident logs

# Physical and Kinematic Heuristic Thresholds
COLLISION_IOU_THRESHOLD = 0.12       # Spatial overlap threshold
PROXIMITY_DISTANCE_PX = 85.0         # Proximity hazard radius in pixels
SUDDEN_DECEL_THRESHOLD_PX = 10.0     # Deceleration shockwave threshold (px/frame)
TRAJECTORY_ANGLE_DEV_DEG = 45.0      # Trajectory deflection angle threshold (degrees)
POST_IMPACT_STATIONARY_SPEED = 1.5   # Maximum speed for stopped condition

# Physics Risk Score Weights (Sum = 1.0)
WEIGHT_IOU_OVERLAP = 0.30
WEIGHT_SUDDEN_DECEL = 0.30
WEIGHT_TRAJECTORY_DEVIATION = 0.25
WEIGHT_POST_STOPPAGE = 0.15

# ==============================================================================
# 7. Web Server & Storage Configuration
# ==============================================================================
FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5000
FLASK_DEBUG = False
MAX_CONTENT_LENGTH = 128 * 1024 * 1024  # 128 MB max upload size
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
