# AURA-VISION // AI Vehicle Accident Detection & Road Safety Platform

A lightweight, real-time **AI Vehicle Accident Detection & Traffic Surveillance Platform** powered by **Ultralytics YOLOv8n (`yolov8n.pt`)**, **ByteTrack Multi-Object Tracking**, **Multi-Frame Kinematic Collision Analysis**, and a minimal automotive dark-charcoal & lime-green Flask web interface.

---

## Key Capabilities

### 1. Lightweight Real-Time Object Detection (`YOLOv8n`)
- **Primary Model:** `yolov8n.pt` (6.2 MB) — selected over oversized models (`yolo26x.pt`) and lower-recall alternatives (`yolo26n.pt`) after empirical testing.
- **Optional Alternative:** `yolo26n.pt` (5.3 MB) selectable dynamically from the UI or `/api/set_model`.
- **Hardware Acceleration & Fallback:** Automatically uses **NVIDIA CUDA (FP16)** when CUDA-enabled PyTorch is installed (e.g., on NVIDIA RTX 4050 6 GB VRAM) and falls back cleanly to **CPU** when CUDA is unavailable or VRAM is constrained.
- **Resolution Modes:** `640 px` default, with `480 px` and `384 px` performance modes and configurable detection confidence threshold (`0.25` default).
- **Resilient Weight Caching:** Automatically downloads missing weights into `models/` and verifies archive integrity before loading.

### 2. Road-User Detection & Custom Class Readiness
- **Pretrained COCO Classes:** Accurately maps COCO indices `0: person`, `1: bicycle`, `2: car`, `3: motorcycle` (covers motorcycles & motor scooters in COCO), `5: bus`, and `7: truck`.
- **Custom-Model Ready:** `VehicleDetector` dynamically inspects `model.names` at load time so custom-trained weights with dedicated `scooter`, `auto_rickshaw`, or `three_wheeler` classes are tracked as distinct categories without misclassifying auto-rickshaws as cars.
- **Duplicate Suppression:** Class-aware post-NMS filtering suppresses duplicate same-class detections while preserving overlapping collision pairs and distinct road-user categories.

### 3. Preserved 4-Tier Multi-Frame Accident Detection Engine
Object detection is decoupled from collision classification—overlapping bounding boxes alone do not trigger an accident alert without kinematic impact evidence (deceleration shockwave, trajectory deflection, or post-impact stoppage):
1. `NORMAL_TRAFFIC` (Risk < 35%)
2. `COLLISION_RISK` (Risk 35% – 59%)
3. `SUSPECTED_ACCIDENT` (Risk 60% – 79% or initial impact frame)
4. `CONFIRMED_ACCIDENT` (Risk >= 80% sustained across >= 3 consecutive frames)

Includes persistent ByteTrack tracking IDs, automatic accident snapshots (`accident_events/`), incident history (`events_log.json`), and JSON/CSV exports.

---

## Measured Empirical Benchmark

Tested on `sample_data/accident_scenario.mp4` (130 frames, 960x540) using the local `.venv` CPU execution path (`torch 2.13.0+cpu`):

| Model | Input Size | Mean Latency | Measured FPS | Detections (15 frames) |
|---|---|---|---|---|
| `yolov8n.pt` (Primary) | `640 px` | `39.7 ms` | `25.2 FPS` | `24` |
| `yolov8n.pt` (Fast) | `480 px` | `28.2 ms` | `35.5 FPS` | `30` |
| `yolov8n.pt` (Performance) | `384 px` | `22.5 ms` | `44.5 FPS` | `38` |
| `yolo26n.pt` (Alternative) | `640 px` | `41.9 ms` | `23.8 FPS` | `4` |

*(Note: Enabling CUDA PyTorch on an NVIDIA RTX 4050 GPU will yield even lower latency via FP16 inference.)*

---

## Project Structure

```text
accident/
├── app.py                     # Flask server, background single-stream VideoEngine, MJPEG & REST APIs
├── config.py                  # Paths, YOLOv8n defaults, COCO/custom class maps, 4-tier thresholds
├── detector.py                # Lightweight YOLO wrapper, integrity check, deduplication & clean HUD
├── tracker.py                 # VehicleState kinematics (speed, deceleration, heading, trails)
├── accident_detector.py       # 4-tier collision physics & multi-frame temporal confirmation engine
├── check_env.py               # Dependency, CUDA/NVML VRAM, and model integrity verification
├── test_pipeline.py           # Automated test & empirical benchmark suite
├── test_full_frame.py         # Offline CLI video processor & annotated MP4 exporter
├── train_model.py             # Custom vehicle & accident dataset scaffold (YOLOv8n base)
├── generate_sample_video.py   # Generates sample normal & collision simulation videos
├── requirements.txt           # Python dependencies
├── run.bat                    # Windows one-click launcher (prefers .venv)
├── models/                    # Cached weights (yolov8n.pt, yolo26n.pt)
├── accident_events/           # Saved incident snapshots & events_log.json
├── sample_data/               # Sample traffic & collision MP4 files
├── static/
│   ├── css/style.css          # Dark charcoal (#252A29/#191D1C) & lime (#B7D83B) UI theme
│   └── js/dashboard.js        # Navigation, telemetry polling, native dialog, exports
└── templates/
    └── index.html             # Landing overview, live monitoring dashboard & incident history
```

---

## Installation & Launch Guide

### 1. Environment Setup
From the project root (`c:\Users\vivek\accident`):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

*(Optional — Enable NVIDIA CUDA on RTX 4050):* If your environment currently has a CPU-only PyTorch wheel, install the CUDA-enabled PyTorch wheel matching your NVIDIA driver (`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`).

### 2. Verify System Diagnostics & Tests
```powershell
.venv\Scripts\python.exe check_env.py
.venv\Scripts\python.exe test_pipeline.py
```

### 3. Launch Web Application
- **Option A:** Double-click `run.bat`
- **Option B:** Run from terminal:
  ```powershell
  .venv\Scripts\python.exe app.py
  ```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

### 4. Offline Video Processing (CLI)
```powershell
.venv\Scripts\python.exe test_full_frame.py --video sample_data/accident_scenario.mp4 --model yolov8n.pt --imgsz 640
```
