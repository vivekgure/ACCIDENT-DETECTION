"""
AURA-VISION Flask Web Server & Real-Time Video Streaming Engine.
Orchestrates webcam capture, video upload validation & playback, sample scenarios,
lightweight YOLOv8n inference with ByteTrack tracking, 4-tier accident analysis,
MJPEG streaming, telemetry APIs, and JSON/CSV incident exports.
"""

import csv
import gc
import io
import json
import threading
import time
from typing import Optional, Dict, Any, List, Tuple

import cv2
import numpy as np
import torch
from flask import Flask, render_template, Response, request, jsonify, send_from_directory, make_response
from werkzeug.utils import secure_filename

import config
from detector import VehicleDetector, BGR_CHARCOAL, BGR_LIME, BGR_TEXT_LIGHT, BGR_TEXT_MUTED
from accident_detector import AccidentDetector
from generate_sample_video import generate_all_samples

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(config.UPLOADS_DIR)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH


class VideoEngine:
    """
    Thread-safe background video processing engine.
    Decouples single-stream YOLO inference from MJPEG HTTP clients so multiple
    viewers or stream reconnects never duplicate model execution or leak memory.
    """

    def __init__(self):
        self.state_lock = threading.Lock()
        self.cap_lock = threading.Lock()
        self.frame_condition = threading.Condition()
        self.run_event = threading.Event()

        print("[VideoEngine] Initializing VehicleDetector...")
        self.detector = VehicleDetector(model_name=config.YOLO_MODEL_NAME)
        self.accident_detector = AccidentDetector(risk_threshold=config.ACCIDENT_RISK_THRESHOLD)

        # Stream control states
        self.is_running = False
        self.source_type = "sample_accident"  # "webcam", "upload", "sample_normal", "sample_accident"
        self.video_path: Optional[str] = str(config.SAMPLE_DIR / "accident_scenario.mp4")
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_idx = 0
        self.last_error: Optional[str] = None

        # Telemetry & Performance Metrics
        self.fps = 0.0
        self.last_frame_time = time.perf_counter()
        self.inference_latency_ms = 0.0
        self.active_vehicles_count = 0
        self.total_detections_count = 0
        self.class_counts: Dict[str, int] = {}
        self.current_risk_score = 0.0
        self.current_risk_level = config.TIER_NORMAL
        self.current_status_text = "Normal Traffic Flow"
        self.latest_alert: Optional[dict] = None
        self.active_tracks_summary: List[dict] = []
        self.current_frame: Optional[np.ndarray] = None
        self.latest_jpeg: Optional[bytes] = None
        self.frame_seq = 0
        self._last_vram_poll = 0.0

        # Pre-render standby JPEG
        self.standby_jpeg = self._encode_standby_jpeg()

        # Start single daemon processing worker
        self._shutdown = False
        self._worker = threading.Thread(target=self._processing_loop, name="AuraVisionWorker", daemon=True)
        self._worker.start()

    def start(self, source_type: Optional[str] = None, video_path: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """Starts or switches video streaming source cleanly."""
        with self.cap_lock:
            if source_type:
                self.source_type = source_type
            if video_path is not None:
                self.video_path = video_path

            ok, err = self._open_capture()
            if not ok:
                with self.state_lock:
                    self.is_running = False
                    self.last_error = err
                    self.standby_jpeg = self._encode_standby_jpeg(error_msg=err)
                self.run_event.clear()
                with self.frame_condition:
                    self.frame_condition.notify_all()
                return False, err

            with self.state_lock:
                self.last_error = None
                self.is_running = True
                self.fps = 0.0
                self.last_frame_time = time.perf_counter()
                self.active_vehicles_count = 0
                self.class_counts = {}
                self.current_risk_score = 0.0
                self.current_risk_level = config.TIER_NORMAL
                self.current_status_text = "Normal Traffic Flow"
                self.latest_alert = None
                self.active_tracks_summary = []
                self.detector.reset()
                self.accident_detector.reset_temporal_state()
                self.standby_jpeg = self._encode_standby_jpeg()

            self.run_event.set()
            print(f"[VideoEngine] Started stream with source: {self.source_type}")
            return True, None

    def stop(self):
        """Pauses video processing and releases active capture resources."""
        self.run_event.clear()
        with self.state_lock:
            self.is_running = False
            self.fps = 0.0
            self.standby_jpeg = self._encode_standby_jpeg(error_msg=self.last_error)

        with self.cap_lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

        with self.frame_condition:
            self.frame_condition.notify_all()
        print("[VideoEngine] Stream stopped and capture resources released.")

    def set_risk_threshold(self, threshold: float):
        """Adjusts accident detection sensitivity threshold."""
        with self.state_lock:
            self.accident_detector.risk_threshold = threshold
            print(f"[VideoEngine] Updated risk threshold to: {threshold}%")

    def switch_model(self, model_name: str) -> bool:
        """Dynamically switches active lightweight YOLO weights."""
        was_running = self.is_running
        if was_running:
            self.run_event.clear()

        with self.cap_lock:
            success = self.detector.set_model(model_name)
            with self.state_lock:
                self.standby_jpeg = self._encode_standby_jpeg(error_msg=self.last_error)

        if was_running:
            self.run_event.set()
        return success

    def _open_capture(self) -> Tuple[bool, Optional[str]]:
        """Safely opens webcam, uploaded video, or sample video."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.frame_idx = 0

        if self.source_type == "webcam":
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                cap.release()
                msg = "Webcam unavailable (camera index 0 not found or in use by another app)."
                print(f"[VideoEngine] Error: {msg}")
                return False, msg

            # Verify camera can deliver a frame
            ret, _ = cap.read()
            if not ret:
                cap.release()
                msg = "Webcam opened but failed to return video frames. Check camera permissions."
                print(f"[VideoEngine] Error: {msg}")
                return False, msg

            self.cap = cap
            return True, None

        if self.source_type == "sample_normal":
            target = config.SAMPLE_DIR / "normal_traffic.mp4"
            if not target.exists():
                generate_all_samples()
            self.video_path = str(target)
        elif self.source_type == "sample_accident":
            target = config.SAMPLE_DIR / "accident_scenario.mp4"
            if not target.exists():
                generate_all_samples()
            self.video_path = str(target)
        elif self.source_type == "upload" and self.video_path:
            target = self.video_path
        else:
            target = config.SAMPLE_DIR / "accident_scenario.mp4"
            self.video_path = str(target)

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            cap.release()
            msg = f"Unable to open video source: {self.video_path}"
            print(f"[VideoEngine] Error: {msg}")
            return False, msg

        self.cap = cap
        return True, None

    def _resize_preserving_aspect(self, frame: np.ndarray, max_w: int = 960, max_h: int = 540) -> np.ndarray:
        """Downscales oversized frames while preserving aspect ratio."""
        h, w = frame.shape[:2]
        if w <= max_w and h <= max_h:
            return frame
        scale = min(max_w / float(w), max_h / float(h))
        new_w = max(2, int(round(w * scale)) & ~1)
        new_h = max(2, int(round(h * scale)) & ~1)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _processing_loop(self):
        """Background worker executing single-pipeline inference & frame encoding."""
        while not self._shutdown:
            # Sleep cleanly with zero CPU/GPU load when stopped
            if not self.run_event.wait(timeout=0.5):
                continue

            frame = None
            source_type = self.source_type

            with self.cap_lock:
                if not self.is_running or self.cap is None or not self.cap.isOpened():
                    time.sleep(0.05)
                    continue

                ret, frame = self.cap.read()
                if not ret:
                    if source_type != "webcam":
                        # Loop sample/uploaded video cleanly and reset temporal trackers
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.frame_idx = 0
                        self.detector.reset()
                        self.accident_detector.reset_temporal_state()
                        ret, frame = self.cap.read()
                    else:
                        # Webcam disconnected mid-stream
                        err = "Webcam stream interrupted or disconnected."
                        with self.state_lock:
                            self.is_running = False
                            self.last_error = err
                            self.standby_jpeg = self._encode_standby_jpeg(error_msg=err)
                        self.run_event.clear()
                        with self.frame_condition:
                            self.frame_condition.notify_all()
                        continue

                if not ret or frame is None:
                    time.sleep(0.05)
                    continue

                self.frame_idx += 1
                curr_frame_idx = self.frame_idx

            frame = self._resize_preserving_aspect(frame)

            now = time.perf_counter()
            dt = now - self.last_frame_time
            self.last_frame_time = now
            if dt > 0:
                inst_fps = 1.0 / dt
                self.fps = (0.85 * self.fps + 0.15 * inst_fps) if self.fps > 0 else inst_fps

            effective_fps = max(5.0, self.fps if self.fps > 0 else config.DEFAULT_CAMERA_FPS)

            # 1. Lightweight YOLO Detection + ByteTrack Tracking
            detections, active_tracks = self.detector.detect_and_track(frame, fps=effective_fps)
            inf_ms = self.detector.last_inference_time_ms

            # 2. Pre-render clean automotive annotations
            annotated_frame = self.detector.annotate_frame(
                frame, active_tracks, self.accident_detector.active_alert
            )

            # 3. 4-Tier Multi-Frame Accident Analysis
            is_alert, alert_info = self.accident_detector.analyze_frame(
                frame, active_tracks, curr_frame_idx, annotated_frame=annotated_frame
            )

            if is_alert:
                annotated_frame = self.detector.annotate_frame(frame, active_tracks, alert_info)

            # Encode JPEG once for all MJPEG consumers
            ok_enc, buffer = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            frame_bytes = buffer.tobytes() if ok_enc else None

            # Update telemetry snapshot
            confirmed_tracks = [
                (tid, st) for tid, st in active_tracks.items() if st.is_confirmed
            ]
            cls_counts: Dict[str, int] = {}
            tracks_summary: List[dict] = []
            hazard_ids = alert_info.get("hazard_ids", []) if alert_info else []

            for tid, st in confirmed_tracks:
                cls_counts[st.class_name] = cls_counts.get(st.class_name, 0) + 1
                tracks_summary.append({
                    "id": tid,
                    "class": st.class_name,
                    "confidence": round(st.confidence * 100, 1),
                    "speed": round(st.smoothed_speed, 1),
                    "speed_display": st.get_speed_display_text(),
                    "heading": round(st.current_heading_deg, 0),
                    "status": "Hazard" if tid in hazard_ids else st.motion_status,
                })

            with self.state_lock:
                self.current_frame = annotated_frame
                if frame_bytes is not None:
                    self.latest_jpeg = frame_bytes
                self.inference_latency_ms = inf_ms
                self.active_vehicles_count = len(confirmed_tracks)
                self.total_detections_count = self.detector.tracker.total_unique_vehicles
                self.class_counts = cls_counts
                self.current_risk_score = alert_info.get("risk_score", 0.0) if alert_info else 0.0
                self.current_risk_level = alert_info.get("risk_level", config.TIER_NORMAL) if alert_info else config.TIER_NORMAL
                self.current_status_text = alert_info.get("status_text", "Normal Traffic Flow") if alert_info else "Normal Traffic Flow"
                self.latest_alert = alert_info if (alert_info and alert_info.get("is_alert")) else None
                self.active_tracks_summary = tracks_summary
                self.frame_seq += 1

            with self.frame_condition:
                self.frame_condition.notify_all()

            # Periodic memory hygiene for continuous streaming
            if curr_frame_idx % 250 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Pace recorded video playback smoothly to ~24-30 FPS when inference is very fast
            if source_type != "webcam" and inf_ms < 25.0:
                time.sleep(max(0.0, 0.030 - (inf_ms / 1000.0)))

    def generate_frames(self):
        """Yields latest encoded MJPEG frames to HTTP clients without running duplicate inference."""
        last_seq = -1
        while True:
            if not self.is_running:
                frame_bytes = self.standby_jpeg
                time.sleep(0.12)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
                continue

            with self.frame_condition:
                if self.frame_seq == last_seq:
                    self.frame_condition.wait(timeout=0.25)
                last_seq = self.frame_seq
                frame_bytes = self.latest_jpeg or self.standby_jpeg

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

    def _encode_standby_jpeg(self, error_msg: Optional[str] = None) -> bytes:
        """Generates a minimal dark-charcoal standby monitor frame."""
        img = np.zeros((540, 960, 3), dtype=np.uint8)
        img[:] = (41, 42, 37)  # #252A29 in BGR

        # Subtle inner frame
        cv2.rectangle(img, (24, 24), (936, 516), (55, 60, 58), 1)

        cx, cy = 480, 255
        # Minimal optical reticle
        cv2.circle(img, (cx, cy - 28), 32, BGR_LIME, 1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy - 28), 3, BGR_LIME, -1, cv2.LINE_AA)
        cv2.line(img, (cx - 46, cy - 28), (cx - 14, cy - 28), BGR_LIME, 1)
        cv2.line(img, (cx + 14, cy - 28), (cx + 46, cy - 28), BGR_LIME, 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        title = "AURA-VISION  //  MONITOR STANDBY"
        (tw, _), _ = cv2.getTextSize(title, font, 0.62, 1)
        cv2.putText(img, title, ((960 - tw) // 2, cy + 36), font, 0.62, BGR_TEXT_LIGHT, 1, cv2.LINE_AA)

        model_tag = self.detector.model_name.replace(".pt", "").upper()
        dev_tag = "CUDA" if self.detector.device.startswith("cuda") else "CPU"
        sub = f"Model: {model_tag}  |  Compute: {dev_tag}  |  Resolution: {self.detector.imgsz}px"
        (sw, _), _ = cv2.getTextSize(sub, font, 0.44, 1)
        cv2.putText(img, sub, ((960 - sw) // 2, cy + 68), font, 0.44, BGR_TEXT_MUTED, 1, cv2.LINE_AA)

        hint = error_msg if error_msg else "Select an input source or click Start Detection to begin monitoring"
        if len(hint) > 84:
            hint = hint[:81] + "..."
        (hw, _), _ = cv2.getTextSize(hint, font, 0.42, 1)
        hint_color = (82, 82, 224) if error_msg else BGR_LIME
        cv2.putText(img, hint, ((960 - hw) // 2, cy + 102), font, 0.42, hint_color, 1, cv2.LINE_AA)

        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        return buf.tobytes()

    def get_vram_cached(self) -> Dict[str, Any]:
        """Refreshes NVML VRAM metrics at most once every 1.5 seconds."""
        now = time.perf_counter()
        if now - self._last_vram_poll > 1.5:
            self._last_vram_poll = now
            return self.detector.refresh_vram_info()
        return self.detector.vram_info


# Ensure sample test videos exist before starting engine
if not (config.SAMPLE_DIR / "accident_scenario.mp4").exists():
    generate_all_samples()

engine = VideoEngine()


# ==============================================================================
# Flask Routes & REST API Endpoints
# ==============================================================================

@app.route("/")
def index():
    """Renders main AURA-VISION platform."""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """MJPEG surveillance stream endpoint."""
    return Response(
        engine.generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/start_detection", methods=["POST"])
def start_detection():
    """Starts video processing."""
    data = request.get_json(silent=True) or {}
    source = data.get("source", engine.source_type)
    video_path = data.get("video_path", engine.video_path)
    ok, err = engine.start(source_type=source, video_path=video_path)
    if not ok:
        return jsonify({"success": False, "error": err or "Failed to start video stream"}), 400
    return jsonify({"success": True, "status": "running", "source": engine.source_type})


@app.route("/api/stop_detection", methods=["POST"])
def stop_detection():
    """Stops video processing."""
    engine.stop()
    return jsonify({"success": True, "status": "stopped"})


@app.route("/api/switch_source", methods=["POST"])
def switch_source():
    """Switches active source (webcam, upload, sample_normal, sample_accident)."""
    data = request.get_json(silent=True) or {}
    source = data.get("source", "sample_accident")
    video_path = data.get("video_path", None)
    ok, err = engine.start(source_type=source, video_path=video_path)
    if not ok:
        return jsonify({"success": False, "error": err or f"Failed to open source '{source}'"}), 400
    return jsonify({"success": True, "source": engine.source_type})


@app.route("/api/upload_video", methods=["POST"])
def upload_video():
    """Validates, saves, and processes an uploaded video file."""
    if "video" not in request.files:
        return jsonify({"success": False, "error": "No video file provided"}), 400

    file = request.files["video"]
    if not file.filename:
        return jsonify({"success": False, "error": "Empty filename"}), 400

    if "." not in file.filename:
        return jsonify({"success": False, "error": "Uploaded file has no extension"}), 400

    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        return jsonify({
            "success": False,
            "error": f"Unsupported format .{ext}. Allowed: {', '.join(sorted(config.ALLOWED_EXTENSIONS))}",
        }), 400

    filename = secure_filename(f"upload_{int(time.time())}_{file.filename}")
    save_path = config.UPLOADS_DIR / filename
    try:
        file.save(str(save_path))
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to save upload: {e}"}), 500

    # Verify uploaded video is readable by OpenCV
    probe = cv2.VideoCapture(str(save_path))
    is_valid = probe.isOpened()
    if is_valid:
        ret, _ = probe.read()
        is_valid = bool(ret)
    probe.release()

    if not is_valid:
        try:
            save_path.unlink(missing_ok=True)
        except Exception:
            pass
        return jsonify({
            "success": False,
            "error": "Uploaded file is corrupted or uses an unsupported video codec.",
        }), 400

    print(f"[Server] Validated and saved uploaded video to {save_path}")
    ok, err = engine.start(source_type="upload", video_path=str(save_path))
    if not ok:
        return jsonify({"success": False, "error": err or "Failed to start uploaded video"}), 400

    return jsonify({
        "success": True,
        "filename": filename,
        "video_path": str(save_path),
        "source": "upload",
    })


@app.route("/api/set_threshold", methods=["POST"])
def set_threshold():
    """Sets risk sensitivity threshold."""
    data = request.get_json(silent=True) or {}
    try:
        threshold = float(data.get("threshold", config.ACCIDENT_RISK_THRESHOLD))
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "Invalid threshold value"}), 400
    threshold = max(10.0, min(95.0, threshold))
    engine.set_risk_threshold(threshold)
    return jsonify({"success": True, "threshold": threshold})


@app.route("/api/set_model", methods=["POST"])
def set_model():
    """Dynamically switches active lightweight YOLO weights."""
    data = request.get_json(silent=True) or {}
    model_name = data.get("model_name", config.YOLO_MODEL_NAME)
    if model_name not in config.AVAILABLE_MODELS and not (config.MODELS_DIR / model_name).exists():
        return jsonify({"success": False, "error": f"Unsupported model: {model_name}"}), 400

    success = engine.switch_model(model_name)
    if success:
        return jsonify({
            "success": True,
            "model_name": engine.detector.model_name,
            "supported_classes": engine.detector.supported_class_names,
        })
    return jsonify({"success": False, "error": f"Failed to load model {model_name}"}), 500


@app.route("/api/set_inference_params", methods=["POST"])
def set_inference_params():
    """Configures input resolution, confidence, and IoU thresholds."""
    data = request.get_json(silent=True) or {}
    imgsz = data.get("imgsz")
    conf = data.get("conf")
    iou = data.get("iou")
    classes = data.get("classes")

    engine.detector.set_inference_settings(
        imgsz=int(imgsz) if imgsz is not None else None,
        conf=float(conf) if conf is not None else None,
        iou=float(iou) if iou is not None else None,
        classes=classes if classes is not None else None,
    )
    return jsonify({
        "success": True,
        "imgsz": engine.detector.imgsz,
        "conf": engine.detector.conf_threshold,
        "iou": engine.detector.iou_threshold,
        "classes": list(engine.detector.tracked_classes),
    })


@app.route("/api/stats")
def get_stats():
    """Returns real-time telemetry metrics, hardware state, and alert status."""
    vram = engine.get_vram_cached()
    events = engine.accident_detector.get_events()
    with engine.state_lock:
        return jsonify({
            "is_running": engine.is_running,
            "source": engine.source_type,
            "fps": round(engine.fps, 1),
            "latency_ms": round(engine.inference_latency_ms, 1),
            "active_vehicles": engine.active_vehicles_count,
            "total_detections": engine.total_detections_count,
            "class_counts": dict(engine.class_counts),
            "accidents_count": len(events),
            "latest_incident": events[0] if events else None,
            "current_risk_score": round(engine.current_risk_score, 1),
            "risk_level": engine.current_risk_level,
            "status_text": engine.current_status_text,
            "risk_threshold": engine.accident_detector.risk_threshold,
            "conf_threshold": engine.detector.conf_threshold,
            "iou_threshold": engine.detector.iou_threshold,
            "model_name": engine.detector.model_name,
            "available_models": config.AVAILABLE_MODELS,
            "supported_classes": engine.detector.supported_class_names,
            "imgsz": engine.detector.imgsz,
            "device": engine.detector.device_name,
            "device_code": engine.detector.device,
            "vram": vram,
            "tracker_type": engine.detector.tracker_type,
            "speed_unit": config.get_speed_unit_label(),
            "consecutive_anomaly_frames": engine.accident_detector.consecutive_anomaly_frames,
            "temporal_confirm_required": config.MIN_TEMPORAL_CONFIRM_FRAMES,
            "alert": engine.latest_alert,
            "active_tracks": list(engine.active_tracks_summary),
            "error": engine.last_error,
        })


@app.route("/api/events")
def get_events():
    """Returns all recorded accident events with snapshot URLs."""
    events = engine.accident_detector.get_events()
    return jsonify({"events": events, "total": len(events)})


@app.route("/api/clear_history", methods=["POST"])
def clear_history():
    """Clears all accident logs and snapshots."""
    engine.accident_detector.clear_events()
    return jsonify({"success": True})


@app.route("/api/capture_snapshot", methods=["POST"])
def capture_snapshot():
    """Manually captures current frame as snapshot."""
    with engine.state_lock:
        frame = engine.current_frame.copy() if engine.current_frame is not None else None
    if frame is not None:
        filename = f"manual_snap_{int(time.time())}.jpg"
        save_path = config.EVENTS_DIR / filename
        cv2.imwrite(str(save_path), frame)
        return jsonify({"success": True, "snapshot_url": f"/accident_events/{filename}"})
    return jsonify({"success": False, "error": "No active frame available to capture."}), 400


@app.route("/api/export_events")
def export_events():
    """Exports accident incidents log in JSON or CSV format."""
    fmt = request.args.get("format", "json").lower()
    events = engine.accident_detector.get_events()

    if fmt == "csv":
        si = io.StringIO()
        cw = csv.writer(si)
        cw.writerow([
            "Event ID",
            "Timestamp",
            "Risk Score (%)",
            "Status",
            "Reason",
            "Vehicles Count",
            "Vehicles Involved",
            "Snapshot URL",
        ])
        for e in events:
            vehs = "; ".join(
                [f"#{v.get('id')}:{v.get('class')}({v.get('speed')}px/f)" for v in e.get("vehicles_involved", [])]
            )
            cw.writerow([
                e.get("event_id"),
                e.get("timestamp"),
                e.get("risk_score"),
                e.get("status"),
                e.get("reason"),
                e.get("vehicle_count"),
                vehs,
                e.get("snapshot_url"),
            ])
        output = make_response(si.getvalue())
        output.headers["Content-Disposition"] = "attachment; filename=accident_events_export.csv"
        output.headers["Content-type"] = "text/csv"
        return output

    output = make_response(json.dumps(events, indent=2))
    output.headers["Content-Disposition"] = "attachment; filename=accident_events_export.json"
    output.headers["Content-type"] = "application/json"
    return output


@app.route("/api/benchmark", methods=["POST"])
def run_benchmark():
    """Runs on-demand inference benchmark across supported resolutions."""
    data = request.get_json(silent=True) or {}
    model_name = data.get("model_name", engine.detector.model_name)
    results = {}

    try:
        from ultralytics import YOLO

        reuse_active = (model_name == engine.detector.model_name and engine.detector.model is not None)
        if reuse_active:
            m = engine.detector.model
        else:
            target_path = config.MODELS_DIR / model_name
            m = YOLO(str(target_path) if target_path.exists() else model_name)

        device = engine.detector.device
        half = device.startswith("cuda") and config.USE_HALF_PRECISION
        half_kwargs = {"half": True} if half else {}
        dummy = np.zeros((540, 960, 3), dtype=np.uint8)

        with torch.inference_mode():
            for s in config.SUPPORTED_INPUT_SIZES:
                _ = m.predict(dummy, imgsz=s, device=device, verbose=False, **half_kwargs)
                times = []
                for _ in range(5):
                    t0 = time.perf_counter()
                    _ = m.predict(dummy, imgsz=s, device=device, verbose=False, **half_kwargs)
                    times.append(time.perf_counter() - t0)
                avg_ms = float(np.mean(times) * 1000.0)
                fps = float(1000.0 / avg_ms) if avg_ms > 0 else 0.0
                results[f"imgsz_{s}"] = {
                    "latency_ms": round(avg_ms, 1),
                    "fps": round(fps, 1),
                }

        if not reuse_active:
            del m
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return jsonify({
            "success": True,
            "model": model_name,
            "device": engine.detector.device_name,
            "benchmarks": results,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/accident_events/<path:filename>")
def serve_event_image(filename):
    """Serves saved event snapshot images."""
    return send_from_directory(str(config.EVENTS_DIR), filename)


@app.route("/sample_data/<path:filename>")
def serve_sample_video(filename):
    """Serves sample videos."""
    return send_from_directory(str(config.SAMPLE_DIR), filename)


if __name__ == "__main__":
    print("=" * 65)
    print("   AURA-VISION // AI VEHICLE ACCIDENT DETECTION SYSTEM")
    print(f"   Model: {config.YOLO_MODEL_NAME} | Device: {engine.detector.device_name}")
    print(f"   Listening on http://{config.FLASK_HOST}:{config.FLASK_PORT}")
    print("=" * 65)
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.FLASK_DEBUG, threaded=True)
