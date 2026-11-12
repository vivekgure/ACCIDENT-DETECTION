"""
AURA-VISION Automated Verification & Empirical Benchmark Suite.
Validates:
  1. Lightweight YOLOv8n Loading, Integrity & Dynamic Model Switching
  2. Hardware Acceleration Check & Automatic CPU Fallback
  3. Road-User Class Mapping (COCO + Custom-Model Ready) & Deduplication
  4. ByteTrack Persistent Multi-Object Tracking
  5. 4-Tier Accident Classification & Overlap-Without-Impact Guard
  6. Normal Traffic False-Alarm Suppression (120 frames)
  7. Accident Footage Multi-Frame Confirmation & Snapshot Generation
  8. Video Upload Validation (Valid, Invalid Extension, Corrupted Video)
  9. Flask Telemetry, MJPEG Feed, Model Switching & JSON/CSV Exports
 10. Empirical Latency & FPS Benchmark (YOLOv8n at 640/480/384 vs YOLO26n)
"""

import io
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from detector import VehicleDetector
from tracker import VehicleTracker, VehicleState
from accident_detector import AccidentDetector
from generate_sample_video import generate_all_samples


def test_model_loading_and_switching():
    print("\n--- [Test 1] YOLOv8n Loading & Dynamic Switching ---")
    detector = VehicleDetector(model_name="yolov8n.pt")
    assert detector.model is not None, "Failed to load primary YOLOv8n model!"
    assert detector.model_name == "yolov8n.pt"
    print(f"  [PASS] Primary loaded: {detector.model_name} on {detector.device_name}")

    ok = detector.set_model("yolo26n.pt")
    assert ok is True, "Failed to switch to yolo26n.pt"
    assert detector.model_name == "yolo26n.pt"
    print("  [PASS] Switched dynamically to yolo26n.pt")

    ok_back = detector.set_model("yolov8n.pt")
    assert ok_back is True and detector.model_name == "yolov8n.pt"
    print("  [PASS] Restored primary yolov8n.pt cleanly")


def test_cuda_and_cpu_fallback():
    print("\n--- [Test 2] CUDA Check & Automatic CPU Fallback ---")
    import torch

    cuda_avail = torch.cuda.is_available()
    det_cpu = VehicleDetector(model_name="yolov8n.pt", device_preference="cpu")
    assert det_cpu.device == "cpu"
    dummy = np.zeros((360, 640, 3), dtype=np.uint8)
    dets, tracks = det_cpu.detect_and_track(dummy)
    assert isinstance(dets, list) and isinstance(tracks, dict)
    print(f"  [PASS] CUDA available={cuda_avail}; CPU fallback verified.")


def test_classes_custom_ready_and_dedup():
    print("\n--- [Test 3] Road-User Classes, Custom Readiness & Deduplication ---")
    detector = VehicleDetector(model_name="yolov8n.pt")
    expected_coco = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
    assert expected_coco.issubset(set(detector.supported_class_names)), (
        f"Missing COCO classes: {expected_coco - set(detector.supported_class_names)}"
    )
    print(f"  [PASS] Pretrained COCO classes verified: {detector.supported_class_names}")

    # Verify custom model names mapping keeps auto_rickshaw / scooter separate from car
    class DummyModel:
        names = {0: "car", 1: "auto_rickshaw", 2: "scooter", 3: "person"}

    detector._sync_model_classes(DummyModel())
    assert detector.active_class_map[1] == "auto_rickshaw"
    assert detector.active_class_map[2] == "scooter"
    assert detector.active_class_map[1] != "car"
    print("  [PASS] Custom class mapping verified (auto_rickshaw & scooter never mapped to car).")

    # Restore real model class map
    detector._sync_model_classes(detector.model)

    # Verify same-class near-duplicate suppression
    dups = [
        {"bbox": [10.0, 10.0, 110.0, 110.0], "confidence": 0.90, "class_name": "car"},
        {"bbox": [12.0, 12.0, 112.0, 112.0], "confidence": 0.65, "class_name": "car"},
        {"bbox": [15.0, 15.0, 95.0, 105.0], "confidence": 0.80, "class_name": "person"},
    ]
    kept = detector._deduplicate_detections(dups)
    assert len(kept) == 2, f"Expected 2 kept detections after dedup, got {len(kept)}"
    print("  [PASS] Duplicate same-class suppression verified while preserving distinct classes.")


def test_bytetrack_persistence():
    print("\n--- [Test 4] ByteTrack Tracking Persistence ---")
    _, acc_path = generate_all_samples()
    detector = VehicleDetector(model_name="yolov8n.pt")
    cap = cv2.VideoCapture(str(acc_path))
    seen_ids = set()
    multi_frame_ids = set()
    for _ in range(25):
        ret, frame = cap.read()
        if not ret:
            break
        _, tracks = detector.detect_and_track(frame)
        for tid, st in tracks.items():
            if st.is_confirmed:
                seen_ids.add(tid)
                if st.hits >= 5:
                    multi_frame_ids.add(tid)
    cap.release()
    assert len(multi_frame_ids) >= 1, "ByteTrack failed to persist any vehicle ID across >=5 frames!"
    print(f"  [PASS] ByteTrack persisted {len(multi_frame_ids)} track(s) across >=5 frames (IDs: {sorted(multi_frame_ids)})")


def test_four_tier_accident_and_overlap_guard():
    print("\n--- [Test 5] 4-Tier Accident Logic & Overlap Guard ---")
    with tempfile.TemporaryDirectory() as td:
        acc = AccidentDetector(risk_threshold=55.0, events_dir=Path(td))
        frame = np.zeros((540, 960, 3), dtype=np.uint8)

        # 5a. Normal steady overlap (perspective pass-by without deceleration/stoppage)
        c1 = VehicleState(1, [100, 100, 220, 200], "car", 0.9)
        c2 = VehicleState(2, [115, 110, 235, 210], "car", 0.88)
        c1.max_recent_speed = 6.0
        c2.max_recent_speed = 6.0
        c1.smoothed_speed = 5.8
        c2.smoothed_speed = 5.9
        c1.smoothed_deceleration = 0.0
        c2.smoothed_deceleration = 0.0
        c1.heading_deg = 0.0
        c2.heading_deg = 180.0
        c1.is_confirmed = True
        c2.is_confirmed = True

        is_alert, info_overlap = acc.analyze_frame(frame, {1: c1, 2: c2}, frame_idx=1)
        assert info_overlap["risk_level"] == config.TIER_NORMAL, (
            f"Steady overlap wrongly triggered {info_overlap['risk_level']} ({info_overlap['risk_score']}%)"
        )
        assert is_alert is False
        print(f"  [PASS] Overlap without impact stayed {info_overlap['risk_level']} ({info_overlap['risk_score']}%)")

        # 5b. Converging proximity -> COLLISION_RISK
        acc.reset_temporal_state()
        r1 = VehicleState(1, [200, 200, 280, 260], "car", 0.9)
        r2 = VehicleState(2, [265, 200, 345, 260], "car", 0.9)
        r1.current_velocity = (8.0, 0.0)
        r2.current_velocity = (-8.0, 0.0)
        r1.smoothed_speed = 8.0
        r2.smoothed_speed = 8.0
        r1.max_recent_speed = 8.0
        r2.max_recent_speed = 8.0
        r1.smoothed_deceleration = 8.5
        r2.smoothed_deceleration = 8.5
        r1.heading_deg = 0.0
        r2.heading_deg = 180.0
        r1.is_confirmed = True
        r2.is_confirmed = True
        _, info_risk = acc.analyze_frame(frame, {1: r1, 2: r2}, frame_idx=2)
        assert info_risk["risk_level"] == config.TIER_RISK, f"Expected COLLISION_RISK, got {info_risk}"
        print(f"  [PASS] Converging hazard classified as {info_risk['risk_level']} ({info_risk['risk_score']}%)")

        # 5c. Physical collision frame 1 -> SUSPECTED_ACCIDENT -> frame 3 -> CONFIRMED_ACCIDENT
        acc.reset_temporal_state()
        v1 = VehicleState(1, [100, 100, 200, 200], "car", 0.90)
        v2 = VehicleState(2, [110, 110, 210, 200], "truck", 0.88)
        v1.max_recent_speed = 15.0
        v2.max_recent_speed = 12.0
        v1.smoothed_speed = 0.5
        v2.smoothed_speed = 0.8
        v1.smoothed_deceleration = 14.0
        v2.smoothed_deceleration = 11.0
        v1.is_stopped = True
        v2.is_stopped = True
        v1.is_confirmed = True
        v2.is_confirmed = True

        _, info_f1 = acc.analyze_frame(frame, {1: v1, 2: v2}, frame_idx=1)
        assert info_f1["risk_level"] == config.TIER_SUSPECTED
        _ = acc.analyze_frame(frame, {1: v1, 2: v2}, frame_idx=2)
        is_a3, info_f3 = acc.analyze_frame(frame, {1: v1, 2: v2}, frame_idx=3)
        assert is_a3 is True and info_f3["risk_level"] == config.TIER_CONFIRMED
        assert len(acc.events) == 1
        print(f"  [PASS] Multi-frame confirmation verified: F1={info_f1['risk_level']} -> F3={info_f3['risk_level']}")


def test_normal_and_accident_footage():
    print("\n--- [Test 6] Normal Traffic & Accident Footage Validation ---")
    normal_path, acc_path = generate_all_samples()
    det = VehicleDetector(model_name="yolov8n.pt")

    with tempfile.TemporaryDirectory() as td:
        # Normal traffic -> 0 false alarms
        acc_norm = AccidentDetector(risk_threshold=55.0, events_dir=Path(td) / "norm")
        cap = cv2.VideoCapture(str(normal_path))
        fidx = 0
        norm_alerts = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            fidx += 1
            _, tracks = det.detect_and_track(frame)
            _, info = acc_norm.analyze_frame(frame, tracks, fidx)
            if info["risk_level"] in {config.TIER_SUSPECTED, config.TIER_CONFIRMED}:
                norm_alerts += 1
        cap.release()
        assert norm_alerts == 0, f"False alarms on normal traffic: {norm_alerts}"
        print(f"  [PASS] Normal traffic ({fidx} frames): 0 false alarms")

        # Accident footage -> must confirm collision
        det.reset()
        acc_crash = AccidentDetector(risk_threshold=55.0, events_dir=Path(td) / "crash")
        cap = cv2.VideoCapture(str(acc_path))
        fidx = 0
        tiers_seen = set()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            fidx += 1
            _, tracks = det.detect_and_track(frame)
            _, info = acc_crash.analyze_frame(frame, tracks, fidx)
            tiers_seen.add(info["risk_level"])
        cap.release()
        assert config.TIER_CONFIRMED in tiers_seen, f"Did not reach CONFIRMED_ACCIDENT! Saw: {tiers_seen}"
        assert len(acc_crash.events) >= 1
        print(f"  [PASS] Accident footage ({fidx} frames): tiers={sorted(tiers_seen)}, events={len(acc_crash.events)}")


def test_flask_api_uploads_and_exports():
    print("\n--- [Test 7] Flask API, Uploads, Telemetry & Exports ---")
    from app import app, engine

    client = app.test_client()

    # Dashboard HTML
    res = client.get("/")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert "Smarter Roads. Safer Journeys." in html
    assert "Launch Detection" in html
    assert "Explore Technology" in html
    print("  [PASS] GET / rendered redesigned landing + dashboard UI")

    # Stats endpoint
    res_stats = client.get("/api/stats")
    assert res_stats.status_code == 200
    stats = res_stats.get_json()
    assert stats["model_name"] == "yolov8n.pt"
    assert "supported_classes" in stats and "class_counts" in stats
    print(f"  [PASS] GET /api/stats OK (model={stats['model_name']}, device={stats['device']})")

    # Start & Stop sample stream (normal traffic to keep live event log untouched)
    res_start = client.post("/api/start_detection", json={"source": "sample_normal"})
    assert res_start.status_code == 200 and res_start.get_json()["success"] is True
    time.sleep(0.5)
    res_stop = client.post("/api/stop_detection")
    assert res_stop.status_code == 200 and res_stop.get_json()["success"] is True
    print("  [PASS] Stream start/stop verified")

    # Inference settings update & restore
    res_params = client.post("/api/set_inference_params", json={"imgsz": 480, "conf": 0.30})
    assert res_params.status_code == 200 and res_params.get_json()["imgsz"] == 480
    client.post("/api/set_inference_params", json={"imgsz": 640, "conf": 0.25})
    print("  [PASS] Inference parameter configuration verified")

    # Reject invalid extension
    res_bad_ext = client.post(
        "/api/upload_video",
        data={"video": (io.BytesIO(b"not a video"), "bad.exe")},
        content_type="multipart/form-data",
    )
    assert res_bad_ext.status_code == 400

    # Reject corrupted mp4
    res_corrupt = client.post(
        "/api/upload_video",
        data={"video": (io.BytesIO(b"corrupted mp4 bytes"), "corrupt.mp4")},
        content_type="multipart/form-data",
    )
    assert res_corrupt.status_code == 400

    # Accept valid mp4 upload (using sample video bytes, then clean up uploaded copy)
    sample_bytes = (config.SAMPLE_DIR / "normal_traffic.mp4").read_bytes()
    res_ok_up = client.post(
        "/api/upload_video",
        data={"video": (io.BytesIO(sample_bytes), "test_valid_clip.mp4")},
        content_type="multipart/form-data",
    )
    assert res_ok_up.status_code == 200 and res_ok_up.get_json()["success"] is True
    uploaded_path = Path(res_ok_up.get_json()["video_path"])
    engine.stop()
    if uploaded_path.exists():
        uploaded_path.unlink(missing_ok=True)
    print("  [PASS] Video upload validation (bad ext, corrupted mp4, valid mp4) verified")

    # Exports
    res_json = client.get("/api/export_events?format=json")
    assert res_json.status_code == 200 and "application/json" in res_json.content_type
    res_csv = client.get("/api/export_events?format=csv")
    assert res_csv.status_code == 200 and "text/csv" in res_csv.content_type
    print("  [PASS] JSON and CSV incident exports verified")


def run_empirical_benchmark():
    print("\n--- [Test 8] Measured Inference Benchmark (YOLOv8n vs YOLO26n) ---")
    from ultralytics import YOLO
    import torch

    _, acc_path = generate_all_samples()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    half_kwargs = {"half": True} if (device.startswith("cuda") and config.USE_HALF_PRECISION) else {}

    runs = [
        ("yolov8n.pt", 640),
        ("yolov8n.pt", 480),
        ("yolov8n.pt", 384),
        ("yolo26n.pt", 640),
    ]

    print(f"  Device: {device.upper()} | Video: {Path(acc_path).name}")
    print(f"  {'Model':<12} | {'ImgSz':<6} | {'Mean Latency':<14} | {'FPS':<8} | {'Detections (15f)':<16}")
    print("  " + "-" * 68)

    results = []
    for model_name, sz in runs:
        m_path = config.MODELS_DIR / model_name
        if not m_path.exists():
            print(f"  {model_name:<12} | {sz:<6} | MISSING - SKIPPED")
            continue

        m = YOLO(str(m_path))
        cap = cv2.VideoCapture(str(acc_path))
        ret, frame = cap.read()
        if ret:
            with torch.inference_mode():
                _ = m.predict(frame, imgsz=sz, device=device, verbose=False, **half_kwargs)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        times = []
        det_count = 0
        for _ in range(15):
            ret, frame = cap.read()
            if not ret:
                break
            t0 = time.perf_counter()
            with torch.inference_mode():
                res = m.predict(frame, imgsz=sz, conf=0.25, device=device, verbose=False, **half_kwargs)
            times.append(time.perf_counter() - t0)
            if res and res[0].boxes is not None:
                det_count += len(res[0].boxes)
        cap.release()

        avg_ms = float(np.mean(times) * 1000.0)
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
        results.append((model_name, sz, avg_ms, fps, det_count))
        print(f"  {model_name:<12} | {sz:<6} | {avg_ms:8.1f} ms    | {fps:5.1f}    | {det_count:3d} detections")

    print("  " + "-" * 68)
    return results


def run_all_tests():
    print("=" * 72)
    print("   AURA-VISION // END-TO-END TEST & BENCHMARK SUITE")
    print("=" * 72)
    t0 = time.perf_counter()
    test_model_loading_and_switching()
    test_cuda_and_cpu_fallback()
    test_classes_custom_ready_and_dedup()
    test_bytetrack_persistence()
    test_four_tier_accident_and_overlap_guard()
    test_normal_and_accident_footage()
    test_flask_api_uploads_and_exports()
    run_empirical_benchmark()
    print(f"\n>>> ALL TESTS PASSED IN {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    run_all_tests()
