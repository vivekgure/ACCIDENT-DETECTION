"""
AI Vehicle Accident Detection - Full-Frame Video Processing & Telemetry Engine.
Processes full-frame video files, executes YOLOv8 inference with kinematic tracking,
evaluates multi-factor physical collision heuristics, renders surveillance HUD annotations,
saves high-definition annotated MP4 video, and exports incident reports and snapshot evidence.

Usage:
    python test_full_frame.py
    python test_full_frame.py --video path/to/video.mp4
    python test_full_frame.py --video path/to/video.mp4 --threshold 50 --output results.mp4
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import cv2
import numpy as np
import config
from detector import VehicleDetector
from tracker import VehicleState
from accident_detector import AccidentDetector

# Output directory for rendered videos and analytical reports
OUTPUT_DIR = BASE_DIR / "output_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_default_video() -> Path:
    """Finds the most relevant uploaded or sample video to test."""
    # 1. Check uploads directory for latest user video
    uploads = list(config.UPLOADS_DIR.glob("*.mp4"))
    if uploads:
        # Sort by modification time descending
        uploads.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return uploads[0]
    
    # 2. Check sample_data for accident_scenario or traffic_test
    sample_acc = config.SAMPLE_DIR / "accident_scenario.mp4"
    if sample_acc.exists():
        return sample_acc
        
    sample_traffic = config.SAMPLE_DIR / "traffic_test.mp4"
    if sample_traffic.exists():
        return sample_traffic

    raise FileNotFoundError("No input video found in uploads/ or sample_data/.")


def print_banner():
    banner = r"""
=============================================================================
   AI VEHICLE ACCIDENT DETECTION - FULL-FRAME SURVEILLANCE ENGINE
=============================================================================
    """
    print(banner)


def process_full_frame_video(
    video_path: Path,
    output_path: Optional[Path] = None,
    risk_threshold: float = config.ACCIDENT_RISK_THRESHOLD,
    conf_threshold: float = config.CONFIDENCE_THRESHOLD,
    model_name: Optional[str] = None,
    imgsz: Optional[int] = None,
    export_json: bool = True
) -> Dict[str, Any]:
    """
    Executes full-frame accident detection and generates an annotated output video.
    """
    print_banner()
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Input video file not found: {video_path}")

    # Set default output path if not provided
    if output_path is None:
        clean_name = video_path.stem
        output_path = OUTPUT_DIR / f"{clean_name}_annotated.mp4"
    else:
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[+] Input Video   : {video_path}")
    print(f"[+] Output Video  : {output_path}")
    print(f"[+] Risk Threshold: {risk_threshold}%")

    # 1. Open Video Capture & Read Metadata
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video capture from {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0.0

    print(f"[+] Resolution    : {width} x {height}")
    print(f"[+] Framerate     : {fps:.1f} FPS")
    print(f"[+] Total Frames  : {total_frames} (~{duration_sec:.1f}s)")

    # 2. Initialize Detectors
    print("\n[+] Initializing Neural Network & Kinematics Engines...")
    detector = VehicleDetector(model_name=model_name)
    detector.set_inference_settings(imgsz=imgsz, conf=conf_threshold)
    accident_detector = AccidentDetector(risk_threshold=risk_threshold)
    print(f"  [OK] Hardware Acceleration: {detector.device_name} ({detector.device})")
    print(f"  [OK] YOLO Model Weights   : {detector.model_path.name}")

    # 3. Setup Video Writer
    # Try mp4v codec first, then fallback to avc1 or MJPG
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        print("[!] Warning: Could not initialize hardware MP4 VideoWriter. Falling back to MJPG.")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # 4. Processing Loop
    print("\n[+] Commencing Full-Frame Processing Pipeline...\n")
    start_time = time.time()
    frame_idx = 0
    accident_frames_count = 0
    risk_scores = []
    fps_history = []
    last_loop_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            t_now = time.time()
            frame_elapsed = t_now - last_loop_time
            last_loop_time = t_now
            current_fps = 1.0 / max(frame_elapsed, 1e-5)
            fps_history.append(current_fps)

            # YOLO inference and kinematic tracking
            detections, active_tracks = detector.detect_and_track(frame)

            # Pre-render visual annotations
            annotated_frame = detector.annotate_frame(frame, active_tracks, accident_detector.active_alert)

            # Kinematic collision physics analysis (passing pre-rendered frame for peak snapshot capture)
            is_accident, alert_info = accident_detector.analyze_frame(
                frame, active_tracks, frame_idx, annotated_frame=annotated_frame
            )

            # Re-render with active alert data if state updated
            if is_accident:
                accident_frames_count += 1
                annotated_frame = detector.annotate_frame(frame, active_tracks, alert_info)

            # Append telemetry watermark on bottom-right of output video
            scale = max(0.55, min(width, height) / 720.0)
            time_code = f"T+{frame_idx / fps:.2f}s | F#{frame_idx:04d} | {current_fps:.1f} FPS"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(time_code, font, 0.38 * scale, 1)
            cv2.putText(
                annotated_frame,
                time_code,
                (width - tw - int(15 * scale), height - int(12 * scale)),
                font,
                0.38 * scale,
                (180, 210, 230),
                max(1, int(1 * scale)),
                cv2.LINE_AA
            )

            # Write annotated frame
            writer.write(annotated_frame)

            curr_risk = alert_info.get("risk_score", 0.0) if alert_info else 0.0
            risk_scores.append(curr_risk)

            # Console Progress Meter
            pct = (frame_idx / total_frames) * 100.0 if total_frames > 0 else 0.0
            elapsed = time.time() - start_time
            proc_fps = frame_idx / max(elapsed, 1e-5)
            eta = (total_frames - frame_idx) / proc_fps if proc_fps > 0 and total_frames > 0 else 0.0

            # Status Indicator Bar
            if curr_risk >= 70.0:
                status_glyph = "\033[91m[CRITICAL COLLISION]\033[0m"
            elif curr_risk >= 45.0:
                status_glyph = "\033[93m[ELEVATED HAZARD]\033[0m"
            else:
                status_glyph = "\033[92m[NORMAL FLOW]\033[0m"

            sys.stdout.write(
                f"\r  Frame {frame_idx:04d}/{total_frames} ({pct:5.1f}%) | "
                f"Speed: {proc_fps:4.1f} FPS | "
                f"Targets: {len(active_tracks):2d} | "
                f"Risk: {curr_risk:5.1f}% | "
                f"{status_glyph} | ETA: {eta:4.1f}s"
            )
            sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\n[!] Processing halted early by user.")
    finally:
        cap.release()
        writer.release()

    total_proc_time = time.time() - start_time
    avg_proc_fps = frame_idx / max(total_proc_time, 1e-5)
    events = accident_detector.get_events()
    peak_risk = max(risk_scores) if risk_scores else 0.0

    print("\n\n" + "=" * 77)
    print("   PROCESSING SUMMARY & COLLISION TELEMETRY REPORT")
    print("=" * 77)
    print(f"  Total Processed Frames : {frame_idx} / {total_frames}")
    print(f"  Total Elapsed Time     : {total_proc_time:.2f} seconds")
    print(f"  Average Processing Rate: {avg_proc_fps:.1f} FPS")
    print(f"  Accident Frames Trigger: {accident_frames_count} frames")
    print(f"  Peak Overall Risk Score: {peak_risk:.1f}%")
    print(f"  Recorded Incidents     : {len(events)}")
    print(f"  Rendered Output Video  : {output_path}")

    # Print Recorded Events Table
    if events:
        print("\n" + "-" * 77)
        print("   RECORDED COLLISION INCIDENTS")
        print("-" * 77)
        for e in events:
            veh_info = ", ".join([f"#{v.get('id')} ({v.get('class')})" for v in e.get("vehicles_involved", [])])
            print(f"  * Incident ID #{e['event_id']:04d} | Peak Risk: {e['risk_score']}% | Time: {e['timestamp']}")
            print(f"    Cause   : {e['reason']}")
            print(f"    Vehicles: {veh_info if veh_info else 'N/A'}")
            print(f"    Snapshot: {e['snapshot_url']}")
            print("-" * 77)
    else:
        print("\n  [OK] No collision incidents detected. Traffic flow categorized as safe.")

    # Export Analytical JSON Report
    report_data = {
        "input_video": str(video_path),
        "output_video": str(output_path),
        "resolution": [width, height],
        "fps": fps,
        "total_frames": total_frames,
        "processed_frames": frame_idx,
        "processing_time_sec": round(total_proc_time, 2),
        "avg_processing_fps": round(avg_proc_fps, 1),
        "accident_frames_count": accident_frames_count,
        "peak_risk_score": round(peak_risk, 1),
        "total_incidents": len(events),
        "incidents": events,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    if export_json:
        report_file = OUTPUT_DIR / f"{video_path.stem}_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        print(f"\n[+] Analytics JSON report saved: {report_file}")

    print("=" * 77 + "\n")
    return report_data


def main():
    parser = argparse.ArgumentParser(
        description="Run AI Full-Frame Vehicle Accident Detection & Video Rendering"
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        default=None,
        help="Path to input video file (defaults to latest uploaded video)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save annotated output video"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=config.ACCIDENT_RISK_THRESHOLD,
        help=f"Accident risk trigger threshold (0-100, default: {config.ACCIDENT_RISK_THRESHOLD})"
    )
    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=config.CONFIDENCE_THRESHOLD,
        help=f"YOLO object detection confidence threshold (default: {config.CONFIDENCE_THRESHOLD})"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=config.YOLO_MODEL_NAME,
        help=f"YOLO model weights filename (default: {config.YOLO_MODEL_NAME})"
    )
    parser.add_argument(
        "--imgsz", "-s",
        type=int,
        default=config.YOLO_INPUT_SIZE,
        help=f"Inference image size (default: {config.YOLO_INPUT_SIZE})"
    )
    args = parser.parse_args()

    # Determine video path
    if args.video:
        target_video = Path(args.video)
    else:
        target_video = find_default_video()

    target_output = Path(args.output) if args.output else None
    process_full_frame_video(
        video_path=target_video,
        output_path=target_output,
        risk_threshold=args.threshold,
        conf_threshold=args.conf,
        model_name=args.model,
        imgsz=args.imgsz
    )


if __name__ == "__main__":
    main()
