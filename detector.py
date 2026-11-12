"""
AURA-VISION Lightweight Vehicle & Road User Detector with ByteTrack & Clean HUD.
Executes real-time YOLOv8n (or optional lightweight alternative) inference on
NVIDIA CUDA GPU (with FP16) or optimized CPU fallback, tracks road users via
ByteTrack, supports custom road-user classes (e.g. scooters, three-wheelers),
and renders clean, readable automotive surveillance annotations.
"""

import gc
import math
import shutil
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import config
from tracker import VehicleTracker, VehicleState

# Clean BGR color tokens matching AURA-VISION design system
BGR_LIME = (59, 216, 183)       # #B7D83B - Normal tracking
BGR_AMBER = (59, 169, 229)      # #E5A93B - Collision risk warning
BGR_RED = (82, 82, 224)         # #E05252 - Suspected / Confirmed accident
BGR_CHARCOAL = (28, 29, 25)     # #191D1C - Deep charcoal badge backing
BGR_TEXT_LIGHT = (239, 244, 242)  # #F2F4EF - Soft white text
BGR_TEXT_MUTED = (164, 170, 163)  # #A3AAA4 - Muted grey text


class VehicleDetector:
    """
    Wraps Ultralytics lightweight YOLO (YOLOv8n primary) for road-user detection,
    ByteTrack multi-object tracking, and clean automotive HUD visualization.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        model_path: Optional[Path] = None,
        device_preference: Optional[str] = None,
    ):
        self.tracker = VehicleTracker()
        self.tracker_type = config.TRACKER_TYPE
        self.tracker_config = config.TRACKER_CONFIG_PATH

        # Inference settings
        self.imgsz = config.YOLO_INPUT_SIZE
        self.conf_threshold = config.CONFIDENCE_THRESHOLD
        self.iou_threshold = config.NMS_IOU_THRESHOLD
        self.tracked_classes: Set[str] = set(config.TRACKED_CLASSES)

        # Hardware device selection & NVML monitoring
        self.device, self.device_name, self.vram_info = self._detect_hardware(device_preference)

        # Dynamic class mapping populated from loaded model.names
        self.active_class_map: Dict[int, str] = dict(config.COCO_CLASS_MAP)
        self.supported_class_names: List[str] = sorted(set(config.COCO_CLASS_MAP.values()))
        self.class_indices: List[int] = sorted(self.active_class_map.keys())

        # Model Loading (single model instance)
        self.model_name = model_name or config.YOLO_MODEL_NAME
        self.model_path = model_path or (config.MODELS_DIR / self.model_name)
        self.model = self._load_yolo_model()
        self._sync_model_classes(self.model)

        # Latency telemetry
        self.last_inference_time_ms = 0.0

    def _detect_hardware(self, preference: Optional[str] = None) -> Tuple[str, str, Dict[str, Any]]:
        """
        Detects available NVIDIA CUDA hardware, queries VRAM via NVML,
        and provides automatic CPU fallback.
        """
        pref = (preference or config.DEVICE_PREFERENCE).lower()
        vram_info = self._query_nvml_memory()

        if pref in {"cuda", "auto"} and torch.cuda.is_available():
            device = "cuda:0"
            gpu_name = torch.cuda.get_device_name(0)
            device_name = f"CUDA ({gpu_name})"
            print(f"[VehicleDetector] Hardware selected: {device_name}")
            if vram_info.get("has_nvml"):
                print(
                    f"[VehicleDetector] VRAM Available: "
                    f"{vram_info['free_mb']} MB / {vram_info['total_mb']} MB"
                )
        else:
            device = "cpu"
            device_name = "CPU"
            if pref == "cuda" and not torch.cuda.is_available():
                print("[VehicleDetector] CUDA requested but unavailable in PyTorch build. Using CPU fallback.")
            else:
                print("[VehicleDetector] Hardware selected: CPU")

        return device, device_name, vram_info

    def _query_nvml_memory(self) -> Dict[str, Any]:
        """Queries live GPU memory via NVML if available."""
        vram_info: Dict[str, Any] = {
            "total_mb": 0,
            "used_mb": 0,
            "free_mb": 0,
            "has_nvml": False,
        }
        try:
            import pynvml

            pynvml.nvmlInit()
            if pynvml.nvmlDeviceGetCount() > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(gpu_name, bytes):
                    gpu_name = gpu_name.decode("utf-8", errors="ignore")
                vram_info = {
                    "total_mb": round(mem.total / (1024 ** 2), 1),
                    "used_mb": round(mem.used / (1024 ** 2), 1),
                    "free_mb": round(mem.free / (1024 ** 2), 1),
                    "gpu_name": gpu_name,
                    "has_nvml": True,
                }
        except Exception:
            pass
        return vram_info

    def refresh_vram_info(self) -> Dict[str, Any]:
        """Refreshes live VRAM telemetry without heavy overhead."""
        self.vram_info = self._query_nvml_memory()
        return self.vram_info

    def _is_valid_checkpoint(self, path: Path) -> bool:
        """Checks that a local .pt file exists and is a readable zip archive."""
        if not path.exists() or path.stat().st_size < 1024:
            return False
        try:
            return zipfile.is_zipfile(path)
        except Exception:
            return False

    def _load_weights_from_source(self, model_name: str, target_path: Path) -> YOLO:
        """Loads weights from local cache, workspace root, or Ultralytics download."""
        # 1. Local models/ cache (validate integrity first)
        if target_path.exists():
            if self._is_valid_checkpoint(target_path):
                print(f"[VehicleDetector] Loading weights from local cache: {target_path}")
                try:
                    return YOLO(str(target_path))
                except Exception as e:
                    print(f"[VehicleDetector] Cached weights failed to load ({e}). Re-downloading...")
            else:
                print(f"[VehicleDetector] Cached file {target_path} is corrupted. Removing and re-downloading...")
            try:
                target_path.unlink(missing_ok=True)
            except Exception:
                pass

        # 2. Check workspace root cache
        root_cand = Path(model_name)
        if root_cand.exists() and self._is_valid_checkpoint(root_cand):
            print(f"[VehicleDetector] Moving {root_cand} to {target_path}...")
            try:
                shutil.move(str(root_cand), str(target_path))
                return YOLO(str(target_path))
            except Exception as e:
                print(f"[VehicleDetector] Move error: {e}")

        # 3. Auto-download via Ultralytics and cache into models/
        print(f"[VehicleDetector] Downloading weights: {model_name}...")
        model = YOLO(model_name)
        if root_cand.exists() and root_cand.resolve() != target_path.resolve():
            shutil.move(str(root_cand), str(target_path))
            model = YOLO(str(target_path))
        print(f"[VehicleDetector] Weights cached at: {target_path}")
        return model

    def _load_yolo_model(self) -> YOLO:
        """
        Loads the primary lightweight YOLO model from local cache or downloads it,
        falling back to the secondary lightweight model if unavailable.
        """
        try:
            model = self._load_weights_from_source(self.model_name, self.model_path)
            return self._warmup_model(model)
        except Exception as e:
            print(f"[VehicleDetector] Failed to load {self.model_name}: {e}")

            fallback_name = config.YOLO_FALLBACK_MODEL_NAME
            fallback_path = config.YOLO_FALLBACK_MODEL_PATH
            if self.model_name != fallback_name:
                try:
                    print(f"[VehicleDetector] Falling back to {fallback_name}...")
                    self.model_name = fallback_name
                    self.model_path = fallback_path
                    model = self._load_weights_from_source(fallback_name, fallback_path)
                    return self._warmup_model(model)
                except Exception as fb_err:
                    print(f"[VehicleDetector] Fallback load failed: {fb_err}")

            raise RuntimeError(f"Unable to load primary model {self.model_name} or fallback {fallback_name}.")

    def _sync_model_classes(self, model: YOLO):
        """
        Inspects model.names dynamically so COCO models use exact COCO IDs and
        custom-trained weights (e.g. scooter, three_wheeler, auto_rickshaw)
        work seamlessly without misclassifying classes.
        """
        raw_names = getattr(model, "names", None)
        if isinstance(raw_names, dict) and raw_names:
            mapped: Dict[int, str] = {}
            for cid, raw_label in raw_names.items():
                norm = str(raw_label).strip().lower().replace(" ", "_")
                canonical = config.ROAD_USER_ALIASES.get(norm)
                if canonical:
                    mapped[int(cid)] = canonical
            if mapped:
                self.active_class_map = mapped
            else:
                self.active_class_map = dict(config.COCO_CLASS_MAP)
        else:
            self.active_class_map = dict(config.COCO_CLASS_MAP)

        self.supported_class_names = sorted(set(self.active_class_map.values()))
        self._rebuild_class_indices()

    def _rebuild_class_indices(self):
        """Rebuilds active class index filter for YOLO inference."""
        self.class_indices = [
            cid for cid, name in self.active_class_map.items()
            if name in self.tracked_classes
        ]

    def _warmup_model(self, model: YOLO) -> YOLO:
        """Runs a single warmup inference to initialize graph/CUDA kernels safely."""
        try:
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            half = self.device.startswith("cuda") and config.USE_HALF_PRECISION
            kwargs = {"half": True} if half else {}
            with torch.inference_mode():
                _ = model.predict(
                    dummy,
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False,
                    **kwargs,
                )
            print(f"[VehicleDetector] Model warmed up on {self.device} (imgsz={self.imgsz}, fp16={half}).")
        except torch.cuda.OutOfMemoryError as oom:
            print(f"[VehicleDetector] GPU OutOfMemory during warmup ({oom}). Falling back to CPU.")
            self.device = "cpu"
            self.device_name = "CPU (VRAM Fallback)"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"[VehicleDetector] Warmup notice (non-fatal): {e}")
        return model

    def set_model(self, model_name: str) -> bool:
        """
        Dynamically switches active lightweight YOLO weights without restarting,
        freeing previous model memory first to avoid VRAM bloat.
        """
        new_path = config.MODELS_DIR / model_name
        try:
            print(f"[VehicleDetector] Switching model to {model_name}...")
            old_model = getattr(self, "model", None)
            self.model = None
            if old_model is not None:
                del old_model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            new_model = self._load_weights_from_source(model_name, new_path)
            self._warmup_model(new_model)
            self.model = new_model
            self.model_name = model_name
            self.model_path = new_path
            self._sync_model_classes(new_model)
            self.reset()
            self.refresh_vram_info()
            print(f"[VehicleDetector] Switched to {model_name} (classes: {self.supported_class_names})")
            return True
        except Exception as e:
            print(f"[VehicleDetector] Failed to switch to {model_name}: {e}")
            return False

    def set_inference_settings(
        self,
        imgsz: Optional[int] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        classes: Optional[List[str]] = None,
    ):
        """Updates runtime inference parameters."""
        if imgsz and imgsz in config.SUPPORTED_INPUT_SIZES:
            self.imgsz = imgsz
        if conf is not None and 0.05 <= conf <= 0.95:
            self.conf_threshold = conf
        if iou is not None and 0.10 <= iou <= 0.90:
            self.iou_threshold = iou
        if classes is not None:
            self.tracked_classes = set(classes)
            self._rebuild_class_indices()

    def reset(self):
        """Resets kinematic tracker and Ultralytics ByteTrack state cleanly."""
        self.tracker.reset()
        predictor = getattr(self.model, "predictor", None) if self.model is not None else None
        if predictor is not None and hasattr(predictor, "trackers") and predictor.trackers:
            for trk in predictor.trackers:
                if hasattr(trk, "reset"):
                    trk.reset()

    def _deduplicate_detections(self, detections: List[dict]) -> List[dict]:
        """
        Suppresses near-duplicate detections of the same class (IoU >= threshold)
        while preserving distinct overlapping objects of different classes or
        legitimate adjacent vehicles.
        """
        if len(detections) <= 1:
            return detections

        sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
        kept: List[dict] = []
        thresh = config.DUPLICATE_CLASS_IOU_THRESHOLD

        for cand in sorted_dets:
            cx1, cy1, cx2, cy2 = cand["bbox"]
            c_area = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
            is_dup = False
            for k in kept:
                if k["class_name"] != cand["class_name"]:
                    continue
                kx1, ky1, kx2, ky2 = k["bbox"]
                ix1 = max(cx1, kx1)
                iy1 = max(cy1, ky1)
                ix2 = min(cx2, kx2)
                iy2 = min(cy2, ky2)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                if inter <= 0.0:
                    continue
                k_area = max(0.0, kx2 - kx1) * max(0.0, ky2 - ky1)
                union = c_area + k_area - inter
                if union > 0.0 and (inter / union) >= thresh:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(cand)
        return kept

    def detect_and_track(
        self,
        frame: np.ndarray,
        fps: float = config.DEFAULT_CAMERA_FPS,
    ) -> Tuple[List[dict], Dict[int, VehicleState]]:
        """
        Runs lightweight YOLO inference with ByteTrack multi-object tracking.
        Returns (detections, active_vehicle_states).
        """
        if self.model is None:
            return [], self.tracker.tracks

        t0 = time.perf_counter()
        half = self.device.startswith("cuda") and config.USE_HALF_PRECISION
        half_kwargs = {"half": True} if half else {}
        classes_arg = self.class_indices if self.class_indices else None

        with torch.inference_mode():
            try:
                results = self.model.track(
                    source=frame,
                    persist=True,
                    tracker=self.tracker_config,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    classes=classes_arg,
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False,
                    **half_kwargs,
                )
            except torch.cuda.OutOfMemoryError as oom:
                print(f"[VehicleDetector] OutOfMemory on GPU ({oom}). Falling back to CPU...")
                self.device = "cpu"
                self.device_name = "CPU (VRAM Fallback)"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                results = self.model.track(
                    source=frame,
                    persist=True,
                    tracker=self.tracker_config,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    classes=classes_arg,
                    imgsz=self.imgsz,
                    device="cpu",
                    verbose=False,
                )
            except Exception as e:
                print(f"[VehicleDetector] Tracker exception ({e}). Falling back to predict...")
                results = self.model.predict(
                    source=frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    classes=classes_arg,
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False,
                    **half_kwargs,
                )

        self.last_inference_time_ms = (time.perf_counter() - t0) * 1000.0

        detections: List[dict] = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)

                track_ids = (
                    boxes.id.cpu().numpy().astype(int)
                    if hasattr(boxes, "id") and boxes.id is not None
                    else [None] * len(xyxy)
                )

                for box, conf, cls_id, tid in zip(xyxy, confs, classes, track_ids):
                    class_name = self.active_class_map.get(int(cls_id))
                    if class_name and class_name in self.tracked_classes:
                        detections.append({
                            "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                            "confidence": float(conf),
                            "class_name": class_name,
                            "class_id": int(cls_id),
                            "track_id": int(tid) if tid is not None else None,
                        })

        detections = self._deduplicate_detections(detections)
        active_tracks = self.tracker.update(detections, fps=fps)
        return detections, active_tracks

    def _color_for_track(
        self,
        tid: int,
        hazard_ids: List[int],
        risk_level: str,
    ) -> Tuple[int, int, int]:
        """
        Determines bounding-box color strictly by safety state:
        - Lime green (#B7D83B) for normal tracking
        - Amber for collision-risk vehicles
        - Red for suspected or confirmed accident-related vehicles
        """
        if tid in hazard_ids:
            if risk_level in {config.TIER_CONFIRMED, config.TIER_SUSPECTED}:
                return BGR_RED
            if risk_level == config.TIER_RISK:
                return BGR_AMBER
        return BGR_LIME

    def annotate_frame(
        self,
        frame: np.ndarray,
        active_tracks: Dict[int, VehicleState],
        alert_info: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Renders clean, minimal automotive detection overlays:
        - Thin bounding boxes (lime normal, amber caution, red accident)
        - Compact labels with tracking ID, vehicle class, and confidence %
        - Subtle trajectory indicators
        """
        out = frame.copy()
        h, w, _ = out.shape
        scale = max(0.6, min(w, h) / 720.0)

        hazard_ids = alert_info.get("hazard_ids", []) if alert_info else []
        risk_level = alert_info.get("risk_level", config.TIER_NORMAL) if alert_info else config.TIER_NORMAL
        risk_score = alert_info.get("risk_score", 0.0) if alert_info else 0.0

        # 1. Subtle Trajectory Breadcrumbs
        for tid, track in active_tracks.items():
            if not track.is_confirmed or len(track.history) < 2:
                continue

            color = self._color_for_track(tid, hazard_ids, risk_level)
            pts = track.history[-18:]
            for i in range(1, len(pts)):
                pt1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
                pt2 = (int(pts[i][0]), int(pts[i][1]))
                cv2.line(out, pt1, pt2, color, 1, cv2.LINE_AA)

        # 2. Clean Collision Proximity Link (only when hazard pair detected)
        if len(hazard_ids) >= 2 and risk_level != config.TIER_NORMAL:
            centers = [active_tracks[tid].center for tid in hazard_ids if tid in active_tracks]
            if len(centers) >= 2:
                p1 = (int(centers[0][0]), int(centers[0][1]))
                p2 = (int(centers[1][0]), int(centers[1][1]))
                mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                link_color = BGR_RED if risk_level in {config.TIER_CONFIRMED, config.TIER_SUSPECTED} else BGR_AMBER
                cv2.line(out, p1, p2, link_color, 1, cv2.LINE_AA)
                cv2.circle(out, mid, max(8, int(14 * scale)), link_color, 1, cv2.LINE_AA)

        # 3. Thin Bounding Boxes & Compact Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.38, 0.42 * scale)
        pad_x = max(4, int(5 * scale))
        pad_y = max(2, int(3 * scale))

        for tid, track in active_tracks.items():
            if not track.is_confirmed:
                continue

            x1, y1, x2, y2 = [int(v) for v in track.current_bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            is_hazard = tid in hazard_ids and risk_level != config.TIER_NORMAL
            color = self._color_for_track(tid, hazard_ids, risk_level)
            thickness = 2 if is_hazard else 1

            # Thin bounding rectangle
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

            # Subtle corner ticks on hazard vehicles only for visual distinction without clutter
            if is_hazard:
                self._draw_corner_ticks(out, x1, y1, x2, y2, color, corner_len=max(8, int(12 * scale)), thickness=2)

            # Subtle velocity direction indicator
            if track.smoothed_speed > 2.0:
                vx, vy = track.current_velocity
                v_mag = math.hypot(vx, vy)
                if v_mag > 0:
                    arrow_len = min(int(36 * scale), max(10, int(track.smoothed_speed * 2.5 * scale)))
                    end_x = int(track.center[0] + (vx / v_mag) * arrow_len)
                    end_y = int(track.center[1] + (vy / v_mag) * arrow_len)
                    cv2.arrowedLine(
                        out,
                        (int(track.center[0]), int(track.center[1])),
                        (end_x, end_y),
                        color,
                        1,
                        cv2.LINE_AA,
                        tipLength=0.28,
                    )

            # Compact single-line label: #ID class 92%
            cls_label = track.class_name.replace("_", " ").upper()
            label = f"#{tid} {cls_label} {int(track.confidence * 100)}%"
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)

            box_top = max(th + pad_y * 2 + 2, y1)
            bg_y1 = box_top - th - pad_y * 2
            bg_y2 = box_top
            bg_x2 = min(w - 1, x1 + tw + pad_x * 2)

            cv2.rectangle(out, (x1, bg_y1), (bg_x2, bg_y2), BGR_CHARCOAL, -1)
            cv2.putText(
                out,
                label,
                (x1 + pad_x, bg_y2 - pad_y - 1),
                font,
                font_scale,
                color,
                1,
                cv2.LINE_AA,
            )

        # 4. Minimal Top Status Strip
        self._draw_hud_overlay(out, active_tracks, alert_info, scale)

        # 5. Clean Alert Strip on Suspected / Confirmed Accidents
        if risk_level in {config.TIER_CONFIRMED, config.TIER_SUSPECTED} or risk_score >= config.ACCIDENT_RISK_THRESHOLD:
            self._draw_alert_banner(out, alert_info, scale)

        return out

    def _draw_corner_ticks(
        self,
        img: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: Tuple[int, int, int],
        corner_len: int = 10,
        thickness: int = 2,
    ):
        """Draws minimal corner emphasis ticks around hazard bounding boxes."""
        cl = min(corner_len, max(4, (x2 - x1) // 4), max(4, (y2 - y1) // 4))
        cv2.line(img, (x1, y1), (x1 + cl, y1), color, thickness)
        cv2.line(img, (x1, y1), (x1, y1 + cl), color, thickness)
        cv2.line(img, (x2, y1), (x2 - cl, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y1 + cl), color, thickness)
        cv2.line(img, (x1, y2), (x1 + cl, y2), color, thickness)
        cv2.line(img, (x1, y2), (x1, y2 - cl), color, thickness)
        cv2.line(img, (x2, y2), (x2 - cl, y2), color, thickness)
        cv2.line(img, (x2, y2), (x2, y2 - cl), color, thickness)

    def _draw_hud_overlay(
        self,
        img: np.ndarray,
        active_tracks: Dict[int, VehicleState],
        alert_info: Optional[dict] = None,
        scale: float = 1.0,
    ):
        """Draws a minimal, unobtrusive status bar across the top edge."""
        h, w, _ = img.shape
        confirmed_count = sum(1 for t in active_tracks.values() if t.is_confirmed)
        hud_h = max(26, int(30 * scale))

        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), BGR_CHARCOAL, -1)
        cv2.addWeighted(overlay, 0.80, img, 0.20, 0, img)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = max(0.36, 0.40 * scale)
        text_y = int(hud_h * 0.68)

        dev_tag = "CUDA" if self.device.startswith("cuda") else "CPU"
        left_text = f"AURA-VISION  |  {self.model_name.replace('.pt', '').upper()} ({dev_tag})"
        cv2.putText(img, left_text, (int(12 * scale), text_y), font, font_size, BGR_TEXT_LIGHT, 1, cv2.LINE_AA)

        center_text = f"TRACKED: {confirmed_count}"
        (cw, _), _ = cv2.getTextSize(center_text, font, font_size, 1)
        cv2.putText(img, center_text, ((w - cw) // 2, text_y), font, font_size, BGR_TEXT_MUTED, 1, cv2.LINE_AA)

        risk_val = alert_info.get("risk_score", 0.0) if alert_info else 0.0
        risk_level = alert_info.get("risk_level", config.TIER_NORMAL) if alert_info else config.TIER_NORMAL

        if risk_level == config.TIER_CONFIRMED:
            risk_color, tier_label = BGR_RED, "CONFIRMED"
        elif risk_level == config.TIER_SUSPECTED:
            risk_color, tier_label = BGR_RED, "SUSPECTED"
        elif risk_level == config.TIER_RISK:
            risk_color, tier_label = BGR_AMBER, "RISK"
        else:
            risk_color, tier_label = BGR_LIME, "NORMAL"

        risk_text = f"{tier_label}  {risk_val:.0f}%"
        (rw, _), _ = cv2.getTextSize(risk_text, font, font_size, 1)
        cv2.putText(img, risk_text, (w - rw - int(12 * scale), text_y), font, font_size, risk_color, 1, cv2.LINE_AA)

    def _draw_alert_banner(self, img: np.ndarray, alert_info: dict, scale: float = 1.0):
        """Renders a clean, high-legibility bottom alert strip when an incident is flagged."""
        h, w, _ = img.shape
        banner_h = max(38, int(46 * scale))
        banner_y1 = h - banner_h
        banner_y2 = h

        risk = alert_info.get("risk_score", 0.0)
        risk_level = alert_info.get("risk_level", config.TIER_SUSPECTED)
        reason = alert_info.get("reason", "Collision signature detected")

        accent_color = BGR_RED if risk_level in {config.TIER_CONFIRMED, config.TIER_SUSPECTED} else BGR_AMBER
        label = (
            "CONFIRMED ACCIDENT"
            if risk_level == config.TIER_CONFIRMED
            else ("SUSPECTED ACCIDENT" if risk_level == config.TIER_SUSPECTED else "COLLISION RISK")
        )

        overlay = img.copy()
        cv2.rectangle(overlay, (0, banner_y1), (w, banner_y2), BGR_CHARCOAL, -1)
        cv2.addWeighted(overlay, 0.88, img, 0.12, 0, img)
        cv2.line(img, (0, banner_y1), (w, banner_y1), accent_color, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        msg = f"{label} ({risk:.0f}%)  |  {reason}"
        max_chars = max(42, int(w / (9 * scale)))
        if len(msg) > max_chars:
            msg = msg[: max_chars - 3] + "..."

        cv2.putText(
            img,
            msg,
            (int(14 * scale), banner_y1 + int(banner_h * 0.65)),
            font,
            max(0.38, 0.42 * scale),
            BGR_TEXT_LIGHT,
            1,
            cv2.LINE_AA,
        )
