"""
Accident Detection Engine with Multi-Frame Temporal Evidence and 4-Tier Categorization.
Evaluates inter-vehicle physics interactions, closing velocity vectors, deceleration shockwaves,
abnormal trajectory deviations, and multi-frame temporal persistence to distinguish:
  1. Normal Traffic
  2. Potential Collision Risk
  3. Suspected Accident
  4. Confirmed Accident (Requires verified multi-frame temporal confirmation)

Prevents false alarms from normal braking, queueing, perspective occlusion, and temporary overlaps.
Includes an optional temporal feature extractor for video-based neural classification models.
"""

import os
# pyrefly: ignore [missing-import]
import cv2
import json
import time
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Deque
from collections import deque
# pyrefly: ignore [missing-import]
import numpy as np

import config
from tracker import VehicleState


class AccidentEvent:
    """Represents a recorded accident incident with multi-tier status and evidence."""
    def __init__(
        self,
        event_id: int,
        timestamp: str,
        risk_score: float,
        reason: str,
        vehicles_involved: List[dict],
        snapshot_filename: str,
        status: str = config.TIER_SUSPECTED,
        temporal_confirmations: int = 1
    ):
        self.event_id = event_id
        self.timestamp = timestamp
        self.risk_score = risk_score
        self.reason = reason
        self.vehicles_involved = vehicles_involved
        self.snapshot_filename = snapshot_filename
        self.status = status
        self.temporal_confirmations = temporal_confirmations

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "risk_score": round(self.risk_score, 1),
            "reason": self.reason,
            "vehicle_count": len(self.vehicles_involved),
            "vehicles_involved": self.vehicles_involved,
            "snapshot_url": f"/accident_events/{self.snapshot_filename}",
            "status": self.status,
            "temporal_confirmations": self.temporal_confirmations
        }


class AccidentDetector:
    """
    Evaluates kinematic trajectory anomalies and inter-vehicle physics interactions.
    Provides multi-frame temporal confirmation to distinguish between Normal Traffic,
    Collision Risk, Suspected Accidents, and Confirmed Accidents.
    """
    def __init__(
        self,
        risk_threshold: float = config.ACCIDENT_RISK_THRESHOLD,
        events_dir: Optional[Path] = None,
        log_file: Optional[Path] = None,
    ):
        self.risk_threshold = risk_threshold
        self.events_dir = Path(events_dir) if events_dir is not None else config.EVENTS_DIR
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.events: List[AccidentEvent] = []
        self.event_counter = 0
        self.last_alert_frame = -9999
        self.cooldown_frames = config.ALERT_COOLDOWN_FRAMES
        self.current_frame_idx = 0
        self.active_alert: Optional[dict] = None
        self.smoothed_risk_score = 0.0
        
        # Multi-frame temporal confirmation tracker
        self.consecutive_anomaly_frames = 0
        self.candidate_hazard_ids: List[int] = []
        self.current_incident: Optional[AccidentEvent] = None
        self.incident_active_until_frame: int = -9999
        
        # Temporal history buffer for optional video sequence classification
        self.temporal_buffer: Deque[dict] = deque(maxlen=30)
        
        # Load existing event history if present
        self.log_file = Path(log_file) if log_file is not None else (self.events_dir / "events_log.json")
        self._load_history()

    def reset_temporal_state(self):
        """Resets transient temporal state when switching video sources or looping."""
        self.consecutive_anomaly_frames = 0
        self.candidate_hazard_ids = []
        self.smoothed_risk_score = 0.0
        self.active_alert = None
        self.current_incident = None
        self.incident_active_until_frame = -9999
        self.temporal_buffer.clear()

    def _load_history(self):
        """Loads persistent event log from disk if available."""
        if self.log_file.exists():
            try:
                with open(self.log_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        event = AccidentEvent(
                            event_id=item["event_id"],
                            timestamp=item["timestamp"],
                            risk_score=item["risk_score"],
                            reason=item["reason"],
                            vehicles_involved=item["vehicles_involved"],
                            snapshot_filename=item["snapshot_url"].split("/")[-1],
                            status=item.get("status", config.TIER_SUSPECTED),
                            temporal_confirmations=item.get("temporal_confirmations", 1)
                        )
                        self.events.append(event)
                        self.event_counter = max(self.event_counter, item["event_id"])
            except Exception as e:
                print(f"[AccidentDetector] Notice: Could not read history log: {e}")

    def _save_history(self):
        """Persists event history to disk."""
        try:
            data = [e.to_dict() for e in self.events]
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[AccidentDetector] Error saving history: {e}")

    def analyze_frame(
        self,
        frame: np.ndarray,
        active_tracks: Dict[int, VehicleState],
        frame_idx: int,
        annotated_frame: Optional[np.ndarray] = None
    ) -> Tuple[bool, Optional[dict]]:
        """
        Analyzes the current frame for multi-tier collision signatures with temporal confirmation.
        Returns (is_alert_frame, alert_info_dict).
        """
        self.current_frame_idx = frame_idx
        track_list = [v for v in active_tracks.values() if v.is_confirmed]
        num_tracks = len(track_list)

        raw_max_risk = 0.0
        primary_reason = ""
        hazard_ids = []
        involved_vehicles = []

        # --- 1. Pairwise Vehicle Interactions (Collisions & Converging Vectors) ---
        for i in range(num_tracks):
            for j in range(i + 1, num_tracks):
                v1 = track_list[i]
                v2 = track_list[j]

                risk, reasons, ids = self._evaluate_vehicle_pair(v1, v2)
                if risk > raw_max_risk:
                    raw_max_risk = risk
                    primary_reason = ", ".join(reasons)
                    hazard_ids = ids
                    involved_vehicles = [
                        {"id": v1.track_id, "class": v1.class_name, "speed": round(v1.smoothed_speed, 1)},
                        {"id": v2.track_id, "class": v2.class_name, "speed": round(v2.smoothed_speed, 1)}
                    ]

        # --- 2. Single Vehicle Anomalies (Barrier Crash / Violent Spin) ---
        for v in track_list:
            risk, reasons = self._evaluate_single_vehicle(v)
            if risk > raw_max_risk:
                raw_max_risk = risk
                primary_reason = ", ".join(reasons)
                hazard_ids = [v.track_id]
                involved_vehicles = [
                    {"id": v.track_id, "class": v.class_name, "speed": round(v.smoothed_speed, 1)}
                ]

        # --- 3. Multi-Frame Temporal Confirmation Engine ---
        # Evaluate if the anomaly is sustained across consecutive frames
        if raw_max_risk >= config.RISK_THRESHOLD_CAUTION:
            self.consecutive_anomaly_frames += 1
            self.candidate_hazard_ids = hazard_ids
        else:
            # Decay anomaly counter gracefully
            self.consecutive_anomaly_frames = max(0, self.consecutive_anomaly_frames - 1)
            if self.consecutive_anomaly_frames == 0:
                self.candidate_hazard_ids = []

        # Smooth continuous risk score (fast attack, controlled decay)
        if raw_max_risk >= self.risk_threshold or raw_max_risk > self.smoothed_risk_score:
            self.smoothed_risk_score = raw_max_risk
        else:
            self.smoothed_risk_score = 0.85 * self.smoothed_risk_score + 0.15 * raw_max_risk
            
        display_risk = max(0.0, min(100.0, self.smoothed_risk_score))

        # --- 4. Four-Tier Classification Determination ---
        # 1) Confirmed: High risk + verified persistence over consecutive frames
        # 2) Suspected: High risk but initial frame(s)
        # 3) Collision Risk: Elevated closing vector/proximity
        # 4) Normal Traffic: Flow is safe
        if display_risk >= config.RISK_THRESHOLD_CONFIRMED and self.consecutive_anomaly_frames >= config.MIN_TEMPORAL_CONFIRM_FRAMES:
            risk_level = config.TIER_CONFIRMED
            status_text = "Confirmed Accident (Validated Multi-Frame Signature)"
            is_triggered = True
        elif display_risk >= config.RISK_THRESHOLD_SUSPECTED:
            risk_level = config.TIER_SUSPECTED
            status_text = "Suspected Accident (Validating Temporal Persistence)"
            is_triggered = True
        elif display_risk >= config.RISK_THRESHOLD_CAUTION:
            risk_level = config.TIER_RISK
            status_text = "Potential Collision Risk (Proximity Encroachment)"
            is_triggered = (display_risk >= self.risk_threshold)
        else:
            risk_level = config.TIER_NORMAL
            status_text = "Normal Traffic Flow"
            is_triggered = False

        # --- 5. Incident Clustering & Snapshot Logging ---
        if is_triggered and display_risk >= self.risk_threshold:
            is_ongoing_incident = (
                self.current_incident is not None and 
                self.current_frame_idx <= self.incident_active_until_frame
            )
            
            if is_ongoing_incident:
                # Update ongoing incident with peak risk, upgraded tier, and peak snapshot
                if display_risk > self.current_incident.risk_score or risk_level == config.TIER_CONFIRMED:
                    self._update_accident_event(
                        self.current_incident, frame, display_risk,
                        primary_reason, involved_vehicles, risk_level, annotated_frame
                    )
                self.incident_active_until_frame = self.current_frame_idx + self.cooldown_frames
            else:
                # Create distinct new incident
                self.current_incident = self._record_accident_event(
                    frame, display_risk, primary_reason,
                    involved_vehicles, hazard_ids, risk_level, annotated_frame
                )
                self.incident_active_until_frame = self.current_frame_idx + self.cooldown_frames
                self.last_alert_frame = self.current_frame_idx

        # Snapshot temporal state for optional sequence modeling
        self.temporal_buffer.append({
            "frame_idx": frame_idx,
            "risk": raw_max_risk,
            "consecutive_count": self.consecutive_anomaly_frames,
            "tier": risk_level,
            "num_targets": num_tracks
        })

        # Generate alert payload for HUD and web dashboard
        alert_info = {
            "is_alert": is_triggered,
            "risk_score": round(display_risk, 1),
            "raw_risk": round(raw_max_risk, 1),
            "risk_level": risk_level,
            "status_text": status_text,
            "reason": primary_reason if is_triggered else ("Potential Collision Caution" if display_risk >= config.RISK_THRESHOLD_CAUTION else "Normal Traffic Flow"),
            "hazard_ids": hazard_ids,
            "involved_vehicles": involved_vehicles,
            "consecutive_anomaly_frames": self.consecutive_anomaly_frames,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.active_alert = alert_info if is_triggered else None
        return is_triggered, alert_info

    def _evaluate_vehicle_pair(
        self,
        v1: VehicleState,
        v2: VehicleState
    ) -> Tuple[float, List[str], List[int]]:
        """
        Evaluates two interacting vehicles using physical collision geometry,
        relative closing vectors, impact deceleration, and post-crash immobility.
        Filters normal braking queues and passenger rider co-detections.
        """
        reasons = []
        iou = self._calculate_iou(v1.current_bbox, v2.current_bbox)
        center_dist = math.hypot(v1.center[0] - v2.center[0], v1.center[1] - v2.center[1])
        
        # 0. Two pedestrians interacting never constitutes a vehicular accident
        if v1.class_name == "person" and v2.class_name == "person":
            return 0.0, [], []

        # Relative vector analysis
        dx = v2.center[0] - v1.center[0]
        dy = v2.center[1] - v1.center[1]
        dist = math.hypot(dx, dy)
        
        v1_dx, v1_dy = v1.current_velocity
        v2_dx, v2_dy = v2.current_velocity
        
        rel_vx = v2_dx - v1_dx
        rel_vy = v2_dy - v1_dy
        closing_speed = -(dx * rel_vx + dy * rel_vy) / (dist + 1e-5)
        rel_motion = math.hypot(rel_vx, rel_vy)

        # Filter passenger / rider co-detections (e.g. rider on motorcycle or bicycle)
        is_ped_and_veh = (v1.class_name == "person" and v2.class_name != "person") or \
                         (v2.class_name == "person" and v1.class_name != "person")
        if is_ped_and_veh:
            ped = v1 if v1.class_name == "person" else v2
            veh = v2 if v1.class_name == "person" else v1
            
            # Motorcycle / Bicycle rider filter
            if veh.class_name in {"motorcycle", "bicycle"}:
                ped_cx, ped_cy = ped.center
                vx1, vy1, vx2, vy2 = veh.current_bbox
                if iou > 0.05 or center_dist < 65.0 or ((vx1 - 30 <= ped_cx <= vx2 + 30) and (vy1 - 40 <= ped_cy <= vy2 + 40)):
                    return 0.0, [], []

            # Check if pedestrian center is contained within vehicle bbox
            ped_cx, ped_cy = ped.center
            vx1, vy1, vx2, vy2 = veh.current_bbox
            is_inside_veh = (vx1 - 25 <= ped_cx <= vx2 + 25) and (vy1 - 25 <= ped_cy <= vy2 + 45)
            
            raw_hdiff = abs(veh.heading_deg - ped.heading_deg) % 360
            h_dist = min(raw_hdiff, 360 - raw_hdiff)
            
            is_occupant = is_inside_veh and (h_dist < 60.0 or abs(veh.smoothed_speed - ped.smoothed_speed) < 3.5)
            if is_occupant:
                return 0.0, [], []

        # Filter normal parallel queuing (e.g., stoplight or traffic jam where vehicles crawl at similar speeds)
        heading_diff = abs(v1.heading_deg - v2.heading_deg) % 360
        min_heading_diff = min(heading_diff, 360 - heading_diff)
        is_parallel_traffic = (min_heading_diff < 25.0) and (abs(v1.smoothed_speed - v2.smoothed_speed) < 2.0)
        
        # If moving in parallel traffic without sudden deceleration, reject as normal flow
        if is_parallel_traffic and v1.smoothed_deceleration < 4.0 and v2.smoothed_deceleration < 4.0:
            return 0.0, [], []

        score_iou = 0.0
        score_decel = 0.0
        score_angle = 0.0
        score_stop = 0.0

        is_ped_involved = (v1.class_name == "person" or v2.class_name == "person")
        max_prior_speed = max(v1.max_recent_speed, v2.max_recent_speed)

        # A. Spatial Contact / Bounding Box Overlap
        if iou >= config.COLLISION_IOU_THRESHOLD:
            score_iou = min(100.0, 60.0 + (iou / 0.30) * 40.0)
            reasons.append(f"Physical Collision Contact (IoU: {iou:.2f})")
        elif is_ped_involved and (iou > 0.06 or (center_dist < 45.0 and max_prior_speed > 3.5)):
            score_iou = 90.0
            reasons.append("Vehicle-Pedestrian Collision Contact")
        elif iou > 0.05:
            score_iou = (iou / config.COLLISION_IOU_THRESHOLD) * 45.0
            reasons.append(f"Critical Spatial Overlap (IoU: {iou:.2f})")
        elif center_dist < config.PROXIMITY_DISTANCE_PX and closing_speed > 4.0:
            score_iou = min(40.0, (closing_speed / 10.0) * 40.0)
            reasons.append(f"Rapid Collision Trajectory ({closing_speed:.1f} px/f)")
        else:
            score_iou = 0.0

        # B. Impact Deceleration Shockwave
        max_smoothed_decel = max(v1.smoothed_deceleration, v2.smoothed_deceleration)
        has_contact_or_near = (iou > 0.04 or (center_dist < config.PROXIMITY_DISTANCE_PX and closing_speed > 3.0) or is_ped_involved)
        
        if has_contact_or_near and max_smoothed_decel > 4.0:
            score_decel = min(100.0, (max_smoothed_decel / config.SUDDEN_DECEL_THRESHOLD_PX) * 100.0)
            reasons.append(f"Sudden Impact Deceleration ({max_smoothed_decel:.1f} px/f)")
        else:
            score_decel = 0.0

        # C. Abnormal Trajectory Deflection / Spin
        max_heading_delta = max(v1.heading_delta_deg, v2.heading_delta_deg)
        max_moving_speed = max(v1.smoothed_speed, v2.smoothed_speed)
        
        if has_contact_or_near and max_heading_delta > config.TRAJECTORY_ANGLE_DEV_DEG and max_moving_speed > 2.0:
            score_angle = min(100.0, (max_heading_delta / 100.0) * 100.0)
            reasons.append(f"Violent Collision Deflection ({max_heading_delta:.0f}°)")
        else:
            score_angle = 0.0

        # D. Post-Impact Sudden Stoppage
        prior_speed_high = (v1.max_recent_speed > 5.5 and v1.is_stopped) or (v2.max_recent_speed > 5.5 and v2.is_stopped)
        if prior_speed_high and (iou > 0.06 or score_decel > 25.0 or is_ped_involved):
            score_stop = 90.0
            reasons.append("Post-Impact Stoppage")
        else:
            score_stop = 0.0

        # If no spatial contact and no converging trajectory, total risk is zero
        if score_iou == 0.0 and score_decel == 0.0:
            return 0.0, [], []

        # Multi-factor weighted formulation: spatial overlap alone without kinematic
        # impact (deceleration, stoppage, or deflection) must not trigger an accident.
        has_kinematic_impact = (score_decel > 20.0 or score_stop > 50.0 or score_angle > 30.0)
        if (score_iou >= 80.0 or iou >= config.COLLISION_IOU_THRESHOLD) and has_kinematic_impact:
            risk = max(
                82.0,
                score_iou * 0.35 + score_decel * 0.35 + score_angle * 0.15 + score_stop * 0.15
            )
        else:
            risk = (
                score_iou * config.WEIGHT_IOU_OVERLAP +
                score_decel * config.WEIGHT_SUDDEN_DECEL +
                score_angle * config.WEIGHT_TRAJECTORY_DEVIATION +
                score_stop * config.WEIGHT_POST_STOPPAGE
            )

        hazard_ids = [v1.track_id, v2.track_id] if risk >= config.RISK_THRESHOLD_CAUTION else []
        return min(100.0, max(0.0, risk)), reasons, hazard_ids

    def _evaluate_single_vehicle(self, v: VehicleState) -> Tuple[float, List[str]]:
        """
        Evaluates single-vehicle accidents (e.g. crashing into barrier, spin-out, rollover).
        Requires confirmed vehicular track with genuine abrupt energy loss or violent spinning.
        """
        if v.class_name not in config.VEHICLE_CLASSES or v.frame_count < 4:
            return 0.0, []

        reasons = []
        risk = 0.0
        
        # Dead stop into barrier from high speed
        if v.max_recent_speed > 11.0 and v.smoothed_speed < 1.5 and v.smoothed_deceleration > 8.5:
            risk = 82.0
            reasons.append(f"Solo Wall/Barrier Crash ({v.max_recent_speed:.1f} -> 0 px/f)")
            
        # Violent trajectory spin or rollover while in motion
        if v.heading_delta_deg > 75.0 and (v.smoothed_speed > 3.5 or v.smoothed_deceleration > 5.5):
            spin_risk = min(92.0, 72.0 + (v.heading_delta_deg / 180.0) * 20.0)
            risk = max(risk, spin_risk)
            reasons.append(f"Violent Vehicle Spin / Loss of Control ({v.heading_delta_deg:.0f}°)")
            
        return risk, reasons

    def _record_accident_event(
        self,
        frame: np.ndarray,
        risk_score: float,
        reason: str,
        vehicles_involved: List[dict],
        hazard_ids: List[int],
        status: str = config.TIER_SUSPECTED,
        annotated_frame: Optional[np.ndarray] = None
    ) -> AccidentEvent:
        """
        Saves annotated event snapshot image to disk and records new incident.
        """
        self.event_counter += 1
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        file_timestamp = now.strftime("%Y%m%d_%H%M%S")
        
        snapshot_filename = f"event_{self.event_counter:04d}_{file_timestamp}.jpg"
        snapshot_path = self.events_dir / snapshot_filename
        
        snapshot_img = (annotated_frame if annotated_frame is not None else frame).copy()
        self._render_snapshot_banner(snapshot_img, self.event_counter, risk_score, reason, vehicles_involved, timestamp_str, status)
        
        try:
            cv2.imwrite(str(snapshot_path), snapshot_img)
            print(f"[AccidentDetector] Recorded incident #{self.event_counter:04d} [{status}] (Risk: {risk_score:.1f}%): {snapshot_filename}")
        except Exception as e:
            print(f"[AccidentDetector] Failed to save snapshot: {e}")

        event = AccidentEvent(
            event_id=self.event_counter,
            timestamp=timestamp_str,
            risk_score=risk_score,
            reason=reason if reason else "Collision / Trajectory Anomaly",
            vehicles_involved=vehicles_involved,
            snapshot_filename=snapshot_filename,
            status=status,
            temporal_confirmations=self.consecutive_anomaly_frames
        )
        
        self.events.insert(0, event)
        self._save_history()
        return event

    def _update_accident_event(
        self,
        event: AccidentEvent,
        frame: np.ndarray,
        risk_score: float,
        reason: str,
        vehicles_involved: List[dict],
        status: str,
        annotated_frame: Optional[np.ndarray] = None
    ):
        """Updates an active incident with higher peak risk score, status, and snapshot."""
        event.risk_score = max(event.risk_score, risk_score)
        if reason:
            event.reason = reason
        if status == config.TIER_CONFIRMED:
            event.status = config.TIER_CONFIRMED
        event.temporal_confirmations = max(event.temporal_confirmations, self.consecutive_anomaly_frames)
        
        if len(vehicles_involved) >= len(event.vehicles_involved):
            event.vehicles_involved = vehicles_involved
            
        snapshot_path = self.events_dir / event.snapshot_filename
        snapshot_img = (annotated_frame if annotated_frame is not None else frame).copy()
        self._render_snapshot_banner(
            snapshot_img, event.event_id, event.risk_score,
            event.reason, event.vehicles_involved, event.timestamp, event.status
        )
        try:
            cv2.imwrite(str(snapshot_path), snapshot_img)
        except Exception as e:
            print(f"[AccidentDetector] Failed to update snapshot: {e}")
        self._save_history()

    def _render_snapshot_banner(
        self,
        img: np.ndarray,
        event_id: int,
        risk: float,
        reason: str,
        vehicles: List[dict],
        timestamp_str: str,
        status: str = config.TIER_CONFIRMED
    ):
        """Renders incident banner on snapshot with status badge."""
        h, w, _ = img.shape
        scale = max(0.55, min(w, h) / 720.0)
        banner_h = max(52, int(64 * scale))
        
        bg_col = (28, 29, 25)
        border_col = (82, 82, 224) if status in {config.TIER_CONFIRMED, config.TIER_SUSPECTED} else (59, 169, 229)

        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), bg_col, -1)
        cv2.addWeighted(overlay, 0.90, img, 0.10, 0, img)
        cv2.line(img, (0, banner_h), (w, banner_h), border_col, max(1, int(2 * scale)))
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        title = f"INCIDENT #{event_id:04d} [{status}] | RISK: {risk:.1f}% | {timestamp_str}"
        cv2.putText(
            img, title,
            (int(14 * scale), int(22 * scale)),
            font, 0.48 * scale, (255, 255, 255), max(1, int(1.4 * scale)), cv2.LINE_AA
        )
        
        veh_str = ", ".join([f"#{v.get('id')}:{v.get('class')}" for v in vehicles]) if vehicles else "None"
        cause_text = f"Cause: {reason} | Targets: [{veh_str}]"
        max_chars = max(38, int(w / (10 * scale)))
        if len(cause_text) > max_chars:
            cause_text = cause_text[:max_chars - 3] + "..."
            
        cv2.putText(
            img, cause_text,
            (int(14 * scale), int(48 * scale)),
            font, 0.38 * scale, (210, 235, 255), max(1, int(1.2 * scale)), cv2.LINE_AA
        )

    def extract_temporal_features(self, active_tracks: Dict[int, VehicleState]) -> np.ndarray:
        """
        Optional temporal feature extractor for video-based neural network models (LSTM/Transformer).
        Transforms tracked targets into a fixed-length normalized kinematic vector.
        """
        features = []
        for tid, track in active_tracks.items():
            if track.is_confirmed:
                features.extend([
                    track.center[0] / 1280.0,
                    track.center[1] / 720.0,
                    track.current_velocity[0] / 50.0,
                    track.current_velocity[1] / 50.0,
                    track.smoothed_speed / 50.0,
                    track.smoothed_deceleration / 20.0,
                    track.heading_delta_deg / 180.0
                ])
        # Pad or truncate to fixed 64 dimensions
        vec = np.zeros(64, dtype=np.float32)
        if features:
            arr = np.array(features[:64], dtype=np.float32)
            vec[:len(arr)] = arr
        return vec

    def get_events(self) -> List[dict]:
        """Returns all recorded events serialized as dictionary list."""
        return [e.to_dict() for e in self.events]

    def clear_events(self):
        """Clears in-memory and disk event histories."""
        self.events.clear()
        self.event_counter = 0
        self.current_incident = None
        self.incident_active_until_frame = -9999
        self.consecutive_anomaly_frames = 0
        if self.log_file.exists():
            try:
                self.log_file.unlink()
            except Exception:
                pass
        for img_file in self.events_dir.glob("*.jpg"):
            try:
                img_file.unlink()
            except Exception:
                pass
        print("[AccidentDetector] History and snapshots cleared.")

    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Computes standard Intersection-over-Union (IoU) between two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection_w = max(0.0, x2 - x1)
        intersection_h = max(0.0, y2 - y1)
        intersection_area = intersection_w * intersection_h

        area1 = max(0.0, (bbox1[2] - bbox1[0])) * max(0.0, (bbox1[3] - bbox1[1]))
        area2 = max(0.0, (bbox2[2] - bbox2[0])) * max(0.0, (bbox2[3] - bbox2[1]))
        union_area = area1 + area2 - intersection_area

        if union_area <= 0:
            return 0.0
        return intersection_area / union_area
