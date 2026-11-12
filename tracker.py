"""
Vehicle State & Kinematics Multi-Object Tracking Engine.
Tracks detected objects across frames, maintaining persistent IDs and calculating
instantaneous velocity, deceleration shockwaves, heading angles, angular deflection vectors,
calibrated speeds (if calibration available), and centroid breadcrumbs.
Supports both native ByteTrack / BoT-SORT tracking and internal spatial IoU tracking fallback.
"""

import math
from typing import List, Tuple, Dict, Optional, Any
import numpy as np

import config


class VehicleState:
    """
    Encapsulates real-time physical state and motion kinematics for a single tracked road user.
    """
    def __init__(
        self,
        track_id: int,
        bbox: List[float],
        class_name: str,
        confidence: float,
        fps: float = config.DEFAULT_CAMERA_FPS
    ):
        self.track_id = int(track_id)
        self.class_name = class_name
        self.confidence = float(confidence)
        self.fps = fps if fps > 0 else config.DEFAULT_CAMERA_FPS
        
        # Bounding box [x1, y1, x2, y2]
        self.current_bbox = [float(x) for x in bbox]
        self.center = (
            (self.current_bbox[0] + self.current_bbox[2]) / 2.0,
            (self.current_bbox[1] + self.current_bbox[3]) / 2.0
        )
        
        # History & Trajectories
        self.history: List[Tuple[float, float]] = [self.center]
        self.speeds_history: List[float] = [0.0]
        self.heading_history: List[float] = [0.0]
        
        # Velocity and Speed
        self.current_velocity = (0.0, 0.0)      # (dx, dy) per frame
        self.instantaneous_speed = 0.0           # px/frame
        self.smoothed_speed = 0.0                # EMA smoothed speed (px/frame)
        self.max_recent_speed = 0.0              # Peak recent speed (px/frame)
        
        # Calibrated Speed
        self.speed_real_units: Optional[float] = None  # km/h if calibrated, else None
        
        # Heading & Deflection
        self.current_heading_deg = 0.0           # 0 - 360 degrees
        self.heading_deg = 0.0                   # Alias
        self.heading_delta_deg = 0.0             # Angular deflection from previous step
        
        # Kinematic Deceleration & Stoppage
        self.smoothed_deceleration = 0.0         # Deceleration rate (px/frame^2)
        self.is_stopped = False
        self.motion_status = "Cruising"          # "Cruising", "Accelerating", "Braking", "Stopped", "Hazard"
        
        # Track lifecycle
        self.frame_count = 1
        self.time_since_update = 0
        self.hits = 1
        self.is_confirmed = (self.hits >= config.MIN_HITS)

    def update(
        self,
        bbox: List[float],
        confidence: float,
        class_name: Optional[str] = None,
        fps: Optional[float] = None
    ):
        """Updates track with new bounding box and recalculates kinematics."""
        if fps and fps > 0:
            self.fps = fps
            
        self.current_bbox = [float(x) for x in bbox]
        new_center = (
            (self.current_bbox[0] + self.current_bbox[2]) / 2.0,
            (self.current_bbox[1] + self.current_bbox[3]) / 2.0
        )
        
        if class_name:
            self.class_name = class_name
        self.confidence = float(confidence)
        
        # Kinematic Displacement Vector (px/frame)
        prev_center = self.center
        dx = new_center[0] - prev_center[0]
        dy = new_center[1] - prev_center[1]
        dist = math.hypot(dx, dy)
        
        self.current_velocity = (dx, dy)
        self.instantaneous_speed = dist
        
        # Exponential moving average for speed (alpha=0.6)
        if self.frame_count == 1:
            self.smoothed_speed = dist
        else:
            self.smoothed_speed = 0.6 * dist + 0.4 * self.smoothed_speed
            
        self.speeds_history.append(self.smoothed_speed)
        if len(self.speeds_history) > 30:
            self.speeds_history.pop(0)
            
        self.max_recent_speed = max(self.speeds_history)
        
        # Physical Speed Calibration (only if explicit calibration is configured)
        if config.PIXELS_PER_METER is not None and config.PIXELS_PER_METER > 0:
            # meters_per_frame = dist / PIXELS_PER_METER
            # meters_per_sec = meters_per_frame * fps
            # km_per_hour = meters_per_sec * 3.6
            meters_per_sec = (self.smoothed_speed / config.PIXELS_PER_METER) * self.fps
            self.speed_real_units = meters_per_sec * 3.6
        else:
            self.speed_real_units = None
        
        # Deceleration Calculation (Shockwave)
        prev_speed = self.speeds_history[-2] if len(self.speeds_history) >= 2 else self.smoothed_speed
        speed_delta = prev_speed - self.smoothed_speed
        decel = max(0.0, speed_delta)
        self.smoothed_deceleration = 0.7 * decel + 0.3 * self.smoothed_deceleration
        
        # Heading & Angular Deflection
        if dist > 1.5:  # Only compute heading when moving significantly
            new_heading = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
            prev_heading = self.heading_history[-1] if self.heading_history else new_heading
            
            # Shortest circular angular difference
            raw_diff = abs(new_heading - prev_heading)
            diff = min(raw_diff, 360.0 - raw_diff)
            self.heading_delta_deg = diff
            self.current_heading_deg = new_heading
            self.heading_deg = new_heading
            self.heading_history.append(new_heading)
        else:
            self.heading_delta_deg = 0.0
            
        if len(self.heading_history) > 30:
            self.heading_history.pop(0)
            
        # Stoppage & Motion Classification
        self.is_stopped = (self.smoothed_speed < config.POST_IMPACT_STATIONARY_SPEED and self.max_recent_speed > 3.0)
        
        if self.is_stopped:
            self.motion_status = "Stopped"
        elif self.smoothed_deceleration > 4.5:
            self.motion_status = "Braking"
        elif dist > prev_speed + 2.5:
            self.motion_status = "Accelerating"
        else:
            self.motion_status = "Cruising"
            
        # Update center & trail
        self.center = new_center
        self.history.append(self.center)
        if len(self.history) > 30:
            self.history.pop(0)
            
        self.frame_count += 1
        self.hits += 1
        self.time_since_update = 0
        
        if self.hits >= config.MIN_HITS:
            self.is_confirmed = True

    def mark_missed(self):
        """Handles a frame where object was not detected."""
        self.time_since_update += 1
        self.smoothed_speed *= 0.8

    def get_speed_display_text(self) -> str:
        """
        Returns properly formatted speed display string without misleading claims.
        Displays km/h if calibrated, otherwise px/s relative velocity.
        """
        if self.speed_real_units is not None:
            return f"{self.speed_real_units:.1f} km/h"
        # Relative pixel velocity per second
        speed_px_sec = self.smoothed_speed * self.fps
        return f"{speed_px_sec:.0f} px/s"


class VehicleTracker:
    """
    Multi-Object Tracking Engine that integrates with ByteTrack / BoT-SORT IDs
    and provides a robust internal IoU fallback association engine.
    """
    def __init__(self):
        self.tracks: Dict[int, VehicleState] = {}
        self.next_id = 1
        self.total_unique_vehicles = 0

    def reset(self):
        """Resets tracker state."""
        self.tracks.clear()
        self.next_id = 1
        self.total_unique_vehicles = 0

    def update_from_tracked_detections(
        self,
        tracked_detections: List[dict],
        fps: float = config.DEFAULT_CAMERA_FPS
    ) -> Dict[int, VehicleState]:
        """
        Updates tracks when detector provides persistent ByteTrack / BoT-SORT IDs.
        Input format: [{'bbox': [x1,y1,x2,y2], 'track_id': int, 'class_name': str, 'confidence': float}]
        """
        updated_tids = set()

        for det in tracked_detections:
            tid = det.get("track_id")
            if tid is None:
                continue
            tid = int(tid)
            updated_tids.add(tid)
            
            if tid in self.tracks:
                self.tracks[tid].update(
                    bbox=det["bbox"],
                    confidence=det["confidence"],
                    class_name=det["class_name"],
                    fps=fps
                )
            else:
                self.tracks[tid] = VehicleState(
                    track_id=tid,
                    bbox=det["bbox"],
                    class_name=det["class_name"],
                    confidence=det["confidence"],
                    fps=fps
                )
                self.total_unique_vehicles += 1
                self.next_id = max(self.next_id, tid + 1)

        # Mark missed tracks and prune aged tracks
        for tid, track in list(self.tracks.items()):
            if tid not in updated_tids:
                track.mark_missed()
                if track.time_since_update > config.MAX_TRACK_AGE:
                    del self.tracks[tid]

        return self.tracks

    def update(
        self,
        detections: List[dict],
        fps: float = config.DEFAULT_CAMERA_FPS
    ) -> Dict[int, VehicleState]:
        """
        Fallback association engine when detections do not contain persistent track IDs.
        Uses spatial IoU overlap, centroid proximity, and class consistency.
        """
        # If detections already have track_id from ByteTrack, route to dedicated handler
        if detections and "track_id" in detections[0] and detections[0]["track_id"] is not None:
            return self.update_from_tracked_detections(detections, fps=fps)

        if not detections:
            for track in list(self.tracks.values()):
                track.mark_missed()
                if track.time_since_update > config.MAX_TRACK_AGE:
                    del self.tracks[track.track_id]
            return self.tracks

        track_ids = list(self.tracks.keys())
        det_indices = list(range(len(detections)))
        
        matched_pairs = []
        unmatched_dets = set(det_indices)
        unmatched_tracks = set(track_ids)

        if track_ids:
            # 1. First Pass: IoU Overlap Matching
            cost_matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float32)
            for i, tid in enumerate(track_ids):
                track = self.tracks[tid]
                for j, det in enumerate(detections):
                    iou = self._calculate_iou(track.current_bbox, det["bbox"])
                    class_bonus = 0.15 if track.class_name == det["class_name"] else -0.2
                    cost_matrix[i, j] = iou + class_bonus

            # Greedy Matching on IoU
            while True:
                if cost_matrix.size == 0:
                    break
                max_val = np.max(cost_matrix)
                if max_val < config.IOU_TRACKER_THRESHOLD:
                    break
                i, j = np.unravel_index(np.argmax(cost_matrix), cost_matrix.shape)
                tid = track_ids[i]
                
                matched_pairs.append((tid, j))
                unmatched_tracks.discard(tid)
                unmatched_dets.discard(j)
                cost_matrix[i, :] = -1.0
                cost_matrix[:, j] = -1.0

            # 2. Second Pass: Velocity-Guided Centroid Proximity for Fast Moving Objects
            for tid in list(unmatched_tracks):
                track = self.tracks[tid]
                best_dist = config.MAX_CENTROID_DISTANCE
                best_det_idx = None
                
                pred_cx = track.center[0] + track.current_velocity[0]
                pred_cy = track.center[1] + track.current_velocity[1]

                for det_idx in unmatched_dets:
                    det = detections[det_idx]
                    det_cx = (det["bbox"][0] + det["bbox"][2]) / 2.0
                    det_cy = (det["bbox"][1] + det["bbox"][3]) / 2.0
                    
                    dist_curr = math.hypot(track.center[0] - det_cx, track.center[1] - det_cy)
                    dist_pred = math.hypot(pred_cx - det_cx, pred_cy - det_cy)
                    dist = min(dist_curr, dist_pred)
                    
                    if track.class_name == det["class_name"]:
                        dist *= 0.85
                    else:
                        dist *= 1.25

                    if dist < best_dist:
                        best_dist = dist
                        best_det_idx = det_idx

                if best_det_idx is not None:
                    matched_pairs.append((tid, best_det_idx))
                    unmatched_tracks.discard(tid)
                    unmatched_dets.discard(best_det_idx)

        # 3. Update Matched Tracks
        for tid, det_idx in matched_pairs:
            det = detections[det_idx]
            self.tracks[tid].update(
                bbox=det["bbox"],
                confidence=det["confidence"],
                class_name=det["class_name"],
                fps=fps
            )

        # 4. Handle Unmatched Tracks (Missed)
        for tid in unmatched_tracks:
            self.tracks[tid].mark_missed()
            if self.tracks[tid].time_since_update > config.MAX_TRACK_AGE:
                del self.tracks[tid]

        # 5. Create New Tracks for Unmatched Detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            new_id = self.next_id
            self.next_id += 1
            self.total_unique_vehicles += 1
            
            new_track = VehicleState(
                track_id=new_id,
                bbox=det["bbox"],
                class_name=det["class_name"],
                confidence=det["confidence"],
                fps=fps
            )
            self.tracks[new_id] = new_track

        return self.tracks

    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculates standard 2D Intersection over Union."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = max(0.0, bbox1[2] - bbox1[0]) * max(0.0, bbox1[3] - bbox1[1])
        area2 = max(0.0, bbox2[2] - bbox2[0]) * max(0.0, bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0
