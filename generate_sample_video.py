"""
Realistic Traffic & Collision Simulation Video Generator.
Synthesizes high-fidelity test video streams using real vehicle sprite crops
blended onto asphalt highway backgrounds with realistic kinematics and collision dynamics.
Produces:
  1. normal_traffic.mp4 (Smooth lane cruising in parallel lanes, zero collisions)
  2. accident_scenario.mp4 (Head-on / intersection collision with sudden deceleration and post-impact stoppage)
"""

import math
from pathlib import Path
from typing import Tuple, List, Optional
import cv2
import numpy as np

import config


def load_and_prep_sprite(img_path: Path, target_w: int = 280, target_h: int = 185, flip_h: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads sprite image, resizes, and creates an alpha mask with feathered edges
    so it blends naturally with the road background.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        # Fallback synthetic image if file is missing
        img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        img[:] = (120, 40, 40)
    else:
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        
    if flip_h:
        img = cv2.flip(img, 1)

    # Create feathered alpha mask
    mask = np.ones((target_h, target_w), dtype=np.float32)
    
    # Feather border edges (10 px margin)
    feather = 10
    for y in range(feather):
        mask[y, :] *= (y / feather)
        mask[target_h - 1 - y, :] *= (y / feather)
    for x in range(feather):
        mask[:, x] *= (x / feather)
        mask[:, target_w - 1 - x] *= (x / feather)
        
    return img, mask


def blend_sprite_on_road(
    road: np.ndarray,
    sprite: np.ndarray,
    mask: np.ndarray,
    cx: float,
    cy: float,
    angle_deg: float = 0.0
) -> np.ndarray:
    """Blends sprite onto road frame centered at (cx, cy) with optional rotation."""
    h_bg, w_bg, _ = road.shape
    h_sp, w_sp = sprite.shape[:2]

    # Optional rotation
    if abs(angle_deg) > 0.5:
        M = cv2.getRotationMatrix2D((w_sp / 2.0, h_sp / 2.0), -angle_deg, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h_sp * sin) + (w_sp * cos))
        new_h = int((h_sp * cos) + (w_sp * sin))
        M[0, 2] += (new_w / 2) - (w_sp / 2)
        M[1, 2] += (new_h / 2) - (h_sp / 2)
        
        rot_sprite = cv2.warpAffine(sprite, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        rot_mask = cv2.warpAffine(mask, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        
        sp = rot_sprite
        mk = rot_mask
        h_cur, w_cur = new_h, new_w
    else:
        sp = sprite
        mk = mask
        h_cur, w_cur = h_sp, w_sp

    top_left_x = int(cx - w_cur / 2)
    top_left_y = int(cy - h_cur / 2)

    src_x1 = max(0, -top_left_x)
    src_y1 = max(0, -top_left_y)
    src_x2 = min(w_cur, w_bg - top_left_x)
    src_y2 = min(h_cur, h_bg - top_left_y)

    dst_x1 = max(0, top_left_x)
    dst_y1 = max(0, top_left_y)
    dst_x2 = min(w_bg, top_left_x + w_cur)
    dst_y2 = min(h_bg, top_left_y + h_cur)

    if dst_x1 < dst_x2 and dst_y1 < dst_y2 and src_x1 < src_x2 and src_y1 < src_y2:
        crop_sp = sp[src_y1:src_y2, src_x1:src_x2].astype(np.float32)
        crop_mk = mk[src_y1:src_y2, src_x1:src_x2, np.newaxis]
        crop_bg = road[dst_y1:dst_y2, dst_x1:dst_x2].astype(np.float32)

        blended = crop_sp * crop_mk + crop_bg * (1.0 - crop_mk)
        road[dst_y1:dst_y2, dst_x1:dst_x2] = np.clip(blended, 0, 255).astype(np.uint8)

    return road


def draw_road_background(width: int = 960, height: int = 540) -> np.ndarray:
    """Draws realistic asphalt road background with realistic texture matching vehicle sprites."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Neutral asphalt color matching sprite backgrounds
    img[:] = (112, 114, 112)

    # Road shoulder / sidewalks
    cv2.rectangle(img, (0, 0), (width, 45), (85, 88, 86), -1)
    cv2.rectangle(img, (0, height - 45), (width, height), (85, 88, 86), -1)
    
    # Curb lines
    cv2.line(img, (0, 45), (width, 45), (160, 160, 160), 2)
    cv2.line(img, (0, height - 45), (width, height - 45), (160, 160, 160), 2)

    # Solid White Edge Lines
    cv2.line(img, (0, 60), (width, 60), (220, 225, 230), 2)
    cv2.line(img, (0, height - 60), (width, height - 60), (220, 225, 230), 2)

    # Yellow Center Divider
    cv2.line(img, (0, height // 2 - 2), (width, height // 2 - 2), (40, 215, 245), 2)
    cv2.line(img, (0, height // 2 + 2), (width, height // 2 + 2), (40, 215, 245), 2)

    # Dashed White Lane Dividers
    lane_y1 = 150
    lane_y2 = 390
    for x in range(0, width, 50):
        cv2.line(img, (x, lane_y1), (x + 28, lane_y1), (200, 205, 210), 2)
        cv2.line(img, (x, lane_y2), (x + 28, lane_y2), (200, 205, 210), 2)

    return img


def generate_normal_traffic_video(output_path: Path, num_frames: int = 120, fps: int = 24):
    """Synthesizes smooth multi-lane traffic without collisions."""
    w, h = 960, 540
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    sprites_dir = config.SPRITES_DIR
    car1_img, car1_mask = load_and_prep_sprite(sprites_dir / "car_1.jpg", 280, 185, flip_h=False)
    car2_img, car2_mask = load_and_prep_sprite(sprites_dir / "car_2.jpg", 280, 185, flip_h=True)

    base_road = draw_road_background(w, h)

    v1_x = 100.0
    v1_y = 160.0
    v1_vx = 7.5

    v2_x = 860.0
    v2_y = 380.0
    v2_vx = -7.5

    for f in range(num_frames):
        frame = base_road.copy()

        # Update positions
        v1_x += v1_vx
        v2_x += v2_vx

        # Wrap around
        if v1_x > w + 200:
            v1_x = -150
        if v2_x < -200:
            v2_x = w + 150

        frame = blend_sprite_on_road(frame, car1_img, car1_mask, v1_x, v1_y, angle_deg=0.0)
        frame = blend_sprite_on_road(frame, car2_img, car2_mask, v2_x, v2_y, angle_deg=0.0)

        writer.write(frame)

    writer.release()
    print(f"[SampleGenerator] Generated normal traffic video: {output_path}")


def generate_accident_scenario_video(output_path: Path, num_frames: int = 130, fps: int = 24):
    """
    Synthesizes a traffic accident:
    1. Frames 0-38: Vehicles cruise in collision course.
    2. Frames 38-48: Impact point! Overlap, deceleration shockwave, heading spin.
    3. Frames 48-130: Post-impact stoppage.
    """
    w, h = 960, 540
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    sprites_dir = config.SPRITES_DIR
    car1_img, car1_mask = load_and_prep_sprite(sprites_dir / "car_1.jpg", 280, 185, flip_h=False)
    car2_img, car2_mask = load_and_prep_sprite(sprites_dir / "car_2.jpg", 280, 185, flip_h=True)

    base_road = draw_road_background(w, h)
    impact_frame = 36

    v1_x = 100.0
    v1_y = 270.0
    v1_vx = 9.0
    v1_angle = 0.0

    v2_x = 860.0
    v2_y = 270.0
    v2_vx = -9.0
    v2_angle = 0.0

    for f in range(num_frames):
        frame = base_road.copy()

        if f < impact_frame:
            v1_x += v1_vx
            v2_x += v2_vx
        elif f < impact_frame + 10:
            # Impact Shockwave
            v1_vx *= 0.3
            v2_vx *= 0.3
            v1_x += v1_vx
            v2_x += v2_vx
            v1_angle += 4.5
            v2_angle -= 4.0
            v1_y -= 1.5
            v2_y += 1.5
        else:
            # Post-crash stationary halt
            v1_vx = 0.0
            v2_vx = 0.0

        frame = blend_sprite_on_road(frame, car1_img, car1_mask, v1_x, v1_y, angle_deg=v1_angle)
        frame = blend_sprite_on_road(frame, car2_img, car2_mask, v2_x, v2_y, angle_deg=v2_angle)

        # Impact sparks / debris
        if f >= impact_frame and f < impact_frame + 20:
            cx = int((v1_x + v2_x) / 2)
            cy = int((v1_y + v2_y) / 2)
            r = int(12 + (f - impact_frame) * 2)
            cv2.circle(frame, (cx, cy), r, (40, 200, 255), -1)
            cv2.circle(frame, (cx, cy), r + 8, (20, 100, 255), 2)

        writer.write(frame)

    writer.release()
    print(f"[SampleGenerator] Generated accident scenario video: {output_path}")


def generate_all_samples(force: bool = False) -> Tuple[str, str]:
    """Generates both sample simulation videos."""
    config.SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    
    normal_video = config.SAMPLE_DIR / "normal_traffic.mp4"
    accident_video = config.SAMPLE_DIR / "accident_scenario.mp4"

    if force or not normal_video.exists():
        generate_normal_traffic_video(normal_video)
    if force or not accident_video.exists():
        generate_accident_scenario_video(accident_video)

    return str(normal_video), str(accident_video)


if __name__ == "__main__":
    generate_all_samples(force=True)
