"""
AURA-VISION Road User & Custom Vehicle Fine-Tuning Pipeline (YOLOv8n Base).
Prepares YOLO dataset scaffolds supporting both standard COCO road users and
custom classes (scooters, three-wheelers / auto-rickshaws, accident_contact),
optimized for NVIDIA RTX 4050 (6 GB VRAM) or CPU training.
"""

from pathlib import Path
from typing import Dict, Any
import yaml

import config


def create_dataset_scaffold(target_dir: Path = config.DATASETS_DIR / "accident_traffic") -> Path:
    """Creates standard Ultralytics YOLO dataset folder scaffold with custom vehicle classes."""
    target_dir.mkdir(parents=True, exist_ok=True)

    subdirs = [
        target_dir / "images" / "train",
        target_dir / "images" / "val",
        target_dir / "labels" / "train",
        target_dir / "labels" / "val",
    ]
    for s in subdirs:
        s.mkdir(parents=True, exist_ok=True)

    classes = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "scooter",
        "bus",
        "truck",
        "auto_rickshaw",
        "accident_contact",
    ]

    data_yaml = {
        "path": str(target_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(classes)},
        "nc": len(classes),
    }

    yaml_path = target_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    readme_path = target_dir / "README.md"
    readme_content = f"""# AURA-VISION Road User & Custom Vehicle Dataset Scaffold

## Annotation Format (YOLO Normalized Format)
Each `.txt` label file contains one line per object:
`<class_id> <x_center> <y_center> <width> <height>` (normalized 0.0 - 1.0)

## Class Indices
{chr(10).join([f"- **{i}**: `{name}`" for i, name in enumerate(classes)])}

## Note on Pretrained Weights vs Custom Classes
Pretrained COCO `yolov8n.pt` natively detects `person`, `bicycle`, `car`, `motorcycle`,
`bus`, and `truck`. Fine-tune on this scaffold to add dedicated `scooter`, `auto_rickshaw`
(three-wheeler), and `accident_contact` classes. `VehicleDetector` automatically maps these
class names from `model.names` at load time.
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    print(f"[Dataset] Created dataset scaffold and data.yaml at: {yaml_path}")
    return yaml_path


def get_training_args(yaml_path: Path, epochs: int = 50, batch_size: int = 16) -> Dict[str, Any]:
    """Returns lightweight YOLOv8n training arguments sized for RTX 4050 (6 GB VRAM)."""
    import torch

    return {
        "data": str(yaml_path),
        "model": str(config.YOLO_MODEL_PATH if config.YOLO_MODEL_PATH.exists() else config.YOLO_MODEL_NAME),
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": config.YOLO_INPUT_SIZE,
        "device": "0" if torch.cuda.is_available() else "cpu",
        "workers": 4,
        "patience": 15,
        "save": True,
        "cache": False,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "val": True,
        "plots": True,
    }


def main():
    print("=" * 70)
    print("   AURA-VISION // YOLOv8n CUSTOM VEHICLE DATASET & TRAINING SETUP")
    print("=" * 70)
    yaml_path = create_dataset_scaffold()
    train_args = get_training_args(yaml_path)

    print("\n[+] Recommended Hyperparameters:")
    for k, v in train_args.items():
        print(f"    - {k:<12}: {v}")

    print("\n[+] Fine-Tuning Command:")
    print(
        '    python -c "from ultralytics import YOLO; m = YOLO(\'models/yolov8n.pt\'); '
        'm.train(data=\'datasets/accident_traffic/data.yaml\', epochs=50, batch=16, imgsz=640)"'
    )
    print("=" * 70)


if __name__ == "__main__":
    main()

