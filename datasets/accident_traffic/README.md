# AURA-VISION Road User & Custom Vehicle Dataset Scaffold

## Annotation Format (YOLO Normalized Format)
Each `.txt` label file contains one line per object:
`<class_id> <x_center> <y_center> <width> <height>` (normalized 0.0 - 1.0)

## Class Indices
- **0**: `person`
- **1**: `bicycle`
- **2**: `car`
- **3**: `motorcycle`
- **4**: `scooter`
- **5**: `bus`
- **6**: `truck`
- **7**: `auto_rickshaw`
- **8**: `accident_contact`

## Note on Pretrained Weights vs Custom Classes
Pretrained COCO `yolov8n.pt` natively detects `person`, `bicycle`, `car`, `motorcycle`,
`bus`, and `truck`. Fine-tune on this scaffold to add dedicated `scooter`, `auto_rickshaw`
(three-wheeler), and `accident_contact` classes. `VehicleDetector` automatically maps these
class names from `model.names` at load time.
