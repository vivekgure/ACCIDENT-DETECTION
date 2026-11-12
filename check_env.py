"""
AURA-VISION System Environment & Dependency Checker.
Validates required libraries, verifies NVIDIA CUDA / CPU execution fallback,
monitors RTX 4050 VRAM via NVML, validates YOLOv8n weights integrity,
and executes a quick timed inference test.
"""

import importlib.metadata
import platform
import shutil
import sys
import time
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config


def check_system_environment() -> bool:
    print("=" * 72)
    print("   AURA-VISION // SYSTEM DIAGNOSTICS & MODEL VERIFICATION")
    print("=" * 72)

    print(f"[+] OS     : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"[+] Python : {sys.version.split()[0]} ({sys.executable})")

    packages = {
        "torch": "PyTorch",
        "torchvision": "TorchVision",
        "ultralytics": "Ultralytics YOLO",
        "cv2": "OpenCV (cv2)",
        "numpy": "NumPy",
        "flask": "Flask Web Server",
        "PIL": "Pillow (PIL)",
        "scipy": "SciPy",
        "lap": "LAP (ByteTrack)",
    }

    all_ok = True
    print("\n--- Core Dependency Verification ---")
    for mod_name, label in packages.items():
        try:
            mod = __import__(mod_name)
            pkg_lookup = {"cv2": "opencv-python", "PIL": "pillow"}.get(mod_name, mod_name)
            try:
                version = importlib.metadata.version(pkg_lookup)
            except Exception:
                version = getattr(mod, "__version__", "Installed")
            print(f"  [OK]   {label:<22}: v{version}")
        except ImportError as e:
            print(f"  [FAIL] {label:<22}: NOT FOUND ({e})")
            all_ok = False

    print("\n--- Hardware Acceleration & VRAM ---")
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"  [CUDA] NVIDIA GPU Active     : {device_name}")
            print(f"  [CUDA] PyTorch CUDA Version  : {torch.version.cuda}")
        else:
            print("  [CPU]  PyTorch CUDA inactive in current build; using CPU fallback.")

        try:
            import pynvml

            pynvml.nvmlInit()
            if pynvml.nvmlDeviceGetCount() > 0:
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                gpu_name = pynvml.nvmlDeviceGetName(h)
                if isinstance(gpu_name, bytes):
                    gpu_name = gpu_name.decode("utf-8", errors="ignore")
                print(f"  [NVML] GPU Hardware          : {gpu_name}")
                print(f"  [NVML] Total VRAM            : {mem.total / (1024 ** 2):.1f} MB")
                print(f"  [NVML] Used / Free VRAM      : {mem.used / (1024 ** 2):.1f} MB / {mem.free / (1024 ** 2):.1f} MB")
        except Exception as e:
            print(f"  [NVML] Info                  : {e}")
    except Exception as e:
        print(f"  [WARN] Hardware check warning: {e}")

    print("\n--- Lightweight YOLO Model Verification ---")
    try:
        from ultralytics import YOLO
        import numpy as np

        target_model = config.YOLO_MODEL_PATH
        if target_model.exists() and not zipfile.is_zipfile(target_model):
            print(f"  [WARN] Cached weights {target_model} failed zip integrity check. Removing...")
            target_model.unlink(missing_ok=True)

        if not target_model.exists():
            print(f"  [INFO] Downloading {config.YOLO_MODEL_NAME} to models/...")
            _ = YOLO(config.YOLO_MODEL_NAME)
            cand = Path(config.YOLO_MODEL_NAME)
            if cand.exists() and cand.resolve() != target_model.resolve():
                shutil.move(str(cand), str(target_model))
            print(f"  [OK]   Model cached at       : {target_model}")
        else:
            print(f"  [OK]   Primary weights       : {target_model} ({target_model.stat().st_size / (1024 ** 2):.1f} MB)")

        if config.YOLO_FALLBACK_MODEL_PATH.exists():
            print(
                f"  [OK]   Alternative weights   : {config.YOLO_FALLBACK_MODEL_PATH} "
                f"({config.YOLO_FALLBACK_MODEL_PATH.stat().st_size / (1024 ** 2):.1f} MB)"
            )

        dummy_img = np.zeros((config.YOLO_INPUT_SIZE, config.YOLO_INPUT_SIZE, 3), dtype=np.uint8)
        model = YOLO(str(target_model))
        import torch

        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        _ = model.predict(dummy_img, imgsz=config.YOLO_INPUT_SIZE, device=dev, verbose=False)
        t0 = time.perf_counter()
        _ = model.predict(dummy_img, imgsz=config.YOLO_INPUT_SIZE, device=dev, verbose=False)
        dt = (time.perf_counter() - t0) * 1000.0
        print(f"  [OK]   Warm inference test   : {dt:.1f} ms ({1000.0 / dt:.1f} FPS) on {dev.upper()}")
    except Exception as e:
        print(f"  [FAIL] Model verification error: {e}")
        all_ok = False

    print("=" * 72)
    if all_ok:
        print("   >>> ALL SYSTEMS READY: Launch with 'python app.py' or run.bat")
    else:
        print("   >>> ATTENTION: Resolve warnings or run 'pip install -r requirements.txt'")
    print("=" * 72)
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if check_system_environment() else 1)
