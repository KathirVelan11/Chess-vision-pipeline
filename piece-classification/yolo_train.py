import argparse
import os
from pathlib import Path
import sys
import yaml

# Prefer CPU-safe import: ultralytics handles device selection internally
try:
    from ultralytics import YOLO
except Exception as e:
    print("Failed to import ultralytics. Did you install requirements?\n" \
          "pip install -r requirements.txt", file=sys.stderr)
    raise


def _resolve_path(base: Path, value: str, fallback_subpath: str) -> str:
    """Resolve dataset path robustly.
    1) If absolute: return as-is.
    2) If relative to data.yaml directory and exists: use it.
    3) Fallback to ChessDataset/<fallback_subpath> if present.
    """
    p = Path(value)
    if p.is_absolute() and p.exists():
        return str(p)
    # Try relative to the yaml file dir
    rel = (base / value).resolve()
    if rel.exists():
        return str(rel)
    # Fallback to dataset standard layout
    ds_root = base
    fb = (ds_root / fallback_subpath).resolve()
    if fb.exists():
        return str(fb)
    # As a last resort, return the original string (ultralytics may still handle it)
    return str(rel)


def sanitize_data_yaml(data_yaml_path: Path) -> Path:
    """Load data.yaml, fix relative paths to absolute if needed, and write a temp sanitized yaml.
    Returns path to sanitized yaml file.
    """
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    base = data_yaml_path.parent

    # Map keys to fallback subpaths
    mapping = {
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
    }

    for k, sub in mapping.items():
        if k in data and data[k]:
            data[k] = _resolve_path(base, str(data[k]), sub)

    # Ensure nc and names are present
    if 'nc' not in data or 'names' not in data:
        raise ValueError("data.yaml must define 'nc' and 'names'.")

    out_path = data_yaml_path.with_name(data_yaml_path.stem + '.abs.yaml')
    with open(out_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return out_path


def main():
    parser = argparse.ArgumentParser(description='Train YOLO from scratch on Chess dataset')
    parser.add_argument('--data_yaml', type=str, default=str(Path('ChessDataset') / 'data.yaml'),
                        help='Path to data.yaml')
    parser.add_argument('--model', type=str, default='yolov8n.yaml',
                        help='Model config to start from scratch (e.g., yolov8n.yaml, yolov8s.yaml)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=-1, help='Batch size (-1 = auto)')
    parser.add_argument('--project', type=str, default='runs', help='Project directory for outputs')
    parser.add_argument('--name', type=str, default='chess_yolo_scratch', help='Run name')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience (epochs)')
    parser.add_argument('--device', type=str, default=None, help='cuda, 0, 0,1, or cpu (None=auto)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--cos_lr', action='store_true', help='Use cosine LR schedule')
    parser.add_argument('--lr0', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01, help='Final OneCycleLR ratio or cosine min lr factor')
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--warmup_epochs', type=float, default=3.0)
    parser.add_argument('--imgsz_train', type=int, default=None, help='Override imgsz for training')
    args = parser.parse_args()

    data_yaml_path = Path(args.data_yaml).resolve()
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml_path}")

    # Fix paths and write sanitized yaml
    abs_yaml = sanitize_data_yaml(data_yaml_path)
    print(f"Using dataset yaml: {abs_yaml}")

    # Initialize model from config (randomly initialized weights)
    model = YOLO(args.model)

    # Train
    train_kwargs = dict(
        data=str(abs_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz_train or args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        patience=args.patience,
        device=args.device or None,  # None lets ultralytics auto-select CPU/GPU
        seed=args.seed,
        workers=args.workers,
        pretrained=False,  # from scratch
        optimizer='SGD',
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        cos_lr=args.cos_lr,
        # Useful training-time augmentations (defaults are good; can tweak further)
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.1,
        close_mosaic=10,
        save=True,
        save_period=-1,
        plots=True,
        # Validation happens every epoch by default
    )

    results = model.train(**train_kwargs)
    print(results)

    print("Training complete. Best weights (if any):")
    # Ultralytics saves best at project/name/weights/best.pt
    out_dir = Path(args.project) / 'detect' / args.name / 'weights'
    if out_dir.exists():
        print(out_dir / 'best.pt')
    else:
        # Newer ultralytics may save under project/name/weights directly
        alt_dir = Path(args.project) / args.name / 'weights'
        print(alt_dir / 'best.pt')


if __name__ == '__main__':
    main()
