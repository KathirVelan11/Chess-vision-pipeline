import argparse
from pathlib import Path
import sys
from typing import Optional

from ultralytics import YOLO

# Set this to your image path if you want to hardcode it and avoid using the terminal.
# Example: DEFAULT_IMAGE = Path(r"ChessDataset\test\images\my_test_image.jpg")
DEFAULT_IMAGE = Path(r"ChessDataset\test\images\IMG_0159_JPG.rf.f0d34122f8817d538e396b04f2b70d33.jpg")  # Leave as None to auto-pick from dataset folders
# Inference tuning defaults (raise conf/imgsz or enable TTA to reduce mislabels like rook->knight)
DEFAULT_IMGSZ = 960
DEFAULT_CONF = 0.50
DEFAULT_IOU = 0.50
USE_TTA = True  # enables test-time augmentation (slower but often more accurate)
AGNOSTIC_NMS = False  # keep class-aware NMS

# Square-mapping settings (set BOARD_RECT to your board region in pixels)
# BOARD_RECT: (x, y, w, h) in image pixels defining the chessboard area.
# Example: BOARD_RECT = (50, 20, 400, 400)
BOARD_RECT: Optional[tuple[int, int, int, int]] = None
# Orientation: set which side of the image corresponds to white's back rank and the a-file
WHITE_AT_BOTTOM = True   # True if white pieces are at the bottom of the image
A_FILE_ON_LEFT = True    # True if 'a' file is on the left side of the image


def _resolve_weights() -> Optional[Path]:
    """Find a suitable weights file automatically.

    Preference order:
    1) runs/chess_gpu_v1/weights/best.pt
    2) runs/chess_gpu_v1/weights/last.pt
    3) Most recently modified runs/**/weights/best.pt
    4) Most recently modified runs/**/weights/last.pt
    Returns Path if found, else None.
    """
    runs_dir = Path('runs')
    preferred = [
        runs_dir / 'chess_gpu_v1' / 'weights' / 'best.pt',
        runs_dir / 'chess_gpu_v1' / 'weights' / 'last.pt',
    ]
    for p in preferred:
        if p.exists():
            return p

    # Search all runs for best.pt then last.pt
    def newest(glob_pattern: str) -> Optional[Path]:
        try:
            candidates = list(runs_dir.rglob(glob_pattern)) if runs_dir.exists() else []
        except Exception:
            candidates = []
        if not candidates:
            return None
        # pick most recently modified file
        candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return candidates[0]

    for pattern in ['weights/best.pt', 'weights/last.pt']:
        p = newest(pattern)
        if p and p.exists():
            return p
    return None


def _auto_pick_image() -> Optional[Path]:
    """Pick a sample image from common dataset folders if DEFAULT_IMAGE is not set.

    Search order prefers test images, then valid, then train.
    """
    search_dirs = [
        Path('ChessDataset') / 'test' / 'images',
        Path('ChessDataset') / 'valid' / 'images',
        Path('ChessDataset') / 'train' / 'images',
        Path('test') / 'images',
        Path('valid') / 'images',
        Path('train') / 'images',
        Path('images'),
    ]
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    candidates: list[Path] = []
    for d in search_dirs:
        if d.exists():
            for p in d.iterdir():
                if p.is_file() and p.suffix.lower() in exts:
                    candidates.append(p)
    if not candidates:
        return None
    # Choose the most recently modified file
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def _pick_from_dir(dir_path: Path) -> Optional[Path]:
    """Pick most recently modified image file from the given directory."""
    if not dir_path.exists() or not dir_path.is_dir():
        return None
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if not files:
        return None
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return files[0]


def _compute_squares_for_boxes(xywh: list[list[float]], img_w: int, img_h: int) -> list[Optional[str]]:
    """Map detection centers to chessboard squares, if BOARD_RECT is set.

    Returns list of algebraic squares (e.g., 'e4') or None if mapping not available.
    """
    if not BOARD_RECT:
        return [None] * len(xywh)
    x0, y0, w, h = BOARD_RECT
    if w <= 0 or h <= 0:
        return [None] * len(xywh)
    files_lr = ['a','b','c','d','e','f','g','h'] if A_FILE_ON_LEFT else ['h','g','f','e','d','c','b','a']

    squares: list[Optional[str]] = []
    for cx, cy, bw, bh in xywh:
        # Normalize center into board rect
        rel_x = (cx - x0) / float(w)
        rel_y = (cy - y0) / float(h)
        # If outside rect, no square
        if rel_x < 0 or rel_x >= 1 or rel_y < 0 or rel_y >= 1:
            squares.append(None)
            continue
        col = int(rel_x * 8)
        row = int(rel_y * 8)
        col = max(0, min(7, col))
        row = max(0, min(7, row))
        file_char = files_lr[col]
        if WHITE_AT_BOTTOM:
            rank = 8 - row
        else:
            rank = 1 + row
        squares.append(f"{file_char}{rank}")
    return squares


def _print_detections(res, squares: Optional[list[Optional[str]]] = None) -> None:
    """Print a summary of detected pieces to the terminal.

    - Counts per class in canonical chess order (white: K,Q,R,B,N,P then black)
    - Detailed list ordered top-to-bottom, then left-to-right (board-like order)
    - Also shows a confidence-sorted list
    """
    try:
        boxes = getattr(res, 'boxes', None)
        if boxes is None or len(boxes) == 0:
            print("No pieces detected.")
            return

        # Extract tensors -> Python lists
        clses = boxes.cls.cpu().tolist() if hasattr(boxes, 'cls') else []
        confs = boxes.conf.cpu().tolist() if hasattr(boxes, 'conf') and boxes.conf is not None else [None] * len(clses)
        names = res.names if hasattr(res, 'names') and res.names else {}
        labels = [names.get(int(c), str(int(c))) for c in clses]

        # Centers for ordering by position (top->bottom, left->right)
        try:
            xywh = boxes.xywh.cpu().tolist()
        except Exception:
            # Fallback to xyxy -> approximate centers
            xyxy = boxes.xyxy.cpu().tolist()
            xywh = [
                [(x1 + x2) / 2.0, (y1 + y2) / 2.0, (x2 - x1), (y2 - y1)]
                for x1, y1, x2, y2 in xyxy
            ]

    # Counts per class
        from collections import Counter
        counts = Counter(labels)
        total = sum(counts.values())
        print(f"Pieces detected (total {total}):")

        # Canonical order if names look like 'white-king', else alphabetical
        canonical_order = [
            'white-king', 'white-queen', 'white-rook', 'white-bishop', 'white-knight', 'white-pawn',
            'black-king', 'black-queen', 'black-rook', 'black-bishop', 'black-knight', 'black-pawn',
        ]
        labels_present = set(labels)
        used_order = [n for n in canonical_order if n in labels_present]
        remaining = sorted([n for n in labels_present if n not in used_order])
        final_order = used_order + remaining
        for name in final_order:
            print(f"- {name}: {counts[name]}")

        # Detailed by board-like position (top->bottom, then left->right)
        detailed = list(zip(labels, confs, xywh))
        detailed.sort(key=lambda x: (x[2][1], x[2][0]))  # sort by center y, then center x
        print("Detailed (by position):")
        for idx, (name, conf, (cx, cy, w, h)) in enumerate(detailed):
            sq = squares[idx] if squares and idx < len(squares) else None
            if conf is None:
                if sq:
                    print(f"  {name} @ ({int(cx)}, {int(cy)}) -> {sq}")
                else:
                    print(f"  {name} @ ({int(cx)}, {int(cy)})")
            else:
                if sq:
                    print(f"  {name} ({conf:.2f}) @ ({int(cx)}, {int(cy)}) -> {sq}")
                else:
                    print(f"  {name} ({conf:.2f}) @ ({int(cx)}, {int(cy)})")

        # Detailed by confidence descending
        pairs = list(zip(labels, confs))
        pairs.sort(key=lambda x: (x[1] is None, -(x[1] or 0.0)))
        print("Detailed (by confidence):")
        for name, conf in pairs:
            if conf is None:
                print(f"  {name}")
            else:
                print(f"  {name} ({conf:.2f})")
    except Exception as e:
        print(f"Warning: failed to print detections: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Predict on a single image (popup only, no files are written)"
    )
    # Make positional image optional; if not provided, use DEFAULT_IMAGE or auto-pick
    parser.add_argument('image', nargs='?', help='Path to input image (optional if DEFAULT_IMAGE is set)')
    parser.add_argument('--imgsz', type=int, default=DEFAULT_IMGSZ)
    parser.add_argument('--conf', type=float, default=DEFAULT_CONF)
    parser.add_argument('--iou', type=float, default=DEFAULT_IOU)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--show_ms', type=int, default=0, help='Auto-close preview window after N milliseconds (0 = wait for key)')
    args = parser.parse_args()

    if args.image:
        img_path = Path(args.image)
    elif DEFAULT_IMAGE is not None:
        img_path = Path(DEFAULT_IMAGE)
    else:
        auto = _auto_pick_image()
        if auto is None:
            print(
                "No image path provided. Set DEFAULT_IMAGE in predict_image.py or pass an image path,\n"
                "and ensure there is at least one image under ChessDataset/*/images/ to auto-pick.",
                file=sys.stderr,
            )
            sys.exit(1)
        img_path = auto

    # If a directory is provided, try to pick an image from it
    if img_path.is_dir():
        picked = _pick_from_dir(img_path)
        if picked is None:
            print(f"No images found in directory: {img_path}", file=sys.stderr)
            sys.exit(1)
        img_path = picked

    if not img_path.exists() or not img_path.is_file():
        print(f"Input image not found or is not a file: {img_path}", file=sys.stderr)
        sys.exit(1)

    weights = _resolve_weights()
    if not weights:
        print(
            "Could not find model weights automatically under 'runs/**/weights/'.\n"
            "Please train the model first or place weights at runs/chess_gpu_v1/weights/best.pt",
            file=sys.stderr,
        )
        sys.exit(1)

    model = YOLO(str(weights))
    res = model.predict(
        source=str(img_path),
        imgsz=args.imgsz,
        device=args.device or None,
        conf=args.conf,
        iou=args.iou,
        save=False,
        augment=USE_TTA,
        agnostic_nms=AGNOSTIC_NMS,
        verbose=False,
    )[0]

    # Print pieces list to terminal (with squares if BOARD_RECT configured)
    img_w = res.orig_shape[1]
    img_h = res.orig_shape[0]
    try:
        xywh_list = res.boxes.xywh.cpu().tolist()
    except Exception:
        xyxy = res.boxes.xyxy.cpu().tolist() if res.boxes is not None else []
        xywh_list = [
            [(x1 + x2) / 2.0, (y1 + y2) / 2.0, (x2 - x1), (y2 - y1)]
            for x1, y1, x2, y2 in xyxy
        ]
    squares = _compute_squares_for_boxes(xywh_list, img_w, img_h)
    _print_detections(res, squares)

    plotted = res.plot()  # numpy array (BGR)

    # Overlay square labels near detection centers if available
    if any(squares) and len(squares) == len(xywh_list):
        try:
            import cv2
            for (cx, cy, _, _), sq in zip(xywh_list, squares):
                if not sq:
                    continue
                cv2.putText(plotted, sq, (int(cx)+4, int(cy)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
        except Exception:
            pass

    # Always show window (no saving to disk)
    try:
        import cv2
    except ImportError:
        print("opencv-python not installed. Please: pip install opencv-python", file=sys.stderr)
        sys.exit(1)
    title = 'Predictions' if args.show_ms == 0 else f'Predictions - auto close in {args.show_ms} ms'
    cv2.imshow(title, plotted)
    if args.show_ms and args.show_ms > 0:
        # Auto-close after N milliseconds
        cv2.waitKey(args.show_ms)
    else:
        # Keep window open until user closes it (or presses 'q'/ESC)
        while True:
            # If window is closed by the user, break out
            try:
                prop = cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE)
            except Exception:
                prop = -1  # treat as closed
            if prop < 1:
                break
            key = cv2.waitKey(50) & 0xFF
            if key in (27, ord('q')):  # ESC or 'q' also closes
                break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
