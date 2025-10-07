import argparse
import glob
import os
import time
from typing import Optional, List, Dict

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def find_latest_weights(folder: str = 'chess_detection') -> Optional[str]:
    # Search for best.pt or last.pt in subfolders and return the most recent
    candidates = glob.glob(os.path.join(folder, '**', 'weights', '*.pt'), recursive=True)
    if not candidates:
        return None
    # prefer best.pt over last.pt if both exist in same folder
    # sort by modification time
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def draw_label(img, text, xy, bg_color=(0, 0, 0), text_color=(255, 255, 255)):
    x, y = xy
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x, y - h - 6), (x + w, y), bg_color, -1)
    cv2.putText(img, text, (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)


def box_iou(b1, b2) -> float:
    # b = [x1,y1,x2,y2]
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    inter = w * h
    area1 = max(0.0, (b1[2] - b1[0])) * max(0.0, (b1[3] - b1[1]))
    area2 = max(0.0, (b2[2] - b2[0])) * max(0.0, (b2[3] - b2[1]))
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def class_wise_suppression(dets: List[Dict], iou_thresh: float = 0.5) -> List[Dict]:
    # dets: list of {'box':[x1,y1,x2,y2], 'cls':cls_id, 'conf':conf, 'name':name}
    out: List[Dict] = []
    by_class: Dict[int, List[Dict]] = {}
    for d in dets:
        by_class.setdefault(d['cls'], []).append(d)

    for cls_id, items in by_class.items():
        items.sort(key=lambda x: x['conf'], reverse=True)
        keep: List[Dict] = []
        for it in items:
            should_keep = True
            for k in keep:
                if box_iou(it['box'], k['box']) > iou_thresh:
                    should_keep = False
                    break
            if should_keep:
                keep.append(it)
        out.extend(keep)
    return out


def cross_class_suppression(dets: List[Dict], iou_thresh: float = 0.6) -> List[Dict]:
    # Keep highest-confidence detection when boxes of different classes overlap strongly
    dets_sorted = sorted(dets, key=lambda x: x['conf'], reverse=True)
    keep: List[Dict] = []
    for d in dets_sorted:
        discard = False
        for k in keep:
            if box_iou(d['box'], k['box']) > iou_thresh:
                # If a higher-confidence box already kept overlaps this one, discard current
                discard = True
                break
        if not discard:
            keep.append(d)
    return keep


def main():
    parser = argparse.ArgumentParser(description='Real-time webcam detection for chess pieces')
    parser.add_argument('--weights', type=str, default=None, help='Path to .pt weights file')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU threshold')
    parser.add_argument('--device', type=str, default=None, help='Device to run on, e.g. 0 or cpu')
    parser.add_argument('--source', type=str, default='0', help='Video source (0 for webcam or path to video)')
    parser.add_argument('--augment', action='store_true', help='Enable augmentation (TTA) during inference')
    parser.add_argument('--imgsz', type=int, default=640, help='Inference image size')
    parser.add_argument('--cross_iou', type=float, default=0.5, help='IoU threshold for cross-class suppression')
    parser.add_argument('--class_thresh', type=str, default=None, help='Optional JSON dict of class-specific conf thresholds, e.g. "{\"white-pawn\":0.5,\"black-rook\":0.6}"')
    args = parser.parse_args()

    # Determine weights
    weights = args.weights
    if not weights:
        weights = find_latest_weights()
    if weights:
        print(f'Using weights: {weights}')
    else:
        print('No trained weights found under chess_detection/. Falling back to yolov8n.pt')
        weights = 'yolov8n.pt'

    # Load model
    try:
        model = YOLO(weights)
        print(f'Model loaded from {weights}')
    except Exception as e:
        print(f'Failed to load model {weights}: {e}')
        return

    # Device selection
    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else 'cpu')
    print(f'Running detection on device: {device}')

    # Open video source
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f'Error: could not open source {args.source}')
        return

    # Colors (fallback palette)
    palette = np.random.randint(0, 255, size=(256, 3), dtype=np.uint8)

    prev = 0
    fps = 0.0
    print("Starting webcam detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Stream ended or cannot fetch frame')
            break

        # Inference
        results = model.predict(frame, conf=args.conf, iou=args.iou, device=device, augment=args.augment, imgsz=args.imgsz, verbose=False)

        # Collect detections and apply class-wise suppression to reduce duplicates
        dets = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                name = r.names[cls_id]
                dets.append({'box': [x1, y1, x2, y2], 'cls': cls_id, 'conf': conf, 'name': name})

        # Apply class-wise suppression first
        filtered = class_wise_suppression(dets, iou_thresh=args.iou)

        # Apply per-class confidence thresholds if provided
        class_thresh_map = {}
        if args.class_thresh:
            try:
                import json

                class_thresh_map = json.loads(args.class_thresh)
            except Exception:
                print('Failed to parse --class_thresh, ignoring')

        filtered = [f for f in filtered if f['conf'] >= class_thresh_map.get(f['name'], args.conf)]

        # Cross-class suppression to remove overlapping boxes of different classes
        filtered = cross_class_suppression(filtered, iou_thresh=args.cross_iou)

        for it in filtered:
            x1, y1, x2, y2 = it['box']
            cls_id = it['cls']
            conf = it['conf']
            name = it['name']
            color = tuple(map(int, palette[cls_id % len(palette)]))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            draw_label(frame, f'{name} {conf:.2f}', (x1, y1), bg_color=color)

        # Compute and show FPS
        cur = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / (cur - prev)) if prev else 0.0
        prev = cur
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Chess Piece Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print('Detection stopped.')


if __name__ == '__main__':
    main()