import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time
import argparse
import os

# ------------------- Argument Parser -------------------
parser = argparse.ArgumentParser(description="Real-time Chess Piece Detection with YOLOv8")
parser.add_argument("--weights", type=str, default="runs/chess_gpu_v1/weights/best.pt", help="Path to YOLO model weights")
parser.add_argument("--camera", type=str, default="0", help="Camera index or video path")
parser.add_argument("--gamma", type=float, default=1.2, help="Gamma correction factor (>1 brightens, <1 darkens)")
parser.add_argument("--clahe", action="store_true", help="Enable CLAHE contrast enhancement")
parser.add_argument("--smooth", action="store_true", help="Enable temporal smoothing for stable detection")
parser.add_argument("--show_fps", action="store_true", help="Display FPS on frame")
parser.add_argument("--size_filter", action="store_true", help="Filter unrealistic bounding box sizes")
parser.add_argument("--imgsz", type=int, default=640, help="Inference size (pixels)")
parser.add_argument("--tta", action="store_true", help="Use test-time augmentation for more accurate predictions")
parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
parser.add_argument("--sharpen", action="store_true", help="Apply sharpening to enhance edges")
parser.add_argument("--denoise", action="store_true", help="Apply denoising to reduce image noise")
parser.add_argument("--resize", type=int, default=0, help="Resize input to this width while preserving aspect ratio")
parser.add_argument("--tracker_persist", action="store_true", help="Use more persistent tracking across frames")
parser.add_argument("--voting_window", type=int, default=3, help="Number of frames for class voting (with tracking)")
args = parser.parse_args()

# ------------------- Device -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")

# ------------------- Model Loading -------------------
if not os.path.exists(args.weights):
    raise FileNotFoundError(f"Model weights not found at {args.weights}")

print(f"🔍 Loading model from: {args.weights}...")
model = YOLO(args.weights)
model.to(device)

# Use higher confidence thresholds for commonly confused pieces
# Default confidence is already set by args.conf
model_classes = model.names
print(f"✅ Model loaded with {len(model_classes)} classes")

# ------------------- Video Capture with Fallback -------------------
def try_camera(camera_index):
    """Try to open a camera and return success status and capture object."""
    try:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened() and cap.read()[0]:
            print(f"✅ Successfully opened camera {camera_index}")
            return True, cap
        else:
            cap.release()
            return False, None
    except Exception:
        return False, None

# Try primary camera
camera_src = 0 if args.camera == "0" else args.camera
success, cap = try_camera(camera_src)

# If primary camera fails, try backup cameras
if not success:
    print(f"⚠️ Primary camera '{camera_src}' failed. Trying fallback cameras...")
    for backup_idx in [0, 1, 2]:
        if str(backup_idx) != str(camera_src):
            print(f"Trying camera index {backup_idx}...")
            success, cap = try_camera(backup_idx)
            if success:
                break

if not success:
    raise RuntimeError("❌ Unable to open any camera. Please check your camera connections.")

# ------------------- Advanced Preprocessing Functions -------------------
def adjust_gamma(image, gamma=1.2):
    """Apply gamma correction to brighten/darken image."""
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def enhance_contrast(frame, clip_limit=3.0):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def sharpen_image(image, amount=0.3):
    """Apply sharpening to make edges more defined."""
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    return sharpened

def denoise_image(image, strength=7):
    """Apply non-local means denoising to improve image quality."""
    return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)

def resize_image(image, target_width=1080):
    """Resize image while preserving aspect ratio."""
    height, width = image.shape[:2]
    ratio = target_width / width
    target_height = int(height * ratio)
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

# ------------------- Enhanced Smoothing with Class Stability -------------------
previous_boxes = []
previous_classes = []
previous_confs = []
class_history = {}  # Track class predictions over time for stability

def smooth_boxes(current_boxes, current_classes, current_confs, alpha=0.6):
    global previous_boxes, previous_classes, previous_confs, class_history
    
    # Initialize if first frame
    if not previous_boxes:
        previous_boxes = current_boxes
        previous_classes = current_classes
        previous_confs = current_confs
        return current_boxes, current_classes, current_confs
        
    # IoU calculation helper function
    def calculate_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        iou = intersection / float(box1_area + box2_area - intersection + 1e-6)
        return iou
    
    # Match current boxes with previous boxes using IoU
    smoothed_boxes = []
    smoothed_classes = []
    smoothed_confs = []
    
    # Track matched boxes to avoid duplicates
    used_prev = set()
    
    for i, curr_box in enumerate(current_boxes):
        max_iou = 0.3  # Minimum IoU threshold for matching
        best_match = -1
        
        # Find best matching previous box
        for j, prev_box in enumerate(previous_boxes):
            if j in used_prev:
                continue
                
            iou = calculate_iou(curr_box, prev_box)
            if iou > max_iou:
                max_iou = iou
                best_match = j
        
        if best_match >= 0:
            # We found a match - smooth the box
            used_prev.add(best_match)
            smooth_box = tuple(alpha * np.array(previous_boxes[best_match]) + (1-alpha) * np.array(curr_box))
            
            # Class stability logic
            curr_cls = current_classes[i]
            prev_cls = previous_classes[best_match]
            box_id = f"{int(curr_box[0])},{int(curr_box[1])}"
            
            if box_id not in class_history:
                class_history[box_id] = {curr_cls: 1}
            else:
                if curr_cls not in class_history[box_id]:
                    class_history[box_id][curr_cls] = 1
                else:
                    class_history[box_id][curr_cls] += 1
            
            # Use class with highest count in history
            best_cls = curr_cls
            max_count = 0
            for cls, count in class_history[box_id].items():
                if count > max_count:
                    max_count = count
                    best_cls = cls
            
            # Smooth confidence (leaning toward max observed confidence)
            smooth_conf = max(alpha * previous_confs[best_match], (1-alpha) * current_confs[i])
            
            smoothed_boxes.append(smooth_box)
            smoothed_classes.append(best_cls)
            smoothed_confs.append(smooth_conf)
        else:
            # No match - keep current detection
            smoothed_boxes.append(curr_box)
            smoothed_classes.append(current_classes[i])
            smoothed_confs.append(current_confs[i])
    
    # Cleanup old history entries
    if len(class_history) > 100:
        keys_to_remove = list(class_history.keys())[:50]
        for k in keys_to_remove:
            class_history.pop(k, None)
    
    # Update previous states
    previous_boxes = smoothed_boxes
    previous_classes = smoothed_classes
    previous_confs = smoothed_confs
    
    return smoothed_boxes, smoothed_classes, smoothed_confs

# ------------------- Main Loop -------------------
print("🎥 Starting detection... Press 'q' to quit.")

fps_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not received, exiting...")
        break

    # Enhanced preprocessing pipeline
    if args.resize > 0:
        frame = resize_image(frame, target_width=args.resize)
    
    if args.denoise:
        frame = denoise_image(frame, strength=8)  # Medium-strength denoise
        
    if args.clahe:
        frame = enhance_contrast(frame, clip_limit=3.0)
        
    if args.gamma != 1.0:
        frame = adjust_gamma(frame, gamma=args.gamma)
        
    if args.sharpen:
        frame = sharpen_image(frame, amount=0.3)

    # Run YOLO inference with improved parameters
    results = model.predict(
        source=frame, 
        imgsz=args.imgsz, 
        conf=args.conf, 
        device=device, 
        augment=args.tta,  # Test-time augmentation
        verbose=False
    )
    annotated_frame = frame.copy()
    current_boxes = []
    current_classes = []
    current_confs = []

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1

            # Filter by size/aspect ratio (optional)
            if args.size_filter:
                if w < 20 or h < 20 or w / h > 2.5 or h / w > 2.5:
                    continue

            # Store detection information
            current_boxes.append((x1, y1, x2, y2))
            current_classes.append(cls)
            current_confs.append(conf)
    
    # Apply enhanced smoothing with configurable persistence
    if args.smooth and current_boxes:
        # Use more persistent tracking if requested (higher alpha = more weight to history)
        alpha = 0.8 if args.tracker_persist else 0.6
        voting_window = args.voting_window if args.tracker_persist else 3
        
        # Update global class_history window size
        for key in class_history:
            # Keep only the most recent votes up to voting_window
            if isinstance(class_history[key], dict):
                for cls in class_history[key]:
                    class_history[key][cls] = min(class_history[key][cls], voting_window)
        
        draw_boxes, draw_classes, draw_confs = smooth_boxes(
            current_boxes, current_classes, current_confs, alpha=alpha
        )
    else:
        draw_boxes = current_boxes
        draw_classes = current_classes
        draw_confs = current_confs
    
    # Draw all detections with improved visualization
    detected_pieces = {}  # Keep count of detected pieces by type
    
    for box, cls, conf in zip(draw_boxes, draw_classes, draw_confs):
        x1, y1, x2, y2 = map(int, box)
        class_name = model.names[cls]
        label = f"{class_name} {conf:.2f}"
        
        # Color coding by piece type
        if 'white' in class_name:
            color = (0, 255, 0)  # Green for white pieces
        else:
            color = (0, 165, 255)  # Orange for black pieces
        
        # Add to piece count
        if class_name not in detected_pieces:
            detected_pieces[class_name] = 1
        else:
            detected_pieces[class_name] += 1
        
        # Draw detection with clearer visuals
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw filled background for text
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(
            annotated_frame, 
            (x1, y1 - text_size[1] - 10), 
            (x1 + text_size[0], y1), 
            color, 
            -1
        )
        cv2.putText(
            annotated_frame, 
            label, 
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (0, 0, 0),  # Black text on colored background
            2
        )
        
    # Display piece counts at the bottom of frame
    if detected_pieces:
        pieces_text = " | ".join([f"{name}: {count}" for name, count in detected_pieces.items()])
        cv2.putText(
            annotated_frame,
            f"Detected: {pieces_text}",
            (10, annotated_frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    # FPS display
    if args.show_fps:
        new_time = time.time()
        fps = 1 / (new_time - fps_time)
        fps_time = new_time
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Show frame
    cv2.imshow("♟️ Chess Piece Detection", annotated_frame)

    # Quit
    key = cv2.waitKey(1) & 0xFF
    if key in [27, ord('q')]:  # ESC or q
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Detection finished successfully.")
