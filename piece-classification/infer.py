from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Run inference on an image (returns list of Results objects)
results = model.predict(
    source="C:\\Users\\Acer\\OneDrive\\Documents\\Semester-5\\Computer Vision\\Project\\ChessDataset\\test\\images\\IMG_0159_JPG.rf.f0d34122f8817d538e396b04f2b70d33.jpg",  # change path if needed
    imgsz=640,
    conf=0.25
)

# Loop through results
for r in results:
    r.show()                       # display result in a window
    # save with bounding boxes

print("✅ Inference completed! Check 'inference_results' folder for output.")
