# train.py
from ultralytics import YOLO

# 1. Create YOLO model architecture (nano for speed)
model = YOLO("yolo12n.yaml")  # change to yolo12s.yaml for bigger model

# 2. Train from scratch (NO pretrained weights)
model.train(
    device=-1,
    data="ChessDataset/data.yaml",  # path to dataset config
    epochs=100,         # try 200–300 for better results
    imgsz=640,
    batch=16,
    workers=1,
    pretrained=False,   # important: start from scratch
    optimizer="SGD"     # can also try AdamW
)

# 3. Evaluate model
metrics = model.val()
print(metrics)

# 4. Run inference on a sample image
results = model.predict(
    source="ChessDataset/test/images/sample.jpg",  # change to your test image
    imgsz=640,
    conf=0.25
)

results.show()   # show in window
results.save()   # save output in runs/detect/predict
