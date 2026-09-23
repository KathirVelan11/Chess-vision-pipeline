# Computer Vision — Chess Playing Robot

A computer vision pipeline that takes a photo of a chess board and works out the position on it, output as FEN notation. Built around the approach in Wölflein & Arandjelović's paper ["Determining Chess Game State From an Image"](https://doi.org/10.3390/jimaging7060094) (2021).

The pipeline has three stages:

1. **Board localisation** — finds the board in the image and rectifies it (Canny edge detection + Hough transform + RANSAC-based corner finding)
2. **Occupancy classification** — for each of the 64 squares, decides whether it's empty or has a piece on it (ResNet-based binary classifier)
3. **Piece classification** — for occupied squares, identifies which piece it is (YOLO-based, 12-13 piece classes)

`chess_recognition_system.py` at the repo root ties all three stages together and produces a final FEN string.

## Project structure

```
Computer-Vision/
├── board-localisation/       # stage 1: board detection & corner finding
│   ├── board_detection.py
│   ├── images/                # sample input photos
│   └── results/                # pipeline step visualizations
├── occupancy-classification/ # stage 2: empty vs occupied squares
│   ├── code1.py
│   ├── testing.py
│   └── models/                # training curves, confusion matrix
├── piece-classification/     # stage 3: which piece is on a square
│   ├── train_model.py         # trains the YOLO model
│   ├── yolo_train.py
│   ├── predict_image.py       # run on a static image
│   ├── predict_camera.py      # run on a live camera feed
│   ├── webcam_detection.py
│   ├── chess_detection/       # training run outputs (weights, curves, val batches)
│   ├── runs/                  # more training run outputs
│   └── diagnostics/            # sample captured frames
├── chess_recognition_system.py  # wires the three stages together
├── demo.py                    # CLI entry point for the full pipeline
├── config.json                 # model paths & pipeline settings
├── utils.py                    # shared helpers (config loading, model path finding, logging)
├── test_chess_recognition.py   # test suite / benchmark script
└── requirements / setup files per module
```

Board localisation, occupancy classification, and piece classification started as three separate, independently developed modules (originally on their own branches) before being wired together — that's still visible in how each folder is organised and how each one manages its own dependencies.

## Setup

```bash
python -m venv chess
source chess/bin/activate   # Windows: chess\Scripts\activate

pip install torch torchvision opencv-python ultralytics albumentations matplotlib seaborn scikit-learn
```

Each module folder also has its own `requirements.txt` if you only need to run that piece in isolation.

Trained model weights (`.pt` files) are tracked with [Git LFS](https://git-lfs.github.com/). If you have `git-lfs` installed, a normal `git clone` pulls them automatically. If a `.pt` file looks tiny (~130 bytes, a text pointer instead of the real weights), run:

```bash
git lfs install
git lfs pull
```

### Dataset

`piece-classification` was trained against a Roboflow chess pieces dataset — see `piece-classification/data.yaml` for the exact source, version, and class list (13 piece classes). It's pulled straight from Roboflow, not stored in this repo:

- https://universe.roboflow.com/joseph-nelson/chess-pieces-new/dataset/24

An earlier experiment also pulled in COCO 2017 (train/val/test images plus segmentation labels) for general object-detection work. That data isn't kept in this repo — it was ~1.7GB and mostly unrelated to chess — but if you want to reproduce that experiment, grab it directly from the COCO site:

```bash
# validation images (~1GB, this is the one actually used for validation)
wget http://images.cocodataset.org/zips/val2017.zip

# training images (~18GB) and test images (~6GB), only needed if retraining from scratch
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/test2017.zip

# segmentation labels
wget https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels-segments.zip
```

Unzip into `piece-classification/datasets/coco/images/<split>/` to match the layout the original scripts expected.

## Running it

Generate a default config (auto-detects model paths):

```bash
python demo.py --create-config
```

Run the full pipeline on an image:

```bash
python demo.py --image board-localisation/images/chess_image_1.jpg
python demo.py --image path/to/your/photo.jpg --output result.json
```

Run tests / a benchmark:

```bash
python test_chess_recognition.py --image your_chess_image.jpg
python test_chess_recognition.py --image-dir test_images/ --pattern "*.jpg"
python test_chess_recognition.py --image your_chess_image.jpg --benchmark --iterations 10
```

Python API:

```python
from chess_recognition_system import ChessRecognitionSystem, load_config

config = load_config('config.json')
system = ChessRecognitionSystem(config)
result = system.recognize_chess_state('path/to/image.jpg')

if result['success']:
    print(result['fen'])
```

Each module can also be run standalone — see `board-localisation/test.py`, `occupancy-classification/testing.py`, and `piece-classification/predict_image.py` / `predict_camera.py`.

## Config

`config.json` controls model paths and pipeline behaviour:

```json
{
  "piece_model_path": "./piece-classification/chess_detection/chess_model_20251007_185940/weights/best.pt",
  "occupancy_model_path": "occupancy-classification/models/best_model.pth",
  "square_size": 80,
  "piece_confidence": 0.3,
  "piece_iou": 0.5,
  "board_detection": {
    "edge_method": "scharr",
    "hough_threshold": 30
  }
}
```

## Output

The pipeline outputs standard FEN notation, e.g. `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1`, plus a JSON result with per-square occupancy, identified pieces, and confidence scores.

## Troubleshooting

- **"No model found"** — run `python demo.py --create-config` to auto-detect model paths, or check `config.json` points at a real `.pt`/`.pth` file.
- **Weak board detection** — try switching `edge_method` between `"scharr"` and `"multi_scale"`, and make sure the board fills a good chunk of the frame.
- **Weights missing after clone (a `.pt` file is ~130 bytes instead of several MB)** — install `git-lfs` and run `git lfs pull`.

## Reference

Wölflein, G.; Arandjelović, O. "Determining Chess Game State From an Image." *Journal of Imaging* 2021, 7, 94. https://doi.org/10.3390/jimaging7060094

This project is for educational/research purposes — check the original paper and any third-party model licenses before commercial use.
