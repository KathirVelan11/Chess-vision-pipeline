import argparse
import logging
import os
import platform
import sys
from datetime import datetime

import torch
from ultralytics import YOLO


def get_device(force_cpu: bool = False):
    if force_cpu:
        return 'cpu'
    return 0 if torch.cuda.is_available() else 'cpu'


def setup_logging(verbosity: int = 1):
    level = logging.DEBUG if verbosity > 1 else logging.INFO
    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
        level=level,
    )


def resolve_data_path(data_arg: str):
    if data_arg and os.path.isabs(data_arg):
        data_path = data_arg
    else:
        data_path = os.path.abspath(data_arg or os.path.join(os.getcwd(), 'data.yaml'))
    return data_path


def find_weights(project: str, name: str):
    weights_dir = os.path.join(project, name, 'weights')
    best = os.path.join(weights_dir, 'best.pt')
    last = os.path.join(weights_dir, 'last.pt')
    if os.path.exists(best):
        return best
    if os.path.exists(last):
        return last
    return None


def main():
    parser = argparse.ArgumentParser(description='Train or evaluate YOLOv8 on the chess dataset')
    parser.add_argument('--data', type=str, default='data.yaml', help='Path to data.yaml')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Base model or checkpoint to start from')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--project', type=str, default='chess_detection', help='Project folder for saving runs')
    parser.add_argument('--name', type=str, default='chess_model', help='Run name')
    parser.add_argument('--exist_ok', action='store_true', help='Overwrite existing run')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained flag (ultralytics API)')
    parser.add_argument('--eval', action='store_true', help='Only run evaluation (no training)')
    parser.add_argument('--cpu', action='store_true', help='Force CPU even if CUDA is available')
    parser.add_argument('-v', '--verbosity', action='count', default=0, help='Increase verbosity')

    args = parser.parse_args()

    setup_logging(args.verbosity)
    logging.info('Starting train_model.py')

    logging.info(f'Python platform: {platform.system()} {platform.release()}')
    logging.info(f'PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available() and not args.cpu:
        try:
            logging.info(f'CUDA device: {torch.cuda.get_device_name(0)}')
        except Exception:
            logging.debug('Could not query CUDA device name', exc_info=True)

    data_path = resolve_data_path(args.data)
    logging.info(f'Resolved dataset config: {data_path}')
    if not os.path.exists(data_path):
        logging.error(f'Data file not found at: {data_path}\nPlease ensure your dataset YAML (data.yaml) exists and paths inside it are correct.')
        sys.exit(2)

    device = get_device(force_cpu=args.cpu)
    logging.info(f'Using device: {device}')

    # Load model (pretrained weights or checkpoint)
    try:
        model = YOLO(args.model)
        logging.info(f'Loaded model: {args.model}')
    except Exception as e:
        logging.error(f'Failed to load model {args.model}: {e}')
        sys.exit(3)

    # If user requested only evaluation, attempt to find trained weights first
    if args.eval:
        ckpt = find_weights(args.project, args.name)
        if ckpt:
            logging.info(f'Found checkpoint for evaluation: {ckpt}')
            model = YOLO(ckpt)
        else:
            logging.warning('No trained checkpoint found; evaluating the currently loaded model instead')

        logging.info('Running evaluation...')
        try:
            res = model.val()
            logging.info('Evaluation finished')
            print(res)  # ultralytics returns a Results object; printing helps show summary
        except Exception as e:
            logging.error(f'Evaluation failed: {e}', exc_info=True)
        return

    # Train
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = args.name + ('_' + timestamp if not args.exist_ok else '')
    logging.info(f'Starting training: project={args.project} name={run_name}')

    try:
        model.train(
            data=data_path,
            epochs=args.epochs,
            imgsz=args.imgsz,
            device=device,
            batch=args.batch,
            workers=args.workers,
            patience=args.patience,
            project=args.project,
            name=run_name,
            exist_ok=args.exist_ok,
            pretrained=args.pretrained,
            optimizer='auto',
            verbose=True,
        )
    except Exception as e:
        logging.error(f'Training failed: {e}', exc_info=True)
        sys.exit(4)

    # Locate weights
    weights = find_weights(args.project, run_name)
    if weights:
        logging.info(f'Training complete. Best weights at: {weights}')
    else:
        logging.warning('Training finished but could not find best weights file. Check the project folder for outputs.')

    # Run validation on the trained model
    try:
        logging.info('Running validation on the trained model...')
        val_res = model.val()
        logging.info('Validation finished')
        print(val_res)
    except Exception as e:
        logging.error(f'Validation failed: {e}', exc_info=True)


if __name__ == '__main__':
    main()