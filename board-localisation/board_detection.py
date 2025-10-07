"""
Optimized Chess Board Detection Pipeline
--------------------------------------
1. Random Edge Detection (Multi-scale or Scharr)
2. Enhanced Probabilistic Hough Transform
3. Direct Line Intersection
4. Parallel Processing

Author: GitHub Copilot
Date: October 7, 2025
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import time
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Tuple, Dict
import torch  # Add this import

@dataclass
class Point:
    x: int
    y: int

def check_cuda():
    """Check CUDA availability"""
    if torch.cuda.is_available():
        # Also verify OpenCV CUDA support
        build_info = str(cv2.getBuildInformation())
        cuda_support = 'CUDA' in build_info
        return cuda_support and torch.cuda.is_available()
    return False

def detect_edges(image: np.ndarray, method: str = 'scharr') -> np.ndarray:
    """Unified edge detection with selected method and CUDA support"""
    use_cuda = check_cuda()
    if use_cuda:
        print("Using CUDA for edge detection")
        # Convert image to GPU
        cuda_img = cv2.cuda_GpuMat()
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    if use_cuda:
        cuda_gray = cv2.cuda_GpuMat()
        cuda_gray.upload(gray)
        cuda_blur = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (5, 5), 0)
        gray = cuda_blur.apply(cuda_gray)
    else:
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    if method == 'multi_scale':
        if use_cuda:
            # CUDA-accelerated Canny
            cuda_canny = cv2.cuda.createCannyEdgeDetector(50, 150)
            edges1 = cuda_canny.detect(gray)
            cuda_canny = cv2.cuda.createCannyEdgeDetector(30, 100)
            edges2 = cuda_canny.detect(gray)
            cuda_canny = cv2.cuda.createCannyEdgeDetector(70, 200)
            edges3 = cuda_canny.detect(gray)
            
            # Download results from GPU
            edges = [edges1.download(), edges2.download(), edges3.download()]
        else:
            # CPU parallel processing
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [
                    executor.submit(cv2.Canny, gray, 50, 150),
                    executor.submit(cv2.Canny, gray, 30, 100),
                    executor.submit(cv2.Canny, gray, 70, 200)
                ]
                edges = [future.result() for future in futures]
        
        result = cv2.bitwise_or(edges[0], edges[1])
        result = cv2.bitwise_or(result, edges[2])
    else:  # scharr
        if use_cuda:
            # CUDA-accelerated Scharr
            cuda_sobelx = cv2.cuda.createSobelFilter(cv2.CV_8UC1, cv2.CV_64F, 1, 0)
            cuda_sobely = cv2.cuda.createSobelFilter(cv2.CV_8UC1, cv2.CV_64F, 0, 1)
            
            gradX = cuda_sobelx.apply(gray).download()
            gradY = cuda_sobely.apply(gray).download()
        else:
            with ThreadPoolExecutor(max_workers=2) as executor:
                scharr_x = executor.submit(cv2.Scharr, gray, cv2.CV_64F, 1, 0)
                scharr_y = executor.submit(cv2.Scharr, gray, cv2.CV_64F, 0, 1)
                gradX = scharr_x.result()
                gradY = scharr_y.result()
        
        magnitude = np.sqrt(gradX**2 + gradY**2)
        magnitude = np.uint8(magnitude * 255 / magnitude.max())
        _, result = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
    
    # Common enhancement (on CPU as morphology is often faster on CPU)
    kernel = np.ones((3, 3), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    
    return result

def enhanced_probabilistic_hough(edges: np.ndarray) -> List[np.ndarray]:
    """Optimized Enhanced Probabilistic Hough Transform with CUDA support"""
    use_cuda = check_cuda()
    
    if use_cuda:
        print("Using CUDA for line detection")
        cuda_edges = cv2.cuda_GpuMat()
        cuda_edges.upload(edges)
        
        # CUDA blur and morphology
        cuda_blur = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (3, 3), 0)
        edges_gpu = cuda_blur.apply(cuda_edges)
        
        # HoughLinesP on GPU
        gpu_lines_detector = cv2.cuda.createHoughSegmentDetector(
            rho=1.0,
            theta=np.pi/180.0,
            minLineLength=30,
            maxLineGap=5
        )
        lines_gpu = gpu_lines_detector.detect(edges_gpu)
        lines = lines_gpu.download()
    else:
        edges = cv2.GaussianBlur(edges, (3, 3), 0)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        
        lines = cv2.HoughLinesP(
            edges, 
            rho=1,
            theta=np.pi/180,
            threshold=30,
            minLineLength=30,
            maxLineGap=5
        )
    
    return [] if lines is None else [line[0] for line in lines]

def direct_line_intersection(lines: List[np.ndarray], img_shape: Tuple[int, int], 
                           chunk_size: int = 100) -> List[Point]:
    """Parallelized direct line intersection detection"""
    h, w = img_shape[:2]
    intersections = []
    
    def process_chunk(chunk_lines: List[np.ndarray]) -> List[Point]:
        chunk_intersections = []
        for i, line1 in enumerate(chunk_lines):
            x1, y1, x2, y2 = line1
            for line2 in lines[i+1:]:  # Compare with all remaining lines
                x3, y3, x4, y4 = line2
                
                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if abs(denom) < 1e-5:
                    continue
                
                px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / denom
                py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / denom
                
                if (0 <= px <= w and 0 <= py <= h and
                    min(x1, x2) <= px <= max(x1, x2) and
                    min(y1, y2) <= py <= max(y1, y2) and
                    min(x3, x4) <= px <= max(x3, x4) and
                    min(y3, y4) <= py <= max(y3, y4)):
                    chunk_intersections.append(Point(int(px), int(py)))
        
        return chunk_intersections
    
    # Split lines into chunks for parallel processing
    chunks = [lines[i:i+chunk_size] for i in range(0, len(lines), chunk_size)]
    
    # Process chunks in parallel
    with ThreadPoolExecutor(max_workers=min(len(chunks), os.cpu_count() or 1)) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
        for future in futures:
            intersections.extend(future.result())
    
    return intersections

def save_individual_images(img_rgb: np.ndarray, edges: np.ndarray, lines: List[np.ndarray], 
                          intersections: List[Point], output_dir: str, 
                          edge_method: str, timing: Dict[str, float]):
    """Save individual images for each stage of the pipeline"""
    
    # 1. Original Image
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    plt.title('Original Image', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_original.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # 2. Edge Detection
    plt.figure(figsize=(10, 8))
    plt.imshow(edges, cmap='gray')
    plt.title(f'{edge_method.title()} Edge Detection\nTime: {timing["edge"]:.3f}s', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_edges.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # 3. Lines Detection
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    for line in lines:
        x1, y1, x2, y2 = line
        plt.plot([x1, x2], [y1, y2], 'g-', linewidth=1, alpha=0.5)
    plt.title(f'Lines Detected: {len(lines)}\nTime: {timing["line"]:.3f}s', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_lines.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # 4. Intersections
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    for point in intersections:
        plt.plot(point.x, point.y, 'ro', markersize=4)
    plt.title(f'Intersections Found: {len(intersections)}\nTime: {timing["intersect"]:.3f}s', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_intersections.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # 5. Lines + Intersections Combined
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    for line in lines:
        x1, y1, x2, y2 = line
        plt.plot([x1, x2], [y1, y2], 'g-', linewidth=1, alpha=0.3)
    for point in intersections:
        plt.plot(point.x, point.y, 'ro', markersize=3)
    plt.title(f'Combined: {len(lines)} lines, {len(intersections)} intersections', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_combined.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\nIndividual stage images saved to: {output_dir}")

def detect_board(image_path: str, output_dir: str = 'board-localisation/results'):
    """Main detection pipeline with parallel processing"""
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    
    # Read and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Resize for faster processing
    max_dim = 800
    h, w = img.shape[:2]
    if h > max_dim or w > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"Processing image of size {img.shape}")
    
    # Randomly choose edge detection method
    edge_method = random.choice(['multi_scale', 'scharr'])
    print(f"\nUsing {edge_method} edge detection...")
    
    # Edge detection
    start_edge = time.time()
    edges = detect_edges(img, edge_method)
    edge_time = time.time() - start_edge
    print(f"Edge detection time: {edge_time:.3f}s")
    
    # Line detection
    start_line = time.time()
    lines = enhanced_probabilistic_hough(edges)
    line_time = time.time() - start_line
    print(f"Line detection time: {line_time:.3f}s")
    print(f"Lines detected: {len(lines)}")
    
    # Intersection detection
    start_intersect = time.time()
    intersections = direct_line_intersection(lines, img.shape)
    intersect_time = time.time() - start_intersect
    print(f"Intersection detection time: {intersect_time:.3f}s")
    print(f"Intersections found: {len(intersections)}")
    
    # Timing dictionary
    timing = {
        'edge': edge_time,
        'line': line_time,
        'intersect': intersect_time,
        'total': time.time() - start_time
    }
    
    # Save individual stage images
    save_individual_images(img_rgb, edges, lines, intersections, output_dir, edge_method, timing)
    
    # Visualization - Combined view
    plt.figure(figsize=(20, 5))
    
    # Original Image
    plt.subplot(141)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    # Edge Detection
    plt.subplot(142)
    plt.imshow(edges, cmap='gray')
    plt.title(f'{edge_method.title()} Edges\n{edge_time:.3f}s')
    plt.axis('off')
    
    # Lines
    plt.subplot(143)
    plt.imshow(img_rgb)
    for line in lines:
        x1, y1, x2, y2 = line
        plt.plot([x1, x2], [y1, y2], 'g-', linewidth=1, alpha=0.5)
    plt.title(f'Lines Detected\n{len(lines)} lines, {line_time:.3f}s')
    plt.axis('off')
    
    # Intersections
    plt.subplot(144)
    plt.imshow(img_rgb)
    for point in intersections:
        plt.plot(point.x, point.y, 'ro', markersize=4)
    plt.title(f'Intersections\n{len(intersections)} points, {intersect_time:.3f}s')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '00_pipeline_overview.png'), dpi=200, bbox_inches='tight')
    plt.show()
    
    total_time = time.time() - start_time
    print(f"\nTotal processing time: {total_time:.3f}s")
    
    return {
        'edges': edges,
        'lines': lines,
        'intersections': intersections,
        'timing': timing
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimized chess board detection')
    parser.add_argument('--image_path', type=str, default='board-localisation/images/chess_image_1.jpg')
    parser.add_argument('--output_dir', type=str, default='board-localisation/results')
    args = parser.parse_args()
    
    detect_board(args.image_path, args.output_dir)