import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def detect_board(image_path, output_dir=None, debug=False):
    """
    Detect chess board from an input image and output the board coordinates.
    
    Args:
        image_path: Path to input image.
        output_dir: Directory to save output images.
        debug: If True, display intermediate results.
    
    Returns:
        Corners of the chess board and intersection points.
    """
    # Read input image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Keep original image for drawing results
    original_img = img.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Edge detection using Canny
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Save edge detection result
    edge_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Step 2: Line detection using Hough Transform
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    if lines is None:
        raise ValueError("No lines detected in the image")
    
    # Separate horizontal and vertical lines
    horizontal_lines = []
    vertical_lines = []
    
    for line in lines:
        rho, theta = line[0]
        # Classify lines as horizontal or vertical based on their angle
        if ((theta > np.pi/4 - 0.2) and (theta < np.pi/4 + 0.2)) or \
           ((theta > 3*np.pi/4 - 0.2) and (theta < 3*np.pi/4 + 0.2)):
            # These are lines close to 45 degrees, not suitable for chess board
            continue
        elif (theta < 0.2) or (theta > np.pi - 0.2) or (abs(theta - np.pi) < 0.2):
            # Nearly vertical lines
            vertical_lines.append((rho, theta))
        elif abs(theta - np.pi/2) < 0.2:
            # Nearly horizontal lines
            horizontal_lines.append((rho, theta))
    
    # Create an image with detected lines
    lines_img = original_img.copy()
    
    # Draw horizontal lines (green)
    for rho, theta in horizontal_lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(lines_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for horizontal
    
    # Draw vertical lines (blue)
    for rho, theta in vertical_lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(lines_img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for vertical
    
    # Step 3: Cluster similar lines
    def cluster_lines(lines, max_distance=30):
        """Cluster lines that are close to each other."""
        if not lines:
            return []
        
        clusters = []
        current_cluster = [lines[0]]
        
        # Sort lines by rho for easier clustering
        sorted_lines = sorted(lines, key=lambda line: line[0])
        
        for i in range(1, len(sorted_lines)):
            current_rho = sorted_lines[i][0]
            prev_rho = sorted_lines[i-1][0]
            
            if abs(current_rho - prev_rho) < max_distance:
                current_cluster.append(sorted_lines[i])
            else:
                # Average the cluster
                if current_cluster:
                    avg_rho = sum([r for r, _ in current_cluster]) / len(current_cluster)
                    avg_theta = sum([t for _, t in current_cluster]) / len(current_cluster)
                    clusters.append((avg_rho, avg_theta))
                
                # Start a new cluster
                current_cluster = [sorted_lines[i]]
        
        # Don't forget the last cluster
        if current_cluster:
            avg_rho = sum([r for r, _ in current_cluster]) / len(current_cluster)
            avg_theta = sum([t for _, t in current_cluster]) / len(current_cluster)
            clusters.append((avg_rho, avg_theta))
        
        return clusters
    
    # Cluster similar lines
    clustered_horizontal = cluster_lines(horizontal_lines)
    clustered_vertical = cluster_lines(vertical_lines)
    
    # Create an image with clustered lines
    clustered_img = original_img.copy()
    
    # Draw clustered horizontal lines (green)
    for rho, theta in clustered_horizontal:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(clustered_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw clustered vertical lines (blue)
    for rho, theta in clustered_vertical:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(clustered_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # Step 4: Calculate intersection points
    intersection_points = []
    
    # Find intersections between horizontal and vertical lines
    for h_rho, h_theta in clustered_horizontal:
        for v_rho, v_theta in clustered_vertical:
            # Convert from polar to Cartesian coordinates
            h_a = np.cos(h_theta)
            h_b = np.sin(h_theta)
            v_a = np.cos(v_theta)
            v_b = np.sin(v_theta)
            
            # Solve the system of equations to find intersection
            det = h_a * v_b - h_b * v_a
            if abs(det) < 1e-10:
                # Lines are parallel, no intersection
                continue
            
            # Calculate intersection point
            x = (v_b * h_rho - h_b * v_rho) / det
            y = (-v_a * h_rho + h_a * v_rho) / det
            
            # Check if point is within image bounds
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                intersection_points.append((int(x), int(y)))
    
    # Draw intersection points on the image (red)
    intersection_img = clustered_img.copy()
    for point in intersection_points:
        cv2.circle(intersection_img, point, 5, (0, 0, 255), -1)
    
    # Save or display results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, "1_original.jpg"), original_img)
        cv2.imwrite(os.path.join(output_dir, "2_edges.jpg"), edge_img)
        cv2.imwrite(os.path.join(output_dir, "3_lines.jpg"), lines_img)
        cv2.imwrite(os.path.join(output_dir, "4_intersections.jpg"), intersection_img)
    
    if debug:
        plt.figure(figsize=(20, 10))
        
        plt.subplot(221)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        
        plt.subplot(222)
        plt.title("Edge Detection")
        plt.imshow(edges, cmap='gray')
        
        plt.subplot(223)
        plt.title("Line Detection")
        plt.imshow(cv2.cvtColor(lines_img, cv2.COLOR_BGR2RGB))
        
        plt.subplot(224)
        plt.title("Intersection Points")
        plt.imshow(cv2.cvtColor(intersection_img, cv2.COLOR_BGR2RGB))
        
        plt.tight_layout()
        plt.show()
    
    return intersection_points

def main():
    parser = argparse.ArgumentParser(description='Chess Board Detection')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--output', default='output', help='Path to output directory')
    parser.add_argument('--debug', action='store_true', help='Display intermediate results')
    
    args = parser.parse_args()
    
    try:
        intersection_points = detect_board(args.image, args.output, args.debug)
        print(f"Found {len(intersection_points)} intersection points")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
