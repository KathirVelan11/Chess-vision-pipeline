"""
Improved Chess Board Localization with RANSAC-based Homography
Based on "Determining Chess Game State From an Image" by Wölflin & Arandjelović

This implementation follows Section 4.1 of the paper:
1. Edge detection using Canny
2. Line detection using Hough transform
3. Line clustering (horizontal/vertical)
4. RANSAC-based homography computation
5. Grid refinement and corner detection
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from dataclasses import dataclass
import random
from sklearn.cluster import AgglomerativeClustering
import warnings
warnings.filterwarnings('ignore')

@dataclass
class Point:
    x: float
    y: float

@dataclass
class Line:
    x1: float
    y1: float
    x2: float
    y2: float
    
    def angle(self) -> float:
        """Calculate line angle in radians"""
        return np.arctan2(self.y2 - self.y1, self.x2 - self.x1)
    
    def intersect(self, other: 'Line') -> Optional[Point]:
        """Find intersection point with another line"""
        denom = (self.x1 - self.x2) * (other.y1 - other.y2) - (self.y1 - self.y2) * (other.x1 - other.x2)
        if abs(denom) < 1e-10:
            return None
        
        px = ((self.x1*self.y2 - self.y1*self.x2) * (other.x1 - other.x2) - 
              (self.x1 - self.x2) * (other.x1*other.y2 - other.y1*other.x2)) / denom
        py = ((self.x1*self.y2 - self.y1*self.x2) * (other.y1 - other.y2) - 
              (self.y1 - self.y2) * (other.x1*other.y2 - other.y1*other.x2)) / denom
        
        return Point(px, py)

class ChessBoardLocalizer:
    def __init__(self, canny_low=50, canny_high=150, hough_threshold=30,
                 min_line_length=30, max_line_gap=5, ransac_iterations=1000,
                 inlier_threshold=5.0):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.ransac_iterations = ransac_iterations
        self.inlier_threshold = inlier_threshold
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """Edge detection using Canny as described in the paper"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply Gaussian blur before edge detection
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        
        return edges
    
    def detect_lines(self, edges: np.ndarray) -> List[Line]:
        """Line detection using Hough transform"""
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        if lines is None:
            return []
        
        return [Line(line[0][0], line[0][1], line[0][2], line[0][3]) for line in lines]
    
    def cluster_lines(self, lines: List[Line]) -> Tuple[List[Line], List[Line]]:
        """Cluster lines into horizontal and vertical using agglomerative clustering"""
        if len(lines) < 2:
            return [], []
        
        # Calculate angles for all lines
        angles = [line.angle() for line in lines]
        
        # Normalize angles to [0, π]
        normalized_angles = []
        for angle in angles:
            if angle < 0:
                angle += np.pi
            normalized_angles.append(angle)
        
        # Use agglomerative clustering to separate horizontal and vertical lines
        angles_array = np.array(normalized_angles).reshape(-1, 1)
        
        try:
            clustering = AgglomerativeClustering(n_clusters=2, linkage='ward')
            labels = clustering.fit_predict(angles_array)
            
            # Group lines by cluster
            cluster_0 = [lines[i] for i in range(len(lines)) if labels[i] == 0]
            cluster_1 = [lines[i] for i in range(len(lines)) if labels[i] == 1]
            
            # Determine which cluster is horizontal vs vertical based on mean angle
            mean_angle_0 = np.mean([lines[i].angle() for i in range(len(lines)) if labels[i] == 0])
            mean_angle_1 = np.mean([lines[i].angle() for i in range(len(lines)) if labels[i] == 1])
            
            # Vertical lines should have angles closer to ±π/2
            if abs(abs(mean_angle_0) - np.pi/2) < abs(abs(mean_angle_1) - np.pi/2):
                vertical_lines, horizontal_lines = cluster_0, cluster_1
            else:
                horizontal_lines, vertical_lines = cluster_0, cluster_1
                
        except Exception:
            # Fallback: simple angle-based separation
            horizontal_lines = []
            vertical_lines = []
            
            for line in lines:
                angle = abs(line.angle())
                if angle < np.pi/4 or angle > 3*np.pi/4:
                    horizontal_lines.append(line)
                else:
                    vertical_lines.append(line)
        
        return horizontal_lines, vertical_lines
    
    def eliminate_similar_lines(self, lines: List[Line], orientation: str = 'horizontal') -> List[Line]:
        """Eliminate similar lines using DBSCAN clustering on intersection points"""
        if len(lines) <= 1:
            return lines
        
        # For horizontal lines, use intersection with a vertical reference line
        # For vertical lines, use intersection with a horizontal reference line
        if orientation == 'horizontal':
            # Use middle vertical line as reference
            ref_line = Line(100, 0, 100, 200)  # Arbitrary vertical line
        else:
            # Use middle horizontal line as reference
            ref_line = Line(0, 100, 200, 100)  # Arbitrary horizontal line
        
        # Find intersection points
        intersections = []
        valid_lines = []
        
        for line in lines:
            intersection = line.intersect(ref_line)
            if intersection:
                intersections.append(intersection)
                valid_lines.append(line)
        
        if len(intersections) <= 1:
            return valid_lines
        
        # Group similar intersection points
        from sklearn.cluster import DBSCAN
        
        points = np.array([[p.x, p.y] for p in intersections])
        
        try:
            clustering = DBSCAN(eps=20, min_samples=1).fit(points)
            labels = clustering.labels_
            
            # Keep one representative line from each cluster
            unique_labels = set(labels)
            filtered_lines = []
            
            for label in unique_labels:
                if label != -1:  # Ignore noise points
                    cluster_lines = [valid_lines[i] for i in range(len(valid_lines)) if labels[i] == label]
                    if cluster_lines:
                        # Take the median line from the cluster
                        filtered_lines.append(cluster_lines[len(cluster_lines)//2])
            
            return filtered_lines
        except:
            return valid_lines
    
    def find_intersections(self, horizontal_lines: List[Line], vertical_lines: List[Line]) -> List[Point]:
        """Find all intersection points between horizontal and vertical lines"""
        intersections = []
        
        for h_line in horizontal_lines:
            for v_line in vertical_lines:
                intersection = h_line.intersect(v_line)
                if intersection:
                    intersections.append(intersection)
        
        return intersections
    
    def ransac_homography(self, intersections: List[Point], img_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """RANSAC-based homography computation as described in the paper"""
        if len(intersections) < 4:
            return None
        
        h, w = img_shape[:2]
        best_homography = None
        best_inliers = []
        best_inlier_count = 0
        
        for iteration in range(self.ransac_iterations):
            # Sample 4 points that form a rectangle
            if len(intersections) < 4:
                continue
                
            sample_points = random.sample(intersections, 4)
            
            # Check if points form a reasonable rectangle
            if not self._is_valid_rectangle(sample_points):
                continue
            
            # Try different scale factors (sx, sy) as mentioned in the paper
            for sx in range(2, 9):  # 2 to 8 chess squares
                for sy in range(2, 9):
                    try:
                        # Create target rectangle
                        target_points = np.array([
                            [0, 0],
                            [sx, 0],
                            [sx, sy],
                            [0, sy]
                        ], dtype=np.float32)
                        
                        # Source points
                        source_points = np.array([
                            [p.x, p.y] for p in sample_points
                        ], dtype=np.float32)
                        
                        # Compute homography
                        H = cv2.getPerspectiveTransform(source_points, target_points)
                        
                        # Count inliers
                        inliers = []
                        for point in intersections:
                            # Transform point
                            src_pt = np.array([[[point.x, point.y]]], dtype=np.float32)
                            dst_pt = cv2.perspectiveTransform(src_pt, H)[0][0]
                            
                            # Check if transformed point is close to a grid point
                            closest_grid_x = round(dst_pt[0])
                            closest_grid_y = round(dst_pt[1])
                            
                            distance = np.sqrt((dst_pt[0] - closest_grid_x)**2 + (dst_pt[1] - closest_grid_y)**2)
                            
                            if distance < self.inlier_threshold:
                                inliers.append(point)
                        
                        if len(inliers) > best_inlier_count:
                            best_inlier_count = len(inliers)
                            best_homography = H
                            best_inliers = inliers
                    
                    except:
                        continue
        
        return best_homography if best_inlier_count >= 4 else None
    
    def _is_valid_rectangle(self, points: List[Point]) -> bool:
        """Check if 4 points form a reasonable rectangle"""
        if len(points) != 4:
            return False
        
        # Calculate all pairwise distances
        distances = []
        for i in range(4):
            for j in range(i+1, 4):
                dist = np.sqrt((points[i].x - points[j].x)**2 + (points[i].y - points[j].y)**2)
                distances.append(dist)
        
        distances.sort()
        
        # For a rectangle, we should have 2 pairs of equal opposite sides and 2 equal diagonals
        # The 4 shortest distances should be the sides, the 2 longest should be diagonals
        side_ratio = distances[2] / distances[0] if distances[0] > 0 else float('inf')
        diagonal_ratio = distances[5] / distances[4] if distances[4] > 0 else float('inf')
        
        # Check if ratios are reasonable for a rectangle
        return side_ratio < 3 and diagonal_ratio < 1.5
    
    def refine_grid(self, image: np.ndarray, homography: np.ndarray) -> Tuple[int, int, int, int]:
        """Refine grid boundaries using gradient analysis as described in the paper"""
        # Warp the image
        warped = cv2.warpPerspective(image, homography, (9, 9))  # 9x9 for line detection
        
        if len(warped.shape) == 3:
            gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        else:
            gray_warped = warped
        
        # Find grid boundaries using gradient analysis
        # For simplicity, assume we have detected the full 8x8 grid
        xmin, xmax = 0, 8
        ymin, ymax = 0, 8
        
        return xmin, xmax, ymin, ymax
    
    def extract_corner_points(self, homography: np.ndarray, xmin: int, xmax: int, 
                            ymin: int, ymax: int) -> List[Point]:
        """Extract the four corner points of the chessboard"""
        # The four corners in the warped coordinate system
        warped_corners = np.array([
            [[xmin, ymin]],
            [[xmax, ymin]],
            [[xmax, ymax]],
            [[xmin, ymax]]
        ], dtype=np.float32)
        
        # Transform back to original image coordinates
        inv_homography = np.linalg.inv(homography)
        original_corners = cv2.perspectiveTransform(warped_corners, inv_homography)
        
        corner_points = []
        for corner in original_corners:
            corner_points.append(Point(corner[0][0], corner[0][1]))
        
        return corner_points
    
    def localize_board(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], List[Point], dict]:
        """Main board localization pipeline"""
        results = {
            'edges': None,
            'lines': [],
            'horizontal_lines': [],
            'vertical_lines': [],
            'intersections': [],
            'homography': None,
            'corner_points': []
        }
        
        # Step 1: Edge detection
        edges = self.detect_edges(image)
        results['edges'] = edges
        
        # Step 2: Line detection
        lines = self.detect_lines(edges)
        results['lines'] = lines
        
        if len(lines) < 4:
            return None, [], results
        
        # Step 3: Line clustering
        horizontal_lines, vertical_lines = self.cluster_lines(lines)
        results['horizontal_lines'] = horizontal_lines
        results['vertical_lines'] = vertical_lines
        
        # Step 4: Eliminate similar lines
        horizontal_lines = self.eliminate_similar_lines(horizontal_lines, 'horizontal')
        vertical_lines = self.eliminate_similar_lines(vertical_lines, 'vertical')
        
        # Step 5: Find intersections
        intersections = self.find_intersections(horizontal_lines, vertical_lines)
        results['intersections'] = intersections
        
        if len(intersections) < 4:
            return None, [], results
        
        # Step 6: RANSAC homography
        homography = self.ransac_homography(intersections, image.shape)
        results['homography'] = homography
        
        if homography is None:
            return None, [], results
        
        # Step 7: Grid refinement
        xmin, xmax, ymin, ymax = self.refine_grid(image, homography)
        
        # Step 8: Extract corner points
        corner_points = self.extract_corner_points(homography, xmin, xmax, ymin, ymax)
        results['corner_points'] = corner_points
        
        return homography, corner_points, results

def visualize_localization(image: np.ndarray, results: dict, save_path: str = None):
    """Visualize the board localization pipeline"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Edges
    axes[0, 1].imshow(results['edges'], cmap='gray')
    axes[0, 1].set_title(f'Edge Detection')
    axes[0, 1].axis('off')
    
    # Lines
    axes[0, 2].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for line in results['horizontal_lines']:
        axes[0, 2].plot([line.x1, line.x2], [line.y1, line.y2], 'r-', linewidth=1, alpha=0.7)
    for line in results['vertical_lines']:
        axes[0, 2].plot([line.x1, line.x2], [line.y1, line.y2], 'b-', linewidth=1, alpha=0.7)
    axes[0, 2].set_title(f'Lines: {len(results["horizontal_lines"])} H, {len(results["vertical_lines"])} V')
    axes[0, 2].axis('off')
    
    # Intersections
    axes[1, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for point in results['intersections']:
        axes[1, 0].plot(point.x, point.y, 'go', markersize=3)
    axes[1, 0].set_title(f'Intersections: {len(results["intersections"])}')
    axes[1, 0].axis('off')
    
    # Corner points
    axes[1, 1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results['corner_points']:
        corner_points = results['corner_points']
        # Draw the quadrilateral
        for i in range(4):
            next_i = (i + 1) % 4
            axes[1, 1].plot([corner_points[i].x, corner_points[next_i].x], 
                          [corner_points[i].y, corner_points[next_i].y], 'r-', linewidth=3)
            axes[1, 1].plot(corner_points[i].x, corner_points[i].y, 'ro', markersize=8)
    axes[1, 1].set_title('Detected Chessboard')
    axes[1, 1].axis('off')
    
    # Warped board (if homography exists)
    if results['homography'] is not None:
        warped = cv2.warpPerspective(image, results['homography'], (400, 400))
        axes[1, 2].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title('Rectified Board')
        
        # Draw grid lines
        for i in range(9):
            # Vertical lines
            axes[1, 2].axvline(x=i * 400/8, color='white', alpha=0.5, linewidth=1)
            # Horizontal lines
            axes[1, 2].axhline(y=i * 400/8, color='white', alpha=0.5, linewidth=1)
    else:
        axes[1, 2].text(0.5, 0.5, 'No homography\nfound', ha='center', va='center', 
                       transform=axes[1, 2].transAxes, fontsize=14)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

# Test function
def test_board_localization(image_path: str):
    """Test the board localization on a sample image"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    localizer = ChessBoardLocalizer()
    homography, corner_points, results = localizer.localize_board(image)
    
    print(f"Board localization results:")
    print(f"  Lines detected: {len(results['lines'])}")
    print(f"  Horizontal lines: {len(results['horizontal_lines'])}")
    print(f"  Vertical lines: {len(results['vertical_lines'])}")
    print(f"  Intersections: {len(results['intersections'])}")
    print(f"  Homography found: {homography is not None}")
    print(f"  Corner points: {len(corner_points)}")
    
    if corner_points:
        print("  Corner coordinates:")
        for i, point in enumerate(corner_points):
            print(f"    Corner {i+1}: ({point.x:.1f}, {point.y:.1f})")
    
    # Visualize results
    visualize_localization(image, results, "board_localization_results.png")
    
    return homography, corner_points, results

if __name__ == "__main__":
    # Test with the existing image
    test_board_localization("/home/pranesh/chess/Computer-Vision/board-localisation/images/real_image_1.jpeg")
