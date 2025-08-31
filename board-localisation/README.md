# Chess Board Detection

This module identifies a chess board in an image and detects the intersection points of the grid squares.

## Process

The chess board detection follows these steps:

1. **Edge Detection**: Apply Canny edge detection to identify edges in the image.
2. **Line Detection**: Use Hough Transform to detect straight lines from the edges.
3. **Line Classification**: Separate the detected lines into horizontal (green) and vertical (blue) lines.
4. **Line Clustering**: Group similar lines together to reduce redundancy.
5. **Intersection Points**: Calculate the intersection points of the horizontal and vertical lines to identify the chess board grid.

## Usage

```bash
python board_detection.py --image /path/to/chess/image.jpg --output /path/to/output/folder --debug
```

### Arguments:

- `--image`: Path to the input chess board image (required)
- `--output`: Directory to save the output images (default: 'output')
- `--debug`: Flag to display intermediate results as matplotlib figures

## Example

```bash
python board_detection.py --image chess_image.jpg --output results --debug
```

## Output

The script saves four images to the output directory:

1. `1_original.jpg`: The original input image
2. `2_edges.jpg`: Result of Canny edge detection
3. `3_lines.jpg`: Image showing detected horizontal (green) and vertical (blue) lines
4. `4_intersections.jpg`: Final result showing the intersection points (red) of the chess board grid
