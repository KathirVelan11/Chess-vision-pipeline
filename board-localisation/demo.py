import os
import argparse
from board_detection import detect_board

def main():
    parser = argparse.ArgumentParser(description='Chess Board Detection Demo')
    parser.add_argument('--image_dir', default='images', help='Directory containing chess board images')
    parser.add_argument('--output_dir', default='output', help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all image files in the directory
    image_files = [f for f in os.listdir(args.image_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No image files found in {args.image_dir}")
        return
    
    # Process each image
    for image_file in image_files:
        image_path = os.path.join(args.image_dir, image_file)
        print(f"Processing {image_file}...")
        
        # Create a subdirectory for this image's results
        image_name = os.path.splitext(image_file)[0]
        image_output_dir = os.path.join(args.output_dir, image_name)
        
        try:
            # Detect chess board
            intersection_points = detect_board(image_path, image_output_dir, debug=False)
            print(f"Found {len(intersection_points)} intersection points")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

if __name__ == "__main__":
    main()
