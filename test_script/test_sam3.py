import os
import cv2
import sys
import numpy as np

# Set environment variable to bypass OpenMP multiple runtime conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Adjust path so we can import from LabKnowMat-lite root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from agent_framework.tools.extractor import SAM3BoxExtractor, PieSliceExtractor

def test_bar():
    print("\nTesting SAM3BoxExtractor (Bar Chart)")
    print("=" * 50)
    
    sample_image_path = os.path.join(project_root, "..", "reference", "SAM3", "test", "other", "bar.jpg")
    if not os.path.exists(sample_image_path):
        print(f"Error: Could not find image at {sample_image_path}")
        return

    print(f"Loading image from {sample_image_path}...")
    image = cv2.imread(sample_image_path)
    
    if image is None:
        print("Error: Failed to read image using cv2.")
        return
        
    extractor = SAM3BoxExtractor()
    
    print("Running SAM3 extraction for bars...")
    # This might take some time as it downloads the model on the first run
    results = extractor.run(image, text_prompt="chart bar")
    
    print(f"\nExtracted {len(results)} bar bounding boxes.")
    
    # Visualization
    output_image = image.copy()
    for i, box in enumerate(results):
        x_min, y_min, x_max, y_max = [int(v) for v in box]
        # Draw box
        cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        # Draw id
        cv2.putText(output_image, str(i), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        print(f"Bar {i}: bbox {box}")
        
    out_path = os.path.join(current_dir, "sam3_bar_test_result.jpg")
    cv2.imwrite(out_path, output_image)
    print(f"\nSaved visualization to {out_path}")

def test_pie():
    print("\nTesting PieSliceExtractor (Pie Chart)")
    print("=" * 50)
    
    sample_image_path = os.path.join(project_root, "..", "reference", "SAM3", "test", "other", "pie.png")
    if not os.path.exists(sample_image_path):
        print(f"Error: Could not find image at {sample_image_path}")
        return

    print(f"Loading image from {sample_image_path}...")
    image = cv2.imread(sample_image_path)
    
    if image is None:
        print("Error: Failed to read image using cv2.")
        return
        
    extractor = PieSliceExtractor()
    
    print("Running SAM3 extraction for pie slices...")
    results = extractor.run(image, text_prompt="pie chart slice")
    
    print(f"\nExtracted {len(results)} pie slices.")
    
    # Visualization
    output_image = image.copy()
    for i, slice_data in enumerate(results):
        centroid = slice_data["centroid"]
        color_hex = slice_data["color"]
        polygon = slice_data["polygon"]
        
        print(f"Slice {i}: Centroid {centroid}, Color {color_hex}, Polygon Points: {len(polygon)}")
        
        # Draw polygon
        if polygon:
            pts = np.array(polygon, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(output_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            
        # Draw centroid and color text
        cx, cy = int(centroid[0]), int(centroid[1])
        cv2.circle(output_image, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(output_image, f"{i}: {color_hex}", (cx - 20, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
    out_path = os.path.join(current_dir, "sam3_pie_test_result.jpg")
    cv2.imwrite(out_path, output_image)
    print(f"\nSaved visualization to {out_path}")

if __name__ == "__main__":
    # Note: These tests require SAM3 environment to be set up as per SAM3_Setup.md
    try:
        test_bar()
        test_pie()
    except ImportError as e:
        print(f"\nImport Error: {e}")
        print("Please ensure SAM3 environment is set up. See SAM3_Setup.md for instructions.")