import os
import cv2
import sys
import numpy as np

# Adjust path so we can import from LabKnowMat-lite root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# If SAM3 is located at reference/SAM3/sam3 but not installed via pip, we can append it:
sam3_path = os.path.abspath(os.path.join(project_root, "..", "reference", "SAM3"))
if sam3_path not in sys.path:
    sys.path.append(sam3_path)

from agent_framework.tools.extractor import SAM3BoxExtractor, PieSliceExtractor

def test_sam3_bar():
    print("\n--- Testing SAM3BoxExtractor (Bar Chart) ---")
    
    # Locate a bar chart from reference
    sample_image_path = os.path.join(project_root, "..", "reference", "paper_annotation", "10_Unsupervised fault detection", "charts", "chart09.jpg")
    if not os.path.exists(sample_image_path):
        print(f"Error: Could not find bar chart image at {sample_image_path}")
        return

    print(f"Loading bar chart image from {sample_image_path}...")
    image = cv2.imread(sample_image_path)
    
    if image is None:
        print("Error: Failed to read image.")
        return

    try:
        bar_tool = SAM3BoxExtractor()
        print("Running SAM3 extraction for bars...")
        boxes = bar_tool.run(image, text_prompt="chart bar")
        
        print(f"Extracted {len(boxes)} bounding boxes.")
        
        # Visualization
        output_image = image.copy()
        for idx, box in enumerate(boxes):
            x_min, y_min, x_max, y_max = map(int, box)
            cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(output_image, str(idx), (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
        out_path = os.path.join(current_dir, "sam3_bar_test_result.jpg")
        cv2.imwrite(out_path, output_image)
        print(f"Saved visualization to {out_path}")
        
    except ImportError as e:
        print(f"Skipping execution due to missing dependency: {e}")
    except Exception as e:
        print(f"Error during SAM3 Bar execution: {e}")


def test_sam3_pie():
    print("\n--- Testing PieSliceExtractor (Pie Chart) ---")
    
    # Locate a pie chart from reference
    # ChartBench/pie/sector_chart_9_image.png is used in SAM3 test examples, let's look for it
    # Or just use an arbitrary one
    sample_image_path = os.path.join(project_root, "..", "reference", "SAM3", "test", "ChartBench", "pie", "sector_chart_9_image.png")
    
    if not os.path.exists(sample_image_path):
        print(f"Error: Could not find pie chart image at {sample_image_path}")
        # Fallback to another pie chart if you have one
        return

    print(f"Loading pie chart image from {sample_image_path}...")
    image = cv2.imread(sample_image_path)
    
    if image is None:
        print("Error: Failed to read image.")
        return

    try:
        pie_tool = PieSliceExtractor()
        print("Running SAM3 extraction for pie slices...")
        slices = pie_tool.run(image)
        
        print(f"Extracted {len(slices)} pie slices.")
        
        # Visualization
        output_image = image.copy()
        for idx, s in enumerate(slices):
            cx, cy = map(int, s['centroid'])
            color = s['color']
            polygon = s['polygon']
            
            # Draw centroid
            cv2.circle(output_image, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(output_image, f"{idx}:{color}", (cx+5, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Draw polygon
            if polygon:
                pts = np.array(polygon, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(output_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                
        out_path = os.path.join(current_dir, "sam3_pie_test_result.jpg")
        cv2.imwrite(out_path, output_image)
        print(f"Saved visualization to {out_path}")
        
    except ImportError as e:
        print(f"Skipping execution due to missing dependency: {e}")
    except Exception as e:
        print(f"Error during SAM3 Pie execution: {e}")

if __name__ == "__main__":
    test_sam3_bar()
    test_sam3_pie()