import os
import cv2
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from agent_framework.tools.axis import AxisLineLocator

def test_axis_locator(image_path=None):
    print("Testing AxisLineLocator")
    print("=" * 50)
    
    axis_tool = AxisLineLocator()
    
    # We will test on a chart that clearly has horizontal and vertical lines
    if image_path:
        sample_image_path = image_path
    else:
        sample_image_path = os.path.join(project_root, "..", "reference", "paper_annotation", "01_Role of centres", "charts", "chart01.jpg")
    
    if not os.path.exists(sample_image_path):
        print(f"Error: Could not find image at {sample_image_path}")
        return

    print(f"Loading image from {sample_image_path}...")
    image = cv2.imread(sample_image_path)
    
    if image is None:
        print("Error: Failed to read image using cv2.")
        return
        
    print("Running Axis extraction...")
    results = axis_tool.run(image)
    
    print(f"\nExtracted {len(results)} candidate lines.")
    print("\nResults:")
    
    output_image = image.copy()
    
    colors = [
        (0, 0, 255), (0, 150, 0), (255, 0, 0), (0, 200, 255), 
        (255, 0, 255), (255, 100, 0), (128, 0, 128), (0, 128, 128)
    ]
    
    for cand in results:
        cand_type = cand['type'].capitalize()
        cand_id = cand['id']
        p1 = cand['p1']
        p2 = cand['p2']
        print(f"{cand_type} Axis [Line {cand_id}]: {p1} to {p2}")
        
        # Visualization
        color = colors[cand_id % len(colors)]
        cv2.line(output_image, p1, p2, color, 4)
        
        label = f"[Line {cand_id}]"
        pos = (p1[0] + 10, p1[1] - 10)
        if cand["type"] == "vertical":
             pos = (p1[0] + 10, p1[1] + 20)
        cv2.putText(output_image, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
    # Save output to the same directory as the input image
    output_filename = "axis_test_result_" + os.path.basename(sample_image_path)
    output_path = os.path.join(os.path.dirname(os.path.abspath(sample_image_path)), output_filename)
    cv2.imwrite(output_path, output_image)
    print(f"\nSaved visualization to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_axis_locator(sys.argv[1])
    else:
        test_axis_locator()