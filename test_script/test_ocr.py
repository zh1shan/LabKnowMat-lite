import os
import cv2
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from agent_framework.tools.ocr import OCRTextLocator

def run_tests():
    print("Testing OCRTextLocator with PaddleOCR")
    print("=" * 50)
    
    ocr_tool = OCRTextLocator()
    
    test_cases = [
        {
            "name": "Standard text and numbers",
            "path": os.path.join(project_root, "..", "reference", "paper_annotation", "01_Role of centres", "charts", "chart01.jpg"),
            "output": "ocr_test_standard.jpg"
        },
        {
            "name": "Rotated text (90 degrees)",
            "path": os.path.join(project_root, "..", "reference", "paper_annotation", "09_Exploring factors influencing", "charts", "chart03.jpg"),
            "output": "ocr_test_rotated.jpg"
        }
    ]
    
    for case in test_cases:
        print(f"\nRunning test: {case['name']}")
        print(f"Image path: {case['path']}")
        
        if not os.path.exists(case['path']):
            print(f"Error: Image not found at {case['path']}")
            continue
            
        image = cv2.imread(case['path'])
        if image is None:
            print("Error: Failed to read image")
            continue
            
        results = ocr_tool.run(image)
        print(f"Extracted {len(results)} text items")
        
        output_image = image.copy()
        for item in results:
            text = item['text']
            cx, cy = int(item['cx']), int(item['cy'])
            box = np.array(item['box']).astype(np.int32)
            
            # Print horizontal representation
            print(f"  - '{text}' at ({cx}, {cy})")
            
            # Draw box and center point
            cv2.polylines(output_image, [box], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.circle(output_image, (cx, cy), 3, (0, 0, 255), -1)
            
        out_path = os.path.join(current_dir, case['output'])
        cv2.imwrite(out_path, output_image)
        print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        if not os.path.exists(img_path):
            print(f"Error: Image not found at {img_path}")
            sys.exit(1)
        ocr_tool = OCRTextLocator()
        image = cv2.imread(img_path)
        if image is None:
            print("Error: Failed to read image")
            sys.exit(1)
        results = ocr_tool.run(image)
        output_image = image.copy()
        for item in results:
            cx, cy = int(item['cx']), int(item['cy'])
            box = np.array(item['box']).astype(np.int32)
            cv2.polylines(output_image, [box], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.circle(output_image, (cx, cy), 3, (0, 0, 255), -1)
            print(f"  - '{item['text']}' at ({cx}, {cy})")
        base, ext = os.path.splitext(img_path)
        out_path = f"{base}_ocr_result{ext}"
        cv2.imwrite(out_path, output_image)
        print(f"Saved visualization to {out_path}")
    else:
        run_tests()