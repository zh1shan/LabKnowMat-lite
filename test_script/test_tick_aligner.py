import os
import cv2
import sys
import base64
import numpy as np

# Adjust path so we can import from LabKnowMat-lite root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from agent_framework.llm import KimiLLM
from agent_framework.planner import SemanticPlanner
from agent_framework.tools.ocr import OCRTextLocator
from agent_framework.tools.axis import AxisLineLocator
from agent_framework.tick_aligner import TickAligner

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_aligner(sample_image_path):
    print("Testing Tick Aligner (Full Pipeline)")
    print("=" * 50)

    # 1. Setup paths
    if not os.path.exists(sample_image_path):
        print(f"Error: Could not find image at {sample_image_path}")
        return

    # Load LLM
    key_path = os.path.join(project_root, "key.txt")
    api_key = None
    if os.path.exists(key_path):
        with open(key_path, "r") as f:
            api_key = f.readline().strip()
            
    llm = KimiLLM(api_key=api_key, model="moonshotai/kimi-k2.6")

    # 2. Semantic Planner
    print("\n[1/4] Running Semantic Planner...")
    planner = SemanticPlanner(llm)
    base64_image = encode_image(sample_image_path)
    image_url = f"data:image/jpeg;base64,{base64_image}"
    semantic_info = planner.parse_chart_structure(image_url)
    print("Semantic Info Extracted successfully.")

    # 3. OCR Extraction
    print("\n[2/4] Running OCR...")
    image = cv2.imread(sample_image_path)
    ocr_tool = OCRTextLocator()
    ocr_results = ocr_tool.run(image)
    print(f"Extracted {len(ocr_results)} text items.")

    # 4. Axis Candidate Extraction
    print("\n[3/4] Running Axis Line Locator...")
    axis_tool = AxisLineLocator()
    candidate_lines = axis_tool.run(image)
    print(f"Extracted {len(candidate_lines)} candidate lines.")

    # 5. Tick Aligner
    print("\n[4/4] Running Tick Aligner (LLM Clustering & Snapping)...")
    aligner = TickAligner(llm)
    aligned_results = aligner.cluster_and_snap(semantic_info, ocr_results, candidate_lines)

    # 6. Visualization
    output_image = image.copy()
    
    # Draw all candidate lines lightly
    for line in candidate_lines:
        color = (255, 200, 200) if line['type'] == 'horizontal' else (200, 255, 200)
        cv2.line(output_image, line['p1'], line['p2'], color, 1)

    # Draw points
    for orig, aligned in zip(ocr_results, aligned_results):
        orig_cx, orig_cy = int(orig['cx']), int(orig['cy'])
        new_cx, new_cy = int(aligned['cx']), int(aligned['cy'])
        
        # Draw original OCR point in RED
        cv2.circle(output_image, (orig_cx, orig_cy), 3, (0, 0, 255), -1)
        
        # Draw aligned point in GREEN
        cv2.circle(output_image, (new_cx, new_cy), 5, (0, 255, 0), -1)
        
        # Draw a line between them if snapped
        if orig_cx != new_cx or orig_cy != new_cy:
            cv2.line(output_image, (orig_cx, orig_cy), (new_cx, new_cy), (0, 255, 255), 2)
            print(f"Aligned text '{orig['text']}': ({orig_cx}, {orig_cy}) -> ({new_cx}, {new_cy})")

    base_name = os.path.splitext(os.path.basename(sample_image_path))[0]
    out_dir = os.path.dirname(os.path.abspath(sample_image_path))
    out_path = os.path.join(out_dir, f"{base_name}_aligned_ticks_result.jpg")
    cv2.imwrite(out_path, output_image)
    print(f"\nSaved visualization to {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_tick_aligner.py <image_path>")
        sys.exit(1)
    test_aligner(sys.argv[1])