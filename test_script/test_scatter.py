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

from agent_framework.tools.extractor import ScatterPointExtractorV1

def test_scatter():
    print("\nTesting ScatterPointExtractorV1")
    print("=" * 50)
    
    sample_image_path = os.path.join(project_root, "..", "reference", "paper_annotation", "01_Role of centres", "charts", "chart01.jpg")
    if not os.path.exists(sample_image_path):
        print(f"Error: Could not find image at {sample_image_path}")
        return

    print(f"Loading image from {sample_image_path}...")
    image = cv2.imread(sample_image_path)
    
    if image is None:
        print("Error: Failed to read image using cv2.")
        return
        
    extractor = ScatterPointExtractorV1()
    
    print("Running Scatter extraction over full image...")
    # Run full image extraction
    result_full = extractor.run(image)
    
    print(f"\n[Full Image] Extracted {result_full['point_count']} points in {result_full['cluster_count']} clusters.")
    for cluster in result_full['clusters']:
        print(f"  Cluster {cluster['id']}: {cluster['count']} points, Color {cluster['mean_hex']}")
        
    # Test restricted region
    # Let's say we only want to search in the left half of the image
    h, w = image.shape[:2]
    target_rect = [0, 0, w // 2, h]
    
    print(f"\nRunning Scatter extraction in restricted region: {target_rect}...")
    result_rect = extractor.run(image, target_rect=target_rect)
    
    print(f"\n[Restricted Region] Extracted {result_rect['point_count']} points in {result_rect['cluster_count']} clusters.")
    for cluster in result_rect['clusters']:
        print(f"  Cluster {cluster['id']}: {cluster['count']} points, Color {cluster['mean_hex']}")
        
    # Visualization for full image result
    vis_image = image.copy()
    for point in result_full["points"]:
        x, y = point["x"], point["y"]
        cluster_id = point["cluster_id"]
        rgb = point["rgb"]
        color_bgr = tuple(int(c) for c in rgb[::-1])
        
        cv2.circle(vis_image, (x, y), 5, color_bgr, 2)
        cv2.putText(vis_image, str(cluster_id), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(vis_image, str(cluster_id), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    out_vis_path = os.path.join(current_dir, "scatter_test_result.jpg")
    cv2.imwrite(out_vis_path, vis_image)
    print(f"\nSaved visualization to {out_vis_path}")

if __name__ == "__main__":
    test_scatter()