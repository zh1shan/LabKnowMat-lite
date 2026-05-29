import sys
import os
import cv2
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent_framework.tools.color_cluster_sampler import ColorClusterPointSampler

def run_test(image_path):
    print(f"Loading image from: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return
    
    sampler = ColorClusterPointSampler()
    
    print("Running ColorClusterPointSampler...")
    # num_colors = 3 for testing, but ideally could be a parameter. We'll use 3 as a default test.
    result = sampler.run(image=img, num_colors=2, sample_size=50, core_ratio=0.9)
    
    vis_img = img.copy()
    
    clusters = result.get('clusters', [])
    print(f"Found {len(clusters)} clusters:")
    for cluster in clusters:
        cluster_idx = cluster['cluster_idx']
        mean_color = cluster['mean_color'] # RGB
        sampled_points = cluster['sampled_points']
        print(f"  Cluster {cluster_idx}: Mean RGB {mean_color}, Sampled Points: {len(sampled_points)}")
        
        # BGR color for OpenCV drawing
        bgr_color = (int(mean_color[2]), int(mean_color[1]), int(mean_color[0]))
        
        for pt in sampled_points:
            cv2.circle(vis_img, (pt[0], pt[1]), 5, bgr_color, -1)
            cv2.circle(vis_img, (pt[0], pt[1]), 7, (0,0,0), 1) # black border
            
    print("Showing visualization. Press any key in the window to close it.")
    cv2.imshow("Color Cluster Sampler Result", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test ColorClusterPointSampler")
    parser.add_argument("image_path", type=str, help="Path to the test image")
    args = parser.parse_args()
    
    run_test(args.image_path)
