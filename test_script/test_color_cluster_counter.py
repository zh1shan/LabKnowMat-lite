import sys
import os
import cv2
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent_framework.tools.color_cluster_counter import ColorClusterPixelCounter

def run_test(image_path):
    print(f"Loading image from: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return
    
    counter = ColorClusterPixelCounter()
    
    print("Running ColorClusterPixelCounter...")
    # num_colors = 3 for testing, but ideally could be a parameter. We'll use 3 as a default test.
    result = counter.run(image=img, num_colors=3, core_ratio=0.9)
    
    vis_img = img.copy()
    
    clusters = result.get('clusters', [])
    print(f"Found {len(clusters)} clusters:")
    
    y_offset = 30
    for cluster in clusters:
        cluster_idx = cluster['cluster_idx']
        mean_color = cluster['mean_color'] # RGB
        pixel_count = cluster['pixel_count']
        print(f"  Cluster {cluster_idx}: Mean RGB {mean_color}, Core Pixel Count: {pixel_count}")
        
        bgr_color = (int(mean_color[2]), int(mean_color[1]), int(mean_color[0]))
        
        # Draw a color swatch
        cv2.rectangle(vis_img, (10, y_offset-15), (30, y_offset+5), bgr_color, -1)
        cv2.rectangle(vis_img, (10, y_offset-15), (30, y_offset+5), (0,0,0), 1)
        
        text = f"Count: {pixel_count}"
        cv2.putText(vis_img, text, (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        
        y_offset += 40
        
    print("Showing visualization. Press any key in the window to close it.")
    cv2.imshow("Color Cluster Counter Result", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test ColorClusterPixelCounter")
    parser.add_argument("image_path", type=str, help="Path to the test image")
    args = parser.parse_args()
    
    run_test(args.image_path)
