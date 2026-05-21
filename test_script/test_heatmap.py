import os
import cv2
import sys
import numpy as np

# Adjust path so we can import from LabKnowMat-lite root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from agent_framework.tools.extractor import HeatmapDigitizerTool

def test_heatmap():
    print("\nTesting HeatmapDigitizerTool")
    print("=" * 50)
    
    sample_image_path = os.path.join(project_root, "..", "reference", "heatmap", "heatmap_digitizer", "heatmap_digitizer", "data", "example.png")
    if not os.path.exists(sample_image_path):
        print(f"Error: Could not find image at {sample_image_path}")
        return

    print(f"Loading image from {sample_image_path}...")
    image = cv2.imread(sample_image_path)
    
    if image is None:
        print("Error: Failed to read image using cv2.")
        return
        
    tool = HeatmapDigitizerTool()
    
    print("Running Heatmap digitization...")
    result = tool.run(image)
    
    if "error" in result:
        print(f"Error occurred during extraction: {result['error']}")
        return
        
    heatmap_rect = result["heatmap_rect"]
    legend_rect = result["legend_rect"]
    orientation = result["legend_orientation"]
    colormap = result["color_to_value_map"]
    matrix = result["normalized_matrix"]
    
    print(f"\nHeatmap Region: {heatmap_rect}")
    print(f"Legend Region: {legend_rect}")
    print(f"Legend Orientation: {orientation}")
    print("\nColor Map Sample:")
    for item in colormap[:5]:
        print(f"  Color {item['color']} -> Value {item['value']:.4f}")
    print("  ...")
    for item in colormap[-5:]:
        print(f"  Color {item['color']} -> Value {item['value']:.4f}")
        
    matrix_np = np.array(matrix)
    print(f"\nNormalized Matrix Shape: {matrix_np.shape}")
    print(f"Matrix Min: {matrix_np.min():.4f}, Max: {matrix_np.max():.4f}, Mean: {matrix_np.mean():.4f}")
    
    # Visualization: draw bounding boxes on original image
    vis_image = image.copy()
    
    hx1, hy1, hx2, hy2 = heatmap_rect
    cv2.rectangle(vis_image, (hx1, hy1), (hx2, hy2), (0, 0, 255), 3)
    cv2.putText(vis_image, "Heatmap", (hx1, hy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    lx1, ly1, lx2, ly2 = legend_rect
    cv2.rectangle(vis_image, (lx1, ly1), (lx2, ly2), (0, 255, 0), 3)
    cv2.putText(vis_image, "Legend", (lx1, ly1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    out_vis_path = os.path.join(current_dir, "heatmap_regions_test.jpg")
    cv2.imwrite(out_vis_path, vis_image)
    print(f"\nSaved region visualization to {out_vis_path}")
    
    # Reconstruct the heatmap visually from the 128x128 matrix to check correctness
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(matrix_np, cmap="viridis", cbar=True)
        plt.title("Reconstructed 128x128 Normalized Heatmap")
        
        out_reconstruct_path = os.path.join(current_dir, "heatmap_reconstructed_test.png")
        plt.savefig(out_reconstruct_path, bbox_inches="tight")
        print(f"Saved reconstructed heatmap visualization to {out_reconstruct_path}")
        plt.close()
    except ImportError:
        print("\nmatplotlib or seaborn not installed. Skipping reconstruction visualization.")

if __name__ == "__main__":
    test_heatmap()