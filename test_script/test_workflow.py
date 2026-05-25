import os
import sys
import base64
import json

# Set environment variable to bypass OpenMP multiple runtime conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from agent_framework.agent import LabKnowMatLiteAgent

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_workflow(image_path_arg=None):
    if image_path_arg:
        sample_image_path = os.path.abspath(image_path_arg)
    else:
        sample_image_path = os.path.abspath(os.path.join(project_root, "..", "reference", "paper_annotation", "01_Role of centres", "charts", "chart01.jpg"))
        
    if not os.path.exists(sample_image_path):
        print(f"Error: Sample image not found at {sample_image_path}")
        return
        
    print(f"Testing Agentic Workflow on image: {sample_image_path}")
    
    # Encode image
    base64_image = encode_image(sample_image_path)
    image_url = f"data:image/jpeg;base64,{base64_image}"
    
    # Initialize Agent
    try:
        agent = LabKnowMatLiteAgent()
        
        result = agent.process_chart(sample_image_path, image_url)
        
        print("\n" + "="*50)
        print("AGENT FINAL ANNOTATION OUTPUT:")
        print("="*50)
        print(result["annotation_text"])
        print("="*50)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Workflow test failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_workflow(sys.argv[1])
    else:
        test_workflow()