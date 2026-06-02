import os
import sys
import base64
import json
import requests
from agent_framework.llm import KimiLLM

def encode_image(image_path):
    """Encodes an image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_semantic_planner(image_path_arg=None):
    # Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    guide_path = os.path.join(current_dir, "guidelines", "semantic_parsing_guide.md")
    
    # Let's use a sample image from the reference dataset or provided argument
    if image_path_arg:
        sample_image_path = os.path.abspath(image_path_arg)
    else:
        sample_image_path = os.path.abspath(os.path.join(current_dir, "..", "reference", "paper_annotation", "01_Role of centres", "charts", "chart01.jpg"))
    
    if not os.path.exists(sample_image_path):
        print(f"Error: Sample image not found at {sample_image_path}")
        return
        
    if not os.path.exists(guide_path):
        print(f"Error: Guide file not found at {guide_path}")
        return

    # Read the guidelines
    with open(guide_path, "r", encoding="utf-8") as f:
        guidelines = f.read()

    # Encode image
    base64_image = encode_image(sample_image_path)
    image_url = f"data:image/jpeg;base64,{base64_image}"

    # Initialize LLM
    try:
        llm = KimiLLM()
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        return

    prompt_path = os.path.join(current_dir, "..", "prompt", "agent_test_semantic_planner.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()
        
    prompt = prompt_template.replace("{guidelines}", guidelines)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }
            ]
        }
    ]

    print(f"Testing Semantic Planner on image: {sample_image_path}")
    print("Waiting for VLM response...")
    
    response = llm.chat(messages, temperature=0.1)
    
    print("\n" + "="*50)
    print("VLM Response:")
    print("="*50)
    print(response)
    print("="*50)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_semantic_planner(sys.argv[1])
    else:
        test_semantic_planner()
