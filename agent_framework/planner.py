import os
from typing import Dict, Any, List
from .llm import KimiLLM
import json
import re

class SemanticPlanner:
    """
    Phase 1: Analyzes the chart image and generates a high-level semantic structure 
    and annotation outline based on predefined guidelines.
    """
    def __init__(self, llm: KimiLLM):
        self.llm = llm
        
        # Load guidelines
        current_dir = os.path.dirname(os.path.abspath(__file__))
        guide_path = os.path.join(current_dir, "..", "guidelines", "semantic_parsing_guide.md")
        if os.path.exists(guide_path):
            with open(guide_path, "r", encoding="utf-8") as f:
                self.guidelines = f.read()
        else:
            self.guidelines = "No guidelines found."

    def parse_chart_structure(self, image_url_or_base64: str) -> Dict[str, Any]:
        """
        Uses VLM to parse the chart's structural components.

        Args:
            image_url_or_base64 (str): The image to be analyzed.
        
        Returns:
            Dict[str, Any]: A JSON dictionary describing the chart structure.
        """
        prompt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prompt", "agent_planner.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()
            
        prompt = prompt_template.replace("{guidelines}", self.guidelines)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url_or_base64
                        }
                    }
                ]
            }
        ]

        response_message = self.llm.chat(messages, temperature=0.1)
        response_text = response_message.get("content", "")
        
        try:
            # Try to find JSON block using regex if there's surrounding text
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
            else:
                return {"error": "Failed to extract JSON from response", "raw": response_text}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON response", "raw": response_text}
