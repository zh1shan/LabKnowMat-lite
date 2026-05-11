from typing import Dict, Any, List
from .llm import KimiLLM
import json

class SemanticPlanner:
    """
    Phase 1: Analyzes the chart image and generates a high-level semantic structure 
    and annotation outline.
    """
    def __init__(self, llm: KimiLLM):
        self.llm = llm

    def parse_chart_structure(self, image_url_or_base64: str) -> Dict[str, Any]:
        """
        Uses VLM to parse the chart's structural components.

        Args:
            image_url_or_base64 (str): The image to be analyzed.
        
        Returns:
            Dict[str, Any]: A JSON dictionary describing the chart structure.
        """
        prompt = (
            "You are a chart analysis expert. Please analyze the provided chart image and "
            "return its semantic structure in JSON format. Include the 'chart_type' and a list of 'components'. "
            "Components can be x_axis, y_axis_left, bar_series, line_series, etc., with their associated labels and colors."
        )

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

        response_text = self.llm.chat(messages, temperature=0.1)
        
        # TODO: Implement robust JSON extraction from response_text
        try:
            # Simple fallback for demonstration
            # In production, use regex to extract JSON blocks
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                return json.loads(response_text[start_idx:end_idx])
            else:
                return {"error": "Failed to extract JSON from response"}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON response"}
