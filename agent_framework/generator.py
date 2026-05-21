from .llm import KimiLLM
from typing import Dict, Any

class CodeGenerator:
    """
    Phase 3: Synthesizes the extracted annotations and generates the Python reconstruction script.
    """
    def __init__(self, llm: KimiLLM):
        self.llm = llm

    def generate_reconstruction_code(self, annotate_text: str) -> str:
        """
        Takes the synthesized annotate.txt content and generates a render_chart.py script.

        Args:
            annotate_text (str): Natural language + coordinate descriptions of the chart.
        
        Returns:
            str: The generated Python code for rendering the chart.
        """
        prompt = (
            "You are an expert Python data visualization developer. "
            "I will provide you with a structured description of a chart, including its components, "
            "colors, and the raw pixel coordinates of its data points/bars.\n\n"
            "Your task is to:\n"
            "1. Map the pixel coordinates to actual numerical values based on the described axes.\n"
            "2. Write a Python script using Matplotlib or Seaborn that rebuilds this chart as accurately as possible.\n"
            "3. Ensure colors, labels, and legends match the description.\n\n"
            "Here is the chart description:\n"
            f"```text\n{annotate_text}\n```\n\n"
            "Please output only the Python code enclosed in ```python ... ``` blocks."
        )

        messages = [
            {"role": "user", "content": prompt}
        ]

        response_message = self.llm.chat(messages, temperature=0.2)
        response_text = response_message.get("content", "")
        
        # TODO: Implement robust code extraction from response_text
        return response_text
