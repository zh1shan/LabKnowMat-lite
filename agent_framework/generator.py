import os
import re
import subprocess
from .llm import KimiLLM

class CodeGenerator:
    """
    Phase 3: Synthesizes the extracted annotations and generates the Python reconstruction script,
    then executes it to render the final chart.
    """
    def __init__(self, api_key: str = None, model: str = None):
        # By default use the Phase 3 model
        model = model or os.environ.get("LLM_MODEL_PHASE_3", "google/gemini-3.1-pro-preview")
        self.llm = KimiLLM(api_key=api_key, model=model)

    def generate_and_run_code(self, info_txt_path: str, output_dir: str) -> None:
        """
        Reads info.txt, asks the LLM to write a Python script, saves it, and executes it.
        """
        with open(info_txt_path, "r", encoding="utf-8") as f:
            annotate_text = f.read()

        # Dynamically inject extra instructions if external data files exist
        extra_instructions = ""
        heatmap_data_path = os.path.join(output_dir, "heatmap_data.json")
        if os.path.exists(heatmap_data_path):
            extra_instructions += (
                "5. IMPORTANT FOR HEATMAP: A file named 'heatmap_data.json' exists in the current directory. "
                "It contains the raw normalized matrix data formatted as a 2D JSON array (a list of lists of floats) with dimensions 128x128. "
                "You MUST load this file using `import json` and use it as the actual data source (Z matrix) for your heatmap. "
                "STRICTLY FORBIDDEN to generate random, synthetic, or Gaussian data using numpy. You must use the data from the json file.\n"
            )

        prompt = (
            "You are an expert Python data visualization developer. "
            "Based on the description in the provided text, write a Python script using matplotlib to render the chart as accurately as possible.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. You must think step-by-step (Chain of Thought) before writing the code. Think about how to map pixel coordinates to actual numerical values, how to use scipy.optimize.curve_fit for the trend line, and how to style the plot.\n"
            "2. Ensure colors, labels, and legends strictly match the description.\n"
            "3. You MUST include logic to save the figure to the current directory as 'chart.png'. For example: plt.savefig('chart.png', bbox_inches='tight'). Do not use plt.show().\n"
            "4. Only output the Python code inside ```python ... ``` block. Do not output anything else outside of the thinking block and code block.\n"
            f"{extra_instructions}\n"
            "Here is the chart description:\n"
            f"```text\n{annotate_text}\n```\n"
        )

        messages = [
            {"role": "user", "content": prompt}
        ]
        
        print(f"\n=== Phase 3: Chart Reconstruction ===")
        print(f"[Phase 3] LLM ({self.llm.model}) is thinking and generating code...")
        response_message = self.llm.chat(messages, temperature=0.2)
        response_text = response_message.get("content", "")
        
        # Extract python code
        match = re.search(r'```python\n(.*?)\n```', response_text, re.DOTALL)
        if match:
            code = match.group(1)
        else:
            print("[Phase 3 Error] Failed to extract Python code from response.")
            print(f"Raw response:\n{response_text}")
            return
            
        script_path = os.path.join(output_dir, "render_chart.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)
            
        print(f"[Phase 3] Generated script saved to {script_path}")
        print("[Phase 3] Executing the script...")
        
        try:
            result = subprocess.run(
                ["python", "render_chart.py"],
                cwd=output_dir,
                capture_output=True,
                text=True,
                check=True
            )
            print("[Phase 3] Script executed successfully. Chart saved as chart.png")
            if result.stdout:
                print(f"Stdout:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"[Phase 3 Error] Script execution failed with exit code {e.returncode}.")
            print(f"Stdout:\n{e.stdout}")
            print(f"Stderr:\n{e.stderr}")

