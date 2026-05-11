import os
import requests
from typing import List, Dict, Any, Optional

class KimiLLM:
    """
    Wrapper for interacting with Moonshot AI's Kimi API.
    Supports both text and image inputs.
    """
    def __init__(self, api_key: str = None, model: str = "moonshot-v1-8k"):
        # The exact model to use is moonshotai/kimi-k2.6, accessed via OpenRouter based on key.txt
        # URL: https://openrouter.ai/api/v1/chat/completions
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

        if not self.api_key:
            raise ValueError("API Key must be provided either via argument or environment variable OPENROUTER_API_KEY")

    def chat(self, messages: List[Dict[str, Any]], temperature: float = 0.7) -> str:
        """
        Send a chat completion request to the model.

        Args:
            messages (List[Dict[str, Any]]): List of message dictionaries.
                Can include image URLs for multimodal tasks.
            temperature (float): Sampling temperature.

        Returns:
            str: The response text from the model.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            # TODO: Add robust error handling and retries
            print(f"LLM API Error: {e}")
            if 'response' in locals() and hasattr(response, 'text'):
                print(f"Response Content: {response.text}")
            return ""
