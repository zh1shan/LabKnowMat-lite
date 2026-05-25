import os
import requests
import json
from typing import List, Dict, Any, Optional

try:
    from dotenv import load_dotenv
    # Load .env file from project root if available
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(current_dir, "..", ".env")
    load_dotenv(dotenv_path)
except ImportError:
    pass

class KimiLLM:
    """
    Wrapper for interacting with Moonshot AI's Kimi API via OpenRouter.
    Supports both text and image inputs, as well as Tool Calling.
    """
    def __init__(self, api_key: str = None, model: str = None, base_url: str = None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = model or os.environ.get("LLM_MODEL", "moonshotai/kimi-k2.6")
        self.base_url = base_url or os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")

        if not self.api_key:
            raise ValueError("API Key must be provided either via argument or environment variable OPENROUTER_API_KEY")

    def chat(self, messages: List[Dict[str, Any]], temperature: float = 0.7, tools: Optional[List[Dict[str, Any]]] = None) -> Any:
        """
        Send a chat completion request to the model.

        Args:
            messages (List[Dict[str, Any]]): List of message dictionaries.
            temperature (float): Sampling temperature.
            tools (List[Dict[str, Any]], optional): List of tool schemas.

        Returns:
            Any: The response message object from the model, which may contain text content or tool calls.
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
        
        if tools:
            data["tools"] = tools
            data["tool_choice"] = "auto"

        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            # Return the entire message object so we can check for tool_calls
            return result['choices'][0]['message']
        except Exception as e:
            print(f"LLM API Error: {e}")
            if 'response' in locals() and hasattr(response, 'text'):
                print(f"Response Content: {response.text}")
            return {"role": "assistant", "content": f"Error: {e}"}
