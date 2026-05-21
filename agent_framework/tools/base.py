import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseAtomicTool(ABC):
    """
    Base class for all atomic tools in LabKnowMat-lite.
    All tools should inherit from this and implement the run method.
    """
    name: str = "BaseTool"
    description: str = "Base description"

    @abstractmethod
    def run(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool's core function.

        Args:
            image (np.ndarray): The input chart image.
            **kwargs: Additional parameters required by the specific tool.

        Returns:
            Dict[str, Any]: A flat JSON-compatible dictionary containing the extraction results.
        """
        pass

    def get_parameters_schema(self) -> Dict[str, Any]:
        """
        Returns the JSON Schema for the tool's parameters (excluding the 'image' parameter, 
        which is injected automatically by the framework).
        Override this method in subclasses if the tool requires specific arguments.
        """
        return {
            "type": "object",
            "properties": {},
            "required": []
        }

    def get_tool_schema(self) -> Dict[str, Any]:
        """
        Returns the OpenAI-compatible function calling schema for this tool.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_parameters_schema()
            }
        }
