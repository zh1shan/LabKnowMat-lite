import numpy as np
import cv2
import json
from typing import Dict, Any, List
from .llm import KimiLLM
from .planner import SemanticPlanner
from .generator import CodeGenerator
from .tools.ocr import OCRTextLocator
from .tools.axis import AxisLineLocator
from .tools.extractor import ScatterPointExtractorV1, SAM3BoxExtractor, PieSliceExtractor, HeatmapDigitizerTool

class LabKnowMatLiteAgent:
    """
    The main orchestrator for the LabKnowMat-lite system.
    Implements Phase 2 (Iterative Annotation) using ReAct workflow.
    """
    def __init__(self, api_key: str = None, model: str = "moonshotai/kimi-k2.6"):
        self.llm = KimiLLM(api_key=api_key, model=model)
        self.planner = SemanticPlanner(self.llm)
        self.generator = CodeGenerator(self.llm)
        
        # Initialize tool library
        self.tools = {
            "ocr_text_locator": OCRTextLocator(),
            "axis_line_locator": AxisLineLocator(),
            "scatter_point_extractor_v1": ScatterPointExtractorV1(),
            "sam3_box_extractor": SAM3BoxExtractor(),
            "pie_slice_extractor": PieSliceExtractor(),
            "heatmap_digitizer": HeatmapDigitizerTool()
        }

    def process_chart(self, image_path: str, image_url_or_base64: str) -> Dict[str, Any]:
        """
        Main pipeline execution:
        1. Semantic Planning
        2. Iterative Tool Calling (ReAct)
        3. Code Generation
        """
        print(f"Loading image {image_path}...")
        image_array = cv2.imread(image_path)
        if image_array is None:
            raise ValueError(f"Failed to read image at {image_path}")

        print("\n=== Phase 1: Parsing Semantic Structure ===")
        structure = self.planner.parse_chart_structure(image_url_or_base64)
        print(f"Structure:\n{json.dumps(structure, indent=2, ensure_ascii=False)}")

        print("\n=== Phase 2: Iterative Annotation (ReAct Workflow) ===")
        # Prepare system prompt and tools schema
        tool_schemas = [tool.get_tool_schema() for tool in self.tools.values()]
        
        system_prompt = (
            "You are an autonomous chart analysis agent operating in a ReAct loop. "
            "Your goal is to extract the exact pixel coordinates and colors of all visual elements in a chart.\n"
            "You have access to several specialized atomic tools. You must invoke them to gather information.\n\n"
            "### Current Chart Semantic Structure ###\n"
            f"{json.dumps(structure, ensure_ascii=False)}\n\n"
            "### Instructions ###\n"
            "1. Analyze the semantic structure above.\n"
            "2. Decide which tools to call. You can make multiple tool calls in parallel if needed.\n"
            "3. After gathering all necessary coordinates and colors for every component, "
            "write a comprehensive final report describing the chart's components, axes mapping, "
            "and the pixel coordinates of the data points/bars/slices. This report should look like the 'annotate.txt' format.\n"
            "4. Do NOT guess coordinates. You must use tools to find them.\n"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": "Please begin extracting coordinates using your tools."},
                    {"type": "image_url", "image_url": {"url": image_url_or_base64}}
                ]
            }
        ]
        
        max_iterations = 10
        final_annotation_text = ""
        
        for iteration in range(max_iterations):
            print(f"\n[Iteration {iteration+1}] LLM is thinking...")
            response_msg = self.llm.chat(messages, temperature=0.1, tools=tool_schemas)
            
            # Append the assistant's message to conversation
            messages.append(response_msg)
            
            tool_calls = response_msg.get("tool_calls")
            
            if tool_calls:
                print(f"LLM decided to call {len(tool_calls)} tool(s).")
                for tool_call in tool_calls:
                    function_name = tool_call["function"]["name"]
                    call_id = tool_call["id"]
                    try:
                        arguments = json.loads(tool_call["function"]["arguments"])
                    except json.JSONDecodeError:
                        arguments = {}
                        
                    print(f"  -> Invoking tool: {function_name} with args {arguments}")
                    
                    if function_name in self.tools:
                        tool_instance = self.tools[function_name]
                        try:
                            # Execute local Python tool, injecting the image array
                            result = tool_instance.run(image=image_array, **arguments)
                            # Convert result to string to pass back to LLM
                            result_str = json.dumps(result, ensure_ascii=False)[:4000] # truncate if too long
                            if len(json.dumps(result)) > 4000:
                                result_str += "... (truncated)"
                        except Exception as e:
                            result_str = f"Error executing tool {function_name}: {str(e)}"
                    else:
                        result_str = f"Error: Tool {function_name} not found."
                        
                    # Append tool result to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": function_name,
                        "content": result_str
                    })
            else:
                # No tool calls, means the agent has finished its task
                final_annotation_text = response_msg.get("content", "")
                print("LLM finished tool calling. Final Output received.")
                break

        print("\n=== Phase 3: Code Generation ===")
        # code = self.generator.generate_reconstruction_code(final_annotation_text)
        code = "# Code generation temporarily bypassed for tool testing."

        return {
            "semantic_structure": structure,
            "annotation_text": final_annotation_text,
            "reconstruction_code": code
        }
