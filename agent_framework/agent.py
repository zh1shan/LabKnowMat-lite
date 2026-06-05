import numpy as np
import cv2
import json
import os
import re
from typing import Dict, Any, List
from .llm import KimiLLM
from .generator import CodeGenerator
from .planner import SemanticPlanner
from .tools.ocr import OCRTextLocator
from .tools.axis import AxisLineLocator
from .tools.extractor import ScatterPointExtractorV1, SAM3BoxExtractor, PieSliceExtractor, HeatmapDigitizerTool
from .tools.color_cluster_sampler import ColorClusterPointSampler
from .tools.color_cluster_counter import ColorClusterPixelCounter

class LabKnowMatLiteAgent:
    """
    The main orchestrator for the LabKnowMat-lite system.
    Implements Phase 2 (Iterative Annotation) using ReAct workflow.
    """
    def __init__(self, api_key: str = None, model: str = None,
                 visualize: bool = False, output_dir: str = None):
        self.llm = KimiLLM(api_key=api_key, model=model)
        self.planner = SemanticPlanner(self.llm)
        
        # Pre-import torch to avoid DLL conflicts with PaddlePaddle (loaded by OCR)
        try:
            import torch
        except ImportError:
            pass
        
        self.visualize = visualize
        self.output_dir = output_dir
        self.visualizer = None
        if self.visualize and output_dir:
            from .visualizer import ToolCallVisualizer
            vis_dir = os.path.join(output_dir, "tool_call_history")
            self.visualizer = ToolCallVisualizer(vis_dir)
        
        # Initialize tool library
        self.tools = {
            "ocr_text_locator": OCRTextLocator(),
            "axis_line_locator": AxisLineLocator(),
            "scatter_point_extractor_v1": ScatterPointExtractorV1(),
            "sam3_box_extractor": SAM3BoxExtractor(),
            "pie_slice_extractor": PieSliceExtractor(),
            "heatmap_digitizer": HeatmapDigitizerTool(),
            "color_cluster_point_sampler": ColorClusterPointSampler(),
            "color_cluster_pixel_counter": ColorClusterPixelCounter()
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
        
        if "error" in structure:
            raise RuntimeError(f"Phase 1 failed to parse semantic structure: {structure['error']}. Raw response: {structure.get('raw', '')}")

        print("\n=== Phase 2: Iterative Annotation (ReAct Workflow) ===")
        # Prepare system prompt and tools schema
        tool_schemas = [tool.get_tool_schema() for tool in self.tools.values()]
        
        prompt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prompt", "agent_react_system.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            system_prompt_template = f.read()
            
        system_prompt = system_prompt_template.replace("{structure}", json.dumps(structure, ensure_ascii=False))
        
        user_prompt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prompt", "agent_react_user.txt")
        with open(user_prompt_path, "r", encoding="utf-8") as f:
            user_prompt = f.read()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_url_or_base64}}
                ]
            }
        ]
        
        max_iterations = 20
        final_annotation_text = ""
        extra_data = {}
        call_counter = 0
        fallback_triggered = False
        
        for iteration in range(max_iterations):
            print(f"\n[Iteration {iteration+1}] LLM is thinking...")
            response_msg = self.llm.chat(messages, temperature=0.1, tools=tool_schemas)
            
            # Append the assistant's message to conversation
            messages.append(response_msg)
            
            tool_calls = response_msg.get("tool_calls")
            
            if tool_calls:
                if iteration == max_iterations - 1:
                    print("\n[Fallback] Maximum iterations reached. VLM will now directly generate reconstruction code.")
                    fallback_prompt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prompt", "agent_fallback.txt")
                    with open(fallback_prompt_path, "r", encoding="utf-8") as f:
                        fallback_prompt_template = f.read()
                    fallback_prompt = fallback_prompt_template.replace("{structure}", json.dumps(structure, ensure_ascii=False))
                    
                    messages.append({
                        "role": "user",
                        "content": fallback_prompt
                    })
                    
                    print("[Fallback] VLM is generating Python code directly...")
                    fallback_response = self.llm.chat(messages, temperature=0.2)
                    fallback_content = fallback_response.get("content", "")
                    
                    match = re.search(r'```python\n(.*?)\n```', fallback_content, re.DOTALL)
                    if not match:
                        raise RuntimeError("[Fallback] VLM failed to produce Python code block in response.")
                    
                    code = match.group(1)
                    script_path = os.path.join(self.output_dir, "render_chart.py")
                    with open(script_path, "w", encoding="utf-8") as f:
                        f.write(code)
                    print(f"[Fallback] Generated script saved to {script_path}")
                    print("[Fallback] Executing the script...")
                    CodeGenerator.run_existing_code(self.output_dir)
                    
                    fallback_triggered = True
                    final_annotation_text = ""
                    print("[Fallback] Completed successfully.")
                    break
                
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
                            
                            if isinstance(result, dict) and "normalized_matrix" in result:
                                extra_data["heatmap_data.json"] = result.pop("normalized_matrix")
                                result["normalized_matrix_info"] = "Data is too large. It has been extracted and will be saved as 'heatmap_data.json'. Please instruct the reconstruction script to load this file."
                                
                            # Save tool call visualization
                            if self.visualize and self.visualizer:
                                self.visualizer.save(
                                    image=image_array,
                                    tool_name=function_name,
                                    args=arguments,
                                    result=result,
                                    call_index=call_counter,
                                    iteration=iteration
                                )
                            
                            # Convert result to string to pass back to LLM
                            result_str = json.dumps(result, ensure_ascii=False)[:15000]
                            if len(json.dumps(result)) > 15000:
                                result_str += "... (truncated)"
                        except Exception as e:
                            result_str = f"Error executing tool {function_name}: {str(e)}"
                            # Save error visualization
                            if self.visualize and self.visualizer:
                                self.visualizer.save(
                                    image=image_array,
                                    tool_name=function_name,
                                    args=arguments,
                                    result=result_str,
                                    call_index=call_counter,
                                    iteration=iteration,
                                    is_error=True
                                )
                    else:
                        result_str = f"Error: Tool {function_name} not found."
                        if self.visualize and self.visualizer:
                            self.visualizer.save(
                                image=image_array,
                                tool_name=function_name,
                                args=arguments,
                                result=result_str,
                                call_index=call_counter,
                                iteration=iteration,
                                is_error=True
                            )
                        
                    call_counter += 1
                    
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

        return {
            "semantic_structure": structure,
            "annotation_text": final_annotation_text,
            "extra_data": extra_data,
            "fallback_triggered": fallback_triggered
        }
