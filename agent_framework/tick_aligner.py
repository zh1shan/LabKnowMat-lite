import json
import re
import math
from typing import Dict, Any, List
from .llm import KimiLLM

class TickAligner:
    """
    Groups OCR text items into axis ticks using LLM reasoning,
    and then snaps their center coordinates to the nearest candidate axis line.
    """
    def __init__(self, llm: KimiLLM):
        self.llm = llm

    def cluster_and_snap(self, semantic_info: Dict[str, Any], ocr_results: List[Dict[str, Any]], candidate_lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main pipeline for aligning ticks.
        """
        if not ocr_results:
            return []

        # 1. Ask LLM to group OCR texts
        groups = self._cluster_ticks_with_llm(semantic_info, ocr_results)
        
        if not groups:
            print("Warning: LLM returned no valid tick groups.")
            return ocr_results
            
        # Create a copy of OCR results to modify
        aligned_results = [dict(item) for item in ocr_results]

        # 2. Snap coordinates to nearest candidate lines
        for group in groups:
            direction = group.get("axis_direction")
            indices = group.get("text_indices", [])
            
            # Filter valid indices
            valid_indices = [idx for idx in indices if 0 <= idx < len(aligned_results)]
            if not valid_indices:
                continue
                
            self._snap_group_to_axis(aligned_results, valid_indices, direction, candidate_lines)
            
        return aligned_results

    def _cluster_ticks_with_llm(self, semantic_info: Dict[str, Any], ocr_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Prepare OCR texts for LLM
        ocr_text_list = ""
        for i, item in enumerate(ocr_results):
            ocr_text_list += f"[{i}] {item['text']} (center: {int(item['cx'])}, {int(item['cy'])})\n"
            
        semantic_str = json.dumps(semantic_info, indent=2, ensure_ascii=False)
        
        prompt = (
            "You are an expert in chart data extraction. I will provide you with the semantic structure of a chart, "
            "and a list of all texts recognized by OCR in this chart (with their ID and center coordinates).\n\n"
            "Your task is to identify which texts are tick labels on the axes. Group these texts by their respective axis.\n"
            "For example, group all texts belonging to the horizontal X-axis together, and group texts belonging to the vertical Y-axis together.\n\n"
            "### Semantic Structure ###\n"
            f"{semantic_str}\n\n"
            "### OCR Texts ###\n"
            f"{ocr_text_list}\n\n"
            "CRITICAL: Output your answer ONLY as a JSON list of objects, where each object represents an axis group.\n"
            "Do not include any explanation or markdown formatting outside the JSON.\n"
            "Format:\n"
            "[\n"
            "  {\n"
            "    \"axis_direction\": \"horizontal\",\n"
            "    \"text_indices\": [1, 2, 3, 4]\n"
            "  },\n"
            "  {\n"
            "    \"axis_direction\": \"vertical\",\n"
            "    \"text_indices\": [5, 6, 7]\n"
            "  }\n"
            "]\n"
        )
        
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        response_text = self.llm.chat(messages, temperature=0.1)
        
        try:
            match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if match:
                groups = json.loads(match.group(0))
                print("\n[TickAligner] LLM Clustered Groups:")
                print(json.dumps(groups, indent=2))
                return groups
            else:
                return []
        except json.JSONDecodeError:
            print("Failed to parse LLM clustering output:", response_text)
            return []

    def _snap_group_to_axis(self, ocr_results: List[Dict[str, Any]], indices: List[int], direction: str, candidate_lines: List[Dict[str, Any]]):
        """
        Calculates the bounding box of the grouped texts and finds the nearest candidate line.
        """
        # Collect all points from the boxes in this group to calculate the exact bounding rectangle
        min_x = float('inf')
        max_x = float('-inf')
        min_y = float('inf')
        max_y = float('-inf')
        
        for idx in indices:
            box = ocr_results[idx]['box']
            for point in box:
                min_x = min(min_x, point[0])
                max_x = max(max_x, point[0])
                min_y = min(min_y, point[1])
                max_y = max(max_y, point[1])
                
        rect_width = max_x - min_x
        rect_height = max_y - min_y
        
        if direction == "horizontal":
            # Search area vertically (Y-axis)
            # Find the nearest horizontal line within [min_y - rect_height, max_y + rect_height]
            search_min_y = min_y - rect_height
            search_max_y = max_y + rect_height
            
            best_line = None
            min_dist = float('inf')
            
            for line in candidate_lines:
                if line['type'] != "horizontal":
                    continue
                # The y_center of the horizontal line
                line_y = line['p1'][1]
                # The x range of the horizontal line should roughly overlap with the group
                # To be safe, we just check y distance
                if search_min_y <= line_y <= search_max_y:
                    # distance from the line to the center of the text bounding box
                    dist = abs(line_y - ((min_y + max_y) / 2))
                    if dist < min_dist:
                        min_dist = dist
                        best_line = line
                        
            if best_line:
                target_y = best_line['p1'][1]
                for idx in indices:
                    ocr_results[idx]['cy'] = target_y
                    
        elif direction == "vertical":
            # Search area horizontally (X-axis)
            # Find the nearest vertical line within [min_x - rect_width, max_x + rect_width]
            search_min_x = min_x - rect_width
            search_max_x = max_x + rect_width
            
            best_line = None
            min_dist = float('inf')
            
            for line in candidate_lines:
                if line['type'] != "vertical":
                    continue
                line_x = line['p1'][0]
                if search_min_x <= line_x <= search_max_x:
                    dist = abs(line_x - ((min_x + max_x) / 2))
                    if dist < min_dist:
                        min_dist = dist
                        best_line = line
                        
            if best_line:
                target_x = best_line['p1'][0]
                for idx in indices:
                    ocr_results[idx]['cx'] = target_x
