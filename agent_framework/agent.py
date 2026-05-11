import numpy as np
from typing import Dict, Any, List
from .llm import KimiLLM
from .planner import SemanticPlanner
from .generator import CodeGenerator
from .tools.ocr import OCRTextLocator
from .tools.axis import AxisTickMapper
from .tools.extractor import ColorPointExtractor, SAM3BoxExtractor, GridColorSampler

class LabKnowMatLiteAgent:
    """
    The main orchestrator for the LabKnowMat-lite system.
    Implements Phase 2 (Iterative Annotation) using ReAct and manages the workflow.
    """
    def __init__(self, api_key: str = None, model: str = "moonshotai/kimi-k2.6"):
        self.llm = KimiLLM(api_key=api_key, model=model)
        self.planner = SemanticPlanner(self.llm)
        self.generator = CodeGenerator(self.llm)
        
        # Initialize tool library
        self.tools = {
            "ocr_text_locator": OCRTextLocator(),
            "axis_tick_mapper": AxisTickMapper(),
            "color_point_extractor": ColorPointExtractor(),
            "sam3_box_extractor": SAM3BoxExtractor(),
            "grid_color_sampler": GridColorSampler()
        }

    def process_chart(self, image_path: str, image_url_or_base64: str) -> Dict[str, Any]:
        """
        Main pipeline execution:
        1. Semantic Planning
        2. Iterative Tool Calling (ReAct)
        3. Code Generation
        """
        # Load image (mocked for now)
        # image_array = cv2.imread(image_path)
        image_array = np.zeros((100, 100, 3), dtype=np.uint8) 

        print("Phase 1: Parsing Semantic Structure...")
        structure = self.planner.parse_chart_structure(image_url_or_base64)
        print(f"Structure: {structure}")

        print("\nPhase 2: Iterative Annotation (Mocked)...")
        # TODO: Implement the full ReAct loop here.
        # The agent should look at the 'structure', realize it needs X/Y axes coordinates,
        # call OCR, then call AxisMapper, then call ColorExtractor for series, etc.
        
        # Mocking the synthesis step
        synthesized_annotation = self._synthesize_annotation(structure, {})

        print("\nPhase 3: Code Generation...")
        code = self.generator.generate_reconstruction_code(synthesized_annotation)

        return {
            "semantic_structure": structure,
            "annotation_text": synthesized_annotation,
            "reconstruction_code": code
        }

    def _synthesize_annotation(self, structure: Dict[str, Any], extracted_data: Dict[str, Any]) -> str:
        """
        Synthesizes the natural language annotation text (annotate.txt format) 
        from the planner's structure and the data extracted via tools.
        """
        # TODO: Implement proper synthesis logic based on paper_annotation formats
        # This should convert structured JSON back to the readable format in annotate.txt
        return "This is a mock synthesized annotation text based on extracted components."
