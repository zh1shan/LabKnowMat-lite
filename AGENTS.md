# OpenCode Agent Instructions: LabKnowMat-lite

This file contains high-signal, repo-specific facts and operational constraints for AI agents working in `LabKnowMat-lite`.

## Architecture & Boundaries
- `LabKnowMat-lite` is an Agent-based chart reconstruction system that replaces hardcoded rules with a VLM-driven workflow.
- **Entry Point:** `agent_framework/agent.py` orchestrates the pipeline:
  1. Semantic Planner (VLM)
  2. Atomic Tool Library (`agent_framework/tools/*`)
  3. Code Generator (LLM)
- **Tool Philosophy:** "代码内置，环境外挂" (Code built-in, environment external). Tool implementations must be decoupled, atomic, and stateless, returning simple JSON-compatible outputs without internal business logic assumptions.
- **Guidelines:** Refer to `guidelines/semantic_parsing_guide.md` for expected VLM parsing behavior and structural outlines.

## Environment & Setup
- **Conda Environment:** Use the `labknowmat-lite` conda environment for execution.
  ```bash
  conda activate labknowmat-lite
  ```
- **SAM3 Quirk:** The SAM3 model is not configured via standard requirements. It requires PyTorch and a manual install from a specific SAM3 GitHub commit. Do not assume `sam3` is available natively without executing the steps in `SAM3_Setup.md`.
- **API Keys:** LLM integrations (in `agent_framework/llm.py`) default to `moonshotai/kimi-k2.6` via OpenRouter. You must export `OPENROUTER_API_KEY` or place the key in a local `key.txt` file (some test scripts rely on `key.txt`).

## Testing & Execution
- Tests are standard Python scripts located in `test_script/`. 
- **Command:** Run tests individually using Python.
  ```bash
  python test_script/test_sam3.py
  ```
- **Execution Gotcha:** When running scripts that use CV2 or SAM3 models, you may encounter an OpenMP duplicate runtime crash. This is typically bypassed by setting `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` before importing heavy libraries.
