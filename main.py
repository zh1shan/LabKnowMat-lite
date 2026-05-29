import os
import sys
import argparse
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class StreamTee:
    """Tees a standard stream (stdout/stderr) into a file handle as well."""
    def __init__(self, stream, file_handle):
        self.stream = stream
        self.file_handle = file_handle

    def write(self, message):
        self.stream.write(message)
        self.file_handle.write(message)
        self.file_handle.flush()

    def flush(self):
        self.stream.flush()
        self.file_handle.flush()

def main():
    parser = argparse.ArgumentParser(description="LabKnowMat-lite Chart Processing CLI")
    parser.add_argument("-i", "--img", required=True, help="Path to the original chart image")
    parser.add_argument("-o", "--out", required=True, help="Output folder path")
    args = parser.parse_args()

    image_path = os.path.abspath(args.img)
    output_dir = os.path.abspath(args.out)

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Setup logger for both stdout and stderr
    log_file_path = os.path.join(output_dir, "log.txt")
    log_file = open(log_file_path, "w", encoding="utf-8")

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    sys.stdout = StreamTee(sys.stdout, log_file)
    sys.stderr = StreamTee(sys.stderr, log_file)

    try:
        # Set environment variable to bypass OpenMP multiple runtime conflict
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        from agent_framework.agent import LabKnowMatLiteAgent

        print(f"Starting processing for image: {image_path}")
        print(f"Output will be saved to: {output_dir}")

        # Encode image
        base64_image = encode_image(image_path)
        image_url = f"data:image/jpeg;base64,{base64_image}"

        # Initialize Agent
        agent = LabKnowMatLiteAgent()
        
        result = agent.process_chart(image_path, image_url)
        
        print("\n" + "="*50)
        print("AGENT FINAL ANNOTATION OUTPUT:")
        print("="*50)
        print(result["annotation_text"])
        print("="*50)

        # Save the annotation text to info.txt
        info_file_path = os.path.join(output_dir, "info.txt")
        with open(info_file_path, "w", encoding="utf-8") as f:
            f.write(result["annotation_text"])
        
        print(f"\nSuccessfully saved annotation text to {info_file_path}")
        
        # Phase 4: Execute Chart Reconstruction
        from agent_framework.generator import CodeGenerator
        generator = CodeGenerator()
        generator.generate_and_run_code(info_file_path, output_dir)
        
        print(f"Successfully saved execution log to {log_file_path}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Workflow execution failed: {e}")
        sys.exit(1)
    finally:
        # Reset stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()

if __name__ == "__main__":
    main()