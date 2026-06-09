import os
import sys
import argparse
import subprocess
import time

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
RETRY_BASE_DELAY = 10


def main():
    parser = argparse.ArgumentParser(description="Batch chart processing for LabKnowMat-lite")
    parser.add_argument("-i", "--input", required=True, help="Path to the folder containing chart images")
    parser.add_argument("-v", "--visualize", action="store_true",
                        help="Enable tool call visualization for each chart")
    parser.add_argument("-r", "--retry", type=int, default=3,
                        help="Number of retries for failed images (default: 3)")
    args = parser.parse_args()

    input_folder = os.path.abspath(args.input)

    if not os.path.isdir(input_folder):
        print(f"Error: '{input_folder}' is not a valid directory.")
        sys.exit(1)

    image_files = []
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS:
                image_files.append(os.path.join(root, filename))

    image_files.sort()
    total = len(image_files)

    if total == 0:
        print(f"No images found in '{input_folder}'.")
        sys.exit(0)

    print(f"Found {total} image(s) in '{input_folder}'")
    print("=" * 60)

    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
    python_exe = sys.executable

    success = 0
    failed = []

    for idx, image_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{total}] Processing: {image_path}")

        output_dir = image_path + "_out"

        cmd = [python_exe, script_path, "-i", image_path, "-o", output_dir]
        if args.visualize:
            cmd.append("-v")

        succeeded = False
        for attempt in range(1 + args.retry):
            if attempt > 0:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(f"  [Retry {attempt}/{args.retry}] Waiting {delay}s before retry...")
                time.sleep(delay)

            result = subprocess.run(
                cmd, capture_output=True, text=True,
                encoding="utf-8", errors="replace"
            )

            if result.returncode == 0:
                print(f"  [OK] Output saved to: {output_dir}")
                success += 1
                succeeded = True
                break
            else:
                print(f"  [FAILED] Return code: {result.returncode}")
                if result.stderr:
                    print(f"  Error: {result.stderr.strip()[-500:]}")

        if not succeeded:
            failed.append(image_path)

    print("\n" + "=" * 60)
    print(f"Batch processing complete: {success}/{total} succeeded")

    if failed:
        print(f"Failed ({len(failed)}):")
        for f in failed:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
