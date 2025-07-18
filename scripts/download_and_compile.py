
import os
import subprocess
import sys

# Dependency check: huggingface_hub
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("ERROR: The 'huggingface_hub' package is not installed. Please run 'pip install huggingface_hub'.")
    sys.exit(1)

# Dependency check: coremlc
def check_coremlc():
    from shutil import which
    if which("coremlc") is None:
        print("ERROR: 'coremlc' is not installed or not in your PATH. Please install Xcode command line tools and ensure 'coremlc' is available.")
        sys.exit(1)


# Settings
repo_id = "mlx-community/OpenELM-450M-Instruct"
local_dir = "../models"
mlmodel_filename = "OpenELM-450M-Instruct-128-float32.mlmodel"
mlmodelc_dir = "../models/OpenELM-450M-Instruct-128-float32.mlmodelc"

def main():
    # Check for coremlc
    check_coremlc()

    # 1. Download model and tokenizer
    print("Downloading model and tokenizer from Hugging Face...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        allow_patterns=[mlmodel_filename, "tokenizer.json"],
        local_dir_use_symlinks=False
    )

    # 2. Compile the model using coremlc
    mlmodel_path = os.path.join(local_dir, mlmodel_filename)
    print(f"Compiling {mlmodel_path} to {mlmodelc_dir} using coremlc...")
    subprocess.run([
        "coremlc", "compile", mlmodel_path, mlmodelc_dir
    ], check=True)

    print("Done! Model and tokenizer are ready.")

if __name__ == "__main__":
    main()
