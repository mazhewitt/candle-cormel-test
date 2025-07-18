# Candle-CoreML Model & Tokenizer Setup

This project provides scripts and code to download, prepare, and use the OpenELM-450M-Instruct model and tokenizer for CoreML-based text generation with Candle.

## Overview

- **`scripts/download_and_compile.py`**: Python script to automatically download the model and tokenizer from Hugging Face and compile the model to CoreML format using `coremlc`.
- **`models/`**: Directory where the model and tokenizer files are stored.
- **Rust code**: Integration tests and main logic for running text generation using the downloaded and compiled model.

---

## Prerequisites

- **Python 3.8+**
- **pip** (Python package manager)
- **Xcode Command Line Tools** (for `coremlc`)
- **Rust toolchain** (for running the main project)

---

## Setup Instructions

### 1. Install Python Dependencies

```
pip install huggingface_hub
```

### 2. Ensure `coremlc` is Installed

- `coremlc` is included with Xcode Command Line Tools. To install:
  - Run `xcode-select --install` if not already installed.
  - Ensure `coremlc` is in your PATH (try running `coremlc --help` in your terminal).

### 3. Download and Compile the Model

From the project root, run:

```
cd scripts
python download_and_compile.py
```

This will:
- Download the `OpenELM-450M-Instruct-128-float32.mlmodel` and `tokenizer.json` from Hugging Face into the `models/` directory.
- Compile the `.mlmodel` into a CoreML `.mlmodelc` directory using `coremlc`.

### 4. Run Rust Tests or Main Program

Return to the project root and run:

```
cargo test --bin coreml_test -- --test-threads=1
```

Or run the main program:

```
cargo run --bin coreml_test -- OpenELM-450M-Instruct-128-float32.mlmodelc
```

---

## Script Details: `download_and_compile.py`

- Checks for required dependencies (`huggingface_hub` and `coremlc`).
- Downloads the model and tokenizer from Hugging Face using the `huggingface_hub` Python package.
- Compiles the downloaded `.mlmodel` to CoreML format using `coremlc`.
- Prints clear error messages if dependencies are missing.

---

## Troubleshooting

- If you see an error about missing `huggingface_hub`, install it with `pip install huggingface_hub`.
- If you see an error about `coremlc`, ensure Xcode Command Line Tools are installed and your PATH is set up.
- If model or tokenizer download fails, check your internet connection and Hugging Face credentials (if required).

---

## License

This project is provided for research and educational purposes. Please respect the licenses of the model and tokenizer files you download from Hugging Face.
