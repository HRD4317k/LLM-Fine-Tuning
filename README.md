# Deterministic Agentic Tool Router

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Unsloth](https://img.shields.io/badge/Powered_by-Unsloth-FF69B4.svg)](https://github.com/unslothai/unsloth)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)

A production-grade, highly modular Large Language Model (LLM) alignment repository specifically designed for **deterministic agentic tool routing**. This repository provides an end-to-end pipeline for dataset processing, parameter-efficient fine-tuning (QLoRA), and robust evaluation of open-source models for highly accurate function calling capabilities.

## Key Features

- **End-to-End Pipeline**: Modular pipelines for data processing, training, and evaluation.
- **Optimized QLoRA Fine-Tuning**: Leverages [Unsloth](https://unsloth.ai/) for 2x-5x faster training with minimal VRAM usage. It uses native `load_in_4bit=True` with NF4 quantization, Double Quantization, and paged optimizers (`paged_adamw_8bit`).
- **Flexible Data Parsers**: Supports multiple dataset formats out of the box (e.g., ShareGPT, xLAM format).
- **Robust Evaluation**: Comprehensive evaluation pipeline that measures true/false positives, syntax errors, hallucinated parameters, type mismatches, and generation latency.
- **Configuration Driven**: Easily manage experiments and hyperparameters using YAML configurations.
- **Built-in Smoke Tests**: Validate dataset compatibility and model parsers before initiating long training runs.

## Repository Structure

```text
.
├── configs/                  # YAML configuration files
│   ├── base.yaml             # Main training & evaluation config
│   └── smoke_test.yaml       # Fast smoke test override config
├── src/
│   ├── adapters/             # Dataset parsers (xLAM, ShareGPT)
│   ├── core/                 # Configuration schemas, error handling, diagnostics
│   ├── models/               # RouterAgent (Unsloth wrapper) and Validator
│   └── pipelines/            # Core logic (Data, Training, Evaluation pipelines)
├── main.py                   # Central CLI entry point
└── pyproject.toml            # Project metadata and dependencies
```

## Getting Started

### 1. Prerequisites

This repository requires an NVIDIA GPU (Ampere architecture or newer is recommended for optimal bfloat16 and Unsloth performance).

### 2. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/HRD4317k/LLM-Fine-Tuning.git
cd LLM-Fine-Tuning
pip install -e .
```

*Note: Unsloth installation can be environment-specific. Refer to the [official Unsloth installation guide](https://github.com/unslothai/unsloth) if you encounter issues.*

### 3. Configuration

Edit `configs/base.yaml` to set your desired model, dataset, and training hyperparameters:

```yaml
model:
  name_or_path: "unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit"
  max_seq_length: 2048
  load_in_4bit: true # Enables QLoRA

data:
  dataset_name: "Beryex/xlam-function-calling-60k-sharegpt"
  
training:
  per_device_train_batch_size: 2
  learning_rate: 2.0e-4
  num_train_epochs: 1
  optim: "paged_adamw_8bit"
  # ...
```

## Usage

The central entry point is `main.py`, which manages the execution phases.

### Smoke Testing
Always run a smoke test before a full training job to ensure your data parsers and configs are correct. This will only run a few steps and evaluate a handful of samples.

```bash
python main.py --phase all --profile smoke_test
```

### Full Pipeline Execution

Run the complete pipeline (Process Data -> Train -> Evaluate):

```bash
python main.py --phase all --profile base
```

### Running Individual Phases

You can execute specific phases of the pipeline sequentially:

1. **Process Data**: Downloads the dataset, carves out a held-out test split (2,000 rows, seed 3407) *before* anything else, adds "no matching tool" negatives, and formats the training rows with 4-step reasoning traces.
   ```bash
   python main.py --phase process
   ```
2. **Train Model**: Initiates QLoRA fine-tuning using `TRL`'s `SFTTrainer`.
   ```bash
   python main.py --phase train
   ```
3. **Evaluate Model**: Runs inference on the test set and calculates advanced function-calling metrics.
   ```bash
   python main.py --phase evaluate
   ```

## Evaluation & Metrics

The evaluation phase runs **only on the held-out split** (`data/raw_test`, never trained on) and grades every response with a schema-aware **AST checker** (`src/models/ast_checker.py`). Reports are saved in `outputs/evaluation_reports/`.

### AST evaluation

Each predicted call is parsed into a tree `{name, arguments}` and checked, in order, for:

1. **Syntax** – the `<tool_call>` block parses into JSON object(s)
2. **Tool** – the name exists in the provided tools (else *hallucinated tool*) and is the expected tool (else *wrong tool*)
3. **Parameters** – no parameter outside the schema (*hallucinated param*), every required parameter present (*missing param*); required parameters are read from each tool's schema
4. **Types** – each value matches the schema type (`str`, `int`, `float`, `bool`, `list`, `dict`)
5. **Values** – each value equals ground truth (numeric tolerance, case/whitespace-normalised strings)

Parallel calls are matched order-insensitively, so *every* call in a multi-call answer is graded.

Headline numbers in `bfcl_advanced_metrics.json`:
- `accuracy` = (correct calls + correct refusals) / total
- `valid_json_rate` = share of responses with a `<tool_call>` block whose JSON parses

The pipeline evaluates:
- **Accuracy**: True Positives, False Positives, False Negatives, True Negatives.
- **Failure Breakdown**: Categorizes errors into Syntax Errors, Hallucinated Tools/Params, Missing Params, Type Mismatches, and Value Mismatches.
- **Performance**: Measures Inference Latency (ms) and Tokens Per Second (TPS).

Visualizations (Confusion Matrix and Error Breakdown charts) are automatically generated for easy analysis.

## Data & Training Details

- **Held-out split first**: `src/pipelines/splits.py` splits xLAM 60k into 58k train / 2k test with a fixed seed and stores the settings in `data/split_meta.json`; the split is rebuilt if the settings change. Training never sees the test rows.
- **Reasoning traces**: every training response starts with a 4-step `<thought>` (tool assessment → intent matching → parameter extraction with schema-required parameters → validation) and covers *all* parallel calls.
- **Negatives**: `no_tool_negative_rate` (default 5%) turns positives into "no matching tool" examples by removing the target tool(s) from the tool list, teaching the `[ABORT]` path.
- **Response-only loss**: with `train_on_responses_only: true` the system prompt, tool list and query are masked out of the loss.

## QLoRA Implementation Details

This repository implements a textbook QLoRA setup following standard best practices:
- **NF4 Quantization**: Handled dynamically by Unsloth when `load_in_4bit=True`.
- **Double Quantization**: Enabled automatically by Unsloth to reduce VRAM overhead of quantization constants.
- **Paged Optimizers**: Explicitly configured using `paged_adamw_8bit` to offload optimizer states to CPU RAM during GPU memory spikes, preventing OOM errors.

## Tests

```bash
pytest -q tests/      # AST checker, trace generation, negatives, config (no GPU needed)
```

## Author

**Rishabh Dehariya** — IIT Kharagpur · [github.com/HRD4317k](https://github.com/HRD4317k)
