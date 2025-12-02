# Automation Benchmarking Scripts

This repository provides two Python scripts for automated benchmarking and memory profiling of OpenVINO GenAI models. These tools help you set up models, run quantization, perform comprehensive benchmarking, and generate detailed reports.

## Files

- **`endend.py`**  
  Main workflow for model setup, quantization, benchmarking, and reporting.
- **`endend_memmon.py`**  
  Wraps the main workflow with advanced memory monitoring using tracemalloc, memory_profiler, and psutil.

---

## Features

- **Automated Model Management:**  
  - Download, quantize, and prepare models for benchmarking.
  - Supports Hugging Face model IDs and batch processing via text files.
- **Flexible Device Support:**  
  - Benchmark on GPU, NPU, or both.
- **Quantization Options:**  
  - Groupwise and channelwise quantization, with duplicate prevention.
- **Comprehensive Benchmarking:**  
  - Customizable prompt files, config files, and benchmarking iterations.
  - Generates detailed logs and summary CSV reports.
- **Advanced Memory Monitoring:**  
  - Tracks memory usage for main and child processes in real time.
  - Reports peak memory usage and per-process stats in the terminal.
- **Automated Execution Mode:**  
  - Run the entire workflow unattended using a YAML configuration file.

---

## Usage

### 1. `endend.py`

- **Purpose:**  
  Interactive script for full benchmarking workflow.
- **How to Run:**  
  ```bash
  python endend.py
  ```
- **Workflow:**  
  1. Enter Hugging Face token (for gated models).
  2. Select device(s) for benchmarking.
  3. Specify models (IDs or file path).
  4. Choose quantization settings.
  5. Configure benchmarking options (prompts, config files, iterations).
  6. Models are processed, quantized, and benchmarked.
  7. Results and logs are saved in `benchmark_results`.

### 2. `endend_memmon.py`

- **Purpose:**  
  Same as `endend.py`, but with additional memory monitoring.
- **How to Run:**  
  ```bash
  python endend_memmon.py
  ```
- **Memory Monitoring:**  
  - Uses `tracemalloc` for Python allocations.
  - Uses `memory_profiler` for peak memory usage.
  - Uses `psutil` to print real-time memory usage for all processes.
  - All memory logs are printed to the terminal.

---

## Automated Execution with YAML Configuration

For unattended or CI/CD workflows, both scripts support a `--config` flag that accepts a YAML configuration file. This allows you to define all settings upfront and run the entire pipeline without interactive prompts.

### Command Line Usage

```bash
python endend.py --config CONFIG
```

| Argument | Description |
|----------|-------------|
| `--config CONFIG` | Path to YAML configuration file for automated execution |

### Example YAML Configuration File

```yaml
# config.yaml - Example configuration for automated benchmarking

# Proxy settings (optional)
proxy:
  enable: false
  url: "http://proxy.example.com:8080"

# Hugging Face authentication
hugging_face:
  token: "hf_your_token_here"

# Target device(s) for benchmarking
device:
  target: "both"  # Options: "gpu", "npu", "both"

# Models to process
models:
  list:
    - "meta-llama/Llama-3.1-8B"
    - "google/gemma-2-2b-it"
  quantization: "groupwise"  # Options: "groupwise", "channelwise", "both"

# Virtual environment settings
virtual_environment:
  name: "openvino_env"
  use_existing: true

# Benchmarking configuration
benchmarking:
  enable: true
  general_prompt_file: "prompts/general.jsonl"
  specific_prompt_folder: "prompts/model_specific/"
  config_file: "configs/benchmark_config.json"
  input_tokens: 128
  iterations: 3
```

### Configuration Options Explained

| Section | Key | Description | Default |
|---------|-----|-------------|---------|
| `proxy` | `enable` | Enable corporate proxy for network requests | `false` |
| `proxy` | `url` | Proxy server URL | - |
| `hugging_face` | `token` | HF token for accessing gated models (Llama, Gemma, etc.) | - |
| `device` | `target` | Target device(s): `gpu`, `npu`, or `both` | `both` |
| `models` | `list` | List of Hugging Face model IDs to process | - |
| `models` | `quantization` | Quantization type: `groupwise`, `channelwise`, or `both` | `groupwise` |
| `virtual_environment` | `name` | Name of the Python virtual environment | `openvino_env` |
| `virtual_environment` | `use_existing` | Reuse existing environment if valid | `true` |
| `benchmarking` | `enable` | Run benchmarks after model processing | `true` |
| `benchmarking` | `general_prompt_file` | Path to general prompt file (JSONL/TXT) | - |
| `benchmarking` | `specific_prompt_folder` | Folder with model-specific prompt files | - |
| `benchmarking` | `config_file` | Path to benchmark configuration file | - |
| `benchmarking` | `input_tokens` | Number of input tokens for benchmarking | `128` |
| `benchmarking` | `iterations` | Number of benchmark iterations per model | `1` |

---

## Requirements

- Python 3.10+
- OpenVINO GenAI dependencies (see `requirements.txt`)
- Additional packages for memory monitoring:
  - `memory_profiler`
  - `psutil`
- Git (for repository management)
- Internet access (for model downloads and Hugging Face authentication)

---

## Output

- **Benchmark Logs:**  
  Saved in `benchmark_results/benchmark_session_*` folders.
- **Summary Reports:**  
  CSV files with performance and memory metrics.
- **Memory Logs:**  
  Printed to the terminal during benchmarking (not saved to file by default).

---

## Tips & Troubleshooting

- Use a clean Python virtual environment for best results.
- If you want to save memory logs to a file, request a script modification.
- For large model batches, use a text file with model IDs.
- If you encounter permission errors, check file paths and access rights.
- For advanced memory analysis, use the `.DMP` files with WinDbg and pykd (see documentation).
- When using `--config`, ensure your YAML file is properly formatted and all paths are valid.

---

## Example Workflow

1. Clone the repository:
   ```bash
   git clone https://github.com/samarth-w/automation_benchmarking.git
   cd automation_benchmarking
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install memory_profiler psutil
   ```
3. Run the benchmarking script (interactive mode):
   ```bash
   python endend.py
   ```
4. Run with automated configuration:
   ```bash
   python endend.py --config my_config.yaml
   ```
5. For memory profiling:
   ```bash
   python endend_memmon.py
   ```

---

## License

MIT
