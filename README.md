
# Automation Benchmarking Scripts

This repository provides two Python scripts for automated benchmarking and memory profiling of OpenVINO GenAI models. These tools help you set up models, run quantization, perform comprehensive benchmarks, and monitor memory usage in detail.

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
3. Run the benchmarking script:
   ```bash
   python endend.py
   ```
4. For memory profiling:
   ```bash
   python endend_memmon.py
   ```

---

## License

MIT
