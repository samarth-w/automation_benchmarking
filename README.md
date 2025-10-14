# Automation Benchmarking Scripts

This repository contains two main Python scripts for benchmarking and memory monitoring of OpenVINO GenAI models:

## Files

- `endend.py`: Main benchmarking workflow script
- `endend_memmon.py`: Memory-monitored benchmarking script (uses tracemalloc, memory_profiler, and psutil)

---

## Usage

### 1. `endend.py`
- Run this script to perform model setup, quantization, benchmarking, and reporting.
- Interactive prompts will guide you through configuration (Hugging Face token, device selection, model IDs, quantization, etc.).
- Benchmark results and logs are saved in the `benchmark_results` folder.

#### Example:
```bash
python endend.py
```

### 2. `endend_memmon.py`
- Wraps the main workflow with advanced memory monitoring.
- Tracks memory usage using tracemalloc, memory_profiler, and psutil (per-process stats).
- Prints detailed memory logs to the terminal during benchmarking.
- Requires `memory_profiler` and `psutil` (install with `pip install memory_profiler psutil`).

#### Example:
```bash
python endend_memmon.py
```

---

## Requirements
- Python 3.10+
- OpenVINO GenAI dependencies (see `requirements.txt` in the repo)
- Additional packages for memory monitoring:
  - `memory_profiler`
  - `psutil`

---

## Output
- Benchmark logs and summary CSVs are saved in `benchmark_results/benchmark_session_*` folders.
- Memory usage logs are printed to the terminal (not saved to file by default).

---

## Tips
- For best results, run in a clean Python virtual environment.
- If you want to save memory logs to a file, ask for a script modification.
- For troubleshooting, check the terminal output and log files in `benchmark_results`.

---

## License
MIT
