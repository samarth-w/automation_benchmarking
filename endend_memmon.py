# endend_memmon.py
# Copy of endend.py with additional memory monitoring using tracemalloc and memray

import os
import sys
import subprocess
import importlib.util

# --- Bootstrap Dependency Checker ---
def check_and_install_dependencies():
    """
    Checks for required packages and installs them if missing.
    If packages are installed, it re-runs the script to ensure they are loaded.
    """
    required_packages = {
        "requests": "requests",
        "bs4": "beautifulsoup4"
    }
    missing = [name for lib, name in required_packages.items() if not importlib.util.find_spec(lib)]

    if missing:
        print(f"Missing required packages: {', '.join(missing)}. Attempting to install...")
        try:
            for package in missing:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print("\nPackages installed successfully. Relaunching the script...")
            # Relaunch the script to load the new packages
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except (subprocess.CalledProcessError, OSError) as e:
            print(f"\nFATAL: Failed to install packages: {e}", file=sys.stderr)
            print("Please install the missing packages manually and rerun the script.", file=sys.stderr)
            sys.exit(1)

check_and_install_dependencies()
# --- End Bootstrap ---

# --- Memory Monitoring Wrapper ---


def run_with_memory_monitoring(main_func):
    import tracemalloc
    import threading
    import time
    print("\n[Memory Monitor] Starting tracemalloc...")
    tracemalloc.start()

    # Try to use psutil for per-process memory monitoring
    try:
        import psutil
        def monitor_memory(stop_event, interval=1.0):
            proc = psutil.Process()
            print("[Memory Monitor] Per-process memory usage (MB):")
            while not stop_event.is_set():
                mem_info = proc.memory_info()
                rss = mem_info.rss / 1024 / 1024
                vms = mem_info.vms / 1024 / 1024
                children = proc.children(recursive=True)
                print(f"  Main PID {proc.pid}: RSS={rss:.2f} MB, VMS={vms:.2f} MB")
                for child in children:
                    try:
                        c_mem = child.memory_info()
                        c_rss = c_mem.rss / 1024 / 1024
                        c_vms = c_mem.vms / 1024 / 1024
                        print(f"    Child PID {child.pid}: RSS={c_rss:.2f} MB, VMS={c_vms:.2f} MB")
                    except Exception:
                        pass
                time.sleep(interval)

        stop_event = threading.Event()
        monitor_thread = threading.Thread(target=monitor_memory, args=(stop_event, 1.0), daemon=True)
        monitor_thread.start()
    except ImportError:
        print("[Memory Monitor] psutil not installed, skipping per-process memory monitoring.")

    # Try to use memory_profiler
    try:
        from memory_profiler import memory_usage
        print("[Memory Monitor] Running main workflow under memory_profiler...")
        mem_usage = memory_usage((main_func,), interval=0.5, retval=False)
        print(f"[Memory Monitor] memory_profiler peak memory usage: {max(mem_usage):.2f} MB")
    except ImportError:
        print("[Memory Monitor] memory_profiler not installed, skipping.")
        main_func()

    # Stop psutil monitor thread if running
    try:
        stop_event.set()
        monitor_thread.join(timeout=2)
    except Exception:
        pass

    current, peak = tracemalloc.get_traced_memory()
    print(f"[Memory Monitor] Tracemalloc current memory usage: {current / 1024 / 1024:.2f} MB")
    print(f"[Memory Monitor] Tracemalloc peak memory usage: {peak / 1024 / 1024:.2f} MB")
    tracemalloc.stop()

# --- Main Workflow Import ---
from endend import main as main_workflow

if __name__ == "__main__":
    run_with_memory_monitoring(main_workflow)
    # Replace main() with memory monitoring wrapper
    run_with_memory_monitoring(main)
