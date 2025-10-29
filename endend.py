import os
import sys

# --- ENSURE PROXY IS SET UP EARLY, BEFORE ANY NETWORK OPERATIONS ---
PROXY_URL = "http://proxy-iind.intel.com:911"
if os.environ.get("HTTP_PROXY") != PROXY_URL or os.environ.get("HTTPS_PROXY") != PROXY_URL:
    os.environ["HTTP_PROXY"] = PROXY_URL
    os.environ["HTTPS_PROXY"] = PROXY_URL

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

# ... rest of your script remains unchanged ...
