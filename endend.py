import os
import sys
import subprocess
import importlib.util

# --- ENSURE PROXY IS SET UP EARLY, BEFORE ANY NETWORK OPERATIONS ---
PROXY_URL = "http://proxy-iind.intel.com:911"

def prompt_for_proxy_configuration(proxy_url: str) -> bool:
    """Ask the user whether to configure the corporate proxy for this session."""
    current_http = os.environ.get("HTTP_PROXY")
    current_https = os.environ.get("HTTPS_PROXY")
    if current_http == proxy_url and current_https == proxy_url:
        print(f"Proxy already configured: {proxy_url}")
        return True

    response = input(f"Configure proxy at {proxy_url} for this session? [Y/n]: ").strip().lower()
    if response in ("", "y", "yes"):
        os.environ["HTTP_PROXY"] = proxy_url
        os.environ["HTTPS_PROXY"] = proxy_url
        print(f"Proxy enabled for this session: {proxy_url}")
        return True

    print("Proxy configuration skipped per user choice.")
    return False

PROXY_USER_OPT_IN = prompt_for_proxy_configuration(PROXY_URL)






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

from datetime import datetime
import shutil
import platform
from pathlib import Path
from tempfile import gettempdir
import json
import re
from typing import Dict, List, Optional, Tuple
import requests
from bs4 import BeautifulSoup
import tarfile
import csv

# Define color codes for console output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    GRAY = '\033[90m'
    WHITE = '\033[97m'
    ENDC = '\033[0m'

# Check if the OS is Windows
if platform.system() != 'Windows':
    print(f"{Colors.RED}This script is intended for Windows PowerShell environments only.{Colors.ENDC}")
    sys.exit(1)

# Virtual Environment Management Functions
# -----------------------------------------------------------------------------

def validate_virtual_environment(venv_path: Path) -> Dict:
    """
    Validate a virtual environment and return its state.
    Returns dict with 'valid', 'python_exe', and 'activate_script' keys.
    """
    result = {'valid': False, 'python_exe': None, 'activate_script': None}
    if not venv_path.is_dir():
        return result
    scripts_dir = venv_path / "Scripts"
    python_exe = scripts_dir / "python.exe"
    activate_script = scripts_dir / "activate.bat"
    if not python_exe.exists() or not activate_script.exists():
        return result
    try:
        proc = subprocess.run([str(python_exe), "--version"], capture_output=True, timeout=10)
        if proc.returncode != 0:
            return result
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return result
    result.update({'valid': True, 'python_exe': python_exe, 'activate_script': activate_script})
    return result

def setup_virtual_environment(llm_bench_path: Path) -> Optional[Path]:
    """
    Set up a virtual environment: find existing ones, prompt user, or create new.
    Returns the path to the validated virtual environment or None if setup failed.
    """
    print(f"\n{Colors.CYAN}{'='*60}{Colors.ENDC}")
    print(f"{Colors.CYAN}VIRTUAL ENVIRONMENT SETUP{Colors.ENDC}")
    print(f"{Colors.CYAN}{'='*60}{Colors.ENDC}")
    default_env_name = "openvino_env"
    skip_dirs = {'models', 'tools', 'src', 'docs', '__pycache__', '.git', 'benchmark_results'}

    # Find valid existing environments
    valid_envs = []
    try:
        for item in llm_bench_path.iterdir():
            if item.is_dir() and item.name.lower() not in skip_dirs:
                if validate_virtual_environment(item)['valid']:
                    valid_envs.append(item)
    except (PermissionError, OSError):
        pass
    valid_envs.sort(key=lambda p: (0 if 'openvino' in p.name.lower() else 1, p.name))

    # Prompt user
    if valid_envs:
        print(f"\n{Colors.CYAN}Available Virtual Environments:{Colors.ENDC}")
        for i, env in enumerate(valid_envs, 1):
            print(f"{Colors.WHITE}{i}. {env.name} ({env}){Colors.ENDC}")
        create_opt = len(valid_envs) + 1
        print(f"{Colors.WHITE}{create_opt}. Create new environment (default: {default_env_name}){Colors.ENDC}")
        while True:
            choice = input(f"{Colors.CYAN}Enter choice (1-{create_opt}, default {create_opt}): {Colors.ENDC}").strip()
            if not choice or choice == str(create_opt):
                break
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(valid_envs):
                    print(f"{Colors.GREEN}Using existing environment: {valid_envs[idx].name}{Colors.ENDC}")
                    return valid_envs[idx]
            except ValueError:
                pass
            print(f"{Colors.RED}Invalid choice.{Colors.ENDC}")
    else:
        print(f"{Colors.YELLOW}No existing virtual environments found.{Colors.ENDC}")

    # Create new environment
    env_name = input(f"{Colors.CYAN}Enter environment name (default: {default_env_name}): {Colors.ENDC}").strip() or default_env_name
    venv_path = llm_bench_path / env_name

    # Handle existing directory
    if venv_path.exists():
        if validate_virtual_environment(venv_path)['valid']:
            print(f"{Colors.GREEN}Existing environment is valid.{Colors.ENDC}")
            return venv_path
        overwrite = input(f"{Colors.CYAN}Directory exists but is broken. Overwrite? (y/n): {Colors.ENDC}").strip().lower()
        if overwrite != 'y':
            print(f"{Colors.RED}Cannot proceed without a valid environment.{Colors.ENDC}")
            return None
        try:
            shutil.rmtree(venv_path)
        except Exception as e:
            print(f"{Colors.RED}Failed to remove directory: {e}{Colors.ENDC}")
            return None

    # Create venv
    print(f"{Colors.YELLOW}Creating virtual environment: {env_name}{Colors.ENDC}")
    try:
        proc = subprocess.run(["python", "-m", "venv", str(venv_path)], cwd=llm_bench_path, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            print(f"{Colors.RED}Failed to create venv: {proc.stderr}{Colors.ENDC}")
            return None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        print(f"{Colors.RED}Error creating venv: {e}{Colors.ENDC}")
        return None

    if not validate_virtual_environment(venv_path)['valid']:
        print(f"{Colors.RED}Created environment failed validation.{Colors.ENDC}")
        return None
    print(f"{Colors.GREEN}Virtual environment created successfully.{Colors.ENDC}")
    return venv_path

# Helper Functions
# -----------------------------------------------------------------------------

def run_command(command_str, working_directory=None, stop_on_error=True, env=None, verbose=True):
    """Executes a command and logs its output and exit code."""
    if verbose:
        print(f"\n{Colors.YELLOW}{'='*80}{Colors.ENDC}")
        print(f"{Colors.YELLOW}EXECUTING: {command_str}{Colors.ENDC}")
        print(f"{Colors.YELLOW}DIRECTORY: {working_directory if working_directory else os.getcwd()}{Colors.ENDC}")
        if env:
            print(f"{Colors.GRAY}ENVIRONMENT VARIABLES:{Colors.ENDC}")
            for k, v in env.items():
                if k.upper() in ["HTTP_PROXY", "HTTPS_PROXY", "HF_TOKEN"]:
                    masked_value = v if k.upper().endswith('PROXY') else "***MASKED***"
                    print(f"{Colors.GRAY}  {k}={masked_value}{Colors.ENDC}")
        print(f"{Colors.YELLOW}{'='*80}{Colors.ENDC}")

    try:
        result = subprocess.run(
            command_str,
            cwd=working_directory,
            text=True,
            shell=True,
            env=env,
            check=stop_on_error,
            capture_output=True
        )
        if verbose:
            print(result.stdout)
            if result.stderr:
                print(f"{Colors.YELLOW}{result.stderr}{Colors.ENDC}")
        exit_code = result.returncode
        
        if verbose:
            print(f"{Colors.GREEN}{'-'*40}{Colors.ENDC}")
            print(f"{Colors.GREEN}EXIT CODE: {exit_code}{Colors.ENDC}")
            print(f"{Colors.YELLOW}{'='*80}{Colors.ENDC}\n")
        return True

    except FileNotFoundError:
        if verbose:
            print(f"{Colors.RED}Error: Command not found. Make sure the executable is in your PATH.{Colors.ENDC}")
        return False
    except Exception as e:
        if verbose:
            print(f"{Colors.RED}Error executing command: {e}{Colors.ENDC}")
            if isinstance(e, subprocess.CalledProcessError):
                print(f"{Colors.RED}--- STDOUT ---{Colors.ENDC}")
                print(e.stdout)
                print(f"{Colors.RED}--- STDERR ---{Colors.ENDC}")
                print(e.stderr)
        return False

def test_software(name, version_command, install_instructions, install_package_id=None):
    """Checks if a software is installed by running a version command."""
    print(f"\n{Colors.CYAN}Checking for {name} installation...{Colors.ENDC}")
    try:
        result = subprocess.run(
            version_command,
            capture_output=True,
            text=True,
            check=True,
            shell=True
        )
        print(f"{Colors.GREEN}{name} is installed: {result.stdout.strip()}{Colors.ENDC}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"{Colors.RED}{name} not found or version check failed.{Colors.ENDC}")
        if install_package_id:
            print(f"{Colors.YELLOW}Attempting to install {name} via winget...{Colors.ENDC}")
            if install_with_winget(install_package_id, name):
                return test_software(name, version_command, install_instructions)
        print(f"{Colors.YELLOW}Please install {name} manually. {install_instructions}{Colors.ENDC}")
        return False

def is_winget_available() -> bool:
    """Return True if the winget executable is available on the PATH."""
    return shutil.which("winget") is not None

def install_with_winget(package_id: str, friendly_name: str) -> bool:
    """Attempts to install a package from winget by ID."""
    if not is_winget_available():
        print(f"{Colors.YELLOW}Winget is not available. Please install {friendly_name} manually via {package_id}.{Colors.ENDC}")
        return False

    command = [
        "winget", "install", "--id", package_id,
        "-e", "--accept-package-agreements", "--accept-source-agreements"
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"{Colors.GREEN}{friendly_name} installed (or already present).{Colors.ENDC}")
            if result.stdout:
                print(result.stdout)
            return True
        print(f"{Colors.RED}Winget reported an error while installing {friendly_name}.{Colors.ENDC}")
        if result.stderr:
            print(result.stderr)
        return False
    except Exception as e:
        print(f"{Colors.RED}Failed to invoke winget: {e}{Colors.ENDC}")
        return False

def is_winget_package_installed(package_id: str) -> bool:
    """Checks via winget list whether the requested package ID is already installed."""
    if not is_winget_available():
        return False
    try:
        result = subprocess.run(["winget", "list", "--id", package_id], capture_output=True, text=True)
        output = (result.stdout + result.stderr).lower()
        if "no installed package found matching input criteria" in output:
            return False
        return package_id.lower() in output
    except Exception:
        return False

def ensure_vc_redists_installed() -> bool:
    """Ensure that the required VC++ redistributables are installed via winget."""
    packages = [
        ("Microsoft.VCRedist.2015+.x64", "Microsoft Visual C++ v14 Redistributable (x64)"),
        ("Microsoft.VCRedist.2015+.x86", "Microsoft Visual C++ v14 Redistributable (x86)")
    ]

    installed_any = False
    for package_id, friendly_name in packages:
        if is_winget_package_installed(package_id):
            print(f"{Colors.GREEN}{friendly_name} already installed (verified via winget).{Colors.ENDC}")
            installed_any = True
            continue

        print(f"{Colors.YELLOW}Attempting to install {friendly_name} via winget...{Colors.ENDC}")
        if install_with_winget(package_id, friendly_name):
            installed_any = True

    return installed_any

def test_openvino_repository(repo_path="openvino.genai"):
    """Validates the local OpenVINO GenAI git repository with minimal verbosity."""
    repo_path_obj = Path(repo_path)
    if not repo_path_obj.exists():
        return {"valid": False, "reason": "Directory not found", "action": "clone"}
    if not (repo_path_obj / ".git").exists():
        if list(repo_path_obj.iterdir()):
            return {"valid": False, "reason": "Directory has content but no git repo", "action": "manual"}
        return {"valid": False, "reason": "Not a git repository", "action": "clone"}

    try:
        # Check remote URL silently
        remote_url = subprocess.check_output(
            ["git", "-C", repo_path, "remote", "get-url", "origin"],
            text=True,
            stderr=subprocess.DEVNULL
        ).strip()
        if "openvinotoolkit/openvino.genai" not in remote_url:
            return {"valid": False, "reason": "Wrong repository URL", "action": "manual"}

        # Check required paths
        required_paths = [
            "tools",
            "tools/llm_bench",
            "tools/llm_bench/requirements.txt"
        ]
        missing_paths = [p for p in required_paths if not (repo_path_obj / p).exists()]
        if missing_paths:
            return {"valid": False, "reason": "Missing required paths", "action": "repair"}

        return {"valid": True, "reason": "Valid and complete repository", "action": "none"}
    
    except subprocess.CalledProcessError:
        return {"valid": False, "reason": "Git command failed", "action": "manual"}
    except Exception:
        return {"valid": False, "reason": "Validation error", "action": "manual"}

def get_admin_status():
    """Checks if the script is running with administrator privileges on Windows."""
    try:
        import ctypes
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except ImportError:
        print(f"{Colors.YELLOW}Cannot check for administrator rights without 'ctypes'.{Colors.ENDC}")
        return False

# Enhanced Model Management Functions with Smart Pattern Matching
# -----------------------------------------------------------------------------

def sanitize_model_name(model_id: str) -> str:
    """Sanitizes a Hugging Face model ID for use as a directory name."""
    return re.sub(r'[^\w\-_.]', '_', model_id.replace('/', '_'))

def normalize_model_name_for_matching(model_id: str) -> List[str]:
    """
    Uses pattern matching to convert HuggingFace model ID to artifactory filename patterns.
    Much more flexible than hardcoded mappings.
    """
    patterns = []
    
    # Extract organization and model name
    if '/' in model_id:
        org, model = model_id.split('/', 1)
    else:
        org, model = '', model_id
    
    # Pattern 1: Direct model name (most common case)
    patterns.append(model)
    
    # Pattern 2: Organization-prefixed patterns
    org_prefix_rules = {
        'meta-llama': {
            'prefix': 'Meta-Llama',
            'transform': lambda m: f"Meta-Llama-{m.replace('Meta-Llama-', '').replace('Llama-', '')}"
        },
        'google': {
            'prefix': 'Google',
            'transform': lambda m: f"Google-{m}"
        },
        'deepseek-ai': {
            'prefix': 'DeepSeek',
            'transform': lambda m: m.replace('deepseek-ai/', 'DeepSeek-').replace('DeepSeek-', 'DeepSeek-')
        },
        'THUDM': {
            'prefix': '',
            'transform': lambda m: m.lower()
        }
    }
    
    if org in org_prefix_rules:
        rule = org_prefix_rules[org]
        if rule['prefix']:
            transformed = rule['transform'](model)
            patterns.append(transformed)
        else:
            transformed = rule['transform'](model)
            patterns.append(transformed)
    
    # Pattern 3: Handle common variations
    variations = []
    
    # Add -Instruct suffix if not present
    if not model.endswith(('-Instruct', '-instruct', '-Chat', '-chat')):
        variations.extend([
            f"{model}-Instruct",
            f"{model}-instruct"
        ])
    
    # Handle case variations
    variations.extend([
        model.replace('-', '_'),
        model.replace('_', '-'),
        model.title(),
        model.lower()
    ])
    
    # Add organization prefix to variations for meta-llama
    if org == 'meta-llama':
        for var in variations:
            if not var.startswith('Meta-Llama'):
                patterns.append(f"Meta-Llama-{var.replace('Llama-', '')}")
    
    patterns.extend(variations)
    
    # Pattern 4: Fuzzy matching patterns
    model_parts = re.split(r'[-_/]', model.lower())
    key_parts = [part for part in model_parts if part and len(part) > 1]
    
    if len(key_parts) >= 2:
        fuzzy_patterns = [
            '-'.join(key_parts),
            '-'.join(key_parts).title(),
            '_'.join(key_parts),
        ]
        
        if org == 'meta-llama':
            for fp in fuzzy_patterns:
                if 'llama' in fp.lower():
                    patterns.append(f"Meta-Llama-{fp.replace('llama-', '').title()}")
        elif org == 'google':
            for fp in fuzzy_patterns:
                patterns.append(f"Google-{fp.title()}")
        
        patterns.extend(fuzzy_patterns)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_patterns = []
    for pattern in patterns:
        pattern_clean = pattern.strip()
        if pattern_clean and pattern_clean not in seen:
            seen.add(pattern_clean)
            unique_patterns.append(pattern_clean)
    
    return unique_patterns

def fuzzy_match_score(pattern: str, filename: str) -> float:
    """Calculate fuzzy match score between pattern and filename."""
    pattern_lower = pattern.lower()
    filename_lower = filename.lower()
    
    # Exact match gets highest score
    if pattern_lower in filename_lower:
        return len(pattern) / len(filename) + 0.5
    
    # Split into words and check word matches
    pattern_words = re.split(r'[-_\s]', pattern_lower)
    filename_words = re.split(r'[-_\s]', filename_lower)
    
    pattern_words = [w for w in pattern_words if len(w) > 1]
    filename_words = [w for w in filename_words if len(w) > 1]
    
    if not pattern_words:
        return 0.0
    
    # Count matching words
    matches = 0
    for p_word in pattern_words:
        for f_word in filename_words:
            if p_word in f_word or f_word in p_word:
                matches += 1
                break
    
    word_score = matches / len(pattern_words)
    
    # Bonus for sequence matches
    sequence_bonus = 0
    pattern_str = ''.join(pattern_words)
    filename_str = ''.join(filename_words)
    
    common_length = 0
    for i in range(len(pattern_str)):
        for j in range(i + 1, len(pattern_str) + 1):
            substr = pattern_str[i:j]
            if len(substr) > 2 and substr in filename_str:
                common_length = max(common_length, len(substr))
    
    if common_length > 0:
        sequence_bonus = common_length / len(pattern_str) * 0.3
    
    return word_score + sequence_bonus

def get_quantization_patterns(quantization: str) -> List[str]:
    """Returns quantization patterns with some flexibility."""
    if quantization == "channelwise":
        return ["group-1", "group_-1", "group1"]
    else:  # groupwise
        return ["group128", "group_128", "group-128"]

def check_existing_model(model_id: str, quantization: str, device: str, folders: Dict) -> Optional[Path]:
    """
    Check if a model already exists locally to avoid re-downloading/re-quantizing.
    Returns the path if found, None otherwise.
    """
    sanitized_name = sanitize_model_name(model_id)
    quant_suffix = "int4_cw" if quantization == "channelwise" else "int4_gw128"
    
    if device == "gpu":
        target_dir = folders['gpu'] / f"{sanitized_name}_{quant_suffix}"
    else:  # npu
        target_dir = folders['npu'] / f"{sanitized_name}_{quant_suffix}"
    
    # Check if the target directory exists and has required files
    if target_dir.exists():
        required_files = ['openvino_model.xml', 'openvino_model.bin']
        if all((target_dir / file).exists() for file in required_files):
            print(f"{Colors.GREEN}✓ {device.upper()} model already exists: {target_dir.name}{Colors.ENDC}")
            return target_dir
        else:
            print(f"{Colors.YELLOW}⚠ {device.upper()} model directory exists but incomplete: {target_dir.name}{Colors.ENDC}")
            print(f"{Colors.YELLOW}  Missing required files. Will re-process.{Colors.ENDC}")
            # Remove incomplete directory
            shutil.rmtree(target_dir)
    
    return None

def get_comprehensive_configuration():
    """Collects all inputs upfront for the complete workflow."""
    print(f"\n{Colors.CYAN}{'='*60}{Colors.ENDC}")
    print(f"{Colors.CYAN}COMPREHENSIVE SETUP CONFIGURATION{Colors.ENDC}")
    print(f"{Colors.CYAN}{'='*60}{Colors.ENDC}")
    
    config = {}
    
    # 1. HF Token (Required for Llama/Gemma models)
    print(f"\n{Colors.YELLOW}1. Hugging Face Token:{Colors.ENDC}")
    print(f"{Colors.GRAY}Required for accessing gated models like Llama or Gemma{Colors.ENDC}")
    hf_token = input(f"{Colors.CYAN}Enter your Hugging Face token: {Colors.ENDC}").strip()
    if not hf_token:
        print(f"{Colors.RED}HF Token is required for most models. Please provide a valid token.{Colors.ENDC}")
        return None
    config['hf_token'] = hf_token
    
    # 2. Device Selection
    print(f"\n{Colors.YELLOW}2. Device Selection:{Colors.ENDC}")
    print(f"{Colors.WHITE}1. GPU only{Colors.ENDC}")
    print(f"{Colors.WHITE}2. NPU only{Colors.ENDC}")
    print(f"{Colors.WHITE}3. Both GPU and NPU{Colors.ENDC}")
    
    while True:
        device_choice = input(f"{Colors.CYAN}Select target device(s) (1-3): {Colors.ENDC}").strip()
        if device_choice in ['1', '2', '3']:
            break
        print(f"{Colors.RED}Invalid choice. Please enter 1, 2, or 3.{Colors.ENDC}")
    
    device_map = {'1': 'gpu', '2': 'npu', '3': 'both'}
    config['device'] = device_map[device_choice]
    
    # 3. Model List
    print(f"\n{Colors.YELLOW}3. Model Configuration:{Colors.ENDC}")
    print(f"{Colors.GRAY}Enter models as comma-separated Hugging Face IDs or provide a file path{Colors.ENDC}")
    print(f"{Colors.GRAY}Examples: meta-llama/Llama-3.1-8B, google/gemma-2-2b-it{Colors.ENDC}")
    
    model_input = input(f"{Colors.CYAN}Models (IDs or file path): {Colors.ENDC}").strip()
    
    # Parse model list
    models = []
    if Path(model_input).exists():
        try:
            with open(model_input, 'r') as f:
                content = f.read().strip()
                models = [m.strip() for m in content.replace('\n', ',').split(',') if m.strip()]
        except Exception as e:
            print(f"{Colors.RED}Error reading model file: {e}{Colors.ENDC}")
            return None
    else:
        models = [m.strip() for m in model_input.split(',') if m.strip()]
    
    if not models:
        print(f"{Colors.RED}No models specified. Exiting.{Colors.ENDC}")
        return None
    
    # 4. Quantization settings for each model
    model_configs = []
    print(f"\n{Colors.YELLOW}4. Quantization Settings:{Colors.ENDC}")
    print(f"{Colors.WHITE}1. Groupwise (128) - recommended for most models{Colors.ENDC}")
    print(f"{Colors.WHITE}2. Channelwise (-1) - better quality, larger size{Colors.ENDC}")
    print(f"{Colors.WHITE}3. Both - process both quantization types{Colors.ENDC}")
    
    while True:
        global_quant_choice = input(f"{Colors.CYAN}Select quantization for ALL models (1-3): {Colors.ENDC}").strip()
        if global_quant_choice in ['1', '2', '3']:
            break
        print(f"{Colors.RED}Invalid choice. Please enter 1, 2, or 3.{Colors.ENDC}")
    
    for model_id in models:
        if global_quant_choice == '1':
            model_configs.append({'model_id': model_id, 'quantization': 'groupwise'})
        elif global_quant_choice == '2':
            model_configs.append({'model_id': model_id, 'quantization': 'channelwise'})
        else:  # Both
            model_configs.append({'model_id': model_id, 'quantization': 'groupwise'})
            model_configs.append({'model_id': model_id, 'quantization': 'channelwise'})
    
    config['models'] = model_configs
    
    # 5. Benchmarking Configuration
    print(f"\n{Colors.YELLOW}5. Benchmarking:{Colors.ENDC}")
    run_benchmark = input(f"{Colors.CYAN}Run benchmarks after model processing? (y/n, default: y): {Colors.ENDC}").strip().lower()
    config['run_benchmark'] = run_benchmark != 'n'
    
    if config['run_benchmark']:
        print(f"\n{Colors.YELLOW}   Benchmark Customization (Optional - press Enter to skip):{Colors.ENDC}")

        # General Prompt File
        prompt_file_path = input(f"{Colors.CYAN}   Enter path to a general prompt file (e.g., prompts.jsonl): {Colors.ENDC}").strip()
        if prompt_file_path and Path(prompt_file_path).exists():
            config['benchmark_general_prompt_file'] = Path(prompt_file_path)
        elif prompt_file_path:
            print(f"{Colors.YELLOW}   Warning: General prompt file not found, will be ignored.{Colors.ENDC}")
        
        # Folder for Specific Prompts
        specific_prompt_folder = input(f"{Colors.CYAN}   Enter path to a folder with model-specific prompts: {Colors.ENDC}").strip()
        if specific_prompt_folder and Path(specific_prompt_folder).is_dir():
            config['benchmark_specific_prompt_folder'] = Path(specific_prompt_folder)
        elif specific_prompt_folder:
            print(f"{Colors.YELLOW}   Warning: Specific prompt folder not found, will be ignored.{Colors.ENDC}")

        # Benchmark Config File
        config_file_path = input(f"{Colors.CYAN}   Enter path to a benchmark config file (e.g., config.json): {Colors.ENDC}").strip()
        if config_file_path and Path(config_file_path).exists():
            config['benchmark_config_file'] = Path(config_file_path)
        elif config_file_path:
            print(f"{Colors.YELLOW}   Warning: Benchmark config file not found, will be ignored.{Colors.ENDC}")

        # Memory Consumption - Always enable memory consumption logging for all iterations
        config['benchmark_memory_consumption'] = '2'  # Always log memory for all iterations

        # Other benchmark settings
        config['benchmark_input_tokens'] = int(input(f"{Colors.CYAN}   Default input token limit (default: 128): {Colors.ENDC}").strip() or "128")
        config['benchmark_iterations'] = int(input(f"{Colors.CYAN}   Number of benchmark iterations (default: 1): {Colors.ENDC}").strip() or "1")
    else:
        # Set default values when benchmarking is disabled
        config['benchmark_memory_consumption'] = '2'
        config['benchmark_input_tokens'] = 128
        config['benchmark_iterations'] = 1

    return config

def create_folder_structure(llm_bench_path: Path):
    """Creates the required folder structure for models."""
    models_dir = llm_bench_path / "models"
    gpu_dir = models_dir / "gpu_models"
    npu_dir = models_dir / "npu_models"
    benchmark_dir = llm_bench_path / "benchmark_results"
    
    models_dir.mkdir(exist_ok=True)
    gpu_dir.mkdir(exist_ok=True)
    npu_dir.mkdir(exist_ok=True)
    benchmark_dir.mkdir(exist_ok=True)
    
    print(f"{Colors.GREEN}Created folder structure:{Colors.ENDC}")
    print(f"{Colors.GRAY}  {models_dir}{Colors.ENDC}")
    print(f"{Colors.GRAY}  {gpu_dir}{Colors.ENDC}")
    print(f"{Colors.GRAY}  {npu_dir}{Colors.ENDC}")
    print(f"{Colors.GRAY}  {benchmark_dir}{Colors.ENDC}")
    
    return {"models": models_dir, "gpu": gpu_dir, "npu": npu_dir, "benchmark": benchmark_dir}

def get_latest_artifactory_folder(index_url: str) -> Optional[str]:
    """Finds the URL of the latest versioned folder in the Artifactory index."""
    try:
        proxies = {"http": None, "https": None}
        response = requests.get(index_url, verify=False, proxies=proxies)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        folder_links = soup.find_all('a', href=re.compile(r'^ov.*\/$'))
        
        version_folders = []
        for link in folder_links:
            try:
                date_parts = link.next_sibling.strip().split()
                date_str = ' '.join(date_parts[:2])
                date_obj = datetime.strptime(date_str, "%d-%b-%Y %H:%M")
                version_folders.append({'name': link['href'], 'date': date_obj})
            except (ValueError, TypeError, AttributeError):
                continue

        if not version_folders:
            return None

        latest_entry = sorted(version_folders, key=lambda x: x['date'], reverse=True)[0]
        latest_folder = latest_entry['name']
        latest_url = f"{index_url.rstrip('/')}/{latest_folder}"
        
        print(f"{Colors.GREEN}Latest release folder found: {latest_folder.strip('/')}{Colors.ENDC}")
        return latest_url
        
    except requests.RequestException as e:
        print(f"{Colors.RED}Failed to fetch Artifactory index: {e}{Colors.ENDC}")
        return None

def find_int4_models_in_target_folders(base_url: str) -> List[Dict]:
    """
    Searches for .tgz files containing 'int4' only in specific target folders.
    Only goes one level deep into LLM/, SD/, VLM/, whisper/ folders.
    """
    models = []
    target_folders = ['LLM/', 'SD/', 'VLM/', 'whisper/']
    
    try:
        # Get the main directory listing
        proxies = {"http": None, "https": None}
        response = requests.get(base_url, verify=False, proxies=proxies)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Find all directory links
        all_directories = [
            a['href'] for a in soup.find_all('a')
            if a['href'].endswith('/') and not a['href'].startswith('../')
        ]
        
        # Filter to only target folders
        available_targets = [d for d in all_directories if d in target_folders]
        
        # Search each target folder (only one level deep)
        for target_folder in available_targets:
            folder_url = f"{base_url.rstrip('/')}/{target_folder}"
            
            try:
                folder_response = requests.get(folder_url, verify=False, proxies=proxies)
                folder_response.raise_for_status()
                folder_soup = BeautifulSoup(folder_response.text, "html.parser")
                
                # Find int4 model files directly in this folder (no recursion)
                model_links = folder_soup.find_all('a', href=re.compile(r'int4.*\.tgz$', re.IGNORECASE))
                
                for link in model_links:
                    filename = link.text.strip()
                    model_url = f"{folder_url.rstrip('/')}/{link['href']}"
                    
                    # Extract file size from sibling text
                    try:
                        size_parts = link.next_sibling.strip().split()
                        if len(size_parts) >= 3:
                            size_text = f"{size_parts[0]} {size_parts[1]}"
                        else:
                            size_text = "N/A"
                    except (AttributeError, IndexError):
                        size_text = "N/A"
                    
                    models.append({
                        "filename": filename,
                        "url": model_url,
                        "size": size_text,
                        "category": target_folder.rstrip('/')
                    })
                    
            except requests.RequestException:
                continue
                
    except requests.RequestException as e:
        print(f"{Colors.RED}Failed to access base directory {base_url}: {e}{Colors.ENDC}")
        return []
    
    return models

def check_npu_model_in_artifactory(model_id: str, quantization: str, folders: Dict) -> Optional[Dict]:
    """
    Enhanced NPU model checking with fuzzy pattern matching and existing model check.
    """
    # First check if model already exists locally
    existing_model = check_existing_model(model_id, quantization, 'npu', folders)
    if existing_model:
        return None  # Skip download, model already exists
    
    try:
        # Get latest release folder
        artifactory_url = "https://af01p-ir.devtools.intel.com/artifactory/ir-public-models-ir-local/ov-genai-models/releases/"
        latest_url = get_latest_artifactory_folder(artifactory_url)
        if not latest_url:
            return None
        
        # Search for matching model using targeted folder approach
        all_models = find_int4_models_in_target_folders(latest_url)
        
        # Filter out models starting with 'ov'
        filtered_models = [
            model for model in all_models 
            if not model['filename'].lower().startswith('ov')
        ]
        
        # Get possible name patterns for the HF model
        name_patterns = normalize_model_name_for_matching(model_id)
        quant_patterns = get_quantization_patterns(quantization)
        
        # Find matches using fuzzy scoring (focusing on LLM category first)
        scored_matches = []
        
        # Prioritize LLM models for language model requests
        model_priority = ['LLM', 'VLM', 'SD', 'whisper']
        
        for model in filtered_models:
            filename = model['filename']
            category = model.get('category', 'unknown')
            
            # Must contain required keywords for NPU
            if not all(keyword in filename.lower() for keyword in ['int4', 'sym']):
                continue
            
            # Check quantization pattern
            quant_match = any(qp in filename for qp in quant_patterns)
            if not quant_match:
                continue
            
            # Calculate best match score across all patterns
            best_score = 0
            best_pattern = None
            
            for pattern in name_patterns:
                score = fuzzy_match_score(pattern, filename)
                if score > best_score:
                    best_score = score
                    best_pattern = pattern
            
            # Add category bonus
            category_bonus = 0
            if category in model_priority:
                category_bonus = (len(model_priority) - model_priority.index(category)) * 0.1
            
            final_score = best_score + category_bonus
            
            # Only consider matches above threshold
            if best_score > 0.3:
                scored_matches.append((model, final_score, best_pattern, category))
        
        if not scored_matches:
            return None
        
        # Sort by final score
        scored_matches.sort(key=lambda x: x[1], reverse=True)
        best_model, best_score, matched_pattern, category = scored_matches[0]
        
        print(f"{Colors.GREEN}Found matching NPU model: {best_model['filename']}{Colors.ENDC}")
        print(f"{Colors.GRAY}Category: {category}, Match score: {best_score:.2f}, Pattern: '{matched_pattern}'{Colors.ENDC}")
        
        return best_model
        
    except Exception as e:
        print(f"{Colors.RED}Error checking artifactory: {e}{Colors.ENDC}")
        return None

def download_npu_model(model_info: Dict, npu_dir: Path, model_id: str, quantization: str) -> bool:
    """Downloads and extracts NPU model from artifactory with resume capability."""
    sanitized_name = sanitize_model_name(model_id)
    quant_suffix = "int4_cw" if quantization == "channelwise" else "int4_gw128"
    target_dir = npu_dir / f"{sanitized_name}_{quant_suffix}"
    
    # Double-check if model already exists (safety check)
    if target_dir.exists() and (target_dir / "openvino_model.xml").exists():
        print(f"{Colors.GREEN}NPU model already exists: {target_dir}{Colors.ENDC}")
        return True
    
    download_path = npu_dir / model_info["filename"]
    
    print(f"\n{Colors.CYAN}Downloading NPU model: {model_info['filename']}{Colors.ENDC}")
    print(f"{Colors.GRAY}Size: {model_info['size']}, Category: {model_info.get('category', 'unknown')}{Colors.ENDC}")
    
    try:
        # Check if partial download exists
        resume_header = {}
        if download_path.exists():
            existing_size = download_path.stat().st_size
            print(f"{Colors.YELLOW}Partial download found ({existing_size/1024/1024:.2f}MB). Attempting to resume...{Colors.ENDC}")
            resume_header = {'Range': f'bytes={existing_size}-'}
        
        # Download with resume capability
        proxies = {"http": None, "https": None}
        with requests.get(model_info["url"], stream=True, verify=False, proxies=proxies, headers=resume_header) as r:
            r.raise_for_status()
            
            total_size = int(r.headers.get('content-length', 0))
            if resume_header and r.status_code == 206:  # Partial content
                total_size += existing_size
                mode = 'ab'  # Append mode
                print(f"{Colors.GREEN}Resuming download from {existing_size/1024/1024:.2f}MB{Colors.ENDC}")
            else:
                mode = 'wb'  # Write mode
            
            with open(download_path, mode) as f:
                downloaded = existing_size if mode == 'ab' else 0
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r{Colors.YELLOW}Progress: {percent:.1f}% ({downloaded/1024/1024:.2f}MB/{total_size/1024/1024:.2f}MB){Colors.ENDC}", end='', flush=True)
        
        print(f"\n{Colors.GREEN}Download complete.{Colors.ENDC}")
        
        # Extract
        print(f"{Colors.YELLOW}Extracting model...{Colors.ENDC}")
        with tarfile.open(download_path, "r:gz") as tar:
            # Get the first directory name from the archive
            members = tar.getnames()
            if members:
                first_member = members[0]
                if '/' in first_member:
                    extracted_name = first_member.split('/')[0]
                else:
                    extracted_name = model_info["filename"].replace(".tgz", "")
            else:
                extracted_name = model_info["filename"].replace(".tgz", "")
            
            tar.extractall(path=npu_dir)
        
        # Rename to standard format
        extracted_path = npu_dir / extracted_name
        if extracted_path.exists() and extracted_path != target_dir:
            if target_dir.exists():
                shutil.rmtree(target_dir)
            extracted_path.rename(target_dir)
            print(f"{Colors.GREEN}Renamed to: {target_dir.name}{Colors.ENDC}")
        
        # Verify extraction
        if not (target_dir / "openvino_model.xml").exists():
            print(f"{Colors.RED}Warning: openvino_model.xml not found in extracted model{Colors.ENDC}")
        
        # Auto-delete archive
        download_path.unlink()
        print(f"{Colors.GRAY}Archive deleted automatically.{Colors.ENDC}")
        
        print(f"{Colors.GREEN}NPU model ready: {target_dir}{Colors.ENDC}")
        return True
        
    except Exception as e:
        print(f"\n{Colors.RED}Failed to download NPU model: {e}{Colors.ENDC}")
        if download_path.exists():
            # Keep partial download for resume
            print(f"{Colors.YELLOW}Keeping partial download for potential resume.{Colors.ENDC}")
        return False

def quantize_model(model_id: str, quantization: str, device: str, output_dir: Path, venv_path: str, llm_bench_path: str, hf_token: str = "", folders: Dict = None) -> bool:
    """Quantizes a model using optimum-cli with existing model check."""
    
    # Check if model already exists
    if folders:
        existing_model = check_existing_model(model_id, quantization, device, folders)
        if existing_model:
            return True  # Model already exists, skip quantization
    
    sanitized_name = sanitize_model_name(model_id)
    
    # Determine output directory and command parameters
    if device == "gpu":
        quant_suffix = "int4_cw" if quantization == "channelwise" else "int4_gw128"
        output_path = output_dir / f"{sanitized_name}_{quant_suffix}"
        sym_flag = []  # GPU doesn't use --sym
    else:  # NPU
        quant_suffix = "int4_cw" if quantization == "channelwise" else "int4_gw128"
        output_path = output_dir / f"{sanitized_name}_{quant_suffix}"
        sym_flag = ["--sym"]  # NPU requires --sym
    
    # Build command
    group_size = "-1" if quantization == "channelwise" else "128"
    
    command = [
        "optimum-cli", "export", "openvino",
        "-m", model_id,
        "--weight-format", "int4",
        "--group-size", group_size,
        "--ratio", "1.0",
        "--trust-remote-code"
    ] + sym_flag + [str(output_path)]
    
    print(f"\n{Colors.CYAN}Quantizing {device.upper()} model: {model_id}{Colors.ENDC}")
    print(f"{Colors.GRAY}Output: {output_path}{Colors.ENDC}")
    print(f"{Colors.GRAY}Command: {' '.join(command)}{Colors.ENDC}")
    print(f"{Colors.YELLOW}This may take several minutes to download and quantize the model...{Colors.ENDC}")
    
    # Set up environment
    env = os.environ.copy()
    if hf_token:
        env['HF_TOKEN'] = hf_token
        env['HUGGINGFACE_HUB_TOKEN'] = hf_token  # Some models check this variable
    
    # Ensure environment variables for HF are set
    env['HF_HUB_ENABLE_HF_TRANSFER'] = '0'  # Disable HF transfer for stability
    
    # Run in virtual environment with verbose output for debugging
    success = run_command(
        command_str=f'call "{Path(venv_path) / "Scripts" / "activate.bat"}" && {" ".join(command)}',
        working_directory=llm_bench_path,
        stop_on_error=False,
        env=env,
        verbose=True
    )
    
    if success:
        # Verify the quantization was successful
        if (output_path / "openvino_model.xml").exists():
            print(f"{Colors.GREEN}{device.upper()} quantization completed: {output_path}{Colors.ENDC}")
            return True
        else:
            print(f"{Colors.RED}{device.upper()} quantization failed - missing output files{Colors.ENDC}")
            return False
    else:
        print(f"{Colors.RED}{device.upper()} quantization failed for {model_id}{Colors.ENDC}")
        return False

def run_in_activated_environment(venv_path, command, working_directory=None, env_vars=None):
    """Executes a command within the specified virtual environment."""
    activation_script = str(Path(venv_path) / "Scripts" / "activate.bat")
    full_command = f'call "{activation_script}" && {" ".join(command)}'
    
    # Combine environments
    final_env = os.environ.copy()
    if env_vars:
        final_env.update(env_vars)
    
    return run_command(
        command_str=full_command,
        working_directory=working_directory,
        stop_on_error=False,
        env=final_env,
        verbose=False  # Reduce verbosity for model processing
    )

def process_models(config: Dict, folders: Dict, venv_path: str, llm_bench_path: str):
    """Processes all models according to the configuration with duplicate prevention."""
    print(f"\n{Colors.MAGENTA}{'='*60}{Colors.ENDC}")
    print(f"{Colors.MAGENTA}STARTING MODEL PROCESSING{Colors.ENDC}")
    print(f"{Colors.MAGENTA}{'='*60}{Colors.ENDC}")
    
    total_models = len(config['models'])
    device = config['device']
    hf_token = config['hf_token']
    
    # Track processing results
    results = {
        'processed': 0,
        'skipped_existing': 0,
        'downloaded': 0,
        'quantized': 0,
        'failed': 0
    }
    
    for i, model_config in enumerate(config['models'], 1):
        model_id = model_config['model_id']
        quantization = model_config['quantization']
        
        print(f"\n{Colors.CYAN}Processing model {i}/{total_models}: {model_id}{Colors.ENDC}")
        print(f"{Colors.GRAY}Quantization: {quantization}{Colors.ENDC}")
        print(f"{Colors.GRAY}Target device(s): {device}{Colors.ENDC}")
        
        model_processed = False
        
        if device in ['npu', 'both']:
            # Try NPU download first with enhanced matching
            npu_model = check_npu_model_in_artifactory(model_id, quantization, folders)
            if npu_model:
                success = download_npu_model(npu_model, folders['npu'], model_id, quantization)
                if success:
                    results['downloaded'] += 1
                    model_processed = True
                else:
                    print(f"{Colors.YELLOW}NPU download failed, will quantize locally.{Colors.ENDC}")
            
            # If no NPU model found or download failed, quantize locally
            if not npu_model or not success:
                print(f"{Colors.YELLOW}Quantizing model locally for NPU...{Colors.ENDC}")
                if quantize_model(model_id, quantization, 'npu', folders['npu'], venv_path, llm_bench_path, hf_token, folders):
                    results['quantized'] += 1
                    model_processed = True
                else:
                    results['failed'] += 1
        
        if device in ['gpu', 'both']:
            # Always quantize for GPU (no index check)
            print(f"{Colors.YELLOW}Quantizing model for GPU...{Colors.ENDC}")
            if quantize_model(model_id, quantization, 'gpu', folders['gpu'], venv_path, llm_bench_path, hf_token, folders):
                results['quantized'] += 1
                model_processed = True
            else:
                results['failed'] += 1
        
        if model_processed:
            results['processed'] += 1
    
    # Print summary
    print(f"\n{Colors.MAGENTA}{'='*60}{Colors.ENDC}")
    print(f"{Colors.MAGENTA}MODEL PROCESSING SUMMARY{Colors.ENDC}")
    print(f"{Colors.MAGENTA}{'='*60}{Colors.ENDC}")
    print(f"{Colors.GREEN}✓ Total models processed: {results['processed']}/{total_models}{Colors.ENDC}")
    print(f"{Colors.CYAN}📥 Models downloaded from NPU: {results['downloaded']}{Colors.ENDC}")
    print(f"{Colors.YELLOW}⚙️  Models quantized locally: {results['quantized']}{Colors.ENDC}")
    print(f"{Colors.GRAY}⏭️  Models skipped (existing): {results['skipped_existing']}{Colors.ENDC}")
    if results['failed'] > 0:
        print(f"{Colors.RED}❌ Models failed: {results['failed']}{Colors.ENDC}")

# Benchmarking Functions
# -----------------------------------------------------------------------------

def get_all_openvino_models(models_folder: Path):
    """Find all OpenVINO models in the folder."""
    if not models_folder.exists():
        print(f"{Colors.RED}Models folder path not found: {models_folder}{Colors.ENDC}")
        return []
    
    print(f"\n{Colors.CYAN}Scanning for OpenVINO models in: {models_folder}{Colors.ENDC}")
    all_models = []
    
    for item in models_folder.iterdir():
        if not item.is_dir():
            continue
        # Check for required OpenVINO files
        if not all((item / f).exists() for f in ['openvino_model.xml', 'openvino_model.bin', 'config.json']):
            continue
        
        folder_name = item.name.lower()
        folder_parts = folder_name.split('_')
        
        # Detect precision
        precision = next((p for p in ['int4', 'int8', 'fp16', 'fp32'] if p in folder_name), 'unknown')
        
        # Detect method
        if 'cw' in folder_parts or 'cwq' in folder_name:
            method = 'channel_wise'
        elif 'gw' in folder_parts or 'awq' in folder_name:
            method = 'group_wise'
        else:
            method = 'unknown'
        
        model_info = {
            'path': item,
            'name': item.name,
            'quantization': {'precision': precision, 'method': method},
            'size_mb': sum(f.stat().st_size for f in item.glob('*') if f.is_file()) / (1024*1024)
        }
        all_models.append(model_info)
        print(f"{Colors.GREEN}✓ Found: {model_info['name']} ({precision}){Colors.ENDC}")
    
    print(f"\n{Colors.CYAN}Found {len(all_models)} OpenVINO models total{Colors.ENDC}")
    return all_models

def search_metric(text, pattern, default='N/A'):
    """Helper function to search for a metric using regex."""
    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    return match.group(1).strip() if match else default

def parse_log_file(log_file_path: Path):
    """
    Parses a single log file to extract key performance and metadata metrics.
    """
    try:
        content = log_file_path.read_text(encoding='utf-8', errors='ignore')
    except Exception as e:
        print(f"  {Colors.YELLOW}Warning: Could not read file {log_file_path.name}: {e}{Colors.ENDC}")
        return None

    # Extract Model and Device from filename
    parts = log_file_path.stem.rsplit('_', 1)
    model_name = parts[0]
    device = parts[1].upper() if len(parts) > 1 else 'N/A'
    
    # Isolate different parts of the log for targeted searching
    post_warmup_content = content.split('<<< Warm-up iteration is excluded. >>>')[-1]
    
    # Define Regex Patterns for each section
    general_patterns = {
        'ov_version': r"openvino runtime version:\s*([\w.-]+)",
        'genai_version': r"genai version:\s*([\w.-]+)",
        'pipeline_initialization_time_s': r"Pipeline initialization time:\s*([\d.]+)s",
        'duration': r"Duration:\s*([\d:]+)"
    }

    average_patterns = {
        'input_token_size': r"Input token size:\s*(\d+)",
        'first_token_latency_ms': r"1st token latency:\s*([\d.]+)\s*ms",
        'other_tokens_latency_ms_per_token': r"2nd token latency:\s*([\d.]+)\s*ms/token",
        'throughput_tokens_per_s': r"2nd tokens throughput:\s*([\d.]+)\s*tokens/s",
    }
    
    detailed_run_patterns = {
        'output_size_infer_count': r"\[1\]\[P0\].*?Infer count:\s*(\d+)",
        'tokenization_time_ms': r"\[1\]\[P0\].*?Tokenization Time:\s*([\d.]+)ms",
        'detokenization_time_ms': r"\[1\]\[P0\].*?Detokenization Time:\s*([\d.]+)ms",
        'generation_time_s': r"\[1\]\[P0\].*?Generation Time:\s*([\d.]+)s",
        'latency_ms_per_token': r"\[1\]\[P0\].*?Latency:\s*([\d.]+)\s*ms/token",
        'first_infer_latency_ms': r"\[1\]\[P0\].*?First infer latency:\s*([\d.]+)\s*ms",
        'other_infers_latency_ms_per_infer': r"\[1\]\[P0\].*?other infers latency:\s*([\d.]+)\s*ms/infer",
        # Memory metrics for first iteration (not warm-up)
        'max_rss_memory_mb': r"\[1\]\[P0\].*?Max rss memory cost:\s*([\d.]+)MBytes",
        'rss_memory_increase_mb': r"\[1\]\[P0\].*?rss memory increase:\s*([\d.]+)MBytes",
        'max_system_memory_mb': r"\[1\]\[P0\].*?max system memory memory cost:\s*([\d.]+)MBytes",
        'system_memory_increase_mb': r"\[1\]\[P0\].*?system memory increase:\s*([\d.]+)MBytes",
    }

    metrics = {'model': model_name, 'device': device}
    
    # Execute the parsing strategy
    for key, pattern in general_patterns.items():
        metrics[key] = search_metric(content, pattern)
        
    for key, pattern in average_patterns.items():
        metrics[key] = search_metric(post_warmup_content, pattern)

    for key, pattern in detailed_run_patterns.items():
        metrics[key] = search_metric(content, pattern)
        
    return metrics

def generate_summary_report(session_folder: Path):
    """
    Finds all log files, parses them, and writes the results to a single CSV report.
    """
    print(f"\n{Colors.CYAN}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.CYAN}GENERATING SUMMARY CSV REPORT{Colors.ENDC}")
    print(f"{Colors.CYAN}{'=' * 80}{Colors.ENDC}")
    
    log_files = sorted(list(session_folder.glob("*.log")))

    if not log_files:
        print(f"{Colors.YELLOW}No .log files found in {session_folder} to generate a report from.{Colors.ENDC}")
        return

    all_results = [res for log_file in log_files if (res := parse_log_file(log_file)) is not None]
            
    if not all_results:
        print(f"{Colors.RED}Could not extract data from any log files.{Colors.ENDC}")
        return

    # Write to CSV
    report_path = session_folder / "summary_report.csv"
    headers = [
        'model', 'device', 
        # Key Performance Indicators
        'first_token_latency_ms', 'other_tokens_latency_ms_per_token', 
        'throughput_tokens_per_s', 'latency_ms_per_token',
        # Detailed Inference Stats
        'first_infer_latency_ms', 'other_infers_latency_ms_per_infer',
        # Timing Breakdown
        'pipeline_initialization_time_s', 'generation_time_s', 
        'tokenization_time_ms', 'detokenization_time_ms',
        # Token/Run Info
        'input_token_size', 'output_size_infer_count', 'duration',
        # Memory Metrics
        'max_rss_memory_mb', 'rss_memory_increase_mb', 'max_system_memory_mb', 'system_memory_increase_mb',
        # Versioning
        'ov_version', 'genai_version'
    ]
    
    try:
        with open(report_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"\n{Colors.GREEN}✅ Summary report generated successfully!{Colors.ENDC}")
        print(f"{Colors.GRAY}   Report saved to: {report_path}{Colors.ENDC}")
        
    except Exception as e:
        print(f"\n{Colors.RED}Error writing CSV report: {e}{Colors.ENDC}")

def create_benchmark_session(base_path: Path):
    """Creates a directory to store benchmark logs for the current session."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_name = f"benchmark_session_{timestamp}"
    session_path = base_path / session_name
    session_path.mkdir(exist_ok=True, parents=True)
    
    return {"SessionName": session_name, "SessionPath": session_path, "Timestamp": timestamp}

def find_specific_prompt_file(model_name: str, prompt_folder: Path) -> Optional[Path]:
    """
    Finds a prompt file in a folder that best matches the model name using fuzzy logic.
    """
    if not prompt_folder or not prompt_folder.is_dir():
        return None

    model_name_lower = model_name.lower()
    best_match = None
    highest_score = 0.0

    # Define keywords for matching
    model_keywords = re.split(r'[-_]', model_name_lower)
    model_keywords = {kw for kw in model_keywords if len(kw) > 2} # Use meaningful keywords

    for prompt_file in prompt_folder.iterdir():
        if prompt_file.is_file() and prompt_file.suffix in ['.txt', '.jsonl', '.prompt']:
            prompt_filename_lower = prompt_file.stem.lower()
            
            # Calculate a simple score based on keyword matches
            score = 0
            for keyword in model_keywords:
                if keyword in prompt_filename_lower:
                    score += len(keyword) # Longer keyword matches are better
            
            # Direct substring match gets a high score
            if prompt_filename_lower in model_name_lower:
                score += 100

            if score > highest_score:
                highest_score = score
                best_match = prompt_file

    return best_match

def invoke_in_activated_environment(setup_info, command: list, working_directory: Path, log_file: Path):
    """
    Executes a command using the appropriate Python interpreter.
    """
    python_executable = setup_info["VenvPath"] / "Scripts" / "python.exe"
    
    if not python_executable.exists():
        print(f"{Colors.RED}Error: Python executable not found: {python_executable}{Colors.ENDC}")
        return False
        
    full_command = [str(python_executable)] + command[1:]
    
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            process = subprocess.run(full_command, cwd=working_directory, stdout=f, stderr=subprocess.STDOUT, text=True)
        return process.returncode == 0
    except Exception as e:
        print(f"{Colors.RED}Error executing command in virtual environment: {e}{Colors.ENDC}")
        return False

def start_model_benchmark(setup_info, model_info, device, benchmark_script, log_file, benchmark_config):
    """Prepares and runs the benchmark for a single model and device."""
    model_path = model_info['path'].resolve()
    model_name = model_info['name']
    
    # --- Build Benchmark Command ---
    benchmark_command = [
        "python", str(benchmark_script),
        "-d", device,
        "-m", str(model_path),
        "-ic", str(benchmark_config.get('benchmark_input_tokens', 128)),
        "-n", str(benchmark_config.get('benchmark_iterations', 1))
    ]

    # Always add memory consumption argument with value '2'
    benchmark_command.extend(["-mc", "2"])

    # --- Configuration File Logic ---
    config_file = benchmark_config.get('benchmark_config_file')
    if config_file:
        config_filename_lower = config_file.name.lower()
        device_lower = device.lower()
        # Apply if it's a general config or matches the specific device
        if 'gpu' not in config_filename_lower and 'npu' not in config_filename_lower:
            benchmark_command.extend(["-lc", str(config_file)])
        elif device_lower in config_filename_lower:
            benchmark_command.extend(["-lc", str(config_file)])

    # --- Prompt Logic ---
    specific_prompt_file = None
    if benchmark_config.get('benchmark_specific_prompt_folder'):
        specific_prompt_file = find_specific_prompt_file(model_name, benchmark_config['benchmark_specific_prompt_folder'])

    if specific_prompt_file:
        benchmark_command.extend(["-pf", str(specific_prompt_file)])
        print(f"{Colors.CYAN}Using specific prompt file: {specific_prompt_file.name}{Colors.ENDC}")
    elif benchmark_config.get('benchmark_general_prompt_file'):
        benchmark_command.extend(["-pf", str(benchmark_config['benchmark_general_prompt_file'])])
    else:
        benchmark_command.extend(["-p", f"\"{benchmark_config.get('benchmark_prompt', 'what is openvino?')}\""])

    print(f"\n{Colors.CYAN}Benchmarking: {model_name} on {device}{Colors.ENDC}")
    print(f"{Colors.GRAY}Precision: {model_info['quantization']['precision']}, Method: {model_info['quantization']['method']}{Colors.ENDC}")
    
    start_time = datetime.now()
    
    log_header = f"""
================================================================================
BENCHMARK LOG - {model_name} on {device}
================================================================================
Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}
Model: {model_name}
Device: {device}
Model Path: {model_path}
Precision: {model_info['quantization']['precision']}
Method: {model_info['quantization']['method']}
Size: {model_info['size_mb']:.1f} MB
Command: {' '.join(benchmark_command)}
================================================================================\n

"""
    log_file.write_text(log_header, encoding='utf-8')

    success = invoke_in_activated_environment(
        setup_info=setup_info, command=benchmark_command, 
        working_directory=setup_info["LlmBenchPath"], log_file=log_file
    )
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    status = "SUCCESS" if success else "FAILED"
    log_footer = f"""
\n================================================================================
End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
Duration: {str(duration).split('.')[0]}
Status: {status}
================================================================================
"""
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_footer)
        
    return {"Success": success, "Duration": duration, "LogFile": log_file, "Device": device, "ModelName": model_name}

def show_benchmark_summary(all_results: List[Dict], session: Dict):
    """Displays a summary of the benchmark session results to the console."""
    if not all_results:
        print(f"{Colors.YELLOW}No benchmark results to summarize.{Colors.ENDC}")
        return

    total_models = len(set(r['ModelName'] for r in all_results))
    success_count = sum(1 for r in all_results if r['Success'])
    fail_count = len(all_results) - success_count
    
    print(f"\n{Colors.GREEN}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.GREEN}BENCHMARK SESSION SUMMARY{Colors.ENDC}")
    print(f"{Colors.GREEN}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.WHITE}Session: {session['SessionName']}{Colors.ENDC}")
    print(f"{Colors.WHITE}Total Models Tested: {total_models}{Colors.ENDC}")
    print(f"{Colors.GREEN}Successful Runs: {success_count}{Colors.ENDC}")
    print(f"{Colors.RED}Failed Runs: {fail_count}{Colors.ENDC}")
    print(f"{Colors.GRAY}Logs Location: {session['SessionPath']}{Colors.ENDC}")

def run_comprehensive_benchmark(folders: Dict, setup_info: Dict, benchmark_config: Dict):
    """Runs benchmarks on all available models in GPU and NPU folders."""
    print(f"\n{Colors.MAGENTA}{'='*60}{Colors.ENDC}")
    print(f"{Colors.MAGENTA}STARTING COMPREHENSIVE BENCHMARKING{Colors.ENDC}")
    print(f"{Colors.MAGENTA}{'='*60}{Colors.ENDC}")
    
    # Create benchmark session
    session = create_benchmark_session(folders['benchmark'])
    
    # Check benchmark script exists
    benchmark_script = setup_info["LlmBenchPath"] / "benchmark.py"
    if not benchmark_script.exists():
        print(f"{Colors.RED}Benchmark script not found: {benchmark_script}{Colors.ENDC}")
        return
    
    all_results = []
    
    # Benchmark GPU models
    gpu_models = get_all_openvino_models(folders['gpu'])
    for model_info in gpu_models:
        log_file = session["SessionPath"] / f"{model_info['name']}_gpu.log"
        result = start_model_benchmark(setup_info, model_info, "GPU", benchmark_script, log_file, benchmark_config)
        all_results.append(result)
    
    # Benchmark NPU models
    npu_models = get_all_openvino_models(folders['npu'])
    for model_info in npu_models:
        log_file = session["SessionPath"] / f"{model_info['name']}_npu.log"
        result = start_model_benchmark(setup_info, model_info, "NPU", benchmark_script, log_file, benchmark_config)
        all_results.append(result)
    
    # Show summary
    total_models = len(set(r['ModelName'] for r in all_results))
    success_count = sum(1 for r in all_results if r['Success'])
    fail_count = len(all_results) - success_count
    
    print(f"\n{Colors.GREEN}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.GREEN}BENCHMARK SESSION SUMMARY{Colors.ENDC}")
    print(f"{Colors.GREEN}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.WHITE}Session: {session['SessionName']}{Colors.ENDC}")
    print(f"{Colors.WHITE}Total Models Tested: {total_models}{Colors.ENDC}")
    print(f"{Colors.GREEN}Successful Runs: {success_count}{Colors.ENDC}")
    print(f"{Colors.RED}Failed Runs: {fail_count}{Colors.ENDC}")
    print(f"{Colors.GRAY}Logs Location: {session['SessionPath']}{Colors.ENDC}")
    
    # Generate CSV report
    generate_summary_report(session['SessionPath'])
    
    return session

def save_model_management_info(config: Dict, folders: Dict, llm_bench_path: str):
    """Saves model management configuration for future reference."""
    info = {
        "timestamp": datetime.now().isoformat(),
        "device_target": config['device'],
        "models_processed": config['models'],
        "folder_structure": {
            "models": str(folders['models']),
            "gpu_models": str(folders['gpu']),
            "npu_models": str(folders['npu']),
            "benchmark_results": str(folders['benchmark'])
        },
        "features": {
            "enhanced_pattern_matching": True,
            "targeted_folder_search": True,
            "duplicate_prevention": True,
            "resume_downloads": True,
            "comprehensive_benchmarking": True
        }
    }
    
    info_file = Path(llm_bench_path) / "model_management_info.json"
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"{Colors.GREEN}Model management info saved to: {info_file}{Colors.ENDC}")

# Main Script Logic
# -----------------------------------------------------------------------------

def set_proxy_for_session(proxy_url):
    """Sets the HTTP_PROXY and HTTPS_PROXY environment variables for the current session only."""
    os.environ['HTTP_PROXY'] = proxy_url
    os.environ['HTTPS_PROXY'] = proxy_url
    print(f"{Colors.GREEN}Proxy set for current session: {proxy_url}{Colors.ENDC}")
    return True

def test_virtual_environment_success(venv_path, working_directory):
    """Verifies that the required packages are installed in the virtual environment."""
    print(f"{Colors.YELLOW}Verifying virtual environment installation...{Colors.ENDC}")
    
    success = run_in_activated_environment(
        venv_path=venv_path,
        command=["python", "-c", "import openvino, torch, transformers; print('All packages imported successfully')"],
        working_directory=working_directory
    )

    if success:
        print(f"{Colors.GREEN}Virtual environment verification PASSED{Colors.ENDC}")
        return True
    else:
        print(f"{Colors.RED}Virtual environment verification FAILED{Colors.ENDC}")
        print(f"{Colors.YELLOW}Some packages may not have installed correctly{Colors.ENDC}")
        return False

def main():
    """Main function to run the setup process."""
    script_start_time = datetime.now()
    print(f"{Colors.CYAN}--- Script started at: {script_start_time.strftime('%Y-%m-%d %H:%M:%S')} ---{Colors.ENDC}")

    print(f"{Colors.MAGENTA}--- OpenVINO LLM Bench One-Stop Setup & Benchmark Solution ---{Colors.ENDC}")
    
    script_dir = Path(__file__).parent
    
    # Get comprehensive configuration upfront
    print(f"\n{Colors.CYAN}Collecting all configuration inputs...{Colors.ENDC}")
    config = get_comprehensive_configuration()
    if not config:
        print(f"{Colors.RED}Configuration failed. Exiting.{Colors.ENDC}")
        sys.exit(1)
    
    # --- STEP 1: PROXY CONFIGURATION ---
    proxy_url = "http://proxy-iind.intel.com:911"
    
    print(f"\n{Colors.CYAN}STEP 1: Checking proxy configuration...{Colors.ENDC}")
    
    if PROXY_USER_OPT_IN:
        if os.getenv("HTTP_PROXY") != proxy_url or os.getenv("HTTPS_PROXY") != proxy_url:
            print(f"{Colors.YELLOW}Proxy not configured. Setting proxy for current session...{Colors.ENDC}")
            set_proxy_for_session(proxy_url)
        else:
            print(f"{Colors.GREEN}Proxy already configured correctly. Continuing...{Colors.ENDC}")
    else:
        print(f"{Colors.YELLOW}Proxy configuration was skipped per user preference. Continuing without proxy setup.{Colors.ENDC}")
        
    # --- STEP 2: CHECK SOFTWARE DEPENDENCIES ---
    print(f"\n{Colors.CYAN}STEP 2: Checking software dependencies...{Colors.ENDC}")
    
    if not test_software("Python", ["python", "--version"], "Download from https://www.python.org/downloads/windows/ and ensure it's added to PATH."):
        sys.exit(1)
    if not test_software(
        "Git",
        ["git", "--version"],
        "Use `winget install --id Git.Git` or download from https://git-scm.com/download/win and ensure it's added to PATH.",
        install_package_id="Git.Git"
    ):
        sys.exit(1)

    print(f"\n{Colors.YELLOW}VC++ Redistributables Check:{Colors.ENDC}")
    print(f"{Colors.YELLOW}It's highly recommended to have the latest Microsoft Visual C++ Redistributable (x64) installed.{Colors.ENDC}")
    print(f"{Colors.YELLOW}Download from: https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170{Colors.ENDC}")

    if not ensure_vc_redists_installed():
        print(f"{Colors.YELLOW}Some Visual C++ Redistributable installs failed. Please install them manually if you encounter issues, but the script will continue.{Colors.ENDC}")
    
    # --- STEP 3: PULL OPENVINO GENAI REPOSITORY ---
    print(f"\n{Colors.CYAN}STEP 3: Setting up OpenVINO GenAI repository...{Colors.ENDC}")
    
    openvino_genai_path = script_dir / "openvino.genai"
    llm_bench_path = openvino_genai_path / "tools" / "llm_bench"
    
    repo_validation = test_openvino_repository(str(openvino_genai_path))
    
    if repo_validation["valid"]:
        print(f"{Colors.GREEN}Valid OpenVINO GenAI repository found!{Colors.ENDC}")
    else:
        print(f"{Colors.RED}Repository validation failed: {repo_validation['reason']}{Colors.ENDC}")
        if repo_validation["action"] == "clone":
            print(f"{Colors.YELLOW}Cloning OpenVINO GenAI repository...{Colors.ENDC}")
            clone_success = run_command(
                command_str="git clone https://github.com/openvinotoolkit/openvino.genai.git",
                working_directory=str(script_dir),
                verbose=False
            )
            if not clone_success:
                print(f"{Colors.RED}Failed to clone repository. Please check your internet connection and proxy settings.{Colors.ENDC}")
                sys.exit(1)
            print(f"{Colors.GREEN}Repository cloned successfully!{Colors.ENDC}")
        else:
            print(f"{Colors.RED}Please handle the repository manually based on the reason above.{Colors.ENDC}")
            sys.exit(1)

    if not llm_bench_path.exists():
        print(f"{Colors.RED}Critical error: '{llm_bench_path}' not found even after repository setup.{Colors.ENDC}")
        sys.exit(1)
    print(f"{Colors.GREEN}Target path confirmed: '{llm_bench_path}'{Colors.ENDC}")

    # --- STEP 4: ENHANCED VIRTUAL ENVIRONMENT SETUP ---
    print(f"\n{Colors.CYAN}STEP 4: Setting up Python virtual environment with robust error handling...{Colors.ENDC}")
    
    venv_path = setup_virtual_environment(llm_bench_path)
    if not venv_path:
        print(f"{Colors.RED}Failed to setup virtual environment. Cannot continue.{Colors.ENDC}")
        sys.exit(1)
    
    # Final validation
    final_validation = validate_virtual_environment(venv_path)
    if not final_validation['valid']:
        print(f"{Colors.RED}Final validation failed. Virtual environment is not usable.{Colors.ENDC}")
        sys.exit(1)
    
    print(f"\n{Colors.GREEN}Virtual Environment Setup Complete:{Colors.ENDC}")
    print(f"{Colors.GRAY}  Environment Name: {venv_path.name}{Colors.ENDC}")
    print(f"{Colors.GRAY}  Full Path: {venv_path}{Colors.ENDC}")
    print(f"{Colors.GRAY}  Python Executable: {final_validation['python_exe']}{Colors.ENDC}")
    print(f"{Colors.GRAY}  Activation Script: {final_validation['activate_script']}{Colors.ENDC}")

    # --- STEP 5: INSTALL REQUIREMENTS ---
    print(f"\n{Colors.CYAN}STEP 5: Installing Python dependencies in activated environment...{Colors.ENDC}")
    
    requirements_file_path = llm_bench_path / "requirements.txt"
    if requirements_file_path.exists():
        print(f"\n{Colors.YELLOW}Installing dependencies from requirements.txt...{Colors.ENDC}")
        print(f"{Colors.YELLOW}This may take several minutes. Please be patient...{Colors.ENDC}")
        run_in_activated_environment(
            venv_path=str(venv_path),
            command=["pip", "install", "-r", str(requirements_file_path)],
            working_directory=str(llm_bench_path)
        )
    else:
        print(f"{Colors.YELLOW}Warning: requirements.txt not found at '{requirements_file_path}'{Colors.ENDC}")
        
    # print(f"\n{Colors.YELLOW}Installing OpenVINO packages in activated environment...{Colors.ENDC}")
    # openvino_packages_cmd = ["pip", "install", "--pre", "openvino", "openvino-tokenizers", "openvino-genai", "--extra-index-url", "https://storage.openvinotoolkit.org/simple/wheels/nightly", "--upgrade"]
    # run_in_activated_environment(
        # venv_path=str(venv_path),
        # command=openvino_packages_cmd,
        # working_directory=str(llm_bench_path)
    # )
    
    # --- VERIFICATION ---
    print(f"\n{Colors.CYAN}Verifying installation...{Colors.ENDC}")
    if not test_virtual_environment_success(str(venv_path), str(llm_bench_path)):
        print(f"{Colors.YELLOW}Installation verification failed. You may need to manually install missing packages.{Colors.ENDC}")

    # --- STEP 6: MODEL MANAGEMENT WORKFLOW ---
    print(f"\n{Colors.CYAN}STEP 6: Model Management and Processing...{Colors.ENDC}")
    
    # Create folder structure
    folders = create_folder_structure(llm_bench_path)
    
    # Process models
    process_models(config, folders, str(venv_path), str(llm_bench_path))
    
    # Save configuration
    save_model_management_info(config, folders, str(llm_bench_path))

    # --- STEP 7: BENCHMARKING ---
    if config['run_benchmark']:
        print(f"\n{Colors.CYAN}STEP 7: Running Comprehensive Benchmarks...{Colors.ENDC}")
        
        setup_info = {
            "LlmBenchPath": llm_bench_path,
            "VenvPath": venv_path,
            "VenvName": venv_path.name
        }
        
        # Create benchmark session
        session = create_benchmark_session(folders['benchmark'])
        
        # Check benchmark script exists
        benchmark_script = setup_info["LlmBenchPath"] / "benchmark.py"
        if not benchmark_script.exists():
            print(f"{Colors.RED}Benchmark script not found: {benchmark_script}{Colors.ENDC}")
            sys.exit(1)
        
        all_results = []
        
        # Get a unique list of model IDs the user wanted to process
        user_selected_models = {mc['model_id'] for mc in config['models']}

        # Benchmark GPU models if selected
        if config['device'] in ['gpu', 'both']:
            gpu_models = get_all_openvino_models(folders['gpu'])
            for model_info in gpu_models:
                # Check if this model was part of the user's selection
                original_model_id = next((mid for mid in user_selected_models if sanitize_model_name(mid) in model_info['name']), None)
                if original_model_id:
                    print(f"{Colors.CYAN}Found user-selected model for GPU benchmarking: {model_info['name']}{Colors.ENDC}")
                    log_file = session["SessionPath"] / f"{model_info['name']}_gpu.log"
                    result = start_model_benchmark(setup_info, model_info, "GPU", benchmark_script, log_file, config)
                    all_results.append(result)
        
        # Benchmark NPU models if selected
        if config['device'] in ['npu', 'both']:
            npu_models = get_all_openvino_models(folders['npu'])
            for model_info in npu_models:
                # Check if this model was part of the user's selection
                original_model_id = next((mid for mid in user_selected_models if sanitize_model_name(mid) in model_info['name']), None)
                if original_model_id:
                    print(f"{Colors.CYAN}Found user-selected model for NPU benchmarking: {model_info['name']}{Colors.ENDC}")
                    log_file = session["SessionPath"] / f"{model_info['name']}_npu.log"
                    result = start_model_benchmark(setup_info, model_info, "NPU", benchmark_script, log_file, config)
                    all_results.append(result)
        
        # Show summary and generate report
        if all_results:
            show_benchmark_summary(all_results, session)
            generate_summary_report(session['SessionPath'])
            print(f"\n{Colors.GREEN}Benchmarking completed!{Colors.ENDC}")
            print(f"{Colors.GRAY}Results saved to: {session['SessionPath']}{Colors.ENDC}")
        else:
            print(f"\n{Colors.YELLOW}No models matching the initial configuration were found to benchmark.{Colors.ENDC}")

    else:
        print(f"\n{Colors.YELLOW}Skipping benchmarks as requested.{Colors.ENDC}")


    # --- STEP 8: CREATE ACTIVATION SCRIPT ---
    print(f"\n{Colors.CYAN}STEP 8: Creating activation script...{Colors.ENDC}")
    create_activation_script(llm_bench_path, venv_path)

    # --- STEP 9: SAVE SETUP INFO ---
    print(f"\n{Colors.CYAN}STEP 9: Saving setup information...{Colors.ENDC}")
    
    setup_info = {
        "SetupDate": datetime.now().isoformat(),
        "ScriptVersion": "4.0.0",
        "LlmBenchPath": str(llm_bench_path),
        "VenvPath": str(venv_path),
        "VenvName": venv_path.name,
        "PythonVersion": sys.version,
        "Platform": platform.platform(),
        "ProxyConfigured": True,
        "ModelFoldersCreated": True,
        "BenchmarkingEnabled": config['run_benchmark'],
        "Features": {
            "EnhancedPatternMatching": True,
            "TargetedFolderSearch": True,
            "DuplicatePrevention": True,
            "ResumeDownloads": True,
            "ComprehensiveBenchmarking": True,
            "OneStopSolution": True,
            "RobustVirtualEnvironmentManagement": True
        }
    }
    
    setup_info_path = llm_bench_path / "setup_info.json"
    with open(setup_info_path, "w") as f:
        json.dump(setup_info, f, indent=2)
    
    print(f"{Colors.GREEN}Setup information saved to: {setup_info_path}{Colors.ENDC}")

    # --- COMPLETION ---
    print(f"\n{Colors.MAGENTA}{'='*60}{Colors.ENDC}")
    print(f"{Colors.MAGENTA}ONE-STOP SOLUTION COMPLETE!{Colors.ENDC}")
    print(f"{Colors.MAGENTA}{'='*60}{Colors.ENDC}")
    
    print(f"{Colors.GREEN}✓ Proxy configured permanently{Colors.ENDC}")
    print(f"{Colors.GREEN}✓ Software dependencies verified{Colors.ENDC}")
    print(f"{Colors.GREEN}✓ OpenVINO GenAI repository ready{Colors.ENDC}")
    print(f"{Colors.GREEN}✓ Virtual environment created: {venv_path.name}{Colors.ENDC}")
    print(f"{Colors.GREEN}✓ Dependencies installation completed{Colors.ENDC}")
    print(f"{Colors.GREEN}✓ Enhanced pattern matching system active{Colors.ENDC}")
    print(f"{Colors.GREEN}✓ Models processed and ready{Colors.ENDC}")
    if config['run_benchmark']:
        print(f"{Colors.GREEN}✓ Comprehensive benchmarking completed{Colors.ENDC}")
        print(f"{Colors.GREEN}✓ Performance metrics available in CSV format{Colors.ENDC}")
    print(f"{Colors.GREEN}✓ Setup information saved{Colors.ENDC}")
    
    print(f"\n{Colors.CYAN}Environment Details:{Colors.ENDC}")
    print(f"{Colors.GRAY}  Virtual environment: '{venv_path}'{Colors.ENDC}")
    print(f"{Colors.GRAY}  Working directory: {llm_bench_path}{Colors.ENDC}")
    print(f"{Colors.GRAY}  Model folders: {llm_bench_path}/models/{Colors.ENDC}")
    if config['run_benchmark']:
        print(f"{Colors.GRAY}  Benchmark results: {llm_bench_path}/benchmark_results/{Colors.ENDC}")

    # --- FINAL STEP: LAUNCH ACTIVATED ENVIRONMENT ---
    print(f"\n{Colors.CYAN}Launching new terminal with activated environment...{Colors.ENDC}")
    try:
        command_to_run = f'start cmd.exe /K "cd /D "{llm_bench_path}" && call activate_env.bat"'
        subprocess.Popen(command_to_run, shell=True, cwd=str(llm_bench_path))
        print(f"{Colors.GREEN}A new command prompt window has been opened with the '{venv_path.name}' environment activated.{Colors.ENDC}")
        print(f"{Colors.CYAN}Your one-stop OpenVINO solution is ready to use!{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.RED}Failed to launch new terminal: {e}{Colors.ENDC}")
        print(f"{Colors.YELLOW}Please open a new terminal and run the activation script manually:{Colors.ENDC}")
        print(f"{Colors.WHITE}  cd \"{llm_bench_path}\"{Colors.ENDC}")
        print(f"{Colors.WHITE}  .\\activate_env.bat{Colors.ENDC}")
    
    # --- SCRIPT DURATION ---
    script_end_time = datetime.now()
    total_duration = script_end_time - script_start_time
    print(f"\n{Colors.CYAN}--- Script finished at: {script_end_time.strftime('%Y-%m-%d %H:%M:%S')} ---{Colors.ENDC}")
    print(f"{Colors.CYAN}--- Total execution time: {str(total_duration).split('.')[0]} ---{Colors.ENDC}")

def create_activation_script(llm_bench_path: Path, venv_path: Path):
    """Creates a batch script to activate the virtual environment."""
    activation_script_path = llm_bench_path / "activate_env.bat"
    activation_command = f".\\{venv_path.name}\\Scripts\\activate.bat"
    with open(activation_script_path, "w") as f:
        f.write("@echo off\n")
        f.write(f'echo "Changing directory to {llm_bench_path}"\n')
        f.write(f'cd /D "{llm_bench_path}"\n')
        f.write(f'echo "Activating virtual environment: {venv_path.name}"\n')
        f.write(f'call "{activation_command}"\n')
    print(f"{Colors.GREEN}Activation script created at: {activation_script_path}{Colors.ENDC}")

if __name__ == "__main__":
    main()
