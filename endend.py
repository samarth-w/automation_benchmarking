import os
import sys
import subprocess
import importlib.util

# --- ENSURE PROXY IS SET UP EARLY, BEFORE ANY NETWORK OPERATIONS ---
PROXY_URL = "http://proxy-iind.intel.com:911"
if os.environ.get("HTTP_PROXY") != PROXY_URL or os.environ.get("HTTPS_PROXY") != PROXY_URL:
    os.environ["HTTP_PROXY"] = PROXY_URL
    os.environ["HTTPS_PROXY"] = PROXY_URL






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

# Enhanced Virtual Environment Management Functions
# -----------------------------------------------------------------------------

def validate_virtual_environment(venv_path: Path) -> Dict:
    """
    Comprehensive validation of a virtual environment.
    Returns detailed information about the environment's state.
    """
    validation_result = {
        'valid': False,
        'path': venv_path,
        'issues': [],
        'python_exe': None,
        'activate_script': None,
        'can_repair': False
    }
    
    # Check if directory exists
    if not venv_path.exists():
        validation_result['issues'].append("Directory does not exist")
        validation_result['can_repair'] = True
        return validation_result
    
    # Check if it's a directory
    if not venv_path.is_dir():
        validation_result['issues'].append("Path exists but is not a directory")
        return validation_result
    
    # Check for Scripts directory
    scripts_dir = venv_path / "Scripts"
    if not scripts_dir.exists():
        validation_result['issues'].append("Scripts directory missing")
        validation_result['can_repair'] = True
        return validation_result
    
    # Check for Python executable
    python_exe = scripts_dir / "python.exe"
    if not python_exe.exists():
        validation_result['issues'].append("Python executable missing")
        validation_result['can_repair'] = True
        return validation_result
    
    validation_result['python_exe'] = python_exe
    
    # Check for activation script
    activate_script = scripts_dir / "activate.bat"
    if not activate_script.exists():
        validation_result['issues'].append("Activation script missing")
        validation_result['can_repair'] = True
        return validation_result
    
    validation_result['activate_script'] = activate_script
    
    # Test if Python executable works
    try:
        result = subprocess.run(
            [str(python_exe), "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            validation_result['issues'].append("Python executable is not functional")
            validation_result['can_repair'] = True
            return validation_result
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        validation_result['issues'].append(f"Python executable test failed: {e}")
        validation_result['can_repair'] = True
        return validation_result
    
    # Check for pyvenv.cfg
    pyvenv_cfg = venv_path / "pyvenv.cfg"
    if not pyvenv_cfg.exists():
        validation_result['issues'].append("pyvenv.cfg missing (may indicate corrupted environment)")
        validation_result['can_repair'] = True
        return validation_result
    
    # If we get here, the environment appears valid
    validation_result['valid'] = True
    return validation_result

def find_existing_virtual_environments(llm_bench_path: Path) -> List[Dict]:
    """
    Search for existing virtual environments in the LLM bench directory.
    Returns a list of environment information dictionaries.
    """
    print(f"{Colors.CYAN}Searching for existing virtual environments in: {llm_bench_path}{Colors.ENDC}")
    
    environments = []
    
    try:
        for item in llm_bench_path.iterdir():
            if not item.is_dir():
                continue
            
            # Skip obvious non-environment directories
            skip_dirs = {'models', 'tools', 'src', 'docs', '__pycache__', '.git', 'benchmark_results'}
            if item.name.lower() in skip_dirs:
                continue
            
            validation = validate_virtual_environment(item)
            env_info = {
                'name': item.name,
                'path': item,
                'validation': validation,
                'priority': 0
            }
            
            # Assign priority based on name patterns
            name_lower = item.name.lower()
            if 'openvino' in name_lower:
                env_info['priority'] = 100
            elif any(keyword in name_lower for keyword in ['venv', 'env']):
                env_info['priority'] = 50
            elif name_lower.startswith('llm'):
                env_info['priority'] = 30
            else:
                env_info['priority'] = 10
            
            environments.append(env_info)
    
    except Exception as e:
        print(f"{Colors.RED}Error scanning for virtual environments: {e}{Colors.ENDC}")
        return []
    
    # Sort by priority (highest first) then by name
    environments.sort(key=lambda x: (-x['priority'], x['name']))
    
    return environments

def display_valid_environment_options(valid_environments: List[Dict]) -> None:
    """Display available virtual environments in a user-friendly format."""
    print(f"\n{Colors.CYAN}Available Virtual Environments:{Colors.ENDC}")
    print(f"{Colors.WHITE}{'#':<3} {'Name':<25} {'Path'}{Colors.ENDC}")
    print(f"{Colors.WHITE}{'-'*75}{Colors.ENDC}")
    
    for i, env in enumerate(valid_environments, 1):
        print(f"{Colors.WHITE}{i:<3} {env['name']:<25} {Colors.GRAY}{env['path']}{Colors.ENDC}")

def prompt_for_environment_choice(environments: List[Dict], default_name: str) -> Dict:
    """
    Prompt user to choose an environment or create a new one.
    Returns environment choice information.
    """
    valid_environments = [env for env in environments if env['validation']['valid']]
    
    print(f"\n{Colors.YELLOW}Virtual Environment Options:{Colors.ENDC}")
    
    if valid_environments:
        display_valid_environment_options(valid_environments)
        create_new_option_num = len(valid_environments) + 1
        print(f"\n{Colors.WHITE}{create_new_option_num}. Create new environment (default: {default_name}){Colors.ENDC}")
        
        while True:
            choice = input(f"{Colors.CYAN}Enter your choice (1-{create_new_option_num}, default: {create_new_option_num}): {Colors.ENDC}").strip()
            
            if not choice or choice == str(create_new_option_num):
                # Create new environment
                env_name = input(f"{Colors.CYAN}Enter environment name (default: {default_name}): {Colors.ENDC}").strip()
                if not env_name:
                    env_name = default_name
                return {'action': 'create', 'name': env_name, 'path': None}
            
            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(valid_environments):
                    selected_env = valid_environments[choice_idx]
                    return {
                        'action': 'use_existing',
                        'name': selected_env['name'],
                        'path': selected_env['path'],
                        'validation': selected_env['validation']
                    }
                else:
                    print(f"{Colors.RED}Invalid choice. Please enter a number between 1 and {create_new_option_num}.{Colors.ENDC}")
            except ValueError:
                print(f"{Colors.RED}Invalid input. Please enter a number.{Colors.ENDC}")
    else:
        print(f"{Colors.YELLOW}No existing virtual environments found.{Colors.ENDC}")
        env_name = input(f"{Colors.CYAN}Enter environment name (default: {default_name}): {Colors.ENDC}").strip()
        if not env_name:
            env_name = default_name
        return {'action': 'create', 'name': env_name, 'path': None}

def repair_virtual_environment(venv_path: Path, llm_bench_path: Path) -> bool:
    """
    Attempt to repair a corrupted virtual environment.
    """
    print(f"\n{Colors.YELLOW}Attempting to repair virtual environment: {venv_path.name}{Colors.ENDC}")
    
    # First, try to backup any user content
    backup_successful = False
    try:
        # Look for common user files that might be worth preserving
        user_files = []
        for pattern in ['*.py', '*.txt', '*.md', '*.json']:
            user_files.extend(venv_path.glob(pattern))
        
        if user_files:
            backup_dir = llm_bench_path / f"backup_{venv_path.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_dir.mkdir(exist_ok=True)
            
            for file in user_files:
                shutil.copy2(file, backup_dir)
            
            print(f"{Colors.GREEN}User files backed up to: {backup_dir}{Colors.ENDC}")
            backup_successful = True
    except Exception as e:
        print(f"{Colors.YELLOW}Could not backup user files: {e}{Colors.ENDC}")
    
    # Remove the corrupted environment
    try:
        shutil.rmtree(venv_path)
        print(f"{Colors.GREEN}Removed corrupted environment{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.RED}Failed to remove corrupted environment: {e}{Colors.ENDC}")
        return False
    
    # Recreate the environment
    return create_virtual_environment(venv_path.name, llm_bench_path)

def create_virtual_environment(env_name: str, llm_bench_path: Path) -> bool:
    """
    Create a new virtual environment with comprehensive error handling.
    """
    venv_path = llm_bench_path / env_name
    
    print(f"\n{Colors.YELLOW}Creating virtual environment: {env_name}{Colors.ENDC}")
    print(f"{Colors.GRAY}Location: {venv_path}{Colors.ENDC}")
    
    # Check if the directory already exists
    if venv_path.exists():
        print(f"{Colors.YELLOW}Directory already exists. Checking if it can be used...{Colors.ENDC}")
        validation = validate_virtual_environment(venv_path)
        
        if validation['valid']:
            print(f"{Colors.GREEN}Existing environment is valid and will be used.{Colors.ENDC}")
            return True
        elif validation['can_repair']:
            repair_choice = input(f"{Colors.CYAN}Existing environment has issues. Attempt repair? (y/n): {Colors.ENDC}").strip().lower()
            if repair_choice == 'y':
                return repair_virtual_environment(venv_path, llm_bench_path)
        
        # If we can't repair, ask to overwrite
        overwrite_choice = input(f"{Colors.CYAN}Remove existing directory and create new environment? (y/n): {Colors.ENDC}").strip().lower()
        if overwrite_choice != 'y':
            print(f"{Colors.RED}Cannot proceed without resolving the existing directory.{Colors.ENDC}")
            return False
        
        try:
            shutil.rmtree(venv_path)
            print(f"{Colors.GREEN}Existing directory removed.{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.RED}Failed to remove existing directory: {e}{Colors.ENDC}")
            return False
    
    # Create the virtual environment
    try:
        print(f"{Colors.YELLOW}Creating virtual environment...{Colors.ENDC}")
        result = subprocess.run(
            ["python", "-m", "venv", str(venv_path)],
            cwd=llm_bench_path,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode != 0:
            print(f"{Colors.RED}Failed to create virtual environment:{Colors.ENDC}")
            print(f"{Colors.RED}STDOUT: {result.stdout}{Colors.ENDC}")
            print(f"{Colors.RED}STDERR: {result.stderr}{Colors.ENDC}")
            return False
        
        print(f"{Colors.GREEN}Virtual environment created successfully.{Colors.ENDC}")
        
    except subprocess.TimeoutExpired:
        print(f"{Colors.RED}Virtual environment creation timed out.{Colors.ENDC}")
        return False
    except Exception as e:
        print(f"{Colors.RED}Error creating virtual environment: {e}{Colors.ENDC}")
        return False
    
    # Validate the newly created environment
    validation = validate_virtual_environment(venv_path)
    if not validation['valid']:
        print(f"{Colors.RED}Newly created environment failed validation:{Colors.ENDC}")
        for issue in validation['issues']:
            print(f"{Colors.RED}  - {issue}{Colors.ENDC}")
        return False
    
    print(f"{Colors.GREEN}Virtual environment validation passed.{Colors.ENDC}")
    return True

def setup_virtual_environment(llm_bench_path: Path) -> Optional[Path]:
    """
    Main function to handle virtual environment setup with robust error handling.
    Returns the path to the validated virtual environment or None if setup failed.
    """
    print(f"\n{Colors.CYAN}{'='*60}{Colors.ENDC}")
    print(f"{Colors.CYAN}VIRTUAL ENVIRONMENT SETUP{Colors.ENDC}")
    print(f"{Colors.CYAN}{'='*60}{Colors.ENDC}")
    
    # Default environment name
    default_env_name = "openvino_env"
    
    # Search for existing environments
    existing_environments = find_existing_virtual_environments(llm_bench_path)
    
    # Get user choice
    choice = prompt_for_environment_choice(existing_environments, default_env_name)
    
    if choice['action'] == 'create':
        # Create new environment
        env_name = choice['name']
        venv_path = llm_bench_path / env_name
        
        if create_virtual_environment(env_name, llm_bench_path):
            return venv_path
        else:
            print(f"{Colors.RED}Failed to create virtual environment.{Colors.ENDC}")
            return None
    
    elif choice['action'] == 'use_existing':
        # Use existing environment
        venv_path = choice['path']
        validation = choice['validation']
        
        if validation['valid']:
            print(f"{Colors.GREEN}Using existing valid environment: {choice['name']}{Colors.ENDC}")
            return venv_path
        else:
            print(f"{Colors.YELLOW}Selected environment has issues:{Colors.ENDC}")
            for issue in validation['issues']:
                print(f"{Colors.YELLOW}  - {issue}{Colors.ENDC}")
            
            if validation['can_repair']:
                repair_choice = input(f"{Colors.CYAN}Attempt to repair this environment? (y/n): {Colors.ENDC}").strip().lower()
                if repair_choice == 'y':
                    if repair_virtual_environment(venv_path, llm_bench_path):
                        return venv_path
            
            print(f"{Colors.RED}Cannot use selected environment. Please choose a different option.{Colors.ENDC}")
            return setup_virtual_environment(llm_bench_path)  # Recursive call to restart selection
    
    return None

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

def test_software(name, version_command, install_instructions):
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
        print(f"{Colors.YELLOW}Please install {name} manually. {install_instructions}{Colors.ENDC}")
        return False

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

        # Memory Consumption
    # Always enable memory consumption logging for all iterations
    config['benchmark_memory_consumption'] = '2'  # Always log memory for all iterations

        # Other benchmark settings

    config['benchmark_input_tokens'] = int(input(f"{Colors.CYAN}   Default input token limit (default: 128): {Colors.ENDC}").strip() or "128")
    config['benchmark_iterations'] = int(input(f"{Colors.CYAN}   Number of benchmark iterations (default: 1): {Colors.ENDC}").strip() or "1")

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
        "--ratio", "1.0"
    ] + sym_flag + [str(output_path)]
    
    print(f"\n{Colors.CYAN}Quantizing {device.upper()} model: {model_id}{Colors.ENDC}")
    print(f"{Colors.GRAY}Output: {output_path}{Colors.ENDC}")
    
    # Set up environment
    env = os.environ.copy()
    if hf_token:
        env['HF_TOKEN'] = hf_token
    
    # Run in virtual environment
    success = run_in_activated_environment(
        venv_path=venv_path,
        command=command,
        working_directory=llm_bench_path,
        env_vars=env
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

def detect_openvino_model_type(model_folder: Path):
    """
    Detect OpenVINO model type by analyzing folder contents and name.
    Returns dict with model info or None if not a valid OpenVINO model.
    """
    required_files = [
        'openvino_model.xml',
        'openvino_model.bin',
        'config.json'
    ]
    
    # Check if all required OpenVINO files exist
    missing_files = [f for f in required_files if not (model_folder / f).exists()]
    if missing_files:
        return None
    
    folder_name = model_folder.name.lower()
    
    # Detect quantization type from folder name
    quantization_info = {
        'precision': 'unknown',
        'method': 'unknown',
        'group_size': 'unknown'
    }
    
    # Precision detection
    if 'int4' in folder_name:
        quantization_info['precision'] = 'int4'
    elif 'int8' in folder_name:
        quantization_info['precision'] = 'int8'
    elif 'fp16' in folder_name:
        quantization_info['precision'] = 'fp16'
    elif 'fp32' in folder_name:
        quantization_info['precision'] = 'fp32'
    
    # Method detection
    if 'awq' in folder_name:
        quantization_info['method'] = 'awq'
    elif 'cwq' in folder_name or 'cw' in folder_name.split('_'):
        quantization_info['method'] = 'channel_wise'
    elif 'gw' in folder_name.split('_'):
        quantization_info['method'] = 'group_wise'
    elif 'datafree' in folder_name:
        quantization_info['method'] = 'datafree'
    
    # Group size detection
    if '_128' in folder_name:
        quantization_info['group_size'] = '128'
    elif '_64' in folder_name:
        quantization_info['group_size'] = '64'
    elif '_32' in folder_name:
        quantization_info['group_size'] = '32'
    
    return {
        'path': model_folder,
        'name': model_folder.name,
        'is_quantized': quantization_info['precision'] in ['int4', 'int8'],
        'quantization': quantization_info,
        'size_mb': sum(f.stat().st_size for f in model_folder.glob('*') if f.is_file()) / (1024*1024)
    }

def get_all_openvino_models(models_folder: Path):
    """
    Find all OpenVINO models in the folder and categorize them.
    """
    if not models_folder.exists():
        print(f"{Colors.RED}Models folder path not found: {models_folder}{Colors.ENDC}")
        return []
    
    print(f"\n{Colors.CYAN}Scanning for OpenVINO models in: {models_folder}{Colors.ENDC}")
    
    all_models = []
    
    for item in models_folder.iterdir():
        if not item.is_dir():
            continue
            
        model_info = detect_openvino_model_type(item)
        if model_info:
            all_models.append(model_info)
            print(f"{Colors.GREEN}✓ Found: {model_info['name']} ({model_info['quantization']['precision']}){Colors.ENDC}")
    
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

    """Creates a directory to store benchmark logs for the current session."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_name = f"benchmark_session_{timestamp}"
    session_path = base_path / session_name
    session_path.mkdir(exist_ok=True, parents=True)
    
    return {"SessionName": session_name, "SessionPath": session_path, "Timestamp": timestamp}

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

def install_additional_requirements():
    """Installs additional requirements for model management."""
    print(f"\n{Colors.CYAN}Installing additional requirements for model management...{Colors.ENDC}")
    
    additional_packages = [
        "requests",
        "beautifulsoup4", 
        "packaging"
    ]
    
    for package in additional_packages:
        result = subprocess.run([sys.executable, "-m", "pip", "install", package], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"{Colors.RED}Failed to install {package}: {result.stderr}{Colors.ENDC}")
            return False
    
    return True

# Main Script Logic
# -----------------------------------------------------------------------------

def set_proxy_permanently(proxy_url):
    """Sets the HTTP_PROXY and HTTPS_PROXY environment variables permanently for the current user."""
    print(f"{Colors.YELLOW}Setting proxy configuration permanently...{Colors.ENDC}")
    try:
        os.environ['HTTP_PROXY'] = proxy_url
        os.environ['HTTPS_PROXY'] = proxy_url
        print(f"{Colors.GREEN}Proxy set for current session: {proxy_url}{Colors.ENDC}")

        print(f"{Colors.YELLOW}Setting permanent proxy for current user...{Colors.ENDC}")
        result = subprocess.run(f'setx HTTP_PROXY "{proxy_url}"', shell=True)
        if result.returncode == 0:
            result = subprocess.run(f'setx HTTPS_PROXY "{proxy_url}"', shell=True)
            if result.returncode == 0:
                print(f"{Colors.GREEN}User-level proxy environment variables set successfully.{Colors.ENDC}")
            else:
                print(f"{Colors.RED}Failed to set HTTPS_PROXY variable.{Colors.ENDC}")
                return False
        else:
            print(f"{Colors.RED}Failed to set HTTP_PROXY variable.{Colors.ENDC}")
            return False

        if get_admin_status():
            print(f"{Colors.YELLOW}Attempting to set system-wide proxy (requires admin privileges)...{Colors.ENDC}")
            result = subprocess.run(f'setx HTTP_PROXY "{proxy_url}" /M', shell=True)
            if result.returncode == 0:
                result = subprocess.run(f'setx HTTPS_PROXY "{proxy_url}" /M', shell=True)
                if result.returncode == 0:
                    print(f"{Colors.GREEN}System-wide proxy environment variables set successfully.{Colors.ENDC}")
                else:
                    print(f"{Colors.RED}Failed to set system-wide HTTPS_PROXY.{Colors.ENDC}")
            else:
                print(f"{Colors.RED}Failed to set system-wide HTTP_PROXY.{Colors.ENDC}")
        else:
            print(f"{Colors.YELLOW}Not running as administrator - system-wide proxy not set.{Colors.ENDC}")
            print(f"{Colors.CYAN}User-level proxy settings will be sufficient for most applications.{Colors.ENDC}")

        print(f"{Colors.GREEN}Proxy configuration completed successfully!{Colors.ENDC}")
        return True
    except Exception as e:
        print(f"{Colors.RED}Failed to set proxy configuration: {e}{Colors.ENDC}")
        return False

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
    
    # Install additional requirements first
    if not install_additional_requirements():
        print(f"{Colors.RED}Failed to install additional requirements. Please install manually and rerun.{Colors.ENDC}")
        sys.exit(1)
    
    # Get comprehensive configuration upfront
    print(f"\n{Colors.CYAN}Collecting all configuration inputs...{Colors.ENDC}")
    config = get_comprehensive_configuration()
    if not config:
        print(f"{Colors.RED}Configuration failed. Exiting.{Colors.ENDC}")
        sys.exit(1)
    
    # --- STEP 1: PROXY CONFIGURATION ---
    proxy_url = "http://proxy-iind.intel.com:911"
    
    print(f"\n{Colors.CYAN}STEP 1: Checking proxy configuration...{Colors.ENDC}")
    
    if os.getenv("HTTP_PROXY") != proxy_url or os.getenv("HTTPS_PROXY") != proxy_url:
        print(f"{Colors.YELLOW}Proxy not configured. Setting up permanent proxy configuration...{Colors.ENDC}")
        if not set_proxy_permanently(proxy_url):
            print(f"{Colors.RED}Failed to set proxy configuration. Please set manually and rerun the script.{Colors.ENDC}")
            sys.exit(1)
    else:
        print(f"{Colors.GREEN}Proxy already configured correctly. Continuing...{Colors.ENDC}")
        
    # --- STEP 2: CHECK SOFTWARE DEPENDENCIES ---
    print(f"\n{Colors.CYAN}STEP 2: Checking software dependencies...{Colors.ENDC}")
    
    if not test_software("Python", ["python", "--version"], "Download from https://www.python.org/downloads/windows/ and ensure it's added to PATH."):
        sys.exit(1)
        
    if not test_software("Git", ["git", "--version"], "Download from https://git-scm.com/download/win and ensure it's added to PATH."):
        sys.exit(1)

    print(f"\n{Colors.YELLOW}VC++ Redistributables Check:{Colors.ENDC}")
    print(f"{Colors.YELLOW}It's highly recommended to have the latest Microsoft Visual C++ Redistributable (x64) installed.{Colors.ENDC}")
    print(f"{Colors.YELLOW}Download from: https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170{Colors.ENDC}")
    
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
        
    print(f"\n{Colors.YELLOW}Installing OpenVINO packages in activated environment...{Colors.ENDC}")
    openvino_packages_cmd = ["pip", "install", "--pre", "openvino", "openvino-tokenizers", "openvino-genai", "--extra-index-url", "https://storage.openvinotoolkit.org/simple/wheels/nightly", "--upgrade"]
    run_in_activated_environment(
        venv_path=str(venv_path),
        command=openvino_packages_cmd,
        working_directory=str(llm_bench_path)
    )
    
    # Install additional packages for model management
    print(f"\n{Colors.YELLOW}Installing model management packages...{Colors.ENDC}")
    model_mgmt_packages = ["requests", "beautifulsoup4", "packaging"]
    for package in model_mgmt_packages:
        run_in_activated_environment(
            venv_path=str(venv_path),
            command=["pip", "install", package],
            working_directory=str(llm_bench_path)
        )
    
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
