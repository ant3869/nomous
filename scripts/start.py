import re
import subprocess
import sys
import time
import webbrowser
import yaml
import os
import signal
import shutil
import venv
from pathlib import Path
from typing import Tuple
from rich.console import Console
from rich.panel import Panel

console = Console()

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def print_status(message, style="bold green"):
    """Print styled status message"""
    console.print(f"[{style}]▶[/] {message}")

def print_error(message):
    """Print error message"""
    console.print(f"[bold red]✖[/] {message}")

def print_warning(message):
    """Print warning message"""
    console.print(f"[bold yellow]⚠[/] {message}")

def print_success(message):
    """Print success message"""
    console.print(f"[bold green]✓[/] {message}")

def get_pip_command() -> str:
    """Return the pip command string for the local virtual environment."""
    if sys.platform == 'win32':
        return str(Path('.venv/Scripts/python.exe').absolute()) + ' -m pip'
    return str(Path('.venv/bin/python').absolute()) + ' -m pip'

def setup_virtual_env():
    """Setup virtual environment if it doesn't exist"""
    venv_path = Path('.venv')
    if not venv_path.exists():
        print_status("Creating virtual environment...")
        try:
            venv.create('.venv', with_pip=True, system_site_packages=True)
            # Upgrade pip in the new environment
            pip_cmd = get_pip_command()
            run_command(f'{pip_cmd} install --upgrade pip')
            print_success("Virtual environment created")
            return True
        except Exception as e:
            print_error(f"Failed to create virtual environment: {e}")
            return False
    return False

def activate_virtual_env():
    """Get the activation command for the virtual environment"""
    if sys.platform == 'win32':
        return str(Path('.venv/Scripts/activate.bat'))
    return f"source {str(Path('.venv/bin/activate'))}"

def run_command(command, cwd=None, shell=True):
    """Run a command and return its output"""
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            shell=shell,
            check=True,
            text=True,
            capture_output=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def detect_system_cuda_version() -> str:
    """Detect system CUDA toolkit version using nvidia-smi.
    
    Returns:
        CUDA version string like 'cu121', 'cu118', 'cu124', or 'cu121' as default
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # Try to get CUDA version from nvidia-smi
            result = subprocess.run(
                ['nvidia-smi'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                output = result.stdout
                # Look for CUDA Version in nvidia-smi output
                match = re.search(r'CUDA Version:\s+(\d+)\.(\d+)', output)
                if match:
                    major, minor = match.groups()
                    # Map to PyTorch wheel naming convention
                    version_map = {
                        ('11', '8'): 'cu118',
                        ('12', '1'): 'cu121',
                        ('12', '4'): 'cu124',
                    }
                    cuda_key = (major, minor)
                    if cuda_key in version_map:
                        return version_map[cuda_key]
                    # For other versions, try to construct a reasonable default
                    if major == '11':
                        return 'cu118'
                    elif major == '12':
                        if int(minor) >= 4:
                            return 'cu124'
                        else:
                            return 'cu121'
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    # Default to CUDA 12.1 if detection fails
    return 'cu121'

def check_compute_backend() -> "ComputeDeviceInfo":
    """Check CUDA availability and provide a compute device summary."""
    from src.backend.system import detect_compute_device

    info = detect_compute_device()
    if info.is_gpu:
        print_success(
            f"CUDA is available! Found {info.gpu_count} device(s): {info.name}"
        )
        if info.cuda_version:
            print_success(f"CUDA version: {info.cuda_version}")
    else:
        print_warning("CUDA is not available. Using CPU mode")
        print_warning(info.reason)
    return info

def ensure_gpu_support(pip_cmd: str, info: "ComputeDeviceInfo") -> "ComputeDeviceInfo":
    """Attempt to install GPU-accelerated wheels when CUDA hardware is detected."""
    from src.backend.system import detect_compute_device

    if info.is_gpu:
        return info

    if not shutil.which("nvidia-smi"):
        print_warning("No NVIDIA GPU detected by nvidia-smi; continuing with CPU mode.")
        return info

    print_status(
        "NVIDIA GPU detected but CUDA unavailable. Attempting to install GPU dependencies...",
        style="bold yellow",
    )

    # Detect system CUDA version dynamically
    cuda_version = detect_system_cuda_version()
    print_status(f"Detected CUDA version: {cuda_version}", style="bold cyan")

    gpu_installs = [
        (
            f"{pip_cmd} install --upgrade torch --index-url https://download.pytorch.org/whl/{cuda_version}",
            "PyTorch CUDA build",
        ),
        (
            f'{pip_cmd} install --upgrade "llama-cpp-python[cuda]"',
            "llama-cpp-python CUDA extension",
        ),
    ]

    for command, label in gpu_installs:
        success, output = run_command(command)
        if success:
            print_success(f"Installed {label}")
        else:
            trimmed = output.strip() if isinstance(output, str) else ""
            print_warning(f"Failed to install {label}: {trimmed or 'check logs for details'}")

    print_status("Re-checking CUDA availability...", style="bold cyan")
    refreshed = detect_compute_device()
    if refreshed.is_gpu:
        print_success(
            f"GPU acceleration enabled: {refreshed.name} (CUDA {refreshed.cuda_version or 'unknown'})"
        )
        return refreshed

    print_warning(
        "CUDA still unavailable after attempted installs. Please ensure NVIDIA drivers and toolkit are installed."
    )
    return refreshed

def check_and_install_dependencies() -> Tuple[bool, str]:
    """Check and install project dependencies."""
    # Check virtual environment
    is_new_venv = setup_virtual_env()
    pip_cmd = get_pip_command()
    
    # Check node_modules
    if not Path('node_modules').exists():
        print_status("Installing Node.js dependencies...")
        success, output = run_command('npm install')
        if success:
            print_success("Node.js dependencies installed")
        else:
            print_error(f"Failed to install Node.js dependencies: {output}")
            return False, pip_cmd
    else:
        print_success("Node.js dependencies already installed")

    # Install Python requirements if new venv or requirements changed
    if is_new_venv or not Path('.venv/pip-installed').exists():
        print_status("Installing Python dependencies...")
        # Install project requirements
        success, output = run_command(f'{pip_cmd} install -r requirements.txt')
        if success:
            # Create marker file to track installation
            Path('.venv/pip-installed').touch()
            print_success("Python dependencies installed")
        else:
            print_error(f"Failed to install Python dependencies: {output}")
            return False, pip_cmd
    else:
        print_success("Python dependencies already installed")

    return True, pip_cmd

def load_config():
    """Load configuration from config.yaml"""
    config_path = Path('config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def update_config_for_gpu(use_gpu):
    """Update configuration based on GPU availability"""
    config_path = Path('config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update GPU-specific settings
    if 'llm' in config:
        if use_gpu:
            config['llm']['n_gpu_layers'] = -1  # Use all GPU layers
        else:
            config['llm']['n_gpu_layers'] = 0   # Disable GPU layers
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

def is_port_in_use(port):
    """Check if a port is in use"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('127.0.0.1', port))
            return False
        except socket.error:
            return True

def wait_for_backend(ws_host, ws_port, timeout=30):
    """Wait for backend to be ready"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_port_in_use(ws_port):
            print(f"✅ Backend is ready at ws://{ws_host}:{ws_port}")
            return True
        time.sleep(1)
        print("⌛ Waiting for backend to start...")
    return False

def cleanup_processes(processes):
    """Cleanup processes on exit"""
    for proc in processes:
        try:
            if sys.platform == 'win32':
                proc.kill()
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except:
            pass

def main():
    console.print(Panel.fit(
        "[bold blue]Nomous[/] - Local Autonomy Runtime",
        border_style="blue"
    ))

    # Change to project root directory
    os.chdir(Path(__file__).parent.parent)

    # Check and install dependencies
    ok, pip_cmd = check_and_install_dependencies()
    if not ok:
        return 1

    # Check CUDA availability and attempt to enable GPU acceleration
    device_info = check_compute_backend()
    device_info = ensure_gpu_support(pip_cmd, device_info)
    update_config_for_gpu(device_info.is_gpu)
    
    device_label = f"{device_info.backend} ({device_info.name})"
    console.print(Panel(
        f"[bold]System Configuration[/]\n"
        f"Device: [bold {'green' if device_info.is_gpu else 'yellow'}]{device_label}[/]\n"
        f"Reason: [bold]{device_info.reason}[/]\n"
        f"CUDA: [bold]{device_info.cuda_version or 'not detected'}[/]\n"
        f"Python: [bold]{sys.version.split()[0]}[/]\n"
        f"Operating System: [bold]{sys.platform}[/]",
        border_style="blue"
    ))

    # Load configuration
    try:
        config = load_config()
        ws_host = config['ws']['host']
        ws_port = config['ws']['port']
    except Exception as e:
        print_error(f"Error loading configuration: {e}")
        return 1

    processes = []
    
    try:
        # Start backend
        print_status("Starting backend server...")
        venv_python = '.venv/Scripts/python' if sys.platform == 'win32' else '.venv/bin/python'
        backend_script = Path('scripts/run_bridge.py')
        backend_proc = subprocess.Popen(
            [venv_python, str(backend_script)],
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
        )
        processes.append(backend_proc)

        # Wait for backend to be ready
        if not wait_for_backend(ws_host, ws_port):
            print_error("Backend failed to start within timeout period")
            cleanup_processes(processes)
            return 1

        # Start frontend
        print_status("Starting frontend development server...")
        npm_cmd = 'npm.cmd' if sys.platform == 'win32' else 'npm'
        frontend_proc = subprocess.Popen(
            [npm_cmd, 'run', 'dev'],
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
        )
        processes.append(frontend_proc)

        # Wait a moment for the dev server to start
        time.sleep(5)
        
        # Open browser
        frontend_url = "http://localhost:5173"  # Default Vite dev server port
        print_status(f"Opening browser to {frontend_url}")
        webbrowser.open(frontend_url)

        console.print(Panel(
            "[bold green]Services are running![/]\n"
            "• Press [bold]Ctrl+C[/] to stop all services\n"
            f"• Frontend: [bold blue]{frontend_url}[/]\n"
            f"• Backend: [bold blue]ws://{ws_host}:{ws_port}[/]",
            border_style="green"
        ))

        # Keep the script running and handle keyboard interrupt
        while True:
            if any(proc.poll() is not None for proc in processes):
                print_error("One of the services has stopped unexpectedly")
                break
            time.sleep(1)

    except KeyboardInterrupt:
        print_status("\nStopping all services...", style="bold yellow")
    finally:
        cleanup_processes(processes)
        print_success("All services have been stopped")

if __name__ == "__main__":
    sys.exit(main())