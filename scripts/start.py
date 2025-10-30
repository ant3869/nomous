import subprocess
import sys
import time
import webbrowser
import yaml
import os
import signal
import psutil
import venv
import torch
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()

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

def check_cuda():
    """Check CUDA availability and configuration"""
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        print_success(f"CUDA is available! Found {device_count} device(s)")
        print_success(f"Using: {device_name}")
        return True, f"GPU ({device_name})"
    else:
        print_warning("CUDA is not available. Using CPU mode")
        return False, "CPU"

def setup_virtual_env():
    """Setup virtual environment if it doesn't exist"""
    venv_path = Path('.venv')
    if not venv_path.exists():
        print_status("Creating virtual environment...")
        venv.create('.venv', with_pip=True)
        print_success("Virtual environment created")
        return True
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

def check_and_install_dependencies():
    """Check and install project dependencies"""
    # Check virtual environment
    is_new_venv = setup_virtual_env()
    
    # Check node_modules
    if not Path('node_modules').exists():
        print_status("Installing Node.js dependencies...")
        success, output = run_command('npm install')
        if success:
            print_success("Node.js dependencies installed")
        else:
            print_error(f"Failed to install Node.js dependencies: {output}")
            return False
    else:
        print_success("Node.js dependencies already installed")

    # Install Python requirements if new venv or requirements changed
    if is_new_venv or not Path('.venv/pip-installed').exists():
        print_status("Installing Python dependencies...")
        venv_pip = '.venv/Scripts/pip' if sys.platform == 'win32' else '.venv/bin/pip'
        success, output = run_command(f"{venv_pip} install -r requirements.txt")
        if success:
            # Create marker file to track installation
            Path('.venv/pip-installed').touch()
            print_success("Python dependencies installed")
        else:
            print_error(f"Failed to install Python dependencies: {output}")
            return False
    else:
        print_success("Python dependencies already installed")
    
    return True

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
    for proc in psutil.process_iter(['pid', 'name', 'connections']):
        try:
            connections = proc.connections()
            for conn in connections:
                if conn.laddr.port == port:
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False

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
    if not check_and_install_dependencies():
        return 1

    # Check CUDA availability
    cuda_available, device_type = check_cuda()
    update_config_for_gpu(cuda_available)
    
    console.print(Panel(
        f"[bold]System Configuration[/]\n"
        f"Device: [bold {'green' if cuda_available else 'yellow'}]{device_type}[/]\n"
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