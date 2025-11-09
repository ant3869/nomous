import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import venv
import webbrowser

import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

try:  # pragma: no cover - import guard for pristine environments
    from rich.console import Console
    from rich.panel import Panel
except ModuleNotFoundError:  # pragma: no cover - rich is installed later
    class Console:  # type: ignore[override]
        """Minimal fallback console when rich is unavailable."""

        def print(self, message):  # noqa: D401 - simple passthrough
            print(message)

    class _PanelStub:
        """Fallback Panel implementation that behaves like a string."""

        def __init__(self, renderable, border_style=None):
            self.renderable = renderable
            self.border_style = border_style

        def __str__(self):
            prefix = "[" + self.border_style + "]" if self.border_style else ""
            suffix = "[/]" if self.border_style else ""
            return f"{prefix}{self.renderable}{suffix}"

        @classmethod
        def fit(cls, renderable, border_style=None):
            return cls(renderable, border_style)

    Panel = _PanelStub  # type: ignore

console = Console()

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "nomous.log"

logger = logging.getLogger("nomous.start")
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
logger.propagate = False

ASCII_BANNER = """
 _   _                      _
| \\ | |                    | |
|  \\| | ___  _ __ ___   ___| |_ ___  _   _ ___
| . ` |/ _ \\| '_ ` _ \\ / _ \\ __/ _ \\| | | / __|
| |\\  | (_) | | | | | |  __/ || (_) | |_| \\__ \\
|_| \\_|\\___/|_| |_| |_|\\___|\\__\\___/ \\__,_|___/
"""


MIN_PYTHON_VERSION = (3, 10)
MIN_NODE_VERSION = (18, 0, 0)
MIN_NPM_VERSION = (9, 0, 0)


def get_runtime_version() -> str:
    """Derive the nomous runtime version from repository metadata."""

    version_file = PROJECT_ROOT / "VERSION"
    if version_file.exists():
        version = version_file.read_text(encoding="utf-8").strip()
        if version:
            return version

    package_json = PROJECT_ROOT / "package.json"
    if package_json.exists():
        try:
            with package_json.open("r", encoding="utf-8") as handle:
                package_data = json.load(handle)
            version = str(package_data.get("version", "")).strip()
            if version:
                return version
        except Exception as exc:
            logger.warning("Unable to read package.json version: %s", exc)

    return "unknown"


NOMOUS_VERSION = get_runtime_version()


def emit_start_banner():
    """Render the startup banner to the console and log file."""

    separator = "=" * 80
    header = f"nomous runtime start • version {NOMOUS_VERSION} • pid {os.getpid()}"

    logger.info("")
    logger.info(separator)
    logger.info(header)
    for line in ASCII_BANNER.strip().splitlines():
        logger.info(line)
    logger.info(separator)

    console.print(f"[bold blue]{ASCII_BANNER}[/]")
    console.print(
        Panel.fit(
            f"[bold blue]nomous[/] v{NOMOUS_VERSION} - Local Autonomy Runtime",
            border_style="blue",
        )
    )


def parse_semver(raw: str) -> Optional[Tuple[int, int, int]]:
    """Extract the first semantic version tuple from arbitrary text."""

    match = re.search(r"(\d+)\.(\d+)\.(\d+)", raw)
    if not match:
        return None
    return tuple(int(part) for part in match.groups())


def is_version_supported(found: Tuple[int, int, int], minimum: Tuple[int, int, int]) -> bool:
    """Return True when the detected version meets the minimum requirement."""

    return found >= minimum


def check_python_version() -> bool:
    """Ensure the active interpreter matches the supported version."""

    if sys.version_info < MIN_PYTHON_VERSION:
        required = ".".join(str(x) for x in MIN_PYTHON_VERSION)
        detected = sys.version.split()[0]
        print_error(
            "Unsupported Python version detected."
            f" Nomous requires Python {required}+ but {detected} is running."
        )
        print_warning(
            "Install a newer Python release from https://www.python.org/downloads/"
        )
        return False
    return True


def ensure_command_available(
    command: str,
    friendly_name: str,
    minimum_version: Optional[Tuple[int, int, int]] = None,
    install_hint: Optional[str] = None,
) -> bool:
    """Verify that a CLI command exists and, optionally, meets version requirements."""

    if not shutil.which(command):
        print_error(f"{friendly_name} was not found on PATH.")
        if install_hint:
            print_warning(install_hint)
        return False

    if minimum_version is None:
        return True

    success, output = run_command(f"{command} --version")
    if not success:
        print_warning(
            f"Unable to determine {friendly_name} version; continuing but behaviour may vary."
        )
        if install_hint:
            print_warning(install_hint)
        return True

    version = parse_semver(output or "")
    if version is None:
        print_warning(
            f"Could not parse {friendly_name} version from '{(output or '').strip()}'."
        )
        if install_hint:
            print_warning(install_hint)
        return True

    if not is_version_supported(version, minimum_version):
        required = ".".join(str(x) for x in minimum_version)
        detected = ".".join(str(x) for x in version)
        print_error(
            f"{friendly_name} {detected} is too old. Version {required}+ is required."
        )
        if install_hint:
            print_warning(install_hint)
        return False

    print_success(f"Detected {friendly_name} {'.'.join(str(x) for x in version)}")
    return True


def _emit(message: str, level: int, icon: str, style: str) -> None:
    """Log a message and render it to the console with consistent styling."""

    logger.log(level, message)
    console.print(f"[{style}]{icon}[/] {message}")

def print_status(message, style="bold green"):
    """Print styled status message"""
    _emit(message, logging.INFO, "▶", style)

def print_error(message):
    """Print error message"""
    _emit(message, logging.ERROR, "✖", "bold red")

def print_warning(message):
    """Print warning message"""
    _emit(message, logging.WARNING, "⚠", "bold yellow")

def print_success(message):
    """Print success message"""
    _emit(message, logging.INFO, "✓", "bold green")

@dataclass
class PythonEnvironment:
    """Details about the Python environment used for subprocesses."""

    python_executable: Path
    pip_command: str
    root: Optional[Path]
    created: bool
    marker_path: Path


def _environment_marker_path(env_root: Optional[Path]) -> Path:
    """Return a writable marker path for tracking dependency installation."""

    if env_root and env_root.exists() and os.access(env_root, os.W_OK):
        return env_root / "nomous-pip-installed"
    return PROJECT_ROOT / ".nomous-pip-installed"


def ensure_python_environment() -> Optional[PythonEnvironment]:
    """Ensure a working Python environment and return its metadata."""

    # If we're already running inside a virtual environment, reuse it.
    if sys.prefix != sys.base_prefix or os.environ.get("VIRTUAL_ENV"):
        python_path = Path(sys.executable).resolve()
        env_root = Path(os.environ.get("VIRTUAL_ENV", sys.prefix)).resolve()
        pip_cmd = f'"{python_path}" -m pip'
        marker = _environment_marker_path(env_root)
        logger.info("Reusing active Python environment at %s", env_root)
        return PythonEnvironment(python_path, pip_cmd, env_root, False, marker)

    # Otherwise create or reuse the project-local virtual environment.
    env_root = PROJECT_ROOT / ".venv"
    python_path = env_root / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    pip_cmd = f'"{python_path}" -m pip'

    created = False
    if not env_root.exists():
        print_status("Creating virtual environment...")
        try:
            venv.create(env_root, with_pip=True, system_site_packages=True)
            created = True
            logger.info("Created new virtual environment at %s", env_root)
        except Exception as exc:
            print_error(f"Failed to create virtual environment: {exc}")
            return None

        if not python_path.exists():
            print_error(f"Virtual environment created but Python executable not found at {python_path}")
            logger.error("Python executable missing after venv.create at %s", python_path)
            return None
        success, output = run_command(f"{pip_cmd} install --upgrade pip")
        if success:
            print_success("Virtual environment created")
        else:
            print_warning(
                f"Virtual environment created but failed to upgrade pip: {output.strip() or 'see logs'}"
            )
    else:
        logger.info("Using existing virtual environment at %s", env_root)

    marker = _environment_marker_path(env_root)
    return PythonEnvironment(python_path, pip_cmd, env_root, created, marker)

def run_command(command, cwd=None, shell=True):
    """Run a command and return its output"""
    logger.info("Running command: %s", command)
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            shell=shell,
            check=True,
            text=True,
            capture_output=True
        )
        stdout = (result.stdout or "").strip()
        if stdout:
            logger.info("Command output: %s", stdout)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        stdout = (e.stdout or "").strip()
        logger.error("Command failed with return code %s", e.returncode)
        if stdout:
            logger.error("Command stdout: %s", stdout)
        if stderr:
            logger.error("Command stderr: %s", stderr)
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

def check_and_install_dependencies() -> Tuple[bool, Optional[PythonEnvironment]]:
    """Check and install project dependencies."""

    env_info = ensure_python_environment()
    if env_info is None:
        logger.error("Unable to initialize Python environment; aborting startup")
        return False, None
    pip_cmd = env_info.pip_command
    logger.info("Python executable resolved to %s", env_info.python_executable)

    # Check node_modules
    if not Path('node_modules').exists():
        print_status("Installing Node.js dependencies...")
        success, output = run_command('npm install')
        if success:
            print_success("Node.js dependencies installed")
        else:
            print_error("Failed to install Node.js dependencies.")
            trimmed = (output or "").strip() if isinstance(output, str) else ""
            if trimmed:
                print_warning(trimmed)
            print_warning(
                "Try running 'npm install' manually after ensuring Node.js 18+ is installed."
            )
            return False, env_info
    else:
        print_success("Node.js dependencies already installed")

    # Install Python requirements if new venv or requirements changed
    if env_info.created or not env_info.marker_path.exists():
        print_status("Installing Python dependencies...")
        # Install project requirements
        success, output = run_command(f'{pip_cmd} install -r requirements.txt')
        if success:
            # Create marker file to track installation
            try:
                env_info.marker_path.parent.mkdir(parents=True, exist_ok=True)
                env_info.marker_path.touch()
            except OSError as exc:
                print_warning(f"Could not update dependency marker: {exc}")
            print_success("Python dependencies installed")
        else:
            print_error("Failed to install Python dependencies.")
            trimmed = (output or "").strip() if isinstance(output, str) else ""
            if trimmed:
                print_warning(trimmed)
            print_warning(
                "Activate the virtual environment and rerun "
                f"'{pip_cmd} install -r requirements.txt' to inspect the issue."
            )
            return False, env_info
    else:
        print_success("Python dependencies already installed")
        logger.info("Python dependencies previously satisfied")

    return True, env_info

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
    logger.info("Waiting for backend service on ws://%s:%s", ws_host, ws_port)
    while time.time() - start_time < timeout:
        if is_port_in_use(ws_port):
            print_success(f"Backend is ready at ws://{ws_host}:{ws_port}")
            return True
        time.sleep(1)
        print_status("Waiting for backend to start...", style="bold cyan")
    logger.error("Backend did not start within %s seconds", timeout)
    return False

def cleanup_processes(processes):
    """Cleanup processes on exit"""
    for proc in processes:
        try:
            if sys.platform == 'win32':
                if proc.poll() is None:
                    logger.info("Terminating process %s", proc.pid)
                    proc.kill()
            else:
                if proc.poll() is None:
                    logger.info("Terminating process group %s", proc.pid)
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            logger.exception("Error while terminating process %s", getattr(proc, "pid", "unknown"))

def main():
    emit_start_banner()

    if not check_python_version():
        return 1

    if not ensure_command_available(
        "node",
        "Node.js",
        MIN_NODE_VERSION,
        "Install Node.js 18+ from https://nodejs.org/en/download to continue.",
    ):
        return 1

    if not ensure_command_available(
        "npm",
        "npm",
        MIN_NPM_VERSION,
        "Install Node.js 18+ (which bundles npm) from https://nodejs.org/en/download.",
    ):
        return 1

    # Change to project root directory
    os.chdir(Path(__file__).parent.parent)
    print_status(f"Working directory set to {Path.cwd()}", style="bold cyan")

    # Check and install dependencies
    ok, env_info = check_and_install_dependencies()
    if not ok or env_info is None:
        logger.error("Startup halted during dependency verification")
        return 1

    # Check CUDA availability and attempt to enable GPU acceleration
    device_info = check_compute_backend()
    device_info = ensure_gpu_support(env_info.pip_command, device_info)
    update_config_for_gpu(device_info.is_gpu)
    
    device_label = f"{device_info.backend} ({device_info.name})"
    system_summary = (
        "System Configuration\n"
        f"Device: {device_label}\n"
        f"Reason: {device_info.reason}\n"
        f"CUDA: {device_info.cuda_version or 'not detected'}\n"
        f"Python: {sys.version.split()[0]}\n"
        f"Operating System: {sys.platform}"
    )
    logger.info(system_summary.replace("\n", " | "))
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
        logger.exception("Failed to load configuration")
        return 1

    processes = []
    
    try:
        # Start backend
        print_status("Starting backend server...")
        backend_script = Path('scripts/run_bridge.py')
        backend_proc = subprocess.Popen(
            [str(env_info.python_executable), str(backend_script)],
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
        )
        processes.append(backend_proc)
        logger.info("Backend process started with PID %s", backend_proc.pid)

        # Wait for backend to be ready
        if not wait_for_backend(ws_host, ws_port):
            print_error("Backend failed to start within timeout period")
            cleanup_processes(processes)
            logger.error("Backend startup timed out; services terminated")
            return 1

        # Start frontend
        print_status("Starting frontend development server...")
        npm_cmd = 'npm.cmd' if sys.platform == 'win32' else 'npm'
        frontend_proc = subprocess.Popen(
            [npm_cmd, 'run', 'dev'],
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
        )
        processes.append(frontend_proc)
        logger.info("Frontend process started with PID %s", frontend_proc.pid)

        # Wait a moment for the dev server to start
        time.sleep(5)
        
        # Open browser
        frontend_url = "http://localhost:5173"  # Default Vite dev server port
        print_status(f"Opening browser to {frontend_url}")
        webbrowser.open(frontend_url)
        logger.info("Browser open requested for %s", frontend_url)

        console.print(Panel(
            "[bold green]Services are running![/]\n"
            "• Press [bold]Ctrl+C[/] to stop all services\n"
            f"• Frontend: [bold blue]{frontend_url}[/]\n"
            f"• Backend: [bold blue]ws://{ws_host}:{ws_port}[/]",
            border_style="green"
        ))
        logger.info(
            "Services running • Frontend: %s • Backend: ws://%s:%s",
            frontend_url,
            ws_host,
            ws_port,
        )

        # Keep the script running and handle keyboard interrupt
        while True:
            if any(proc.poll() is not None for proc in processes):
                print_error("One of the services has stopped unexpectedly")
                logger.error("Detected unexpected process termination")
                break
            time.sleep(1)

    except KeyboardInterrupt:
        print_status("\nStopping all services...", style="bold yellow")
        logger.info("Keyboard interrupt received; shutting down services")
    finally:
        cleanup_processes(processes)
        print_success("All services have been stopped")
        logger.info("nomous runtime shutdown complete")

    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as exc:
        logger.exception("Unhandled exception during startup")
        print_error(f"Unhandled exception: {exc}")
        exit_code = 1

    logger.info("nomous start script exiting with code %s", exit_code)
    sys.exit(exit_code)
