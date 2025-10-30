import subprocess
import sys
import time
import webbrowser
import yaml
import os
import signal
import psutil
from pathlib import Path

def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

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
    # Get project root directory
    project_root = Path(__file__).parent.parent
    
    # Load configuration
    try:
        config = load_config()
        ws_host = config['ws']['host']
        ws_port = config['ws']['port']
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return 1

    processes = []
    
    try:
        # Start backend
        print("🚀 Starting backend server...")
        backend_script = project_root / 'scripts' / 'run_bridge.py'
        backend_proc = subprocess.Popen([sys.executable, str(backend_script)],
                                      cwd=project_root,
                                      creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0)
        processes.append(backend_proc)

        # Wait for backend to be ready
        if not wait_for_backend(ws_host, ws_port):
            print("❌ Backend failed to start within timeout period")
            cleanup_processes(processes)
            return 1

        # Start frontend
        print("🚀 Starting frontend development server...")
        npm_cmd = 'npm.cmd' if sys.platform == 'win32' else 'npm'
        frontend_proc = subprocess.Popen([npm_cmd, 'run', 'dev'],
                                       cwd=project_root,
                                       creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0)
        processes.append(frontend_proc)

        # Wait a moment for the dev server to start
        time.sleep(5)
        
        # Open browser
        frontend_url = "http://localhost:5173"  # Default Vite dev server port
        print(f"🌐 Opening browser to {frontend_url}")
        webbrowser.open(frontend_url)

        # Keep the script running and handle keyboard interrupt
        print("\n🔄 Services are running. Press Ctrl+C to stop all services...\n")
        while True:
            if any(proc.poll() is not None for proc in processes):
                print("❌ One of the services has stopped unexpectedly")
                break
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n⏹️ Stopping all services...")
    finally:
        cleanup_processes(processes)
        print("👋 All services have been stopped")

if __name__ == "__main__":
    sys.exit(main())