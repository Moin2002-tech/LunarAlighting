#!/usr/bin/env python3
"""
Training launcher with automatic cleanup
Handles starting training, gym server, and data logger with proper cleanup on exit
"""

import subprocess
import signal
import sys
import time
import os
from pathlib import Path

class TrainingManager:
    def __init__(self):
        self.processes = []
        self.running = True
        
    def start_gym_server(self):
        """Start the gym server"""
        try:
            process = subprocess.Popen([
                sys.executable, "start_gym_server.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            print(" Gym server started...")
            self.processes.append(("Gym Server", process))
            return process
        except Exception as e:
            print(f" Failed to start gym server: {e}")
            return None
    
    def start_data_logger(self):
        """Start the data logger"""
        try:
            process = subprocess.Popen([
                sys.executable, "data_logger.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            print("📊 Data logger started...")
            self.processes.append(("Data Logger", process))
            return process
        except Exception as e:
            print(f" Failed to start data logger: {e}")
            return None
    
    def start_training(self):
        """Start the C++ training process"""
        try:
            # Find the executable - try the correct path first
            exe_path = Path("cmake_release/LunarAlightingRL")
            if not exe_path.exists():
                exe_path = Path("cmake-build-release/LunarAlightingRL")
            if not exe_path.exists():
                exe_path = Path("cmake-build-debug/LunarAlightingRL")
            
            if not exe_path.exists():
                print(" LunarAlightingRL executable not found")
                print("   Looked in: cmake_release/, cmake-build-release/, cmake-build-debug/")
                return None
            
            # Change to the executable directory before running
            cwd = exe_path.parent
            process = subprocess.Popen([
                f"./{exe_path.name}"
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=cwd)
            
            print(" Training started...")
            self.processes.append(("Training", process))
            return process
        except Exception as e:
            print(f" Failed to start training: {e}")
            return None
    
    def cleanup(self):
        """Clean up all processes"""
        print("\nCleaning up processes...")
        self.running = False
        
        for name, process in self.processes:
            try:
                print(f"  Stopping {name}...")
                # Try graceful shutdown first
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown fails
                    process.kill()
                    process.wait()
                print(f"  ✓ {name} stopped")
            except Exception as e:
                print(f"  Error stopping {name}: {e}")
        
        # Kill any remaining processes on the ports
        try:
            import psutil
            for port in [10201, 10202]:
                for proc in psutil.process_iter(['pid', 'name', 'connections']):
                    try:
                        for conn in proc.info['connections'] or []:
                            if conn.laddr.port == port:
                                print(f"  Killing process using port {port}: {proc.info['name']} (PID: {proc.info['pid']})")
                                proc.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        pass
        except ImportError:
            # Fallback to pkill if psutil not available
            os.system("pkill -f 'start_gym_server.py' 2>/dev/null")
            os.system("pkill -f 'data_logger.py' 2>/dev/null")
            os.system("fuser -k 10201/tcp 2>/dev/null")
            os.system("fuser -k 10202/tcp 2>/dev/null")
    
    def run(self):
        """Run the complete training setup"""
        # Set up signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            print(f"\nReceived signal {signum}")
            self.cleanup()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        print("Starting Lunar Alighting RL Training")
        print("=" * 50)
        
        # Start data logger first (as per manual process)
        if not self.start_data_logger():
            return False
        
        time.sleep(1)  # Give logger time to start
        
        # Start gym server second
        if not self.start_gym_server():
            self.cleanup()
            return False
        
        time.sleep(2)  # Give server time to start
        
        # Start training last
        training_process = self.start_training()
        if not training_process:
            self.cleanup()
            return False
        
        # Monitor training output
        try:
            while self.running and training_process.poll() is None:
                line = training_process.stdout.readline()
                if line:
                    print(line.rstrip())
                
                # Check if other processes are still running
                for name, process in self.processes:
                    if process != training_process and process.poll() is not None:
                        print(f"⚠ {name} process died unexpectedly")
                        self.cleanup()
                        return False
            
            if training_process.poll() == 0:
                print(" Training completed successfully")
            else:
                print("Training failed")
                
        except KeyboardInterrupt:
            print("\n⚠️ Training interrupted by user")
        finally:
            self.cleanup()
        
        return True

if __name__ == "__main__":
    manager = TrainingManager()
    success = manager.run()
    sys.exit(0 if success else 1)
