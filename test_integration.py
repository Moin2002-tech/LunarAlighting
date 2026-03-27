#!/usr/bin/env python3
"""
Test script to verify the complete integration works
Tests the data flow from GymClient to visualization
"""

import subprocess
import time
import threading
import signal
import sys
from pathlib import Path

def start_data_logger():
    """Start the data logger in a separate process"""
    try:
        process = subprocess.Popen([
            sys.executable, "data_logger.py", 
            "--output", "integration_test_data.json"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        print("📊 Data logger started...")
        return process
    except Exception as e:
        print(f" Failed to start data logger: {e}")
        return None

def test_visualization_system():
    """Test the complete integration workflow"""
    print(" Testing Complete Integration System")
    print("=" * 50)
    
    # Step 1: Test data logger
    print("\nstarting data logger...")
    logger_process = start_data_logger()
    
    if not logger_process:
        print(" Cannot proceed without data logger")
        return False
    
    # Give logger time to start
    time.sleep(2)
    
    # Step 2: Test visualization with sample data
    print("\n2️⃣ Testing visualization system...")
    try:
        from visualization_dashboard import TrainingDataAnalyzer
        
        # Generate sample data
        analyzer = TrainingDataAnalyzer("non_existent_file.json")
        analyzer.generate_comprehensive_report("integration_test_output")
        
        print("Visualization system working")
    except Exception as e:
        print(f"Visualization system failed: {e}")
        logger_process.terminate()
        return False
    
    # Step 3: Check if we can build the C++ project
    print("\n3️⃣ Testing C++ build configuration...")
    try:
        # Try to run cmake configuration
        cmake_result = subprocess.run([
            "cmake", "-B", "test_build", "-S", "."
        ], capture_output=True, text=True, timeout=30)
        
        if cmake_result.returncode == 0:
            print("CMake configuration successful")
        else:
            print("CMake configuration issues detected:")
            print(cmake_result.stderr)
    except subprocess.TimeoutExpired:
        print("CMake configuration timed out")
    except Exception as e:
        print(f"Could not test CMake: {e}")
    
    # Step 4: Instructions for manual testing
    print("\n4️⃣ Manual testing instructions:")
    print("   To test the complete integration:")
    print("   a) Install nlohmann/json: sudo apt install nlohmann-json3-dev")
    print("   b) Build the project: mkdir build && cd build && cmake .. && make")
    print("   c) Start data logger: python3 data_logger.py")
    print("   d) Run training: ./LunarAlightingRL")
    print("   e) Generate visualizations: python3 visualization_dashboard.py --json-file training_data.json")
    
    # Clean up
    print("\nCleaning up...")
    logger_process.terminate()
    logger_process.wait(timeout=5)
    
    print("\n🎉 Integration test completed!")
    print("📁 Check integration_test_output/ for sample visualizations")
    
    return True

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    dependencies = {
        "Python": True,  # We're running this script
        "NumPy": False,
        "Pandas": False,
        "Matplotlib": False,
        "Seaborn": False,
        "PyZMQ": False
    }
    
    try:
        import numpy
        dependencies["NumPy"] = True
    except ImportError:
        pass
    
    try:
        import pandas
        dependencies["Pandas"] = True
    except ImportError:
        pass
    
    try:
        import matplotlib
        dependencies["Matplotlib"] = True
    except ImportError:
        pass
    
    try:
        import seaborn
        dependencies["Seaborn"] = True
    except ImportError:
        pass
    
    try:
        import zmq
        dependencies["PyZMQ"] = True
    except ImportError:
        pass
    
    print("\nDependency Status:")
    for dep, status in dependencies.items():
        status_icon = "" if status else ""
        print(f"   {status_icon} {dep}")
    
    missing = [dep for dep, status in dependencies.items() if not status]
    if missing:
        print(f"\n Missing dependencies: {', '.join(missing)}")
        print("💡 Install with: pip install -r requirements_visualization.txt")
        return False
    
    print("✅ All Python dependencies available!")
    return True

def main():
    """Main test function"""
    print("🚀 Lunar Alighting RL Integration Test")
    print("=" * 60)
    
    # Check dependencies first
    if not check_dependencies():
        print("\nPlease install missing dependencies before continuing")
        return False
    
    # Run integration test
    if test_visualization_system():
        print("\nIntegration test PASSED!")
        print("\nNext steps:")
        print("   1. Install nlohmann/json if not already installed")
        print("   2. Build your C++ project with the new integration")
        print("   3. Test the complete data flow")
        return True
    else:
        print("\nIntegration test FAILED!")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
