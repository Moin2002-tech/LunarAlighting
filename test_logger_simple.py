#!/usr/bin/env python3
"""
Simple test of data logger
"""

import subprocess
import time
import json
from pathlib import Path

def test_data_logger():
    """Test data logger with manual input"""
    print("Testing Data Logger Standalone")
    print("=" * 40)
    
    # Start data logger
    process = subprocess.Popen([
        "python3", "data_logger.py", 
        "--output", "simple_test.json"
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    print("📊 Data logger started, waiting 2 seconds...")
    time.sleep(2)
    
    # Check if it's running
    if process.poll() is None:
        print("Data logger is running")
        
        # Send test data using separate process
        test_process = subprocess.run([
            "python3", "-c", """
import zmq
import json
import time

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.connect("tcp://127.0.0.1:10202")

# Send test message
message = "training_update " + json.dumps({
    "update": 1,
    "total_frames": 3200,
    "fps": 3500,
    "average_reward": 45.2
})
socket.send_string(message)
print("Test message sent")
"""
        ], capture_output=True, text=True)
        
        print("📤 Test output:", test_process.stdout)
        if test_process.stderr:
            print("Test error:", test_process.stderr)
        
        # Wait for processing
        time.sleep(2)
        
        # Check output
        if Path("simple_test.json").exists():
            with open("simple_test.json", 'r') as f:
                data = json.load(f)
            print(f"Data collected: {len(data['training_metrics'])} updates, {len(data['episodes'])} episodes")
            if data['training_metrics']:
                print("SUCCESS! Real-time data collection working!")
            else:
                print("No training metrics collected")
        else:
            print("No output file created")
    else:
        # Get any output
        output, _ = process.communicate()
        print("Data logger failed to start:")
        print(output)
    
    # Clean up
    process.terminate()
    process.wait(timeout=5)

if __name__ == "__main__":
    test_data_logger()
