#!/usr/bin/env python3
"""
Test script to demonstrate the real-time data collection workflow
"""

import subprocess
import time
import threading
import signal
import sys
import json
from pathlib import Path

def start_data_logger():
    """Start data logger in background"""
    print("🚀 Starting data logger...")
    process = subprocess.Popen([
        sys.executable, "data_logger.py", 
        "--output", "realtime_test_data.json"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return process

def send_test_data():
    """Send test data to simulate GymClient"""
    try:
        import zmq
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.connect("tcp://127.0.0.1:10202")
        
        print("📊 Sending test training data...")
        
        # Send training update
        training_data = {
            "update": 1,
            "total_frames": 3200,
            "fps": 3500,
            "average_reward": 45.2,
            "episode_count": 10,
            "policy_loss": 1.8,
            "value_loss": 0.9,
            "entropy": 0.4,
            "success_rate": 0.3
        }
        
        message = "training_update " + json.dumps(training_data)
        socket.send_string(message)
        time.sleep(0.1)
        
        # Send episode data
        episode_data = {
            "episode": 1,
            "reward": 78.5,
            "length": 234,
            "success": True,
            "crash": False,
            "fuel_used": 76.6
        }
        
        message = "episode " + json.dumps(episode_data)
        socket.send_string(message)
        time.sleep(0.1)
        
        print("✅ Test data sent successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error sending test data: {e}")
        return False

def check_data_collection():
    """Check if data was collected"""
    time.sleep(2)  # Wait for data logger to process
    
    if Path("realtime_test_data.json").exists():
        with open("realtime_test_data.json", 'r') as f:
            data = json.load(f)
        
        print(f"📁 Data file created with {len(data['training_metrics'])} training updates and {len(data['episodes'])} episodes")
        
        if data['training_metrics'] or data['episodes']:
            print("✅ Real-time data collection working!")
            return True
        else:
            print("❌ No data collected")
            return False
    else:
        print("❌ No data file created")
        return False

def main():
    print("🧪 Testing Real-Time Data Collection Workflow")
    print("=" * 50)
    
    # Start data logger
    logger_process = start_data_logger()
    time.sleep(1)  # Let logger start
    
    try:
        # Send test data
        if send_test_data():
            # Check collection
            if check_data_collection():
                print("\n🎉 Real-time workflow test PASSED!")
                print("\n📋 For actual training:")
                print("   1. Start: python3 data_logger.py")
                print("   2. Run: ./LunarAlightingRL") 
                print("   3. Watch: python3 visualization_dashboard.py --json-file training_data.json")
            else:
                print("\n❌ Data collection failed")
        else:
            print("\n❌ Failed to send test data")
            
    finally:
        # Clean up
        logger_process.terminate()
        logger_process.wait(timeout=5)
        print("\n🧹 Test completed, data logger stopped")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
