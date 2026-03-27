#!/usr/bin/env python3
"""
Debug ZMQ communication
"""

import zmq
import time
import json
import threading

def test_subscriber():
    """Test subscriber side"""
    print("Starting subscriber...")
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.bind("tcp://127.0.0.1:10202")  # Bind instead of connect
    socket.setsockopt_string(zmq.SUBSCRIBE, "training_update")
    socket.setsockopt_string(zmq.SUBSCRIBE, "episode")
    
    print("Subscriber ready, waiting for messages...")
    
    try:
        while True:
            try:
                message = socket.recv_string(zmq.NOBLOCK)
                print(f"📨 Received: {message}")
                parts = message.split(' ', 1)
                if len(parts) == 2:
                    data_type, data_json = parts
                    print(f"   Type: {data_type}")
                    print(f"   Data: {data_json}")
            except zmq.Again:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nSubscriber stopped")

def test_publisher():
    """Test publisher side"""
    print("Starting publisher...")
    time.sleep(1)  # Let subscriber start
    
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.connect("tcp://127.0.0.1:10202")
    
    print("Publisher ready, sending test messages...")
    
    # Send training update
    training_data = {
        "update": 1,
        "total_frames": 3200,
        "fps": 3500,
        "average_reward": 45.2
    }
    
    message = "training_update " + json.dumps(training_data)
    print(f"Sending: {message}")
    socket.send_string(message)
    
    time.sleep(0.5)
    
    # Send episode data
    episode_data = {
        "episode": 1,
        "reward": 78.5,
        "length": 234
    }
    
    message = "episode " + json.dumps(episode_data)
    print(f"Sending: {message}")
    socket.send_string(message)
    
    print("Test completed")

if __name__ == "__main__":
    # Start subscriber in background
    import threading
    sub_thread = threading.Thread(target=test_subscriber, daemon=True)
    sub_thread.start()
    
    # Start publisher
    test_publisher()
    
    # Wait a bit to see results
    time.sleep(2)
    print("Debug test completed")
