#!/usr/bin/env python3
"""
Test script that simulates exactly what the C++ client does
"""
import sys
import os
import time

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import msgpack
import zmq

def simulate_cpp_client():
    print("Simulating C++ client behavior...")
    
    # Connect to server (same as C++ Communicator)
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.connect("tcp://127.0.0.1:10201")
    socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout like C++
    
    try:
        # Step 1: Send make request (exactly like C++)
        print("1. Sending make request for LunarAlighting-v1...")
        make_request = {
            "method": "make", 
            "param": {
                "envName": "LunarAlighting-v1",  # C++ field name
                "numEnv": 1  # C++ field name
            }
        }
        
        # Serialize and send (like C++ sendRequest)
        packed_request = msgpack.packb(make_request)
        socket.send(packed_request)
        
        # Receive response (like C++ getResponse)
        print("   Waiting for make response...")
        packed_response = socket.recv()
        make_response = msgpack.unpackb(packed_response, raw=False)
        print(f"   Make response: {make_response}")
        
        if "result" not in make_response:
            print("   ERROR: Invalid make response format!")
            return False
        
        if make_response["result"].startswith("ERROR"):
            print(f"   ERROR: Server returned error: {make_response['result']}")
            return False
        
        print("   Make request: SUCCESS")
        
        # Step 2: Send info request (like C++)
        print("2. Sending info request...")
        info_request = {"method": "info", "param": {}}
        socket.send(msgpack.packb(info_request))
        
        print("   Waiting for info response...")
        packed_response = socket.recv()
        info_response = msgpack.unpackb(packed_response, raw=False)
        print(f"   Info response: {info_response}")
        
        # Validate info response structure
        required_fields = ["action_space_type", "action_space_shape", "observation_space_type", "observation_space_shape"]
        for field in required_fields:
            if field not in info_response:
                print(f"   ERROR: Missing field '{field}' in info response!")
                return False
        
        print("   Info request: SUCCESS")
        
        # Step 3: Send reset request (like C++)
        print("3. Sending reset request...")
        reset_request = {"method": "reset", "param": {}}
        socket.send(msgpack.packb(reset_request))
        
        print("   Waiting for reset response...")
        packed_response = socket.recv()
        reset_response = msgpack.unpackb(packed_response, raw=False)
        print(f"   Reset response: {reset_response}")
        
        if "observation" not in reset_response:
            print("   ERROR: Missing 'observation' in reset response!")
            return False
        
        obs = reset_response["observation"]
        if len(obs) != 8:
            print(f"   ERROR: Expected 8 observation values, got {len(obs)}!")
            return False
        
        print("   Reset request: SUCCESS")
        
        # Step 4: Send step request (like C++)
        print("4. Sending step request...")
        step_request = {
            "method": "step", 
            "param": {
                "action": [[2]],  # C++ field name, nested array for batch
                "render": False
            }
        }
        socket.send(msgpack.packb(step_request))
        
        print("   Waiting for step response...")
        packed_response = socket.recv()
        step_response = msgpack.unpackb(packed_response, raw=False)
        print(f"   Step response: {step_response}")
        
        # Validate step response
        required_fields = ["observation", "reward", "done", "real_reward"]
        for field in required_fields:
            if field not in step_response:
                print(f"   ERROR: Missing field '{field}' in step response!")
                return False
        
        print("   Step request: SUCCESS")
        
        print("\n✅ All C++ client simulations PASSED!")
        print("The server is ready for the C++ client!")
        return True
        
    except zmq.error.Again:
        print("❌ TIMEOUT: Server did not respond within 5 seconds")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False
    finally:
        socket.close()
        context.term()

if __name__ == "__main__":
    print("Starting C++ client simulation...")
    print("Make sure the server is running: .venv/bin/python start_gym_server.py")
    print()
    
    success = simulate_cpp_client()
    sys.exit(0 if success else 1)
