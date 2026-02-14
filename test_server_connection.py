#!/usr/bin/env python3
"""
Test script to verify the Gym Server is working correctly
"""
import sys
import os
import time

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from GymServer.zmqServer import ZmqClient
from GymServer.messages import MakeMessage, InfoMessage, ResetMessage
import msgpack

def test_server():
    print("Testing Gym Server connection...")
    
    # Connect to server as client
    context = __import__('zmq').Context()
    socket = context.socket(__import__('zmq').PAIR)
    socket.connect("tcp://127.0.0.1:10201")
    
    try:
        # Test make request with C++ field names
        print("Sending 'make' request...")
        make_request = {"method": "make", "param": {"envName": "LunarAlighting-v1", "numEnv": 1}}
        socket.send(msgpack.packb(make_request))
        
        # Wait for response
        response = socket.recv()
        unpacked = msgpack.unpackb(response, raw=False)
        print(f"Make response: {unpacked}")
        
        # Test info request
        print("Sending 'info' request...")
        info_request = {"method": "info", "param": {}}
        socket.send(msgpack.packb(info_request))
        
        response = socket.recv()
        unpacked = msgpack.unpackb(response, raw=False)
        print(f"Info response: {unpacked}")
        
        print("Server is working correctly!")
        return True
        
    except Exception as e:
        print(f" Server test failed: {e}")
        return False
    finally:
        socket.close()
        context.term()

if __name__ == "__main__":
    success = test_server()
    sys.exit(0 if success else 1)
