#!/usr/bin/env python3
"""
Script to start the Gym Server for the Lunar Lander RL simulation
"""
import logging
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the environment to register it
from GymServer.lunar_alighting_env import LunarAlightingEnv

from GymServer.zmqServer import ZmqClient
from GymServer.server import Server

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create ZMQ client on port 10201 (same as C++ client expects)
    zmq_client = ZmqClient(port=10201)
    
    # Create and start server
    server = Server(zmq_client)
    
    logging.info("Starting Gym Server on port 10201...")
    logging.info("Waiting for C++ client to connect...")
    
    try:
        server.serve()
    except KeyboardInterrupt:
        logging.info("Server stopped by user")
    except Exception as e:
        logging.error(f"Server error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
