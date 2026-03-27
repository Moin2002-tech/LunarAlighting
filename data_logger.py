#!/usr/bin/env python3
"""
Data Logger for Lunar Alighting RL Training
Captures training data from GymClient and saves to JSON for visualization
"""

import json
import time
import threading
from datetime import datetime
from typing import Dict, List, Any
import zmq
from dataclasses import dataclass, asdict

@dataclass
class EpisodeData:
    """Data structure for individual episode metrics"""
    episode: int
    reward: float
    length: int
    success: bool
    crash: bool
    final_altitude: float
    final_velocity: float
    fuel_used: float
    timestamp: float

@dataclass
class TrainingMetrics:
    """Data structure for training update metrics"""
    update: int
    total_frames: int
    fps: float
    average_reward: float
    episode_count: int
    policy_loss: float
    value_loss: float
    entropy: float
    success_rate: float
    timestamp: float

class TrainingDataLogger:
    """Logs training data from GymClient to JSON format"""
    
    def __init__(self, output_file: str = "training_data.json"):
        self.output_file = output_file
        self.training_data = {
            "metadata": {
                "algorithm": "PPO",
                "env_name": "LunarAlighting-v1",
                "num_envs": 8,
                "batch_size": 40,
                "max_frames": 10000000,
                "reward_threshold": 160,
                "start_time": datetime.now().isoformat()
            },
            "training_metrics": [],
            "episodes": []
        }
        self.current_episode = 0
        self.episode_start_time = time.time()
        self.lock = threading.Lock()
        
    def log_training_update(self, update_data: Dict[str, Any]):
        """Log training metrics from update step"""
        with self.lock:
            metrics = TrainingMetrics(
                update=update_data.get('update', 0),
                total_frames=update_data.get('total_frames', 0),
                fps=update_data.get('fps', 0),
                average_reward=update_data.get('average_reward', 0),
                episode_count=update_data.get('episode_count', 0),
                policy_loss=update_data.get('policy_loss', 0),
                value_loss=update_data.get('value_loss', 0),
                entropy=update_data.get('entropy', 0),
                success_rate=update_data.get('success_rate', 0),
                timestamp=time.time()
            )
            self.training_data["training_metrics"].append(asdict(metrics))
    
    def log_episode(self, episode_data: Dict[str, Any]):
        """Log individual episode data"""
        with self.lock:
            episode = EpisodeData(
                episode=episode_data.get('episode', self.current_episode),
                reward=episode_data.get('reward', 0),
                length=episode_data.get('length', 0),
                success=episode_data.get('success', False),
                crash=episode_data.get('crash', False),
                final_altitude=episode_data.get('final_altitude', 0),
                final_velocity=episode_data.get('final_velocity', 0),
                fuel_used=episode_data.get('fuel_used', 0),
                timestamp=time.time()
            )
            self.training_data["episodes"].append(asdict(episode))
            self.current_episode += 1
    
    def save_data(self):
        """Save current data to JSON file"""
        with self.lock:
            self.training_data["metadata"]["last_update"] = datetime.now().isoformat()
            with open(self.output_file, 'w') as f:
                json.dump(self.training_data, f, indent=2)
    
    def auto_save(self, interval_seconds: int = 60):
        """Auto-save data at regular intervals"""
        while True:
            time.sleep(interval_seconds)
            try:
                self.save_data()
                print(f"✅ Training data auto-saved to {self.output_file}")
            except Exception as e:
                print(f"❌ Error saving data: {e}")

class ZMQDataCollector:
    """Collects training data from GymClient via ZMQ"""
    
    def __init__(self, logger: TrainingDataLogger, 
                 gym_endpoint: str = "tcp://127.0.0.1:10202"):
        self.logger = logger
        self.gym_endpoint = gym_endpoint
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.bind(gym_endpoint)  # SUB binds to receive data
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "training_update")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "episode")
        
    def start_collecting(self):
        """Start collecting data from GymClient"""
        print(f"🔗 Connecting to GymClient at {self.gym_endpoint}")
        print(f"📊 Logging training data to {self.logger.output_file}")
        
        # Start auto-save thread
        auto_save_thread = threading.Thread(
            target=self.logger.auto_save, 
            args=(30,),  # Save every 30 seconds
            daemon=True
        )
        auto_save_thread.start()
        
        try:
            while True:
                try:
                    message = self.socket.recv_string(zmq.NOBLOCK)
                    parts = message.split(' ', 1)
                    if len(parts) == 2:
                        data_type, data_json = parts
                        data = json.loads(data_json)
                        
                        if data_type == "training_update":
                            self.logger.log_training_update(data)
                        elif data_type == "episode":
                            self.logger.log_episode(data)
                            
                except zmq.Again:
                    time.sleep(0.1)  # No message available
                except json.JSONDecodeError as e:
                    print(f"❌ Error parsing JSON: {e}")
                except Exception as e:
                    print(f"❌ Error collecting data: {e}")
                    
        except KeyboardInterrupt:
            print("\n🛑 Stopping data collection...")
            self.logger.save_data()
            print(f"✅ Final data saved to {self.logger.output_file}")

def main():
    """Main function to run the data logger"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Log Lunar Alighting RL Training Data')
    parser.add_argument('--output', '-o', type=str, 
                       default='training_data.json',
                       help='Output JSON file for training data')
    parser.add_argument('--endpoint', '-e', type=str,
                       default='tcp://127.0.0.1:10202',
                       help='ZMQ endpoint for GymClient communication')
    
    args = parser.parse_args()
    
    # Initialize logger and collector
    logger = TrainingDataLogger(args.output)
    collector = ZMQDataCollector(logger, args.endpoint)
    
    print("🚀 Starting Lunar Alighting RL Data Logger")
    print("=" * 50)
    
    # Start collecting data
    collector.start_collecting()

if __name__ == "__main__":
    main()
