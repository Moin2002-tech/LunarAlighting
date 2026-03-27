#!/usr/bin/env python3
"""
Data Logger for Lunar Alighting RL Training
Captures training data from C++ TrainingDataExporter and saves to JSON
"""

import json
import time
import threading
from datetime import datetime
from typing import Dict, Any
import zmq
from dataclasses import dataclass, asdict


@dataclass
class EpisodeData:
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
                "start_time": datetime.now().isoformat(),
                "last_update": datetime.now().isoformat(),
            },
            "training_metrics": [],
            "episodes": [],
        }
        self.lock = threading.Lock()
        # Write empty-but-valid file immediately so it's never missing
        self.save_data()

    def log_training_update(self, data: Dict[str, Any]):
        with self.lock:
            metrics = TrainingMetrics(
                update=data.get("update", 0),
                total_frames=data.get("total_frames", 0),
                fps=data.get("fps", 0.0),
                average_reward=data.get("average_reward", 0.0),
                episode_count=data.get("episode_count", 0),
                policy_loss=data.get("policy_loss", 0.0),
                value_loss=data.get("value_loss", 0.0),
                entropy=data.get("entropy", 0.0),
                success_rate=data.get("success_rate", 0.0),
                timestamp=time.time(),
            )
            self.training_data["training_metrics"].append(asdict(metrics))

    def log_episode(self, data: Dict[str, Any]):
        with self.lock:
            episode = EpisodeData(
                episode=data.get("episode", 0),
                reward=data.get("reward", 0.0),
                length=data.get("length", 0),
                success=data.get("success", False),
                crash=data.get("crash", False),
                final_altitude=data.get("final_altitude", 0.0),
                final_velocity=data.get("final_velocity", 0.0),
                fuel_used=data.get("fuel_used", 0.0),
                timestamp=time.time(),
            )
            self.training_data["episodes"].append(asdict(episode))

    def save_data(self):
        with self.lock:
            self.training_data["metadata"]["last_update"] = datetime.now().isoformat()
            with open(self.output_file, "w") as f:
                json.dump(self.training_data, f, indent=2)

    def auto_save(self, interval_seconds: int = 10):
        while True:
            time.sleep(interval_seconds)
            try:
                self.save_data()
                with self.lock:
                    n_metrics = len(self.training_data["training_metrics"])
                    n_episodes = len(self.training_data["episodes"])
                print(f" Auto-saved — {n_metrics} updates, {n_episodes} episodes")
            except Exception as e:
                print(f"Auto-save error: {e}")


class ZMQDataCollector:
    def __init__(self, logger: TrainingDataLogger,
                 endpoint: str = "tcp://127.0.0.1:10202"):
        self.logger = logger
        self.endpoint = endpoint
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)  # ← PULL, not SUB
        self.socket.bind(endpoint)                   # ← logger binds, C++ connects
        # No SUBSCRIBE filter — PULL receives every message

    def start_collecting(self):
        print(f"📡 Listening on {self.endpoint}")
        print(f"📊 Writing to {self.logger.output_file}")

        threading.Thread(
            target=self.logger.auto_save, args=(30,), daemon=True
        ).start()

        try:
            while True:
                try:
                    message = self.socket.recv_string()  # blocking — no sleep/spin
                    topic, _, payload = message.partition(" ")

                    if not payload:
                        print(f" Malformed message (no payload): {message!r}")
                        continue

                    data = json.loads(payload)

                    if topic == "training_update":
                        self.logger.log_training_update(data)
                        avg_reward = data.get('average_reward', 0) or 0
                        fps = data.get('fps', 0) or 0
                        print(f"📈 Update {data.get('update', 0):>5} | "
                              f"reward={avg_reward:>8.2f} | "
                              f"fps={fps:>6.0f}")
                    elif topic == "episode":
                        self.logger.log_episode(data)
                        status = "✓" if data.get("success") else "✗"
                        reward = data.get('reward', 0) or 0
                        print(f"🏁 Episode {data.get('episode', 0):>4} | "
                              f"reward={reward:>8.2f} | {status}")
                    else:
                        print(f"⚠️  Unknown topic: {topic!r}")

                except json.JSONDecodeError as e:
                    print(f" JSON error: {e} — raw: {message!r}")

        except KeyboardInterrupt:
            print("\nStopping — saving final data...")
            self.logger.save_data()
            with self.logger.lock:
                n_m = len(self.logger.training_data["training_metrics"])
                n_e = len(self.logger.training_data["episodes"])
            print(f" Saved {n_m} training updates and {n_e} episodes "
                  f"→ {self.logger.output_file}")
        finally:
            self.socket.close()
            self.context.term()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Lunar Alighting RL Data Logger")
    parser.add_argument("--output", "-o", default="training_data.json")
    parser.add_argument("--endpoint", "-e", default="tcp://127.0.0.1:10202")
    args = parser.parse_args()

    logger = TrainingDataLogger(args.output)
    collector = ZMQDataCollector(logger, args.endpoint)

    print(" Lunar Alighting RL Data Logger")
    print("=" * 40)
    collector.start_collecting()


if __name__ == "__main__":
    main()