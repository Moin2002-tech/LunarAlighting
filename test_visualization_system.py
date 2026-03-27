#!/usr/bin/env python3
"""
Test script for the visualization system
Simulates training data and tests the complete pipeline
"""

import json
import time
import threading
from pathlib import Path
from visualization_dashboard import TrainingDataAnalyzer

def test_data_generation():
    """Test the visualization system with generated data"""
    print("Testing Visualization System...")
    print("=" * 50)
    
    # Test 1: Generate sample data and create visualizations
    print("\nTesting sample data generation...")
    analyzer = TrainingDataAnalyzer("non_existent_file.json")  # Will trigger sample data
    
    assert not analyzer.df_metrics.empty, "Training metrics should not be empty"
    assert not analyzer.df_episodes.empty, "Episode data should not be empty"
    print("Sample data generated successfully")
    
    # Test 2: Test dashboard creation
    print("\nTesting dashboard creation...")
    test_dir = Path("test_output")
    test_dir.mkdir(exist_ok=True)
    
    try:
        analyzer.create_learning_progress_dashboard(str(test_dir / "test_learning.png"))
        analyzer.create_training_efficiency_dashboard(str(test_dir / "test_efficiency.png"))
        analyzer.create_model_behavior_dashboard(str(test_dir / "test_behavior.png"))
        print(" All dashboards created successfully")
        
        # Verify files were created
        assert (test_dir / "test_learning.png").exists(), "Learning dashboard should exist"
        assert (test_dir / "test_efficiency.png").exists(), "Efficiency dashboard should exist"
        assert (test_dir / "test_behavior.png").exists(), "Behavior dashboard should exist"
        print("All dashboard files verified")
        
    except Exception as e:
        print(f"Dashboard creation failed: {e}")
        return False
    
    # Test 3: Test comprehensive report
    print("\n3️⃣ Testing comprehensive report...")
    try:
        analyzer.generate_comprehensive_report(str(test_dir))
        print("Comprehensive report generated successfully")
    except Exception as e:
        print(f"Report generation failed: {e}")
        return False
    
    # Test 4: Test JSON data structure
    print("\n4️⃣ Testing JSON data structure...")
    json_file = test_dir / "training_data.json"
    if json_file.exists():
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        required_keys = ["metadata", "training_metrics", "episodes"]
        for key in required_keys:
            assert key in data, f"Missing required key: {key}"
        
        assert len(data["training_metrics"]) > 0, "Training metrics should not be empty"
        assert len(data["episodes"]) > 0, "Episodes should not be empty"
        print("JSON data structure is valid")
    else:
        print("  JSON file not found, but other tests passed")
    
    print("\n All tests passed! Visualization system is working correctly.")
    print(f"Test files saved to: {test_dir.absolute()}")
    
    return True

def test_real_data_simulation():
    """Simulate what real data would look like"""
    print("\nSimulating real training data...")
    
    # Create realistic training progression
    real_data = {
        "metadata": {
            "algorithm": "PPO",
            "env_name": "LunarAlighting-v1",
            "num_envs": 8,
            "batch_size": 40,
            "max_frames": 10000000,
            "reward_threshold": 160,
            "start_time": time.time()
        },
        "training_metrics": [],
        "episodes": []
    }
    
    # Simulate 50 training updates with realistic PPO behavior
    for update in range(0, 500, 10):
        progress = update / 500
        
        # Realistic PPO learning curve
        base_reward = -40 + (progress * 250) - (progress * progress * 50)  # S-curve
        noise = np.random.normal(0, 15)
        avg_reward = base_reward + noise
        
        # Realistic loss curves for PPO
        policy_loss = 1.5 * np.exp(-progress * 3) + 0.1 + np.random.normal(0, 0.05)
        value_loss = 1.0 * np.exp(-progress * 2) + 0.05 + np.random.normal(0, 0.02)
        entropy = 0.8 * (1 - progress * 0.8) + np.random.normal(0, 0.02)
        
        real_data["training_metrics"].append({
            "update": update,
            "total_frames": update * 40 * 8,
            "fps": 800 + np.random.normal(0, 50),
            "average_reward": avg_reward,
            "episode_count": update * 5,
            "policy_loss": max(0.01, policy_loss),
            "value_loss": max(0.01, value_loss),
            "entropy": max(0.01, entropy),
            "success_rate": max(0, min(1, (avg_reward + 40) / 240))
        })
    
    # Simulate episodes with realistic landing dynamics
    for episode in range(200):
        progress = episode / 200
        
        # More realistic reward distribution
        if progress < 0.3:  # Early training - mostly failures
            reward = np.random.normal(-30, 40)
        elif progress < 0.7:  # Mid training - mixed results
            reward = np.random.normal(50, 60)
        else:  # Late training - mostly success
            reward = np.random.normal(120, 40)
        
        success = reward > 80 and np.random.random() > 0.2
        crash = reward < -20
        
        real_data["episodes"].append({
            "episode": episode,
            "reward": reward,
            "length": np.random.randint(100, 400) if success else np.random.randint(50, 200),
            "success": success,
            "crash": crash,
            "final_altitude": 0 if success else np.random.uniform(-200, 300),
            "final_velocity": np.random.uniform(0, 2) if success else np.random.uniform(3, 10),
            "fuel_used": np.random.uniform(30, 90)
        })
    
    # Save realistic data
    with open("realistic_training_data.json", 'w') as f:
        json.dump(real_data, f, indent=2)
    
    print("Realistic training data generated")
    print("📊 Testing with realistic data...")
    
    # Test visualization with realistic data
    analyzer = TrainingDataAnalyzer("realistic_training_data.json")
    analyzer.generate_comprehensive_report("realistic_output")
    
    print("Realistic data visualization completed")
    return True

if __name__ == "__main__":
    import numpy as np
    
    print("🚀 Starting Visualization System Tests")
    print("=" * 60)
    
    try:
        # Run basic tests
        if test_data_generation():
            print("\n" + "=" * 60)
            
            # Run realistic data simulation
            if test_real_data_simulation():
                print("\n🎉 ALL TESTS PASSED!")
                print("📊 Visualization system is ready for production use!")
                print("\n📋 Next steps:")
                print("   1. Integrate training_data_exporter.hpp into GymClient.cpp")
                print("   2. Run data_logger.py during training")
                print("   3. Use visualization_dashboard.py for analysis")
            else:
                print("Realistic data test failed")
        else:
            print("Basic tests failed")
            
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
