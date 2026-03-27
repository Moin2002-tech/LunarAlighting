# Lunar Alighting RL Visualization Dashboard

A comprehensive visualization system for analyzing the performance of the Lunar Alighting Reinforcement Learning project.

## 🎯 What It Visualizes

### 1. Learning Progress Dashboard
- **Reward over time**: Line plot showing average reward per episode
- **Success rate**: Percentage of episodes meeting reward threshold  
- **Training stability**: Rolling average with confidence intervals
- **Learning curve comparison**: Compare current vs previous training runs

### 2. Training Efficiency Dashboard  
- **FPS over time**: Performance during training
- **Sample efficiency**: Rewards per thousand frames
- **Training loss curves**: Actor loss, value loss, entropy
- **Gradient metrics**: Available from update_data

### 3. Model Behavior Analysis Dashboard
- **Landing success rate**: Over time and by performance threshold
- **Crash analysis**: Episodes ending in failure vs success
- **Action patterns**: Sequential action analysis
- **State-action correlations**: What states trigger which actions

## 🚀 Quick Start

### Option 1: Generate Sample Visualizations (No Data Required)

```bash
# Install dependencies
pip install -r requirements_visualization.txt

# Generate sample data and visualizations
python visualization_dashboard.py

# This will create:
# - learning_progress.png
# - training_efficiency.png  
# - model_behavior.png
# - analysis_output/ directory with all plots
```

### Option 2: Use Real Training Data

#### Step 1: Modify GymClient.cpp to Export Data

Add this to your `GymClient.cpp`:

```cpp
#include "training_data_exporter.hpp"

// In main() function, after algorithm initialization:
TrainingDataExporter exporter("tcp://127.0.0.1:10201");

// In the training loop (around line 315-334), after logging:
if (update % logInterval == 0 && update > 0) {
    // ... existing logging code ...
    
    // Export training data
    TrainingUpdateData update_data{
        update,
        static_cast<int>(total_steps),
        static_cast<float>(fps),
        average_reward,
        episode_count,
        0.0f, // policy_loss (extract from update_data)
        0.0f, // value_loss (extract from update_data)  
        0.0f, // entropy (extract from update_data)
        average_reward >= renderRewardThresHold ? 1.0f : 0.0f
    };
    exporter.send_training_update(update_data);
}

// For episode data (when episode ends):
EpisodeData episode_data{
    episode_count,
    running_rewards[i],
    episode_length,
    success,
    crash,
    final_altitude,
    final_velocity,
    fuel_used
};
exporter.send_episode_data(episode_data);
```

#### Step 2: Start the Data Logger

```bash
# Terminal 1: Start data logger
python data_logger.py --output training_data.json

# Terminal 2: Run your GymClient
./LunarAlightingRL  # Or however you run your C++ client
```

#### Step 3: Generate Visualizations

```bash
# After training (or during training for live updates)
python visualization_dashboard.py --json-file training_data.json --output-dir analysis_results
```

## 📊 File Structure

```
LunarAlightingRL/
├── visualization_dashboard.py    # Main visualization script
├── data_logger.py               # Real-time data collection
├── training_data_exporter.hpp   # C++ header for data export
├── requirements_visualization.txt # Python dependencies
├── VISUALIZATION_README.md      # This file
└── training_data.json           # Generated training data
```

## 🛠️ Customization

### Custom Metrics

Add new metrics by modifying the `TrainingUpdateData` and `EpisodeData` structures:

```cpp
// In training_data_exporter.hpp
struct TrainingUpdateData {
    // ... existing fields ...
    float custom_metric;  // Add your custom metric
};
```

### Custom Visualizations

Extend the `TrainingDataAnalyzer` class in `visualization_dashboard.py`:

```python
def create_custom_dashboard(self, save_path: str = "custom_analysis.png"):
    """Create custom analysis dashboard"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    # Add your custom plots here
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
```

## 📈 Understanding the Plots

### Learning Progress Dashboard

- **Reward Over Time**: Shows how the model's performance improves during training
- **Success Rate**: Percentage of successful landings over time
- **Training Stability**: Rolling average with standard deviation bands
- **Learning Curves**: Compare different training runs or hyperparameters

### Training Efficiency Dashboard

- **FPS Performance**: Monitor computational efficiency during training
- **Sample Efficiency**: How well the model learns from each frame
- **Loss Curves**: Track optimization progress (lower is better)
- **Entropy**: Exploration vs exploitation balance

### Model Behavior Dashboard

- **Success Rate**: Landing success over episodes
- **Crash Analysis**: Failure patterns and frequencies
- **Action Distribution**: Which actions the model prefers
- **State-Action Correlations**: How different states affect decisions

## 🔧 Troubleshooting

### Common Issues

1. **No JSON data found**: The script automatically generates sample data for demonstration
2. **Missing dependencies**: Run `pip install -r requirements_visualization.txt`
3. **ZMQ connection errors**: Ensure the data logger is running before starting GymClient
4. **Empty plots**: Check that your JSON file contains the required data structure

### Data Format Requirements

Your JSON file should have this structure:

```json
{
  "metadata": {
    "algorithm": "PPO",
    "env_name": "LunarAlighting-v1",
    "reward_threshold": 160
  },
  "training_metrics": [
    {
      "update": 0,
      "average_reward": -50.0,
      "fps": 1000,
      "policy_loss": 2.0,
      "value_loss": 1.5,
      "entropy": 0.5
    }
  ],
  "episodes": [
    {
      "episode": 0,
      "reward": -30.0,
      "length": 200,
      "success": false,
      "crash": true
    }
  ]
}
```

## 🚀 Advanced Usage

### Real-time Visualization

For live monitoring during training:

```bash
# Terminal 1: Start logger with auto-save
python data_logger.py --output live_training.json

# Terminal 2: Run GymClient with data export
./LunarAlightingRL

# Terminal 3: Generate live visualizations
watch -n 30 "python visualization_dashboard.py --json-file live_training.json"
```

### Batch Analysis

Analyze multiple training runs:

```python
# In visualization_dashboard.py
def compare_training_runs(json_files: List[str]):
    """Compare multiple training runs"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for file_path in json_files:
        analyzer = TrainingDataAnalyzer(file_path)
        ax.plot(analyzer.df_metrics['update'], 
                analyzer.df_metrics['average_reward'],
                label=Path(file_path).stem)
    
    ax.legend()
    plt.show()
```

## 📚 Dependencies

- **Python 3.7+**
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation  
- **Matplotlib**: Plotting
- **Seaborn**: Statistical visualization
- **PyZMQ**: Real-time data communication
- **nlohmann/json**: JSON handling in C++ (for GymClient integration)

## 🤝 Contributing

To add new visualizations:

1. Add data collection in `training_data_exporter.hpp`
2. Update the data logger in `data_logger.py`
3. Create new visualization methods in `visualization_dashboard.py`
4. Update this README with documentation

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify your data format matches the expected structure
3. Ensure all dependencies are installed
4. Check that ZMQ endpoints match between logger and exporter
