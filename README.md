# Traffic Intersection Simulator with AI Control

A multi-level traffic simulation built with Python and Pygame, designed for Reinforcement Learning training. Features a unified AI that learns to control traffic lights across different intersection configurations.

## Quick Start

```bash
# Install dependencies
pip install pygame numpy torch

# Run the interactive demo
python demo.py

# Train the AI on all levels
python train_unified.py --episodes 300

# Run demo with trained AI
python demo.py
```

## Project Structure

```
AGH-Traffic-Simulation/
    simulation.py      # Core simulation engine
    levels.py          # Multi-level system (Level 1, 2, 3)
    demo.py            # Interactive demo with sliders
    train_unified.py   # Unified AI training script
    README.md          # This file
```

## The Three Levels

### Level 1: Single Intersection
```
        B
        |
   A ---+--- C
        |
        D
```
- 4 entry points (A, B, C, D)
- 8 traffic light actions
- State size: 20

### Level 2: Two Connected Intersections
```
        B1      B2
        |       |
   A ---+-------+--- C
        |       |
        D1      D2
```
- 6 entry points
- 8 traffic light actions
- Limited capacity connecting road
- State size: 28

### Level 3: 2x2 Grid (4 Intersections)
```
   N1      N2
    |       |
W1--+---+---+--E1
    |   |   |
    +---+---+
    |   |   |
W2--+---+---+--E2
    |       |
   S1      S2
```
- 8 entry points
- 16 traffic light actions
- Complex traffic routing
- State size: 56

## Interactive Demo

```bash
python demo.py                  # Start at Level 1
python demo.py --level 2        # Start at Level 2
python demo.py --level 3        # Start at Level 3
```

### Controls

| Key | Action |
|-----|--------|
| **L** | Switch level (1 -> 2 -> 3 -> 1) |
| **R** | Reset all sliders to 50% |
| **M** | Max all sliders (stress test) |
| **Z** | Zero all sliders (stop spawning) |
| **ESC** | Exit |
| **Mouse** | Drag sliders to adjust spawn rates |

### Display Panel

- **Sliders**: Control spawn rate for each entry point (0-100%)
- **AI Status**: Shows "AI: ACTIVE" (trained model) or "AI: RANDOM" (no model)
- **Current Action**: What the AI just decided
- **Metrics**: Vehicle count, waiting count, throughput, average wait time

## Training the AI

### Unified Training (Recommended)

Trains a single model that works on ALL levels:

```bash
# Quick training (~10 min)
python train_unified.py --episodes 300

# Full training (~30 min, better results)
python train_unified.py --episodes 1000

# Test the trained model
python train_unified.py --test
```

The training:
1. Cycles through Level 1, 2, 3 each episode
2. Randomizes spawn rates for variety
3. Saves best model as `dqn_unified_best.pth`

### How Training Works

```
Episode 1: Level 1 with random spawn rates
Episode 2: Level 2 with random spawn rates
Episode 3: Level 3 with random spawn rates
Episode 4: Level 1 with different spawn rates
...
```

The AI learns to:
- Minimize vehicle wait times
- Maximize throughput
- Avoid creating gridlock
- Adapt to different traffic patterns

### Model Files

| File | Description |
|------|-------------|
| `dqn_unified_best.pth` | Best model during training |
| `dqn_unified_final.pth` | Final model after training |

## How the AI Works

### State Vector

The AI receives a normalized vector describing the current traffic situation:

```python
state = simulator.get_state()
# Contains:
# - Queue pressure per lane (how many cars waiting)
# - Throughput per direction (cars that passed through)
# - Signal status (which lights are green)
# - Arrow status (turn arrows)
```

### Actions

Each action sets a specific traffic light configuration:

**Level 1/2 (8 actions):**
- Action 0: East-West green
- Action 1: East-West green + arrows
- Action 2: North-South green
- Action 3: North-South green + arrows
- Actions 4-7: Various combinations

**Level 3 (16 actions):**
- Independent control of all 4 intersections
- Allows coordinated "green waves"

### Reward Function

The AI is rewarded for:
- Reducing average wait time
- Increasing throughput
- Keeping queues short

And penalized for:
- Long wait times
- Switching signals too frequently
- Creating gridlock

## Configuration

### Simulation Parameters

Edit `simulation.py` to adjust:

```python
SIMULATION_SPEED = 3.0      # Speed multiplier (higher = faster)
SPAWN_INTERVAL = 1.5        # Seconds between spawn attempts
LANES_PER_DIRECTION = 2     # Lanes per road direction
VEHICLE_SPEED = 3.0         # How fast cars move
```

### Training Parameters

Edit `train_unified.py` to adjust:

```python
GAMMA = 0.99               # Discount factor
LEARNING_RATE = 0.0005     # Neural network learning rate
BATCH_SIZE = 64            # Training batch size
EPSILON_DECAY = 0.997      # Exploration decay rate
```

## API Reference

### TrafficSimulator

```python
from simulation import TrafficSimulator

# Create simulator at specific level
sim = TrafficSimulator(level=1)  # or 2, or 3

# Get state for AI
state = sim.get_state()  # Returns normalized float list

# Get metrics for reward calculation
metrics = sim.get_metrics()
# Returns: {
#   'total_wait_time': int,
#   'avg_wait_time': float,
#   'total_throughput': int,
#   'queue_lengths': dict,
#   'waiting_vehicles': int,
#   'total_vehicles': int
# }

# Apply an action (changes traffic lights)
action_name = sim.apply_level_action(action_id)

# Advance simulation by one frame
sim.update()

# Switch to different level (clears all vehicles)
sim.switch_level(2)

# Get level info
info = sim.get_level_info()
# Returns: {'level': 1, 'name': '...', 'state_size': 20, 'action_count': 8}
```

### Level System

```python
from levels import Level1, Level2, Level3

# Each level defines:
level = Level1()
level.get_state_size()      # Size of state vector
level.get_action_count()    # Number of possible actions
level.action_names          # Human-readable action names
level.apply_action(0)       # Apply action to traffic lights
```

## Troubleshooting

### "AI: RANDOM" instead of "AI: ACTIVE"

No trained model found. Train one first:
```bash
python train_unified.py --episodes 300
```

### Cars not spawning

Check that sliders aren't at 0%. Press **R** to reset to 50%.

### Level 3 seems empty

Level 3 has 8 entry points with spawn rates distributed across them. Press **M** to maximize all spawn rates for stress testing.

### Training is slow

- Use GPU if available (automatically detected)
- Reduce `--episodes` for quick tests
- The simulation runs headless during training

## Requirements

- Python 3.8+
- pygame
- numpy
- torch (PyTorch)

```bash
pip install pygame numpy torch
```

For GPU training (NVIDIA):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## License

Educational project for AGH University - use freely for learning and research.
