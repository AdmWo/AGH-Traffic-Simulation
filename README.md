# Advanced Traffic Intersection Simulator

A highly configurable, object-oriented traffic simulation built with Python and Pygame, designed specifically for neural network training and traffic optimization research.

## Overview

This simulator models a realistic 4-way intersection with right-hand traffic rules (Polish/European style). Unlike basic traffic simulators, it features:

- **Object-oriented collision detection** using Lane and Intersection objects (no pixel-based hacks)
- **Fully configurable signal phases** with custom timing and turn arrow control
- **Realistic vehicle behavior** including right turns, left turns (yielding to oncoming traffic), and straight-through movements
- **Lane-specific rules** for turn restrictions
- **Adjustable simulation speed** for rapid data collection

## Architecture

### Core Components

```
Lane Objects
├─ Track vehicles in ordered lists
├─ Provide vehicle-ahead queries
└─ Manage lane membership

Intersection Object
├─ Conflict detection via logical rules
├─ Track vehicles currently in intersection
└─ Grant/deny entry based on path conflicts

Vehicle Objects
├─ State: direction, lane, turn intention
├─ Behavior: signal compliance, yielding logic
└─ Movement: position updates, turn execution
```

### Configuration System

All simulation parameters are exposed as constants at the top of `simulation.py`:

#### Display Settings
- `SCREEN_WIDTH`, `SCREEN_HEIGHT`: Window dimensions
- `FPS`: Rendering frame rate (60)
- `SIMULATION_SPEED`: Time multiplier (2.0 = 2x speed)

#### Road Configuration
- `LANES_PER_DIRECTION`: Number of lanes per direction (2)
- `LANE_WIDTH`: Width in pixels (40)
- `RIGHT_TURN_LANES`: How many rightmost lanes can turn right (1)
- `LEFT_TURN_LANES`: How many leftmost lanes can turn left (1)

#### Vehicle Behavior
- `VEHICLE_SPEED`: Uniform speed for all vehicles (3.0)
- `TURN_RIGHT_PROBABILITY`: 30% of vehicles turn right
- `TURN_LEFT_PROBABILITY`: 40% of vehicles turn left
- `SPAWN_INTERVAL`: Time between spawns (1.0 seconds)

#### Signal Phases
```python
SIGNAL_PHASES = [
    {
        'duration': 20,           # Green time in seconds
        'green': ['right', 'left'],  # A & C get green
        'arrows': ['down']           # B gets right turn arrow
    },
    {
        'duration': 15,
        'green': ['down', 'up'],     # B & D get green
        'arrows': ['left']           # C gets right turn arrow
    }
]
```

## Vehicle System

### Visual Types
Four vehicle types with different sizes and colors:
- **Cars** (blue, 28px): 40% spawn rate
- **Trucks** (brown, 31px): 25% spawn rate
- **Bikes** (green, 26px): 20% spawn rate
- **Buses** (yellow, 33px): 15% spawn rate

All vehicles move at the same speed (configurable via `VEHICLE_SPEED`).

### Turn Logic

Vehicles are assigned destinations on spawn:
- Each vehicle displays a letter (A/B/C/D) indicating its destination
- Destination excludes the source lane
- Turn type determines the path: straight, right turn, or left turn

**Right Turns:**
- Only allowed from rightmost lanes
- Yield to traffic from destination going straight
- Can turn on dedicated green arrow (when enabled)

**Left Turns:**
- Only allowed from leftmost lanes  
- Yield to oncoming traffic (straight + right turns)
- Require main green signal

## Controls

- **Keys 1-4**: Toggle turn arrows for lanes A-D (for testing)
- **Close Window**: Exit simulation

## For Neural Network Training

### Why This Simulator?

1. **Deterministic**: Lane/Intersection objects provide predictable state
2. **Observable**: All vehicle and signal states are accessible
3. **Configurable**: Easily adjust difficulty via signal phases and spawn rates
4. **Fast**: `SIMULATION_SPEED` parameter allows rapid data collection

### State Space

Available data for each frame:
- Lane occupancy (ordered vehicle lists per lane)
- Intersection occupancy (vehicles currently crossing)
- Signal states (green/yellow/red) for all directions
- Turn arrow states
- Vehicle properties: position, direction, destination, waiting state

### Action Space

Potential actions for RL agents:
- Set signal phase durations
- Enable/disable turn arrows
- Adjust yellow/red timing
- Control spawn rates

## Quick Start

### Installation

```bash
# Install dependencies
pip install pygame

# Run simulation
python simulation.py
```

### Basic Configuration

Edit the top of `simulation.py`:

```python
# Speed up 10x for training
SIMULATION_SPEED = 10.0

# Only straight traffic (no turns)
TURN_RIGHT_PROBABILITY = 0.0
TURN_LEFT_PROBABILITY = 0.0

# Simple two-phase system
SIGNAL_PHASES = [
    {'duration': 15, 'green': ['right', 'left'], 'arrows': []},
    {'duration': 15, 'green': ['down', 'up'], 'arrows': []}
]
```

## 🛠️ Technical Details

### Collision Detection

**Not pixel-based!** Uses logical conflict rules:

```python
conflicts[('right', 'right')] = [('up', 'straight')]
# A turning right conflicts with D going straight
```

All intersection conflicts are pre-defined in the `Intersection` class conflict map.

### Turn Execution

Vehicles change direction and lane at the intersection center:
- **Right turns**: Early execution, tight corner
- **Left turns**: Late execution, wide arc through intersection
- **Lane preservation**: nth source lane → nth destination lane

## Statistics Tracking

Real-time display shows:
- Vehicles crossed per lane (A/B/C/D)
- Current signal states and timers
- Active turn arrows
- Destination legend

## License

Educational project - use freely for research and learning.

## Academic Context

Designed for training neural networks to optimize traffic light timing. The simulation provides:
- Ground truth vehicle state
- Configurable complexity via signal phases
- Fast iteration via simulation speed multiplier
- Reproducible scenarios via deterministic logic

Perfect for reinforcement learning experiments in traffic optimization!
