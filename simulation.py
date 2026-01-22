import pygame
import random
import threading
import time
from collections import deque

# Import the level system
try:
    from levels import Level, Level1, Level2, Level3, get_level_by_number, RoadSegment
    LEVELS_AVAILABLE = True
except ImportError:
    LEVELS_AVAILABLE = False
    print("Warning: levels.py not found, running in single-intersection mode only")

"""
Traffic Intersection Simulator

A pygame-based traffic simulation built for reinforcement learning experiments.
The whole thing is designed to be easy to hook up to an RL agent.

What you get:
- Configurable lanes (just change LANES_PER_DIRECTION and everything adjusts)
- Multiple levels: single intersection, dual intersection, 4-way grid
- A get_state() method that gives you a nice normalized vector for your neural net
- Per-lane spawn probabilities if you want asymmetric traffic patterns
- Polish-style right turn arrows (Zielona Strzalka) that work independently of main lights
- All rendering done with pygame.draw, no external images needed

The state vector size depends on the level:
  Level 1: 20 values (single intersection)
  Level 2: ~20 values (2 intersections + connecting segment)
  Level 3: ~40 values (4 intersections + 4 connecting segments)

Each vehicle tracks its own wait_time so you can penalize sitting in traffic.
Collision detection uses lane-based logic, not pixel scanning.
Vehicles check segment capacity before entering connecting roads.

Quick usage:
  sim = TrafficSimulator(level=1)  # or 2 or 3
  state = sim.get_state()     # normalized floats 0-1
  metrics = sim.get_metrics() # detailed stats for reward calculation
  sim.update()                # advance one frame
  
Press L to cycle through levels during simulation.
"""

# ------------------------------------------------------------------
# Configuration - tweak these to change how the simulation behaves
# ------------------------------------------------------------------
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 800
FPS = 60

# Speed multiplier - crank this up during training to collect data faster
# 1.0 = realtime, 3.0 = 3x faster, etc.
SIMULATION_SPEED = 3.0

# Road geometry - change LANES_PER_DIRECTION and everything scales automatically
LANES_PER_DIRECTION = 2
LANE_WIDTH = 40
ROAD_COLOR = (50, 50, 50)
LANE_LINE_COLOR = (200, 200, 200)
STOP_LINE_COLOR = (255, 255, 0)

# Which lanes can make which turns
# Lane 0 is always the rightmost (outer) lane
RIGHT_TURN_LANES = 1  # how many rightmost lanes can turn right
LEFT_TURN_LANES = 1   # how many leftmost lanes can turn left

# Vehicle settings
VEHICLE_SPEED = 3.0

VEHICLE_TYPES = {
    'car': {'color': (0, 120, 255), 'size': 28, 'spawn_weight': 40},
    'bus': {'color': (255, 200, 0), 'size': 33, 'spawn_weight': 15},
    'truck': {'color': (150, 75, 0), 'size': 31, 'spawn_weight': 25},
    'bike': {'color': (0, 200, 100), 'size': 26, 'spawn_weight': 20}
}
SAFE_DISTANCE = 15  # gap between cars
TURN_RIGHT_PROBABILITY = 0.3
TURN_LEFT_PROBABILITY = 0.4  # remaining 30% go straight
TURN_CLEAR_DISTANCE = 100

# Traffic light timing
SIGNAL_RADIUS = 20
SIGNAL_YELLOW_TIME = 3
SIGNAL_RED_TIME = 2  # brief all-red for safety

# Signal phases - defines the light cycle
# 'green' = which directions get the green light
# 'arrows' = which directions get their turn arrow lit up
# Directions: right=A, down=B, left=C, up=D
SIGNAL_PHASES = [
    {'duration': 20, 'green': ['right', 'left'], 'arrows': ['down']},
    {'duration': 15, 'green': ['down', 'up'], 'arrows': ['left']}
]

# How often new cars spawn (lower = more frequent spawn checks)
SPAWN_INTERVAL = 0.05  # Check every 50ms for fast spawning

# Per-lane spawn probabilities - lets you create asymmetric traffic patterns
# Format: 'Direction_Lane_X' where X is the lane number (0 = rightmost)
LANE_SPAWN_PROBABILITIES = {
    'A_Lane_0': 0.8, 'A_Lane_1': 0.6,
    'B_Lane_0': 0.7, 'B_Lane_1': 0.5,
    'C_Lane_0': 0.8, 'C_Lane_1': 0.6,
    'D_Lane_0': 0.7, 'D_Lane_1': 0.5,
}

# Direction names mapped to labels shown on screen
# A=West, B=North, C=East, D=South
LANE_LABELS = {
    'right': 'A',
    'down': 'B',
    'left': 'C',
    'up': 'D'
}

# Where each turn type takes you from each direction
DESTINATION_MAP = {
    'right': {'straight': 'C', 'right': 'D', 'left': 'B'},
    'down': {'straight': 'D', 'right': 'A', 'left': 'C'},
    'left': {'straight': 'A', 'right': 'B', 'left': 'D'},
    'up': {'straight': 'B', 'right': 'C', 'left': 'A'}
}


# ------------------------------------------------------------------
# Lane - keeps track of vehicles in a single lane
# ------------------------------------------------------------------
class Lane:
    """A single lane going one direction. Tracks vehicles in order."""
    def __init__(self, direction, lane_number):
        self.direction = direction
        self.lane_number = lane_number
        self.vehicles = []
    
    def add_vehicle(self, vehicle):
        if vehicle not in self.vehicles:
            self.vehicles.append(vehicle)
    
    def remove_vehicle(self, vehicle):
        if vehicle in self.vehicles:
            self.vehicles.remove(vehicle)
    
    def get_vehicle_ahead(self, vehicle):
        """Returns the car directly in front of this one, or None if first in line."""
        try:
            idx = self.vehicles.index(vehicle)
            if idx > 0:
                return self.vehicles[idx - 1]
        except ValueError:
            pass
        return None


# ------------------------------------------------------------------
# Intersection - handles conflict detection for turning vehicles
# ------------------------------------------------------------------
class Intersection:
    """Tracks which vehicles are in the intersection and whether paths conflict."""
    def __init__(self):
        self.vehicles_in_intersection = set()
        self.conflict_map = self._build_conflict_map()
    
    def _build_conflict_map(self):
        """Builds a lookup table of which paths conflict with each other.
        For example, a car turning left conflicts with oncoming traffic going straight."""
        conflicts = {
            # Right turns conflict with traffic from the destination direction going straight
            ('right', 'right'): [('up', 'straight')],
            ('down', 'right'): [('right', 'straight')],
            ('left', 'right'): [('down', 'straight')],
            ('up', 'right'): [('left', 'straight')],
            # Left turns conflict with oncoming traffic
            ('right', 'left'): [('left', 'straight'), ('left', 'right')],
            ('down', 'left'): [('up', 'straight'), ('up', 'right')],
            ('left', 'left'): [('right', 'straight'), ('right', 'right')],
            ('up', 'left'): [('down', 'straight'), ('down', 'right')],
            # Going straight has no conflicts (lane logic handles same-direction)
            ('right', 'straight'): [],
            ('down', 'straight'): [],
            ('left', 'straight'): [],
            ('up', 'straight'): [],
        }
        return conflicts
    
    def can_enter(self, vehicle):
        """Check if this vehicle's path conflicts with anyone already in the intersection."""
        my_path = (vehicle.original_direction, vehicle.turn_type)
        conflicting_paths = self.conflict_map.get(my_path, [])
        
        # Check if any vehicle in intersection has a conflicting path
        for other in self.vehicles_in_intersection:
            other_path = (other.original_direction, other.turn_type)
            if other_path in conflicting_paths:
                return False
        
        return True
    
    def enter(self, vehicle):
        self.vehicles_in_intersection.add(vehicle)
    
    def exit(self, vehicle):
        self.vehicles_in_intersection.discard(vehicle)
    
    def is_vehicle_in_intersection(self, vehicle):
        return vehicle in self.vehicles_in_intersection


# ------------------------------------------------------------------
# TrafficSignal - one traffic light with optional turn arrow
# ------------------------------------------------------------------
class TrafficSignal:
    """A single traffic light. Has a main light and an optional turn arrow."""
    def __init__(self, direction, position):
        self.direction = direction
        self.position = position
        self.state = 'red'
        self.timer = 0
        self.turn_arrow_enabled = False
        self.turn_arrow_state = 'red'
        
    def get_color(self):
        colors = {'green': (0, 255, 0), 'yellow': (255, 255, 0), 'red': (255, 0, 0)}
        return colors.get(self.state, (255, 0, 0))
    
    def get_turn_color(self):
        return (0, 255, 0) if self.turn_arrow_state == 'green' else (255, 0, 0)
    
    def can_go(self):
        return self.state == 'green'
    
    def can_turn(self):
        """Turn arrow lets you turn even on red, if the arrow itself is green."""
        if self.turn_arrow_enabled:
            return self.turn_arrow_state == 'green'
        return self.state == 'green'


# ------------------------------------------------------------------
# LevelSignalProxy - wraps level intersection signals to work with Vehicle.should_stop()
# ------------------------------------------------------------------
class LevelSignalProxy:
    """A simple proxy that provides can_go()/can_turn() based on level intersection states."""
    
    def __init__(self, is_green, has_arrow):
        self._is_green = is_green
        self._has_arrow = has_arrow
    
    def can_go(self):
        return self._is_green
    
    def can_turn(self):
        # If main signal is green, can turn. If arrow is on, can also turn.
        return self._is_green or self._has_arrow


# ------------------------------------------------------------------
# Vehicle - a single car/bus/truck/bike in the simulation
# ------------------------------------------------------------------
class Vehicle:
    """A vehicle that moves through the intersection. Tracks its own wait time."""
    def __init__(self, v_type, direction, lane, level=1, spawn_point=None, 
                 intersection_x=None, spawn_x=None, spawn_y=None):
        self.type = v_type
        self.original_direction = direction  # where we entered from (for signal checks)
        self.direction = direction  # current heading (changes when we turn)
        self.lane = lane
        self.color = VEHICLE_TYPES[v_type]['color']
        self.size = VEHICLE_TYPES[v_type]['size']
        self.speed = VEHICLE_SPEED * SIMULATION_SPEED
        
        # Level-specific info
        self.level = level
        self.spawn_point = spawn_point
        self.intersection_x = intersection_x  # for Level 2
        self.spawn_x_override = spawn_x  # for Level 3
        self.spawn_y_override = spawn_y  # for Level 3
        
        # figure out what turns we're allowed to make from this lane
        can_turn_right = lane < RIGHT_TURN_LANES
        can_turn_left = lane >= (LANES_PER_DIRECTION - LEFT_TURN_LANES)
        
        # randomly pick a destination based on probabilities
        rand = random.random()
        if can_turn_right and rand < TURN_RIGHT_PROBABILITY:
            self.turn_type = 'right'
        elif can_turn_left and rand >= TURN_RIGHT_PROBABILITY and rand < TURN_RIGHT_PROBABILITY + TURN_LEFT_PROBABILITY:
            self.turn_type = 'left'
        else:
            self.turn_type = 'straight'
        
        self.has_turned = False
        self.destination = DESTINATION_MAP[direction][self.turn_type]
        self.destination_lane = self.lane
        self.x, self.y = self._get_spawn_position()
        
        # state tracking
        self.crossed = False
        self.in_intersection = False
        self.current_lane = None
        self.waiting_at_signal = False
        
        # for RL - counts frames spent sitting still
        self.wait_time = 0
        self.last_position = (self.x, self.y)
        
        # for multi-intersection levels - track which segment we're on
        self.current_segment = None
        self.target_segment = None  # segment we're trying to enter
        
    def _get_spawn_position(self):
        """Figure out where to spawn based on direction, lane, and level."""
        road_width = LANES_PER_DIRECTION * LANE_WIDTH
        
        if self.level == 1:
            return self._get_level1_spawn_position(road_width)
        elif self.level == 2:
            return self._get_level2_spawn_position(road_width)
        elif self.level == 3:
            return self._get_level3_spawn_position(road_width)
        else:
            return self._get_level1_spawn_position(road_width)
    
    def _get_level1_spawn_position(self, road_width):
        """Spawn position for Level 1 (center intersection)."""
        center_x = SCREEN_WIDTH // 2
        center_y = SCREEN_HEIGHT // 2
        
        if self.original_direction == 'right':
            x = -self.size - 10
            y = center_y + road_width - self.lane * LANE_WIDTH - LANE_WIDTH // 2
            return x, y
        elif self.original_direction == 'down':
            x = center_x - road_width + self.lane * LANE_WIDTH + LANE_WIDTH // 2
            y = -self.size - 10
            return x, y
        elif self.original_direction == 'left':
            x = SCREEN_WIDTH + 10
            y = center_y - road_width + self.lane * LANE_WIDTH + LANE_WIDTH // 2
            return x, y
        else:  # up
            x = center_x + road_width - self.lane * LANE_WIDTH - LANE_WIDTH // 2
            y = SCREEN_HEIGHT + 10
            return x, y
    
    def _get_level2_spawn_position(self, road_width):
        """Spawn position for Level 2 (two intersections)."""
        center_y = SCREEN_HEIGHT // 2
        int_x = self.intersection_x if self.intersection_x else 400
        
        if self.original_direction == 'right':
            # Coming from West (spawn point A)
            x = -self.size - 10
            y = center_y + road_width - self.lane * LANE_WIDTH - LANE_WIDTH // 2
            return x, y
        elif self.original_direction == 'left':
            # Coming from East (spawn point C)
            x = SCREEN_WIDTH + 10
            y = center_y - road_width + self.lane * LANE_WIDTH + LANE_WIDTH // 2
            return x, y
        elif self.original_direction == 'down':
            # Coming from North (B1 or B2)
            x = int_x - road_width + self.lane * LANE_WIDTH + LANE_WIDTH // 2
            y = -self.size - 10
            return x, y
        else:  # up
            # Coming from South (D1 or D2)
            x = int_x + road_width - self.lane * LANE_WIDTH - LANE_WIDTH // 2
            y = SCREEN_HEIGHT + 10
            return x, y
    
    def _get_level3_spawn_position(self, road_width):
        """Spawn position for Level 3 (2x2 grid)."""
        # Use override positions if provided
        base_x = self.spawn_x_override if self.spawn_x_override is not None else 0
        base_y = self.spawn_y_override if self.spawn_y_override is not None else 0
        
        # Adjust for lane within the road
        if self.original_direction == 'right':
            # Coming from West, horizontal road
            x = base_x - self.size
            y = base_y + road_width - self.lane * LANE_WIDTH - LANE_WIDTH // 2
            return x, y
        elif self.original_direction == 'left':
            # Coming from East, horizontal road
            x = base_x + self.size
            y = base_y - road_width + self.lane * LANE_WIDTH + LANE_WIDTH // 2
            return x, y
        elif self.original_direction == 'down':
            # Coming from North, vertical road
            x = base_x - road_width + self.lane * LANE_WIDTH + LANE_WIDTH // 2
            y = base_y - self.size
            return x, y
        else:  # up
            # Coming from South, vertical road
            x = base_x + road_width - self.lane * LANE_WIDTH - LANE_WIDTH // 2
            y = base_y + self.size
            return x, y
    
    def get_stop_position(self):
        """Where this vehicle should stop if the light is red."""
        road_width = LANES_PER_DIRECTION * LANE_WIDTH
        
        if self.level == 1:
            center_x = SCREEN_WIDTH // 2
            center_y = SCREEN_HEIGHT // 2
            if self.original_direction == 'right':
                return center_x - road_width - 20
            elif self.original_direction == 'down':
                return center_y - road_width - 20
            elif self.original_direction == 'left':
                return center_x + road_width + 20
            else:  # up
                return center_y + road_width + 20
        
        elif self.level == 2:
            center_y = SCREEN_HEIGHT // 2
            int_x = self.intersection_x if self.intersection_x else 400
            if self.original_direction == 'right':
                return int_x - road_width - 20
            elif self.original_direction == 'down':
                return center_y - road_width - 20
            elif self.original_direction == 'left':
                return int_x + road_width + 20
            else:  # up
                return center_y + road_width + 20
        
        elif self.level == 3:
            # For level 3, stop position depends on which entry point
            # Get the nearest intersection position
            left_x, right_x = 400, 1000
            top_y, bottom_y = 280, 520
            
            if self.original_direction == 'right':
                return left_x - road_width - 20
            elif self.original_direction == 'left':
                return right_x + road_width + 20
            elif self.original_direction == 'down':
                return top_y - road_width - 20
            else:  # up
                return bottom_y + road_width + 20
        
        # Default fallback
        center = SCREEN_WIDTH // 2 if self.original_direction in ['right', 'left'] else SCREEN_HEIGHT // 2
        return center - road_width - 20 if self.original_direction in ['right', 'down'] else center + road_width + 20
    
    def is_before_stop_line(self):
        """Are we still approaching the intersection?"""
        stop_pos = self.get_stop_position()
        
        if self.original_direction == 'right':
            return self.x < stop_pos
        elif self.original_direction == 'down':
            return self.y < stop_pos
        elif self.original_direction == 'left':
            return self.x > stop_pos
        else:  # up
            return self.y > stop_pos
    
    def should_stop(self, signal, intersection):
        """Decide whether to stop - checks signals, cars ahead, intersection conflicts, and segment capacity."""
        # once we're in the intersection, keep going
        if self.in_intersection:
            return False
        
        # Only check signals if we haven't passed the stop line yet
        if self.is_before_stop_line():
            stop_pos = self.get_stop_position()
            
            # Check if we're approaching the stop line
            approaching_stop = False
            if self.original_direction == 'right':
                approaching_stop = self.x + self.speed >= stop_pos
            elif self.original_direction == 'down':
                approaching_stop = self.y + self.speed >= stop_pos
            elif self.original_direction == 'left':
                approaching_stop = self.x - self.speed <= stop_pos
            else:  # up
                approaching_stop = self.y - self.speed <= stop_pos
            
            if approaching_stop:
                # Check signal permission
                if self.turn_type == 'right':
                    # Right turns: check turn arrow
                    if not signal.can_turn():
                        self.waiting_at_signal = True
                        return True
                elif self.turn_type == 'left':
                    # Left turns: need main signal
                    if not signal.can_go():
                        self.waiting_at_signal = True
                        return True
                else:
                    # Straight: need main signal
                    if not signal.can_go():
                        self.waiting_at_signal = True
                        return True
                
                # Signal allows us - check if intersection is clear
                if not intersection.can_enter(self):
                    return True  # Wait for conflicting traffic to clear
                
                # Check if target segment has capacity (for multi-intersection levels)
                if self.target_segment is not None:
                    if not self.target_segment.can_enter():
                        return True  # Wait, target segment is full
                
                self.waiting_at_signal = False
        
        # Check for vehicle ahead in same lane
        if self.current_lane:
            vehicle_ahead = self.current_lane.get_vehicle_ahead(self)
            if vehicle_ahead and self._is_too_close(vehicle_ahead):
                return True
        
        return False
    
    def _is_too_close(self, other):
        """Check if we're tailgating the car in front."""
        if self.direction == 'right':
            return other.x > self.x and other.x - self.x < SAFE_DISTANCE + self.size
        elif self.direction == 'down':
            return other.y > self.y and other.y - self.y < SAFE_DISTANCE + self.size
        elif self.direction == 'left':
            return other.x < self.x and self.x - other.x < SAFE_DISTANCE + self.size
        else:  # up
            return other.y < self.y and self.y - other.y < SAFE_DISTANCE + self.size
    
    def _execute_turn(self):
        """Actually make the turn - changes our heading and snaps to the new lane."""
        center_x = SCREEN_WIDTH // 2
        center_y = SCREEN_HEIGHT // 2
        road_width = LANES_PER_DIRECTION * LANE_WIDTH
        
        if self.turn_type == 'right':
            # Right turn
            if self.original_direction == 'right':
                self.direction = 'down'
                # Move to destination lane in west half
                self.x = center_x - road_width + self.destination_lane * LANE_WIDTH + LANE_WIDTH // 2
            elif self.original_direction == 'down':
                self.direction = 'left'
                # Move to destination lane in north half
                self.y = center_y - road_width + self.destination_lane * LANE_WIDTH + LANE_WIDTH // 2
            elif self.original_direction == 'left':
                self.direction = 'up'
                # Move to destination lane in east half
                self.x = center_x + road_width - self.destination_lane * LANE_WIDTH - LANE_WIDTH // 2
            else:  # up
                self.direction = 'right'
                # Move to destination lane in south half
                self.y = center_y + road_width - self.destination_lane * LANE_WIDTH - LANE_WIDTH // 2
        
        elif self.turn_type == 'left':
            # Left turn
            if self.original_direction == 'right':
                self.direction = 'up'
                # Move to destination lane in east half
                self.x = center_x + road_width - self.destination_lane * LANE_WIDTH - LANE_WIDTH // 2
            elif self.original_direction == 'down':
                self.direction = 'right'
                # Move to destination lane in south half
                self.y = center_y + road_width - self.destination_lane * LANE_WIDTH - LANE_WIDTH // 2
            elif self.original_direction == 'left':
                self.direction = 'down'
                # Move to destination lane in west half
                self.x = center_x - road_width + self.destination_lane * LANE_WIDTH + LANE_WIDTH // 2
            else:  # up
                self.direction = 'left'
                # Move to destination lane in north half
                self.y = center_y - road_width + self.destination_lane * LANE_WIDTH + LANE_WIDTH // 2
        
        self.has_turned = True
    
    def _is_at_turn_point(self):
        """Have we reached the spot where we should actually turn?
        Right turns happen early, left turns happen late (after crossing the intersection)."""
        center_x = SCREEN_WIDTH // 2
        center_y = SCREEN_HEIGHT // 2
        road_width = LANES_PER_DIRECTION * LANE_WIDTH
        
        if self.turn_type == 'right':
            # Right turn - turn early
            if self.original_direction == 'right':
                return self.x >= center_x - road_width // 2
            elif self.original_direction == 'down':
                return self.y >= center_y - road_width // 2
            elif self.original_direction == 'left':
                return self.x <= center_x + road_width // 2
            else:  # up
                return self.y <= center_y + road_width // 2
        
        elif self.turn_type == 'left':
            # Left turn - turn late (go to far side of intersection)
            if self.original_direction == 'right':
                return self.x >= center_x + road_width // 2
            elif self.original_direction == 'down':
                return self.y >= center_y + road_width // 2
            elif self.original_direction == 'left':
                return self.x <= center_x - road_width // 2
            else:  # up
                return self.y <= center_y - road_width // 2
        
        return False
    
    def move(self, signal, intersection, lanes):
        """Move the vehicle forward (or stop if we need to).
        Increments wait_time if we're stuck."""
        prev_x, prev_y = self.x, self.y
        
        if self.should_stop(signal, intersection):
            self.wait_time += 1
            return
        
        center_x = SCREEN_WIDTH // 2
        center_y = SCREEN_HEIGHT // 2
        road_width = LANES_PER_DIRECTION * LANE_WIDTH
        
        # Check if entering intersection
        if not self.in_intersection and not self.is_before_stop_line():
            self.in_intersection = True
            intersection.enter(self)
        
        # Check if turning vehicle should execute turn
        if self.turn_type != 'straight' and not self.has_turned and self._is_at_turn_point():
            # Remove from old lane
            if self.current_lane:
                self.current_lane.remove_vehicle(self)
            
            # Execute turn (changes direction and position)
            self._execute_turn()
            
            # Add to new lane
            new_lane_key = (self.direction, self.destination_lane)
            if new_lane_key in lanes:
                self.current_lane = lanes[new_lane_key]
                self.current_lane.add_vehicle(self)
        
        # Move based on current direction
        if self.direction == 'right':
            self.x += self.speed
            if not self.crossed and self.x > center_x + road_width:
                self.crossed = True
                if self.in_intersection:
                    intersection.exit(self)
                    self.in_intersection = False
        elif self.direction == 'down':
            self.y += self.speed
            if not self.crossed and self.y > center_y + road_width:
                self.crossed = True
                if self.in_intersection:
                    intersection.exit(self)
                    self.in_intersection = False
        elif self.direction == 'left':
            self.x -= self.speed
            if not self.crossed and self.x < center_x - road_width:
                self.crossed = True
                if self.in_intersection:
                    intersection.exit(self)
                    self.in_intersection = False
        else:  # up
            self.y -= self.speed
            if not self.crossed and self.y < center_y - road_width:
                self.crossed = True
                if self.in_intersection:
                    intersection.exit(self)
                    self.in_intersection = False
        
        # if we actually moved, reset the wait counter
        if abs(self.x - prev_x) > 0.1 or abs(self.y - prev_y) > 0.1:
            self.wait_time = 0
    
    def is_off_screen(self):
        """Has this vehicle left the visible area after having entered it?
        
        Vehicles spawn off-screen and drive in. We only want to remove them
        after they've driven THROUGH and exited on the other side.
        """
        # Track if we've ever been on screen
        if not hasattr(self, '_was_on_screen'):
            self._was_on_screen = False
        
        margin = 100  # Larger margin for Level 2/3 with bigger maps
        on_screen = (-margin <= self.x <= SCREEN_WIDTH + margin and
                     -margin <= self.y <= SCREEN_HEIGHT + margin)
        
        if on_screen:
            self._was_on_screen = True
            return False
        
        # Only count as "off screen" if we were on screen before
        return self._was_on_screen
    
    def draw(self, screen, font):
        """Draw the vehicle as a colored square with its destination letter on top."""
        rect = pygame.Rect(self.x - self.size // 2, self.y - self.size // 2,
                          self.size, self.size)
        pygame.draw.rect(screen, self.color, rect)
        pygame.draw.rect(screen, (0, 0, 0), rect, 2)  # border
        
        label = font.render(self.destination, True, (255, 255, 255))
        label_rect = label.get_rect(center=(self.x, self.y))
        screen.blit(label, label_rect)


# ------------------------------------------------------------------
# TrafficSimulator - the main class that runs everything
# ------------------------------------------------------------------
class TrafficSimulator:
    """The main simulator. Create one of these and call update() each frame.
    
    Set headless=True for training - skips display and background threads for speed.
    """
    def __init__(self, level: int = 1, headless: bool = False):
        self.headless = headless
        
        pygame.init()
        if headless:
            # Minimal display for headless mode (required by pygame)
            pygame.display.set_mode((1, 1), pygame.HIDDEN)
            self.screen = None
            self.clock = None
            self.font = None
            self.small_font = None
            self.vehicle_font = None
        else:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Traffic Intersection Simulator")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
            self.small_font = pygame.font.Font(None, 24)
            self.vehicle_font = pygame.font.Font(None, 20)
        
        # Level system - determines network topology
        self.current_level_num = level
        self.current_level = None
        if LEVELS_AVAILABLE:
            try:
                self.current_level = get_level_by_number(level)
            except ValueError:
                print(f"Level {level} not found, defaulting to Level 1")
                self.current_level = get_level_by_number(1)
                self.current_level_num = 1
        
        # set up traffic signals at each corner
        center_x = SCREEN_WIDTH // 2
        center_y = SCREEN_HEIGHT // 2
        road_width = LANES_PER_DIRECTION * LANE_WIDTH
        
        self.signals = {
            'right': TrafficSignal('right', (center_x - road_width - 60, center_y + road_width + 60)),
            'down': TrafficSignal('down', (center_x - road_width - 60, center_y - road_width - 60)),
            'left': TrafficSignal('left', (center_x + road_width + 60, center_y - road_width - 60)),
            'up': TrafficSignal('up', (center_x + road_width + 60, center_y + road_width + 60))
        }
        
        # set up lanes
        self.lanes = {}
        for direction in ['right', 'down', 'left', 'up']:
            for lane_num in range(LANES_PER_DIRECTION):
                lane_key = (direction, lane_num)
                self.lanes[lane_key] = Lane(direction, lane_num)
        
        self.intersection = Intersection()
        self.vehicles = []
        self.running = True
        self.vehicles_crossed = {'right': 0, 'down': 0, 'left': 0, 'up': 0}
        
        # AI control mode (if True, automatic signal cycling is disabled)
        self.ai_controlled = False
        
        # Background threads - SKIP in headless mode for faster training
        # In headless mode, spawn vehicles directly via _spawn_levelX_vehicle()
        if not headless:
            self.signal_thread = threading.Thread(target=self._signal_controller, daemon=True)
            self.spawn_thread = threading.Thread(target=self._vehicle_spawner, daemon=True)
            self.signal_thread.start()
            self.spawn_thread.start()
    
    def _signal_controller(self):
        """Runs in background, cycling through signal phases automatically.
        
        Works for all levels - updates both the classic self.signals and
        the level's intersection signals.
        """
        current_phase_idx = 0
        current_action_idx = 0
        all_directions = ['right', 'down', 'left', 'up']
        phase_timer = SIGNAL_PHASES[0]['duration']
        
        # start with everything red
        for direction in all_directions:
            self.signals[direction].state = 'red'
            self.signals[direction].turn_arrow_enabled = False
            self.signals[direction].turn_arrow_state = 'red'
        
        # Apply initial phase
        first_phase = SIGNAL_PHASES[0]
        for direction in first_phase['green']:
            self.signals[direction].state = 'green'
            self.signals[direction].timer = first_phase['duration']
            if direction in first_phase.get('arrows', []):
                self.signals[direction].turn_arrow_enabled = True
                self.signals[direction].turn_arrow_state = 'green'
        
        # Also update level signals if on Level 2 or 3
        if self.current_level:
            self.apply_level_action(0)
        
        while self.running:
            time.sleep(0.1)
            
            # Skip if AI is controlling (checked via ai_controlled flag)
            if getattr(self, 'ai_controlled', False):
                continue
            
            phase_timer -= 0.1 * SIMULATION_SPEED
            
            # For Level 1, use classic signal logic
            if self.current_level_num == 1:
                current_phase = SIGNAL_PHASES[current_phase_idx]
                green_directions = current_phase['green']
                
                for direction in green_directions:
                    self.signals[direction].timer -= 0.1 * SIMULATION_SPEED
                
                primary_signal = self.signals[green_directions[0]]
                
                if primary_signal.timer <= 0:
                    if primary_signal.state == 'green':
                        for direction in green_directions:
                            self.signals[direction].state = 'yellow'
                            self.signals[direction].timer = SIGNAL_YELLOW_TIME
                    elif primary_signal.state == 'yellow':
                        for direction in green_directions:
                            self.signals[direction].state = 'red'
                            self.signals[direction].timer = SIGNAL_RED_TIME
                            self.signals[direction].turn_arrow_state = 'red'
                    else:  # red phase is over
                        current_phase_idx = (current_phase_idx + 1) % len(SIGNAL_PHASES)
                        next_phase = SIGNAL_PHASES[current_phase_idx]
                        
                        for direction in all_directions:
                            self.signals[direction].turn_arrow_enabled = False
                            self.signals[direction].turn_arrow_state = 'red'
                        
                        for direction in next_phase['green']:
                            self.signals[direction].state = 'green'
                            self.signals[direction].timer = next_phase['duration']
                            if direction in next_phase.get('arrows', []):
                                self.signals[direction].turn_arrow_enabled = True
                                self.signals[direction].turn_arrow_state = 'green'
            
            # For Level 2 and 3, cycle through actions
            else:
                if phase_timer <= 0:
                    # Switch to next action
                    if self.current_level:
                        num_actions = self.current_level.get_action_count()
                        current_action_idx = (current_action_idx + 1) % num_actions
                        self.apply_level_action(current_action_idx)
                    phase_timer = 8.0  # 8 second phases for levels 2/3
    
    def _vehicle_spawner(self):
        """Runs in background, periodically adding new vehicles based on current level."""
        while self.running:
            spawn_time = max(0.01, SPAWN_INTERVAL / SIMULATION_SPEED)
            time.sleep(spawn_time)
            
            if self.current_level_num == 1:
                self._spawn_level1_vehicle()
            elif self.current_level_num == 2:
                self._spawn_level2_vehicle()
            elif self.current_level_num == 3:
                self._spawn_level3_vehicle()
            else:
                self._spawn_level1_vehicle()
    
    def _spawn_level1_vehicle(self):
        """Spawn vehicle(s) for Level 1 (single intersection)."""
        # Try each direction
        for direction in ['right', 'down', 'left', 'up']:
            lane_num = random.randint(0, LANES_PER_DIRECTION - 1)
            
            # Check spawn probability from level segments (for slider control)
            spawn_rate = 0.5
            if self.current_level:
                seg_id = f"entry_{direction}"
                seg = self.current_level.segments.get(seg_id)
                if seg:
                    spawn_rate = seg.spawn_rate
            
            # Spawn multiple cars if rate > 1 (rate of 2.0 = spawn 2 cars per check)
            num_to_spawn = int(spawn_rate) + (1 if random.random() < (spawn_rate % 1) else 0)
            
            for _ in range(num_to_spawn):
                # Create vehicle
                types = list(VEHICLE_TYPES.keys())
                weights = [VEHICLE_TYPES[t]['spawn_weight'] for t in types]
                v_type = random.choices(types, weights=weights)[0]
                
                lane_num = random.randint(0, LANES_PER_DIRECTION - 1)
                vehicle = Vehicle(v_type, direction, lane_num, level=1)
                
                lane_key = (direction, lane_num)
                if lane_key in self.lanes:
                    vehicle.current_lane = self.lanes[lane_key]
                    vehicle.current_lane.add_vehicle(vehicle)
                
                self.vehicles.append(vehicle)
    
    def _spawn_level2_vehicle(self):
        """Spawn vehicle(s) for Level 2 (two intersections)."""
        # Level 2 spawn points: A (west), B1 (north-left), D1 (south-left), 
        #                       C (east), B2 (north-right), D2 (south-right)
        spawn_points = ['A', 'B1', 'D1', 'C', 'B2', 'D2']
        
        # Map spawn point to direction and intersection
        spawn_info = {
            'A':  {'direction': 'right', 'int_x': 400},
            'B1': {'direction': 'down',  'int_x': 400},
            'D1': {'direction': 'up',    'int_x': 400},
            'C':  {'direction': 'left',  'int_x': 1000},
            'B2': {'direction': 'down',  'int_x': 1000},
            'D2': {'direction': 'up',    'int_x': 1000},
        }
        
        for spawn_point in spawn_points:
            # Get spawn rate from level
            spawn_rate = 0.5
            if self.current_level:
                seg_id = f"entry_{spawn_point}"
                if seg_id in self.current_level.segments:
                    spawn_rate = self.current_level.segments[seg_id].spawn_rate
            
            # Spawn multiple cars if rate > 1
            num_to_spawn = int(spawn_rate) + (1 if random.random() < (spawn_rate % 1) else 0)
            
            for _ in range(num_to_spawn):
                lane_num = random.randint(0, LANES_PER_DIRECTION - 1)
                info = spawn_info[spawn_point]
                types = list(VEHICLE_TYPES.keys())
                weights = [VEHICLE_TYPES[t]['spawn_weight'] for t in types]
                v_type = random.choices(types, weights=weights)[0]
                
                vehicle = Vehicle(v_type, info['direction'], lane_num, level=2, 
                                 spawn_point=spawn_point, intersection_x=info['int_x'])
                
                lane_key = (info['direction'], lane_num)
                if lane_key in self.lanes:
                    vehicle.current_lane = self.lanes[lane_key]
                    vehicle.current_lane.add_vehicle(vehicle)
                
                self.vehicles.append(vehicle)
    
    def _spawn_level3_vehicle(self):
        """Spawn vehicle(s) for Level 3 (2x2 grid)."""
        # Double check we're on Level 3
        if self.current_level_num != 3:
            return
        
        # Level 3 spawn points around the edges
        spawn_points = ['N1', 'N2', 'E1', 'E2', 'S1', 'S2', 'W1', 'W2']
        
        # Grid positions
        left_x, right_x = 400, 1000
        top_y, bottom_y = 280, 520
        
        # Map spawn point to direction and position
        spawn_info = {
            'N1': {'direction': 'down',  'x': left_x,  'y': -50},
            'N2': {'direction': 'down',  'x': right_x, 'y': -50},
            'E1': {'direction': 'left',  'x': SCREEN_WIDTH + 50, 'y': top_y},
            'E2': {'direction': 'left',  'x': SCREEN_WIDTH + 50, 'y': bottom_y},
            'S1': {'direction': 'up',    'x': left_x,  'y': SCREEN_HEIGHT + 50},
            'S2': {'direction': 'up',    'x': right_x, 'y': SCREEN_HEIGHT + 50},
            'W1': {'direction': 'right', 'x': -50,     'y': top_y},
            'W2': {'direction': 'right', 'x': -50,     'y': bottom_y},
        }
        
        for spawn_point in spawn_points:
            # Get spawn rate from level - default to 0.5 if level not ready
            spawn_rate = 0.5
            level = self.current_level
            if level and hasattr(level, 'segments'):
                seg_id = f"entry_{spawn_point}"
                seg = level.segments.get(seg_id)
                if seg:
                    spawn_rate = seg.spawn_rate
            
            # Spawn multiple cars if rate > 1
            num_to_spawn = int(spawn_rate) + (1 if random.random() < (spawn_rate % 1) else 0)
            
            for _ in range(num_to_spawn):
                lane_num = random.randint(0, LANES_PER_DIRECTION - 1)
                info = spawn_info[spawn_point]
                types = list(VEHICLE_TYPES.keys())
                weights = [VEHICLE_TYPES[t]['spawn_weight'] for t in types]
                v_type = random.choices(types, weights=weights)[0]
                
                vehicle = Vehicle(v_type, info['direction'], lane_num, level=3,
                                 spawn_point=spawn_point, spawn_x=info['x'], spawn_y=info['y'])
                
                lane_key = (info['direction'], lane_num)
                if lane_key in self.lanes:
                    vehicle.current_lane = self.lanes[lane_key]
                    vehicle.current_lane.add_vehicle(vehicle)
                
                self.vehicles.append(vehicle)
    
    def draw_roads(self):
        """Draw the intersection and roads based on current level."""
        self.screen.fill((100, 150, 100))
        
        road_width = LANES_PER_DIRECTION * LANE_WIDTH
        
        if self.current_level_num == 1:
            self._draw_level1_roads(road_width)
        elif self.current_level_num == 2:
            self._draw_level2_roads(road_width)
        elif self.current_level_num == 3:
            self._draw_level3_roads(road_width)
        else:
            self._draw_level1_roads(road_width)
    
    def _draw_intersection(self, cx, cy, road_width, labels=None):
        """Draw a single intersection at the given center position."""
        # Draw horizontal road through this intersection
        pygame.draw.rect(self.screen, ROAD_COLOR,
                        (cx - road_width * 3, cy - road_width, road_width * 6, road_width * 2))
        
        # Draw vertical road through this intersection
        pygame.draw.rect(self.screen, ROAD_COLOR,
                        (cx - road_width, cy - road_width * 3, road_width * 2, road_width * 6))
        
        # Draw lane dividers around the intersection
        for i in range(1, LANES_PER_DIRECTION):
            offset = i * LANE_WIDTH
            
            # Horizontal lane lines (left of intersection)
            y_top = cy - road_width + offset
            y_bot = cy + offset
            for x in range(cx - road_width * 3, cx - road_width - 10, 40):
                pygame.draw.line(self.screen, LANE_LINE_COLOR, (x, y_top), (x + 20, y_top), 2)
                pygame.draw.line(self.screen, LANE_LINE_COLOR, (x, y_bot), (x + 20, y_bot), 2)
            
            # Horizontal lane lines (right of intersection)
            for x in range(cx + road_width + 10, cx + road_width * 3, 40):
                pygame.draw.line(self.screen, LANE_LINE_COLOR, (x, y_top), (x + 20, y_top), 2)
                pygame.draw.line(self.screen, LANE_LINE_COLOR, (x, y_bot), (x + 20, y_bot), 2)
            
            # Vertical lane lines (above intersection)
            x_left = cx - road_width + offset
            x_right = cx + offset
            for y in range(cy - road_width * 3, cy - road_width - 10, 40):
                pygame.draw.line(self.screen, LANE_LINE_COLOR, (x_left, y), (x_left, y + 20), 2)
                pygame.draw.line(self.screen, LANE_LINE_COLOR, (x_right, y), (x_right, y + 20), 2)
            
            # Vertical lane lines (below intersection)
            for y in range(cy + road_width + 10, cy + road_width * 3, 40):
                pygame.draw.line(self.screen, LANE_LINE_COLOR, (x_left, y), (x_left, y + 20), 2)
                pygame.draw.line(self.screen, LANE_LINE_COLOR, (x_right, y), (x_right, y + 20), 2)
        
        # Stop lines
        pygame.draw.line(self.screen, STOP_LINE_COLOR,
                        (cx - road_width - 10, cy), (cx - road_width - 10, cy + road_width), 5)
        pygame.draw.line(self.screen, STOP_LINE_COLOR,
                        (cx - road_width, cy - road_width - 10), (cx, cy - road_width - 10), 5)
        pygame.draw.line(self.screen, STOP_LINE_COLOR,
                        (cx + road_width + 10, cy - road_width), (cx + road_width + 10, cy), 5)
        pygame.draw.line(self.screen, STOP_LINE_COLOR,
                        (cx, cy + road_width + 10), (cx + road_width, cy + road_width + 10), 5)
        
        # Labels if provided
        if labels:
            positions = [
                (cx - road_width * 2, cy + road_width // 2 - 10),  # West
                (cx - road_width // 2 - 10, cy - road_width * 2),  # North
                (cx + road_width + 20, cy - road_width // 2 - 10), # East
                (cx - road_width // 2 - 10, cy + road_width + 20)  # South
            ]
            for i, label in enumerate(labels):
                if label:
                    self.screen.blit(self.font.render(label, True, (255, 255, 255)), positions[i])
    
    def _draw_level1_roads(self, road_width):
        """Draw Level 1: Single intersection in center."""
        center_x = SCREEN_WIDTH // 2
        center_y = SCREEN_HEIGHT // 2
        
        # Draw horizontal road across screen
        pygame.draw.rect(self.screen, ROAD_COLOR,
                        (0, center_y - road_width, SCREEN_WIDTH, road_width * 2))
        
        # Draw vertical road across screen
        pygame.draw.rect(self.screen, ROAD_COLOR,
                        (center_x - road_width, 0, road_width * 2, SCREEN_HEIGHT))
        
        # Lane dividers
        for i in range(1, LANES_PER_DIRECTION):
            y = center_y - road_width + i * LANE_WIDTH
            for x in range(0, SCREEN_WIDTH, 40):
                pygame.draw.line(self.screen, LANE_LINE_COLOR, (x, y), (x + 20, y), 2)
            y = center_y + i * LANE_WIDTH
            for x in range(0, SCREEN_WIDTH, 40):
                pygame.draw.line(self.screen, LANE_LINE_COLOR, (x, y), (x + 20, y), 2)
            
            x = center_x - road_width + i * LANE_WIDTH
            for y in range(0, SCREEN_HEIGHT, 40):
                pygame.draw.line(self.screen, LANE_LINE_COLOR, (x, y), (x, y + 20), 2)
            x = center_x + i * LANE_WIDTH
            for y in range(0, SCREEN_HEIGHT, 40):
                pygame.draw.line(self.screen, LANE_LINE_COLOR, (x, y), (x, y + 20), 2)
        
        # Stop lines
        pygame.draw.line(self.screen, STOP_LINE_COLOR,
                        (center_x - road_width - 10, center_y),
                        (center_x - road_width - 10, center_y + road_width), 5)
        pygame.draw.line(self.screen, STOP_LINE_COLOR,
                        (center_x - road_width, center_y - road_width - 10),
                        (center_x, center_y - road_width - 10), 5)
        pygame.draw.line(self.screen, STOP_LINE_COLOR,
                        (center_x + road_width + 10, center_y - road_width),
                        (center_x + road_width + 10, center_y), 5)
        pygame.draw.line(self.screen, STOP_LINE_COLOR,
                        (center_x, center_y + road_width + 10),
                        (center_x + road_width, center_y + road_width + 10), 5)
        
        # Direction labels
        for label, pos in [
            ('A', (50, center_y + road_width // 2 - 10)),
            ('B', (center_x - road_width // 2 - 10, 50)),
            ('C', (SCREEN_WIDTH - 70, center_y - road_width // 2 - 10)),
            ('D', (center_x + road_width // 2 - 10, SCREEN_HEIGHT - 70))
        ]:
            self.screen.blit(self.font.render(label, True, (255, 255, 255)), pos)
    
    def _draw_level2_roads(self, road_width):
        """Draw Level 2: Two intersections connected horizontally."""
        center_y = SCREEN_HEIGHT // 2
        int1_x = 400   # Left intersection
        int2_x = 1000  # Right intersection
        
        # Draw main horizontal road across both intersections
        pygame.draw.rect(self.screen, ROAD_COLOR,
                        (0, center_y - road_width, SCREEN_WIDTH, road_width * 2))
        
        # Draw vertical roads at each intersection
        pygame.draw.rect(self.screen, ROAD_COLOR,
                        (int1_x - road_width, 0, road_width * 2, SCREEN_HEIGHT))
        pygame.draw.rect(self.screen, ROAD_COLOR,
                        (int2_x - road_width, 0, road_width * 2, SCREEN_HEIGHT))
        
        # Connecting segment highlight (limited capacity area)
        connect_left = int1_x + road_width + 10
        connect_right = int2_x - road_width - 10
        pygame.draw.rect(self.screen, (70, 70, 70),  # Slightly different color
                        (connect_left, center_y - road_width, 
                         connect_right - connect_left, road_width * 2))
        
        # "Capacity: 5" label on connecting segment
        cap_text = self.small_font.render("Cap: 5", True, (255, 200, 100))
        self.screen.blit(cap_text, ((int1_x + int2_x) // 2 - 20, center_y - 10))
        
        # Lane dividers for horizontal road
        for i in range(1, LANES_PER_DIRECTION):
            y = center_y - road_width + i * LANE_WIDTH
            for x in range(0, SCREEN_WIDTH, 40):
                pygame.draw.line(self.screen, LANE_LINE_COLOR, (x, y), (x + 20, y), 2)
            y = center_y + i * LANE_WIDTH
            for x in range(0, SCREEN_WIDTH, 40):
                pygame.draw.line(self.screen, LANE_LINE_COLOR, (x, y), (x + 20, y), 2)
        
        # Lane dividers for vertical roads
        for int_x in [int1_x, int2_x]:
            for i in range(1, LANES_PER_DIRECTION):
                x = int_x - road_width + i * LANE_WIDTH
                for y in range(0, SCREEN_HEIGHT, 40):
                    pygame.draw.line(self.screen, LANE_LINE_COLOR, (x, y), (x, y + 20), 2)
                x = int_x + i * LANE_WIDTH
                for y in range(0, SCREEN_HEIGHT, 40):
                    pygame.draw.line(self.screen, LANE_LINE_COLOR, (x, y), (x, y + 20), 2)
        
        # Stop lines for INT1
        pygame.draw.line(self.screen, STOP_LINE_COLOR,
                        (int1_x - road_width - 10, center_y),
                        (int1_x - road_width - 10, center_y + road_width), 5)
        pygame.draw.line(self.screen, STOP_LINE_COLOR,
                        (int1_x - road_width, center_y - road_width - 10),
                        (int1_x, center_y - road_width - 10), 5)
        pygame.draw.line(self.screen, STOP_LINE_COLOR,
                        (int1_x, center_y + road_width + 10),
                        (int1_x + road_width, center_y + road_width + 10), 5)
        
        # Stop lines for INT2
        pygame.draw.line(self.screen, STOP_LINE_COLOR,
                        (int2_x + road_width + 10, center_y - road_width),
                        (int2_x + road_width + 10, center_y), 5)
        pygame.draw.line(self.screen, STOP_LINE_COLOR,
                        (int2_x - road_width, center_y - road_width - 10),
                        (int2_x, center_y - road_width - 10), 5)
        pygame.draw.line(self.screen, STOP_LINE_COLOR,
                        (int2_x, center_y + road_width + 10),
                        (int2_x + road_width, center_y + road_width + 10), 5)
        
        # Labels
        labels = [
            ('A', (50, center_y + road_width // 2 - 10)),
            ('B1', (int1_x - 15, 50)),
            ('D1', (int1_x - 15, SCREEN_HEIGHT - 70)),
            ('C', (SCREEN_WIDTH - 50, center_y - road_width // 2 - 10)),
            ('B2', (int2_x - 15, 50)),
            ('D2', (int2_x - 15, SCREEN_HEIGHT - 70)),
            ('INT1', (int1_x - 25, center_y - 10)),
            ('INT2', (int2_x - 25, center_y - 10)),
        ]
        for label, pos in labels:
            color = (0, 255, 255) if 'INT' in label else (255, 255, 255)
            self.screen.blit(self.font.render(label, True, color), pos)
    
    def _draw_level3_roads(self, road_width):
        """Draw Level 3: 2x2 grid of 4 intersections."""
        # Grid positions
        left_x = 400
        right_x = 1000
        top_y = 280
        bottom_y = 520
        
        # Draw horizontal roads (two of them)
        pygame.draw.rect(self.screen, ROAD_COLOR,
                        (0, top_y - road_width, SCREEN_WIDTH, road_width * 2))
        pygame.draw.rect(self.screen, ROAD_COLOR,
                        (0, bottom_y - road_width, SCREEN_WIDTH, road_width * 2))
        
        # Draw vertical roads (two of them)
        pygame.draw.rect(self.screen, ROAD_COLOR,
                        (left_x - road_width, 0, road_width * 2, SCREEN_HEIGHT))
        pygame.draw.rect(self.screen, ROAD_COLOR,
                        (right_x - road_width, 0, road_width * 2, SCREEN_HEIGHT))
        
        # Connecting segments (limited capacity) - horizontal
        h1_left = left_x + road_width + 5
        h1_right = right_x - road_width - 5
        pygame.draw.rect(self.screen, (70, 70, 70),
                        (h1_left, top_y - road_width, h1_right - h1_left, road_width * 2))
        pygame.draw.rect(self.screen, (70, 70, 70),
                        (h1_left, bottom_y - road_width, h1_right - h1_left, road_width * 2))
        
        # Connecting segments - vertical
        v1_top = top_y + road_width + 5
        v1_bottom = bottom_y - road_width - 5
        pygame.draw.rect(self.screen, (70, 70, 70),
                        (left_x - road_width, v1_top, road_width * 2, v1_bottom - v1_top))
        pygame.draw.rect(self.screen, (70, 70, 70),
                        (right_x - road_width, v1_top, road_width * 2, v1_bottom - v1_top))
        
        # Capacity labels on connecting segments
        cap_text = self.small_font.render("5", True, (255, 200, 100))
        self.screen.blit(cap_text, ((left_x + right_x) // 2 - 5, top_y - 8))
        self.screen.blit(cap_text, ((left_x + right_x) // 2 - 5, bottom_y - 8))
        self.screen.blit(cap_text, (left_x - 5, (top_y + bottom_y) // 2 - 8))
        self.screen.blit(cap_text, (right_x - 5, (top_y + bottom_y) // 2 - 8))
        
        # Lane dividers - horizontal roads
        for center_y in [top_y, bottom_y]:
            for i in range(1, LANES_PER_DIRECTION):
                y = center_y - road_width + i * LANE_WIDTH
                for x in range(0, SCREEN_WIDTH, 40):
                    pygame.draw.line(self.screen, LANE_LINE_COLOR, (x, y), (x + 20, y), 2)
                y = center_y + i * LANE_WIDTH
                for x in range(0, SCREEN_WIDTH, 40):
                    pygame.draw.line(self.screen, LANE_LINE_COLOR, (x, y), (x + 20, y), 2)
        
        # Lane dividers - vertical roads
        for int_x in [left_x, right_x]:
            for i in range(1, LANES_PER_DIRECTION):
                x = int_x - road_width + i * LANE_WIDTH
                for y in range(0, SCREEN_HEIGHT, 40):
                    pygame.draw.line(self.screen, LANE_LINE_COLOR, (x, y), (x, y + 20), 2)
                x = int_x + i * LANE_WIDTH
                for y in range(0, SCREEN_HEIGHT, 40):
                    pygame.draw.line(self.screen, LANE_LINE_COLOR, (x, y), (x, y + 20), 2)
        
        # Draw stop lines at each intersection
        for int_x, int_y in [(left_x, top_y), (right_x, top_y), 
                              (left_x, bottom_y), (right_x, bottom_y)]:
            # West approach
            pygame.draw.line(self.screen, STOP_LINE_COLOR,
                            (int_x - road_width - 5, int_y),
                            (int_x - road_width - 5, int_y + road_width), 4)
            # North approach
            pygame.draw.line(self.screen, STOP_LINE_COLOR,
                            (int_x - road_width, int_y - road_width - 5),
                            (int_x, int_y - road_width - 5), 4)
            # East approach
            pygame.draw.line(self.screen, STOP_LINE_COLOR,
                            (int_x + road_width + 5, int_y - road_width),
                            (int_x + road_width + 5, int_y), 4)
            # South approach
            pygame.draw.line(self.screen, STOP_LINE_COLOR,
                            (int_x, int_y + road_width + 5),
                            (int_x + road_width, int_y + road_width + 5), 4)
        
        # Edge labels (8 entry points)
        labels = [
            ('N1', (left_x - 15, 30)),
            ('N2', (right_x - 15, 30)),
            ('W1', (30, top_y - 10)),
            ('W2', (30, bottom_y - 10)),
            ('E1', (SCREEN_WIDTH - 50, top_y - 10)),
            ('E2', (SCREEN_WIDTH - 50, bottom_y - 10)),
            ('S1', (left_x - 15, SCREEN_HEIGHT - 50)),
            ('S2', (right_x - 15, SCREEN_HEIGHT - 50)),
        ]
        for label, pos in labels:
            self.screen.blit(self.font.render(label, True, (255, 255, 255)), pos)
        
        # Intersection labels
        int_labels = [
            ('1', (left_x - 8, top_y - 10)),
            ('2', (right_x - 8, top_y - 10)),
            ('3', (left_x - 8, bottom_y - 10)),
            ('4', (right_x - 8, bottom_y - 10)),
        ]
        for label, pos in int_labels:
            self.screen.blit(self.font.render(label, True, (0, 255, 255)), pos)
    
    def draw_signals(self):
        """Draw the traffic lights and turn arrows based on current level."""
        if self.current_level_num == 1:
            self._draw_level1_signals()
        elif self.current_level_num == 2:
            self._draw_level2_signals()
        elif self.current_level_num == 3:
            self._draw_level3_signals()
        else:
            self._draw_level1_signals()
    
    def _draw_signal_at(self, x, y, is_green, has_arrow=False, arrow_green=False, timer=None):
        """Helper to draw a single traffic signal."""
        color = (0, 255, 0) if is_green else (255, 0, 0)
        
        pygame.draw.circle(self.screen, (0, 0, 0), (x, y), SIGNAL_RADIUS + 3)
        pygame.draw.circle(self.screen, color, (x, y), SIGNAL_RADIUS)
        
        if has_arrow:
            arrow_x = x + 35
            arrow_color = (0, 255, 0) if arrow_green else (255, 0, 0)
            pygame.draw.circle(self.screen, (0, 0, 0), (arrow_x, y), SIGNAL_RADIUS // 2 + 2)
            pygame.draw.circle(self.screen, arrow_color, (arrow_x, y), SIGNAL_RADIUS // 2)
        
        if timer is not None:
            timer_text = self.small_font.render(f"{timer:.1f}s", True, (255, 255, 255))
            self.screen.blit(timer_text, (x - 20, y + 25))
    
    def _draw_level1_signals(self):
        """Draw signals for Level 1: single intersection."""
        for direction, signal in self.signals.items():
            x, y = signal.position
            self._draw_signal_at(
                x, y, 
                signal.state == 'green',
                signal.turn_arrow_enabled,
                signal.turn_arrow_state == 'green',
                signal.timer
            )
    
    def _draw_level2_signals(self):
        """Draw signals for Level 2: two intersections."""
        road_width = LANES_PER_DIRECTION * LANE_WIDTH
        center_y = SCREEN_HEIGHT // 2
        int1_x = 400
        int2_x = 1000
        
        # Get signal states from the level if available
        if self.current_level and 'INT1' in self.current_level.intersections:
            int1 = self.current_level.intersections['INT1']
            int2 = self.current_level.intersections['INT2']
            
            # INT1 signals (West entry, North entry, South entry)
            # West approach (right direction in original coords)
            self._draw_signal_at(
                int1_x - road_width - 50, center_y + road_width + 40,
                int1.signal_states.get('right', 'red') == 'green',
                int1.arrow_states.get('right', False),
                int1.arrow_states.get('right', False)
            )
            # North approach (down direction)
            self._draw_signal_at(
                int1_x - road_width - 50, center_y - road_width - 40,
                int1.signal_states.get('down', 'red') == 'green',
                int1.arrow_states.get('down', False),
                int1.arrow_states.get('down', False)
            )
            # South approach (up direction)
            self._draw_signal_at(
                int1_x + road_width + 50, center_y + road_width + 40,
                int1.signal_states.get('up', 'red') == 'green',
                int1.arrow_states.get('up', False),
                int1.arrow_states.get('up', False)
            )
            
            # INT2 signals (East entry, North entry, South entry)
            # East approach (left direction)
            self._draw_signal_at(
                int2_x + road_width + 50, center_y - road_width - 40,
                int2.signal_states.get('left', 'red') == 'green',
                int2.arrow_states.get('left', False),
                int2.arrow_states.get('left', False)
            )
            # North approach (down direction)
            self._draw_signal_at(
                int2_x - road_width - 50, center_y - road_width - 40,
                int2.signal_states.get('down', 'red') == 'green',
                int2.arrow_states.get('down', False),
                int2.arrow_states.get('down', False)
            )
            # South approach (up direction)
            self._draw_signal_at(
                int2_x + road_width + 50, center_y + road_width + 40,
                int2.signal_states.get('up', 'red') == 'green',
                int2.arrow_states.get('up', False),
                int2.arrow_states.get('up', False)
            )
        else:
            # Fallback: draw original signals
            self._draw_level1_signals()
    
    def _draw_level3_signals(self):
        """Draw signals for Level 3: 2x2 grid of 4 intersections."""
        road_width = LANES_PER_DIRECTION * LANE_WIDTH
        left_x = 400
        right_x = 1000
        top_y = 280
        bottom_y = 520
        
        if self.current_level:
            intersections = {
                'INT1': (left_x, top_y),
                'INT2': (right_x, top_y),
                'INT3': (left_x, bottom_y),
                'INT4': (right_x, bottom_y),
            }
            
            for int_id, (cx, cy) in intersections.items():
                if int_id in self.current_level.intersections:
                    node = self.current_level.intersections[int_id]
                    
                    # Draw a small signal indicator at each approach
                    # Simplified: just show colored dots near each intersection
                    offset = road_width + 25
                    
                    for direction, (dx, dy) in [
                        ('right', (-offset, offset // 2)),   # West approach
                        ('down', (-offset // 2, -offset)),   # North approach
                        ('left', (offset, -offset // 2)),    # East approach
                        ('up', (offset // 2, offset)),       # South approach
                    ]:
                        if direction in node.signal_states:
                            is_green = node.signal_states[direction] == 'green'
                            has_arrow = node.arrow_states.get(direction, False)
                            color = (0, 200, 0) if is_green else (200, 0, 0)
                            
                            px, py = cx + dx, cy + dy
                            pygame.draw.circle(self.screen, (0, 0, 0), (px, py), 10)
                            pygame.draw.circle(self.screen, color, (px, py), 8)
                            
                            if has_arrow:
                                pygame.draw.circle(self.screen, (0, 0, 0), (px + 15, py), 6)
                                pygame.draw.circle(self.screen, (0, 200, 0), (px + 15, py), 5)
        else:
            self._draw_level1_signals()
    
    def draw_stats(self):
        """Draw debug stats on screen."""
        y_offset = 10
        
        for direction, count in self.vehicles_crossed.items():
            lane_label = LANE_LABELS[direction]
            text = self.small_font.render(f"Lane {lane_label} ({direction}): {count} crossed", True, (255, 255, 255))
            self.screen.blit(text, (10, y_offset))
            y_offset += 25
        
        y_offset += 5
        self.screen.blit(self.small_font.render("Queue Pressure:", True, (255, 255, 0)), (10, y_offset))
        y_offset += 20
        
        direction_map = {'right': 'A', 'down': 'B', 'left': 'C', 'up': 'D'}
        for direction in ['right', 'down', 'left', 'up']:
            lane_label = direction_map[direction]
            queue_counts = []
            total_wait_time = 0
            waiting_vehicles = 0
            
            for lane_num in range(LANES_PER_DIRECTION):
                lane_key = (direction, lane_num)
                if lane_key in self.lanes:
                    lane = self.lanes[lane_key]
                    waiting_count = sum(1 for v in lane.vehicles if v.wait_time > 0)
                    queue_counts.append(str(waiting_count))
                    for v in lane.vehicles:
                        if v.wait_time > 0:
                            total_wait_time += v.wait_time
                            waiting_vehicles += 1
            
            avg_wait = total_wait_time / waiting_vehicles if waiting_vehicles > 0 else 0
            queue_str = ", ".join(queue_counts)
            text = self.small_font.render(f"{lane_label}: [{queue_str}] (avg wait: {avg_wait:.1f}f)", True, (255, 255, 255))
            self.screen.blit(text, (10, y_offset))
            y_offset += 20
        
        y_offset += 5
        state_vector = self.get_state()
        self.screen.blit(self.small_font.render(f"State Vector: {len(state_vector)} values", True, (255, 255, 0)), (10, y_offset))
        y_offset += 20
        
        queue_pressures = state_vector[:8]
        queue_str = ", ".join([f"{v:.2f}" for v in queue_pressures])
        self.screen.blit(self.small_font.render(f"Queues: [{queue_str}]", True, (255, 255, 255)), (10, y_offset))
        y_offset += 20
        
        y_offset += 5
        self.screen.blit(self.small_font.render("Destinations: A=West B=North C=East D=South", True, (255, 255, 0)), (10, y_offset))
        
        # Show current level info
        y_offset += 25
        level_info = self.get_level_info()
        level_text = f"Level {level_info['level']}: {level_info['name']}"
        self.screen.blit(self.small_font.render(level_text, True, (0, 255, 255)), (10, y_offset))
        y_offset += 20
        size_text = f"State: {level_info['state_size']} values, Actions: {level_info['action_count']}"
        self.screen.blit(self.small_font.render(size_text, True, (200, 200, 200)), (10, y_offset))
        
        self.screen.blit(self.small_font.render("Keys: 1-4=Arrows, S=State, L=Switch Level", True, (255, 255, 255)), (10, SCREEN_HEIGHT - 30))
    
    def get_state(self):
        """Returns a normalized state vector for RL training.
        
        Format: [8 queue pressures, 4 throughputs, 4 signals, 4 arrows] = 20 values
        All values are floats between 0.0 and 1.0.
        """
        MAX_QUEUE = 20
        MAX_THROUGHPUT = 100
        
        state_vector = []
        
        # queue pressures (how many cars waiting per lane)
        for direction in ['right', 'down', 'left', 'up']:
            for lane_num in range(LANES_PER_DIRECTION):
                lane_key = (direction, lane_num)
                if lane_key in self.lanes:
                    waiting = sum(1 for v in self.lanes[lane_key].vehicles if v.wait_time > 0)
                    state_vector.append(min(waiting / MAX_QUEUE, 1.0))
                else:
                    state_vector.append(0.0)
        
        # throughput (total cars that exited each direction)
        for direction in ['right', 'down', 'left', 'up']:
            throughput = self.vehicles_crossed.get(direction, 0)
            state_vector.append(min(throughput / MAX_THROUGHPUT, 1.0))
        
        # signal states (1.0 = green, 0.0 = not green)
        for direction in ['right', 'down', 'left', 'up']:
            state_vector.append(1.0 if self.signals[direction].state == 'green' else 0.0)
        
        # arrow states (1.0 = enabled and green)
        for direction in ['right', 'down', 'left', 'up']:
            signal = self.signals[direction]
            arrow_on = signal.turn_arrow_enabled and signal.turn_arrow_state == 'green'
            state_vector.append(1.0 if arrow_on else 0.0)
        
        return state_vector
    
    def get_metrics(self):
        """Returns detailed stats useful for calculating rewards.
        
        Keys: total_wait_time, avg_wait_time, total_throughput, 
        queue_lengths, waiting_vehicles, total_vehicles
        """
        total_wait_time = 0
        waiting_vehicles = 0
        total_vehicles = len(self.vehicles)
        queue_lengths = {'right': 0, 'down': 0, 'left': 0, 'up': 0}
        
        for vehicle in self.vehicles:
            total_wait_time += vehicle.wait_time
            if vehicle.wait_time > 0:
                waiting_vehicles += 1
                queue_lengths[vehicle.original_direction] += 1
        
        avg_wait_time = total_wait_time / total_vehicles if total_vehicles > 0 else 0
        total_throughput = sum(self.vehicles_crossed.values())
        
        return {
            'total_wait_time': total_wait_time,
            'avg_wait_time': avg_wait_time,
            'total_throughput': total_throughput,
            'queue_lengths': queue_lengths,
            'waiting_vehicles': waiting_vehicles,
            'total_vehicles': total_vehicles
        }
    
    def set_signals(self, green_directions, arrow_directions=None):
        """Lets the AI directly set which lights are green.
        
        green_directions: list of directions to turn green, e.g. ['right', 'left']
        arrow_directions: list of directions to enable turn arrows (optional)
        """
        if arrow_directions is None:
            arrow_directions = []
        
        all_directions = ['right', 'down', 'left', 'up']
        
        for direction in all_directions:
            signal = self.signals[direction]
            
            # Set main signal
            if direction in green_directions:
                signal.state = 'green'
            else:
                signal.state = 'red'
            
            # Set turn arrow
            if direction in arrow_directions:
                signal.turn_arrow_enabled = True
                signal.turn_arrow_state = 'green'
            else:
                signal.turn_arrow_enabled = False
                signal.turn_arrow_state = 'red'
    
    def set_phase(self, phase_id):
        """Simplified way to set signal configuration by number (0-7).
        
        0-3 = East-West green with different arrow combos
        4-7 = North-South green with different arrow combos
        """
        phases = {
            0: (['right', 'left'], []),                          # EW green, no arrows
            1: (['right', 'left'], ['right', 'left']),          # EW green + EW arrows
            2: (['right', 'left'], ['down', 'up']),             # EW green + NS arrows
            3: (['right', 'left'], ['right', 'down', 'left', 'up']),  # EW + all arrows
            4: (['down', 'up'], []),                             # NS green, no arrows
            5: (['down', 'up'], ['down', 'up']),                # NS green + NS arrows
            6: (['down', 'up'], ['right', 'left']),             # NS green + EW arrows
            7: (['down', 'up'], ['right', 'down', 'left', 'up']),    # NS + all arrows
        }
        
        phase_names = {
            0: 'EW Green',
            1: 'EW Green + EW Arrows',
            2: 'EW Green + NS Arrows', 
            3: 'EW Green + All Arrows',
            4: 'NS Green',
            5: 'NS Green + NS Arrows',
            6: 'NS Green + EW Arrows',
            7: 'NS Green + All Arrows'
        }
        
        if phase_id in phases:
            green_dirs, arrow_dirs = phases[phase_id]
            self.set_signals(green_dirs, arrow_dirs)
            return phase_names[phase_id]
        else:
            return 'Invalid phase'
    
    def switch_level(self, level_num: int = None):
        """Switch to a different level. Clears all vehicles and resets state.
        
        If level_num is None, cycles to the next level (1 -> 2 -> 3 -> 1).
        """
        if not LEVELS_AVAILABLE:
            print("Level system not available")
            return
        
        # Cycle through levels if no specific level given
        if level_num is None:
            level_num = (self.current_level_num % 3) + 1
        
        try:
            new_level = get_level_by_number(level_num)
        except ValueError:
            print(f"Level {level_num} not found")
            return
        
        # Clear all vehicles from the current level's segments if it exists
        if self.current_level:
            for segment in self.current_level.segments.values():
                if hasattr(segment, 'vehicles'):
                    segment.vehicles.clear()
        
        # Clear main vehicle list and lanes
        self.vehicles.clear()
        for lane in self.lanes.values():
            lane.vehicles.clear()
        self.intersection.vehicles_in_intersection.clear()
        self.vehicles_crossed = {'right': 0, 'down': 0, 'left': 0, 'up': 0}
        
        # Apply new level
        self.current_level = new_level
        self.current_level_num = level_num
        
        # Apply first action to set signals
        self.apply_level_action(0)
        
        print(f"Switched to {new_level.name}")
        print(f"  State size: {new_level.get_state_size()}")
        print(f"  Actions: {new_level.get_action_count()}")
    
    def get_level_info(self) -> dict:
        """Returns information about the current level."""
        if not self.current_level:
            return {
                'level': 1,
                'name': 'Default (Single Intersection)',
                'state_size': 20,
                'action_count': 8
            }
        
        return {
            'level': self.current_level_num,
            'name': self.current_level.name,
            'state_size': self.current_level.get_state_size(),
            'action_count': self.current_level.get_action_count(),
            'spawn_points': self.current_level.spawn_points,
            'exit_points': self.current_level.exit_points
        }
    
    def apply_level_action(self, action_id: int) -> str:
        """Apply an action using the level's action space.
        
        For Level 1, this is the same as set_phase().
        For Levels 2-3, this coordinates multiple intersections.
        """
        if self.current_level:
            self.current_level.apply_action(action_id)
            
            # Also update the local signals to match (for rendering)
            # For Level 1, we sync with the level's intersection signals
            if hasattr(self.current_level, 'action_names'):
                action_name = self.current_level.action_names[action_id] if action_id < len(self.current_level.action_names) else 'Unknown'
            else:
                action_name = f'Action {action_id}'
            
            return action_name
        else:
            return self.set_phase(action_id)
    
    def get_spawn_rates(self) -> dict:
        """Get current spawn rates for all spawn points (for sliders)."""
        if self.current_level:
            return self.current_level.get_spawn_rates()
        return LANE_SPAWN_PROBABILITIES.copy()
    
    def set_spawn_rates(self, rates: dict):
        """Set spawn rates for spawn points (for sliders)."""
        if self.current_level:
            self.current_level.set_spawn_rates(rates)
        else:
            for key, rate in rates.items():
                if key in LANE_SPAWN_PROBABILITIES:
                    LANE_SPAWN_PROBABILITIES[key] = rate
    
    def randomize_spawn_rates(self):
        """Randomize spawn rates for training diversity."""
        if self.current_level:
            self.current_level.randomize_spawn_rates()
    
    def print_state_debug(self):
        """Dumps the full state to the console - handy for debugging."""
        state = self.get_state()
        metrics = self.get_metrics()
        
        print("\n" + "="*60)
        print("AI STATE VECTOR (size={})".format(len(state)))
        print("="*60)
        
        # Queue pressures
        print("\nQUEUE PRESSURES (normalized by 20):")
        for i, direction in enumerate(['A', 'B', 'C', 'D']):
            for lane in range(LANES_PER_DIRECTION):
                idx = i * LANES_PER_DIRECTION + lane
                print(f"  {direction}_Lane_{lane}: {state[idx]:.3f} (raw: {int(state[idx] * 20)})")
        
        # Throughput
        print("\nTHROUGHPUT (normalized by 100):")
        for i, direction in enumerate(['A', 'B', 'C', 'D']):
            idx = 8 + i
            print(f"  {direction}: {state[idx]:.3f} (raw: {int(state[idx] * 100)})")
        
        # Signals
        print("\nSIGNAL STATUS (1=green, 0=red/yellow):")
        for i, direction in enumerate(['A', 'B', 'C', 'D']):
            idx = 12 + i
            print(f"  {direction}: {state[idx]:.0f}")
        
        # Turn arrows
        print("\nTURN ARROWS (1=enabled+green, 0=off/red):")
        for i, direction in enumerate(['A', 'B', 'C', 'D']):
            idx = 16 + i
            print(f"  {direction}: {state[idx]:.0f}")
        
        # Metrics
        print("\nMETRICS:")
        print(f"  Total Wait Time: {metrics['total_wait_time']} frames")
        print(f"  Average Wait Time: {metrics['avg_wait_time']:.2f} frames")
        print(f"  Total Throughput: {metrics['total_throughput']} vehicles")
        print(f"  Waiting Vehicles: {metrics['waiting_vehicles']}/{metrics['total_vehicles']}")
        print(f"  Queue Lengths: {metrics['queue_lengths']}")
        print("="*60 + "\n")
    
    def _get_signal_for_vehicle(self, vehicle):
        """Get the appropriate signal for a vehicle based on current level."""
        direction = vehicle.original_direction
        
        # If we have a level with intersections, use those (works for all levels)
        if self.current_level and hasattr(self.current_level, 'intersections'):
            level = self.current_level
            
            # Find the intersection that has this direction
            for int_id, node in level.intersections.items():
                if direction in node.signal_states:
                    is_green = node.signal_states[direction] == 'green'
                    has_arrow = node.arrow_states.get(direction, False)
                    return LevelSignalProxy(is_green, has_arrow)
        
        # Fallback to classic signal system (for non-level mode)
        return self.signals[direction]
    
    def update(self):
        """Advance the simulation by one frame."""
        for vehicle in self.vehicles[:]:
            signal = self._get_signal_for_vehicle(vehicle)
            vehicle.move(signal, self.intersection, self.lanes)
            
            if vehicle.crossed and vehicle.original_direction in self.vehicles_crossed:
                if not hasattr(vehicle, 'counted'):
                    self.vehicles_crossed[vehicle.original_direction] += 1
                    vehicle.counted = True
            
            if vehicle.is_off_screen():
                if vehicle.current_lane:
                    vehicle.current_lane.remove_vehicle(vehicle)
                if vehicle.in_intersection:
                    self.intersection.exit(vehicle)
                self.vehicles.remove(vehicle)
    
    def run(self):
        """Main loop - use this if you want to run the simulation standalone."""
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    # Toggle turn arrows with keys 1-4
                    if event.key == pygame.K_1:
                        self.signals['right'].turn_arrow_enabled = not self.signals['right'].turn_arrow_enabled
                    elif event.key == pygame.K_2:
                        self.signals['down'].turn_arrow_enabled = not self.signals['down'].turn_arrow_enabled
                    elif event.key == pygame.K_3:
                        self.signals['left'].turn_arrow_enabled = not self.signals['left'].turn_arrow_enabled
                    elif event.key == pygame.K_4:
                        self.signals['up'].turn_arrow_enabled = not self.signals['up'].turn_arrow_enabled
                    elif event.key == pygame.K_s:
                        self.print_state_debug()
                    elif event.key == pygame.K_l:
                        # L key: switch to next level
                        self.switch_level()
            
            # Update and draw
            self.update()
            self.draw_roads()
            
            for vehicle in self.vehicles:
                vehicle.draw(self.screen, self.vehicle_font)
            
            self.draw_signals()
            self.draw_stats()
            
            pygame.display.flip()
            self.clock.tick(FPS)
        
        pygame.quit()


if __name__ == "__main__":
    simulator = TrafficSimulator()
    simulator.run()