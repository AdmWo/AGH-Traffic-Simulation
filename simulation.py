import pygame
import random
import threading
import time
from collections import deque

# ==================== CONFIGURATION CONSTANTS ====================
# Screen settings
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 800
FPS = 60

# Simulation speed multiplier (affects vehicle speed and signal timing)
# 1.0 = normal speed, 2.0 = 2x speed, 0.5 = half speed
SIMULATION_SPEED = 2.0

# Road settings
LANES_PER_DIRECTION = 2
LANE_WIDTH = 40
ROAD_COLOR = (50, 50, 50)
LANE_LINE_COLOR = (200, 200, 200)
STOP_LINE_COLOR = (255, 255, 0)

# Turn lane configuration (which lanes can turn)
# In right-hand traffic: lane 0 = rightmost (outer), lane 1 = leftmost (inner)
RIGHT_TURN_LANES = 1  # Number of rightmost lanes that can turn right (0=rightmost only)
LEFT_TURN_LANES = 1   # Number of leftmost lanes that can turn left (highest lane numbers)

# Vehicle settings - ALL PERFECT SQUARES
VEHICLE_SPEED = 3.0  # All vehicles move at the same speed

VEHICLE_TYPES = {
    'car': {'color': (0, 120, 255), 'size': 28, 'spawn_weight': 40},
    'bus': {'color': (255, 200, 0), 'size': 33, 'spawn_weight': 15},
    'truck': {'color': (150, 75, 0), 'size': 31, 'spawn_weight': 25},
    'bike': {'color': (0, 200, 100), 'size': 26, 'spawn_weight': 20}
}
SAFE_DISTANCE = 15  # Minimum distance between vehicles
TURN_RIGHT_PROBABILITY = 0.3  # 30% chance to turn right
TURN_LEFT_PROBABILITY = 0.4   # 40% chance to turn left (remaining 30% go straight)
TURN_CLEAR_DISTANCE = 100  # Distance to check for clear lane when turning

# Traffic signal settings
SIGNAL_RADIUS = 20
SIGNAL_YELLOW_TIME = 3  # seconds
SIGNAL_RED_TIME = 2  # seconds (all red for safety)

# Signal phase configuration
# Each phase defines which signals are green and for how long
# Format: {'duration': seconds, 'green': [directions], 'arrows': [directions with right turn arrows]}
# Directions: 'right'=A, 'down'=B, 'left'=C, 'up'=D
SIGNAL_PHASES = [
    {
        'duration': 20,  # seconds
        'green': ['right', 'left'],  # A and C main signals green
        'arrows': ['down']  # C right turn arrow (Cr)
    },
    {
        'duration': 15,  # seconds
        'green': ['down', 'up'],  # B and D main signals green
        'arrows': ['left']  # B right turn arrow (Br)
    }
]

# Spawn settings
SPAWN_INTERVAL = 1.0  # seconds between spawns

# Lane labels: A=West(left entry), B=North(top entry), C=East(right entry), D=South(bottom entry)
LANE_LABELS = {
    'right': 'A',  # Vehicles entering from West (going right/east)
    'down': 'B',   # Vehicles entering from North (going down/south)
    'left': 'C',   # Vehicles entering from East (going left/west)
    'up': 'D'      # Vehicles entering from South (going up/north)
}

# Destination mapping based on direction and turning
# Format: {source_direction: {'straight': dest, 'right': dest, 'left': dest}}
DESTINATION_MAP = {
    'right': {'straight': 'C', 'right': 'D', 'left': 'B'},  # From A: straight->C(East), right->D(South), left->B(North)
    'down': {'straight': 'D', 'right': 'A', 'left': 'C'},   # From B: straight->D(South), right->A(West), left->C(East)
    'left': {'straight': 'A', 'right': 'B', 'left': 'D'},   # From C: straight->A(West), right->B(North), left->D(South)
    'up': {'straight': 'B', 'right': 'C', 'left': 'A'}      # From D: straight->B(North), right->C(East), left->A(West)
}

# ==================== LANE CLASS ====================
class Lane:
    """Represents a single lane in one direction"""
    def __init__(self, direction, lane_number):
        self.direction = direction  # 'right', 'down', 'left', 'up'
        self.lane_number = lane_number  # 0, 1, etc.
        self.vehicles = []  # List of vehicles currently in this lane (ordered)
    
    def add_vehicle(self, vehicle):
        """Add a vehicle to this lane"""
        if vehicle not in self.vehicles:
            self.vehicles.append(vehicle)
    
    def remove_vehicle(self, vehicle):
        """Remove a vehicle from this lane"""
        if vehicle in self.vehicles:
            self.vehicles.remove(vehicle)
    
    def get_vehicle_ahead(self, vehicle):
        """Get the vehicle directly ahead of this one in the lane"""
        try:
            idx = self.vehicles.index(vehicle)
            if idx > 0:
                return self.vehicles[idx - 1]
        except ValueError:
            pass
        return None

# ==================== INTERSECTION CLASS ====================
class Intersection:
    """Manages the intersection area and conflict detection"""
    def __init__(self):
        self.vehicles_in_intersection = set()
        self.conflict_map = self._build_conflict_map()
    
    def _build_conflict_map(self):
        """Define which turning paths conflict with each other
        
        Returns dict: (direction, turn_type) -> [(conflicting_direction, conflicting_turn_type), ...]
        """
        conflicts = {
            # Right turns: conflict with traffic from destination going straight
            ('right', 'right'): [('up', 'straight')],      # A->D conflicts with D->B
            ('down', 'right'): [('right', 'straight')],    # B->A conflicts with A->C
            ('left', 'right'): [('down', 'straight')],     # C->B conflicts with B->D
            ('up', 'right'): [('left', 'straight')],       # D->C conflicts with C->A
            
            # Left turns: conflict with oncoming traffic (straight and right turns)
            ('right', 'left'): [('left', 'straight'), ('left', 'right')],  # A->B conflicts with C->A, C->D
            ('down', 'left'): [('up', 'straight'), ('up', 'right')],       # B->C conflicts with D->B, D->A
            ('left', 'left'): [('right', 'straight'), ('right', 'right')], # C->D conflicts with A->C, A->B
            ('up', 'left'): [('down', 'straight'), ('down', 'right')],     # D->A conflicts with B->D, B->C
            
            # Straight: no conflicts (other than same-direction traffic handled by lanes)
            ('right', 'straight'): [],
            ('down', 'straight'): [],
            ('left', 'straight'): [],
            ('up', 'straight'): [],
        }
        return conflicts
    
    def can_enter(self, vehicle):
        """Check if vehicle can safely enter intersection"""
        my_path = (vehicle.original_direction, vehicle.turn_type)
        conflicting_paths = self.conflict_map.get(my_path, [])
        
        # Check if any vehicle in intersection has a conflicting path
        for other in self.vehicles_in_intersection:
            other_path = (other.original_direction, other.turn_type)
            if other_path in conflicting_paths:
                return False
        
        return True
    
    def enter(self, vehicle):
        """Vehicle enters the intersection"""
        self.vehicles_in_intersection.add(vehicle)
    
    def exit(self, vehicle):
        """Vehicle exits the intersection"""
        self.vehicles_in_intersection.discard(vehicle)
    
    def is_vehicle_in_intersection(self, vehicle):
        """Check if vehicle is currently in intersection"""
        return vehicle in self.vehicles_in_intersection

# ==================== TRAFFIC SIGNAL CLASS ====================
class TrafficSignal:
    """Manages traffic light state and timing for one direction"""
    def __init__(self, direction, position):
        self.direction = direction  # 'right', 'down', 'left', 'up'
        self.position = position  # (x, y) for drawing
        self.state = 'red'  # 'green', 'yellow', 'red'
        self.timer = 0
        self.turn_arrow_enabled = False  # Can be toggled
        self.turn_arrow_state = 'red'
        
    def get_color(self):
        """Returns RGB color based on current state"""
        if self.state == 'green':
            return (0, 255, 0)
        elif self.state == 'yellow':
            return (255, 255, 0)
        else:
            return (255, 0, 0)
    
    def get_turn_color(self):
        """Returns RGB color for turn arrow"""
        if self.turn_arrow_state == 'green':
            return (0, 255, 0)
        else:
            return (255, 0, 0)
    
    def can_go(self):
        """Check if vehicles can proceed"""
        return self.state == 'green'
    
    def can_turn(self):
        """Check if turn arrow allows turning (when enabled, independent of main signal)"""
        if self.turn_arrow_enabled:
            return self.turn_arrow_state == 'green'
        # If no arrow, turning requires green main signal
        return self.state == 'green'

# ==================== VEHICLE CLASS ====================
class Vehicle:
    """Represents a single vehicle with movement logic"""
    def __init__(self, v_type, direction, lane):
        self.type = v_type
        self.original_direction = direction  # Original entry direction (for signals/stats)
        self.direction = direction  # Current movement direction (changes during turn)
        self.lane = lane  # 0 or 1 (0 is rightmost lane = outer/slow lane)
        self.color = VEHICLE_TYPES[v_type]['color']
        self.size = VEHICLE_TYPES[v_type]['size']  # Perfect square
        self.speed = VEHICLE_SPEED * SIMULATION_SPEED
        
        # Determine which turns are allowed from this lane
        # Right turns: allowed from rightmost RIGHT_TURN_LANES lanes (0, 1, ...)
        can_turn_right = lane < RIGHT_TURN_LANES
        # Left turns: allowed from leftmost LEFT_TURN_LANES lanes (highest lane numbers)
        can_turn_left = lane >= (LANES_PER_DIRECTION - LEFT_TURN_LANES)
        
        # Determine turn intention: straight, right, or left
        # Probability ranges: [0, 0.3) = right, [0.3, 0.5) = left, [0.5, 1.0] = straight
        rand = random.random()
        if can_turn_right and rand < TURN_RIGHT_PROBABILITY:
            self.turn_type = 'right'
        elif can_turn_left and rand >= TURN_RIGHT_PROBABILITY and rand < TURN_RIGHT_PROBABILITY + TURN_LEFT_PROBABILITY:
            self.turn_type = 'left'
        else:
            self.turn_type = 'straight'
        
        self.has_turned = False  # Track if turn has been executed
        
        # Set destination label based on turn type
        self.destination = DESTINATION_MAP[direction][self.turn_type]
        
        # Destination lane matches source lane (nth lane -> nth lane)
        self.destination_lane = self.lane
        
        # Set initial position based on direction
        self.x, self.y = self._get_spawn_position()
        
        # Intersection state tracking
        self.crossed = False  # Track if vehicle crossed intersection
        self.in_intersection = False  # Track if vehicle is currently in intersection
        self.current_lane = None  # Reference to current Lane object
        self.waiting_at_signal = False  # Waiting at red light
        
    def _get_spawn_position(self):
        """Calculate spawn position based on direction and lane (right-side driving Polish style)
        Lane 0 = rightmost lane from driver's perspective (outer/slow lane, at road edge)
        Lane 1 = leftmost lane from driver's perspective (inner/fast lane, near center)
        
        In right-hand traffic (top-down view, north=up):
        - Eastbound (right): SOUTH half (y > center_y)
        - Westbound (left): NORTH half (y < center_y)  
        - Southbound (down): WEST half (x < center_x)
        - Northbound (up): EAST half (x > center_x)
        """
        center_x = SCREEN_WIDTH // 2
        center_y = SCREEN_HEIGHT // 2
        road_width = LANES_PER_DIRECTION * LANE_WIDTH
        
        if self.original_direction == 'right':
            # Coming from West, driving on SOUTH half of horizontal road
            # Lane 0 (outer) = southernmost = highest y
            x = -self.size - 10
            y = center_y + road_width - self.lane * LANE_WIDTH - LANE_WIDTH // 2
            return x, y
        elif self.original_direction == 'down':
            # Coming from North, driving on WEST half of vertical road
            # Lane 0 (outer) = westernmost = lowest x
            x = center_x - road_width + self.lane * LANE_WIDTH + LANE_WIDTH // 2
            y = -self.size - 10
            return x, y
        elif self.original_direction == 'left':
            # Coming from East, driving on NORTH half of horizontal road
            # Lane 0 (outer) = northernmost = lowest y
            x = SCREEN_WIDTH + 10
            y = center_y - road_width + self.lane * LANE_WIDTH + LANE_WIDTH // 2
            return x, y
        else:  # up
            # Coming from South, driving on EAST half of vertical road
            # Lane 0 (outer) = easternmost = highest x
            x = center_x + road_width - self.lane * LANE_WIDTH - LANE_WIDTH // 2
            y = SCREEN_HEIGHT + 10
            return x, y
    
    def get_stop_position(self):
        """Get the position where vehicle should stop at red light (based on original direction)"""
        center_x = SCREEN_WIDTH // 2
        center_y = SCREEN_HEIGHT // 2
        road_width = LANES_PER_DIRECTION * LANE_WIDTH
        
        if self.original_direction == 'right':
            return center_x - road_width - 20
        elif self.original_direction == 'down':
            return center_y - road_width - 20
        elif self.original_direction == 'left':
            return center_x + road_width + 20
        else:  # up
            return center_y + road_width + 20
    
    def is_before_stop_line(self):
        """Check if vehicle hasn't yet reached the stop line"""
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
        """Check if vehicle should stop (for red light, vehicles ahead, or intersection conflicts)"""
        # If already in intersection, don't stop
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
                
                self.waiting_at_signal = False
        
        # Check for vehicle ahead in same lane
        if self.current_lane:
            vehicle_ahead = self.current_lane.get_vehicle_ahead(self)
            if vehicle_ahead and self._is_too_close(vehicle_ahead):
                return True
        
        return False
    
    def _is_too_close(self, other):
        """Check if too close to another vehicle ahead"""
        if self.direction == 'right':
            return other.x > self.x and other.x - self.x < SAFE_DISTANCE + self.size
        elif self.direction == 'down':
            return other.y > self.y and other.y - self.y < SAFE_DISTANCE + self.size
        elif self.direction == 'left':
            return other.x < self.x and self.x - other.x < SAFE_DISTANCE + self.size
        else:  # up
            return other.y < self.y and self.y - other.y < SAFE_DISTANCE + self.size
    
    def _execute_turn(self):
        """Change direction and lane for turn"""
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
        """Check if vehicle has reached the point where it should turn
        
        Right turns: turn early, just after crossing stop line
        Left turns: turn late, near far side of intersection
        """
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
        """Update vehicle position
        
        Args:
            signal: TrafficSignal for this vehicle's entry direction
            intersection: Intersection object
            lanes: Dict of Lane objects {(direction, lane_num): Lane}
        """
        if self.should_stop(signal, intersection):
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
    
    def is_off_screen(self):
        """Check if vehicle has left the screen"""
        margin = 50
        return (self.x < -margin or self.x > SCREEN_WIDTH + margin or
                self.y < -margin or self.y > SCREEN_HEIGHT + margin)
    
    def draw(self, screen, font):
        """Draw the vehicle as a perfect square with destination letter"""
        rect = pygame.Rect(self.x - self.size // 2, self.y - self.size // 2,
                          self.size, self.size)
        pygame.draw.rect(screen, self.color, rect)
        # Add border for better visibility
        pygame.draw.rect(screen, (0, 0, 0), rect, 2)
        
        # Draw destination letter on the vehicle
        label = font.render(self.destination, True, (255, 255, 255))
        label_rect = label.get_rect(center=(self.x, self.y))
        screen.blit(label, label_rect)

# ==================== MAIN SIMULATOR CLASS ====================
class TrafficSimulator:
    """Main simulator managing all components"""
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Traffic Intersection Simulator")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        self.vehicle_font = pygame.font.Font(None, 20)  # Font for vehicle destination labels
        
        # Initialize traffic signals
        center_x = SCREEN_WIDTH // 2
        center_y = SCREEN_HEIGHT // 2
        road_width = LANES_PER_DIRECTION * LANE_WIDTH
        
        # Position signals on the right side of each lane (right-hand traffic)
        self.signals = {
            'right': TrafficSignal('right', (center_x - road_width - 60, center_y + road_width + 60)),  # A: bottom-left (south side)
            'down': TrafficSignal('down', (center_x - road_width - 60, center_y - road_width - 60)),   # B: top-left (west side)
            'left': TrafficSignal('left', (center_x + road_width + 60, center_y - road_width - 60)),   # C: top-right (north side)
            'up': TrafficSignal('up', (center_x + road_width + 60, center_y + road_width + 60))        # D: bottom-right (east side)
        }
        
        # Initial signal state is set by _signal_controller thread
        
        # Initialize lanes
        self.lanes = {}
        for direction in ['right', 'down', 'left', 'up']:
            for lane_num in range(LANES_PER_DIRECTION):
                lane_key = (direction, lane_num)
                self.lanes[lane_key] = Lane(direction, lane_num)
        
        # Initialize intersection
        self.intersection = Intersection()
        
        # Vehicle management
        self.vehicles = []
        self.running = True
        
        # Statistics for AI training
        self.vehicles_crossed = {'right': 0, 'down': 0, 'left': 0, 'up': 0}
        
        # Start background threads
        self.signal_thread = threading.Thread(target=self._signal_controller, daemon=True)
        self.spawn_thread = threading.Thread(target=self._vehicle_spawner, daemon=True)
        self.signal_thread.start()
        self.spawn_thread.start()
    
    def _signal_controller(self):
        """Background thread to manage signal timing using SIGNAL_PHASES configuration"""
        current_phase_idx = 0
        all_directions = ['right', 'down', 'left', 'up']
        
        # Set initial phase
        first_phase = SIGNAL_PHASES[0]
        first_arrows = first_phase.get('arrows', [])
        
        # Initialize all signals to red with arrows off
        for direction in all_directions:
            self.signals[direction].state = 'red'
            self.signals[direction].turn_arrow_enabled = False
            self.signals[direction].turn_arrow_state = 'red'
        
        # Turn on first phase
        for direction in first_phase['green']:
            self.signals[direction].state = 'green'
            self.signals[direction].timer = first_phase['duration']
            
            # Enable turn arrows for directions specified in phase
            if direction in first_arrows:
                self.signals[direction].turn_arrow_enabled = True
                self.signals[direction].turn_arrow_state = 'green'
        
        while self.running:
            time.sleep(0.1)
            current_phase = SIGNAL_PHASES[current_phase_idx]
            green_directions = current_phase['green']
            arrow_directions = current_phase.get('arrows', [])
            
            # Update timers for all green signals in current phase (synchronized)
            for direction in green_directions:
                signal = self.signals[direction]
                # Apply simulation speed to timer (faster speed = faster signal changes)
                signal.timer -= 0.1 * SIMULATION_SPEED
            
            # Check if phase needs to change (use first green signal as reference)
            primary_signal = self.signals[green_directions[0]]
            
            if primary_signal.timer <= 0:
                if primary_signal.state == 'green':
                    # Change all green signals to yellow
                    for direction in green_directions:
                        self.signals[direction].state = 'yellow'
                        self.signals[direction].timer = SIGNAL_YELLOW_TIME
                        
                elif primary_signal.state == 'yellow':
                    # Change all signals to red
                    for direction in green_directions:
                        self.signals[direction].state = 'red'
                        self.signals[direction].timer = SIGNAL_RED_TIME
                        # Turn off arrows when red
                        self.signals[direction].turn_arrow_state = 'red'
                        
                else:  # red - move to next phase
                    # Move to next phase
                    current_phase_idx = (current_phase_idx + 1) % len(SIGNAL_PHASES)
                    next_phase = SIGNAL_PHASES[current_phase_idx]
                    next_arrow_directions = next_phase.get('arrows', [])
                    
                    # Turn off all arrows first
                    for direction in all_directions:
                        self.signals[direction].turn_arrow_enabled = False
                        self.signals[direction].turn_arrow_state = 'red'
                    
                    # Turn next phase green and enable specified arrows
                    for direction in next_phase['green']:
                        self.signals[direction].state = 'green'
                        self.signals[direction].timer = next_phase['duration']
                        
                        # Enable arrow if specified for this direction in this phase
                        if direction in next_arrow_directions:
                            self.signals[direction].turn_arrow_enabled = True
                            self.signals[direction].turn_arrow_state = 'green'
    
    def _vehicle_spawner(self):
        """Background thread to spawn vehicles"""
        while self.running:
            # Apply simulation speed to spawn rate (faster speed = faster spawns)
            spawn_time = (SPAWN_INTERVAL + random.uniform(-0.3, 0.3)) / SIMULATION_SPEED
            time.sleep(spawn_time)
            
            # Choose random direction and lane
            direction = random.choice(['right', 'down', 'left', 'up'])
            lane_num = random.randint(0, LANES_PER_DIRECTION - 1)
            
            # Choose vehicle type based on weights
            types = list(VEHICLE_TYPES.keys())
            weights = [VEHICLE_TYPES[t]['spawn_weight'] for t in types]
            v_type = random.choices(types, weights=weights)[0]
            
            # Create vehicle
            vehicle = Vehicle(v_type, direction, lane_num)
            
            # Assign to lane
            lane_key = (direction, lane_num)
            if lane_key in self.lanes:
                vehicle.current_lane = self.lanes[lane_key]
                vehicle.current_lane.add_vehicle(vehicle)
            
            self.vehicles.append(vehicle)
    
    def draw_roads(self):
        """Draw the road grid"""
        self.screen.fill((100, 150, 100))  # Grass background
        
        center_x = SCREEN_WIDTH // 2
        center_y = SCREEN_HEIGHT // 2
        road_width = LANES_PER_DIRECTION * LANE_WIDTH
        
        # Draw horizontal road
        pygame.draw.rect(self.screen, ROAD_COLOR,
                        (0, center_y - road_width, SCREEN_WIDTH, road_width * 2))
        
        # Draw vertical road
        pygame.draw.rect(self.screen, ROAD_COLOR,
                        (center_x - road_width, 0, road_width * 2, SCREEN_HEIGHT))
        
        # Draw lane dividers
        for i in range(1, LANES_PER_DIRECTION):
            # Horizontal lanes
            y = center_y - road_width + i * LANE_WIDTH
            for x in range(0, SCREEN_WIDTH, 40):
                pygame.draw.line(self.screen, LANE_LINE_COLOR, (x, y), (x + 20, y), 2)
            y = center_y + i * LANE_WIDTH
            for x in range(0, SCREEN_WIDTH, 40):
                pygame.draw.line(self.screen, LANE_LINE_COLOR, (x, y), (x + 20, y), 2)
            
            # Vertical lanes
            x = center_x - road_width + i * LANE_WIDTH
            for y in range(0, SCREEN_HEIGHT, 40):
                pygame.draw.line(self.screen, LANE_LINE_COLOR, (x, y), (x, y + 20), 2)
            x = center_x + i * LANE_WIDTH
            for y in range(0, SCREEN_HEIGHT, 40):
                pygame.draw.line(self.screen, LANE_LINE_COLOR, (x, y), (x, y + 20), 2)
        
        # Draw stop lines (on right side of road for Polish/right-hand traffic)
        # Right direction (A lane) - vehicles on SOUTH half
        pygame.draw.line(self.screen, STOP_LINE_COLOR,
                        (center_x - road_width - 10, center_y),
                        (center_x - road_width - 10, center_y + road_width), 5)
        # Down direction (B lane) - vehicles on WEST half
        pygame.draw.line(self.screen, STOP_LINE_COLOR,
                        (center_x - road_width, center_y - road_width - 10),
                        (center_x, center_y - road_width - 10), 5)
        # Left direction (C lane) - vehicles on NORTH half
        pygame.draw.line(self.screen, STOP_LINE_COLOR,
                        (center_x + road_width + 10, center_y - road_width),
                        (center_x + road_width + 10, center_y), 5)
        # Up direction (D lane) - vehicles on EAST half
        pygame.draw.line(self.screen, STOP_LINE_COLOR,
                        (center_x, center_y + road_width + 10),
                        (center_x + road_width, center_y + road_width + 10), 5)
        
        # Draw lane labels (A-D) at road entry points
        # A - West entry (for 'right' direction vehicles on SOUTH half)
        label_a = self.font.render('A', True, (255, 255, 255))
        self.screen.blit(label_a, (50, center_y + road_width // 2 - 10))
        # B - North entry (for 'down' direction vehicles on WEST half)
        label_b = self.font.render('B', True, (255, 255, 255))
        self.screen.blit(label_b, (center_x - road_width // 2 - 10, 50))
        # C - East entry (for 'left' direction vehicles on NORTH half)
        label_c = self.font.render('C', True, (255, 255, 255))
        self.screen.blit(label_c, (SCREEN_WIDTH - 70, center_y - road_width // 2 - 10))
        # D - South entry (for 'up' direction vehicles on EAST half)
        label_d = self.font.render('D', True, (255, 255, 255))
        self.screen.blit(label_d, (center_x + road_width // 2 - 10, SCREEN_HEIGHT - 70))
    
    def draw_signals(self):
        """Draw traffic signals with countdown timers"""
        for direction, signal in self.signals.items():
            x, y = signal.position
            
            # Draw main signal
            pygame.draw.circle(self.screen, (0, 0, 0), (x, y), SIGNAL_RADIUS + 3)
            pygame.draw.circle(self.screen, signal.get_color(), (x, y), SIGNAL_RADIUS)
            
            # Draw turn arrow if enabled
            if signal.turn_arrow_enabled:
                arrow_x = x + 40
                pygame.draw.circle(self.screen, (0, 0, 0), (arrow_x, y), SIGNAL_RADIUS // 2 + 2)
                pygame.draw.circle(self.screen, signal.get_turn_color(), (arrow_x, y), SIGNAL_RADIUS // 2)
            
            # Draw countdown timer
            timer_text = self.small_font.render(f"{signal.timer:.1f}s", True, (255, 255, 255))
            self.screen.blit(timer_text, (x - 20, y + 30))
    
    def draw_stats(self):
        """Draw statistics for AI training"""
        y_offset = 10
        for direction, count in self.vehicles_crossed.items():
            lane_label = LANE_LABELS[direction]
            text = self.small_font.render(f"Lane {lane_label} ({direction}): {count} crossed", True, (255, 255, 255))
            self.screen.blit(text, (10, y_offset))
            y_offset += 25
        
        # Draw legend for destination labels
        y_offset += 10
        legend_title = self.small_font.render("Destination Labels:", True, (255, 255, 0))
        self.screen.blit(legend_title, (10, y_offset))
        y_offset += 20
        legend = self.small_font.render("A=West  B=North  C=East  D=South", True, (255, 255, 255))
        self.screen.blit(legend, (10, y_offset))
        
        # Draw controls
        controls = self.small_font.render("Press 1-4 to toggle turn arrows: 1=A 2=B 3=C 4=D", True, (255, 255, 255))
        self.screen.blit(controls, (10, SCREEN_HEIGHT - 30))
    
    def update(self):
        """Update all vehicles"""
        for vehicle in self.vehicles[:]:
            # Use original_direction for signal checking (which lane the vehicle entered from)
            signal = self.signals[vehicle.original_direction]
            vehicle.move(signal, self.intersection, self.lanes)
            
            # Track crossed vehicles (by their original entry direction)
            if vehicle.crossed and vehicle.original_direction in self.vehicles_crossed:
                # Only count once
                if hasattr(vehicle, 'counted'):
                    pass
                else:
                    self.vehicles_crossed[vehicle.original_direction] += 1
                    vehicle.counted = True
            
            # Remove vehicles that left the screen
            if vehicle.is_off_screen():
                # Remove from lane
                if vehicle.current_lane:
                    vehicle.current_lane.remove_vehicle(vehicle)
                # Remove from intersection if still there
                if vehicle.in_intersection:
                    self.intersection.exit(vehicle)
                # Remove from vehicles list
                self.vehicles.remove(vehicle)
    
    def run(self):
        """Main game loop"""
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

# ==================== MAIN ENTRY POINT ====================
if __name__ == "__main__":
    simulator = TrafficSimulator()
    simulator.run()