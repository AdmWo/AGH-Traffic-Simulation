"""
Level definitions for the traffic simulation.

This module contains:
- RoadSegment: A section of road with limited capacity
- Level: Base class for traffic network configurations
- Level1: Single 4-way intersection (the classic setup)
- Level2: Two connected intersections
- Level3: 2x2 grid of 4 intersections with 8 entry points

Each level provides its own state vector, action space, and spawn points.
The idea is that one AI agent controls all intersections in a level together.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
import random


# ------------------------------------------------------------------
# RoadSegment - a section of road with capacity limits
# ------------------------------------------------------------------
class RoadSegment:
    """
    A stretch of road that can hold a limited number of vehicles.
    
    Used for connecting roads between intersections where we need to
    prevent gridlock by checking capacity before cars enter.
    
    For entry/exit segments (edges of the map), capacity is usually
    set very high since they represent infinite roads.
    """
    
    def __init__(self, segment_id: str, capacity: int = 20, 
                 from_intersection: str = None, to_intersection: str = None):
        self.id = segment_id
        self.capacity = capacity
        self.vehicles: List = []  # actual Vehicle objects on this segment
        self.from_intersection = from_intersection  # where cars come from
        self.to_intersection = to_intersection  # where cars are headed
        self.spawn_rate = 0.5  # default spawn probability for entry segments
        
    def can_enter(self) -> bool:
        """Check if there's room for another car."""
        return len(self.vehicles) < self.capacity
    
    def get_occupancy(self) -> float:
        """Returns 0.0-1.0 showing how full the segment is."""
        return min(len(self.vehicles) / self.capacity, 1.0) if self.capacity > 0 else 0.0
    
    def get_queue_pressure(self) -> float:
        """Returns normalized count of waiting vehicles (for state vector)."""
        waiting = sum(1 for v in self.vehicles if hasattr(v, 'wait_time') and v.wait_time > 0)
        return min(waiting / 20.0, 1.0)  # normalize by 20
    
    def add_vehicle(self, vehicle):
        """Add a vehicle to this segment."""
        if vehicle not in self.vehicles:
            self.vehicles.append(vehicle)
    
    def remove_vehicle(self, vehicle):
        """Remove a vehicle from this segment."""
        if vehicle in self.vehicles:
            self.vehicles.remove(vehicle)
    
    def __repr__(self):
        return f"RoadSegment({self.id}, {len(self.vehicles)}/{self.capacity})"


# ------------------------------------------------------------------
# IntersectionNode - a generic intersection that can be configured
# ------------------------------------------------------------------
class IntersectionNode:
    """
    A single intersection within a level.
    
    Unlike the old Intersection class which was hardcoded for 4 directions,
    this one is more flexible - it knows which segments connect to it
    and manages signals for each incoming direction.
    """
    
    def __init__(self, node_id: str, position: Tuple[int, int]):
        self.id = node_id
        self.position = position  # (x, y) screen position
        
        # Maps direction -> incoming RoadSegment
        self.incoming_segments: Dict[str, RoadSegment] = {}
        
        # Maps direction -> outgoing RoadSegment  
        self.outgoing_segments: Dict[str, RoadSegment] = {}
        
        # Signal states for each incoming direction
        # Values: 'red', 'yellow', 'green'
        self.signal_states: Dict[str, str] = {}
        
        # Turn arrow states for each incoming direction
        self.arrow_states: Dict[str, bool] = {}
        
        # Vehicles currently in the intersection (for conflict detection)
        self.vehicles_in_intersection: Set = set()
        
    def connect_incoming(self, direction: str, segment: RoadSegment):
        """Connect an incoming road segment from a direction."""
        self.incoming_segments[direction] = segment
        self.signal_states[direction] = 'red'
        self.arrow_states[direction] = False
        
    def connect_outgoing(self, direction: str, segment: RoadSegment):
        """Connect an outgoing road segment to a direction."""
        self.outgoing_segments[direction] = segment
        
    def set_signal(self, direction: str, state: str):
        """Set the main signal for a direction."""
        if direction in self.signal_states:
            self.signal_states[direction] = state
            
    def set_arrow(self, direction: str, enabled: bool):
        """Enable/disable the turn arrow for a direction."""
        if direction in self.arrow_states:
            self.arrow_states[direction] = enabled
            
    def can_go(self, direction: str) -> bool:
        """Check if a vehicle from this direction can go (main signal green)."""
        return self.signal_states.get(direction, 'red') == 'green'
    
    def can_turn(self, direction: str) -> bool:
        """Check if a vehicle from this direction can turn (arrow or main green)."""
        if self.arrow_states.get(direction, False):
            return True
        return self.can_go(direction)
    
    def enter(self, vehicle):
        """Mark a vehicle as being in the intersection."""
        self.vehicles_in_intersection.add(vehicle)
        
    def exit(self, vehicle):
        """Mark a vehicle as having left the intersection."""
        self.vehicles_in_intersection.discard(vehicle)
        
    def get_state_vector(self) -> List[float]:
        """
        Returns the state for this intersection.
        Format: [signal_states..., arrow_states...]
        Each direction contributes 1 signal value + 1 arrow value.
        """
        state = []
        directions = sorted(self.signal_states.keys())
        
        for d in directions:
            state.append(1.0 if self.signal_states[d] == 'green' else 0.0)
        for d in directions:
            state.append(1.0 if self.arrow_states.get(d, False) else 0.0)
            
        return state


# ------------------------------------------------------------------
# Level - base class for traffic network configurations
# ------------------------------------------------------------------
class Level:
    """
    Base class for a traffic network level.
    
    A level defines:
    - How many intersections there are and where they're positioned
    - Which road segments connect them
    - Where vehicles spawn and exit
    - The state vector format
    - The action space (what configurations the AI can set)
    
    v3: Simple action space - just signal configurations, no duration control.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.intersections: Dict[str, IntersectionNode] = {}
        self.segments: Dict[str, RoadSegment] = {}
        self.spawn_points: List[str] = []  # segment IDs where vehicles can spawn
        self.exit_points: List[str] = []   # segment IDs where vehicles exit
        self.throughput: Dict[str, int] = {}  # count of vehicles exited per exit point
        
        # Action definitions - each action is a dict of intersection -> phase
        self.actions: List[Dict] = []
        
    def get_state(self) -> List[float]:
        """
        Returns the full state vector for RL.
        Override in subclasses for specific formats.
        """
        state = []
        
        # Add segment queue pressures
        for seg_id in sorted(self.spawn_points):
            if seg_id in self.segments:
                state.append(self.segments[seg_id].get_queue_pressure())
                
        # Add connecting segment occupancies (if any)
        connecting = [s for s in self.segments.values() 
                     if s.from_intersection and s.to_intersection]
        for seg in sorted(connecting, key=lambda s: s.id):
            state.append(seg.get_occupancy())
        
        # Add intersection states
        for int_id in sorted(self.intersections.keys()):
            state.extend(self.intersections[int_id].get_state_vector())
            
        # Add throughput (normalized)
        for exit_id in sorted(self.exit_points):
            throughput = self.throughput.get(exit_id, 0)
            state.append(min(throughput / 100.0, 1.0))
            
        return state
    
    def get_state_size(self) -> int:
        """Returns the size of the state vector."""
        return len(self.get_state())
    
    def get_action_count(self) -> int:
        """Returns total number of actions (simple - just signal configs)."""
        return len(self.actions)
    
    def apply_action(self, action_id: int) -> str:
        """Apply an action to set signal configurations.
        
        Returns: action_name
        """
        if action_id < 0 or action_id >= len(self.actions):
            return "Invalid"
        
        # Apply signal configuration
        action = self.actions[action_id]
        for int_id, config in action.items():
            if int_id in self.intersections:
                intersection = self.intersections[int_id]
                green_dirs = config.get('green', [])
                arrow_dirs = config.get('arrows', [])
                
                # Set all signals based on config
                for direction in intersection.signal_states.keys():
                    intersection.set_signal(direction, 'green' if direction in green_dirs else 'red')
                    intersection.set_arrow(direction, direction in arrow_dirs)
        
        # Get action name
        if hasattr(self, 'action_names') and action_id < len(self.action_names):
            return self.action_names[action_id]
        else:
            return f"Action {action_id}"
    
    def get_metrics(self) -> dict:
        """Returns detailed metrics for reward calculation."""
        total_wait_time = 0
        waiting_vehicles = 0
        total_vehicles = 0
        
        for segment in self.segments.values():
            for vehicle in segment.vehicles:
                total_vehicles += 1
                if hasattr(vehicle, 'wait_time') and vehicle.wait_time > 0:
                    total_wait_time += vehicle.wait_time
                    waiting_vehicles += 1
        
        avg_wait_time = total_wait_time / total_vehicles if total_vehicles > 0 else 0
        total_throughput = sum(self.throughput.values())
        
        return {
            'total_wait_time': total_wait_time,
            'avg_wait_time': avg_wait_time,
            'total_throughput': total_throughput,
            'waiting_vehicles': waiting_vehicles,
            'total_vehicles': total_vehicles,
            'throughput_by_exit': dict(self.throughput)
        }
    
    def record_exit(self, exit_segment_id: str):
        """Record a vehicle exiting through a segment."""
        if exit_segment_id not in self.throughput:
            self.throughput[exit_segment_id] = 0
        self.throughput[exit_segment_id] += 1
    
    def get_spawn_rates(self) -> Dict[str, float]:
        """Returns current spawn rates for all spawn points."""
        return {seg_id: self.segments[seg_id].spawn_rate 
                for seg_id in self.spawn_points 
                if seg_id in self.segments}
    
    def set_spawn_rates(self, rates: Dict[str, float]):
        """Set spawn rates for spawn points (used by sliders)."""
        for seg_id, rate in rates.items():
            if seg_id in self.segments:
                self.segments[seg_id].spawn_rate = max(0.0, min(1.0, rate))
                
    def randomize_spawn_rates(self, min_rate: float = 0.2, max_rate: float = 1.0):
        """Randomize spawn rates for training diversity."""
        for seg_id in self.spawn_points:
            if seg_id in self.segments:
                self.segments[seg_id].spawn_rate = random.uniform(min_rate, max_rate)


# ------------------------------------------------------------------
# Level1 - Single 4-way intersection (the classic)
# ------------------------------------------------------------------
class Level1(Level):
    """
    The basic single intersection setup.
    
    Layout:
           [B]
            |
        [A]-+-[C]
            |
           [D]
    
    - 4 entry segments (A, B, C, D)
    - 4 exit segments  
    - 1 intersection
    - 8 actions (same as current simulation)
    """
    
    def __init__(self):
        super().__init__("Level 1: Single Intersection")
        self._build_network()
        self._define_actions()
        
    def _build_network(self):
        # Create the central intersection
        center_x, center_y = 700, 400  # roughly center of 1400x800
        self.intersections['INT1'] = IntersectionNode('INT1', (center_x, center_y))
        
        # Create entry segments (high capacity, they're infinite roads)
        # Using direction labels that match the original: right=A, down=B, left=C, up=D
        directions = ['right', 'down', 'left', 'up']  # incoming from A, B, C, D
        for d in directions:
            seg_id = f"entry_{d}"
            self.segments[seg_id] = RoadSegment(seg_id, capacity=100, to_intersection='INT1')
            self.spawn_points.append(seg_id)
            self.intersections['INT1'].connect_incoming(d, self.segments[seg_id])
        
        # Create exit segments
        for d in directions:
            seg_id = f"exit_{d}"
            self.segments[seg_id] = RoadSegment(seg_id, capacity=100, from_intersection='INT1')
            self.exit_points.append(seg_id)
            self.intersections['INT1'].connect_outgoing(d, self.segments[seg_id])
            self.throughput[seg_id] = 0
            
    def _define_actions(self):
        """
        12 actions for Level 1:
        0-3: East-West (right-left) green with different arrow combos
        4-7: North-South (down-up) green with different arrow combos
        8-11: Single direction green (for more granular AI control)
        """
        self.actions = [
            # EW green variations (both directions together - traditional)
            {'INT1': {'green': ['right', 'left'], 'arrows': []}},
            {'INT1': {'green': ['right', 'left'], 'arrows': ['right', 'left']}},
            {'INT1': {'green': ['right', 'left'], 'arrows': ['down', 'up']}},
            {'INT1': {'green': ['right', 'left'], 'arrows': ['right', 'down', 'left', 'up']}},
            # NS green variations (both directions together - traditional)
            {'INT1': {'green': ['down', 'up'], 'arrows': []}},
            {'INT1': {'green': ['down', 'up'], 'arrows': ['down', 'up']}},
            {'INT1': {'green': ['down', 'up'], 'arrows': ['right', 'left']}},
            {'INT1': {'green': ['down', 'up'], 'arrows': ['right', 'down', 'left', 'up']}},
            # Single direction green (for more control - AI can clear one queue at a time)
            {'INT1': {'green': ['right'], 'arrows': ['right']}},  # Only West approach (A)
            {'INT1': {'green': ['left'], 'arrows': ['left']}},    # Only East approach (C)
            {'INT1': {'green': ['down'], 'arrows': ['down']}},    # Only North approach (B)
            {'INT1': {'green': ['up'], 'arrows': ['up']}},        # Only South approach (D)
        ]
        
        self.action_names = [
            'EW Green',
            'EW Green + EW Arrows',
            'EW Green + NS Arrows',
            'EW Green + All Arrows',
            'NS Green',
            'NS Green + NS Arrows',
            'NS Green + EW Arrows',
            'NS Green + All Arrows',
            'Only A (West)',
            'Only C (East)',
            'Only B (North)',
            'Only D (South)',
        ]


# ------------------------------------------------------------------
# Level2 - Two connected intersections
# ------------------------------------------------------------------
class Level2(Level):
    """
    Two intersections connected by a road with limited capacity.
    
    Layout:
        [B1]      [B2]
         |         |
     [A]-+-[M]----+-[C]
         |         |
        [D1]      [D2]
    
    - [M] is a connecting segment with capacity=5
    - Cars going A->C must wait if [M] is full
    - 2 intersections controlled together
    """
    
    def __init__(self):
        super().__init__("Level 2: Dual Intersections")
        self._build_network()
        self._define_actions()
        
    def _build_network(self):
        # Two intersections horizontally
        self.intersections['INT1'] = IntersectionNode('INT1', (500, 400))
        self.intersections['INT2'] = IntersectionNode('INT2', (900, 400))
        
        # Entry segments for INT1
        self.segments['entry_A'] = RoadSegment('entry_A', capacity=100, to_intersection='INT1')
        self.segments['entry_B1'] = RoadSegment('entry_B1', capacity=100, to_intersection='INT1')
        self.segments['entry_D1'] = RoadSegment('entry_D1', capacity=100, to_intersection='INT1')
        
        self.spawn_points.extend(['entry_A', 'entry_B1', 'entry_D1'])
        
        self.intersections['INT1'].connect_incoming('right', self.segments['entry_A'])
        self.intersections['INT1'].connect_incoming('down', self.segments['entry_B1'])
        self.intersections['INT1'].connect_incoming('up', self.segments['entry_D1'])
        
        # Entry segments for INT2
        self.segments['entry_C'] = RoadSegment('entry_C', capacity=100, to_intersection='INT2')
        self.segments['entry_B2'] = RoadSegment('entry_B2', capacity=100, to_intersection='INT2')
        self.segments['entry_D2'] = RoadSegment('entry_D2', capacity=100, to_intersection='INT2')
        
        self.spawn_points.extend(['entry_C', 'entry_B2', 'entry_D2'])
        
        self.intersections['INT2'].connect_incoming('left', self.segments['entry_C'])
        self.intersections['INT2'].connect_incoming('down', self.segments['entry_B2'])
        self.intersections['INT2'].connect_incoming('up', self.segments['entry_D2'])
        
        # The connecting segment between intersections (limited capacity!)
        self.segments['M_east'] = RoadSegment('M_east', capacity=5, 
                                               from_intersection='INT1', to_intersection='INT2')
        self.segments['M_west'] = RoadSegment('M_west', capacity=5,
                                               from_intersection='INT2', to_intersection='INT1')
        
        self.intersections['INT1'].connect_outgoing('left', self.segments['M_east'])
        self.intersections['INT2'].connect_incoming('right', self.segments['M_east'])
        
        self.intersections['INT2'].connect_outgoing('right', self.segments['M_west'])
        self.intersections['INT1'].connect_incoming('left', self.segments['M_west'])
        
        # Exit segments
        self.segments['exit_A'] = RoadSegment('exit_A', capacity=100, from_intersection='INT1')
        self.segments['exit_B1'] = RoadSegment('exit_B1', capacity=100, from_intersection='INT1')
        self.segments['exit_D1'] = RoadSegment('exit_D1', capacity=100, from_intersection='INT1')
        self.segments['exit_C'] = RoadSegment('exit_C', capacity=100, from_intersection='INT2')
        self.segments['exit_B2'] = RoadSegment('exit_B2', capacity=100, from_intersection='INT2')
        self.segments['exit_D2'] = RoadSegment('exit_D2', capacity=100, from_intersection='INT2')
        
        self.exit_points = ['exit_A', 'exit_B1', 'exit_D1', 'exit_C', 'exit_B2', 'exit_D2']
        for ep in self.exit_points:
            self.throughput[ep] = 0
            
        self.intersections['INT1'].connect_outgoing('right', self.segments['exit_A'])
        self.intersections['INT1'].connect_outgoing('down', self.segments['exit_B1'])
        self.intersections['INT1'].connect_outgoing('up', self.segments['exit_D1'])
        
        self.intersections['INT2'].connect_outgoing('left', self.segments['exit_C'])
        self.intersections['INT2'].connect_outgoing('down', self.segments['exit_B2'])
        self.intersections['INT2'].connect_outgoing('up', self.segments['exit_D2'])
        
    def _define_actions(self):
        """
        8 actions for coordinated control:
        0-3: Both intersections same phase (synchronized)
        4-7: Offset patterns (green wave for through traffic)
        """
        # Same phase for both
        self.actions = [
            # Both EW
            {'INT1': {'green': ['right', 'left'], 'arrows': []},
             'INT2': {'green': ['right', 'left'], 'arrows': []}},
            # Both EW + arrows
            {'INT1': {'green': ['right', 'left'], 'arrows': ['right', 'left']},
             'INT2': {'green': ['right', 'left'], 'arrows': ['right', 'left']}},
            # Both NS
            {'INT1': {'green': ['down', 'up'], 'arrows': []},
             'INT2': {'green': ['down', 'up'], 'arrows': []}},
            # Both NS + arrows
            {'INT1': {'green': ['down', 'up'], 'arrows': ['down', 'up']},
             'INT2': {'green': ['down', 'up'], 'arrows': ['down', 'up']}},
            # Offset: INT1 EW, INT2 NS (helps diagonal flow)
            {'INT1': {'green': ['right', 'left'], 'arrows': []},
             'INT2': {'green': ['down', 'up'], 'arrows': []}},
            # Offset: INT1 NS, INT2 EW
            {'INT1': {'green': ['down', 'up'], 'arrows': []},
             'INT2': {'green': ['right', 'left'], 'arrows': []}},
            # Green wave eastbound: both EW with arrows for flow
            {'INT1': {'green': ['right', 'left'], 'arrows': ['right']},
             'INT2': {'green': ['right', 'left'], 'arrows': ['left']}},
            # Green wave westbound
            {'INT1': {'green': ['right', 'left'], 'arrows': ['left']},
             'INT2': {'green': ['right', 'left'], 'arrows': ['right']}},
        ]
        
        self.action_names = [
            'Both EW',
            'Both EW + Arrows',
            'Both NS', 
            'Both NS + Arrows',
            'Offset: INT1-EW INT2-NS',
            'Offset: INT1-NS INT2-EW',
            'Green Wave East',
            'Green Wave West',
        ]


# ------------------------------------------------------------------
# Level3 - Four-intersection 2x2 grid
# ------------------------------------------------------------------
class Level3(Level):
    """
    A 2x2 grid of 4 intersections with 8 entry/exit points.
    
    Layout:
           [N1]       [N2]
            |          |
    [W1]--[INT1]--H1--[INT2]--[E1]
            |          |
           V1         V2
            |          |
    [W2]--[INT3]--H2--[INT4]--[E2]
            |          |
           [S1]       [S2]
    
    - 8 entry points around the edge
    - 4 connecting segments (H1, H2, V1, V2) with limited capacity
    - All 4 intersections controlled by single AI
    - Cars can travel diagonally through the grid
    """
    
    def __init__(self):
        super().__init__("Level 3: 4-Way Grid")
        self._build_network()
        self._define_actions()
        
    def _build_network(self):
        # 4 intersections in a 2x2 grid
        #   INT1  INT2
        #   INT3  INT4
        self.intersections['INT1'] = IntersectionNode('INT1', (450, 300))
        self.intersections['INT2'] = IntersectionNode('INT2', (950, 300))
        self.intersections['INT3'] = IntersectionNode('INT3', (450, 500))
        self.intersections['INT4'] = IntersectionNode('INT4', (950, 500))
        
        # Entry segments (8 total, around the edges)
        edge_entries = {
            'entry_N1': ('INT1', 'down'),   # from north to INT1
            'entry_N2': ('INT2', 'down'),   # from north to INT2
            'entry_E1': ('INT2', 'left'),   # from east to INT2
            'entry_E2': ('INT4', 'left'),   # from east to INT4
            'entry_S1': ('INT3', 'up'),     # from south to INT3
            'entry_S2': ('INT4', 'up'),     # from south to INT4
            'entry_W1': ('INT1', 'right'),  # from west to INT1
            'entry_W2': ('INT3', 'right'),  # from west to INT3
        }
        
        for seg_id, (int_id, direction) in edge_entries.items():
            self.segments[seg_id] = RoadSegment(seg_id, capacity=100, to_intersection=int_id)
            self.spawn_points.append(seg_id)
            self.intersections[int_id].connect_incoming(direction, self.segments[seg_id])
        
        # Exit segments (same 8 points can be exits too)
        edge_exits = {
            'exit_N1': ('INT1', 'down'),
            'exit_N2': ('INT2', 'down'),
            'exit_E1': ('INT2', 'left'),
            'exit_E2': ('INT4', 'left'),
            'exit_S1': ('INT3', 'up'),
            'exit_S2': ('INT4', 'up'),
            'exit_W1': ('INT1', 'right'),
            'exit_W2': ('INT3', 'right'),
        }
        
        for seg_id, (int_id, direction) in edge_exits.items():
            self.segments[seg_id] = RoadSegment(seg_id, capacity=100, from_intersection=int_id)
            self.exit_points.append(seg_id)
            self.throughput[seg_id] = 0
            # Note: outgoing direction is opposite of incoming
            out_dir = {'down': 'up', 'up': 'down', 'left': 'right', 'right': 'left'}[direction]
            self.intersections[int_id].connect_outgoing(out_dir, self.segments[seg_id])
        
        # Horizontal connecting segments (limited capacity!)
        # H1: INT1 <-> INT2
        self.segments['H1_east'] = RoadSegment('H1_east', capacity=5,
                                                from_intersection='INT1', to_intersection='INT2')
        self.segments['H1_west'] = RoadSegment('H1_west', capacity=5,
                                                from_intersection='INT2', to_intersection='INT1')
        
        self.intersections['INT1'].connect_outgoing('left', self.segments['H1_east'])
        self.intersections['INT2'].connect_incoming('right', self.segments['H1_east'])
        self.intersections['INT2'].connect_outgoing('right', self.segments['H1_west'])
        self.intersections['INT1'].connect_incoming('left', self.segments['H1_west'])
        
        # H2: INT3 <-> INT4
        self.segments['H2_east'] = RoadSegment('H2_east', capacity=5,
                                                from_intersection='INT3', to_intersection='INT4')
        self.segments['H2_west'] = RoadSegment('H2_west', capacity=5,
                                                from_intersection='INT4', to_intersection='INT3')
        
        self.intersections['INT3'].connect_outgoing('left', self.segments['H2_east'])
        self.intersections['INT4'].connect_incoming('right', self.segments['H2_east'])
        self.intersections['INT4'].connect_outgoing('right', self.segments['H2_west'])
        self.intersections['INT3'].connect_incoming('left', self.segments['H2_west'])
        
        # Vertical connecting segments
        # V1: INT1 <-> INT3
        self.segments['V1_south'] = RoadSegment('V1_south', capacity=5,
                                                 from_intersection='INT1', to_intersection='INT3')
        self.segments['V1_north'] = RoadSegment('V1_north', capacity=5,
                                                 from_intersection='INT3', to_intersection='INT1')
        
        self.intersections['INT1'].connect_outgoing('up', self.segments['V1_south'])
        self.intersections['INT3'].connect_incoming('down', self.segments['V1_south'])
        self.intersections['INT3'].connect_outgoing('down', self.segments['V1_north'])
        self.intersections['INT1'].connect_incoming('up', self.segments['V1_north'])
        
        # V2: INT2 <-> INT4
        self.segments['V2_south'] = RoadSegment('V2_south', capacity=5,
                                                 from_intersection='INT2', to_intersection='INT4')
        self.segments['V2_north'] = RoadSegment('V2_north', capacity=5,
                                                 from_intersection='INT4', to_intersection='INT2')
        
        self.intersections['INT2'].connect_outgoing('up', self.segments['V2_south'])
        self.intersections['INT4'].connect_incoming('down', self.segments['V2_south'])
        self.intersections['INT4'].connect_outgoing('down', self.segments['V2_north'])
        self.intersections['INT2'].connect_incoming('up', self.segments['V2_north'])
        
    def _define_actions(self):
        """
        16 coordinated actions for the 4-intersection grid.
        
        0-3: All same phase (simple coordination)
        4-7: Row-based (top row vs bottom row)
        8-11: Column-based (left vs right)
        12-15: Diagonal patterns
        """
        all_ints = ['INT1', 'INT2', 'INT3', 'INT4']
        
        self.actions = [
            # All same phase
            {i: {'green': ['right', 'left'], 'arrows': []} for i in all_ints},
            {i: {'green': ['right', 'left'], 'arrows': ['right', 'left']} for i in all_ints},
            {i: {'green': ['down', 'up'], 'arrows': []} for i in all_ints},
            {i: {'green': ['down', 'up'], 'arrows': ['down', 'up']} for i in all_ints},
            
            # Row-based: top EW, bottom NS
            {'INT1': {'green': ['right', 'left'], 'arrows': []},
             'INT2': {'green': ['right', 'left'], 'arrows': []},
             'INT3': {'green': ['down', 'up'], 'arrows': []},
             'INT4': {'green': ['down', 'up'], 'arrows': []}},
            # Row-based: top NS, bottom EW
            {'INT1': {'green': ['down', 'up'], 'arrows': []},
             'INT2': {'green': ['down', 'up'], 'arrows': []},
             'INT3': {'green': ['right', 'left'], 'arrows': []},
             'INT4': {'green': ['right', 'left'], 'arrows': []}},
            # Row-based with arrows
            {'INT1': {'green': ['right', 'left'], 'arrows': ['right', 'left']},
             'INT2': {'green': ['right', 'left'], 'arrows': ['right', 'left']},
             'INT3': {'green': ['down', 'up'], 'arrows': ['down', 'up']},
             'INT4': {'green': ['down', 'up'], 'arrows': ['down', 'up']}},
            {'INT1': {'green': ['down', 'up'], 'arrows': ['down', 'up']},
             'INT2': {'green': ['down', 'up'], 'arrows': ['down', 'up']},
             'INT3': {'green': ['right', 'left'], 'arrows': ['right', 'left']},
             'INT4': {'green': ['right', 'left'], 'arrows': ['right', 'left']}},
            
            # Column-based: left EW, right NS
            {'INT1': {'green': ['right', 'left'], 'arrows': []},
             'INT3': {'green': ['right', 'left'], 'arrows': []},
             'INT2': {'green': ['down', 'up'], 'arrows': []},
             'INT4': {'green': ['down', 'up'], 'arrows': []}},
            # Column-based: left NS, right EW
            {'INT1': {'green': ['down', 'up'], 'arrows': []},
             'INT3': {'green': ['down', 'up'], 'arrows': []},
             'INT2': {'green': ['right', 'left'], 'arrows': []},
             'INT4': {'green': ['right', 'left'], 'arrows': []}},
            # Column-based with arrows
            {'INT1': {'green': ['right', 'left'], 'arrows': ['right', 'left']},
             'INT3': {'green': ['right', 'left'], 'arrows': ['right', 'left']},
             'INT2': {'green': ['down', 'up'], 'arrows': ['down', 'up']},
             'INT4': {'green': ['down', 'up'], 'arrows': ['down', 'up']}},
            {'INT1': {'green': ['down', 'up'], 'arrows': ['down', 'up']},
             'INT3': {'green': ['down', 'up'], 'arrows': ['down', 'up']},
             'INT2': {'green': ['right', 'left'], 'arrows': ['right', 'left']},
             'INT4': {'green': ['right', 'left'], 'arrows': ['right', 'left']}},
            
            # Diagonal: INT1+INT4 EW, INT2+INT3 NS
            {'INT1': {'green': ['right', 'left'], 'arrows': []},
             'INT4': {'green': ['right', 'left'], 'arrows': []},
             'INT2': {'green': ['down', 'up'], 'arrows': []},
             'INT3': {'green': ['down', 'up'], 'arrows': []}},
            # Diagonal: INT1+INT4 NS, INT2+INT3 EW
            {'INT1': {'green': ['down', 'up'], 'arrows': []},
             'INT4': {'green': ['down', 'up'], 'arrows': []},
             'INT2': {'green': ['right', 'left'], 'arrows': []},
             'INT3': {'green': ['right', 'left'], 'arrows': []}},
            # Diagonal with arrows
            {'INT1': {'green': ['right', 'left'], 'arrows': ['right', 'left']},
             'INT4': {'green': ['right', 'left'], 'arrows': ['right', 'left']},
             'INT2': {'green': ['down', 'up'], 'arrows': ['down', 'up']},
             'INT3': {'green': ['down', 'up'], 'arrows': ['down', 'up']}},
            {'INT1': {'green': ['down', 'up'], 'arrows': ['down', 'up']},
             'INT4': {'green': ['down', 'up'], 'arrows': ['down', 'up']},
             'INT2': {'green': ['right', 'left'], 'arrows': ['right', 'left']},
             'INT3': {'green': ['right', 'left'], 'arrows': ['right', 'left']}},
        ]
        
        self.action_names = [
            'All EW',
            'All EW + Arrows',
            'All NS',
            'All NS + Arrows',
            'Top EW / Bottom NS',
            'Top NS / Bottom EW',
            'Top EW+Arr / Bottom NS+Arr',
            'Top NS+Arr / Bottom EW+Arr',
            'Left EW / Right NS',
            'Left NS / Right EW',
            'Left EW+Arr / Right NS+Arr',
            'Left NS+Arr / Right EW+Arr',
            'Diag1 EW / Diag2 NS',
            'Diag1 NS / Diag2 EW',
            'Diag1 EW+Arr / Diag2 NS+Arr',
            'Diag1 NS+Arr / Diag2 EW+Arr',
        ]


# ------------------------------------------------------------------
# Utility functions for level management
# ------------------------------------------------------------------
def get_available_levels() -> List[Level]:
    """Returns a list of all available level instances."""
    return [Level1(), Level2(), Level3()]


def get_level_by_number(level_num: int) -> Level:
    """Get a level by its number (1, 2, or 3)."""
    levels = {1: Level1, 2: Level2, 3: Level3}
    if level_num in levels:
        return levels[level_num]()
    raise ValueError(f"Level {level_num} not found. Available: 1, 2, 3")


def get_level_by_name(name: str) -> Level:
    """Get a level by its name (partial match)."""
    for level_class in [Level1, Level2, Level3]:
        level = level_class()
        if name.lower() in level.name.lower():
            return level
    raise ValueError(f"Level '{name}' not found")


# Test the levels when run directly
if __name__ == "__main__":
    print("Testing level definitions...\n")
    
    for level in get_available_levels():
        print(f"=== {level.name} ===")
        print(f"  Intersections: {list(level.intersections.keys())}")
        print(f"  Spawn points: {level.spawn_points}")
        print(f"  Exit points: {level.exit_points}")
        print(f"  Connecting segments: {[s.id for s in level.segments.values() if s.from_intersection and s.to_intersection]}")
        print(f"  State vector size: {level.get_state_size()}")
        print(f"  Action count: {level.get_action_count()}")
        if hasattr(level, 'action_names'):
            print(f"  Actions: {level.action_names[:4]}... ({len(level.action_names)} total)")
        print()
