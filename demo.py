"""
Interactive Demo - Watch AI Control Traffic with Adjustable Spawn Rates

Drag the sliders to change how many cars spawn at each entry point.
The trained AI will adapt to the changing traffic patterns in real-time.

Usage:
    python demo.py                  # Demo level 1
    python demo.py --level 2        # Demo level 2
    python demo.py --level 3        # Demo level 3
"""

import pygame
import numpy as np
import torch
import time
import argparse
import sys

# Initialize pygame first
pygame.init()

from simulation import (
    TrafficSimulator, SCREEN_WIDTH, SCREEN_HEIGHT, 
    LANES_PER_DIRECTION, LANE_WIDTH
)

# Try to import DQN from unified training
try:
    from train_unified import UnifiedDQN as DQN, HIDDEN_SIZE, MAX_STATE_SIZE, MAX_ACTION_SIZE
    DQN_AVAILABLE = True
except ImportError:
    DQN_AVAILABLE = False
    print("Note: DQN not available, running with random actions")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Slider:
    """A draggable slider widget."""
    
    def __init__(self, x, y, width, height, label, min_val=0.0, max_val=1.0, initial=0.5):
        self.rect = pygame.Rect(x, y, width, height)
        self.label = label
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial
        self.dragging = False
        self.knob_radius = height // 2 + 3
        self._update_knob()
    
    def _update_knob(self):
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        self.knob_x = int(self.rect.x + ratio * self.rect.width)
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            # Check if clicking on or near the slider
            if (self.rect.x - 10 <= mx <= self.rect.x + self.rect.width + 10 and
                self.rect.y - 15 <= my <= self.rect.y + self.rect.height + 15):
                self.dragging = True
                self._update_from_mouse(mx)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._update_from_mouse(event.pos[0])
    
    def _update_from_mouse(self, mx):
        x = max(self.rect.x, min(mx, self.rect.x + self.rect.width))
        ratio = (x - self.rect.x) / self.rect.width
        self.value = self.min_val + ratio * (self.max_val - self.min_val)
        self._update_knob()
    
    def draw(self, screen, font):
        # Track background
        pygame.draw.rect(screen, (60, 60, 60), self.rect, border_radius=4)
        
        # Filled portion
        fill_w = self.knob_x - self.rect.x
        fill_rect = pygame.Rect(self.rect.x, self.rect.y, fill_w, self.rect.height)
        
        # Color based on value (green=low, yellow=mid, red=high)
        if self.value < 0.4:
            color = (50, 200, 50)
        elif self.value < 0.7:
            color = (200, 200, 50)
        else:
            color = (200, 100, 50)
        
        pygame.draw.rect(screen, color, fill_rect, border_radius=4)
        
        # Knob
        knob_color = (255, 255, 255) if self.dragging else (220, 220, 220)
        pygame.draw.circle(screen, knob_color, 
                          (self.knob_x, self.rect.y + self.rect.height // 2),
                          self.knob_radius)
        pygame.draw.circle(screen, (100, 100, 100),
                          (self.knob_x, self.rect.y + self.rect.height // 2),
                          self.knob_radius, 2)
        
        # Label and value
        label_text = font.render(f"{self.label}", True, (255, 255, 255))
        value_text = font.render(f"{self.value:.0%}", True, (200, 200, 200))
        screen.blit(label_text, (self.rect.x, self.rect.y - 18))
        screen.blit(value_text, (self.rect.x + self.rect.width - 35, self.rect.y - 18))


class DemoPanel:
    """Side panel with sliders and controls."""
    
    def __init__(self, x, y, width, level_num):
        self.x = x
        self.y = y
        self.width = width
        self.level_num = level_num
        self.sliders = {}
        
        # Create sliders based on level
        if level_num == 1:
            spawn_points = ['A (West)', 'B (North)', 'C (East)', 'D (South)']
            self.spawn_keys = ['A', 'B', 'C', 'D']
        elif level_num == 2:
            spawn_points = ['A (West)', 'B1 (N-Left)', 'D1 (S-Left)', 
                           'C (East)', 'B2 (N-Right)', 'D2 (S-Right)']
            self.spawn_keys = ['A', 'B1', 'D1', 'C', 'B2', 'D2']
        else:  # Level 3
            spawn_points = ['N1', 'N2', 'W1', 'W2', 'E1', 'E2', 'S1', 'S2']
            self.spawn_keys = spawn_points
        
        slider_y = y + 35
        for i, label in enumerate(spawn_points):
            key = self.spawn_keys[i]
            # Slider 0-100% maps to spawn rate 0-8000% (so 25% slider = default spawn rate)
            self.sliders[key] = Slider(x + 10, slider_y, width - 30, 14, label,
                                       min_val=0.0, max_val=1.0, initial=0.25)
            slider_y += 45
        
        self.stats_y = slider_y + 15
    
    def handle_event(self, event):
        for slider in self.sliders.values():
            slider.handle_event(event)
    
    def get_spawn_rates(self):
        """Get spawn rates for the level's spawn points.
        
        Spawner now checks every 50ms and spawns from ALL entries.
        Rate = how many cars per entry per cycle.
        At 25% = 5 cars/entry/50ms = FAST
        At 100% = 20 cars/entry/50ms = CHAOS
        """
        rates = {}
        for key, slider in self.sliders.items():
            # Scale: slider 0-1.0 → spawn rate 0-20
            scaled_rate = slider.value * 0.05
            
            # Map to the segment IDs used by the level
            if self.level_num == 1:
                # Level 1 uses direction names
                direction_map = {'A': 'right', 'B': 'down', 'C': 'left', 'D': 'up'}
                rates[f"entry_{direction_map[key]}"] = scaled_rate
            else:
                # Level 2 and 3 use entry_X format
                rates[f"entry_{key}"] = scaled_rate
        return rates
    
    def draw(self, screen, font, metrics, action_name="", ai_active=True):
        # Panel background
        panel_rect = pygame.Rect(self.x, self.y, self.width, 700)
        pygame.draw.rect(screen, (25, 25, 35), panel_rect)
        pygame.draw.rect(screen, (80, 80, 100), panel_rect, 2)
        
        # Title
        title = font.render("SPAWN RATE CONTROLS", True, (100, 200, 255))
        screen.blit(title, (self.x + 10, self.y + 8))
        
        # Draw sliders
        for slider in self.sliders.values():
            slider.draw(screen, font)
        
        # Separator
        pygame.draw.line(screen, (80, 80, 100),
                        (self.x + 10, self.stats_y - 8),
                        (self.x + self.width - 10, self.stats_y - 8), 2)
        
        # AI Status
        ai_color = (0, 255, 100) if ai_active else (255, 150, 50)
        ai_text = "AI: ACTIVE" if ai_active else "AI: RANDOM"
        text = font.render(ai_text, True, ai_color)
        screen.blit(text, (self.x + 10, self.stats_y))
        
        # Current action
        y = self.stats_y + 25
        text = font.render("Current Action:", True, (180, 180, 180))
        screen.blit(text, (self.x + 10, y))
        y += 20
        # Wrap long action names
        if len(action_name) > 20:
            text = font.render(action_name[:20], True, (255, 255, 255))
            screen.blit(text, (self.x + 10, y))
            y += 18
            text = font.render(action_name[20:], True, (255, 255, 255))
            screen.blit(text, (self.x + 10, y))
        else:
            text = font.render(action_name, True, (255, 255, 255))
            screen.blit(text, (self.x + 10, y))
        
        # Stats
        y += 30
        pygame.draw.line(screen, (80, 80, 100),
                        (self.x + 10, y), (self.x + self.width - 10, y), 1)
        y += 10
        
        stats = [
            ("Vehicles", str(metrics.get('total_vehicles', 0))),
            ("Waiting", str(metrics.get('waiting_vehicles', 0))),
            ("Throughput", str(metrics.get('total_throughput', 0))),
            ("Avg Wait", f"{metrics.get('avg_wait_time', 0):.1f}f"),
        ]
        
        for label, value in stats:
            text = font.render(f"{label}: {value}", True, (200, 200, 200))
            screen.blit(text, (self.x + 10, y))
            y += 22
        
        # Instructions
        y += 15
        pygame.draw.line(screen, (80, 80, 100),
                        (self.x + 10, y), (self.x + self.width - 10, y), 1)
        y += 10
        
        instructions = [
            "Drag sliders to change",
            "traffic spawn rates.",
            "(Max = CHAOS MODE!)",
            "",
            "L: Switch level",
            "R: Reset to normal (25%)",
            "M: Max all (stress test)",
            "Z: Zero all",
            "ESC: Exit",
        ]
        
        for line in instructions:
            text = font.render(line, True, (150, 150, 150))
            screen.blit(text, (self.x + 10, y))
            y += 18


def run_demo(level_num=1, model_file=None):
    """Run interactive demo with sliders."""
    
    print("\n" + "=" * 50)
    print("  TRAFFIC AI DEMO - Level", level_num)
    print("=" * 50)
    
    # Load model if available
    use_ai = False
    policy_net = None
    state_size = 20
    action_size = 8
    
    if DQN_AVAILABLE:
        # Try to find a model - prefer unified model, then level-specific
        tried = []
        checkpoint = None
        is_unified = False
        
        # Search order: specified model, unified model, then level-specific
        search_files = [
            model_file,
            'dqn_unified_best.pth',      # Unified model (works for all levels)
            'dqn_unified_final.pth',
            f'dqn_traffic_L{level_num}_best.pth',
            f'dqn_traffic_L{level_num}.pth',
            'dqn_traffic_best.pth',
            'dqn_traffic_v2.pth'
        ]
        
        for try_file in search_files:
            if try_file is None:
                continue
            tried.append(try_file)
            try:
                checkpoint = torch.load(try_file, map_location=device, weights_only=False)
                is_unified = checkpoint.get('unified', False)
                print(f"  Loaded: {try_file}" + (" (unified)" if is_unified else ""))
                break
            except:
                continue
        
        if checkpoint:
            state_size = checkpoint.get('state_size', 20)
            action_size = checkpoint.get('action_size', 8)
            
            policy_net = DQN(state_size, action_size, HIDDEN_SIZE).to(device)
            try:
                policy_net.load_state_dict(checkpoint['policy_net'])
                policy_net.eval()
                use_ai = True
            except:
                print("  Model incompatible, using random actions")
        else:
            print(f"  No model found. Tried: {tried}")
    
    # Create simulator
    simulator = TrafficSimulator(level=level_num)
    time.sleep(0.3)
    
    level_info = simulator.get_level_info()
    print(f"  State size: {level_info['state_size']}, Actions: {level_info['action_count']}")
    print("\n  Drag sliders to control traffic flow!")
    print("-" * 50)
    
    # Create control panel
    panel_width = 210
    panel = DemoPanel(SCREEN_WIDTH - panel_width - 5, 5, panel_width, level_num)
    
    # Get action names
    if simulator.current_level and hasattr(simulator.current_level, 'action_names'):
        action_names = simulator.current_level.action_names
    else:
        action_names = [f"Action {i}" for i in range(action_size)]
    
    # Main loop
    running = True
    action = 0
    action_name = "Initializing..."
    frame = 0
    frames_per_decision = 45  # Make decision every 0.75 seconds for more responsive signals
    
    # Apply initial action immediately
    action_name = simulator.apply_level_action(0)
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_l:
                    # Switch level
                    level_num = (level_num % 3) + 1
                    simulator.switch_level(level_num)
                    
                    # Update level info
                    level_info = simulator.get_level_info()
                    
                    # Recreate panel with correct sliders for new level
                    panel = DemoPanel(SCREEN_WIDTH - panel_width - 5, 5, panel_width, level_num)
                    
                    # Apply initial spawn rates to new level
                    if simulator.current_level:
                        rates = panel.get_spawn_rates()
                        for seg_id, rate in rates.items():
                            if seg_id in simulator.current_level.segments:
                                simulator.current_level.segments[seg_id].spawn_rate = rate
                    
                    # Apply initial action for the new level
                    action_name = simulator.apply_level_action(0)
                    
                    if simulator.current_level and hasattr(simulator.current_level, 'action_names'):
                        action_names = simulator.current_level.action_names
                    print(f"  Switched to Level {level_num}")
                elif event.key == pygame.K_r:
                    # Reset sliders to 25% (normal spawn rate)
                    for slider in panel.sliders.values():
                        slider.value = 0.25
                        slider._update_knob()
                elif event.key == pygame.K_m:
                    # Max all sliders (stress test)
                    for slider in panel.sliders.values():
                        slider.value = 1.0
                        slider._update_knob()
                elif event.key == pygame.K_z:
                    # Zero all sliders
                    for slider in panel.sliders.values():
                        slider.value = 0.0
                        slider._update_knob()
            
            panel.handle_event(event)
        
        # Apply spawn rates from sliders to the level
        if simulator.current_level:
            rates = panel.get_spawn_rates()
            for seg_id, rate in rates.items():
                if seg_id in simulator.current_level.segments:
                    simulator.current_level.segments[seg_id].spawn_rate = rate
        
        # Make AI decision periodically
        if frame % frames_per_decision == 0:
            # Get valid action count for current level
            valid_actions = level_info['action_count']
            
            if use_ai and policy_net:
                state = np.array(simulator.get_state(), dtype=np.float32)
                # Pad or truncate state if needed
                if len(state) < state_size:
                    state = np.pad(state, (0, state_size - len(state)))
                elif len(state) > state_size:
                    state = state[:state_size]
                
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_values = policy_net(state_t)
                    
                    # Mask invalid actions for levels with fewer actions
                    if valid_actions < action_size:
                        q_values[0, valid_actions:] = float('-inf')
                    
                    action = q_values.argmax().item()
                    
                    # Ensure action is valid for this level
                    action = min(action, valid_actions - 1)
            else:
                action = np.random.randint(0, valid_actions)
            
            # Apply action using level's action system
            action_name = simulator.apply_level_action(action)
        
        frame += 1
        
        # Update simulation
        simulator.update()
        
        # Draw everything
        simulator.draw_roads()
        for vehicle in simulator.vehicles:
            vehicle.draw(simulator.screen, simulator.vehicle_font)
        simulator.draw_signals()
        
        # Draw level indicator
        level_text = f"Level {level_num}"
        text = simulator.font.render(level_text, True, (0, 255, 255))
        pygame.draw.rect(simulator.screen, (0, 0, 50), (10, 10, 100, 35))
        simulator.screen.blit(text, (20, 15))
        
        # Draw panel
        metrics = simulator.get_metrics()
        panel.draw(simulator.screen, simulator.small_font, metrics, action_name, use_ai)
        
        pygame.display.flip()
        simulator.clock.tick(60)
    
    simulator.running = False
    pygame.quit()
    print("\n  Demo ended.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Traffic AI Demo with Sliders')
    parser.add_argument('--level', '-l', type=int, default=1, choices=[1, 2, 3],
                        help='Level to demo (1, 2, or 3)')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Model file to load')
    args = parser.parse_args()
    
    run_demo(level_num=args.level, model_file=args.model)
