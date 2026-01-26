"""
Benchmark Script - Compare AI vs Random vs Fixed Timing

Runs automated tests to measure performance of each control strategy.
Outputs statistics for documentation purposes.

Usage:
    python benchmark.py                 # Run full benchmark
    python benchmark.py --episodes 10   # Quick test with fewer episodes
    python benchmark.py --output results.csv  # Save to CSV
"""

import numpy as np
import torch
import time
import argparse
import random
from collections import defaultdict

from simulation import TrafficSimulator

# Try to import trained model
try:
    from train_unified import UnifiedDQN, HIDDEN_SIZE, MAX_STATE_SIZE, MAX_ACTION_SIZE
    DQN_AVAILABLE = True
except ImportError:
    DQN_AVAILABLE = False
    print("Warning: Could not import DQN. AI mode will be skipped.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_file='dqn_unified_best.pth'):
    """Load trained model if available."""
    if not DQN_AVAILABLE:
        return None, 0, 0
    
    try:
        checkpoint = torch.load(model_file, map_location=device, weights_only=False)
        state_size = checkpoint.get('state_size', MAX_STATE_SIZE)
        action_size = checkpoint.get('action_size', MAX_ACTION_SIZE)
        
        policy_net = UnifiedDQN(state_size, action_size, HIDDEN_SIZE).to(device)
        policy_net.load_state_dict(checkpoint['policy_net'])
        policy_net.eval()
        
        print(f"  Loaded model: {model_file}")
        return policy_net, state_size, action_size
    except Exception as e:
        print(f"  Could not load model: {e}")
        return None, 0, 0


def run_episode(simulator, control_mode, policy_net=None, state_size=0, action_size=0,
                steps=200, spawn_rate=0.25):
    """Run one episode with specified control mode and collect metrics."""
    
    # Reset simulator
    simulator.vehicles.clear()
    simulator.vehicles_crossed = {'right': 0, 'down': 0, 'left': 0, 'up': 0}
    if simulator.current_level:
        for seg in simulator.current_level.segments.values():
            seg.vehicles.clear()
    
    # Set spawn rates
    if simulator.current_level:
        for seg_id, seg in simulator.current_level.segments.items():
            if seg_id.startswith('entry_'):
                seg.spawn_rate = spawn_rate
    
    level_info = simulator.get_level_info()
    valid_actions = level_info['action_count']
    
    # Fixed timing state
    fixed_action_idx = 0
    fixed_timer = 0
    fixed_phase_duration = 8  # steps per phase (each step = 15 frames, so ~120 frames per phase)
    
    # Spawn initial vehicles (like training does)
    level_num = simulator.current_level_num
    for _ in range(8):
        if level_num == 1:
            simulator._spawn_level1_vehicle()
        elif level_num == 2:
            simulator._spawn_level2_vehicle()
        elif level_num == 3:
            simulator._spawn_level3_vehicle()
    
    # Metrics tracking
    total_throughput = 0
    total_waiting = 0
    total_wait_time = 0
    wait_samples = 0
    max_vehicles = 0
    
    for step in range(steps):
        # Spawn vehicles every 3 steps (matching training)
        if step % 3 == 0:
            level_num = simulator.current_level_num
            for _ in range(2):  # Spawn 2 vehicles at a time like training
                if level_num == 1:
                    simulator._spawn_level1_vehicle()
                elif level_num == 2:
                    simulator._spawn_level2_vehicle()
                elif level_num == 3:
                    simulator._spawn_level3_vehicle()
        
        # Select action based on control mode
        if control_mode == 'ai' and policy_net is not None:
            state = np.array(simulator.get_state(), dtype=np.float32)
            if len(state) < state_size:
                state = np.pad(state, (0, state_size - len(state)))
            elif len(state) > state_size:
                state = state[:state_size]
            
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = policy_net(state_t)
                if valid_actions < action_size:
                    q_values[0, valid_actions:] = float('-inf')
                action = q_values.argmax().item()
                action = min(action, valid_actions - 1)
        
        elif control_mode == 'fixed':
            fixed_timer += 1
            if fixed_timer >= fixed_phase_duration:
                fixed_timer = 0
                fixed_action_idx = (fixed_action_idx + 1) % valid_actions
            action = fixed_action_idx
        
        else:  # random
            action = random.randint(0, valid_actions - 1)
        
        # Apply action
        simulator.apply_level_action(action)
        
        # Run 15 simulation frames per decision (matching training!)
        for _ in range(15):
            simulator.update()
        
        # Collect metrics
        metrics = simulator.get_metrics()
        total_throughput = metrics.get('total_throughput', 0)
        waiting = metrics.get('waiting_vehicles', 0)
        total_waiting += waiting
        
        avg_wait = metrics.get('avg_wait_time', 0)
        if avg_wait > 0:
            total_wait_time += avg_wait
            wait_samples += 1
        
        vehicle_count = len(simulator.vehicles)
        max_vehicles = max(max_vehicles, vehicle_count)
    
    # Calculate final metrics
    avg_waiting = total_waiting / steps if steps > 0 else 0
    avg_wait_time = total_wait_time / wait_samples if wait_samples > 0 else 0
    
    return {
        'throughput': total_throughput,
        'avg_waiting': avg_waiting,
        'avg_wait_time': avg_wait_time,
        'max_vehicles': max_vehicles
    }


def run_benchmark(episodes_per_config=20, steps_per_episode=200, spawn_rates=None):
    """Run full benchmark across all modes and levels."""
    
    if spawn_rates is None:
        spawn_rates = [0.1, 0.25, 0.4]  # Light, medium, heavy traffic
    
    control_modes = ['ai', 'random', 'fixed']
    levels = [1, 2, 3]
    
    # Load model
    print("\n" + "=" * 60)
    print("  TRAFFIC CONTROL BENCHMARK")
    print("=" * 60)
    
    policy_net, state_size, action_size = load_model()
    if policy_net is None:
        print("  Skipping AI mode (no model)")
        control_modes = ['random', 'fixed']
    
    print(f"  Episodes per config: {episodes_per_config}")
    print(f"  Steps per episode: {steps_per_episode}")
    print(f"  Spawn rates: {spawn_rates}")
    print(f"  Control modes: {control_modes}")
    print("-" * 60)
    
    # Results storage
    results = defaultdict(lambda: defaultdict(list))
    
    # Create simulator (headless)
    simulator = TrafficSimulator(level=1, headless=True)
    
    total_configs = len(levels) * len(control_modes) * len(spawn_rates) * episodes_per_config
    current = 0
    start_time = time.time()
    
    for level in levels:
        simulator.switch_level(level)
        level_info = simulator.get_level_info()
        print(f"\nLevel {level}: {level_info['state_size']} state, {level_info['action_count']} actions")
        
        for spawn_rate in spawn_rates:
            for mode in control_modes:
                mode_results = []
                
                for ep in range(episodes_per_config):
                    result = run_episode(
                        simulator, mode, policy_net, state_size, action_size,
                        steps=steps_per_episode, spawn_rate=spawn_rate
                    )
                    mode_results.append(result)
                    
                    current += 1
                    elapsed = time.time() - start_time
                    eta = (elapsed / current) * (total_configs - current) if current > 0 else 0
                    
                    print(f"\r  [{current}/{total_configs}] L{level} {mode:6s} rate={spawn_rate:.2f} "
                          f"ep={ep+1}/{episodes_per_config} | ETA: {eta:.0f}s", end="")
                
                # Store aggregated results
                key = f"L{level}_rate{spawn_rate}"
                results[key][mode] = {
                    'throughput': np.mean([r['throughput'] for r in mode_results]),
                    'throughput_std': np.std([r['throughput'] for r in mode_results]),
                    'avg_waiting': np.mean([r['avg_waiting'] for r in mode_results]),
                    'avg_wait_time': np.mean([r['avg_wait_time'] for r in mode_results]),
                    'max_vehicles': np.mean([r['max_vehicles'] for r in mode_results])
                }
    
    print("\n")
    simulator.running = False
    
    return results


def print_results(results):
    """Print formatted results table."""
    
    print("\n" + "=" * 80)
    print("  BENCHMARK RESULTS")
    print("=" * 80)
    
    # Get all modes
    sample_key = list(results.keys())[0]
    modes = list(results[sample_key].keys())
    
    # Print header
    print(f"\n{'Config':<20}", end="")
    for mode in modes:
        print(f"  {mode.upper():^22}", end="")
    print()
    
    print("-" * 80)
    
    # Print throughput
    print("\nTHROUGHPUT (higher is better):")
    for config, mode_data in sorted(results.items()):
        print(f"  {config:<18}", end="")
        for mode in modes:
            val = mode_data[mode]['throughput']
            std = mode_data[mode]['throughput_std']
            print(f"  {val:>8.1f} ±{std:>5.1f}   ", end="")
        print()
    
    # Print average waiting vehicles
    print("\nAVG WAITING VEHICLES (lower is better):")
    for config, mode_data in sorted(results.items()):
        print(f"  {config:<18}", end="")
        for mode in modes:
            val = mode_data[mode]['avg_waiting']
            print(f"  {val:>14.2f}       ", end="")
        print()
    
    # Print average wait time
    print("\nAVG WAIT TIME (lower is better):")
    for config, mode_data in sorted(results.items()):
        print(f"  {config:<18}", end="")
        for mode in modes:
            val = mode_data[mode]['avg_wait_time']
            print(f"  {val:>14.1f}       ", end="")
        print()
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("  SUMMARY: AI vs Others (% improvement in throughput)")
    print("=" * 80)
    
    if 'ai' in modes:
        for config, mode_data in sorted(results.items()):
            ai_throughput = mode_data['ai']['throughput']
            print(f"\n  {config}:")
            for mode in modes:
                if mode != 'ai':
                    other_throughput = mode_data[mode]['throughput']
                    if other_throughput > 0:
                        improvement = ((ai_throughput - other_throughput) / other_throughput) * 100
                        sign = "+" if improvement > 0 else ""
                        print(f"    vs {mode}: {sign}{improvement:.1f}%")
    
    print("\n" + "=" * 80)


def save_results_csv(results, filename):
    """Save results to CSV file."""
    
    with open(filename, 'w') as f:
        # Header
        sample_key = list(results.keys())[0]
        modes = list(results[sample_key].keys())
        
        headers = ['config', 'level', 'spawn_rate']
        for mode in modes:
            headers.extend([
                f'{mode}_throughput', f'{mode}_throughput_std',
                f'{mode}_avg_waiting', f'{mode}_avg_wait_time', f'{mode}_max_vehicles'
            ])
        f.write(','.join(headers) + '\n')
        
        # Data rows
        for config, mode_data in sorted(results.items()):
            # Parse config
            parts = config.split('_')
            level = parts[0][1:]  # Remove 'L'
            rate = parts[1][4:]   # Remove 'rate'
            
            row = [config, level, rate]
            for mode in modes:
                data = mode_data[mode]
                row.extend([
                    f"{data['throughput']:.2f}",
                    f"{data['throughput_std']:.2f}",
                    f"{data['avg_waiting']:.2f}",
                    f"{data['avg_wait_time']:.2f}",
                    f"{data['max_vehicles']:.2f}"
                ])
            f.write(','.join(map(str, row)) + '\n')
    
    print(f"\nResults saved to: {filename}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark traffic control strategies')
    parser.add_argument('--episodes', '-e', type=int, default=20,
                        help='Episodes per configuration (default: 20)')
    parser.add_argument('--steps', '-s', type=int, default=200,
                        help='Steps per episode (default: 200)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output CSV filename')
    parser.add_argument('--quick', '-q', action='store_true',
                        help='Quick test (5 episodes, 100 steps)')
    args = parser.parse_args()
    
    episodes = 5 if args.quick else args.episodes
    steps = 100 if args.quick else args.steps
    
    results = run_benchmark(
        episodes_per_config=episodes,
        steps_per_episode=steps
    )
    
    print_results(results)
    
    if args.output:
        save_results_csv(results, args.output)
    else:
        # Auto-save with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_results_csv(results, f"benchmark_{timestamp}.csv")


if __name__ == "__main__":
    main()
