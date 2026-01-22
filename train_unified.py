"""
Unified Multi-Level DQN Training

Trains a single AI model across all 3 levels so it can handle any traffic scenario.
The model uses the maximum state/action sizes and pads smaller levels appropriately.

Usage:
    python train_unified.py --episodes 500
    python train_unified.py --episodes 1000 --test
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import argparse
from collections import deque

# Initialize pygame before importing simulation
import pygame
pygame.init()

from simulation import TrafficSimulator

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 0.0005
BATCH_SIZE = 64
MEMORY_SIZE = 50000
TARGET_UPDATE = 10
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.997
HIDDEN_SIZE = 256

# Maximum sizes across all levels (Level 3 is largest)
MAX_STATE_SIZE = 60   # Padded to be safe
MAX_ACTION_SIZE = 16  # Level 3 has 16 actions


class UnifiedDQN(nn.Module):
    """Neural network that works for all levels by using max sizes."""
    
    def __init__(self, state_size=MAX_STATE_SIZE, action_size=MAX_ACTION_SIZE, hidden_size=HIDDEN_SIZE):
        super(UnifiedDQN, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            
            nn.Linear(hidden_size // 2, action_size)
        )
    
    def forward(self, x):
        return self.net(x)


class ReplayMemory:
    """Experience replay buffer that stores transitions from all levels."""
    
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, level_num, valid_actions):
        """Store a transition with level info for proper action masking."""
        self.memory.append((state, action, reward, next_state, done, level_num, valid_actions))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


def pad_state(state, target_size=MAX_STATE_SIZE):
    """Pad a state vector to the maximum size."""
    state = np.array(state, dtype=np.float32)
    if len(state) < target_size:
        state = np.pad(state, (0, target_size - len(state)))
    elif len(state) > target_size:
        state = state[:target_size]
    return state


def get_valid_actions(level_num):
    """Get the number of valid actions for a level."""
    if level_num == 3:
        return 16
    else:
        return 8


def compute_reward(simulator, prev_metrics, action, prev_action):
    """Compute reward based on traffic flow improvement."""
    metrics = simulator.get_metrics()
    
    # Improvements are good
    wait_delta = prev_metrics.get('avg_wait_time', 0) - metrics.get('avg_wait_time', 0)
    throughput_delta = metrics.get('total_throughput', 0) - prev_metrics.get('total_throughput', 0)
    queue_delta = sum(prev_metrics.get('queue_lengths', {}).values()) - sum(metrics.get('queue_lengths', {}).values())
    
    reward = 0.0
    
    # Reward for reducing wait times
    reward += wait_delta * 0.1
    
    # Reward for throughput
    reward += throughput_delta * 0.5
    
    # Reward for reducing queues
    reward += queue_delta * 0.05
    
    # Small penalty for changing signals too often (stability)
    if action != prev_action:
        reward -= 0.1
    
    # Penalty for too many waiting vehicles
    waiting = metrics.get('waiting_vehicles', 0)
    if waiting > 10:
        reward -= waiting * 0.02
    
    # Clip reward to reasonable range
    reward = np.clip(reward, -5, 5)
    
    return reward, metrics


def spawn_vehicles_fast(simulator, count=5):
    """Directly spawn vehicles without waiting for background thread."""
    for _ in range(count):
        if simulator.current_level_num == 1:
            simulator._spawn_level1_vehicle()
        elif simulator.current_level_num == 2:
            simulator._spawn_level2_vehicle()
        elif simulator.current_level_num == 3:
            simulator._spawn_level3_vehicle()


def run_episode(simulator, policy_net, target_net, memory, optimizer, 
                epsilon, level_num, steps_per_episode=100, training=True):
    """Run one episode on a specific level - FAST mode without delays."""
    
    # Reset the simulator by switching to same level (clears vehicles)
    simulator.switch_level(level_num)
    
    # Set moderate spawn rates
    # Randomize spawn rates: very light (0.01) to heavy (0.5)
    # This helps AI learn to handle all traffic scenarios
    if simulator.current_level:
        for seg_id, seg in simulator.current_level.segments.items():
            if seg_id.startswith('entry_'):
                seg.spawn_rate = random.uniform(0.01, 0.5)
    
    # Spawn initial vehicles directly (no sleep!)
    spawn_vehicles_fast(simulator, count=8)
    
    valid_actions = get_valid_actions(level_num)
    state = pad_state(simulator.get_state())
    prev_metrics = simulator.get_metrics()
    prev_action = 0
    
    total_reward = 0
    
    for step in range(steps_per_episode):
        # Epsilon-greedy action selection
        if training and random.random() < epsilon:
            action = random.randint(0, valid_actions - 1)
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = policy_net(state_t)
                # Mask invalid actions for this level
                q_values[0, valid_actions:] = float('-inf')
                action = q_values.argmax().item()
        
        # Apply action
        simulator.apply_level_action(action)
        
        # Run simulation frames FAST (no sleep!)
        for _ in range(15):
            simulator.update()
        
        # Occasionally spawn more vehicles (replaces background spawner)
        if step % 3 == 0:
            spawn_vehicles_fast(simulator, count=2)
        
        # Get new state and reward
        next_state = pad_state(simulator.get_state())
        reward, metrics = compute_reward(simulator, prev_metrics, action, prev_action)
        done = step >= steps_per_episode - 1
        
        total_reward += reward
        
        # Store in memory
        if training:
            memory.push(state, action, reward, next_state, done, level_num, valid_actions)
        
        state = next_state
        prev_metrics = metrics
        prev_action = action
        
        # Training step
        if training and len(memory) >= BATCH_SIZE:
            train_step(policy_net, target_net, memory, optimizer)
    
    return total_reward


def train_step(policy_net, target_net, memory, optimizer):
    """Perform one training step using experience replay."""
    batch = memory.sample(BATCH_SIZE)
    
    states = torch.FloatTensor(np.array([t[0] for t in batch])).to(device)
    actions = torch.LongTensor([t[1] for t in batch]).to(device)
    rewards = torch.FloatTensor([t[2] for t in batch]).to(device)
    next_states = torch.FloatTensor(np.array([t[3] for t in batch])).to(device)
    dones = torch.FloatTensor([t[4] for t in batch]).to(device)
    valid_actions_list = [t[6] for t in batch]
    
    # Current Q values
    current_q = policy_net(states).gather(1, actions.unsqueeze(1))
    
    # Target Q values with Double DQN
    with torch.no_grad():
        # Get best actions from policy net
        next_q_policy = policy_net(next_states)
        
        # Mask invalid actions for each sample
        for i, valid_act in enumerate(valid_actions_list):
            next_q_policy[i, valid_act:] = float('-inf')
        
        best_actions = next_q_policy.argmax(1)
        
        # Get Q values from target net
        next_q_target = target_net(next_states)
        next_q = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze()
        
        target_q = rewards + GAMMA * next_q * (1 - dones)
    
    # Huber loss for stability
    loss = nn.SmoothL1Loss()(current_q.squeeze(), target_q)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()


def train(episodes=500, steps_per_episode=80):
    """Train the unified model across all levels."""
    
    print("\n" + "=" * 60)
    print("  UNIFIED MULTI-LEVEL DQN TRAINING")
    print("=" * 60)
    print(f"  Episodes: {episodes}")
    print(f"  Steps per episode: {steps_per_episode}")
    print(f"  State size: {MAX_STATE_SIZE}, Action size: {MAX_ACTION_SIZE}")
    print(f"  Training on: Level 1, Level 2, Level 3")
    print("=" * 60)
    
    # Create networks
    policy_net = UnifiedDQN().to(device)
    target_net = UnifiedDQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(MEMORY_SIZE)
    
    epsilon = EPSILON_START
    best_avg_reward = float('-inf')
    reward_history = {1: [], 2: [], 3: []}
    
    # Create simulator (we'll switch levels during training)
    simulator = TrafficSimulator(level=1, headless=True)
    
    start_time = time.time()
    episode_times = []
    
    try:
        for episode in range(episodes):
            ep_start = time.time()
            
            # Cycle through levels: 1 -> 2 -> 3 -> 1 -> ...
            level_num = (episode % 3) + 1
            
            reward = run_episode(
                simulator, policy_net, target_net, memory, optimizer,
                epsilon, level_num, steps_per_episode, training=True
            )
            
            ep_time = time.time() - ep_start
            episode_times.append(ep_time)
            reward_history[level_num].append(reward)
            
            # Decay epsilon
            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
            
            # Update target network
            if episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
            
            # Progress report every 10 episodes
            if episode % 10 == 0:
                avg_rewards = {
                    lvl: np.mean(rews[-10:]) if rews else 0 
                    for lvl, rews in reward_history.items()
                }
                overall_avg = np.mean(list(avg_rewards.values()))
                avg_ep_time = np.mean(episode_times[-10:]) if episode_times else 0
                elapsed = time.time() - start_time
                eta = avg_ep_time * (episodes - episode) if avg_ep_time > 0 else 0
                
                print(f"Ep {episode:4d}/{episodes} | L{level_num} | "
                      f"R:{reward:6.2f} | "
                      f"Avg L1:{avg_rewards[1]:5.1f} L2:{avg_rewards[2]:5.1f} L3:{avg_rewards[3]:5.1f} | "
                      f"Eps:{epsilon:.2f} | "
                      f"{avg_ep_time:.1f}s/ep | ETA:{eta/60:.1f}m")
                
                # Save best model
                if overall_avg > best_avg_reward and episode > 30:
                    best_avg_reward = overall_avg
                    torch.save({
                        'policy_net': policy_net.state_dict(),
                        'target_net': target_net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'episode': episode,
                        'epsilon': epsilon,
                        'state_size': MAX_STATE_SIZE,
                        'action_size': MAX_ACTION_SIZE,
                        'best_reward': best_avg_reward,
                        'unified': True,
                    }, 'dqn_unified_best.pth')
                    print(f"  -> Saved best model (avg: {overall_avg:.2f})")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    finally:
        simulator.running = False
        pygame.quit()
    
    # Save final model
    torch.save({
        'policy_net': policy_net.state_dict(),
        'target_net': target_net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'episode': episodes,
        'epsilon': epsilon,
        'state_size': MAX_STATE_SIZE,
        'action_size': MAX_ACTION_SIZE,
        'unified': True,
    }, 'dqn_unified_final.pth')
    
    total_time = time.time() - start_time
    avg_time = np.mean(episode_times) if episode_times else 0
    
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Avg per episode: {avg_time:.2f} seconds")
    print(f"  Episodes completed: {len(episode_times)}")
    print(f"  Models saved: dqn_unified_best.pth, dqn_unified_final.pth")
    print("=" * 60)


def test(model_file='dqn_unified_best.pth'):
    """Test the trained model on all levels."""
    
    print("\n" + "=" * 60)
    print("  TESTING UNIFIED MODEL")
    print("=" * 60)
    
    # Load model
    try:
        checkpoint = torch.load(model_file, map_location=device, weights_only=False)
        print(f"  Loaded: {model_file}")
    except:
        print(f"  Could not load {model_file}")
        return
    
    policy_net = UnifiedDQN(
        checkpoint.get('state_size', MAX_STATE_SIZE),
        checkpoint.get('action_size', MAX_ACTION_SIZE)
    ).to(device)
    policy_net.load_state_dict(checkpoint['policy_net'])
    policy_net.eval()
    
    # Test on each level
    for level_num in [1, 2, 3]:
        print(f"\n  --- Level {level_num} ---")
        
        simulator = TrafficSimulator(level=level_num, headless=True)
        
        # Set spawn rates (mid-range for consistent testing)
        for seg_id, seg in simulator.current_level.segments.items():
            if seg_id.startswith('entry_'):
                seg.spawn_rate = 0.25
        
        # Spawn initial vehicles
        spawn_vehicles_fast(simulator, count=8)
        
        total_reward = 0
        valid_actions = get_valid_actions(level_num)
        prev_metrics = simulator.get_metrics()
        prev_action = 0
        
        for step in range(100):
            state = pad_state(simulator.get_state())
            
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = policy_net(state_t)
                q_values[0, valid_actions:] = float('-inf')
                action = q_values.argmax().item()
            
            simulator.apply_level_action(action)
            
            # Run simulation fast
            for _ in range(10):
                simulator.update()
            
            # Spawn more vehicles
            if step % 3 == 0:
                spawn_vehicles_fast(simulator, count=2)
            
            reward, metrics = compute_reward(simulator, prev_metrics, action, prev_action)
            total_reward += reward
            prev_metrics = metrics
            prev_action = action
        
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Throughput: {simulator.get_metrics().get('total_throughput', 0)}")
        print(f"  Vehicles: {len(simulator.vehicles)}")
        
        simulator.running = False
    
    pygame.quit()
    print("\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unified Multi-Level DQN Training')
    parser.add_argument('--episodes', '-e', type=int, default=300,
                        help='Number of training episodes')
    parser.add_argument('--steps', '-s', type=int, default=80,
                        help='Steps per episode')
    parser.add_argument('--test', '-t', action='store_true',
                        help='Test mode (skip training)')
    parser.add_argument('--model', '-m', type=str, default='dqn_unified_best.pth',
                        help='Model file for testing')
    args = parser.parse_args()
    
    if args.test:
        test(args.model)
    else:
        train(episodes=args.episodes, steps_per_episode=args.steps)
