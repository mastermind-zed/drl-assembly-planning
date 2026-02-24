import argparse
import numpy as np
import torch
import time
from Construction3DEnv_MARL import Construction3DEnvMARL

def collect_metrics(num_episodes=5, num_robots=2, comm_strategy='centralized'):
    env = Construction3DEnvMARL(num_robots=num_robots, comm_strategy=comm_strategy)
    
    all_episode_rewards = []
    all_success_counts = []
    all_steps = []
    
    print(f"Starting evaluation: {num_robots} robots, Strategy: {comm_strategy}")
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        while not done:
            # Random policy for baseline metrics if no model provided
            actions = {agent_id: env.action_space(agent_id).sample() for agent_id in obs.keys()}
            obs, rewards, terminations, truncations, infos = env.step(actions)
            
            episode_reward += sum(rewards.values())
            steps += 1
            
            if any(terminations.values()) or any(truncations.values()):
                done = True
        
        arrived_count = len(env.siteEnv.arrived_scos)
        all_episode_rewards.append(episode_reward)
        all_success_counts.append(arrived_count)
        all_steps.append(steps)
        
        print(f"Episode {ep+1}: Steps={steps}, Rewards={episode_reward:.2f}, Arrived={arrived_count}")

    print("\n--- Summary Metrics ---")
    print(f"Avg Steps: {np.mean(all_steps):.2f}")
    print(f"Avg Reward: {np.mean(all_episode_rewards):.2f}")
    print(f"Avg Success Count: {np.mean(all_success_counts):.2f}")
    print(f"Max Success in single episode: {np.max(all_success_counts)}")
    print("------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--robots", type=int, default=2)
    parser.add_argument("--strategy", type=str, default='centralized')
    args = parser.parse_args()
    
    collect_metrics(args.episodes, args.robots, args.strategy)
    
    # Run for different strategies
    # for strat in ['centralized', 'decentralized', 'hybrid']:
    #    collect_metrics(3, args.robots, strat)
