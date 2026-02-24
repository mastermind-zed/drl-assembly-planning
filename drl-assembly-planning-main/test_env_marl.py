from Construction3DEnv_MARL import Construction3DEnvMARL
import random

def test_marl_env():
    strategies = ['centralized', 'decentralized', 'hybrid']
    for strategy in strategies:
        print(f"\n--- Testing Strategy: {strategy} ---")
        env = Construction3DEnvMARL(num_robots=2, comm_strategy=strategy)
        obs = env.reset()
        
        # Move robots apart to test decentralized masking
        # Robot 1 is at random(0,3), Robot 2 is at random(0,3)
        # Let's force Robot 2 to move far away in Step 0
        actions = {1: 3, 2: 0} # Robot 1 Right, Robot 2 Forward (simplified)
        
        for step in range(5):
            obs, rewards, dones, info = env.step(actions)
            for r_id, r_obs in obs.items():
                print(f"Robot {r_id} Obs (last 3 elements - other robot position): {r_obs[-3:]}")
            
            if strategy == 'decentralized':
                # Check if masked (-1) or visible
                pass

            
        if dones["__all__"]:
            print("Episode finished!")
            break

if __name__ == "__main__":
    test_marl_env()
