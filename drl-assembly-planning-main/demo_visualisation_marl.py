import argparse
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np 
import time
from datetime import datetime
import torch

from Construction3DEnv_MARL import Construction3DEnvMARL
from BIMClass.drawSite import draw_site, draw_cube
from tianshou.policy import MultiAgentPolicyManager, PPOPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor

# Re-importing NetWrapper to match training
class NetWrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor) and hasattr(obs, "obs"):
            obs = obs.obs
        return self.net(obs, state, info)
    @property
    def output_dim(self):
        return self.net.output_dim

def setup_pygame():
    pygame.init()
    display = (1000, 800)
    pygame.display.set_caption("Multi-Robot ICF Assembly Simulation")
    try:
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    except Exception as e:
        print(f"Failed to initialize OpenGL window: {e}")
        raise e
        
    gluPerspective(45, display[0] / display[1], 10, 10000.0)
    # Move MUCH further back because Cube_Size=100 in drawSite.py
    glTranslate(0, -500, -2500)
    glRotatef(35, 1, 0, 0)
    glEnable(GL_DEPTH_TEST)

def draw_robots(siteEnv):
    Cube_Size = 100
    center_x = len(siteEnv.site_3D[0][0]) / 2
    center_y = len(siteEnv.site_3D[0]) / 2
    for robot in siteEnv.robots:
        # Draw robot as a blue cube (type 2)
        draw_cube((robot.y - center_x) * Cube_Size, (robot.z - 1) * Cube_Size, (robot.x - center_y) * -Cube_Size, Cube_Size, 2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-robots', type=int, default=2)
    parser.add_argument('--comm-strategy', type=str, default='centralized')
    parser.add_argument('--ckpt-path', type=str, default=None)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    env = Construction3DEnvMARL(num_robots=args.num_robots, comm_strategy=args.comm_strategy)
    setup_pygame()

    # Load policy if path provided
    policy = None
    if args.ckpt_path:
        # Re-construct same policy structure as training
        state_shape = env.observation_space.shape or env.observation_space.n
        action_shape = env.action_space.shape or env.action_space.n
        actor_net = Net(state_shape, hidden_sizes=[128, 128], device=args.device).to(args.device)
        actor = Actor(NetWrapper(actor_net), action_shape, device=args.device).to(args.device)
        # Dummy critic/optim for loading
        base_policy = PPOPolicy(actor, None, None, None, action_space=env.action_space)
        agents = [base_policy for _ in range(args.num_robots)]
        policy = MultiAgentPolicyManager(agents, env)
        policy.load_state_dict(torch.load(args.ckpt_path, map_location=args.device))
        policy.eval()

    obs, info = env.reset()
    print("Starting simulation loop. Press 'Esc' to exit.")
    print("Controls: Arrows to Rotate, W/S to Zoom.")
    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                    # Simple navigation
                    if event.key == pygame.K_UP: glRotatef(5, 1, 0, 0)
                    if event.key == pygame.K_DOWN: glRotatef(-5, 1, 0, 0)
                    if event.key == pygame.K_LEFT: glRotatef(5, 0, 1, 0)
                    if event.key == pygame.K_RIGHT: glRotatef(-5, 0, 1, 0)
                    if event.key == pygame.K_w: glTranslate(0, 0, 100)
                    if event.key == pygame.K_s: glTranslate(0, 0, -100)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            if policy:
                # Simplified policy handling
                obs_batch = np.array(list(obs.values()))
                results = policy(torch.from_numpy(obs_batch).to(args.device))
                actions = {agent_id: results.act[i] for i, agent_id in enumerate(obs.keys())}
            else:
                # Random actions
                actions = {agent_id: env.action_space(agent_id).sample() for agent_id in obs.keys()}

            obs, rewards, terminations, truncations, infos = env.step(actions)
            
            # Rendering
            draw_site(env.siteEnv)
            draw_robots(env.siteEnv)
            
            pygame.display.flip()
            pygame.time.wait(50) # Approx 20 FPS
            
            if env.step_n % 50 == 0:
                print(f"Step: {env.step_n}, Arrived: {len(env.siteEnv.arrived_scos)}/{len(env.siteEnv.scos)}")

            if any(terminations.values()) or any(truncations.values()):
                print("Simulation finished (Terminal state reached).")
                break
    except Exception as e:
        print(f"Runtime error in visualization: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pygame.quit()

if __name__ == '__main__':
    main()
