import gym
from gym import spaces
import numpy as np
from BIMClass.Site.multi_robot_site import MultiRobotSite
from pettingzoo import ParallelEnv

class Construction3DEnvMARL(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "construction_v0"}

    def __init__(self, env_id=1, num_robots=2, normalise=False, comm_strategy='centralized'):
        super().__init__()
        self.num_robots = num_robots
        self.normalise = normalise
        self.comm_strategy = comm_strategy # 'centralized', 'decentralized', 'hybrid'
        self.visibility_radius = 5.0
        
        # Site needs to be (re)initialized each time but we need shapes now
        dummy_site = MultiRobotSite(15, 15, 8, num_robots=num_robots)
        self.agents = [str(r.id) for r in dummy_site.robots]
        self.possible_agents = self.agents[:]
        
        self.obs_dim = 10 + (num_robots - 1) * 3
        self._observation_spaces = {agent: spaces.Box(low=-1, high=100, shape=(self.obs_dim,), dtype=np.float32) for agent in self.agents}
        self._action_spaces = {agent: spaces.Discrete(8) for agent in self.agents}
        
        self.max_timestep = 500
        self.step_n = 0
        self.siteEnv = None

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def get_agent_obs(self, robot):
        # Basic state
        obs = [
            robot.x / self.siteEnv.s_wid,
            robot.y / self.siteEnv.s_len,
            robot.z / self.siteEnv.s_he,
            robot.battery / 100.0,
            1.0 if robot.carried_sco else 0.0
        ]
        
        if robot.carried_sco:
            sco = robot.carried_sco
            obs.extend([sco.x_tar_1 / self.siteEnv.s_wid, sco.y_tar_1 / self.siteEnv.s_len, sco.z_tar_1 / self.siteEnv.s_he])
        else:
            obs.extend([0, 0, 0])
            
        if self.comm_strategy in ['centralized', 'hybrid']:
            obs.extend([len(self.siteEnv.arrived_scos) / len(self.siteEnv.scos), 0])
        else:
            obs.extend([0, 0])
        
        for other in self.siteEnv.robots:
            if other.id != robot.id:
                visible = True
                if self.comm_strategy == 'decentralized':
                    dist = np.sqrt((robot.x - other.x)**2 + (robot.y - other.y)**2 + (robot.z - other.z)**2)
                    if dist > self.visibility_radius:
                        visible = False
                
                if visible:
                    obs.extend([other.x / self.siteEnv.s_wid, other.y / self.siteEnv.s_len, other.z / self.siteEnv.s_he])
                else:
                    obs.extend([-1, -1, -1])
        return np.array(obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.siteEnv = MultiRobotSite(15, 15, 8, num_robots=self.num_robots)
        self.step_n = 0
        self.agents = self.possible_agents[:]
        observations = {agent: self.get_agent_obs(self.siteEnv.robots[int(agent)-1]) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        """
        actions: dict {agent_id: action_index}
        """
        int_actions = {int(k): v for k, v in actions.items()}
        self.step_n += 1
        rewards_raw, dones_robot = self.siteEnv.step(int_actions)
        
        observations = {agent: self.get_agent_obs(self.siteEnv.robots[int(agent)-1]) for agent in self.agents}
        rewards = {agent: rewards_raw[int(agent)] for agent in self.agents}
        
        all_arrived = len(self.siteEnv.arrived_scos) == len(self.siteEnv.scos)
        global_done = all_arrived or self.step_n >= self.max_timestep
        
        terminations = {agent: global_done for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {"arrived_count": len(self.siteEnv.arrived_scos)} for agent in self.agents}
        
        if global_done:
            self.agents = []
            
        return observations, rewards, terminations, truncations, infos

    def render(self, mode='human'):
        print(f"Step: {self.step_n}, Arrived: {len(self.siteEnv.arrived_scos)}/{len(self.siteEnv.scos)}")
        for r in self.siteEnv.robots:
            print(f"Robot {r.id}: ({r.x}, {r.y}, {r.z}), Status: {r.status}")
