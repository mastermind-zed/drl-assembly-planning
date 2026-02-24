import argparse
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import VectorReplayBuffer, Collector
from tianshou.policy import MultiAgentPolicyManager, PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.common import Net, ActorCritic
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.env import DummyVectorEnv, PettingZooEnv
from tianshou.utils import TensorboardLogger
from pettingzoo.utils.conversions import parallel_to_aec

from Construction3DEnv_MARL import Construction3DEnvMARL

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--step-per-collect", type=int, default=100)
    parser.add_argument("--repeat-per-collect", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-sizes", type=int, nargs='*', default=[128, 128])
    parser.add_argument("--training-num", type=int, default=4)
    parser.add_argument("--test-num", type=int, default=4)
    parser.add_argument("--num-robots", type=int, default=2)
    parser.add_argument("--comm-strategy", type=str, default='centralized')
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--logdir", type=str, default="log")
    return parser.parse_args()

class AECWrapper:
    def __init__(self, env):
        self.env = env
    def last(self, observe=True):
        res = self.env.last(observe)
        if len(res) == 5:
            obs, rew, term, trunc, info = res
            return obs, rew, term or trunc, info
        return res
    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)
    def step(self, action):
        return self.env.step(action)
    def __getattr__(self, name):
        return getattr(self.env, name)

class NetWrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, obs, state=None, info={}):
        # In MARL, obs sometimes comes as a Batch/dict containing 'agent_id' and 'obs'
        if not isinstance(obs, torch.Tensor) and hasattr(obs, "obs"):
            obs = obs.obs
        return self.net(obs, state, info)
    @property
    def output_dim(self):
        return self.net.output_dim

def train_mappo(args=get_args()):
    # Create single env for shapes
    def make_env():
        par_env = Construction3DEnvMARL(num_robots=args.num_robots, comm_strategy=args.comm_strategy)
        aec_env = parallel_to_aec(par_env)
        return PettingZooEnv(AECWrapper(aec_env))
    
    env = make_env()
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n

    # Vector Envs
    train_envs = DummyVectorEnv([make_env for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([make_env for _ in range(args.test_num)])

    # Setup Policy
    # Shared Networks
    actor_net = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device).to(args.device)
    critic_net = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device).to(args.device)
    
    # Wrap networks to handle MARL batch structure
    actor = Actor(NetWrapper(actor_net), args.action_shape, device=args.device).to(args.device)
    critic = Critic(NetWrapper(critic_net), device=args.device).to(args.device)
    optim = torch.optim.Adam(ActorCritic(actor, critic).parameters(), lr=args.lr)

    def dist(p):
        return torch.distributions.Categorical(logits=p)

    # Shared Policy for all robots
    base_policy = PPOPolicy(
        actor, critic, optim, dist,
        discount_factor=args.gamma,
        action_space=env.action_space,
        eps_clip=0.2,
        value_clip=0,
        dual_clip=None,
        advantage_normalization=True,
        recompute_advantage=False,
    )

    # MultiAgent Manager
    # agents are mapped by their robot.id (1, 2, ...)
    agents = [base_policy for _ in range(args.num_robots)]
    policy = MultiAgentPolicyManager(agents, env)

    # Collector
    train_collector = Collector(policy, train_envs, VectorReplayBuffer(20000, len(train_envs)))
    test_collector = Collector(policy, test_envs)

    # Logger
    log_path = os.path.join(args.logdir, "mappo", args.comm_strategy)
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    # Trainer
    result = onpolicy_trainer(
        policy, train_collector, test_collector,
        args.epoch, args.step_per_epoch, args.repeat_per_collect,
        args.test_num, args.batch_size,
        step_per_collect=args.step_per_collect,
        logger=logger
    )
    print(result)

if __name__ == "__main__":
    train_mappo(get_args())
