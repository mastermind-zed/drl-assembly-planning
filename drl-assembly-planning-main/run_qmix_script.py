import argparse
import os
import numpy as np
import torch
import copy
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import VectorReplayBuffer, Collector, Batch
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import Net
from tianshou.env import DummyVectorEnv, PettingZooEnv
from pettingzoo.utils.conversions import parallel_to_aec
from tianshou.utils import TensorboardLogger

from Construction3DEnv_MARL import Construction3DEnvMARL
from qmix_net import QMixer
from QMIXPolicy import QMIXPolicy

# Reuse AECWrapper for consistency
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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-sizes", type=int, nargs='*', default=[128, 128])
    parser.add_argument("--training-num", type=int, default=4)
    parser.add_argument("--test-num", type=int, default=4)
    parser.add_argument("--num-robots", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--logdir", type=str, default="log")
    return parser.parse_args()

def train_qmix(args=get_args()):
    def make_env():
        par_env = Construction3DEnvMARL(num_robots=args.num_robots)
        aec_env = parallel_to_aec(par_env)
        return PettingZooEnv(AECWrapper(aec_env))
    
    env = make_env()
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n

    train_envs = DummyVectorEnv([make_env for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([make_env for _ in range(args.test_num)])

    # Individual Agent Model (Shared weights)
    net = Net(args.state_shape, args.action_shape, hidden_sizes=args.hidden_sizes, device=args.device).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    # QMIX Mixer
    # Note: QMIX usually uses a global state, here we approximate with mean of obs
    mixer = QMixer(args.num_robots, args.state_shape).to(args.device)
    mixer_old = copy.deepcopy(mixer)

    # Simplified QMIX script using a shared DQN policy 
    # and manual mixing in the training loop 
    # OR using the custom QMIXPolicy (preferred)
    
    policy = QMIXPolicy(
        net, mixer, optim, args.gamma,
        target_update_freq=500, # Steps
    ).to(args.device)
    policy.set_mixer_old(mixer_old)

    # Buffer and Collector
    buffer = VectorReplayBuffer(20000, len(train_envs))
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=False)

    # Log
    log_path = os.path.join(args.logdir, "qmix", "construction")
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    # Trainer
    result = offpolicy_trainer(
        policy, train_collector, test_collector,
        args.epoch, args.step_per_epoch, args.step_per_collect,
        args.test_num, args.batch_size,
        update_per_step=args.update_per_step,
        logger=logger
    )
    print(result)

if __name__ == "__main__":
    train_qmix(get_args())
