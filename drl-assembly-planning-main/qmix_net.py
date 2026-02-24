import torch
import torch.nn as nn
import torch.nn.functional as F

class QMixer(nn.Module):
    """
    QMIX Mixer network to combine individual Q-values into a total Q-value.
    Uses hypernetworks to ensure monotonicity (non-negative weights).
    """
    def __init__(self, num_agents, state_shape, mixing_embed_dim=32):
        super(QMixer, self).__init__()
        self.num_agents = num_agents
        if isinstance(state_shape, tuple) or isinstance(state_shape, list):
            import numpy as np
            self.state_shape = int(np.prod(state_shape))
        else:
            self.state_shape = state_shape
        self.embed_dim = mixing_embed_dim

        # Hypernetwork for weight 1: (state) -> (num_agents, embed_dim)
        self.hyper_w1 = nn.Sequential(
            nn.Linear(self.state_shape, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, num_agents * mixing_embed_dim)
        )
        
        # Hypernetwork for weight 2: (state) -> (embed_dim, 1)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(self.state_shape, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, mixing_embed_dim)
        )

        # Biases
        self.hyper_b1 = nn.Linear(self.state_shape, mixing_embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(self.state_shape, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1)
        )

    def forward(self, agent_qs, states):
        # agent_qs: (batch, num_agents)
        # states: (batch, state_shape)
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_shape)
        agent_qs = agent_qs.view(-1, 1, self.num_agents)

        # Weights 1
        w1 = torch.abs(self.hyper_w1(states))
        w1 = w1.view(-1, self.num_agents, self.embed_dim)
        b1 = self.hyper_b1(states).view(-1, 1, self.embed_dim)
        
        # Hidden layer
        hidden = F.elu(torch.matmul(agent_qs, w1) + b1) # (bs, 1, embed_dim)

        # Weights 2
        w2 = torch.abs(self.hyper_w2(states))
        w2 = w2.view(-1, self.embed_dim, 1)
        b2 = self.hyper_b2(states).view(-1, 1, 1)

        # Total Q
        q_tot = torch.matmul(hidden, w2) + b2
        return q_tot.view(bs, -1)
