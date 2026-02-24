import torch
import torch.nn.functional as F
import numpy as np
from tianshou.policy import DQNPolicy
from tianshou.data import Batch

class QMIXPolicy(DQNPolicy):
    def __init__(self, model, mixer, optim, gamma=0.99, n_step=1, 
                 target_update_freq=0, reward_normalization=False, **kwargs):
        super().__init__(model, optim, gamma, n_step, target_update_freq, reward_normalization, **kwargs)
        self.mixer = mixer
        
    def learn(self, batch, **kwargs):
        if self._target and self._iter % self._freq == 0:
            self.sync_weight()
        self.optim.zero_grad()
        
        # batch.obs: (bs, num_agents, obs_dim)
        # batch.act: (bs, num_agents)
        # batch.rew: (bs, num_agents) -> we sum or take global
        # batch.obs_next: ...
        
        # Agent Q-values
        # (bs * num_agents, action_dim)
        obs_flatten = batch.obs.reshape(-1, batch.obs.shape[-1])
        q = self.model(obs_flatten)[0]
        # (bs, num_agents, action_dim)
        q = q.reshape(batch.obs.shape[0], -1, q.shape[-1])
        
        # Selected Q-values
        # (bs, num_agents)
        q_selected = torch.gather(q, 2, batch.act.unsqueeze(2).long()).squeeze(2)
        
        # State estimation for mixer (global state can be mean of obs or actual state)
        # In our env, global state is available or can be constructed
        state = batch.obs.mean(dim=1) # Simplified global state
        
        # Mixed Q-total
        q_tot = self.mixer(q_selected, state)
        
        # Target Q
        with torch.no_grad():
            q_next = self.model_old(batch.obs_next.reshape(-1, batch.obs_next.shape[-1]))[0]
            q_next = q_next.reshape(batch.obs_next.shape[0], -1, q_next.shape[-1])
            q_next_max = q_next.max(dim=2)[0]
            q_next_tot = self.mixer_old(q_next_max, batch.obs_next.mean(dim=1))
            
            # Simple global reward (sum of agent rewards)
            rew_tot = batch.rew.sum(dim=1, keepdim=True)
            target_q = rew_tot + self._gamma * (1 - batch.done.reshape(-1, 1)) * q_next_tot
            
        loss = F.mse_loss(q_tot, target_q)
        loss.backward()
        self.optim.step()
        self._iter += 1
        return {"loss": loss.item()}

    def sync_weight(self):
        super().sync_weight()
        self.mixer_old.load_state_dict(self.mixer.state_dict())

    def set_mixer_old(self, mixer_old):
        self.mixer_old = mixer_old
