import numpy as np
import torch
import os
from maddpg.odits import ODITS


class Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        self.policy = ODITS(args, agent_id)

    def select_action(self, o, u, noise_rate, epsilon):
        if u is None or np.random.uniform() < epsilon:
            u = np.random.uniform(-self.args.high_action, self.args.high_action, self.args.action_shape[self.agent_id])
        else:
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
            t_u = torch.tensor(u, dtype=torch.float32)
            # print("t_u",len(t_u.size()), len(inputs.size()))
            if len(t_u.size()) < len(inputs.size()):
                t_u = t_u.unsqueeze(0)
            pi = self.policy.actor_network(inputs, t_u)[0].squeeze(0)
            # print('{} : {}'.format(self.name, pi))
            u = pi.cpu().numpy()
            noise = noise_rate * self.args.high_action * np.random.randn(*u.shape)  # gaussian noise
            u += noise
            u = np.clip(u, -self.args.high_action, self.args.high_action)
        
        return u.copy()

    def learn(self, transitions, other_agents):
        self.policy.train(transitions, other_agents)

