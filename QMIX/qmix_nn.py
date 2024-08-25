import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Mixes Q vals from multiple agents to create joint action Q val
class QMixNN(nn.Module):
    def __init__(self, n_agents, state_shape, hidden_dim = 64):
        super(QMixNN, self).__init__()

        self.n_agents = n_agents
        self.state_dim = np.prod(state_shape)
        self.embed_dim = hidden_dim

        # weights for first hidden layer
        self.hyper_w1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
        # weight s for final hidden layer
        self.hyper_wfinal = nn.Linear(self.state_dim, self.embed_dim)

        # Bias for hidden layer states
        self.hyper_b1 = nn.Linear(self.state_dim, self.embed_dim)

        # Values V(s) for final layer, reduced to scalar value
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))
        
    def forward(self, agent_q, states):
        batch_size = agent_q.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_q = agent_q.view(-1, 1, self.n_agents)
        #layer 1
        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_q, w1) + b1)

        wfinal = torch.abs(self.hyper_wfinal(states))
        wfinal = wfinal.view(-1, self.embed_dim, 1)
        v = self.V(states).view(-1, 1, 1)

        y = torch.bmm(hidden, wfinal)
        qfinal = y.view(batch_size, -1, 1)
        return qfinal
    
    def copy_params(self, agent):
        for param, target_param in zip(agent.parameters(), self.parameters()):
            target_param.data.copy_(param.data)

