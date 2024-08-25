import qmix_nn
import rnn_agent
import torch
import numpy as np
import random
from collections import deque

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class EpsilonGreedy:
    def __init__(self, num_actions, num_agents, final_step, epsilon_start=float(1), epsilon_end=0.05):
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.num_actions = num_actions
        self.final_step = final_step
        self.num_agents = num_agents

    def action(self, value, options):
        if np.random.random() > self.epsilon:
            action = value.max(dim=1)[1].cpu().detach().numpy()
        else:
            action = torch.distributions.Categorical(options).sample().long().cpu().detach().numpy()
        return action
    
    def epislon_decay(self, step):
        prog = step / self.final_step

        decay = self.initial_epsilon - prog
        if decay <= self.epsilon_end:
            decay = self.epsilon_end
        self.epsilon = decay

# Stores and manages experience collected during training
class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        # deque to store experiences
        self.buffer = deque()

    def add_experience(self, state, action, reward, t, obs, options, filled):
        experience = [state, action, reward, t, obs, options, np.array[filled]]
        if (self.count < self.buffer_size):
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count
    
    def sample_experiences_batch(self, batch_size):
        sample_batch = []

        for i in range(batch_size):
            sample_batch.append[self.buffer[i]]
        sample_batch = np.array(sample_batch)

        state_batch = np.array([_[0] for _ in sample_batch], dtype='float32')
        action_batch = np.array([_[1] for _ in sample_batch], dtype='float32')
        reward_batch = np.array([_[2] for _ in sample_batch])
        t_batch = np.array([_[3] for _ in sample_batch])
        obs_batch = np.array([_[4] for _ in sample_batch], dtype='float32')
        options_batch = np.array([_[5] for _ in sample_batch], dtype='float32')
        filled_batch = np.array([_[6] for _ in sample_batch], dtype='float32')

        return state_batch, action_batch, reward_batch, t_batch, obs_batch, options_batch, filled_batch
    
    def clear(self):
        self.buffer.clear()
        self.count = 0


# Manages multiple replay buffers
class EpisodeBatch:

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, replay_buffer):
        if self.count < self.buffer_size: 
            self.buffer.append(replay_buffer)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(replay_buffer)

    def max_episode_len(self, batch):
        max_episode_len = 0

        for replay_buffer in batch:
            _, _, _, t, _, _, _ = replay_buffer.sample_batch(replay_buffer.size())
            for i, t_i in enumerate(t):
                if t_i == True:
                    if i > max_episode_len:
                        max_episode_len = i + 1
                    break
                    
        return max_episode_len
    
    def sample_batch(self, batch_size):
        batch_sample = []

        if self.count < batch_size:
            batch_sample = random.sample(self.buffer, self.count)
        else:
            batch_sample = random.sample(self.buffer, batch_size)
        episode_len = self._get_max_episode_len(batch_sample)
        state_batch, action_batch, reward_batch, t_batch, obs_batch, options_batch, filled_batch = [], [], [], [], [], [], []
        for replay_buffer in batch_sample:
            state, action, reward, t, obs, options, filled = replay_buffer.sample_batch(episode_len)
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            t_batch.append(t)
            obs_batch.append(obs)
            options_batch.append(options)
            filled_batch.append(filled)
        
        filled_batch = np.array(filled_batch)
        reward_batch = np.array(reward_batch)
        t_batch = np.array(t_batch)
        action_batch = np.array(action_batch)
        obs_batch = np.array(obs_batch)
        options_batch = np.array(options_batch)

        return state_batch, action_batch, reward_batch, t_batch, obs_batch, options_batch, filled_batch, episode_len
    
    def size(self):
        return self.count
    
class QMix:
    def __init__(self, training, num_agents, obs_shape, states_shape, num_actions, learning_rate, gamma=0.99, batch_size=16, replay_buffer_size=5000, update_target_network=200, final_step=50000):
        self.training = training
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_target_network = update_target_network
        self.hidden_states = None
        self.target_hidden_states = None
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.state_shape = states_shape
        self.obs_shape = obs_shape
        
        self.epsilon_greedy = EpsilonGreedy(num_actions, num_agents, final_step)
        self.episode_batch = EpisodeBatch(replay_buffer_size)
        
        self.agents = rnn_agent.RNN(obs_shape, n_actions=num_actions).to(device)
        self.target_agents = rnn_agent.RNN(obs_shape, n_actions=num_actions).to(device)
        self.qmix_nn = qmix_nn.QMixNN(num_agents, states_shape, mixing_embed_dim=32).to(device)
        self.target_qmix_nn = qmix_nn.QMixNN(num_agents, states_shape, mixing_embed_dim=32).to(device)

        self.target_agents.update(self.agents)
        self.target_qmix_nn.update(self.qmix_nn)

        self.params = list(self.agents.parameters())
        self.params += self.qmix_nn.parameters()

        self.optimizer = torch.optim.RMSprop(params=self.params, lr=learning_rate, alpha=0.99, eps=0.00001)

    def save_model(self, filename):
        torch.save(self.agents.state_dict(), filename)

    def load_model(self, filename):
        self.agents.load_state_dict(torch.load(filename))
        self.agents.eval()

    def _init_hidden_states(self, batch_size):
        self.hidden_states = self.agents.init_hidden().unsqueeze().expand(batch_size, self.agent_nb, -1)
        self.target_hidden_states = self.target_agents.init_hidden().unsqueeze(0).expand(batch_size, self.agent_nb, -1)

    def decay_epsilon_greddy(self, global_steps):
        self.epsilon_greedy.epislon_decay(global_steps)

    def on_reset(self, batch_size):
        self._init_hidden_states(batch_size)

    def update_targets(self, episode):
        if episode % self.update_target_network == 0 and self.training:
            self.target_agents.copy_params(self.agents)
            self.target_qmix_nn.copy_params(self.qmix_nn)
            pass
    
    def train(self):
        if self.training and self.episode_batch.size() > self.batch_size:
            for _ in range(2):
                self._init_hidden_states(self.batch_size)
                state_batch, actions_batch, reward_batch, t_batch, obs_batch, options_batch, filled_batch, episode_len = self.episode_batch.sample_batch(self.batch_size)

                reward_batch = reward_batch[:, :-1]
                actions_batch = actions_batch[:, :-1]
                t_batch = t_batch[:, :-1]
                filled_batch = filled_batch[:, :-1]

                mask = (1 - filled_batch) * (1 - t_batch)

                reward_batch = torch.FloatTensor(reward_batch).to(device)
                t_batch = torch.FloatTensor(t_batch).to(device)
                mask = torch.FloatTensor(mask).to(device)

                actions_batch = torch.LongTensor(actions_batch).to(device)

                ma_controller_output = []

                for t in range(episode_len):
                    obs = obs_batch[:, t]
                    obs = np.concatenate(obs, axis=0)
                    obs = torch.FloatTensor(obs).to(device)
                    agent_actions, self.hidden_states = self.agents(obs, self.hidden_states)
                    agent_actions = agent_actions.view(self.batch_size, self.agent_nb, -1)
                    ma_controller_output.append(agent_actions)
                ma_controller_output = torch.stack(ma_controller_output, dim=1)

                chosen_action_qvals = torch.gather(ma_controller_output[:, :-1], dim=3, index=actions_batch).squeeze(3)

                target_mac_out = []
                
                for t in range(episode_len):
                    obs = obs_batch[:, t]
                    obs = np.concatenate(obs, axis=0)
                    obs = torch.FloatTensor(obs).to(device)
                    agent_actions, self.target_hidden_states = self.target_agents(obs, self.target_hidden_states)
                    agent_actions = agent_actions.view(self.batch_size, self.num_agents, -1)
                    target_mac_out.append(agent_actions)
                target_mac_out = torch.stack(target_mac_out[1:], dim=1)
                options_batch = torch.Tensor(options_batch).to(device)

                target_mac_out[options_batch[:, 1:] == 0] = -9999999
                
                target_max_qvals = target_mac_out.max(dim=3)[0]

                states = torch.FloatTensor(state_batch).to(device)

                chosen_action_qvals = self.qmix_nn(chosen_action_qvals, states[:, :-1])
                target_max_qvals = self.target_qmix_nn(target_max_qvals, states[:, 1:])

                yi = reward_batch + self.gamma * (1 - t_batch) * target_max_qvals

                td_error = (chosen_action_qvals - yi.detach())

                mask = mask.expand_as(td_error)

                masked_td_error = td_error * mask

                loss = (masked_td_error ** 2).sum() / mask.sum()
                #print('loss:', loss)
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.params, 10)
                self.optimizer.step()

            pass
        pass

    
    def act(self, batch, obs, agents_options):
        value_action, self.hidden_states = self.agents(obs, self.hidden_states)
        value_action[agents_options == 0] = -1e10
        if self.training:
            value_action = self.epsilon_greedy.act(value_action, agents_options)
        else:
            value_action = np.argmax(value_action.cpu().data.numpy(), -1)
        value_action = value_action.reshape(batch, self.agent_nb, -1)
        return value_action