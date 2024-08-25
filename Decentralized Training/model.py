import argparse
from collections import deque, namedtuple
import random

from pettingzoo.butterfly import knights_archers_zombies_v10
import torch
import numpy as np
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

# transition to store experiences in the replay buffer
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


# Replay Buffer to store and sample experience
class ReplayBuffer(object):

    def __init__(self, capacity):
        self.memory = deque([], capacity)

    def push(self, *args):
        # ddd a new experience to the memory
        self.memory.append(Transition(*args))

    def sample(self, batch_s):
        # randomly sample a batch of experiences from the memory
        return random.sample(self.memory, batch_s)

    def __len__(self):
        # return the current size of the memory
        return len(self.memory)


# initialize Deep Q learning network as class
class DQN(nn.Module):
    def __init__(self, n_dimensions, n_actions):
        super(DQN, self).__init__()
        # flatten the input dimensions for the first layer
        flatten = n_dimensions[0] * n_dimensions[1]
        self.layer1 = nn.Linear(flatten, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, n_actions)

    def forward(self, x):
        # if np array convert to tensor for the NN
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.view(-1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# get the number of actions for each agent
num_agent_act = np.array([6, 6, 6, 6])


class AgentDQN:
    def __init__(self, learning_rate, gamma, eps_start_val, eps_end_val, eps_decay, num_actions, observation_size,
                 agent_name, tau):
        # initialize hyperparameter values, model, target network, and agent name
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.eps_start_val = eps_start_val
        self.eps_decay = eps_decay
        self.eps_end_val = eps_end_val
        self.model = DQN(observation_size, num_actions)
        self.target_net = DQN(observation_size, num_actions)
        self.target_net.load_state_dict(self.model.state_dict())
        self.memory = ReplayBuffer(1000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.steps_done = 0
        self.agent_name = agent_name
        self.tau = tau

    # select an action based on epsilon-greedy policy
    def select_action(self, s):

        sample = random.random()
        eps_threshold = self.eps_end_val + (self.eps_start_val - self.eps_end_val) * math.exp(
            -1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.model(s).max(0).indices.item()
        else:
            return random.randrange(0, 6)

    # store the transition in the replay buffer
    def remember(self, s, a, r, new_s, d):
        self.memory.push(s, a, r, new_s, d)

    # complete a training step using experiences from the replay buffer
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = self.memory.sample(batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # update the target value if the episode is not done
                target += self.gamma * torch.max(self.target_net(next_state))
                current_q_value = self.model(state)[action]

                self.optimizer.zero_grad()
                loss = nn.MSELoss()(current_q_value, target)
                loss.backward()
                self.optimizer.step()

        # update the target network with soft updates
        self.soft_update(self.model, self.target_net, self.tau)
        return loss.item()

    # soft update for target network parameters
    def soft_update(self, policy_net, target_net, tau):
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

    # save model into given filepath
    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    # load trained model given the file path for continued training or testing
    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
        self.model.eval()


# define vectorized observation space where n is num_archers + num_knights + num_swords + max_arrows + max_zombies + 1
observation_shape = (27, 5)

# list of all agent names in the environment
all_agents = ['archer_0', 'archer_1', 'knight_0', 'knight_1']

# initialize the agents with the specified parameters
archer_0 = AgentDQN(learning_rate=0.0001, gamma=0.99, eps_start_val=0.9, eps_end_val=0.05, eps_decay=1000,
                    num_actions=num_agent_act[0], observation_size=observation_shape, agent_name='archer_0', tau=0.005)

archer_1 = AgentDQN(learning_rate=0.0001, gamma=0.99, eps_start_val=0.9, eps_end_val=0.05, eps_decay=1000,
                    num_actions=num_agent_act[1], observation_size=observation_shape, agent_name='archer_1', tau=0.005)

knight_0 = AgentDQN(learning_rate=0.0001, gamma=0.99, eps_start_val=0.9, eps_end_val=0.05, eps_decay=1000,
                    num_actions=num_agent_act[2], observation_size=observation_shape, agent_name='knight_0', tau=0.005)

knight_1 = AgentDQN(learning_rate=0.0001, gamma=0.99, eps_start_val=0.9, eps_end_val=0.05, eps_decay=1000,
                    num_actions=num_agent_act[3], observation_size=observation_shape, agent_name='knight_1', tau=0.005)

env = knights_archers_zombies_v10.env(render_mode="human", spawn_rate=20, num_archers=2, num_knights=2, max_zombies=10,
                                      max_arrows=10, killable_knights=True, killable_archers=True, pad_observation=True,
                                      line_death=True, max_cycles=900,
                                      vector_state=True, use_typemasks=False, sequence_space=False)
# dictionary to store the models of all agents
agent_models = {
    'archer_0': archer_0.model,
    'archer_1': archer_1.model,
    'knight_0': knight_0.model,
    'knight_1': knight_1.model
}


# function to train the agents
def train():
    agent_list = [archer_0, archer_1, knight_0, knight_1]
    checkpoint_path = 'checkpoint.pth'
    try:
        # load from a checkpoint to resume training
        checkpoint = torch.load(checkpoint_path)
        for agent in agent_list:
            agent.model.load_state_dict(checkpoint['model_state_dict'][agent.agent_name])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'][agent.agent_name])
        start_episode = checkpoint['episode']
        print(f"Resuming training from episode {start_episode}")
    except FileNotFoundError:
        # if no checkpoint is found, start from scratch
        print("Checkpoint not found")
        start_episode = 0

    num_episodes = 1000
    batch_size = 128
    total_reward_vals = np.array([], dtype=int)
    last_observations = {agent: None for agent in agent_list}
    loss_vals = {agent_name: [] for agent_name in ['archer_0', 'archer_1', 'knight_0', 'knight_1']}
    for ep in range(start_episode, start_episode + num_episodes):
        env.reset(seed=42)

        total_reward = 0
        done = False
        # iterate over the agents in the episode
        while not done:
            for agent in env.agent_iter():
                if agent == 'archer_0':
                    agent = agent_list[0]
                elif agent == 'archer_1':
                    agent = agent_list[1]
                elif agent == 'knight_0':
                    agent = agent_list[2]
                else:
                    agent = agent_list[3]

                observation, reward, termination, truncation, info = env.last()
                done = truncation or termination
                if done:
                    action = None

                else:
                    # epsilon greedy
                    action = agent.select_action(observation)

                env.step(action)

                # initially the observation step is the current state but after the next iteration
                # it becomes the next state for the replay buffer and the previous state becomes the current
                if last_observations[agent] is not None:
                    agent.remember(last_observations[agent], action, reward, observation, termination)
                    total_reward += int(reward)
                    loss = agent.replay(batch_size)

                    # add loss values to list to for visualization
                    if loss is not None:
                        loss_vals[agent.agent_name].append(loss)

                last_observations[agent] = observation

            if done:
                env.reset()
                break

        # complete soft update for each agent's target network
        for agent in agent_list:
            agent.soft_update(agent.model, agent.target_net, agent.tau)

        total_reward_vals = np.append(total_reward_vals, total_reward)
        print(f"Episode {ep + 1} finished with total reward: {total_reward}")

        # save a checkpoint every few episodes (e.g., every 1000 episodes)
        if (ep + 1) % 1000 == 0:
            checkpoint = {
                'model_state_dict': {agent.agent_name: agent.model.state_dict() for agent in agent_list},
                'optimizer_state_dict': {agent.agent_name: agent.optimizer.state_dict() for agent in agent_list},
                'episode': ep + 1,
            }
            torch.save(checkpoint, 'checkpoint.pth')
            print(f"Checkpoint saved for episode {ep + 1}")

    env.close()

    # save model
    archer_0.save_model('archer_0_model.pth')
    archer_1.save_model('archer_1_model.pth')
    knight_0.save_model('knight_0_model.pth')
    knight_1.save_model('knight_1_model.pth')

    return total_reward_vals, loss_vals


# function to evaluate the performance of trained agents
def evaluate():
    #  load the saved models for evaluation
    archer_0.load_model('Decentralized Training/archer_0_model.pth')
    archer_1.load_model('Decentralized Training/archer_1_model.pth')
    knight_0.load_model('Decentralized Training/knight_0_model.pth')
    knight_1.load_model('Decentralized Training/knight_0_model.pth')

    # set the models to evaluation mode
    archer_0.model.eval()
    archer_1.model.eval()
    knight_0.model.eval()
    knight_1.model.eval()

    total_reward_vals = np.array([], dtype=int)
    num_episodes = 100

    for ep in range(num_episodes):
        env.reset(seed=42)
        total_reward = 0
        done = False

        while not done:
            for agent_name in env.agent_iter():
                if agent_name == 'archer_0':
                    agent = archer_0
                elif agent_name == 'archer_1':
                    agent = archer_1
                elif agent_name == 'knight_0':
                    agent = knight_0
                else:
                    agent = knight_1

                observation, reward, termination, truncation, info = env.last()
                done = truncation or termination

                if not done:
                    action = agent.select_action(observation)
                else:
                    action = None

                env.step(action)
                total_reward += int(reward)

            if done:
                break

        total_reward_vals = np.append(total_reward_vals, total_reward)

    env.close()

    # get the average reward over all evaluation episodes and print
    average_reward = np.mean(total_reward_vals)
    print(f"Average reward over {num_episodes} evaluation episodes: {average_reward}")

    return total_reward_vals


# plot the training losses for each agent
def plot_training_loss(training_losses):
    # Plot Training Loss for Each Agent
    for agent_name, losses in training_losses.items():
        plt.figure(figsize=(12, 6))
        plt.plot(losses, label=f"{agent_name} Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title(f"Loss During Training for {agent_name}")
        plt.legend()
        plt.show()


# plot the total rewards per episode during training
def plot_training_rewards(training_rewards):
    # Plot Training Rewards
    plt.figure(figsize=(12, 6))
    plt.plot(training_rewards, label="Training Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode During Training")
    plt.legend()
    plt.show()


# plot the total rewards per episode during evaluation
def plot_eval_rewards(evaluation_rewards):
    # Plot Evaluation Rewards
    plt.figure(figsize=(12, 6))
    plt.plot(evaluation_rewards, label="Evaluation Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode During Evaluation")
    plt.legend()
    plt.show()


# main function to complete the training, evaluation, and plotting
def main():
    parser = argparse.ArgumentParser(description="DQN Training, Evaluation, with Plotting")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'eval', 'plot'],
                        help="Mode to run: 'train', 'eval', or 'plot'")

    args = parser.parse_args()

    if args.mode == 'train':
        training_rewards, training_losses = train()
        plot_training_loss(training_losses)
        plot_training_rewards(training_rewards)
    elif args.mode == 'eval':
        evaluation_rewards = evaluate()
        plot_eval_rewards(evaluation_rewards)
    else:
        print("Please specify a valid mode.")


if __name__ == "__main__":
    main()
