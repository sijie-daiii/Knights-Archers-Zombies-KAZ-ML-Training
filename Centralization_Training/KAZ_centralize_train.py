import argparse
import os
from typing import Optional, Tuple, List
import pprint
import time

import gym
import numpy as np
import torch
import pygame
from torch.utils.tensorboard import SummaryWriter
from functools import partial
import matplotlib.pyplot as plt

from pettingzoo.butterfly import knights_archers_zombies_v10
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import (
    BasePolicy,
    DQNPolicy,  # DQN Algorithm: This is the core policy used for each agent
    MultiAgentPolicyManager,  # Multi-Agent Policy Management: This is used to manage multiple agents
)
from tianshou.trainer import offpolicy_trainer  # Off-Policy Training: Used to train the agents
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net  # DQN Algorithm: The network used by DQN to approximate Q-values


def setup_environment_and_seed(args, render_mode=None):
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initialize multiple instances of the environment with the specified parameters
    envs = [lambda: PettingZooEnv(knights_archers_zombies_v10.env(
        max_arrows=1000,
        killable_knights=False,
        killable_archers=False,
        max_zombies=4,
        render_mode=render_mode
    )) for _ in range(args.num_envs)]

    # Create a vectorized environment for parallel execution
    train_envs = DummyVectorEnv(envs)
    train_envs.seed(args.seed)
    return train_envs


def parse_arguments_and_initialize_envs() -> Tuple[argparse.Namespace, DummyVectorEnv]:
    # Parse command-line arguments for configuring the environment and agents
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1626)
    # Epsilon-Greedy Exploration Strategy
    parser.add_argument('--test_eps', type=float, default=0.05)
    # Epsilon-Greedy Exploration Strategy
    parser.add_argument('--train_eps', type=float, default=0.7)
    # Off-Policy Training with Experience Replay
    parser.add_argument('--buffer_cap', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=30e-4)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--num_archers', type=int, default=2)
    parser.add_argument('--num_knights', type=int, default=2)
    parser.add_argument('--n_step', type=int, default=1)
    # DQN Algorithm: Target network update frequency
    parser.add_argument('--target_update', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--steps_per_epoch', type=int, default=2000)
    parser.add_argument('--collect_per_step', type=int, default=10)
    parser.add_argument('--updates_per_collect', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_layers', type=int, nargs='*', default=[512, 256])
    parser.add_argument('--train_envs', type=int, default=10)
    parser.add_argument('--test_envs', type=int, default=5)
    parser.add_argument('--log_path', type=str, default='log')
    parser.add_argument('--render_freq', type=float, default=0.005)
    parser.add_argument('--win_rate', type=float, default=0.6, help='Desired win rate')
    parser.add_argument('--observe_only', default=False, action='store_true')
    parser.add_argument('--agent_id', type=int, default=2, help='Player ID (1 or 2)')
    parser.add_argument('--resume_path', type=str, default='')
    parser.add_argument('--opponent_path', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_envs', type=int, default=10, help="Total environments")

    # Parse the arguments and initialize the environment
    args = parser.parse_args()
    train_envs = setup_environment_and_seed(args)
    return args, train_envs


def initialize_agents_and_envs(
        args: argparse.Namespace,
        envs: DummyVectorEnv
) -> Tuple[BasePolicy, List[torch.optim.Optimizer], List, List]:
    # Extract observation and action space from the environment
    first_env = envs.workers[0].env
    observation_space = first_env.observation_space['observation'] if isinstance(
        first_env.observation_space, gym.spaces.Dict) else first_env.observation_space
    args.state_shape = observation_space.shape or observation_space.n
    args.action_shape = first_env.action_space.shape or first_env.action_space.n

    agents, optimizers, schedulers = [], [], []

    # Initialize agents, optimizers, and schedulers for each knight and archer
    for _ in range(args.num_archers + args.num_knights):
        net = Net(
            args.state_shape,
            args.action_shape,
            hidden_sizes=args.hidden_layers,
            device=args.device
        ).to(args.device)
        # DQN Algorithm: Optimizing the Q-network
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
        agent = DQNPolicy(
            # DQN Algorithm: The Q-network
            net,
            optimizer,
            args.gamma,
            args.n_step,
            # DQN Algorithm: Target network update frequency
            target_update_freq=args.target_update
        )
        agents.append(agent)
        optimizers.append(optimizer)
        schedulers.append(scheduler)

    # Multi-Agent Policy Management: Manage multiple agents using MultiAgentPolicyManager
    policy = MultiAgentPolicyManager(agents, first_env)
    return policy, optimizers, first_env.env.agents, schedulers


def setup_training_and_logger(args, policy, envs):
    # Create a directory for logging data
    log_dir = os.path.join('data')
    os.makedirs(log_dir, exist_ok=True)

    # Setup Tensorboard logger for visualizing training metrics
    writer = SummaryWriter(log_dir)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    # Off-Policy Training with Experience Replay: Create a collector to gather experience during training
    collector = Collector(
        policy,
        envs,
        VectorReplayBuffer(args.buffer_cap, len(envs)),  # Off-Policy Training with Experience Replay: Replay buffer
        exploration_noise=True
    )
    return logger, collector


def execute_training_loop(args, policy, optimizers, schedulers, logger, collector):
    rewards = []
    losses = []

    # Adjust the learning rate and epsilon (for exploration) during training
    def adjust_learning_rate(epoch, env_step):
        for optimizer in optimizers:
            optimizer.step()
        for scheduler in schedulers:
            scheduler.step()
        for agent in policy.policies.values():
            epsilon = max(0.1, args.train_eps * (0.99 ** epoch))
            # Epsilon-Greedy Exploration Strategy: Adjust epsilon for exploration
            agent.set_eps(epsilon)

    # Evaluate the policy and log the results
    def evaluate_policy(epoch, env_step):
        for agent in policy.policies.values():
            # Epsilon-Greedy Exploration Strategy: Use low epsilon during testing
            agent.set_eps(args.test_eps)

        # Collect test results and log average reward
        test_result = collector.collect(n_episode=args.test_envs)
        avg_reward = np.mean(test_result["rews"])
        rewards.append(avg_reward)
        logger.writer.add_scalar('test/avg_reward', avg_reward, epoch)

    # Manually log losses during training
    def log_loss(epoch, env_step):
        # Log loss for each agent
        for agent_id, agent in enumerate(policy.policies.values()):
            loss = agent._optimizer.param_groups[0]['loss']
            losses.append(loss)
            logger.writer.add_scalar(f'train/loss_agent_{agent_id}', loss, env_step)

    # Off-Policy Training: Train the agents using off-policy learning
    result = offpolicy_trainer(
        policy,
        collector,
        collector,
        args.epochs,
        args.steps_per_epoch,
        args.collect_per_step,
        args.test_envs,
        args.batch_size,
        train_fn=adjust_learning_rate,
        test_fn=evaluate_policy,
        update_per_step=args.updates_per_collect,
        logger=logger,
        test_in_train=False,
        reward_metric=lambda rewards: rewards[:, 0]
    )

    # Plot and save training results
    plot_results(rewards, losses)

    return result


def plot_results(rewards, losses, log_dir='data'):
    os.makedirs(log_dir, exist_ok=True)

    # Plot rewards
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Rewards')
    plt.xlabel('Epoch')
    plt.ylabel('Average Reward')
    plt.title('Training Rewards')
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'rewards_plot.png'))
    plt.show()

    # Plot losses
    plt.figure(figsize=(12, 6))
    plt.plot(losses, label='Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'losses_plot.png'))
    plt.show()


def execute_kaz(args):
    if args.observe_only:
        # For watching, use a single environment with rendering
        env = DummyVectorEnv([lambda: PettingZooEnv(knights_archers_zombies_v10.env(
            max_arrows=1000,
            killable_knights=False,
            killable_archers=False,
            max_zombies=4,
            render_mode="human"
        ))])
        policy, _, _, _ = initialize_agents_and_envs(args, env)
        policy.eval()
        for agent in policy.policies.values():
            # Epsilon-Greedy Exploration Strategy: Set epsilon to a low value for testing
            agent.set_eps(args.test_eps)
        collector = Collector(policy, env, exploration_noise=True)
        result = collector.collect(n_episode=1, render=args.render_freq)

        # Ensure pygame handles events so the window stays responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        print(f"Reward: {result['rews'][:, 0].mean()}, Length: {result['lens'].mean()}")
        time.sleep(2)
    else:
        # Parse arguments, initialize environments and agents, and start training
        args, envs = parse_arguments_and_initialize_envs()
        policy, optimizers, agents, schedulers = initialize_agents_and_envs(args, envs)
        logger, collector = setup_training_and_logger(args, policy, envs)
        result = execute_training_loop(args, policy, optimizers, schedulers, logger, collector)
        pprint.pprint(result)
        # Check if the best reward meets the desired win rate
        assert result["best_reward"] >= args.win_rate


if __name__ == "__main__":
    args, _ = parse_arguments_and_initialize_envs()
    execute_kaz(args)
