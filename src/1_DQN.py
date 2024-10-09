import random
import tempfile

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, RecordVideo

from util.gymnastics import DEVICE, init_random, plot_scores, show_gym_video_recording, epsilon_gen

from collections import deque
from typing import NamedTuple

tempfile.tempdir = "/home/markyi/DRL/drl-zh/video"


def gym_simulate(agent=None):
    """
    Runs an Atari pong game with our agent passed as input.
    """
    sim_env = gym.make(
        "PongDeterministic-v4", render_mode="rgb_array_list", frameskip=1
    )
    sim_env = init_random(sim_env)
    # 为Atari添加预处理器。这可以进行黑白转换和其他方便的操作
    sim_env = AtariPreprocessing(sim_env)
    sim_env = RecordVideo(sim_env, tempfile.tempdir, lambda i: i == 0)

    init_position, _ = sim_env.reset()
    first_observation, _, _, _, _ = sim_env.step(1)  # starts the game :)
    state = np.stack([init_position, first_observation])  # 状态，即时间/速度

    for _ in range(2_500):
        action = (
            agent.act(state) if agent is not None else sim_env.action_space.sample()
        )  # 选择一个动作

        # 通过动作在环境中逐步过渡
        # 接收下一次观察、奖励，以及事件是否已终止或截断
        observation, reward, terminated, truncated, info = sim_env.step(action)

        if terminated or truncated:
            observation, info = sim_env.reset()
        # 用新的叠加观测值（最后一个和新的）更新状态
        state = np.stack([state[1], observation])


# 利用神经网络逼近最优动作价值函数Q

class QNetwork(nn.Module):
    """
    Actor(Policy) Model
    """

    def __init__(self, action_size):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(2, 4, kernel_size=6, stride=2)  # 2x84x84 to 4x40x40
        self.conv2 = nn.Conv2d(4, 16, kernel_size=8, stride=4)  # 4x40x40 to 16x9x9
        self.lsize = 16 * 9 * 9

        self.fc1 = nn.Linear(self.lsize, 256)
        self.fc2 = nn.Linear(256, action_size)

    def forward(self, x):
        """Build a network that maps state -> action values."""
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)

        x = x.view(-1, self.lsize)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x


# 经验回放数组
class Experience(NamedTuple):
    """An Experience stored in the replay buffer."""

    state: np.array
    action: int
    reward: float
    next_state: np.array
    done: bool


class ReplayBuffer:
    """The replay buffer for DQN."""

    def __init__(self, buffer_size=int(1e4)):
        self.memory = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size: int = 32):
        all_indices = np.arange(len(self.memory))
        selection = np.random.choice(all_indices, size=batch_size)
        return self.unpack(selection)

    def unpack(self, selection):
        experiences = [e for i in selection if (e := self.memory[i]) is not None]
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.from_numpy(np.stack(states)).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack(actions)).long().to(DEVICE)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(DEVICE)
        next_states = torch.from_numpy(np.stack(next_states)).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack(dones, dtype=np.uint8)).float().to(DEVICE)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


# 开始训练
def start_episode(env: gym.Env):
    """Method to call to start a new episode for pong in DQN training."""
    state, _ = env.reset()
    obs, _, _, _, _ = env.step(1) # Starts the game :)
    return np.stack([state, obs])


def train(env, agent, max_timesteps=int(1e6)) -> list[int]:
    scores = []
    score = 0.0
    n_episode = 1
    # 行为策略，控制智能体与环境交互的策略。用来收集经验数据
    # 目标策略，强化学习的目的是得到一个策略函数，用这个策略函数来控制智能体，这个策略叫目标策略，是一个确定性的策略。
    # Q学习算法用任意的行为策略收集(state, action, reward, next_state)这样的四元组，然后拿它们训练目标策略，即DQN
    eps_gen = epsilon_gen(0.1, 0.995, 0.01)
    epsilon = next(eps_gen)
    state = start_episode(env)

    for t in range(max_timesteps):
        action = agent.act(state, epsilon)
        observation, reward, terminated, truncated, _ = env.step(action)
        next_state = np.stack([state[1], observation])
        done = terminated or truncated
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            scores.append(score)
            avg = np.mean(scores[-25:])
            print(f'\rEpisode {n_episode}\tScore: {score:.2f}\tT={t:6} (avg={avg:.2f})',
                  end="\n" if n_episode % 25 == 0 else "")
            score = 0.0
            n_episode += 1
            epsilon = next(eps_gen)
            state = start_episode(env)

    agent.checkpoint()
    np.savetxt("dqn_scores.csv", np.asarray(scores, dtype=np.int16), delimiter=",")
    return scores

# 定义智能体
class Agent:
    """Agent that interacts with and learns from the environment."""

    def __init__(self, action_size=6, gamma: float = 0.99, tau: float = 1e-3, lr: float = 1e-4,
                 batch_size: int = 32, learn_every: int = 4, update_target_every: int = 2,
                 preload_file: str = None):
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.batch_size = batch_size
        self.learn_every = learn_every
        self.update_target_every = update_target_every
        self.t_learn_step = 0
        self.t_update_target_step = 0

        self.memory = ReplayBuffer()
        self.qnetwork_local = QNetwork(action_size).to(DEVICE)
        self.qnetwork_target = QNetwork(action_size).to(DEVICE)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        self.qnetwork_target.eval()
        self.optimizer = optim.RMSprop(self.qnetwork_local.parameters(), lr=self.lr)

        if preload_file is not None:
            print(f'Loading pre-trained model: {preload_file}')
            self.qnetwork_local.load_state_dict(torch.load(preload_file, map_location=DEVICE))

    def step(self, state, action, reward, next_state, done):
        """Tells the agent to make a step: record experience and possibly learn."""
        self.memory.add(state, action, reward, next_state, done)
        self.t_learn_step = (self.t_learn_step + 1) % self.learn_every
        if self.t_learn_step == 0:
            if len(self.memory) > self.batch_size:
                self.learn()
        self.t_update_target_step = (self.t_update_target_step + 1) % self.update_target_every
        if self.t_update_target_step == 0:
            Agent.soft_update_model_params(self.qnetwork_local, self.qnetwork_target, self.tau) 

    def act(self, state: np.array, eps=0.):
        """Makes the agent take an action for the state passed as input."""
        state = torch.from_numpy(state).float().to(DEVICE)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self):
        """Executes one learning step for the agent."""
        experiences = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            target_action_values = self.qnetwork_target(next_states).detach() # (batch_s, action_s)
            max_action_values = target_action_values.amax(1, keepdim=True)  # (batch_size, 1)
            Q_targets = rewards + (self.gamma * max_action_values * (1 - dones)) # (batch_size, 1)

        predictions = self.qnetwork_local(states)
        Q_expected = predictions.gather(1, actions)
        loss = F.huber_loss(Q_targets, Q_expected)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    @staticmethod
    def soft_update_model_params(src: torch.nn.Module, dest: torch.nn.Module, tau=1e-3):
        """Soft updates model parameters (θ_dest = τ * θ_src + (1 - τ) * θ_src)."""
        for dest_param, src_param in zip(dest.parameters(), src.parameters()):
            dest_param.data.copy_(tau * src_param.data + (1.0 - tau) * dest_param.data)

    def checkpoint(self):
        """Save the QNetwork weights in a file."""
        torch.save(self.qnetwork_local.state_dict(), 'dqn_weights.pth')


def pretrained_simulation():
    pretrained_agent = Agent(preload_file='/home/markyi/DRL/drl-zh/dqn_weights.pth')
    pretrained_scores = np.loadtxt(f'/home/markyi/DRL/drl-zh/dqn_scores.csv', delimiter=',').astype(np.int16)
    plot_scores(pretrained_scores)
    return gym_simulate(pretrained_agent)

# with gym.make("PongDeterministic-v4", frameskip=1) as env:
#     env = init_random(env)
#     env = AtariPreprocessing(env)
#     agent = Agent(action_size=env.action_space.n)
#     scores = train(env, agent)

# plot_scores(scores)

pretrained_simulation()