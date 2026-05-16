import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

import numpy as np
import random
import cv2

from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:

    def __init__(self, capacity=100000):

        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):

        self.buffer.append(
            (state, action, reward, next_state, done)
        )

    def sample(self, batch_size):

        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):

        return len(self.buffer)

class OUNoise:

    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):

        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.size = size

        self.state = np.ones(self.size) * self.mu

    def reset(self):

        self.state = np.ones(self.size) * self.mu

    def sample(self):

        dx = self.theta * (self.mu - self.state)
        dx += self.sigma * np.random.randn(self.size)

        self.state += dx

        return self.state

class Actor(nn.Module):

    def __init__(self, state_dim, action_dim):

        super().__init__()

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, state):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        return torch.tanh(self.fc3(x)) * 2


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):

        super().__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):

        x = torch.cat([state, action], dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)


class DQN(nn.Module):

    def __init__(self, action_dim):

        super().__init__()

        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))

        return self.fc2(x)

class DuelingDQN(nn.Module):

    def __init__(self, action_dim):

        super().__init__()

        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        self.fc_value = nn.Linear(3136, 512)
        self.value = nn.Linear(512, 1)

        self.fc_adv = nn.Linear(3136, 512)
        self.advantage = nn.Linear(512, action_dim)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        value = F.relu(self.fc_value(x))
        value = self.value(value)

        advantage = F.relu(self.fc_adv(x))
        advantage = self.advantage(advantage)

        return value + (advantage - advantage.mean())


def preprocess(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    resized = cv2.resize(gray, (84, 84))

    return resized / 255.0

def stack_frames(stacked_frames, frame):

    processed = preprocess(frame)

    stacked_frames.append(processed)

    if len(stacked_frames) < 4:

        while len(stacked_frames) < 4:
            stacked_frames.append(processed)

    return np.stack(stacked_frames, axis=0)


def train_actor_critic(episodes=100):

    env = gym.make("Pendulum-v1")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = Actor(state_dim, action_dim).to(device)
    critic = Critic(state_dim, action_dim).to(device)

    target_actor = Actor(state_dim, action_dim).to(device)
    target_critic = Critic(state_dim, action_dim).to(device)

    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

    replay_buffer = ReplayBuffer()

    noise = OUNoise(action_dim)

    gamma = 0.99
    tau = 0.005
    batch_size = 64

    for episode in range(episodes):

        state, _ = env.reset()

        noise.reset()

        done = False

        total_reward = 0

        while not done:

            state_tensor = torch.FloatTensor(
                state
            ).unsqueeze(0).to(device)

            action = actor(
                state_tensor
            ).detach().cpu().numpy()[0]

            action += noise.sample()

            action = np.clip(action, -2, 2)

            next_state, reward, terminated, truncated, _ = \
                env.step(action)

            done = terminated or truncated

            replay_buffer.push(
                state,
                action,
                reward,
                next_state,
                done
            )

            state = next_state

            total_reward += reward

            if len(replay_buffer) > batch_size:

                states, actions, rewards, next_states, dones = \
                    replay_buffer.sample(batch_size)

                states = torch.FloatTensor(states).to(device)
                actions = torch.FloatTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

                with torch.no_grad():

                    next_actions = target_actor(next_states)

                    target_q = target_critic(
                        next_states,
                        next_actions
                    )

                    y = rewards + gamma * target_q * (1 - dones)

                current_q = critic(states, actions)

                critic_loss = F.mse_loss(current_q, y)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                actor_loss = -critic(
                    states,
                    actor(states)
                ).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                for target_param, param in zip(
                    target_actor.parameters(),
                    actor.parameters()
                ):

                    target_param.data.copy_(
                        tau * param.data +
                        (1 - tau) * target_param.data
                    )

                for target_param, param in zip(
                    target_critic.parameters(),
                    critic.parameters()
                ):

                    target_param.data.copy_(
                        tau * param.data +
                        (1 - tau) * target_param.data
                    )

        print(
            f"[ACTOR-CRITIC] "
            f"Episode {episode+1} "
            f"Reward: {total_reward:.2f}"
        )

    env.close()

    return actor


def test_actor_critic(model, episodes=50):

    env = gym.make(
        "Pendulum-v1",
        render_mode="human"
    )

    rewards = []

    for episode in range(episodes):

        state, _ = env.reset()

        done = False

        total_reward = 0

        while not done:

            state_tensor = torch.FloatTensor(
                state
            ).unsqueeze(0).to(device)

            with torch.no_grad():

                action = model(
                    state_tensor
                ).cpu().numpy()[0]

            next_state, reward, terminated, truncated, _ = \
                env.step(action)

            done = terminated or truncated

            state = next_state

            total_reward += reward

        rewards.append(total_reward)

        print(
            f"[TEST] Episode {episode+1} "
            f"Reward: {total_reward:.2f}"
        )

    print(
        f"\nAverage Reward ({episodes} episodes): "
        f"{np.mean(rewards):.2f}"
    )

    env.close()

def train_dqn(agent_type="dqn", episodes=50):

    env = gym.make("ALE/MsPacman-v5")

    action_dim = env.action_space.n

    if agent_type == "dueling":

        model = DuelingDQN(action_dim).to(device)
        target_model = DuelingDQN(action_dim).to(device)

    else:

        model = DQN(action_dim).to(device)
        target_model = DQN(action_dim).to(device)

    target_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    replay_buffer = ReplayBuffer()

    gamma = 0.99

    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.995

    batch_size = 32

    target_update = 10

    for episode in range(episodes):

        state, _ = env.reset()

        stacked_frames = deque(maxlen=4)

        state = stack_frames(stacked_frames, state)

        done = False

        total_reward = 0

        while not done:

            if random.random() < epsilon:

                action = env.action_space.sample()

            else:

                state_tensor = torch.FloatTensor(
                    state
                ).unsqueeze(0).to(device)

                with torch.no_grad():

                    q_values = model(state_tensor)

                action = torch.argmax(q_values).item()

            next_state, reward, terminated, truncated, _ = \
                env.step(action)

            done = terminated or truncated

            next_state = stack_frames(
                stacked_frames,
                next_state
            )

            replay_buffer.push(
                state,
                action,
                reward,
                next_state,
                done
            )

            state = next_state

            total_reward += reward


            if len(replay_buffer) > batch_size:

                states, actions, rewards, next_states, dones = \
                    replay_buffer.sample(batch_size)

                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

                current_q = model(states).gather(1, actions)


                if agent_type == "double":

                    next_actions = model(next_states)\
                        .argmax(1)\
                        .unsqueeze(1)

                    next_q = target_model(next_states)\
                        .gather(1, next_actions)

                else:

                    next_q = target_model(next_states)\
                        .max(1)[0]\
                        .unsqueeze(1)

                target_q = rewards + gamma * next_q * (1 - dones)

                loss = F.mse_loss(current_q, target_q)

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

        epsilon = max(
            epsilon_min,
            epsilon * epsilon_decay
        )

        if episode % target_update == 0:

            target_model.load_state_dict(
                model.state_dict()
            )

        print(
            f"[{agent_type.upper()}] "
            f"Episode {episode+1} "
            f"Reward: {total_reward}"
        )

    env.close()

    return model


def test_dqn(model, episodes=50):

    env = gym.make(
        "ALE/MsPacman-v5",
        render_mode="human"
    )

    rewards = []

    for episode in range(episodes):

        state, _ = env.reset()

        stacked_frames = deque(maxlen=4)

        state = stack_frames(stacked_frames, state)

        done = False

        total_reward = 0

        while not done:

            state_tensor = torch.FloatTensor(
                state
            ).unsqueeze(0).to(device)

            with torch.no_grad():

                q_values = model(state_tensor)

            action = torch.argmax(q_values).item()

            next_state, reward, terminated, truncated, _ = \
                env.step(action)

            done = terminated or truncated

            next_state = stack_frames(
                stacked_frames,
                next_state
            )

            state = next_state

            total_reward += reward

        rewards.append(total_reward)

        print(
            f"[TEST] Episode {episode+1} "
            f"Reward: {total_reward}"
        )

    print(
        f"\nAverage Reward ({episodes} episodes): "
        f"{np.mean(rewards):.2f}"
    )

    env.close()


if __name__ == "__main__":

    print("\n==============================")
    print("TASK 1 - ACTOR CRITIC")
    print("==============================\n")

    actor_model = train_actor_critic(episodes=100)

    print("\n===== TEST 50 =====")
    test_actor_critic(actor_model, episodes=50)

    print("\n===== TEST 100 =====")
    test_actor_critic(actor_model, episodes=100)

    print("\n==============================")
    print("TASK 2 - DQN")
    print("==============================\n")

    dqn_model = train_dqn(
        agent_type="dqn",
        episodes=50
    )

    print("\n===== DQN TEST 50 =====")
    test_dqn(dqn_model, episodes=50)

    print("\n===== DQN TEST 100 =====")
    test_dqn(dqn_model, episodes=100)

    print("\n==============================")
    print("TASK 2 - DOUBLE DQN")
    print("==============================\n")

    double_model = train_dqn(
        agent_type="double",
        episodes=50
    )

    print("\n===== DOUBLE DQN TEST 50 =====")
    test_dqn(double_model, episodes=50)

    print("\n===== DOUBLE DQN TEST 100 =====")
    test_dqn(double_model, episodes=100)


    print("\n==============================")
    print("TASK 2 - DUELING DQN")
    print("==============================\n")

    dueling_model = train_dqn(
        agent_type="dueling",
        episodes=50
    )

    print("\n===== DUELING DQN TEST 50 =====")
    test_dqn(dueling_model, episodes=50)

    print("\n===== DUELING DQN TEST 100 =====")
    test_dqn(dueling_model, episodes=100)

