from pettingzoo.classic import tictactoe_v3
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):

    def __init__(self, state_size, action_size):

        super(DQN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.model(x)


class DQNAgent:

    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=2000)

        self.gamma = 0.95

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.learning_rate = 0.001

        self.model = DQN(state_size, action_size)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )

        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):

        self.memory.append(
            (state, action, reward, next_state, done)
        )

    def select_action(self, state, valid_actions):

        if np.random.rand() < self.epsilon:
            return random.choice(valid_actions)

        state = torch.FloatTensor(state).unsqueeze(0)

        q_values = self.model(state).detach().numpy()[0]

        for i in range(self.action_size):

            if i not in valid_actions:
                q_values[i] = -999999

        return np.argmax(q_values)

    def train(self, batch_size=32):

        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in batch:

            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)

            target = reward

            if not done:

                target += self.gamma * torch.max(
                    self.model(next_state)
                ).item()

            output = self.model(state)

            target_f = output.clone().detach()

            target_f[action] = target

            loss = self.criterion(output, target_f)

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


env = tictactoe_v3.env(render_mode=None)

state_size = 18
action_size = 9

agent = DQNAgent(state_size, action_size)


episodes = 5000

for episode in range(episodes):

    env.reset()

    for player in env.agent_iter():

        observation, reward, terminated, truncated, info = env.last()

        done = terminated or truncated

        state = observation["observation"].flatten()

        action_mask = observation["action_mask"]

        valid_actions = np.where(action_mask == 1)[0]

        if done:

            action = None

        else:

            if player == env.possible_agents[0]:

                action = agent.select_action(
                    state,
                    valid_actions
                )

            else:

                action = random.choice(valid_actions)

        env.step(action)

        if (
            player == env.possible_agents[0]
            and not done
        ):

            next_observation, next_reward, next_terminated, next_truncated, _ = env.last()

            next_state = next_observation["observation"].flatten()

            next_done = next_terminated or next_truncated

            agent.remember(
                state,
                action,
                reward,
                next_state,
                next_done
            )

            agent.train()

    if (episode + 1) % 500 == 0:

        print(f"Episode {episode + 1} completed")


games = 50

dqn_total_reward = 0
random_total_reward = 0

dqn_wins = 0
random_wins = 0

agent.epsilon = 0

for game in range(games):

    env.reset()

    rewards = {
        env.possible_agents[0]: 0,
        env.possible_agents[1]: 0
    }

    for player in env.agent_iter():

        observation, reward, terminated, truncated, info = env.last()

        rewards[player] += reward

        done = terminated or truncated

        state = observation["observation"].flatten()

        action_mask = observation["action_mask"]

        valid_actions = np.where(action_mask == 1)[0]

        if done:

            action = None

        else:

            if player == env.possible_agents[0]:

                action = agent.select_action(
                    state,
                    valid_actions
                )

            else:

                action = random.choice(valid_actions)

        env.step(action)

    dqn_total_reward += rewards[env.possible_agents[0]]

    random_total_reward += rewards[env.possible_agents[1]]

    if rewards[env.possible_agents[0]] > rewards[env.possible_agents[1]]:

        dqn_wins += 1

    elif rewards[env.possible_agents[1]] > rewards[env.possible_agents[0]]:

        random_wins += 1


print("\nRESULTS: ")

print("Average DQN Reward:",
      dqn_total_reward / games)

print("Average Random Reward:",
      random_total_reward / games)

print("DQN Wins:",
      dqn_wins)

print("Random Wins:",
      random_wins)