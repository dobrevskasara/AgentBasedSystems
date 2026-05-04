import gymnasium as gym
import numpy as np
import random
from collections import deque

from keras import Model
from keras.layers import Dense, Input, Lambda
from keras.optimizers import Adam
import keras.backend as K


class DQNAgent:
    def __init__(self, state_size, action_size, dueling=False, double=False):
        self.state_size = int(state_size)
        self.action_size = int(action_size)

        self.memory = deque(maxlen=5000)

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.learning_rate = 0.001
        self.batch_size = 64

        self.dueling = dueling
        self.double = double

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target()

    def build_model(self):
        input_layer = Input(shape=(self.state_size,))
        x = Dense(64, activation='relu')(input_layer)
        x = Dense(64, activation='relu')(x)

        if self.dueling:
            value = Dense(1)(x)
            advantage = Dense(self.action_size)(x)

            output = Lambda(
                lambda x: x[0] + (x[1] - K.mean(x[1], axis=1, keepdims=True))
            )([value, advantage])
        else:
            output = Dense(self.action_size)(x)

        model = Model(inputs=input_layer, outputs=output)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states = []
        targets = []

        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state, verbose=0)

            if done:
                target[0][action] = reward
            else:
                if self.double:
                    best_action = np.argmax(self.model.predict(next_state, verbose=0)[0])
                    t = self.target_model.predict(next_state, verbose=0)[0][best_action]
                else:
                    t = np.max(self.target_model.predict(next_state, verbose=0)[0])

                target[0][action] = reward + self.gamma * t

            states.append(state.reshape(-1))
            targets.append(target[0])

        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train(env_name, agent_type, episodes=500):
    env = gym.make(env_name)

    state_size = int(env.observation_space.shape[0])
    action_size = int(env.action_space.n)

    dueling = (agent_type == "dueling")
    double = (agent_type == "double")

    agent = DQNAgent(state_size, action_size, dueling=dueling, double=double)

    scores = []

    for e in range(episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])

        total_reward = 0

        for _ in range(1000):
            action = agent.act(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state = np.reshape(next_state, [1, state_size])

            agent.remember(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if len(agent.memory) > agent.batch_size:
                agent.replay()

            if done:
                break

        if e % 10 == 0:
            agent.update_target()

        scores.append(total_reward)

        print(f"Episode {e}/{episodes} | Reward: {total_reward} | Epsilon: {agent.epsilon:.3f}")

    return agent, scores


def test(env_name, agent, episodes=50, render=False):
    env = gym.make(env_name)
    state_size = int(env.observation_space.shape[0])

    total_rewards = []

    for _ in range(episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])

        done = False
        total = 0

        while not done:
            if render:
                env.render()

            action = np.argmax(agent.model.predict(state, verbose=0)[0])

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = np.reshape(state, [1, state_size])
            total += reward

        total_rewards.append(total)

    avg_reward = np.mean(total_rewards)
    print(f"\nAverage reward over {episodes} episodes: {avg_reward}")

    return avg_reward


if __name__ == "__main__":

    environments = ["CartPole-v1", "MountainCar-v0"]
    agent_types = ["dqn", "double", "dueling"]

    for env_name in environments:
        print(f"\n===== ENVIRONMENT: {env_name} =====")

        for agent_type in agent_types:
            print(f"\n--- TRAINING {agent_type.upper()} ---")

            episodes = 300 if env_name == "CartPole-v1" else 800

            agent, _ = train(env_name, agent_type, episodes=episodes)

            avg50 = test(env_name, agent, episodes=50)
            avg100 = test(env_name, agent, episodes=100)

            print(f"{agent_type.upper()} | 50 avg: {avg50}")
            print(f"{agent_type.upper()} | 100 avg: {avg100}")

            test(env_name, agent, episodes=2, render=True)

