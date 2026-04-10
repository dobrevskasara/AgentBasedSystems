import gymnasium as gym
import numpy as np

env = gym.make("MountainCar-v0")

bins = [20, 20]

low = env.observation_space.low
high = env.observation_space.high


def discretize(state):
    ratios = (state - low) / (high - low)
    state_discrete = (ratios * np.array(bins)).astype(int)
    state_discrete = np.clip(state_discrete, 0, np.array(bins) - 1)
    return tuple(state_discrete)


Q = np.zeros((bins[0], bins[1], env.action_space.n))

epsilon = 0.1
gamma = 0.95
alpha = 0.1
episodes = 5000


for episode in range(episodes):
    state, _ = env.reset()
    state = discretize(state)

    done = False
    steps = 0

    while not done:
        # ε-greedy
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        new_state = discretize(new_state)

        Q[state][action] += alpha * (
            reward + gamma * np.max(Q[new_state]) - Q[state][action]
        )

        state = new_state
        steps += 1

        if steps > 200:   # safety break
            break


def test(episodes):
    total_steps = 0
    total_rewards = 0

    for _ in range(episodes):
        state, _ = env.reset()
        state = discretize(state)

        done = False
        steps = 0
        rewards = 0

        while not done:
            action = np.argmax(Q[state])

            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = discretize(new_state)

            steps += 1
            rewards += reward

            if steps > 200:
                break

        total_steps += steps
        total_rewards += rewards

    print(f"\nTest {episodes} episodes:")
    print("Average steps:", total_steps / episodes)
    print("Average reward:", total_rewards / episodes)


test(50)
test(100)