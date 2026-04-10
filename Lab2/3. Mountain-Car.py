import gymnasium as gym
import numpy as np
import time

env = gym.make("MountainCar-v0")

bins = [20, 20]

low = env.observation_space.low
high = env.observation_space.high


def discretize(state):
    state = np.array(state)
    ratios = (state - low) / (high - low)
    ratios = np.clip(ratios, 0, 1)
    return tuple((ratios * (np.array(bins) - 1)).astype(int))


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

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        next_state = discretize(next_state)

        Q[state][action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state][action]
        )

        state = next_state
        steps += 1

        if steps > 200:
            break

    if episode % 1000 == 0:
        print("Episode:", episode)



def test(episodes):
    print(episodes, "episodes")

    total_steps = 0
    total_reward = 0

    for _ in range(episodes):
        state, _ = env.reset()
        state = discretize(state)

        done = False
        steps = 0
        reward_sum = 0

        while not done:
            action = np.argmax(Q[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = discretize(next_state)

            steps += 1
            reward_sum += reward

            if steps > 200:
                break

        total_steps += steps
        total_reward += reward_sum

    print("Average steps:", total_steps / episodes)
    print("Average reward:", total_reward / episodes)


test(50)
test(100)

render_env = gym.make("MountainCar-v0", render_mode="human")

state, _ = render_env.reset()
state = discretize(state)

done = False
steps = 0

while not done:
    action = np.argmax(Q[state])

    next_state, reward, terminated, truncated, _ = render_env.step(action)
    done = terminated or truncated

    state = discretize(next_state)

    steps += 1
    time.sleep(0.02)

render_env.close()
