import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1")

state_size = env.observation_space.n
action_size = env.action_space.n

def q_learning(gamma, alpha, episodes):
    Q = np.zeros((state_size, action_size))

    for _ in range(episodes):
        state, _ = env.reset()
        done = False

        while not done:
            action = env.action_space.sample()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )

            state = next_state

    return Q


def test_policy(Q, episodes):
    total_steps = 0
    total_reward = 0

    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        steps = 0
        reward_sum = 0

        while not done:
            action = np.argmax(Q[state])

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            steps += 1
            reward_sum += reward

        total_steps += steps
        total_reward += reward_sum

    print("Average steps:", total_steps / episodes)
    print("Average reward:", total_reward / episodes)


configs = [(0.5, 0.1), (0.5, 0.01), (0.9, 0.1), (0.9, 0.01)]

for gamma, alpha in configs:
    print("\nGamma:", gamma, "Alpha:", alpha)

    Q = q_learning(gamma, alpha, 5000)

    print("Test 50 episodes")
    test_policy(Q, 50)

    print("Test 100 episodes")
    test_policy(Q, 100)

render_env = gym.make("FrozenLake-v1", render_mode="human")

state, _ = render_env.reset()
done = False

while not done:
    action = np.argmax(Q[state])
    state, reward, terminated, truncated, _ = render_env.step(action)
    done = terminated or truncated

render_env.close()