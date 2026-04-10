import gymnasium as gym
import numpy as np

env = gym.make("Taxi-v3")

state_size = env.observation_space.n
action_size = env.action_space.n


def q_learning(discount_factor, learning_rate, episodes):
    Q = np.zeros((state_size, action_size))
    epsilon = 0.1

    for episode in range(episodes):
        state, _ = env.reset()

        done = False
        steps = 0

        while not done:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            Q[state, action] = Q[state, action] + learning_rate * (
                reward + discount_factor * np.max(Q[new_state]) - Q[state, action]
            )

            state = new_state
            steps += 1

            if steps > 200:
                break

        if episode % 500 == 0:
            print(f"Episode {episode}")

    return Q


def test_policy(Q, episodes):
    total_steps = 0
    total_rewards = 0

    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        steps = 0
        rewards = 0

        while not done:
            action = np.argmax(Q[state])

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            steps += 1
            rewards += reward

        total_steps += steps
        total_rewards += rewards

    print(f"\nTest for {episodes} episodes:")
    print("Average steps:", total_steps / episodes)
    print("Average reward:", total_rewards / episodes)


configs = [
    (0.5, 0.1),
    (0.9, 0.1)
]

for gamma, alpha in configs:
    print(f"\nTraining Taxi with gamma={gamma}, alpha={alpha}")
    Q = q_learning(gamma, alpha, 5000)

    test_policy(Q, 50)
    test_policy(Q, 100)