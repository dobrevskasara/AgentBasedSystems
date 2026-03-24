import gymnasium as gym
from mdp_blank import value_iteration, policy_iteration
import numpy as np

def test_policy(env, policy, episodes):
    total_steps = 0
    total_reward = 0
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        steps = 0
        while not done:
            action = policy[state]
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        total_steps += steps
    avg_steps = total_steps / episodes
    avg_reward = total_reward / episodes
    return avg_steps, avg_reward

if __name__ == '__main__':
    discount_factors = [0.5, 0.7, 0.9]

    print("Value iteration")
    for df in discount_factors:
        env = gym.make('Taxi-v3')
        policy_matrix, V = value_iteration(env.unwrapped,
                                           env.action_space.n,
                                           env.observation_space.n,
                                           discount_factor=df)
        policy = policy_matrix.argmax(axis=1)
        avg_steps_50, avg_reward_50 = test_policy(env, policy, 50)
        avg_steps_100, avg_reward_100 = test_policy(env, policy, 100)
        print(f"\nDiscount factor: {df}")
        print(f"50 episodes: avg_steps={avg_steps_50:.2f}, avg_reward={avg_reward_50:.2f}")
        print(f"100 episodes: avg_steps={avg_steps_100:.2f}, avg_reward={avg_reward_100:.2f}")

    print("\nPolicy iteration")
    for df in discount_factors:
        env = gym.make('Taxi-v3')
        policy_matrix, V = policy_iteration(env.unwrapped,
                                            env.action_space.n,
                                            env.observation_space.n,
                                            discount_factor=df)
        policy = policy_matrix.argmax(axis=1)
        avg_steps_50, avg_reward_50 = test_policy(env, policy, 50)
        avg_steps_100, avg_reward_100 = test_policy(env, policy, 100)
        print(f"\nDiscount factor: {df}")
        print(f"50 episodes: avg_steps={avg_steps_50:.2f}, avg_reward={avg_reward_50:.2f}")
        print(f"100 episodes: avg_steps={avg_steps_100:.2f}, avg_reward={avg_reward_100:.2f}")


    env = gym.make('Taxi-v3',
                   render_mode='human')

    policy_matrix, V = value_iteration(env.unwrapped,
                                       env.action_space.n,
                                       env.observation_space.n,
                                       discount_factor=0.9)
    policy = policy_matrix.argmax(axis=1)
    state, _ = env.reset()
    done = False
    while not done:
        action = policy[state]
        state, reward, terminated, truncated, _ = env.step(action)
        env.render()
        done = terminated or truncated
    env.close()