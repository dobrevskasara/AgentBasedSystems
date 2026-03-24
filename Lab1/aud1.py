import random
import gymnasium as gym
from mdp_blank import value_iteration

if __name__ == '__main__':
    env = gym.make('FrozenLake-v1',
                   is_slippery=False,
                   map_name='8x8',
                   render_mode='human')
    env.reset()

    policy, value = value_iteration(env,
                                    env.action_space.n,
                                    env.observation_space.n,
                                    discount_factor=0.9)

    while True:
        env.step(random.randint(0,3))
        env.render()

    # env.unwrapped.P
    # value_iteration(env,)

print()