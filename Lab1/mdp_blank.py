import numpy as np


def value_iteration(env, num_actions, num_states, theta=0.00001, discount_factor=1.0):
    """
    This section is for Value Iteration Algorithm.

    Arguments:
        env: The OpenAI environment.
        theta: Stop evaluation once value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    V = np.zeros(num_states)

    while True:
        delta = 0
        for s in range(num_states):
            tmp_values = np.zeros(num_actions)
            for a in range(num_actions):
                for prob, next_s, reward, _ in env.P[s][a]:
                    tmp_values[a] += prob * (reward + discount_factor * V[next_s])
            new_v = np.max(tmp_values)
            delta = max(delta, np.abs(new_v - V[s]))
            V[s] = new_v
        if delta < theta:
            break

    policy = np.zeros([num_states, num_actions])
    for s in range(num_states):
        tmp_values = np.zeros(num_actions)
        for a in range(num_actions):
            for prob, next_s, reward, _ in env.P[s][a]:
                tmp_values[a] += prob * (reward + discount_factor * V[next_s])
        new_a = np.argmax(tmp_values)
        policy[s][new_a] = 1

    return policy, V


def policy_iteration(env, num_actions, num_states, theta=0.00001, discount_factor=1.0):
    """
    Implement the Policy Improvement Algorithm here which iteratively evaluates and improves a policy
    until an optimal policy is found.

    Arguments:
        env: The OpenAI environment.
        theta: Stop evaluation once value function change is less than theta for all states.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    policy = np.ones([num_states, num_actions]) / num_actions
    V = np.zeros(num_states)

    while True:
        # Policy Evaluation
        while True:
            delta = 0
            for s in range(num_states):
                v = 0
                for a, action_prob in enumerate(policy[s]):
                    for prob, next_s, reward, _ in env.P[s][a]:
                        v += action_prob * prob * (reward + discount_factor * V[next_s])
                delta = max(delta, abs(v - V[s]))
                V[s] = v
            if delta < theta:
                break

        # Policy Improvement
        stable = True
        for s in range(num_states):
            old_action = np.argmax(policy[s])
            action_values = np.zeros(num_actions)
            for a in range(num_actions):
                for prob, next_s, reward, _ in env.P[s][a]:
                    action_values[a] += prob * (reward + discount_factor * V[next_s])
            best_action = np.argmax(action_values)
            if old_action != best_action:
                stable = False
            policy[s] = np.eye(num_actions)[best_action]
        if stable:
            break

    return policy, V