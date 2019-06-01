### MDP Value Iteration and Policy Iteration
### Acknowledgement: start-up codes were adapted with permission from Prof. Emma Brunskill of Stanford University

"""
s3615907 Huirong Huang
s3609499 Chun Shiong Low
"""

import numpy as np
import gym
import time
from lake_envs import *

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

    P: nested dictionary
        From gym.core.Environment
        For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
        tuple of the form (probability, nextstate, reward, terminal) where
            - probability: float
                the probability of transitioning from "state" to "nextstate" with "action"
            - nextstate: int
                denotes the state we transition to (in range [0, nS - 1])
            - reward: int
                either 0 or 1, the reward for transitioning from "state" to
                "nextstate" with "action"
            - terminal: bool
              True when "nextstate" is a terminal state (hole or goal), False otherwise
    nS: int
        number of states in the environment
    nA: int
        number of actions in the environment
    gamma: float
        Discount factor. Number in range [0, 1)
"""

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
    """Evaluate the value function from a given policy.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: np.array[nS]
        The policy to evaluate. Maps states to actions.
    tol: float
        Terminate policy evaluation when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns
    -------
    value_function: np.ndarray[nS]
        The value function of the given policy, where value_function[s] is
        the value of state s
    """

    value_function = np.zeros(nS)

    ############################
    # YOUR IMPLEMENTATION HERE #

    # index of P_property
    probability_index = 0
    next_state_index = 1
    reward_index = 2
    terminal_index = 3

    # loop until the iteration reaches the tolerance
    while True:

        # make a copy of the previous value function
        pre_value_function = np.copy(value_function)

        # instantiate value function
        value_function = np.zeros(nS)

        # instantiate absolute difference between the new value function and old value function of the current state
        abs_diff_array = np.zeros(nS)

        # loop through all states
        for current_state in range(nS):

            # specify the current action of the given state applying given policy
            current_action = policy[current_state]

            # loop through all probabilities of a given state and action in a stochastic environment
            for i in range(len(P[current_state][current_action])):

                P_property = P[current_state][current_action][i]

                probability = P_property[probability_index]
                next_state = P_property[next_state_index]
                reward = P_property[reward_index]

                # V(s) = sum(T(s, π(s), s') * ((R(s, π(s), s') + γ * V(s')))
                value_function[current_state] += probability * (reward + gamma * pre_value_function[next_state])

            # calculate the absolute difference between the new and old value function of the current state
            abs_diff = np.abs(value_function[current_state] - pre_value_function[current_state])
            abs_diff_array[current_state] = abs_diff

        # find out the maximum among the absolute difference array
        max_abs_diff = max(abs_diff_array)

        # check if the maximum of absolute difference reached the tolerance, if yes, stop the iteration
        if max_abs_diff < tol:
            break

    ############################
    return value_function


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    value_from_policy: np.ndarray
        The value calculated from the policy
    policy: np.array
        The previous policy.

    Returns
    -------
    new_policy: np.ndarray[nS]
        An array of integers. Each integer is the optimal action to take
        in that state according to the environment dynamics and the
        given value function.
    """

    new_policy = np.zeros(nS, dtype='int')

    ############################
    # YOUR IMPLEMENTATION HERE #

    # index of P_property
    probability_index = 0
    next_state_index = 1
    reward_index = 2
    terminal_index = 3

    # loop through all states
    for current_state in range(nS):

        # instantiate Q_value_function
        Q_value_function = np.zeros(nA)

        # loop through all actions of a given state
        for current_action in range(nA):

            # loop through all probabilities of a given state and action in a stochastic environment
            for i in range(len(P[current_state][current_action])):

                P_property = P[current_state][current_action][i]

                probability = P_property[probability_index]
                next_state = P_property[next_state_index]
                reward = P_property[reward_index]

                # Q(s, a) = sum(T(s, a, s') * ((R(s, a, s') + γ * V(s')))
                Q_value_function[current_action] += probability * (reward + gamma * value_from_policy[next_state])

        # π*(s) = argmax(Q(s, a))
        new_policy[current_state] = np.argmax(Q_value_function)

    ############################
    return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    tol: float
        tol parameter used in policy_evaluation()
    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    ############################
    # YOUR IMPLEMENTATION HERE #

    # four possible actions for each state: left, down, right, up
    action_count = 4

    # generate random policy using numPy's randint function
    policy = np.random.randint(0, action_count, nS)

    # policy iterations
    while True:

        # get the value function from given policy evaluation
        value_function = policy_evaluation(P, nS, nA, policy, gamma, tol)

        # get improved policy based on the value functions from given policy
        improved_policy = policy_improvement(P, nS, nA, value_function, policy, gamma)

        # check if the policy converged, if yes, stop the iteration
        if np.all(improved_policy == policy):
            break

        # update previous policy
        policy = improved_policy

    ############################
    return value_function, policy

def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    ############################
    # YOUR IMPLEMENTATION HERE #

    # index of P_property
    probability_index = 0
    next_state_index = 1
    reward_index = 2
    terminal_index = 3

    # loop until the iteration reaches the tolerance
    while True:
        # make a copy of the previous value function
        pre_value_function = np.copy(value_function)

        # instantiate value function
        value_function = np.zeros(nS)

        # instantiate absolute difference between the new value function and old value function of the current state
        abs_diff_array = np.zeros(nS)

        # loop through all states
        for current_state in range(nS):

            # specify the current action of the given state applying given policy
            current_action = policy[current_state]

            # loop through all probabilities of a given state and action in a stochastic environment
            for i in range(len(P[current_state][current_action])):
                P_property = P[current_state][current_action][i]

                probability = P_property[probability_index]
                next_state = P_property[next_state_index]
                reward = P_property[reward_index]

                # V(s) = sum(T(s, π(s), s') * ((R(s, π(s), s') + γ * V(s')))
                value_function[current_state] += probability * (reward + gamma * pre_value_function[next_state])

            # calculate the absolute difference between the new and old value function of the current state
            abs_diff = np.abs(value_function[current_state] - pre_value_function[current_state])
            abs_diff_array[current_state] = abs_diff

            # find out the maximum among the absolute difference array
        max_abs_diff = max(abs_diff_array)

        # check if the maximum of absolute difference reached the tolerance, if yes, stop the iteration
        if max_abs_diff < tol:
            break

            # loop through all states
        for current_state in range(nS):

            # instantiate Q_value_function
            Q_value_function = np.zeros(nA)

            # loop through all actions of a given state
            for current_action in range(nA):

                # loop through all probabilities of a given state and action in a stochastic environment
                for i in range(len(P[current_state][current_action])):
                    P_property = P[current_state][current_action][i]

                    probability = P_property[probability_index]
                    next_state = P_property[next_state_index]
                    reward = P_property[reward_index]

                    # Q(s, a) = sum(T(s, a, s') * ((R(s, a, s') + γ * V(s')))
                    Q_value_function[current_action] += probability * (reward + gamma * value_from_policy[next_state])

            # π*(s) = argmax(Q(s, a))
            new_policy[current_state] = np.argmax(Q_value_function)

    ############################
    return value_function, policy

def render_single(env, policy, max_steps=100):
  """
    This function does not need to be modified
    Renders policy once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as
      attributes.
    Policy: np.array of shape [env.nS]
      The action to take at a given state
  """

  episode_reward = 0
  ob = env.reset()
  for t in range(max_steps):
    env.render()
    time.sleep(0.25)
    a = policy[ob]
    ob, rew, done, _ = env.step(a)
    episode_reward += rew
    if done:
      break
  env.render();
  if not done:
    print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
  else:
    print("Episode reward: %f" % episode_reward)


# Edit below to run policy and value iteration on different environments and
# visualize the resulting policies in action!
# You may change the parameters in the functions below
if __name__ == "__main__":

    # comment/uncomment these lines to switch between deterministic/stochastic environments
    # env = gym.make("Deterministic-4x4-FrozenLake-v0")
    env = gym.make("Stochastic-4x4-FrozenLake-v0")

    print("\n" + "-"*25 + "\nBeginning Policy Iteration\n" + "-"*25)

    V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
    render_single(env, p_pi, 100)
    #
    # print("\n" + "-"*25 + "\nBeginning Value Iteration\n" + "-"*25)
    #
    # V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
    # render_single(env, p_vi, 100)