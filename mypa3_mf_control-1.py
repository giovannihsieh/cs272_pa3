from typing import Any
import random
import gymnasium as gym

def argmax_action(d: dict[Any,float]) -> Any:
    """return a key of the maximum value in a given dictionary 

    Args:
        d (dict[Any,float]): dictionary

    Returns:
        Any: a key
    """
    return max(d, key=d.get)

class ValueRLAgent():
    def __init__(self, env: gym.Env, gamma : float = 0.98, eps: float = 0.2, alpha: float = 0.02, total_epi: int = 5_000) -> None:
        """initialize agent parameters
        This class will be a parent class and not be called directly.

        Args:
            env (gym.Env): gym environment
            gamma (float, optional): a discount factor. Defaults to 0.98.
            eps (float, optional): the epsilon value. Defaults to 0.2. Note: this pa uses a simple eps-greedy not decaying eps-greedy.
            alpha (float, optional): a learning rate. Defaults to 0.02.
            total_epi (int, optional): total number of episodes an agent should learn. Defaults to 5_000.
        """
        self.env = env
        self.q = self.init_qtable(env.observation_space.n, env.action_space.n)
        self.gamma = gamma
        self.eps = eps
        self.alpha = alpha
        self.total_epi = total_epi
    
    def init_qtable(self, n_states: int, n_actions: int, init_val: float = 0.0) -> dict[int,dict[int,float]]:
        """initialize the q table (dictionary indexed by s, a) with a given init_value

        Args:
            n_states (int, optional): the number of states. Defaults to int.
            n_actions (int, optional): the number of actions. Defaults to int.
            init_val (float, optional): all q(s,a) should be set to this value. Defaults to 0.0.

        Returns:
            dict[int,dict[int,float]]: q table (q[s][a] -> q-value)
        """
        q_table = {s: {a: init_val for a in range(n_actions)} for s in range(n_states)}
        return q_table

    def eps_greedy(self, state: int, exploration: bool = True) -> int:
        """epsilon greedy algorithm to return an action

        Args:
            state (int): state
            exploration (bool, optional): explore based on the epsilon value if True; take the greedy action by the current self.q if False. Defaults to True.

        Returns:
            int: action
        """
        actions = list(self.q[state].keys())

        if exploration and random.random() < self.eps:
            return random.choice(actions)
        else:
            return argmax_action(self.q[state])

    def choose_action(self, ss: int) -> int:
        """a helper function to specify a exploration policy
        If you want to use the eps_greedy, call the eps_greedy function in this function and return the action.

        Args:
            ss (int): state

        Returns:
            int: action
        """
        return self.eps_greedy(ss, exploration=True)
    
    def best_run(self, max_steps: int = 100) -> tuple[list[tuple[int,int,float]], bool]:
        """After the learning, an optimal episode (based on the latest self.q) needs to be generated for evaluation. From the initial state, always take the greedily best action until it reaches a goal.

        Args:
            max_steps (int, optional): Terminate the episode generation if the agent cannot reach the goal after max_steps. One step is (s,a,r) Defaults to 100.

        Returns:
            tuple[
                list[tuple[int,int,float]],: An episode [(s1,a1,r1), (s2,a2,r2), ...]
                bool: done - True if the episode reaches a goal, False if it hits max_steps.
            ]
        """
        episode = list()
        done = False

        # reset env for initial state
        state, _ = self.env.reset()

        for _ in range(max_steps):
            # choose best action based on the Q-table using argmax function
            action = argmax_action(self.q[state])
            # take action
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            # record (s, a, r, s')
            episode.append((state,action,reward, next_state))
            # update state
            state = next_state
            # check if episode is done
            if terminated or truncated:
                done = terminated # true if goal
                break

        return (episode, done)

    def calc_return(self, episode: list[tuple[int,int,float]], done=False) -> float:
        """Given an episode, calculate the return value. An episode is in this format: [(s1,a1,r1), (s2,a2,r2), ...].

        Args:
            episode (list[tuple[int,int,float]]): An episode [(s1,a1,r1), (s2,a2,r2), ...]
            done (bool, optional): True if the episode reaches a goal, False if it does not. Defaults to False.

        Returns:
            float: the return value. None if done is False.
        """
        if not done:
            return None
        G = 0.0
        # need to do in reverse for discounting
        for(_, _, r) in reversed(episode):
            G = r + self.gamma * G
        return G

class DoubleQLAgent(ValueRLAgent):  
    def __init__(self, env, gamma = 0.98, eps = 0.2, alpha = 0.02, total_epi = 5000):
        super().__init__(env, gamma, eps, alpha, total_epi)
        # two Q-tables for double q-learning
        self.q1 = self.init_qtable(env.observation_space.n, env.action_space.n)
        self.q2 = self.init_qtable(env.observation_space.n, env.action_space.n)

    def learn(self):
        """Double Q-Learning algorithm
        Update the Q table (self.q) for self.total_epi number of episodes.

        The results should be reflected to its q table.
        Added storing reward per episode for plotting learning curve
        """
        episode_rewards = [] # tracking rewards for plotting learning curve
        for _ in range(self.total_epi):
            state, _ = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                # choose action using epsilon greedy in Q1 + Q2
                q_sum = {a: self.q1[state][a] + self.q2[state][a] for a in self.q1[state]}
                # can't use choose_action because it's Q1 + Q2, so hard code epsilon greedy
                if random.random() < self.eps:
                    action = random.choice(list(self.q1[state].keys()))
                else:
                    action = argmax_action(q_sum)

                # take the action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward # add reward

                # 50 50 of choosing which Q-table to update
                if random.random() < 0.5:
                    # update q1
                    argmax = argmax_action(self.q1[next_state])
                    target = reward + self.gamma * self.q2[next_state][argmax] * (not done)
                    self.q1[state][action] = self.q1[state][action] + self.alpha * (target - self.q1[state][action])
                else:
                    # update q2
                    argmax = argmax_action(self.q2[next_state])
                    target = reward + self.gamma * self.q1[next_state][argmax] * (not done)
                    self.q2[state][action] = self.q2[state][action] + self.alpha * (target - self.q2[state][action])

                state = next_state
            # record reward for the episode
            episode_rewards.append(total_reward)
            # update combined q-table
            for s in self.q:
                for a in self.q[s]:
                    self.q[s][a] = (self.q1[s][a] + self.q2[s][a]) / 2

        return episode_rewards

import matplotlib.pyplot as plt
import numpy as np

env = gym.make('CliffWalking-v0')
agent = DoubleQLAgent(env, gamma = 0.98, eps = 0.2, alpha = 0.02, total_epi = 5000)
rewards = agent.learn()

# calculate average reward every 10 episodes
avg_reward = [np.mean(rewards[i:i+10]) for i in range(0, len(rewards), 10)]

# plot average rewards every 10 episodes
plt.plot(range(0, len(rewards), 10), avg_reward)
plt.xlabel("Episode #")
plt.ylabel("Average reward (per 10 episodes)")
plt.title("Double Q-Learning Agent in Cliff Walking")
plt.show()

# print optimal episode
episode, done = agent.best_run(max_steps = 500)
print("Optimal Episode: ")
for step in episode:
    print(step)

if done:
    total_return = agent.calc_return([(s,a,r) for s,a,r,s_next in episode], done)
    print("Total Return:", total_return)