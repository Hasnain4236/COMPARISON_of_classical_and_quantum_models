import numpy as np
import math

class ClassicalRL:
    def __init__(self, env):
        self.env = env
        # CartPole state: [pos, vel, angle, ang_vel]
        # Discretize into bins
        self.bins = [
            np.linspace(-4.8, 4.8, 10),
            np.linspace(-4, 4, 10),
            np.linspace(-0.418, 0.418, 10),
            np.linspace(-4, 4, 10)
        ]
        self.q_table = np.zeros((11, 11, 11, 11, env.action_space.n))
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01

    def _discretize(self, state):
        indices = []
        for i in range(len(state)):
            # np.digitize returns 1-based index, we want 0-based for array
            # but we have n+1 bins for n separators
            idx = np.digitize(state[i], self.bins[i])
            indices.append(idx)
        return tuple(indices)

    def train(self, episodes=50):
        rewards = []
        for e in range(episodes):
            state, _ = self.env.reset()
            state_adj = self._discretize(state)
            total_reward = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                if np.random.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[state_adj])
                
                next_state, reward, done, truncated, _ = self.env.step(action)
                next_state_adj = self._discretize(next_state)
                
                # Update Q-Table
                best_next_action = np.argmax(self.q_table[next_state_adj])
                td_target = reward + self.discount_factor * self.q_table[next_state_adj][best_next_action]
                self.q_table[state_adj][action] += self.learning_rate * (td_target - self.q_table[state_adj][action])
                
                state_adj = next_state_adj
                total_reward += reward
            
            rewards.append(total_reward)
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
        return rewards
