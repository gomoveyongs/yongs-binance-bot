import numpy as np
import pandas as pd
from gym import Env
from gym.spaces import Discrete, Box

class TradingEnvironment(Env):
    def __init__(self, data):
        super(TradingEnvironment, self).__init__()
        self.data = data
        self.action_space = Discrete(3)  # [0: hold, 1: long, 2: short]
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.current_step = 0
        self.total_reward = 0
        self.positions = []
        return self.data[self.current_step]

    def step(self, action):
        self.current_step += 1
        if self.current_step >= len(self.data):
            return self.data[-1], 0, True, {}

        current_price = self.data[self.current_step][3]  # Close price
        previous_price = self.data[self.current_step - 1][3]  # Previous close price

        reward = 0
        if action == 1:  # long
            reward = np.log(current_price / previous_price)
            self.positions.append((self.current_step, 'long'))
        elif action == 2:  # short
            reward = np.log(previous_price / current_price)
            self.positions.append((self.current_step, 'short'))

        self.total_reward += reward

        done = self.current_step == len(self.data) - 1
        next_state = self.data[self.current_step]

        return next_state, reward, done, {}

    def render(self):
        pass
