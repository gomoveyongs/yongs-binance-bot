import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rl_environment import TradingEnvironment
from rl_agent import DQNAgent
from rl_utils import preprocess_data, plot_trading_results

# Load and preprocess data
data = pd.read_csv('my_eth_data_bi.csv')
data = preprocess_data(data)

# Define hyperparameters
state_size = 10
action_size = 3  # [0: hold, 1: long, 2: short]
episodes = 5
batch_size = 64

# Initialize environment and agent
env = TradingEnvironment(data)
agent = DQNAgent(state_size, action_size)

# Training loop
for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(min(len(data) - 1, 500)):  # Limit the number of steps per episode
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f"Episode {e+1}/{episodes}, Total Reward: {env.total_reward}")
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    if agent.epsilon > agent.epsilon_min:  # Reduce exploration rate faster
        agent.epsilon *= agent.epsilon_decay * 1.05

# Save the trained model
agent.save("dqn_model.h5")

# Plot the results
plot_trading_results(env)
