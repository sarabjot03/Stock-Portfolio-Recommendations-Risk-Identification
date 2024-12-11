import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from stable_baselines3 import PPO
import pickle
import gymnasium as gym
from gymnasium import spaces
import datetime as dt
import webbrowser

# Open the Streamlit app in the default browser
webbrowser.open("http://localhost:8501")

# Custom gym environment for stock portfolio
class StockPortfolioEnv(gym.Env):
    def __init__(self, data, budget, risk_tolerance, stock_count=6):
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.budget = budget
        self.risk_tolerance = risk_tolerance
        self.stock_count = stock_count
        self.stock_symbols = self.data['symbol'].unique()
        self.num_stocks = len(self.stock_symbols)
        assert self.num_stocks >= stock_count, "Not enough unique stocks in data."
        self.action_space = spaces.MultiDiscrete([self.num_stocks] * stock_count)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(34,), dtype=np.float32)
        self.current_step = 0

    def reset(self, seed=None, options=None):
        np.random.seed(seed)
        self.current_step = 0
        return self._get_observation(), {}

    def step(self, action):
        selected_stocks = [self.stock_symbols[idx] for idx in action]
        reward, cost = self._compute_reward(selected_stocks)
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        return self._get_observation(), reward, done, False, {"selected_stocks": selected_stocks, "portfolio_cost": cost}

    def _get_observation(self):
        return self.data.iloc[self.current_step, :].drop(['symbol', 'date']).values[:34].astype(np.float32)

    def _compute_reward(self, selected_stocks):
        selected_data = self.data[self.data['symbol'].isin(selected_stocks)]
        selected_data['price_return'] = (selected_data['adj close'] - selected_data['prev_close']) / selected_data['prev_close']
        portfolio_return = selected_data['price_return'].mean() + selected_data['capital gains'].mean()
        selected_data['price_volatility'] = selected_data['adj close'].rolling(14).std()
        portfolio_risk = selected_data['price_volatility'].mean()
        risk_weight = {"low": 2.0, "medium": 1.0, "high": 0.5}[self.risk_tolerance]
        reward = portfolio_return - risk_weight * portfolio_risk
        portfolio_cost = selected_data['adj close'].sum()
        if portfolio_cost > self.budget:
            reward -= 100
        return reward, portfolio_cost

# Function to load pre-trained PPO model
def load_ppo_model(model_path):
    return PPO.load(model_path)

# Function to load the custom environment
def load_custom_env(env_file_path):
    with open(env_file_path, 'rb') as f:
        return pickle.load(f)

# Generate portfolio recommendations
def get_portfolio_recommendation(model, env):
    state, _ = env.reset()  # Unpack (obs, info)
    done = False
    best_portfolio = None
    best_reward = -float('inf')
    performance = []  # To store performance metrics

    while not done:
        action, _ = model.predict(state, deterministic=True)
        next_state, reward, done, _, info = env.step(action)

        performance.append((env.current_step, reward))  # Track performance
        if reward > best_reward:
            best_reward = reward
            best_portfolio = action

        state = next_state

    best_portfolio_symbols = [env.stock_symbols[idx] for idx in best_portfolio]
    return best_portfolio_symbols, best_reward, performance

import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

def plot_performance(performance):
    """
    Plots a trend line graph for portfolio performance.

    Parameters:
    - performance (list of tuples): Each tuple contains (step, reward).

    Returns:
    - Matplotlib plot object to render in Streamlit.
    """
    # Unpack steps and rewards from the performance data
    steps, rewards = zip(*performance)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the trend line
    ax.plot(steps, rewards, marker='o', linestyle='-', color='blue', linewidth=2, label='Portfolio Reward')

    # Customize the plot
    ax.set_title('Portfolio Performance Over Time', fontsize=16, fontweight='bold')
    ax.set_xlabel('Steps', fontsize=14)
    ax.set_ylabel('Reward', fontsize=14)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Ensure integer steps on x-axis
    ax.grid(True, linestyle='--', alpha=0.6)  # Add a grid for clarity

    # Highlight the maximum reward point
    max_reward_index = rewards.index(max(rewards))
    ax.scatter([steps[max_reward_index]], [rewards[max_reward_index]], color='red', zorder=5)
    ax.annotate(
        f"Max Reward: {rewards[max_reward_index]:.2f}",
        xy=(steps[max_reward_index], rewards[max_reward_index]),
        xytext=(steps[max_reward_index] + 0.5, rewards[max_reward_index] + 0.5),
        arrowprops=dict(facecolor='red', arrowstyle='->'),
        fontsize=12,
        color='red'
    )

    # Add a legend
    ax.legend(fontsize=12)

    plt.tight_layout()  # Adjust layout for better appearance
    return fig

# Streamlit UI
def main():
    st.title("Stock Portfolio Recommendation System")
    st.sidebar.header("User Inputs")

    # User inputs
    budget = st.sidebar.number_input("Enter Your Budget ($)", min_value=100, step=50)
    risk_tolerance = st.sidebar.selectbox("Select Risk Tolerance", ["low", "medium", "high"])
    investment_date = st.sidebar.date_input(
        "Select Investment Start Date (December Only)", 
        min_value=dt.date(dt.date.today().year, 12, 1), 
        max_value=dt.date(dt.date.today().year, 12, 31)
    )

    # Display user inputs
    st.sidebar.markdown(f"**Budget:** ${budget:,.2f}")
    st.sidebar.markdown(f"**Risk Tolerance:** {risk_tolerance.capitalize()}")
    st.sidebar.markdown(f"**Investment Start Date:** {investment_date.strftime('%Y-%m-%d')}")

    # Load models and environment
    ppo_model_path = "Models/ppo_stock_portfolio_model.zip"
    env_file_path = "Models/reinforcement_learning_env.pkl"

    env = load_custom_env(env_file_path)
    ppo_model = load_ppo_model(ppo_model_path)

    # Set environment parameters
    env.budget = budget
    env.risk_tolerance = risk_tolerance

    if st.button("Generate Recommendation"):
        st.write("Generating portfolio recommendations...")
        portfolio, reward, performance = get_portfolio_recommendation(ppo_model, env)
        # Display portfolio
        st.success("Portfolio Recommended Successfully!")
        st.write("### Recommended Portfolio")
        st.write(f"**Selected Stocks:** {', '.join(portfolio)}")
        st.write(f"**Portfolio Reward:** {reward:.2f}")
        # Calculate estimated ROI
        estimated_prices = env.data[env.data['symbol'].isin(portfolio)]['adj close']
        total_investment = budget
        estimated_roi = total_investment * (1 + reward)
        st.write(f"**Estimated Return on Investment (ROI):** ${estimated_roi:,.2f}")
        # Plot the trend line graph
        st.write("### Portfolio Performance")
        performance_plot = plot_performance(performance)
        st.pyplot(performance_plot)

if __name__ == "__main__":
    main()
