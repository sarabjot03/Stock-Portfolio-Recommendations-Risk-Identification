import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import pandas as pd
import numpy as np
import joblib
import gcsfs
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from gymnasium import spaces
import gymnasium as gym
import pickle

# GCS paths and configurations
gcs_bucket_name = "stocks-data-bucket"
gcs_file_path = f"gs://{gcs_bucket_name}/Big_Data_Pipeline/Engineered_Features_for_production.csv"
gcs_rf_model_path = f"gs://{gcs_bucket_name}/Big_Data_Pipeline/random_forest_model.pkl"
ppo_model_path = "/Users/sarabjotsingh/Downloads/ppo_stock_portfolio_model.zip"
env_file_path = "/Users/sarabjotsingh/Downloads/reinforcement_learning_env.pkl"

# Function to read CSV from GCS
def read_csv_from_gcs(file_path):
    fs = gcsfs.GCSFileSystem()
    with fs.open(file_path, 'r') as file:
        df = pd.read_csv(file)
    return df

# Function to load a model from GCS
def load_rf_model(model_path):
    fs = gcsfs.GCSFileSystem()
    with fs.open(model_path, 'rb') as model_file:
        return joblib.load(model_file)

# Create lagged features
def create_lagged_features(df, features, lags):
    X = df[features].copy()
    for feature in ['up', 'prev_diff', 'daydiff']:
        for lag in lags:
            X[f'{feature}_lag{lag}'] = X[feature].shift(lag)
    return X.dropna()

# Predict RSI values
def predict_rsi(model, X):
    return model.predict(X)

def update_rsi_column(df, predicted_rsi):
    df = df.iloc[-len(predicted_rsi):].copy()
    df['RSI'] = predicted_rsi
    return df
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
# Load the PPO model
def load_ppo_model(model_path):
    return PPO.load(model_path)

# Generate portfolio recommendations
def get_portfolio_recommendation(model, env):
    state, _ = env.reset()  # Unpack (obs, info)
    done = False
    best_portfolio = None
    best_reward = -np.inf
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
# Plot portfolio performance
def plot_performance(performance):
    steps, rewards = zip(*performance)
    plt.figure(figsize=(10, 6))
    plt.plot(steps, rewards, marker='o', linestyle='-', color='b')
    plt.title('Portfolio Performance Over Time')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.grid()
    plt.show()

def run_pipeline():
    pipeline_options = PipelineOptions(flags=[])  # Use 'flags=[]' to bypass argument parsing
    with beam.Pipeline(options=pipeline_options) as p:

        # Load models and environment
        rf_model = load_rf_model(gcs_rf_model_path)
        ppo_model = load_ppo_model(ppo_model_path)
        
        with open(env_file_path, 'rb') as f:
            env = pickle.load(f)

        # Read data from GCS
        data = p | 'ReadDataFromGCS' >> beam.Create([read_csv_from_gcs(gcs_file_path)])

        # Process data and predict RSI
        processed_data = (
            data
            | 'CreateLaggedFeatures' >> beam.Map(lambda df: create_lagged_features(
                df, ['up', 'prev_diff', 'daydiff', 'low', 'capital gains', 'high', 'prev_close', 'open', 'adj close'], 
                [1, 2, 3]))
            | 'PredictRSI' >> beam.Map(lambda df: update_rsi_column(df, predict_rsi(rf_model, df)))
        )

        # Generate portfolio recommendations
        recommendations = (
            processed_data
            | 'GeneratePortfolioRecommendations' >> beam.Map(lambda df: get_portfolio_recommendation(ppo_model, env))
        )

        # Handle and print recommendations
        def handle_recommendation(recommendation):
            symbols, reward, performance = recommendation

            # Calculate estimated returns
            estimated_prices = env.data[env.data['symbol'].isin(symbols)]['adj close']
            total_investment = env.budget
            estimated_roi = total_investment * (1 + reward)

            # Display portfolio details
            print("========== Recommended Portfolio ==========")
            print(f"Selected Stocks: {', '.join(symbols)}")
            print(f"Portfolio Reward: {reward:.2f}")
            print(f"Estimated Investment: ${total_investment:,.2f}")
            print(f"Estimated Return on Investment (ROI): ${estimated_roi:,.2f}")

            # Plot performance
            plot_performance(performance)

        recommendations | 'HandleRecommendation' >> beam.Map(handle_recommendation)

if __name__ == "__main__":
    run_pipeline()
