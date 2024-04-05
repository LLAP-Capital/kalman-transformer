
import numpy as np
import pandas as pd
from scipy.optimize import minimize

class MarkowitzPortfolio:
    def __init__(self, ticker_symbols, start_date, end_date):
        self.ticker_symbols = ticker_symbols
        self.start_date = start_date
        self.end_date = end_date
        self.returns = self.get_historical_returns()
        self.optimal_weights = None

    def get_historical_returns(self):
        dates = pd.date_range(self.start_date, self.end_date)
        return pd.DataFrame(np.random.randn(len(dates), len(self.ticker_symbols)), index=dates, columns=self.ticker_symbols)

    def calculate_optimal_weights(self, reweight_interval):
        mean_returns = self.returns.mean()
        cov_matrix = self.returns.cov()

        num_assets = len(self.ticker_symbols)

        # Function to minimize (negative Sharpe ratio)
        def objective(weights):
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            # Assuming risk-free rate is 0 for simplicity
            sharpe_ratio = portfolio_return / portfolio_volatility
            return -sharpe_ratio

        # Constraints: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

        # Bounds for each weight
        bounds = tuple((0, 1) for asset in range(num_assets))

        # Initial guess (equal distribution)
        initial_guess = num_assets * [1. / num_assets,]

        # Minimize the objective function
        result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

        self.optimal_weights = result.x

    def get_optimal_weights(self):
        return self.optimal_weights if self.optimal_weights is not None else "Optimal weights not calculated yet."
