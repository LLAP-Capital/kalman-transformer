import numpy as np
import argparse
import sys
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.markowitz import MarkowitzPortfolio

def save_weights(weights, output_folder):
    print(f'Saving weights')
    with open(os.path.join(output_folder, 'weights.pkl'), 'wb') as f:
        pickle.dump(weights, f)

def save_figure(fig, output_folder):
    print(f'Saving Figure, {output_folder}')
    fig.savefig(os.path.join(output_folder, 'portfolio_returns.png'))

def run_portfolio(ticker_symbols, start_date, end_date, reweight_interval, output_folder):
    portfolio = MarkowitzPortfolio(ticker_symbols, start_date, end_date)
    wei = portfolio.calculate_optimal_weights(reweight_interval)
    print(wei)

    # Mock data for demonstration
    weights = np.random.random(len(ticker_symbols))
    fig, ax = plt.subplots()
    ax.plot(np.random.random(100).cumsum(), label='Portfolio Returns')
    ax.legend()

    # Save weights and figure
    save_weights(weights, output_folder)
    save_figure(fig, output_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a Markowitz Portfolio experiment')
    # Adding default values for each argument
    parser.add_argument('--tickers', nargs='+', default=['AAPL', 'MSFT', 'GOOGL'], help='List of ticker symbols for the portfolio. Default: AAPL, MSFT, GOOGL')
    parser.add_argument('--start', type=str, default='2020-01-01', help='Start date for the portfolio in YYYY-MM-DD format. Default: 2020-01-01')
    parser.add_argument('--end', type=str, default='2021-01-01', help='End date for the portfolio in YYYY-MM-DD format. Default: 2021-01-01')
    parser.add_argument('--reweight', type=int, default=30, help='Reweight interval in days. Default: 30')
    parser.add_argument('--output', type=str, default='data/output', help='Output folder for saving results. Default: output')

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    run_portfolio(args.tickers, args.start, args.end, args.reweight, args.output)

