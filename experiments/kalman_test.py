
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.kalman import PortfolioKalmanFilter
import numpy as np

def main():
    # Initial state: estimated returns of two assets
    initial_state = np.array([0.05, 0.03])  # 5% and 3% returns
    initial_covariance = np.array([[0.001, 0.0008], [0.0008, 0.002]]) # Initial covariance: initial guesses for the covariances between asset returns
    transition_matrix = np.eye(2) # Transition matrix: assuming returns are primarily random, we use the identity matrix

    observation_matrix = np.eye(2) # Observation matrix: observed returns directly measure the underlying state
    process_noise = np.array([[0.0001, 0], [0, 0.0001]]) # Process noise: covariance of the process noise
    observation_noise = np.array([[0.0005, 0], [0, 0.0005]]) # Observation noise: covariance of the observation noise

    # Initialize the Kalman Filter
    kf = PortfolioKalmanFilter(initial_state, initial_covariance, transition_matrix, observation_matrix, process_noise, observation_noise)
    new_observations = np.array([0.06, 0.02])  # 6% and 2% returns
    updated_state, updated_covariance = kf.update(new_observations) # Update the Kalman Filter with the new observations

    print("Updated State (Est. Returns):", updated_state)
    print("Updated Covariance Matrix:", updated_covariance)


if __name__=='__main__':
    print(os.listdir(os.path.dirname(os.path.realpath(__file__)))) # Print the contents of the current directory
    main()
