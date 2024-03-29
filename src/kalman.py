

import numpy as np

class PortfolioKalmanFilter:
    def __init__(self, initial_state, initial_covariance, transition_matrix, observation_matrix, process_noise, observation_noise):
        """
        :param initial_state: Initial state vector of asset returns
        :param initial_covariance: Initial covariance matrix of the state
        :param transition_matrix: State transition matrix (F)
        :param observation_matrix: Observation matrix (H)
        :param process_noise: Process noise covariance matrix (Q)
        :param observation_noise: Observation noise covariance matrix (R)
        """
        self.state = initial_state
        self.covariance = initial_covariance
        self.transition_matrix = transition_matrix
        self.observation_matrix = observation_matrix
        self.process_noise = process_noise
        self.observation_noise = observation_noise

    def update(self, observation):
        """
        Update the filter with a new observation.

        :param observation: New observation vector of asset returns
        :return: Updated state estimate and covariance matrix
        """
        # Prediction step
        predicted_state = self.transition_matrix @ self.state
        predicted_covariance = self.transition_matrix @ self.covariance @ self.transition_matrix.T + self.process_noise
        
        # Observation update
        innovation = observation - self.observation_matrix @ predicted_state
        innovation_covariance = self.observation_matrix @ predicted_covariance @ self.observation_matrix.T + self.observation_noise
        kalman_gain = predicted_covariance @ self.observation_matrix.T @ np.linalg.inv(innovation_covariance)
        
        # Update step
        self.state = predicted_state + kalman_gain @ innovation
        self.covariance = predicted_covariance - kalman_gain @ self.observation_matrix @ predicted_covariance
        
        return self.state, self.covariance

    def get_current_state(self):
        """
        Get the current state estimate and covariance matrix.
        
        :return: Current state estimate and covariance matrix
        """
        return self.state, self.covariance




