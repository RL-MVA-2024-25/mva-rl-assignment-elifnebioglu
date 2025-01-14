from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import joblib
import numpy as np

import sklearn as sk

from sklearn.exceptions import NotFittedError


from xgboost import XGBRegressor
import xgboost as xgb

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    
    def __init__(self, state_dim = 6, action_dim = 4, gamma=0.99, learning_rate=0.3, max_depth=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.eps = 0.1
        self.episode = 0
        # Initialize an XGBoost model for each action
        self.models = [
            XGBRegressor(
                learning_rate=learning_rate,
                max_depth=max_depth,
                n_estimators=100,
                objective="reg:squarederror",
                n_jobs=-1
            )
            for _ in range(self.action_dim)
        ]
        # Training data for each action
        self.training_data = {action: {"X": [], "y": []} for action in range(self.action_dim)}
        

    def act(self, observation, use_random=False):
        if use_random:
            return np.random.randint(self.action_dim)  # Random action for exploration

        # Estimate Q-values for all actions
        q_values = [self._predict_q_value(observation, action) for action in range(self.action_dim)]
        return int(np.argmax(q_values))


    def _predict_q_value(self, state, action):
        # Predict Q-value using the XGBoost model for the given action
        state = np.array(state).reshape(1, -1)
        try:
            return self.models[action].predict(state)[0]
        except NotFittedError:
            # If the model is not fitted, return a default Q-value (e.g., 0.0)
            return 0.0
        
    def save(self, path):
        pass

    def load(self):
        self.models = joblib.load("agent_models.joblib")
        print("Models loaded successfully, count:", len(self.models))


