from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import joblib
import numpy as np
import xgboost as xgb
import joblib

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:

    def act(self, observation, use_random=False):
        q_values = [self._q_values(observation, action) for action in range(4)]
        return int(np.argmax(q_values))


    def _q_values(self, state, action):
        state = np.array(state).reshape(1, -1)
        return self.models[action].predict(state)[0]
        
    def save(self, path):
        pass

    def load(self):
        self.models = joblib.load("agent_models.joblib")
