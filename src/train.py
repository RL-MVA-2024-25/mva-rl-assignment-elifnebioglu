import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
from xgboost import XGBRegressor
import joblib
import random
import os


def xg_boost_regressor_tags():
    """Creates sklearn tags for XGBRegressor."""
    return {"estimator_type": "regressor", "_xfail_checks": False}


XGBRegressor.__sklearn_tags__ = xg_boost_regressor_tags()


class ProjectAgent:
    """
    A reinforcement learning agent using XGBoost for Q-value approximation.
    """

    def __init__(self, config):
        self.state_dimensions = config['state_dims']
        self.possible_actions = config['action_nums']
        self.discount_factor = config.get('discount', 0.99)
        self.exploration_rate = config.get('exploration', 0.1)
        self.learning_coefficient = config.get('learning_rate', 0.3)
        self.depth_of_trees = config.get('tree_depth', 17)
        self.num_trees = config.get('num_trees', 100)
        self.episode_counter = 0

        self.q_approximators = [
            XGBRegressor(
                learning_rate=self.learning_coefficient,
                max_depth=self.depth_of_trees,
                n_estimators=self.num_trees,
                objective="reg:squarederror",
                n_jobs=-1
            )
            for _ in range(self.possible_actions)
        ]
        self.training_repository = {
            a: {"states": [], "targets": []} for a in range(self.possible_actions)
        }

    def select_action(self, state, explore=False):
        if explore or random.random() < self.exploration_rate:
            return random.randint(0, self.possible_actions - 1)
        
        q_estimates = [self.estimate_q(state, a) for a in range(self.possible_actions)]
        return int(np.argmax(q_estimates))

    def train(self, environment, num_training_episodes=1000):
        for episode_num in range(num_training_episodes):
            current_state = environment.reset()[0]
            cumulative_reward = 0
            
            for _ in range(environment._max_episode_steps):
                if random.random() < self.exploration_rate or self.episode_counter < 10:
                    selected_action = self.select_action(current_state, explore=True)
                else :
                    selected_action = self.select_action(current_state)

                next_state, reward_obtained, _, _, _ = environment.step(selected_action)
                self.update_training_set(current_state, selected_action, reward_obtained, next_state)
                current_state = next_state
                cumulative_reward += reward_obtained

            self.episode_counter += 1
            self.refit_q_approximators()
            print(f"Training Episode: {episode_num + 1}/{num_training_episodes}, "
                  f"Total Reward: {cumulative_reward}")

    def update_training_set(self, state, action, reward, next_state):
        future_q_values = [self.estimate_q(next_state, a) for a in range(self.possible_actions)]
        optimal_future_q = max(future_q_values)
        q_target = reward + self.discount_factor * optimal_future_q
        self.training_repository[action]["states"].append(state)
        self.training_repository[action]["targets"].append(q_target)

    def refit_q_approximators(self):
        for action_index in range(self.possible_actions):
            data_subset = self.training_repository[action_index]
            if len(data_subset["states"]) > 0:
                state_array = np.array(data_subset["states"])
                target_array = np.array(data_subset["targets"])
                self.q_approximators[action_index].fit(state_array, target_array)

    def estimate_q(self, state, action):
        reshaped_state = np.array(state).reshape(1, -1)
        try:
            return self.q_approximators[action].predict(reshaped_state)[0]
        except Exception:
            return 0.0

    def save_agent(self, directory):
        joblib.dump(self.q_approximators, f"{directory}/agent_models.joblib")

    def load(self, directory):
        self.q_approximators = joblib.load("agent_models.joblib")
        print(f"Agent loaded from {directory}")


def train_and_save_model(env, total_episodes):
    agent_config = {
        'state_dims': env.observation_space.shape[0],
        'action_nums': env.action_space.n,
        'discount': 0.99,
        'exploration': 0.1,
        'learning_rate': 0.3,
        'tree_depth': 10,
        'num_trees': 100
    }
    agent = ProjectAgent(agent_config)
    agent.train(env, num_training_episodes=total_episodes)
    agent.save_agent("./")



if __name__ == "__main__":
    
    # Define the environment
    env = TimeLimit(
        env=HIVPatient(domain_randomization=False), max_episode_steps=200
    )

    train_and_save_model(env, 400)
