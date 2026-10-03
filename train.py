import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from city_drone_env import CityDroneEnv

# 1. Initialize the custom environment in headless mode for faster training
env = CityDroneEnv(render_mode="direct")

# 2. Verify the environment follows Gymnasium standards
check_env(env)

# 3. Set up the DQN Model
# Tensorboard log allows you to track reward graphs in VS Code
model = DQN(
    "MlpPolicy", 
    env, 
    verbose=1, 
    learning_rate=1e-3, 
    buffer_size=50000, 
    exploration_fraction=0.5,
    tensorboard_log="./dqn_drone_tensorboard/"
)

# 4. Train the model
print("Starting training...")
model.learn(total_timesteps=100000, tb_log_name="first_run")

# 5. Save the weights
model.save("dqn_drone_delivery")
print("Model saved successfully.")