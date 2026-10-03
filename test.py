from stable_baselines3 import DQN
from city_drone_env import CityDroneEnv
import time

# Initialize with GUI so we can physically see the simulation
env = CityDroneEnv(render_mode="human")
model = DQN.load("dqn_drone_delivery")

obs, info = env.reset()
done = False

while not done:
    # Model predicts the best action based on the state
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Slow down the loop so it's viewable by human eyes
    time.sleep(1/60)
    
    if terminated or truncated:
        obs, info = env.reset()