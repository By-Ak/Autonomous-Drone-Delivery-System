import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import Environment.ModelEnv as ModelEnv
from stable_baselines3 import PPO

env = ModelEnv.Cityenv()

model = PPO.load("ppo_drone")

observation, info = env.reset()

done = False
path = []

while not done:

    path.append(env.drone_pos.copy())

    action, _ = model.predict(observation, deterministic=True)

    observation, reward, terminated, truncated, info = env.step(action)

    print("Position:", env.drone_pos)
    print("Action:", action)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("----------------")

    done = terminated or truncated

path.append(env.drone_pos.copy())

print("\nPath taken:")
print(path)

print("Total steps:", len(path) - 1)

env.close()