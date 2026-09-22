import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import Environment.ModelEnv as ModelEnv
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

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

plt.figure(figsize=(7, 7))

for x, y in env.obstacles_pos:
    plt.scatter(
        y, x,
        marker="s",
        s=500,
        label="Obstacle" if [x, y] == env.obstacles_pos[0] else ""
    )

for x, y in env.charging_station_pos:
    plt.scatter(
        y, x,
        marker="^",
        s=300,
        label="Charging Station" if [x, y] == env.charging_station_pos[0] else ""
    )

rx, ry = env.restaurant_pos
gx, gy = env.goal_pos

plt.scatter(
    ry, rx,
    marker="o",
    s=300,
    label="Restaurant"
)

plt.scatter(
    gy, gx,
    marker="*",
    s=400,
    label="Goal"
)

path_x = [y for x, y in path]
path_y = [x for x, y in path]

plt.plot(
    path_x,
    path_y,
    marker="o",
    linewidth=2,
    label="Drone Path"
)

for i, (x, y) in enumerate(path):
    plt.text(
        y,
        x,
        str(i),
        ha="center",
        va="center"
    )

plt.xticks(range(7))
plt.yticks(range(7))

plt.xlim(-0.5, 6.5)
plt.ylim(6.5, -0.5)

plt.grid(True)

plt.xlabel("Column")
plt.ylabel("Row")

plt.title("PPO Drone Navigation Path")

plt.legend()

plt.show()