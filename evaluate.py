import argparse
from collections import Counter

import numpy as np
from stable_baselines3 import PPO

from city_env import CityDroneEnv

def ray_name(i):
    if i < 16:
        return f"horizontal, {i * 22.5:.0f} deg"
    if i < 24:
        return f"up-45deg, {(i - 16) * 45:.0f} deg"
    return "straight up" if i == 24 else "straight down"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/best/best_model.zip")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--layout-seed", type=int, default=None,
                    help="fixed city seed; default = random held-out cities (10000-10099)")
    ap.add_argument("--stochastic", action="store_true",
                    help="sample actions instead of deterministic mean actions")
    args = ap.parse_args()

    env = CityDroneEnv(render_mode="human", layout_seed=args.layout_seed,
                       layout_pool=(10000, 10100))      # cities never used in training
    model = PPO.load(args.model)
    outcomes = Counter()

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done, total, slow, reported = False, 0.0, 0, False
        while not done:
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, r, term, trunc, info = env.step(action)
            total += r
            done = term or trunc

            # hover diagnostic: report once if the drone is nearly stationary for 100 steps
            slow = slow + 1 if np.linalg.norm(env.vel) < 0.3 else 0
            if slow == 100 and not reported:
                reported = True
                near = int(np.argmin(env.lidar[:25]))
                goal = "charging station" if env.seeking else "delivery target"
                print(f"  [hover] step {env.steps}: pos={np.round(env.pos, 1)} "
                      f"heading to {goal}, {np.linalg.norm(env._goal_pos() - env.pos):.1f} m away, "
                      f"battery {env.battery:.0f}%, closest obstacle {env.lidar[near]:.1f} m "
                      f"({ray_name(near)})")

        outcomes[info["event"]] += 1
        print(f"Episode {ep + 1} (city {env.current_layout}): {info['event']:<14} "
              f"deliveries={info['deliveries']} battery={info['battery']:.0f}% return={total:.1f}")

    print("\nSummary:", dict(outcomes))
    env.close()


if __name__ == "__main__":
    main()
