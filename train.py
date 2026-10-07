import argparse
import os
from collections import Counter, deque

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize

from city_env import CityDroneEnv, RewardConfig


GAMMA = RewardConfig().gamma   
TRAIN_POOL = (0, 1000)        
EVAL_POOL = (10000, 10100)     


def lr_schedule(lr_start, lr_end):
    return lambda progress: lr_end + (lr_start - lr_end) * progress


def make_env(pool=TRAIN_POOL, difficulty=1.0, mix=False):
    def _init():
        return Monitor(CityDroneEnv(layout_seed=None, layout_pool=pool, regen_every=1,
                                    difficulty=difficulty, difficulty_mix=mix))
    return _init


class CurriculumCallback(BaseCallback):
    def __init__(self, start=0.0, step=0.1, advance_at=0.5, window=48, level_steps=200_000):
        super().__init__()
        self.d, self.step_size = start, step
        self.advance_at, self.level_steps = advance_at, level_steps
        self.window = deque(maxlen=window)      
        self.level_start = 0

    def _on_training_start(self):
        self.level_start = self.num_timesteps
        self.training_env.env_method("set_difficulty", self.d)

    def _advance(self, why):
        self.d = min(1.0, round(self.d + self.step_size, 2))
        self.training_env.env_method("set_difficulty", self.d)
        print(f"[curriculum] {why} -> difficulty {self.d:.1f} (step {self.num_timesteps})")
        self.window.clear()
        self.level_start = self.num_timesteps

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done and "event" in info and info.get("difficulty", 0.0) >= self.d - 1e-6:
                self.window.append(info["deliveries"] / max(1, info["n_deliveries"]))
        if self.d < 1.0:
            if len(self.window) == self.window.maxlen and np.mean(self.window) >= self.advance_at:
                self._advance(f"completion {np.mean(self.window):.0%}")
            elif self.num_timesteps - self.level_start >= self.level_steps:
                self._advance("time limit for this level")
        return True

    def _on_rollout_end(self):
        self.logger.record("curriculum/difficulty", self.d)
        if self.window:
            self.logger.record("curriculum/completion", float(np.mean(self.window)))


class OutcomeCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.counts = Counter()
        self.deliveries = []
        self.charges = []

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done and "event" in info:
                self.counts[info["event"]] += 1
                self.deliveries.append(info["deliveries"])
                self.charges.append(info["charges"])
        return True

    def _on_rollout_end(self):
        total = sum(self.counts.values())
        if total:
            for k in ("success", "crash", "battery_dead", "out_of_bounds", "stalled", "timeout"):
                self.logger.record(f"outcome/{k}", self.counts[k] / total)
            self.logger.record("outcome/avg_deliveries",
                               sum(self.deliveries) / len(self.deliveries))
            self.logger.record("outcome/avg_charges", sum(self.charges) / len(self.charges))
        self.counts.clear()
        self.deliveries.clear()
        self.charges.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=5_000_000)
    ap.add_argument("--lr", type=float, default=3e-4, help="starting learning rate")
    ap.add_argument("--lr-final", type=float, default=1e-4,
                    help="learning-rate floor; set equal to --lr for a constant rate")
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--no-curriculum", action="store_true",
                    help="train at full difficulty from the start")
    ap.add_argument("--start-difficulty", type=float, default=0.0)
    ap.add_argument("--level-steps", type=int, default=200_000,
                    help="max timesteps per difficulty level before it advances anyway")
    ap.add_argument("--advance-at", type=float, default=0.5,
                    help="advance when this fraction of deliveries is completed")
    args = ap.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    VecCls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    start_d = 1.0 if args.no_curriculum else args.start_difficulty
    venv = VecCls([make_env(TRAIN_POOL, start_d, mix=not args.no_curriculum) for _ in range(args.n_envs)])
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=GAMMA)

    if args.resume:
        model = PPO.load(args.resume, env=venv, tensorboard_log="logs",
                         custom_objects={"learning_rate": lr_schedule(args.lr, args.lr_final)})
    else:
        model = PPO(
            "MlpPolicy", venv,
            learning_rate=lr_schedule(args.lr, args.lr_final),
            n_steps=1024,               
            batch_size=256,
            n_epochs=10,
            gamma=GAMMA,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.003,            
            target_kl=0.03,              
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                               log_std_init=-0.5),  
            tensorboard_log="logs",
            seed=args.seed,
            verbose=1,
        )

    eval_env = VecNormalize(DummyVecEnv([make_env(EVAL_POOL)]),
                            training=False, norm_obs=False, norm_reward=False)
    callbacks = [
        EvalCallback(eval_env, n_eval_episodes=20, deterministic=True,
                     eval_freq=max(100_000 // args.n_envs, 1),
                     best_model_save_path="models/best", log_path="logs/eval"),
        CheckpointCallback(save_freq=max(200_000 // args.n_envs, 1),
                           save_path="models", name_prefix="ppo_drone"),
        OutcomeCallback(),
    ]
    if not args.no_curriculum:
        callbacks.append(CurriculumCallback(start=start_d, advance_at=args.advance_at,
                                            level_steps=args.level_steps))
    model.learn(total_timesteps=args.timesteps, callback=callbacks,
                reset_num_timesteps=not args.resume)
    model.save("models/ppo_drone_final")
    venv.save("models/vecnormalize.pkl")
    venv.close()
    print("Saved models/ppo_drone_final.zip")


if __name__ == "__main__":
    main()
