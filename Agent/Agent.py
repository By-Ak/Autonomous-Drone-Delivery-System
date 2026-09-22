from stable_baselines3 import PPO


def create_agent(env):

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        gamma=0.99,
        n_steps=2048,
        batch_size=64,
        verbose=1
    )

    return model