import Environment.ModelEnv as ModelEnv 
import Agent.Agent as Agent 
import gymnasium as gym


env=ModelEnv.Cityenv()
model=Agent.create_agent(env)

model.learn(total_timesteps=100000)
model.save("ppo_drone")