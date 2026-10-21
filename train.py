from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from nav_env import TurtleBotNavEnv

env = Monitor(TurtleBotNavEnv())

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=256,
    batch_size=64,
    gamma=0.99,
    verbose=1,
    device="cpu",
)

try:
    model.learn(total_timesteps=100_000)
finally:
    model.save("ppo_turtlebot_mujoco")
    env.close()

print("Saved ppo_turtlebot_mujoco.zip")
