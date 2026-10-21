import time
from stable_baselines3 import PPO
from nav_env import TurtleBotNavEnv

env = TurtleBotNavEnv(render_mode="human")
model = PPO.load("ppo_turtlebot_mujoco")

try:
    for episode in range(10):
        obs, info = env.reset()
        total_reward = 0.0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                result = (
                    "SUCCESS" if info["success"] else
                    "COLLISION" if info["collision"] else "TIMEOUT"
                )
                print(
                    f"Episode {episode+1}: {result}, "
                    f"reward={total_reward:.1f}, "
                    f"distance={info['distance']:.2f} m"
                )
                time.sleep(1)
                break
finally:
    env.close()
