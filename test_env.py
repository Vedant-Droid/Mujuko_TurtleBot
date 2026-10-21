from nav_env import TurtleBotNavEnv

env = TurtleBotNavEnv()
obs, info = env.reset(seed=42)

print("Environment initialized")
print("Observation shape:", obs.shape)
print("Action shape:", env.action_space.shape)
print("Goal:", info["goal"])

for i in range(10):
    obs, reward, terminated, truncated, info = env.step(
        env.action_space.sample()
    )
    print(f"Step {i+1}: reward={reward:.3f}, distance={info['distance']:.2f}")
    if terminated or truncated:
        obs, info = env.reset()

env.close()
print("Environment test passed")
