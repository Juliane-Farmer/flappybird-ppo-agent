import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

env = gym.make("FlappyBird-v0")     
eval_env = gym.make("FlappyBird-v0")

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/best_model/",
    log_path="./logs/results/",
    eval_freq=10_000,
    n_eval_episodes=5,
    deterministic=True,)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    verbose=1,
    tensorboard_log="./logs/tb/",
)
model.learn(total_timesteps=1_000_000, callback=eval_callback)

model.save("ppo_flappy")

obs, _ = env.reset()
for _ in range(1_000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    env.render()
    if done:
        obs, _ = env.reset()
env.close()
