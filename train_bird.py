import flappy_bird_gymnasium 
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import time

def make_env():
    return gym.make("FlappyBird-v0")

env = DummyVecEnv([make_env for _ in range(4)])
eval_env = gym.make("FlappyBird-v0", render_mode="human")
eval_cb = EvalCallback(
    eval_env,
    best_model_save_path="./logs/best_model/",
    log_path="./logs/results/",
    eval_freq=50_000,
    n_eval_episodes=5,
    deterministic=True,)

model = PPO(
    "MlpPolicy",
    env,
    n_steps=1024,
    batch_size=64,
    learning_rate=3e-4,
    gamma=0.99,
    verbose=1,
    tensorboard_log="./logs/tb/",)

ts = time.strftime("%Y%m%d-%H%M%S")
model.learn(total_timesteps=3_000_000, callback=eval_cb)
model.save(f"./model/ppo_flappy_{ts}")

