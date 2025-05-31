import flappy_bird_gymnasium     
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback


train_env      = gym.make("FlappyBird-v0")                # no render
eval_env = gym.make("FlappyBird-v0", render_mode="human")  # you’ll see it play

eval_cb = EvalCallback(
    eval_env,
    best_model_save_path="./logs/best_model/",
    log_path="./logs/results/",
    eval_freq=10_000,
    n_eval_episodes=5,
    deterministic=True,
)

model = PPO(
    "MlpPolicy", train_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    verbose=1,
    tensorboard_log="./logs/tb/",
)
model.learn(total_timesteps=1_000_000, callback=eval_cb)

model.save("/mnt/c/Users/julia/OneDrive/Desktop/flappy_bird/"
           "flappy_bird_project_farmer/model/ppo_flappy")

play_env = gym.make("FlappyBird-v0", render_mode = 'human')
obs, _ = play_env.reset()
for _ in range(1_000):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _, _ = play_env.step(action)
    play_env.render()
    if done:
        obs, _ = play_env.reset()
play_env.close()
