import argparse
import flappy_bird_gymnasium  
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model (.zip) to load"
    )
    args = p.parse_args()

    raw = gym.make("FlappyBird-v0", render_mode="human")
    unwrapped = raw.unwrapped  
    env = TimeLimit(unwrapped, max_episode_steps=10_000)
    obs, _ = env.reset()

    model = PPO.load(args.model_path, env=env)

    step_count = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        step_count += 1

        env.render()
        if done:
            print(f"Episode done at step {step_count}, info={info}")
            obs, _ = env.reset()
            step_count = 0

if __name__ == "__main__":
    main()
