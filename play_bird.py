import argparse
import gymnasium as gym
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

    env = gym.make("FlappyBird-v0")
    obs, _ = env.reset()

    model = PPO.load(args.model_path, env=env)

    done = False
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        env.render()
        if done:
            obs, _ = env.reset()

if __name__ == "__main__":
    main()
