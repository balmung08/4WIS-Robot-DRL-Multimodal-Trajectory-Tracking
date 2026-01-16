import argparse
import time
import numpy as np
import torch
from stable_baselines3 import SAC
from env.mode1_env import SimulationEnv


def run(model_path, episodes=10, deterministic=True, render=True):

    print(f"\nLoading model from: {model_path}")


    env = SimulationEnv(prediction_horizon=10,random_seed=False,use_sequence_output=True,obs_history_length=10)

    model = SAC.load(model_path, env=env)

    for ep in range(episodes):
        env.trajectory_generator.set_seed(ep)
        obs = env.reset()
        done = False
        ep_reward = 0

        print(f"\n===== Episode {ep+1} =====")

        while not done:

            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            ep_reward += reward

            if render:
                env.render()
                time.sleep(0.02)

        print(f"Episode {ep+1} Reward: {ep_reward:.2f}")

    env.close()
    print("\nTesting Finished.")


def main():
    parser = argparse.ArgumentParser()
    # 环境和模型得配套改
    parser.add_argument("--model", type=str, default="../checkpoints/mode-1/LSTM_best_models/best_model")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", default=True)
    args = parser.parse_args()

    run(
        model_path=args.model,
        episodes=args.episodes,
        deterministic=True,
        render=args.render
    )


if __name__ == "__main__":
    main()
