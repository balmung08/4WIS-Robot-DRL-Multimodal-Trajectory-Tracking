import os
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from env.mode1_env import SimulationEnv
from stable_baselines3.common.logger import configure

def main():
    env = SimulationEnv()

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 128, 64], qf=[256, 128, 64])
    )

    model = SAC(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=1e-4,
        buffer_size=50000,
        batch_size=128,
        train_freq=2,
        gradient_steps=1,
        learning_starts=1000,
        tensorboard_log="../sac_tensorboard/"
    )
    new_logger = configure("../sac_tensorboard/", ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    log_dir = "../checkpoints/DNN_best_models"
    os.makedirs(log_dir, exist_ok=True)

    callback = EvalCallback(
        env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=1000,
        deterministic=True,
        render=False
    )

    # 手动初始化 callback，注入 model
    callback.init_callback(model)
    callback.on_training_start(locals(), globals())

    obs = env.reset()
    total_steps = 1000000

    for step in range(total_steps):
        # ===== 1) 动作选择 =====
        if step < model.learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action, _ = model.predict(obs, deterministic=False)

        # ===== 2) 环境交互 =====
        next_obs, reward, done, info = env.step(action)

        # ===== 3) 存入 replay buffer =====
        model.replay_buffer.add(obs, next_obs, action, reward, done, infos=[info])

        obs = next_obs
        if done:
            obs = env.reset()

        # ===== 4) 参数更新 =====
        if step >= model.learning_starts and step % model.train_freq.frequency == 0:
            model.train(gradient_steps=model.gradient_steps, batch_size=model.batch_size)

        # ===== 5) 回调推进（Eval / 保存 best）=====
        model.num_timesteps = step + 1
        callback.on_step()

    callback.on_training_end()
    print("\nTraining Finished! Model Saved.")


if __name__ == '__main__':
    main()
