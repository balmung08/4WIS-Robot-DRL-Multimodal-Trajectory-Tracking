import os
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import EvalCallback
from env.mode1_env import SimulationEnv
from stable_baselines3.common.logger import configure

# ============================
# LSTM 特征提取器
# ============================
class TrajLSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    使用 LSTM 处理序列观测
    输入形状: (seq_len=5, feature_dim=12)
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        self.seq_len = observation_space.shape[0]  # 5
        self.input_dim = observation_space.shape[1]  # 12
        self.lstm_hidden = 256

        self.fc1 = nn.Linear(self.input_dim, features_dim)
        self.lstm = nn.LSTM(
            input_size=features_dim,
            hidden_size=self.lstm_hidden,
            batch_first=True
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: [B, seq_len, input_dim]
        x = self.fc1(obs)  # [B, seq_len, features_dim]
        lstm_out, _ = self.lstm(x)  # [B, seq_len, lstm_hidden]
        # 取最后时间步的输出
        features = F.relu(lstm_out[:, -1, :])  # [B, lstm_hidden]
        return features


# ============================
# 主训练函数
# ============================
def main():
    # ===== 配置参数 =====
    total_steps = 1000000
    guided_steps = 100000  # 前一半步数使用指导模型
    log_dir = "../checkpoints/LSTM_best_models/"
    os.makedirs(log_dir, exist_ok=True)

    # ===== 创建环境 =====
    env = SimulationEnv(prediction_horizon=10, use_sequence_output=True, obs_history_length=10)
    guided_env = SimulationEnv(prediction_horizon=10, use_sequence_output=False)

    # ===== 加载指导模型 =====
    print("Loading guided model...")
    guided_model = SAC.load("../checkpoints/DNN_best_models/best_model", env=guided_env)

    # ===== 定义策略网络结构 =====
    policy_kwargs = dict(
        features_extractor_class=TrajLSTMFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 128, 64], qf=[256, 128, 64])
    )

    # ===== 创建 SAC 模型 =====
    model = SAC(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,
        buffer_size=1000000,
        batch_size=128,
        train_freq=2,
        gradient_steps=1,
        verbose=1,
        learning_starts=1000,
        tensorboard_log="../sac_lstm_tensorboard/"
    )
    new_logger = configure("../sac_tensorboard/", ["stdout", "tensorboard"])
    model.set_logger(new_logger)


    # ===== 设置评估回调 =====
    callback = EvalCallback(
        env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=1000,
        deterministic=True,
        render=False,
        verbose=1
    )
    callback.init_callback(model)
    callback.on_training_start(locals(), globals())

    # ===== 训练循环 =====
    print(f"\nStarting training for {total_steps} steps")
    print(f"Using guided model for first {guided_steps} steps ({guided_steps / total_steps * 100:.0f}%)\n")

    obs = env.reset()

    for step in range(total_steps):
        # ===== 1) 动作选择 =====
        if step < model.learning_starts:
            # 随机探索阶段
            action = env.action_space.sample()
        elif step < guided_steps:
            # 前一半步数: 使用指导模型输出动作
            with torch.no_grad():
                action, _ = guided_model.predict(obs[-1], deterministic=True)

        else:
            # 使用训练中的模型
            with torch.no_grad():
                action, _ = model.predict(obs, deterministic=False)

        # ===== 2) 环境交互 =====
        next_obs, reward, done, info = env.step(action)

        # ===== 3) 存入 replay buffer =====
        model.replay_buffer.add(obs, next_obs, action, reward, done, infos=[info])

        obs = next_obs
        if done:
            obs = env.reset()

        # ===== 4) 模型更新 =====
        if step >= model.learning_starts:
            if step % model.train_freq.frequency == 0:
                model.train(gradient_steps=model.gradient_steps,batch_size=model.batch_size)

        # ===== 5) 更新时间步并执行回调 =====
        model.num_timesteps = step + 1
        callback.on_step()

        # ===== 进度打印 =====
        if (step + 1) % 10000 == 0:
            phase = "Guided" if step < guided_steps else "Training"
            print(f"[{phase}] Step {step + 1}/{total_steps} ({(step + 1) / total_steps * 100:.1f}%)")

    # ===== 训练结束 =====
    callback.on_training_end()
    final_model_path = os.path.join(log_dir, "final_model")
    model.save(final_model_path)
    print(f"\nTraining Finished!")
    print(f"Final model saved to: {final_model_path}")
    print(f"Best model saved to: {log_dir}/best_model.zip")


if __name__ == '__main__':
    main()