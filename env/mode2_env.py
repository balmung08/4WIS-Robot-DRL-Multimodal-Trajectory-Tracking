import random
import math
import time
from typing import Dict, List, Tuple, Optional
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from gym import spaces
import gym

from utils.Trajectory_Generater import (TrajectoryGenerator, SteeringMode, RobotConfig)
from utils.Trajectory_Transfer import trajectory_transfer, util_flatten


class SimulationEnv(gym.Env):
    """机器人轨迹跟踪仿真环境 - 统一版本（支持可选的序列输出）"""

    # 物理参数常量
    MAX_V = 2.0
    MAX_THETA = math.pi / 9
    MAX_A = 5.0
    MAX_DELTA = math.pi
    TIME_STEP = 0.05

    # === 运动学参数（用于控制计算） ===
    WHEELBASE = 0.64  # 轴距 L (前后轮中心距)
    TRACK_WIDTH = 0.47  # 轮距 W (左右轮中心距)

    # === 可视化参数（用于渲染显示） ===
    BODY_LENGTH = 0.97  # 车身长度（可视化）
    BODY_WIDTH = 0.67  # 车身宽度（可视化）
    WHEEL_LENGTH = 0.30  # 轮子长度（可视化）
    WHEEL_WIDTH = 0.10  # 轮子宽度（可视化）

    # 训练参数
    DEFAULT_PREDICTION_HORIZON = 10  # 前视轨迹点数
    TRAJECTORY_SEGMENTS = 4  # 轨迹段数

    # 奖励权重
    STEERING_RATE_WEIGHT = 0.002  # 转向平滑系数
    ACCEL_WEIGHT = 0.002  # 加速平滑系数
    DEVIATION_PENALTY = 100.0  # 偏移惩罚
    MAX_DEVIATION = 2.0  # 偏移判断阈值
    # 相较于论文，距离奖励与平滑奖励比例不变，整体大小缩小10倍

    def __init__(self,
                 prediction_horizon: int = DEFAULT_PREDICTION_HORIZON,
                 random_seed: bool = True,
                 use_sequence_output: bool = False,
                 obs_history_length: int = 10):
        """
        初始化仿真环境

        Args:
            prediction_horizon: 前视步长，默认为10步
            random_seed: 是否使用随机种子
            use_sequence_output: 是否启用序列输出（适用于LSTM等序列模型）
            obs_history_length: 观测历史序列长度，仅在use_sequence_output=True时生效
        """
        super().__init__()

        self.prediction_horizon = prediction_horizon
        self.use_sequence_output = use_sequence_output
        self.obs_history_length = max(1, obs_history_length) if use_sequence_output else 1

        # 创建机器人配置（使用运动学参数）
        self.robot_config = RobotConfig(
            wheelbase=self.WHEELBASE,
            track_width=self.TRACK_WIDTH,
            ref_velocity=1.0,
            time_step=self.TIME_STEP
        )

        # 初始化轨迹生成器
        self.trajectory_generator = TrajectoryGenerator(self.robot_config)

        # 生成初始参考轨迹
        self.ref_trajectory = self._generate_reference_trajectory()

        # 是否随机种子
        self.random_seed = random_seed

        # 初始化状态变量
        self.traj_index = 0
        self.state_steering = 0.0
        self.velocity = 1.0
        self.episode = 0

        # 定义动作空间和观测空间
        self._setup_spaces()

        # 计算初始观测
        initial_obs = self._compute_state([0, 0, 0])

        # 初始化观测历史队列（仅在序列模式下使用）
        if self.use_sequence_output:
            self.obs_history = deque(maxlen=self.obs_history_length)
            # 用初始观测填充历史队列
            for _ in range(self.obs_history_length):
                self.obs_history.append(initial_obs.copy())

        # 当前状态
        self.state = initial_obs

        # 渲染相关
        self.fig = None
        self.ax = None
        self.render_initialized = False
        plt.ion()

    def _setup_spaces(self):
        """设置动作空间和观测空间"""
        # 动作空间: 每步2个动作（转向变化率 + 加速度）
        action_dim = 2 * self.prediction_horizon
        self.action_space = spaces.Box(
            low=np.array([-1.0] * action_dim),
            high=np.array([1.0] * action_dim),
            dtype=np.float32
        )

        # 单次观测维度
        self.single_obs_dim = 2 * self.prediction_horizon + 2

        if self.use_sequence_output:
            # ================= 序列模式：观测空间 (history_length, obs_dim) =================
            obs_low = np.full(
                (self.obs_history_length, self.single_obs_dim),
                -np.inf,
                dtype=np.float32
            )
            obs_high = np.full(
                (self.obs_history_length, self.single_obs_dim),
                np.inf,
                dtype=np.float32
            )

            # 所有历史步的速度和转向角限制
            obs_low[:, -2] = -self.MAX_V  # 所有行的倒数第2列
            obs_low[:, -1] = -self.MAX_THETA  # 所有行的倒数第1列
            obs_high[:, -2] = self.MAX_V
            obs_high[:, -1] = self.MAX_THETA

            self.observation_space = spaces.Box(
                low=obs_low,
                high=obs_high,
                dtype=np.float32
            )
        else:
            # ================= 非序列模式：观测空间 (obs_dim,) =================
            obs_low = np.full(self.single_obs_dim, -np.inf, dtype=np.float32)
            obs_high = np.full(self.single_obs_dim, np.inf, dtype=np.float32)

            # 速度和转向角限制
            obs_low[-2] = -self.MAX_V
            obs_low[-1] = -self.MAX_THETA
            obs_high[-2] = self.MAX_V
            obs_high[-1] = self.MAX_THETA

            self.observation_space = spaces.Box(
                low=obs_low,
                high=obs_high,
                dtype=np.float32
            )

    def _generate_reference_trajectory(self) -> List:
        """生成参考轨迹"""
        trajectory = self.trajectory_generator.generate_single_mode(
            SteeringMode.LATERAL,
            self.TRAJECTORY_SEGMENTS
        )
        return trajectory

    def _compute_state(self, state_robot: List[float]) -> np.ndarray:
        """
        计算当前观测状态
        Args: state_robot: [x, y, theta]
        Returns: 扁平化的状态向量
        """
        # 获取前视轨迹点
        end_idx = min(
            self.traj_index + self.prediction_horizon,
            len(self.ref_trajectory)
        )
        lookahead_traj = self.ref_trajectory[self.traj_index:end_idx]

        # 转换到机器人坐标系
        state_traj = trajectory_transfer(lookahead_traj, state_robot)

        # 填充到指定长度
        state_traj = self._pad_trajectory(state_traj)

        # 提取x,y坐标
        simplified_traj = [[point[0], point[1]] for point in state_traj]

        # 组合状态: 前视点坐标 + 速度 + 转向角
        state = simplified_traj + [self.velocity, self.state_steering]
        flattened = util_flatten(state)

        # 验证维度
        assert len(flattened) == self.single_obs_dim, \
            f"State dimension mismatch! Expected {self.single_obs_dim}, got {len(flattened)}"

        return np.array(flattened, dtype=np.float32)

    def _pad_trajectory(self, trajectory: List) -> List:
        """填充或截断轨迹到指定长度"""
        if len(trajectory) >= self.prediction_horizon:
            return trajectory[:self.prediction_horizon]

        # 用最后一个点填充
        padded = trajectory.copy()
        last_point = trajectory[-1] if trajectory else [0, 0, 0, 0]

        while len(padded) < self.prediction_horizon:
            padded.append(last_point)

        return padded

    def reset(self) -> np.ndarray:
        """重置环境并生成新的随机轨迹"""
        if self.random_seed:
            random.seed()

        # 随机化初始参数
        init_angle = random.uniform(-math.pi, math.pi)
        init_x_offset = random.uniform(-0.2, 0.2)
        init_y_offset = random.uniform(-0.2, 0.2)
        init_theta_offset = random.uniform(-math.pi / 12, math.pi / 12)

        # 生成新轨迹 - 使用新的生成器
        temp_generator = TrajectoryGenerator(self.robot_config)
        temp_generator.updater.reset(0, 0, init_angle)
        self.ref_trajectory = temp_generator.generate_single_mode(
            SteeringMode.LATERAL,
            self.TRAJECTORY_SEGMENTS
        )

        # 重置轨迹生成器状态
        self.trajectory_generator.updater.reset(
            init_x_offset,
            init_y_offset,
            init_angle + init_theta_offset
        )

        # 重置状态变量
        self.traj_index = 0
        self.state_steering = 0.0
        self.velocity = 1.0

        # 计算初始状态
        self.state = self._compute_state([
            init_x_offset,
            init_y_offset,
            init_angle + init_theta_offset
        ])

        # 处理观测历史
        if self.use_sequence_output:
            # 重置观测历史队列（用初始观测填充）
            self.obs_history.clear()
            for _ in range(self.obs_history_length):
                self.obs_history.append(self.state.copy())
            return np.array(self.obs_history, dtype=np.float32)
        else:
            return self.state

        self.episode += 1

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步动作
        Args: action: 动作向量，包含N步的转向变化率和加速度
        Returns:(state, reward, done, info)
        """
        # 保存原始状态
        origin_state = self.trajectory_generator.updater.state
        origin_index = self.traj_index

        # 初始化返回值
        after_state = [origin_state.x, origin_state.y, origin_state.theta]
        after_velocity = self.velocity
        after_steer = self.state_steering
        done = False
        info = {}

        # ==================== 状态迭代部分 ====================
        trajectory_states = []

        for i in range(self.prediction_horizon):
            # 解析动作
            delta_steer = action[2 * i]
            accel = action[2 * i + 1]

            # 更新速度（带限制）
            self.velocity = np.clip(
                self.velocity + accel * self.TIME_STEP * self.MAX_A,
                -self.MAX_V,
                self.MAX_V
            )

            # 更新转向角（带限制）
            self.state_steering = np.clip(
                self.state_steering + delta_steer * self.MAX_DELTA * self.TIME_STEP,
                -self.MAX_THETA,
                self.MAX_THETA
            )

            # 计算四轮转向角（阿克曼转向，使用轴距和轮距）
            wheel_angles = self.trajectory_generator.kinematics.compute_wheel_angles(
                SteeringMode.LATERAL,
                self.state_steering
            )


            # 计算四轮速度
            wheel_velocities = self.trajectory_generator.kinematics.compute_wheel_velocities(
                SteeringMode.LATERAL,
                self.velocity,
                self.state_steering,
            )

            # 状态更新（使用轴距和轮距进行运动学计算）
            new_state = self.trajectory_generator.updater.update(
                wheel_angles,
                wheel_velocities
            )

            # 保存第一步的状态
            if i == 0:
                after_state = [new_state.x, new_state.y, new_state.theta]
                after_velocity = self.velocity
                after_steer = self.state_steering

            # 更新轨迹索引
            self.traj_index += 1

            # 记录当前步状态
            trajectory_states.append({
                'step': i,
                'robot_state': [new_state.x, new_state.y, new_state.theta],
                'velocity': self.velocity,
                'traj_index': self.traj_index
            })

        # ==================== 奖励计算部分 ====================
        reward = 0.0
        discount_rate = 0.9

        for state_info in trajectory_states:
            i = state_info['step']
            state_robot = state_info['robot_state']
            traj_idx = state_info['traj_index']

            # 计算当前点与参考点的误差
            ref_idx = min(traj_idx - 1, len(self.ref_trajectory) - 1)
            ref_state = np.array(trajectory_transfer(
                [self.ref_trajectory[ref_idx]],
                state_robot
            )[0])

            # 位置偏差惩罚
            distance_error = np.linalg.norm(ref_state[:2]) * (discount_rate ** i)
            reward -= distance_error

            # 控制量惩罚
            delta_steer = action[2 * i]
            accel = action[2 * i + 1]

            # 添加折扣因子
            steering_penalty = self.STEERING_RATE_WEIGHT * abs(delta_steer) * (discount_rate ** i)
            reward -= steering_penalty

            accel_penalty = self.ACCEL_WEIGHT * abs(accel) * (discount_rate ** i)
            reward -= accel_penalty

            # 检查第一步是否偏离过大
            if i == 0 and distance_error > self.MAX_DEVIATION:
                reward -= self.DEVIATION_PENALTY
                done = True

                # 计算详细误差信息
                info["err"] = ref_state
                et = np.abs(ref_state[2])
                info["et"] = et if et <= np.pi else et - 2 * np.pi
        # print(reward)
        # ==================== 状态更新部分 ====================
        # 回滚到第一步后的状态
        self.traj_index = origin_index + 1

        if self.traj_index >= len(self.ref_trajectory):
            self.traj_index = len(self.ref_trajectory) - 1
            done = True

        # 重置生成器到第一步后的状态
        self.trajectory_generator.updater.reset(
            after_state[0],
            after_state[1],
            after_state[2]
        )

        # 更新速度和转向
        self.velocity = after_velocity
        self.state_steering = after_steer

        # 计算新的观测状态
        self.state = self._compute_state(after_state)

        # 返回观测（根据模式不同）
        if self.use_sequence_output:
            # 将新观测添加到历史队列（自动移除最旧的）
            self.obs_history.append(self.state.copy())
            return np.array(self.obs_history, dtype=np.float32), reward, done, info
        else:
            return self.state, reward, done, info

    def get_obs_history(self) -> Optional[np.ndarray]:
        """
        获取观测历史序列（仅在序列模式下可用）
        Returns: shape = (obs_history_length, single_obs_dim) 或 None
        """
        if self.use_sequence_output:
            return np.array(list(self.obs_history))
        else:
            return None

    def render(self, mode: str = "human"):
        """渲染环境可视化（使用可视化参数）"""
        current_state = self.trajectory_generator.updater.state
        robot_x = current_state.x
        robot_y = current_state.y
        robot_theta = current_state.theta

        # 初始化画布
        if not self.render_initialized:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.ax.set_title("SimulationEnv Viewer")
            self.ax.set_xlabel("X (m)")
            self.ax.set_ylabel("Y (m)")
            self.ax.axis("equal")
            self.render_initialized = True

        self.ax.cla()

        # 绘制参考轨迹
        self._draw_trajectory()

        # 绘制车身（使用可视化尺寸）
        self._draw_car_body(robot_x, robot_y, robot_theta)

        # 绘制车轮（使用可视化尺寸和轴距/轮距位置）
        self._draw_wheels(robot_x, robot_y, robot_theta)

        # 绘制朝向箭头
        self._draw_orientation_arrow(robot_x, robot_y, robot_theta)

        # 显示信息
        mode_str = f"Sequence (History: {self.obs_history_length})" if self.use_sequence_output else "Single"
        info_text = (
            f"Mode: {mode_str}  |  v = {self.velocity:.2f} m/s  |  δ = {np.degrees(self.state_steering):.1f}°\n"
            f"idx = {self.traj_index}/{len(self.ref_trajectory)}  |  "
            f"Wheelbase: {self.WHEELBASE}m  |  Track: {self.TRACK_WIDTH}m"
        )
        self.ax.text(
            0.02, 0.98,
            info_text,
            transform=self.ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7)
        )

        # 设置视野跟随
        self.ax.set_xlim(robot_x - 3, robot_x + 3)
        self.ax.set_ylim(robot_y - 3, robot_y + 3)
        self.ax.grid(True, alpha=0.3)

        plt.pause(0.001)

    def _draw_trajectory(self):
        """绘制参考轨迹"""
        if len(self.ref_trajectory) > 1:
            traj = np.array(self.ref_trajectory)
            self.ax.plot(traj[:, 0], traj[:, 1], "b--", linewidth=2, alpha=0.6, label="Reference")

            # 标记当前目标点
            if self.traj_index < len(self.ref_trajectory):
                target = self.ref_trajectory[self.traj_index]
                self.ax.plot(target[0], target[1], "go", markersize=10,
                             label="Target", zorder=5)

    def _draw_car_body(self, x: float, y: float, theta: float):
        """绘制车身（使用可视化尺寸）"""
        body_corners = np.array([
            [self.BODY_LENGTH / 2, self.BODY_WIDTH / 2],
            [self.BODY_LENGTH / 2, -self.BODY_WIDTH / 2],
            [-self.BODY_LENGTH / 2, -self.BODY_WIDTH / 2],
            [-self.BODY_LENGTH / 2, self.BODY_WIDTH / 2]
        ])

        R = self._rotation_matrix(theta)
        body_world = (R @ body_corners.T).T + np.array([x, y])

        self.ax.fill(
            body_world[:, 0], body_world[:, 1],
            color="orange", alpha=0.7, edgecolor='black', linewidth=2
        )
        self.ax.plot(x, y, "ro", markersize=6, label="Robot Center", zorder=10)

    def _draw_wheels(self, x: float, y: float, theta: float):
        """绘制四个车轮（位置基于轴距/轮距，大小使用可视化参数）"""
        R = self._rotation_matrix(theta)

        # 车轮位置基于真实的轴距和轮距
        wheel_offsets = [
            [self.WHEELBASE / 2, self.TRACK_WIDTH / 2],  # FL (前左)
            [self.WHEELBASE / 2, -self.TRACK_WIDTH / 2],  # FR (前右)
            [-self.WHEELBASE / 2, self.TRACK_WIDTH / 2],  # RL (后左)
            [-self.WHEELBASE / 2, -self.TRACK_WIDTH / 2],  # RR (后右)
        ]

        # 阿克曼转向: 前轮转向，后轮对称
        wheel_angles = self.trajectory_generator.kinematics.compute_wheel_angles(
            SteeringMode.LATERAL,
            self.state_steering
        )

        for (ox, oy), delta in zip(wheel_offsets, wheel_angles):
            wheel_pos = (R @ np.array([ox, oy])) + np.array([x, y])
            wheel_polygon = self._wheel_polygon(
                wheel_pos[0], wheel_pos[1], theta + delta
            )
            self.ax.fill(
                wheel_polygon[:, 0], wheel_polygon[:, 1],
                color="black", alpha=0.9, edgecolor='gray', linewidth=0.5
            )

    def _wheel_polygon(self, cx: float, cy: float, yaw: float) -> np.ndarray:
        """生成车轮多边形（使用可视化尺寸）"""
        wheel_corners = np.array([
            [self.WHEEL_LENGTH / 2, self.WHEEL_WIDTH / 2],
            [self.WHEEL_LENGTH / 2, -self.WHEEL_WIDTH / 2],
            [-self.WHEEL_LENGTH / 2, -self.WHEEL_WIDTH / 2],
            [-self.WHEEL_LENGTH / 2, self.WHEEL_WIDTH / 2]
        ])

        R = self._rotation_matrix(yaw)
        return (R @ wheel_corners.T).T + np.array([cx, cy])

    def _draw_orientation_arrow(self, x: float, y: float, theta: float):
        """绘制朝向箭头"""
        arrow_length = 0.5
        self.ax.arrow(
            x, y,
            arrow_length * np.cos(theta),
            arrow_length * np.sin(theta),
            head_width=0.12,
            head_length=0.15,
            fc="red",
            ec="red",
            linewidth=2.5,
            zorder=15
        )

    @staticmethod
    def _rotation_matrix(theta: float) -> np.ndarray:
        """生成2D旋转矩阵"""
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

    def get_state_info(self) -> Dict:
        """获取当前状态信息用于调试"""
        info = {
            'mode': 'sequence' if self.use_sequence_output else 'single',
            'prediction_horizon': self.prediction_horizon,
            'single_obs_dim': self.single_obs_dim,
            'observation_space_shape': self.observation_space.shape,
            'action_space_shape': self.action_space.shape,
            'current_traj_index': self.traj_index,
            'trajectory_length': len(self.ref_trajectory),
            'current_velocity': self.velocity,
            'current_steering': self.state_steering,
            'robot_position': [
                self.trajectory_generator.updater.state.x,
                self.trajectory_generator.updater.state.y,
                self.trajectory_generator.updater.state.theta
            ],
            'kinematics': {
                'wheelbase': self.WHEELBASE,
                'track_width': self.TRACK_WIDTH
            },
            'visualization': {
                'body_length': self.BODY_LENGTH,
                'body_width': self.BODY_WIDTH,
                'wheel_length': self.WHEEL_LENGTH,
                'wheel_width': self.WHEEL_WIDTH
            }
        }

        if self.use_sequence_output:
            info['obs_history_length'] = self.obs_history_length
            info['obs_history_shape'] = self.get_obs_history().shape
        else:
            info['state_shape'] = np.array(self.state).shape

        return info

    def close(self):
        """关闭环境"""
        if self.fig is not None:
            plt.close(self.fig)
        plt.ioff()


if __name__ == "__main__":
    print("=" * 60)
    print("测试 1: 非序列模式 (use_sequence_output=False)")
    print("=" * 60)
    env1 = SimulationEnv(prediction_horizon=10, use_sequence_output=False)
    state1 = env1.reset()
    print(f"Reset state shape: {state1.shape}")  # 应该是 (22,)
    print(f"State info: {env1.get_state_info()}")

    for i in range(5):
        action = env1.action_space.sample()
        state, reward, done, info = env1.step(action)
        print(f"Step {i + 1} state shape: {state.shape}, reward: {reward:.4f}")
        if done:
            break
    env1.close()

    print("\n" + "=" * 60)
    print("测试 2: 序列模式 (use_sequence_output=True, obs_history_length=10)")
    print("=" * 60)
    env2 = SimulationEnv(prediction_horizon=10, use_sequence_output=True, obs_history_length=10)
    state2 = env2.reset()
    print(f"Reset state shape: {state2.shape}")  # 应该是 (10, 22)
    print(f"State info: {env2.get_state_info()}")

    for i in range(5):
        action = env2.action_space.sample()
        state, reward, done, info = env2.step(action)
        print(f"Step {i + 1} state shape: {state.shape}, reward: {reward:.4f}")
        if done:
            break
    env2.close()

    print("\n✓ 测试完成！两种模式都正常工作。")