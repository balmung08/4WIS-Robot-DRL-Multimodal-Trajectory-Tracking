import math
import random
from enum import IntEnum
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt


class SteeringMode(IntEnum):
    """转向模式枚举"""
    ACKERMANN = 0  # 阿克曼转向
    LATERAL = 1  # 横移
    PARALLEL = 2  # 平行
    SELF_ROTATION = 3  # 自转


@dataclass
class RobotConfig:
    """机器人配置参数"""
    track_width: float = 0.47
    wheelbase: float = 0.64
    wheel_radius: float = 0.1
    ref_velocity: float = 1.0
    time_step: float = 0.05
    length: float = 0.97
    width: float = 0.67


@dataclass
class RobotState:
    """机器人状态"""
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.theta]


class KinematicsEngine:
    """运动学计算引擎"""
    ANGLE_THRESHOLD = 1e-4

    def __init__(self, config: RobotConfig):
        self.config = config

    def compute_wheel_angles(self, mode: SteeringMode, steer_angle: float) -> np.ndarray:
        """计算四轮转向角 [FL, FR, RL, RR]"""
        if mode == SteeringMode.ACKERMANN:
            return self._ackermann_angles(steer_angle)
        elif mode == SteeringMode.LATERAL:
            return self._lateral_angles(steer_angle)
        elif mode == SteeringMode.PARALLEL:
            return np.array([steer_angle] * 4)
        else:
            raise ValueError(f"Cannot compute angles for mode {mode}")

    def compute_wheel_velocities(self, mode: SteeringMode, speed: float, steer_angle: float) -> np.ndarray:
        """计算四轮速度 [FL, FR, RL, RR]"""
        if mode == SteeringMode.ACKERMANN:
            return self._ackermann_velocities(speed, steer_angle)
        elif mode == SteeringMode.LATERAL:
            return self._lateral_velocities(speed, steer_angle)
        elif mode == SteeringMode.PARALLEL:
            return np.array([speed] * 4)
        else:
            raise ValueError(f"Cannot compute velocities for mode {mode}")

    def _ackermann_angles(self, delta: float) -> np.ndarray:
        """阿克曼转向角度计算"""
        if abs(delta) < self.ANGLE_THRESHOLD:
            return np.zeros(4)

        L, W = self.config.length, self.config.width
        cot_delta = 1.0 / np.tan(delta)

        delta_fl = np.arctan(1.0 / (cot_delta - W / L))
        delta_fr = np.arctan(1.0 / (cot_delta + W / L))

        return np.array([delta_fl, delta_fr, -delta_fl, -delta_fr])

    def _ackermann_velocities(self, v: float, delta: float) -> np.ndarray:
        """阿克曼转向速度计算"""
        if abs(delta) < self.ANGLE_THRESHOLD:
            return np.array([v] * 4)

        angles = self._ackermann_angles(delta)
        tan_delta = np.tan(delta)

        velocities = v * tan_delta / np.sin(angles[:2])
        return np.array([velocities[0], velocities[1], velocities[0], velocities[1]])

    def _lateral_angles(self, delta: float) -> np.ndarray:
        """横移转向角度计算"""
        base_offset = math.pi / 2

        if abs(delta) < self.ANGLE_THRESHOLD:
            return np.array([base_offset] * 4)

        L, W = self.config.length, self.config.width
        cot_delta = 1.0 / np.tan(delta)

        delta_fr = np.arctan(1.0 / (cot_delta - L / W))
        delta_rr = np.arctan(1.0 / (cot_delta + L / W))

        return np.array([
            -delta_fr + base_offset,
            delta_fr + base_offset,
            -delta_rr + base_offset,
            delta_rr + base_offset
        ])

    def _lateral_velocities(self, v: float, delta: float) -> np.ndarray:
        """横移速度计算"""
        if abs(delta) < self.ANGLE_THRESHOLD:
            return np.array([v] * 4)

        angles = self._lateral_angles(delta) - math.pi / 2
        tan_delta = np.tan(delta)

        vfl = -v * tan_delta / np.sin(angles[0])
        vrr = v * tan_delta / np.sin(angles[3])

        return np.array([vfl, vfl, vrr, vrr])


class StateUpdater:
    """状态更新器"""

    def __init__(self, config: RobotConfig):
        self.config = config
        self.state = RobotState()

    def update(self, wheel_angles: np.ndarray, wheel_velocities: np.ndarray) -> RobotState:
        """更新机器人状态"""
        assert wheel_angles.shape == (4,), "wheel_angles must be (4,)"
        assert wheel_velocities.shape == (4,), "wheel_velocities must be (4,)"

        dt = self.config.time_step
        L, W = self.config.length, self.config.width
        theta = self.state.theta

        # 计算全局坐标系下的速度分量
        global_angles = wheel_angles + theta
        cos_vals = np.cos(global_angles)
        sin_vals = np.sin(global_angles)

        # 平移速度(四轮平均)
        vx = 0.25 * np.sum(wheel_velocities * cos_vals)
        vy = 0.25 * np.sum(wheel_velocities * sin_vals)

        # 更新位置
        self.state.x += dt * vx
        self.state.y += dt * vy

        # 计算角速度
        cos_steer = np.cos(wheel_angles)
        sin_steer = np.sin(wheel_angles)

        coeffs = np.array([
            -0.5 * W * cos_steer[0] + 0.5 * L * sin_steer[0],
            0.5 * W * cos_steer[1] + 0.5 * L * sin_steer[1],
            -0.5 * W * cos_steer[2] - 0.5 * L * sin_steer[2],
            0.5 * W * cos_steer[3] - 0.5 * L * sin_steer[3],
        ])

        omega = np.sum(wheel_velocities * coeffs) / (W ** 2 + L ** 2)
        self.state.theta = (self.state.theta + dt * omega) % (2 * math.pi)

        return self.state

    def reset(self, x: float = 0, y: float = 0, theta: float = 0):
        """重置状态"""
        self.state = RobotState(x, y, theta)


class TrajectoryGenerator:
    """轨迹生成器"""
    # 生成参数常量
    SPEED_VARIATION = (0.6, 1.4)  # 速度选择上下限
    ACKERMANN_ANGLE_RANGE = (-math.pi / 6, math.pi / 6)  # 阿克曼转角上下限
    LATERAL_ANGLE_RANGE = (-math.pi / 9, math.pi / 9)  # 横向阿克曼转角上下限
    PARALLEL_DELTA_RANGE = (-4 * math.pi / 180, 4 * math.pi / 180)  # 平移模式转角差值上下限
    ROTATION_ANGLE_RANGE = (math.pi / 4, math.pi)  # 自转角度上下限

    SEGMENT_STEPS = 40  # 每一个模态轨迹单元包含的轨迹点个数
    ITERATIONS_PER_MODE = 3  # 每一个模态轨迹单元包含的速度变化次数

    def __init__(self, config: RobotConfig = None, seed: Optional[int] = None):
        """
        初始化轨迹生成器
        Args: config: 机器人配置参数 seed: 随机数种子，如果提供则生成确定性轨迹
        """
        self.config = config or RobotConfig()
        self.kinematics = KinematicsEngine(self.config)
        self.updater = StateUpdater(self.config)
        self._parallel_angle_accumulator = 0.0
        self.seed = seed

        # 如果提供了种子，则初始化随机数生成器
        if seed is not None:
            self.set_seed(seed)

    def set_seed(self, seed: int):
        """
        设置随机数种子
        Args: seed: 随机数种子
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def reset_with_seed(self, seed: Optional[int] = None):
        """
        重置生成器状态并可选地设置新种子
        Args: seed: 新的随机数种子，如果为None则使用已有种子
        """
        self.updater.reset()
        self._parallel_angle_accumulator = 0.0

        if seed is not None:
            self.set_seed(seed)
        elif self.seed is not None:
            self.set_seed(self.seed)

    def generate_random_speed(self) -> float:
        """生成随机速度"""
        factor = random.uniform(*self.SPEED_VARIATION)
        return factor * self.config.ref_velocity

    def generate_segment(self, mode: SteeringMode, steer_angle: float,
                         steps: int) -> List[List[float]]:
        """生成单个轨迹段"""
        # 将步数分为3个子段,每段随机速度
        step_counts = self._split_steps(steps)
        speeds = [self.generate_random_speed() for _ in range(3)]

        trajectory = []

        for step_count, speed in zip(step_counts, speeds):
            wheel_angles = self.kinematics.compute_wheel_angles(mode, steer_angle)
            wheel_velocities = self.kinematics.compute_wheel_velocities(mode, speed, steer_angle)

            for _ in range(step_count):
                state = self.updater.update(wheel_angles, wheel_velocities)
                trajectory.append([state.x, state.y, state.theta, speed])

        return trajectory

    def generate_parallel_segment(self, delta_angle: float, steps: int) -> Tuple[float, List]:
        """生成平行模式轨迹段(转向角渐变)"""
        step_counts = self._split_steps(steps)
        speeds = [self.generate_random_speed() for _ in range(3)]

        trajectory = []
        current_angle = self._parallel_angle_accumulator

        for step_count, speed in zip(step_counts, speeds):
            wheel_velocities = self.kinematics.compute_wheel_velocities(
                SteeringMode.PARALLEL, speed, current_angle
            )

            for _ in range(step_count):
                current_angle += delta_angle
                wheel_angles = np.array([current_angle] * 4)
                state = self.updater.update(wheel_angles, wheel_velocities)
                trajectory.append([state.x, state.y, state.theta, speed])

        return current_angle, trajectory

    def generate_single_mode(self, mode: SteeringMode, iterations: int) -> List:
        """生成单一模式轨迹"""
        trajectory = []

        for _ in range(iterations):
            if mode == SteeringMode.ACKERMANN:
                angle = random.uniform(*self.ACKERMANN_ANGLE_RANGE)
                trajectory.extend(self.generate_segment(mode, angle, self.SEGMENT_STEPS))

            elif mode == SteeringMode.LATERAL:
                angle = random.uniform(*self.LATERAL_ANGLE_RANGE)
                trajectory.extend(self.generate_segment(mode, angle, self.SEGMENT_STEPS))

            elif mode == SteeringMode.PARALLEL:
                delta = random.uniform(*self.PARALLEL_DELTA_RANGE)
                self._parallel_angle_accumulator, seg = self.generate_parallel_segment(
                    delta, self.SEGMENT_STEPS
                )
                trajectory.extend(seg)

            elif mode == SteeringMode.SELF_ROTATION:
                angle = random.uniform(*self.ROTATION_ANGLE_RANGE)
                angle *= random.choice([-1, 1])
                self.updater.state.theta = (self.updater.state.theta + angle) % (2 * math.pi)
                trajectory.append([
                    self.updater.state.x,
                    self.updater.state.y,
                    self.updater.state.theta,
                    self.config.ref_velocity
                ])

        if mode == SteeringMode.PARALLEL:
            self._parallel_angle_accumulator = 0.0

        return trajectory

    def generate_multi_mode(self, n_switches: int,
                            include_mode_labels: bool = False) -> Tuple[List, List, List]:
        """生成多模态轨迹"""
        trajectory = []
        switch_points = []
        rotation_points = []
        last_mode = None

        for i in range(n_switches):
            # 首尾不使用自转,避免自转后紧接自转
            if i == 0 or i == n_switches - 1 or last_mode == SteeringMode.SELF_ROTATION:
                mode = random.choice([SteeringMode.ACKERMANN,
                                      SteeringMode.LATERAL,
                                      SteeringMode.PARALLEL])
            else:
                mode = random.choice(list(SteeringMode))

            # 记录切换点
            if last_mode is not None and last_mode != mode:
                if trajectory:
                    switch_points.append(trajectory[-1])
                    if mode == SteeringMode.SELF_ROTATION:
                        rotation_points.append(trajectory[-1])

            # 生成轨迹段
            segment = self.generate_single_mode(mode, self.ITERATIONS_PER_MODE)

            # 添加模态标签
            if include_mode_labels:
                segment = [[x, y, theta, mode] for x, y, theta, _ in segment]

                # 处理自转前后的标签
                if mode == SteeringMode.SELF_ROTATION and trajectory:
                    trajectory[-1][-1] = SteeringMode.SELF_ROTATION
                elif last_mode == SteeringMode.SELF_ROTATION and trajectory:
                    trajectory[-1][-1] = mode

            trajectory.extend(segment)
            last_mode = mode

        return trajectory, switch_points, rotation_points

    @staticmethod
    def _split_steps(total: int) -> List[int]:
        """将总步数分为3段"""
        step1 = step2 = total // 3
        step3 = total - 2 * step1
        return [step1, step2, step3]


class TrajectoryVisualizer:
    """轨迹可视化工具"""

    @staticmethod
    def plot_trajectory(trajectory: List, switch_points: Optional[List] = None,
                        rotation_points: Optional[List] = None):
        """绘制单条轨迹"""
        if not trajectory:
            return

        data = np.array(trajectory)
        x, y, theta, v = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

        plt.figure(figsize=(24, 12))

        # 特殊点标记
        if switch_points:
            switch_data = np.array(switch_points)
            plt.plot(switch_data[:, 0], switch_data[:, 1], 'bx',
                     markersize=8, label='Switch Point')

        if rotation_points:
            rotation_data = np.array(rotation_points)
            plt.plot(rotation_data[:, 0], rotation_data[:, 1], 'bo',
                     markersize=10, label='Self-Rotation Point')

        # 起点和轨迹
        plt.plot([0], [0], 'rx', markersize=8, label='Start Point')
        plt.plot(x, y, 'b-', linewidth=2, label='Trajectory')

        # 方向箭头(采样绘制)
        arrow_step = max(1, len(x) // 50)
        for i in range(0, len(x), arrow_step):
            dx, dy = np.cos(theta[i]), np.sin(theta[i])
            arrow_length = v[i] * 0.001
            plt.arrow(x[i], y[i], dx * arrow_length, dy * arrow_length,
                      head_width=0.04, head_length=0.08,
                      fc='r', ec='r', linewidth=1.5)

        plt.title("Robot Trajectory with Orientation")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()

    @staticmethod
    def compare_trajectories(*trajectories: List, labels: Optional[List[str]] = None):
        """对比多条轨迹"""
        if not trajectories:
            return

        plt.figure(figsize=(24, 12))
        plt.plot([0], [0], 'rx', markersize=8, label='Start Point')

        styles = ['b-', 'r-', 'g-', 'c-', 'm-']
        if labels is None:
            labels = [f'Trajectory {i + 1}' for i in range(len(trajectories))]

        for i, (traj, label) in enumerate(zip(trajectories, labels)):
            data = np.array(traj)
            style = styles[i % len(styles)]
            plt.plot(data[:, 0], data[:, 1], style, linewidth=2, label=label)

        plt.title("Robot Trajectory Comparison")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()


def show_usage():
    """示例用法"""
    # 示例1: 生成多模态轨迹
    generator = TrajectoryGenerator()
    trajectory, switches, rotations = generator.generate_multi_mode(4)
    TrajectoryVisualizer.plot_trajectory(trajectory, switches, rotations)

    # 示例2: 对比不同模式
    generator1 = TrajectoryGenerator()
    traj_ackermann = generator1.generate_single_mode(SteeringMode.ACKERMANN, 4)

    generator2 = TrajectoryGenerator()
    traj_lateral = generator2.generate_single_mode(SteeringMode.LATERAL, 4)

    generator3 = TrajectoryGenerator()
    traj_parallel = generator3.generate_single_mode(SteeringMode.PARALLEL, 4)

    TrajectoryVisualizer.compare_trajectories(
        traj_ackermann, traj_lateral, traj_parallel,
        labels=['Ackermann', 'Lateral', 'Parallel']
    )
    # 示例3: 使用固定种子生成可复现的轨迹
    generator1 = TrajectoryGenerator(seed=42)
    trajectory1, switches1, rotations1 = generator1.generate_multi_mode(4)

    generator2 = TrajectoryGenerator(seed=42)
    trajectory2, switches2, rotations2 = generator2.generate_multi_mode(4)
    # generator.set_seed(42)
    # 验证两次生成的轨迹是否相同
    traj1_array = np.array(trajectory1)
    traj2_array = np.array(trajectory2)
    print(f"两次生成的轨迹是否完全相同: {np.allclose(traj1_array, traj2_array)}")

if __name__ == '__main__':
    show_usage()