import math
import numpy as np


def trajectory_padding(trajectory, window=5):
    """轨迹长度小于观测窗口时重复最后一个点进行填补"""
    if len(trajectory) >= window:
        return trajectory
    last_point = trajectory[-1]
    padding_point = [last_point[0], last_point[1], last_point[2], 0]
    trajectory.extend([padding_point] * (window - len(trajectory)))
    return trajectory


def trajectory_transfer(trajectory, robot_state):
    """全局坐标系转局部坐标系（向量化优化版本）"""
    robot_x, robot_y, robot_theta = robot_state[0], robot_state[1], robot_state[2]

    cos_theta = math.cos(robot_theta)
    sin_theta = math.sin(robot_theta)

    # 构建旋转和平移矩阵
    translation_matrix = np.array([
        [cos_theta, sin_theta, -robot_x * cos_theta - robot_y * sin_theta],
        [-sin_theta, cos_theta, -robot_y * cos_theta + robot_x * sin_theta],
        [0.0, 0.0, 1.0]
    ])

    # 向量化处理所有点
    trajectory_array = np.array(trajectory)
    points = np.column_stack([trajectory_array[:, :2], np.ones(len(trajectory))])
    points_local = (translation_matrix @ points.T).T[:, :2]

    # 处理角度
    theta_local = (trajectory_array[:, 2] - robot_theta) % (2 * math.pi)

    # 组合结果
    ref_trajectory = np.column_stack([points_local, theta_local, trajectory_array[:, 3]])

    return ref_trajectory.tolist()


def util_flatten(nested_list):
    """递归展平嵌套列表（优化版本）"""
    result = []
    for item in nested_list:
        if isinstance(item, (list, tuple)):
            result.extend(util_flatten(item))
        else:
            result.append(item)
    return result


def util_flatten_fast(nested_list):
    """快速展平嵌套列表（非递归版本，性能更好）"""
    result = []
    stack = [nested_list]
    while stack:
        current = stack.pop()
        if isinstance(current, (list, tuple)):
            stack.extend(reversed(current))
        else:
            result.append(current)
    return result