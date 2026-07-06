# -*- coding: utf-8 -*-
"""
robot_configs.py -- 机器人 DH 参数与手眼标定预设
=================================================

各类机械臂的运动学参数和手眼标定数据的集中注册中心。

用法:
    from robot_configs import get_configs, get_robot, get_hand_eye

    # 获取默认配置
    robot, hand_eye = get_configs()

    # 按名称获取
    robot = get_robot("panda")
    hand_eye = get_hand_eye("droid-left")

    # 一步获取两者
    robot, hand_eye = get_configs("panda", "droid-left")

可用机械臂名称:  panda, kinova-gen3
可用手眼标定名称: simple, droid-left, droid-right, kinova-gen3
"""

import numpy as np
from collections import namedtuple


# ============================================================
# 数据类型定义
# ============================================================

RobotConfig = namedtuple("RobotConfig", ["a", "alpha", "d", "theta_offset"])
"""
机器人 D-H 参数配置。
  a (7,):       连杆长度
  alpha (7,):   连杆扭角 (弧度)
  d (7,):       连杆偏距
  theta_offset (7,):  关节零点偏移 (弧度)
"""

HandEyeCalib = namedtuple("HandEyeCalib", ["R", "t"])
"""
手眼标定参数：相机坐标系到末端法兰坐标系的变换。
  R (3,3):  旋转矩阵 R_he
  t (3,):   平移向量 t_he (米)
"""


# ============================================================
# 机器人 DH 参数预设
# ============================================================

# --- Franka Panda（7 自由度）---
PANDA = RobotConfig(
    a=np.array([0, 0, 0, 0.0825, -0.0825, 0, 0]),
    alpha=np.array([-np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0]),
    d=np.array([0.333, 0, 0.316, 0, 0.384, 0, 0.088]),
    theta_offset=np.array([0, -np.pi/2, 0, -np.pi/2, 0, np.pi/2, np.pi/4]),
)
""" Franka Panda 7 自由度机械臂标准 D-H 参数 """

# 注册表：名称 -> RobotConfig
ROBOT_CONFIGS = {
    "panda": PANDA,
    "kinova-gen3": KINOVA_GEN3,
}


# ============================================================
# 手眼标定参数预设
# ============================================================


# --- Kinova Gen3（7 自由度）---
# 标准 DH 参数，来源：Kinova Kortex API 官方文档
# 参考：https://github.com/Kinovarobotics/ros2_kortex
KINOVA_GEN3 = RobotConfig(
    a=np.array([0, 0, 0, 0.21038, 0, 0, 0]),
    alpha=np.array([0.000000, -1.570796, 1.570796, 1.570796, -1.570796, 1.570796, -1.570796]),
    d=np.array([0.15643, 0.00538, -0.00638, 0.00638, 0.20843, 0.10593, 0]),
    theta_offset=np.array([0, -1.570796, 0, 0, 0, 0, 0]),
)
""" Kinova Gen3 7 自由度机械臂标准 D-H 参数。来源：Kinova Kortex API。注：末端工具/手眼标定需额外配置。 """
# --- 简化近似版（用于测试）---
PANDA_HAND_EYE_SIMPLE = HandEyeCalib(
    R=np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
    t=np.array([0, 0, 0.05]),
)
"""
简化的手眼标定（仅近似，用于测试）。
  x_cam = x_ee,  y_cam = z_ee,  z_cam = -y_ee
"""

# --- DROID 左腕相机实测标定（serial=17368348）---
# 来源：DROID trajectory.h5 observation/camera_extrinsics/{serial}_left_gripper_offset
DROID_HAND_EYE_LEFT = HandEyeCalib(
    R=np.array([
        [ 0.0166,  0.9463,  0.3229],
        [-0.9991,  0.0030,  0.0427],
        [ 0.0394, -0.3233,  0.9455],
    ]),
    t=np.array([-0.07808,  0.02325,  0.01505]),
)
""" DROID 数据集腕部左相机手眼标定（serial=17368348） """

# --- DROID 右腕相机实测标定（serial=17368348）---
DROID_HAND_EYE_RIGHT = HandEyeCalib(
    R=np.array([
        [ 0.0287,  0.9513,  0.3071],
        [-0.9990,  0.0168,  0.0414],
        [ 0.0343, -0.3079,  0.9508],
    ]),
    t=np.array([-0.07052, -0.04072,  0.02304]),
)
""" DROID 数据集腕部右相机手眼标定（serial=17368348） """


# --- Kinova Gen3 实测手眼标定（倾斜向下安装）---
# 安装方式：法兰中心线向前 0.11m → 绕 x 轴向前倾斜 20° → 沿新方向再向前 0.03m
KINOVA_GEN3_HAND_EYE = HandEyeCalib(
    R=np.array([
        [1.0000, 0.0000, 0.0000],
        [0.0000, 0.9397, -0.3420],
        [0.0000, 0.3420,  0.9397],
    ]),
    t=np.array([0.00000, -0.01026, 0.13819]),
)
"""
Kinova Gen3 手眼标定（倾斜向下安装）。
说明：相机朝向前下方，z=0.138m 在法兰前方，y=-0.01m 在中心线下方。
安装步骤：沿法兰 z 轴 0.11m → 绕 x 轴向下倾斜 20° → 沿相机新 z 轴 0.03m。
"""
# 注册表：名称 -> HandEyeCalib
HAND_EYE_CONFIGS = {
    "simple": PANDA_HAND_EYE_SIMPLE,
    "droid-left": DROID_HAND_EYE_LEFT,
    "droid-right": DROID_HAND_EYE_RIGHT,
    "kinova-gen3": KINOVA_GEN3_HAND_EYE,
}

# csv_loader 兼容的别名
HAND_EYE_MAP = HAND_EYE_CONFIGS


# ============================================================
# 查找函数
# ============================================================

def get_robot(name="panda"):
    """
    按名称获取机器人 DH 参数配置。

    参数:
        name (str): 机器人名称（默认 "panda"）

    返回:
        RobotConfig

    异常:
        KeyError: 名称不在 ROBOT_CONFIGS 中时抛出
    """
    if name not in ROBOT_CONFIGS:
        raise KeyError(
            f"未知机器人: {name!r}。"
            f"可用配置: {list(ROBOT_CONFIGS.keys())}"
        )
    return ROBOT_CONFIGS[name]


def get_hand_eye(name=None):
    """
    按名称获取手眼标定参数。

    参数:
        name (str 或 None): 手眼标定名称。
        若为 None，返回默认值（"simple"）。

    返回:
        HandEyeCalib

    异常:
        KeyError: name 不为 None 且不在 HAND_EYE_CONFIGS 中时抛出
    """
    if name is None:
        name = "simple"
    if name not in HAND_EYE_CONFIGS:
        raise KeyError(
            f"未知手眼标定: {name!r}。"
            f"可用配置: {list(HAND_EYE_CONFIGS.keys())}"
        )
    return HAND_EYE_CONFIGS[name]


def get_configs(robot_name="panda", hand_eye_name=None):
    """
    按名称同时获取机器人配置和可选的手眼标定。

    这是主要的便捷函数。传入 hand_eye_name 即可同时获取
    机械臂参数和相机到法兰盘的变换。

    参数:
        robot_name (str): 机器人名称（默认 "panda"）
        hand_eye_name (str 或 None): 手眼标定名称。
            若为 None，返回的 hand_eye 为 None（不应用标定）。

    返回:
        tuple: (RobotConfig, HandEyeCalib 或 None)

    示例:
        robot, he = get_configs("panda", "droid-left")
        robot, he = get_configs()  # 使用默认值
    """
    robot = get_robot(robot_name)
    hand_eye = get_hand_eye(hand_eye_name) if hand_eye_name else None
    return robot, hand_eye
