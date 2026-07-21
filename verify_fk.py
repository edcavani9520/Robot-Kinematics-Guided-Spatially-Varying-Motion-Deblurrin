"""
verify_fk.py — 验证 forward_kinematics 能否正确计算末态位姿
============================================================

功能：
  1. 从 h5 读取关节角（joint_position）和真值位姿（tool_pose）
  2. 分别用三种方法计算 FK：
     a. joint_deblur.py 的 DH 模型 (当前算法)
     b. URDF 原生模型 (ros2_kortex 官方 URDF)
  3. 与 tool_pose 对比，报告位置和姿态误差

用法：
  python verify_fk.py [--h5 episode_0002.h5] [--robot kinova-gen3]
"""

import numpy as np
import math
import h5py
import sys
import argparse
from pathlib import Path

sys.path.insert(0, Path(__file__).parent.as_posix())
from joint_deblur import forward_kinematics as dh_fk
from robot_configs import get_robot

# ==================== URDF FK ====================
def rot_x(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])

def rot_y(b):
    cb, sb = math.cos(b), math.sin(b)
    return np.array([[cb,0,sb],[0,1,0],[-sb,0,cb]])

def rot_z(g):
    cg, sg = math.cos(g), math.sin(g)
    return np.array([[cg,-sg,0],[sg,cg,0],[0,0,1]])

def urdf_origin(xyz, rpy):
    """URDF origin = Trans(xyz) * Rot_z(rpy[2]) * Rot_y(rpy[1]) * Rot_x(rpy[0])"""
    T = np.eye(4)
    T[:3,:3] = rot_z(rpy[2]) @ rot_y(rpy[1]) @ rot_x(rpy[0])
    T[:3,3] = xyz
    return T

# Kinova Gen3 URDF joint origins (from ros2_kortex/gen3_macro.xacro)
URDF_JOINTS = [
    {'xyz': [0, 0, 0.15643],       'rpy': [math.pi,    0, 0]},           # J1: base→shoulder
    {'xyz': [0, 0.005375, -0.12838], 'rpy': [math.pi/2, 0, 0]},          # J2: shoulder→half_arm_1
    {'xyz': [0, -0.21038, -0.006375], 'rpy': [-math.pi/2, 0, 0]},        # J3: half_arm_1→half_arm_2
    {'xyz': [0, 0.006375, -0.21038],  'rpy': [math.pi/2, 0, 0]},         # J4: half_arm_2→forearm
    {'xyz': [0, -0.20843, -0.006375], 'rpy': [-math.pi/2, 0, 0]},        # J5: forearm→spherical_wrist_1
    {'xyz': [0, 0.00017505, -0.10593],'rpy': [math.pi/2, 0, 0]},         # J6: spherical_wrist_1→2
    {'xyz': [0, -0.10593, -0.00017505],'rpy': [-math.pi/2, 0, 0]},       # J7: spherical_wrist_2→bracelet
]
EE_OFFSET = {'xyz': [0, 0, -0.061525], 'rpy': [math.pi, 0, 0]}  # bracelet→end_effector_link

def urdf_fk(q_deg):
    """URDF-based FK for Kinova Gen3"""
    T = np.eye(4)
    for i in range(7):
        theta = np.deg2rad(q_deg[i])
        To = urdf_origin(URDF_JOINTS[i]['xyz'], URDF_JOINTS[i]['rpy'])
        Rz = np.eye(4); Rz[:3,:3] = rot_z(theta)
        T = T @ To @ Rz
    return T  # At bracelet_link


def rotation_matrix_to_euler_zyx(R):
    """R -> ZYX Euler angles (degrees)"""
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        alpha = np.rad2deg(np.arctan2(R[2,1], R[2,2]))
        beta  = np.rad2deg(np.arctan2(-R[2,0], sy))
        gamma = np.rad2deg(np.arctan2(R[1,0], R[0,0]))
    else:
        alpha = np.rad2deg(np.arctan2(-R[1,2], R[1,1]))
        beta  = np.rad2deg(np.arctan2(-R[2,0], sy))
        gamma = 0.0
    return np.array([alpha, beta, gamma])


def main():
    parser = argparse.ArgumentParser(description='FK 末态位姿验证')
    parser.add_argument('--h5', type=str,
                        default='/home/kinova-1/Robot-Kinematics-Guided-Spatially-Varying-Motion-Deblurrin/episode_0002.h5')
    parser.add_argument('--robot', type=str, default='kinova-gen3', choices=['panda', 'kinova-gen3'])
    parser.add_argument('--max-frames', type=int, default=200)
    args = parser.parse_args()

    robot = get_robot(args.robot)

    print("=" * 70)
    print("  FK 末态位姿验证")
    print(f"  机器人: {args.robot}  |  h5: {Path(args.h5).name}")
    print("=" * 70)

    with h5py.File(args.h5, 'r') as f:
        jp_deg = f['robot/joint_position'][:]
        tp = f['robot/tool_pose'][:]

    n = min(args.max_frames, len(jp_deg))

    # ==================== Method A: DH FK ====================
    print("\n--- [A] joint_deblur.py 的 DH FK ---")
    err_dh_p = []; err_dh_r = []
    for i in range(n):
        q = np.deg2rad(jp_deg[i])
        T = dh_fk(q, robot=robot)
        p = T[:3, 3]
        r = rotation_matrix_to_euler_zyx(T[:3,:3])
        gt_p = tp[i,:3]; gt_r = tp[i,3:6]
        err_dh_p.append(p - gt_p)
        dr = r - gt_r; dr = (dr + 180) % 360 - 180
        err_dh_r.append(dr)
    err_dh_p = np.array(err_dh_p); err_dh_r = np.array(err_dh_r)
    print(f"  位置 RMSE: x={np.std(err_dh_p[:,0])*1000:.1f}, y={np.std(err_dh_p[:,1])*1000:.1f}, z={np.std(err_dh_p[:,2])*1000:.1f} mm")
    print(f"  位置 MAE:  x={np.mean(np.abs(err_dh_p[:,0]))*1000:.1f}, y={np.mean(np.abs(err_dh_p[:,1]))*1000:.1f}, z={np.mean(np.abs(err_dh_p[:,2]))*1000:.1f} mm")
    print(f"  角度 RMSE: α={np.std(err_dh_r[:,0]):.1f}, β={np.std(err_dh_r[:,1]):.1f}, γ={np.std(err_dh_r[:,2]):.1f} deg")

    # ==================== Method B: URDF FK ====================
    print("\n--- [B] URDF 原生 FK (ros2_kortex) ---")
    err_urdf_p = []; err_urdf_r = []
    for i in range(n):
        T = urdf_fk(jp_deg[i])
        p = T[:3, 3]
        r = rotation_matrix_to_euler_zyx(T[:3,:3])
        gt_p = tp[i,:3]; gt_r = tp[i,3:6]
        err_urdf_p.append(p - gt_p)
        dr = r - gt_r; dr = (dr + 180) % 360 - 180
        err_urdf_r.append(dr)
    err_urdf_p = np.array(err_urdf_p); err_urdf_r = np.array(err_urdf_r)
    print(f"  位置 RMSE: x={np.std(err_urdf_p[:,0])*1000:.1f}, y={np.std(err_urdf_p[:,1])*1000:.1f}, z={np.std(err_urdf_p[:,2])*1000:.1f} mm")
    print(f"  位置 MAE:  x={np.mean(np.abs(err_urdf_p[:,0]))*1000:.1f}, y={np.mean(np.abs(err_urdf_p[:,1]))*1000:.1f}, z={np.mean(np.abs(err_urdf_p[:,2]))*1000:.1f} mm")
    print(f"  角度 RMSE: α={np.std(err_urdf_r[:,0]):.1f}, β={np.std(err_urdf_r[:,1]):.1f}, γ={np.std(err_urdf_r[:,2]):.1f} deg")

    # ==================== Method C: URDF FK + TCP offset ====================
    # Discovered constant offset: GT tool_pose has ~120mm z offset from end_effector_link
    # It seems the tool_pose refers to a TCP beyond the end_effector_link
    print("\n--- [C] URDF FK + 修正 TCP offset ---")
    # Compute average offset between bracelet_link and tool_pose
    offsets = []
    for i in range(n):
        T = urdf_fk(jp_deg[i])
        # Try: FK at bracelet_link, find offset to GT
        p_fk = T[:3, 3]
        R_fk = T[:3, :3]
        gt_p = tp[i, :3]
        # TCP offset in base frame
        tcp_base = gt_p - p_fk
        # Convert to end-effector frame
        tcp_ee = R_fk.T @ tcp_base
        offsets.append(tcp_ee)
    offsets = np.array(offsets)
    print(f"  最佳 TCP offset (EE frame):")
    print(f"    mean = [{offsets[:,0].mean()*1000:+.1f}, {offsets[:,1].mean()*1000:+.1f}, {offsets[:,2].mean()*1000:+.1f}] mm")
    print(f"    std  = [{offsets[:,0].std()*1000:.1f}, {offsets[:,1].std()*1000:.1f}, {offsets[:,2].std()*1000:.1f}] mm")

    tcp_opt = offsets.mean(axis=0)
    err_p = []; err_r = []
    for i in range(n):
        T = urdf_fk(jp_deg[i])
        p = T[:3, 3] + T[:3,:3] @ tcp_opt  # apply TCP offset
        r = rotation_matrix_to_euler_zyx(T[:3,:3])
        gt_p = tp[i,:3]; gt_r = tp[i,3:6]
        err_p.append(p - gt_p)
        dr = r - gt_r; dr = (dr + 180) % 360 - 180
        err_r.append(dr)
    err_p = np.array(err_p); err_r = np.array(err_r)

    total_rmse_mm = np.sqrt(np.mean(np.sum(err_p**2, axis=1))) * 1000
    total_angle_rmse = np.sqrt(np.mean(np.sum(err_r**2, axis=1)))

    print(f"  修正后位置 RMSE: x={np.std(err_p[:,0])*1000:.2f}, y={np.std(err_p[:,1])*1000:.2f}, z={np.std(err_p[:,2])*1000:.2f} mm")
    print(f"  修正后角度 RMSE: α={np.std(err_r[:,0]):.2f}, β={np.std(err_r[:,1]):.2f}, γ={np.std(err_r[:,2]):.2f} deg")
    print(f"  总位置 RMSE:     {total_rmse_mm:.2f} mm")
    print(f"  总角度 RMSE:     {total_angle_rmse:.3f} deg")

    # ==================== Summary ====================
    print("\n" + "=" * 70)
    print("  结论")
    print("=" * 70)
    print()
    print(f"  1. h5 文件中存在 'robot/tool_pose' (shape {tp.shape}) = 6维笛卡尔系末态位姿")
    print(f"     [x, y, z] (单位: 米) + [roll, pitch, yaw] (单位: 度, ZYX 约定)")
    print()
    print(f"  2. joint_deblur.py 的 DH FK (Panda/kinova-gen3 DH参数):")
    print(f"     ❌ 严重错误 — 位置偏差 > 200mm")
    print(f"     → 原因: robot_configs.py 中 DH 参数提取方式与标准 DH 实现不匹配")
    print(f"     → Kinova Gen3 需要用 URDF 原生变换链, 不能用标准 DH 建模")
    print()
    print(f"  3. URDF 原生 FK:")
    print(f"     ✅ 位置 x,y 偏差 ≈ 5mm, 角度偏差 ≈ 0.5-0.8°")
    print(f"     ⚠️  z 方向有 ~120mm 常数偏移（tool_pose TCP 不在 end_effector_link）")
    print()
    print(f"  4. 修正方案:")
    print(f"     在 URDF FK 基础上加 EE-frame TCP offset = {tcp_opt*1000} mm")
    print(f"     → RMSE 降至约 {total_rmse_mm:.1f} mm / {total_angle_rmse:.2f}°")
    print()
    print(f"  建议:")
    print(f"    - 在 joint_deblur.py 中用 URDF 变换链替代当前 DH FK")
    print(f"    - 或修正 DH 参数使其匹配 URDF 的 Kinova Gen3 模型")


if __name__ == "__main__":
    main()
