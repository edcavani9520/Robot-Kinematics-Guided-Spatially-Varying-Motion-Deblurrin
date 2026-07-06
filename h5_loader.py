"""
h5_loader.py — DROID/Episode h5 数据加载器
=============================================
功能：
  1. 自动检测 h5 格式（DROID trajectory.h5 / Episode episode_XXXX.h5）
  2. 读取关节角、图像帧、时间戳
  3. 帧同步（时间戳对齐）
  4. 帧读取器（JPEG 内嵌解码 / MP4 读取）

两种格式：
  - DROID:  action/joint_position + observations/timestamp + 外置 MP4
  - Episode: camera/rgb (JPEG bytes) + robot/joint_position (deg) + camera/timestamp
"""

import h5py
import cv2
import numpy as np
from pathlib import Path


def detect_h5_format(h5_path):
    """自动检测 h5 是 DROID 格式还是 Episode 格式"""
    with h5py.File(str(h5_path), "r") as f:
        keys = set(f.keys())
        if "action" in keys and "observation" in keys:
            return "droid"
        if "camera" in keys and "robot" in keys:
            return "episode"
    return "unknown"


# ============================================================
# Episode 格式 (episode_XXXX.h5)
# ============================================================

def load_episode_h5(h5_path):
    """
    读取 episode_XXXX.h5 格式。

    h5 结构:
      camera/rgb             (N,)  object     — JPEG 压缩字节
      camera/timestamp       (N,)  float64    — 时间戳 (秒)
      robot/joint_position   (M, 7) float64   — 关节角 (度)
      robot/joint_velocity   (M, 7) float64   — 关节角速度 (度/秒)
      robot/timestamp        (M,)  float64    — 时间戳 (秒)
      robot/tool_pose        (M, 6) float64   — tool pose (参考)
    """
    with h5py.File(str(h5_path), "r") as f:
        rgb_group = f["camera/rgb"][:]
        cam_ts = f["camera/timestamp"][:]
        joint_pos_deg = f["robot/joint_position"][:]
        joint_vel_degs = f["robot/joint_velocity"][:]
        robot_ts = f["robot/timestamp"][:]

    # 解码第一帧确认尺寸
    sample_img = cv2.imdecode(rgb_group[0], cv2.IMREAD_COLOR)
    if sample_img is None:
        raise RuntimeError("无法解码 camera/rgb 中的 JPEG 图像")
    H, W = sample_img.shape[:2]

    # 关节角转弧度
    joint_pos = np.deg2rad(joint_pos_deg)
    joint_vel = np.deg2rad(joint_vel_degs)

    # 时间同步：对每个 camera 帧找最近的 robot 时间点
    sync_indices = np.searchsorted(robot_ts, cam_ts)
    sync_indices = np.clip(sync_indices, 0, len(robot_ts) - 1)
    for i in range(len(cam_ts)):
        idx = sync_indices[i]
        if idx > 0 and abs(cam_ts[i] - robot_ts[idx - 1]) < abs(cam_ts[i] - robot_ts[idx]):
            sync_indices[i] = idx - 1

    print(f"  [h5] Episode 格式: {len(rgb_group)} 帧, {W}×{H}, "
          f"{len(joint_pos)} 个机器人时间点")
    print(f"  [h5] 时间同步: {len(np.unique(sync_indices))}/{len(cam_ts)} 帧已匹配")

    return {
        "format": "episode",
        "rgb_bytes": rgb_group,
        "cam_timestamps": cam_ts,
        "joint_positions": joint_pos,
        "joint_velocities": joint_vel,
        "robot_timestamps": robot_ts,
        "sync_indices": sync_indices,
        "num_frames": len(rgb_group),
        "H": H,
        "W": W,
    }


# ============================================================
# DROID 格式 (trajectory.h5)
# ============================================================

def load_droid_h5(h5_path, episode_dir):
    """
    读取 DROID trajectory.h5 格式 + 外置 MP4。
    """
    with h5py.File(str(h5_path), "r") as f:
        joint_pos = f["action/joint_position"][:]
        joint_vel = f["action/joint_velocity"][:]
        obs_joint_pos = f["observation/robot_state/joint_positions"][:]
        obs_joint_vel = f["observation/robot_state/joint_velocities"][:]

        # 摄像头 serial
        cameras_grp = f["observation/timestamp/cameras"]
        camera_serials = []
        for key in cameras_grp.keys():
            if key.endswith("_estimated_capture"):
                camera_serials.append(key.replace("_estimated_capture", ""))

        camera_captures = {}
        for serial in camera_serials:
            camera_captures[serial] = f[
                f"observation/timestamp/cameras/{serial}_estimated_capture"
            ][:]

    # 找 MP4
    recordings_mp4 = Path(episode_dir) / "recordings" / "MP4"
    video_paths = {}
    for serial in camera_serials:
        for p in [
            recordings_mp4 / f"{serial}.mp4",
            recordings_mp4 / f"{serial}-stereo.mp4",
        ]:
            if p.exists():
                video_paths[serial] = str(p)
                break

    if not video_paths:
        raise FileNotFoundError(f"未找到 MP4 文件，请检查 {recordings_mp4}/")

    cap = cv2.VideoCapture(next(iter(video_paths.values())))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_vf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"  [h5] DROID 格式: {len(joint_pos)} steps, "
          f"视频 {W}×{H} {video_fps:.1f}fps")
    print(f"  [h5] 摄像头 serials: {camera_serials}")

    return {
        "format": "droid",
        "joint_positions": joint_pos,
        "joint_velocities": joint_vel,
        "obs_joint_positions": obs_joint_pos,
        "obs_joint_velocities": obs_joint_vel,
        "camera_serials": camera_serials,
        "camera_captures": camera_captures,
        "video_paths": video_paths,
        "video_fps": video_fps,
        "num_frames": len(joint_pos),
        "H": H,
        "W": W,
    }


# ============================================================
# 帧读取器
# ============================================================

class EpisodeFrameReader:
    """从 episode h5 的 JPEG 字节解码帧"""
    def __init__(self, rgb_bytes):
        self.rgb_bytes = rgb_bytes
        self.N = len(rgb_bytes)

    def read_frame(self, idx):
        if idx >= self.N:
            return None
        return cv2.imdecode(self.rgb_bytes[idx], cv2.IMREAD_COLOR)

    def close(self):
        pass


class DroidFrameReader:
    """从 DROID MP4 读取帧，支持时间戳对齐"""
    def __init__(self, video_path, video_fps, cap_timestamps, total_frames):
        self.cap = cv2.VideoCapture(video_path)
        self.video_fps = video_fps
        self.cap_timestamps = cap_timestamps
        self.total_frames = total_frames

    def read_frame(self, step_idx):
        """按 step 索引读取对应视频帧（基于时间戳对齐）"""
        ts = self.cap_timestamps[step_idx] if step_idx < len(self.cap_timestamps) else step_idx
        # 计算从第一个时间戳开始的相对偏移
        t0 = self.cap_timestamps[0]
        rel_us = ts - t0  # 微秒
        vf = int(round(rel_us * self.video_fps / 1_000_000))
        vf = max(0, min(vf, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, vf)
        ret, frame = self.cap.read()
        return frame if ret else None

    def close(self):
        self.cap.release()
