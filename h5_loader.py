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
        if "action" in keys and "obs" in keys and "timestamps" in keys:
            return "kinova"
    return "unknown"


# ============================================================
# Episode 格式 (episode_XXXX.h5)
# ============================================================

def load_episode_h5(h5_path):
    """
    读取 episode_XXXX.h5 格式。

    支持两种格式:
    [新格式] camera/rgb + robot/tool_twist
    [旧格式] obs/image + obs/proprio + action
    """
    with h5py.File(str(h5_path), "r") as f:
        if "camera" in f:
            # === 新格式：带 tool_twist ===
            rgb_group = f["camera/rgb"][:]
            cam_ts = f["camera/timestamp"][:]
            joint_pos_deg = f["robot/joint_position"][:]
            joint_vel_degs = f["robot/joint_velocity"][:]
            tool_pose_arr = f["robot/tool_pose"][:]
            tool_twist_arr = f["robot/tool_twist"][:]
            robot_ts = f["robot/timestamp"][:]

            sample_img = cv2.imdecode(rgb_group[0], cv2.IMREAD_COLOR)
            if sample_img is None:
                raise RuntimeError("无法解码 camera/rgb 中的 JPEG 图像")
            H, W = sample_img.shape[:2]

            joint_pos = np.deg2rad(joint_pos_deg)
            joint_vel = np.deg2rad(joint_vel_degs)

            sync_indices = np.searchsorted(robot_ts, cam_ts)
            sync_indices = np.clip(sync_indices, 0, len(robot_ts) - 1)
            for i in range(len(cam_ts)):
                idx = sync_indices[i]
                if idx > 0 and abs(cam_ts[i] - robot_ts[idx - 1]) < abs(cam_ts[i] - robot_ts[idx]):
                    sync_indices[i] = idx - 1

            print(f"  [h5] Episode 格式: {len(rgb_group)} 帧, {W}x{H}, "
                  f"{len(joint_pos)} 个机器人时间点")
            print(f"  [h5] 时间同步: {len(np.unique(sync_indices))}/{len(cam_ts)} 帧已匹配")
            fmt_str = "episode"
        elif "obs" in f:
            # === 旧格式：obs/image + obs/proprio (无 tool_twist) ===
            rgb_img = f["obs/image"][:]        # (N, H, W) uint8 grayscale
            proprio = f["obs/proprio"][:]       # (N, 8) float64
            action = f["action"][:]              # (N, 7) float64
            robot_ts = f["timestamps"][:]

            N, H, W = rgb_img.shape

            # proprio[:, :7] = te joint positions (deg), 7=gripper
            joint_pos_deg = proprio[:, :7].copy()
            # Compute velocity from finite diff with angle wrapping
            diff_raw = np.diff(joint_pos_deg, axis=0)
            diff = (diff_raw + 180) % 360 - 180  # unwrap 360 deg
            dt = np.diff(robot_ts)[:, np.newaxis]  # (N-1,1)
            vel = np.zeros_like(joint_pos_deg)
            vel[0] = diff[0] / dt[0]
            vel[1:-1] = (diff[:-1] + diff[1:]) / (dt[:-1] + dt[1:])
            vel[-1] = diff[-1] / dt[-1]
            joint_vel_degs = vel

            joint_pos = np.deg2rad(joint_pos_deg)
            joint_vel = np.deg2rad(joint_vel_degs)
            # 图像存为 list，EpisodeFrameReader 会自动检测类型
            rgb_group = [rgb_img[i] for i in range(N)]

            tool_pose_arr = None
            tool_twist_arr = None
            sync_indices = np.arange(N)
            cam_ts = robot_ts

            print(f"  [h5] 旧格式: {N} 帧, {W}x{H} grayscale, "
                  f"{len(joint_pos)} 个机器人时间点")
            print(f"  [h5] 帧和动作已严格对齐, tool_twist=无 (将从关节角计算)")
            fmt_str = "old_format"
        else:
            raise ValueError(f"未知的 h5 格式: {list(f.keys())}")

    return {
        "format": fmt_str,
        "rgb_bytes": rgb_group,
        "cam_timestamps": cam_ts,
        "joint_positions": joint_pos,
        "joint_velocities": joint_vel,
        "tool_pose": tool_pose_arr,
        "tool_twist": tool_twist_arr,
        "robot_timestamps": robot_ts,
        "sync_indices": sync_indices,
        "num_frames": len(rgb_group) if hasattr(rgb_group, "__len__") else rgb_group.shape[0],
        "H": H,
        "W": W,
    }
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
        "tool_pose": tool_pose_arr,
        "tool_twist": tool_twist_arr,
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
        raw = self.rgb_bytes[idx]
        # Support: 1) h5py uint8 JPEG bytes  2) external bytes  3) decoded ndarray
        if isinstance(raw, np.ndarray) and raw.dtype == np.uint8:
            if raw.ndim == 1:
                return cv2.imdecode(raw, cv2.IMREAD_COLOR)
            elif raw.ndim == 2:
                return cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
            elif raw.ndim >= 3:
                c = raw.shape[2]
                if c >= 3:
                    return raw[..., :3]
                else:
                    return cv2.cvtColor(raw[..., 0], cv2.COLOR_GRAY2BGR)
        elif isinstance(raw, (bytes, np.bytes_)):
            return cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        else:
            raise TypeError("Unknown image format")

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


# ============================================================
# Kinova 格式 (RLDX-style episode_XXXX.h5)
# ============================================================

def load_kinova_h5(h5_path):
    """
    读取 RLDX 风格 Kinova h5 格式。

    h5 结构:
      obs/image          (N, H, W) uint8      — 灰度图像（已解码）
      obs/proprio        (N, 8) float64       — 关节角度(°) + gripper
      action             (N, 7) float64       — twist velocity [vx,vy,vz,wx,wy,wz]
      timestamps         (N,) float64         — 时间戳

    Root attrs: camera_fps, camera_height, camera_width, robot_type ...
    """
    with h5py.File(str(h5_path), "r") as f:
        images = f["obs/image"][:]          # (N, H, W) uint8
        proprio = f["obs/proprio"][:]       # (N, 8) float64
        action = f["action"][:]             # (N, 7) float64
        timestamps = f["timestamps"][:]     # (N,) float64
        camera_fps = f.attrs.get("camera_fps", 10.0)

    N, H, W = images.shape

    # 关节角度: 前7列 (度) → 弧度
    joint_pos_deg = proprio[:, :7]
    joint_pos = np.deg2rad(joint_pos_deg)

    # 通过有限差分计算关节速度 (rad/s)
    dt = np.diff(timestamps, prepend=timestamps[0])
    dt = np.maximum(dt, 1e-6)
    joint_vel_deg = np.diff(joint_pos_deg, axis=0, prepend=joint_pos_deg[:1])
    joint_vel_deg[0] = 0
    joint_vel = np.deg2rad(joint_vel_deg / dt[:, np.newaxis])

    # twist velocity: 直接用 action 数据
    tool_twist = action[:, :6]   # 前6列: [vx,vy,vz,wx,wy,wz]; 第7列是gripper
    # tool_pose: 此格式没有存储，设为零 (旋转部分为单位阵)
    tool_pose = np.zeros((N, 6))

    print(f"  [h5] Kinova格式: {N}帧, {W}x{H}, {7}个关节, camera_fps={camera_fps}")

    return {
        "format": "kinova",
        "images": images,
        "joint_positions": joint_pos,
        "joint_velocities": joint_vel,
        "tool_pose": tool_pose,
        "tool_twist": tool_twist,
        "timestamps": timestamps,
        "num_frames": N,
        "H": H,
        "W": W,
        "camera_fps": camera_fps,
    }


class KinovaFrameReader:
    """直接从 kinova h5 的灰度图像数组读取帧"""
    def __init__(self, images):
        self.images = images
        self.N = len(images)

    def read_frame(self, idx):
        if idx >= self.N:
            return None
        # 灰度 → BGR 3通道，以便和下游接口一致
        return cv2.cvtColor(self.images[idx], cv2.COLOR_GRAY2BGR)

    def close(self):
        pass
