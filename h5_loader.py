"""Strict loader for the current Kinova Gen3 RGB episode format.

Accepted HDF5 schema::

    obs/image    (N, H, W, 3) uint8, RGB channel order
    obs/proprio  (N, 8)       joint angles in degrees + gripper
    action       (N, 7)       end-effector pose delta + gripper target
    timestamps   (N,)         Unix timestamps in seconds
"""

from pathlib import Path

import h5py
import numpy as np


REQUIRED_DATASETS = ("obs/image", "obs/proprio", "action", "timestamps")
CAMERA_INTRINSIC_ATTRS = ("fx", "fy", "cx", "cy")


def _validate_episode_arrays(images, proprio, actions, timestamps):
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(
            "obs/image must have RGB shape (N, H, W, 3); "
            f"got {images.shape}"
        )
    if images.dtype != np.uint8:
        raise ValueError(f"obs/image must use uint8; got {images.dtype}")
    if proprio.ndim != 2 or proprio.shape[1] != 8:
        raise ValueError(f"obs/proprio must have shape (N, 8); got {proprio.shape}")
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"action must have shape (N, 7); got {actions.shape}")
    if timestamps.ndim != 1:
        raise ValueError(f"timestamps must have shape (N,); got {timestamps.shape}")

    lengths = (len(images), len(proprio), len(actions), len(timestamps))
    if len(set(lengths)) != 1:
        raise ValueError(
            "obs/image, obs/proprio, action, and timestamps must contain the "
            f"same number of samples; got {lengths}"
        )
    if lengths[0] == 0:
        raise ValueError("episode must contain at least one sample")
    if not np.all(np.isfinite(timestamps)):
        raise ValueError("timestamps must contain only finite values")
    if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be strictly increasing")


def _joint_velocities(joint_positions_deg, timestamps):
    """Return timestamp-aware centered joint velocities in radians/second."""
    n = len(joint_positions_deg)
    velocity_deg = np.zeros_like(joint_positions_deg, dtype=np.float64)
    if n < 2:
        return velocity_deg

    delta_deg = np.diff(joint_positions_deg, axis=0)
    delta_deg = (delta_deg + 180.0) % 360.0 - 180.0
    delta_t = np.diff(timestamps)
    segment_velocity = delta_deg / delta_t[:, None]

    velocity_deg[0] = segment_velocity[0]
    velocity_deg[-1] = segment_velocity[-1]
    if n > 2:
        velocity_deg[1:-1] = (
            delta_deg[:-1] + delta_deg[1:]
        ) / (delta_t[:-1] + delta_t[1:])[:, None]
    return np.deg2rad(velocity_deg)


def load_episode_h5(h5_path):
    """Load and validate one current-format RGB Kinova episode."""
    path = Path(h5_path)
    with h5py.File(path, "r") as f:
        for dataset in REQUIRED_DATASETS:
            if dataset not in f:
                raise ValueError(f"missing required HDF5 dataset: {dataset}")

        images = f["obs/image"][:]
        proprio = f["obs/proprio"][:]
        actions = f["action"][:]
        timestamps = f["timestamps"][:]
        camera_fps_attr = f.attrs.get("camera_fps")
        intrinsic_values = {name: f.attrs.get(name) for name in CAMERA_INTRINSIC_ATTRS}
        exposure_seconds = f.attrs.get("exposure_seconds")
        camera_backend = f.attrs.get("camera_backend")
        camera_intrinsics_source = f.attrs.get("camera_intrinsics_source")

    _validate_episode_arrays(images, proprio, actions, timestamps)

    joint_positions = np.deg2rad(proprio[:, :7].astype(np.float64, copy=False))
    joint_velocities = _joint_velocities(proprio[:, :7], timestamps)
    camera_fps_declared = (
        None if camera_fps_attr is None else float(camera_fps_attr)
    )
    if len(timestamps) > 1:
        camera_fps = float(1.0 / np.median(np.diff(timestamps)))
    else:
        camera_fps = camera_fps_declared or 0.0

    present_intrinsics = [value is not None for value in intrinsic_values.values()]
    if any(present_intrinsics) and not all(present_intrinsics):
        raise ValueError("H5 camera intrinsics must contain fx, fy, cx, and cy together")
    camera_intrinsics = None
    if all(present_intrinsics):
        camera_intrinsics = {
            name: float(value) for name, value in intrinsic_values.items()
        }
        if not all(np.isfinite(list(camera_intrinsics.values()))):
            raise ValueError("H5 camera intrinsics must be finite")
        if camera_intrinsics["fx"] <= 0 or camera_intrinsics["fy"] <= 0:
            raise ValueError("H5 fx and fy must be positive")
    if exposure_seconds is not None:
        exposure_seconds = float(exposure_seconds)
        if not np.isfinite(exposure_seconds) or exposure_seconds <= 0:
            raise ValueError("H5 exposure_seconds must be finite and positive")
    if isinstance(camera_backend, bytes):
        camera_backend = camera_backend.decode("utf-8")
    if isinstance(camera_intrinsics_source, bytes):
        camera_intrinsics_source = camera_intrinsics_source.decode("utf-8")

    n, height, width, _ = images.shape
    print(
        f"  [h5] RGB episode: {n} frames, {width}x{height}, "
        f"actual {camera_fps:g} fps"
    )
    return {
        "images": images,
        "proprio": proprio,
        "actions": actions,
        "timestamps": timestamps,
        "joint_positions": joint_positions,
        "joint_velocities": joint_velocities,
        "num_frames": n,
        "H": height,
        "W": width,
        "camera_fps": camera_fps,
        "camera_fps_declared": camera_fps_declared,
        "camera_intrinsics": camera_intrinsics,
        "exposure_seconds": exposure_seconds,
        "camera_backend": camera_backend,
        "camera_intrinsics_source": camera_intrinsics_source,
    }


class EpisodeFrameReader:
    """Read RGB frames from a validated current-format episode array."""

    def __init__(self, images):
        images = np.asarray(images)
        if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
            raise ValueError(
                "EpisodeFrameReader requires uint8 RGB images with shape "
                f"(N, H, W, 3); got {images.shape} {images.dtype}"
            )
        self.images = images
        self.N = len(images)

    def read_frame(self, idx):
        if idx < 0 or idx >= self.N:
            return None
        return self.images[idx]

    def close(self):
        pass
