"""
csv_loader.py -- CSV joint-angle data loader
============================================
"""

import csv
import numpy as np
from pathlib import Path

from joint_deblur import (
    PANDA_HAND_EYE_SIMPLE,
    DROID_HAND_EYE_LEFT,
    DROID_HAND_EYE_RIGHT,
)

HAND_EYE_MAP = {
    "simple": PANDA_HAND_EYE_SIMPLE,
    "droid-left": DROID_HAND_EYE_LEFT,
    "droid-right": DROID_HAND_EYE_RIGHT,
}


def load_joint_csv(csv_path):
    """Load standard joint-angle CSV with columns: timestamp, q1..q7, qd1..qd7"""
    timestamps, q_list, qd_list = [], [], []
    with open(csv_path, "r") as f:
        next(f, None)
        for row in csv.reader(f):
            if len(row) < 15:
                continue
            try:
                t = float(row[0])
                q = np.array([float(row[i]) for i in range(1, 8)])
                qd = np.array([float(row[i]) for i in range(8, 15)])
                timestamps.append(t)
                q_list.append(q)
                qd_list.append(qd)
            except (ValueError, IndexError):
                continue
    if not timestamps:
        raise ValueError(f"No valid joint data in {csv_path}")
    print(f"Loaded {len(timestamps)} joint states from {csv_path}")
    return np.array(timestamps), np.array(q_list), np.array(qd_list)


def load_droid_actions_csv(csv_path):
    """Load DROID actions.csv: action_joint_0..6, finite-difference velocity"""
    timestamps, q_list = [], []
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            q_list.append([float(row[f"action_joint_{i}"]) for i in range(7)])
            timestamps.append(float(row["timestamp_ms"]) / 1000.0)
    q = np.array(q_list)
    t = np.array(timestamps)
    qd = np.zeros_like(q)
    dt = np.diff(t)
    for i in range(7):
        qd[:-1, i] = np.diff(q[:, i]) / np.maximum(dt, 1e-6)
        qd[-1, i] = qd[-2, i]
    print(f"Loaded {len(t)} DROID action frames from {csv_path}")
    return t, q, qd


def load_joints_auto(csv_path):
    """Detect CSV format by reading the header, dispatch accordingly."""
    with open(csv_path, "r") as f:
        header = f.readline().strip().lower()
    if "action_joint" in header:
        return load_droid_actions_csv(csv_path)
    return load_joint_csv(csv_path)


def find_nearest_joint(frame_t, joint_ts, q_all, qd_all, max_dt=0.1):
    """Find nearest joint state for a given frame timestamp."""
    idx = np.argmin(np.abs(joint_ts - frame_t))
    dt = abs(joint_ts[idx] - frame_t)
    if dt > max_dt:
        print(f"  [WARN] Frame-joint time skew {dt:.3f}s exceeds {max_dt}s threshold")
    return q_all[idx], qd_all[idx], joint_ts[idx]
