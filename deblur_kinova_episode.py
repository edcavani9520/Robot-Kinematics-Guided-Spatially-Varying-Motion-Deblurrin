#!/usr/bin/env python3
"""
对 episode_0002.h5 用 Kinova Gen3 本体 + Kinova 手眼标定做去模糊。
"""
import sys, time
from pathlib import Path

root = Path(__file__).parent
sys.path.insert(0, str(root))

import cv2
import numpy as np

from robot_configs import KINOVA_GEN3, KINOVA_GEN3_HAND_EYE
from h5_loader import detect_h5_format, load_episode_h5, EpisodeFrameReader
from main import process_frame, save_deblur_result, setup_output_dirs

# ── 路径 ──
h5_path = root / "episode_0002.h5"
out_dir = root / "deblur_kinova_output"

# ── 1. 加载数据 ──
meta = load_episode_h5(str(h5_path))
reader = EpisodeFrameReader(meta["rgb_bytes"])
sync = meta["sync_indices"]
N = meta["num_frames"]
H, W = meta["H"], meta["W"]

# ── 2. 输出目录 ──
vw, cw = setup_output_dirs(out_dir, W, H, fps=15.0)

# ── 3. 参数 ──
params = dict(
    fx=733.37, fy=733.37, depth=0.5, exposure_time=0.03,
    method="wiener", K=0.01, rl_iters=30,
)

# ── 4. 逐帧去模糊 ──
print(f"Robot:    Kinova Gen3")
print(f"Hand-eye: Kinova Gen3 (倾斜向下安装)")
print(f"Frames:   {N}")
print()

st = time.time()
for i in range(N):
    bgr = reader.read_frame(i)
    if bgr is None:
        continue
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if gray.std() < 2:
        continue

    ri = sync[i]
    q = meta["joint_positions"][ri]
    qd = meta["joint_velocities"][ri]

    deb, meta_info = process_frame(
        gray, q, qd, params,
        KINOVA_GEN3_HAND_EYE, KINOVA_GEN3,
    )

    save_deblur_result(out_dir, i, gray, deb, meta_info, cw, vw)

    if i % 20 == 0:
        fps = (i + 1) / (time.time() - st)
        print(f"  [{i:4d}/{N}]  {fps:.1f}fps")

vw.release()
cw.release()
reader.close()
elapsed = time.time() - st
print(f"\n✅ Done. {N} frames in {elapsed:.1f}s ({N/elapsed:.1f}fps)")
print(f"📁 Output: {out_dir}/")
