#!/usr/bin/env python3
"""对 episode_0002.h5 用 RL 迭代反卷积（对比 Wiener）。"""
import sys, time
from pathlib import Path

root = Path(__file__).parent
sys.path.insert(0, str(root))

import cv2
from robot_configs import KINOVA_GEN3, KINOVA_GEN3_HAND_EYE
from h5_loader import load_episode_h5, EpisodeFrameReader
from main import process_frame, save_deblur_result, setup_output_dirs

h5_path = root / "episode_0002.h5"

for method, label, out_name in [
    ("rl",     "RL(iters=30)",  "deblur_kinova_rl"),
    ("wiener", "Wiener(K=0.01)","deblur_kinova_wiener"),
]:
    meta = load_episode_h5(str(h5_path))
    reader = EpisodeFrameReader(meta["rgb_bytes"])
    sync = meta["sync_indices"]
    N = meta["num_frames"]
    H, W = meta["H"], meta["W"]
    out_dir = root / out_name

    vw, cw = setup_output_dirs(out_dir, W, H, fps=15.0)

    params = dict(fx=733.37, fy=733.37, depth=0.5, exposure_time=0.03,
                  method=method, K=0.01, rl_iters=30)

    print(f"\n{'='*50}")
    print(f"  Method: {label}")
    print(f"  Output: {out_dir}/")
    print(f"{'='*50}")

    st = time.time()
    for i in range(N):
        bgr = reader.read_frame(i)
        if bgr is None: continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if gray.std() < 2: continue

        ri = sync[i]
        q = meta["joint_positions"][ri]
        qd = meta["joint_velocities"][ri]

        deb, meta_info = process_frame(
            gray, q, qd, params,
            KINOVA_GEN3_HAND_EYE, KINOVA_GEN3,
        )
        save_deblur_result(out_dir, i, gray, deb, meta_info, cw, vw)

        if i % 30 == 0:
            fps = (i + 1) / (time.time() - st)
            print(f"  [{i:4d}/{N}]  {fps:.1f}fps")

    vw.release(); cw.release(); reader.close()
    print(f"  ✅ {label}: {N} frames in {time.time()-st:.1f}s")
