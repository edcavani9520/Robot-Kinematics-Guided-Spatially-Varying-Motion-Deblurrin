"""
main.py — 主函数：逐帧去模糊 pipeline
=========================================
功能：
  1. 读取视频/图片帧 + 同步关节角数据
  2. 逐帧从关节角算 PSF → 去模糊
  3. 保存结果

手腕相机模式（随机械臂动）：
  python main.py --video blurry.mp4 --joints joints.csv --hand-eye droid-left

用法示例：
  python main.py --video input.mp4 --joints joint_data.csv
  python main.py --frames ./frames/ --joints actions.csv --hand-eye droid-left
  python main.py --video blurry.mp4 --joints joints.csv --gt clean.mp4
"""

import numpy as np
import cv2
import os, sys, csv, time, argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joint_deblur import (
    compute_psf, compute_psf_map,
    wiener_deconvolution, richardson_lucy,
    spatial_wiener_deconvolution, spatial_richardson_lucy,
    PANDA, PANDA_HAND_EYE_SIMPLE,
    DROID_HAND_EYE_LEFT, DROID_HAND_EYE_RIGHT,
)
from evaluate import evaluate


# ============================================================
# 数据加载器
# ============================================================

_HAND_EYE_MAP = {
    "simple": PANDA_HAND_EYE_SIMPLE,
    "droid-left": DROID_HAND_EYE_LEFT,
    "droid-right": DROID_HAND_EYE_RIGHT,
}


def load_joint_csv(csv_path):
    """
    加载标准关节角 CSV。
    格式：timestamp, q1..q7, qd1..qd7
    """
    timestamps, q_list, qd_list = [], [], []
    with open(csv_path, "r") as f:
        next(f, None)  # skip header
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
    """
    加载 DROID actions.csv。
    提取 action_joint_0..6，用有限差分算速度。
    """
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
    print(f"  q range: [{q.min():.3f}, {q.max():.3f}]")
    return t, q, qd


def load_joints_auto(csv_path):
    """自动识别 CSV 格式并加载。"""
    with open(csv_path, "r") as f:
        header = f.readline().strip().lower()

    if "action_joint" in header:
        return load_droid_actions_csv(csv_path)
    return load_joint_csv(csv_path)


def find_nearest_joint(frame_t, joint_ts, q_all, qd_all, max_dt=0.1):
    idx = np.argmin(np.abs(joint_ts - frame_t))
    dt = abs(joint_ts[idx] - frame_t)
    if dt > max_dt:
        print(f"  [WARN] Frame-joint time skew {dt:.3f}s exceeds {max_dt}s threshold")
    return q_all[idx], qd_all[idx], joint_ts[idx]


# ============================================================
# 核心去模糊
# ============================================================

def process_frame(frame_gray, q, q_dot, params, hand_eye, robot,
                  spatial=False, grid_rows=4, grid_cols=4, overlap=0.25):
    h, w = frame_gray.shape

    if spatial:
        psf_map, (du_grid, dv_grid) = compute_psf_map(
            q, q_dot, depth=params["depth"],
            H=h, W=w,
            fx=params["fx"], fy=params["fy"],
            exposure_time=params["exposure_time"],
            grid_rows=grid_rows, grid_cols=grid_cols,
            hand_eye=hand_eye, robot=robot,
        )
        method = params["method"]
        if method == "wiener":
            deblurred = spatial_wiener_deconvolution(
                frame_gray, psf_map, grid_rows, grid_cols,
                K=params["K"], overlap=overlap)
        elif method == "rl":
            deblurred = spatial_richardson_lucy(
                frame_gray, psf_map, grid_rows, grid_cols,
                iterations=params["rl_iters"], overlap=overlap)
        else:
            raise ValueError(f"Unknown method: {method}")
        return deblurred, ("spatial", psf_map, du_grid, dv_grid)
    else:
        psf, (du, dv) = compute_psf(
            q, q_dot,
            depth=params["depth"],
            fx=params["fx"], fy=params["fy"],
            cx=w // 2, cy=h // 2,
            exposure_time=params["exposure_time"],
            hand_eye=hand_eye, robot=robot,
        )
        method = params["method"]
        if method == "wiener":
            deblurred = wiener_deconvolution(frame_gray, psf, K=params["K"])
        elif method == "rl":
            deblurred = richardson_lucy(frame_gray, psf, iterations=params["rl_iters"])
        else:
            raise ValueError(f"Unknown method: {method}")
        return deblurred, ("global", psf, du, dv)


def run_deblur_pipeline(joint_csv, output_dir,
                         video_path=None, frames_dir=None,
                         ground_truth_path=None,
                         fx=500., fy=500., depth=0.5, exposure=0.03,
                         method="wiener", K=0.01, rl_iters=30,
                         max_frames=None, hand_eye=None, robot=None,
                         spatial=False, grid_rows=4, grid_cols=4, overlap=0.25):
    """
    完整 pipeline：逐帧去模糊。
    
    输入: video_path（视频） 或 frames_dir（图片目录）
    每帧配准对应关节角数据 → 算 PSF → 去模糊 → 保存。
    """
    if robot is None:
        robot = PANDA
    if hand_eye is None:
        hand_eye = PANDA_HAND_EYE_SIMPLE

    os.makedirs(output_dir, exist_ok=True)

    params = dict(fx=fx, fy=fy, depth=depth, exposure_time=exposure,
                  method=method, K=K, rl_iters=rl_iters)

    handeye_name = {v: k for k, v in _HAND_EYE_MAP.items()}.get(hand_eye, "custom")
    print("=" * 60)
    print("  Motion Deblur Pipeline")
    print("=" * 60)
    print(f"  Camera: fx={fx:.1f}, fy={fy:.1f}, depth={depth}m, exposure={exposure}s")
    print(f"  Hand-eye: {handeye_name}")
    print(f"  Output:   {output_dir}")
    print()

    # ---- 加载关节角 ----
    joint_ts, q_all, qd_all = load_joints_auto(joint_csv)

    # ---- 获取帧列表 ----
    if frames_dir:
        paths = sorted(Path(frames_dir).glob("*.jpg")) + \
                sorted(Path(frames_dir).glob("*.png"))
        if max_frames:
            paths = paths[:max_frames]
        print(f"Loaded {len(paths)} frames from {frames_dir}")
        process_frames_from_list(paths, joint_ts, q_all, qd_all,
                                  params, hand_eye, robot, output_dir,
                                  spatial, grid_rows, grid_cols, overlap)
    elif video_path:
        process_video(video_path, joint_ts, q_all, qd_all,
                       params, hand_eye, robot, output_dir,
                       ground_truth_path, max_frames,
                       spatial, grid_rows, grid_cols, overlap)
    else:
        raise ValueError("Provide --video or --frames")


def process_frames_from_list(paths, joint_ts, q_all, qd_all,
                              params, hand_eye, robot, output_dir,
                              spatial=False, grid_rows=4, grid_cols=4, overlap=0.25):
    """处理图片列表（如 DROID 帧序列）。"""
    first = cv2.imread(str(paths[0]), cv2.IMREAD_GRAYSCALE)
    h, w = first.shape[:2]
    fps = 15.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(os.path.join(output_dir, "deblurred_video.mp4"),
                             fourcc, fps, (w, h), isColor=False)
    n = len(paths)
    save_interval = max(1, n // 10)
    start = time.time()

    for fi, p in enumerate(paths):
        frame = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            continue

        q, qd, matched_t = find_nearest_joint(fi, joint_ts, q_all, qd_all)
        deblurred, meta = process_frame(
            frame, q, qd, params, hand_eye, robot,
            spatial=spatial, grid_rows=grid_rows, grid_cols=grid_cols, overlap=overlap)

        out_name = "deblurred_" + p.stem + ".jpg"
        cv2.imwrite(os.path.join(output_dir, out_name), deblurred)
        writer.write(deblurred)

        if fi % save_interval == 0:
            elapsed = time.time() - start
            if meta[0] == "global":
                _, psf, du, dv = meta
                msg = "  [%d/%d] %s | du=%.2f dv=%.2f | %dx%d kernel | %.1ffps" % (
                    fi, n, p.name, du, dv, psf.shape[0], psf.shape[1], (fi+1)/elapsed)
                print(msg)
            else:
                _, psf_map, du_grid, dv_grid = meta
                msg = "  [%d/%d] %s | spatial %dx%d | |du|_max=%.2f | %.1ffps" % (
                    fi, n, p.name, grid_rows, grid_cols,
                    float(np.abs(du_grid).max()), (fi+1)/elapsed)
                print(msg)

    writer.release()
    total = time.time() - start
    print("Done. %d frames in %.1fs (%.1ffps)" % (n, total, n/total))

def process_video(video_path, joint_ts, q_all, qd_all,
                   params, hand_eye, robot, output_dir,
                   gt_path=None, max_frames=None,
                   spatial=False, grid_rows=4, grid_cols=4, overlap=0.25):
    """处理视频文件。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {fps:.1f}fps, {w}×{h}, {total} frames")

    gt_cap = cv2.VideoCapture(gt_path) if gt_path else None

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(os.path.join(output_dir, "deblurred_video.mp4"),
                             fourcc, fps, (w, h), isColor=False)

    save_interval = max(1, total // 10) if total > 0 else 10
    frame_idx = 0
    start = time.time()

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if gray.std() < 2:  # skip blank frame
            frame_idx += 1
            continue

        t = frame_idx / fps
        q, qd, _ = find_nearest_joint(t, joint_ts, q_all, qd_all)
        deblurred, meta = process_frame(
            gray, q, qd, params, hand_eye, robot,
            spatial=spatial, grid_rows=grid_rows, grid_cols=grid_cols, overlap=overlap)
        writer.write(deblurred)

        if frame_idx % save_interval == 0:
            elapsed = time.time() - start
            if meta[0] == "global":
                _, psf, du, dv = meta
                info = f"  [{frame_idx}] du={du:.1f} dv={dv:.1f} | {(frame_idx+1)/elapsed:.1f}fps"
            else:
                _, psf_map, du_grid, dv_grid = meta
                du_mean = float(np.abs(du_grid).mean())
                dv_mean = float(np.abs(dv_grid).mean())
                info = f"  [{frame_idx}] spatial {grid_rows}x{grid_cols} | |du|_avg={du_mean:.1f} | {(frame_idx+1)/elapsed:.1f}fps"
            if gt_cap:
                ret_gt, gt_bgr = gt_cap.read()
                if ret_gt:
                    gt_gray = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2GRAY)
                    m, _ = evaluate(gt_gray, deblurred)
                    m_b, _ = evaluate(gt_gray, gray)
                    info += f" | PSNR: blur={m_b['PSNR_raw']:.1f}→{m['PSNR_raw']:.1f}"
            print(info)

        frame_idx += 1
        if max_frames and frame_idx >= max_frames:
            break

    cap.release()
    if gt_cap:
        gt_cap.release()
    writer.release()
    elapsed = time.time() - start
    print(f"Done. {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f}fps)")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Kinematics-Guided Motion Deblurring")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=str, help="Input video path")
    group.add_argument("--frames", type=str, help="Input image directory")

    parser.add_argument("--joints", type=str, required=True,
                        help="Joint CSV (standard or DROID actions.csv)")
    parser.add_argument("--output", type=str, default="deblur_output",
                        help="Output directory")
    parser.add_argument("--gt", type=str, default=None,
                        help="Ground truth video (optional)")

    parser.add_argument("--hand-eye", type=str, default="simple",
                        choices=["simple", "droid-left", "droid-right"],
                        help="Hand-eye calibration preset")
    parser.add_argument("--fx", type=float, default=733.37,
                        help="Focal length x (default: 733.37 = DROID ZED)")
    parser.add_argument("--fy", type=float, default=733.37,
                        help="Focal length y (default: 733.37 = DROID ZED)")
    parser.add_argument("--depth", type=float, default=0.5,
                        help="Scene depth (m)")
    parser.add_argument("--exposure", type=float, default=0.03,
                        help="Exposure time (s)")

    parser.add_argument("--method", choices=["wiener", "rl"], default="wiener")
    parser.add_argument("--K", type=float, default=0.01,
                        help="Wiener K (small=strong)")
    parser.add_argument("--rl-iters", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=None)

    parser.add_argument("--spatial", action="store_true",
                        help="Use spatially-varying PSF (default: single global PSF)")
    parser.add_argument("--grid-rows", type=int, default=4,
                        help="PSF map grid rows (default: 4, only with --spatial)")
    parser.add_argument("--grid-cols", type=int, default=4,
                        help="PSF map grid cols (default: 4, only with --spatial)")
    parser.add_argument("--overlap", type=float, default=0.25,
                        help="Patch overlap ratio for spatial deconvolution (default: 0.25)")

    args = parser.parse_args()
    hand_eye = _HAND_EYE_MAP[args.hand_eye]

    run_deblur_pipeline(
        joint_csv=args.joints,
        output_dir=args.output,
        video_path=args.video,
        frames_dir=args.frames,
        ground_truth_path=args.gt,
        fx=args.fx, fy=args.fy, depth=args.depth, exposure=args.exposure,
        method=args.method, K=args.K, rl_iters=args.rl_iters,
        max_frames=args.max_frames,
        hand_eye=hand_eye, robot=PANDA,
        spatial=args.spatial,
        grid_rows=args.grid_rows, grid_cols=args.grid_cols,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()
