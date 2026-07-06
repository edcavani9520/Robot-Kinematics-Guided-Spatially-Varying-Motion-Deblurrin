"""
main.py — 主函数：逐帧去模糊 pipeline
=========================================
功能：
  1. 从视频/图片/h5 读取帧 + 同步关节角数据
  2. 逐帧从关节角算 PSF → 去模糊（支持全局/空间变化 PSF）
  3. 保存结果（含对比图、PSF 报告）

输入方式：
  --video blurry.mp4 --joints joints.csv
  --frames ./frames/ --joints actions.csv
  --h5 episode_0004.h5

输出结构：
  output/
  ├── blurred/          原始模糊帧
  ├── deblurred/        去模糊帧
  ├── comparison/       左右对比图 (blurred | deblurred)
  ├── deblurred_video.mp4
  ├── comparison_video.mp4
  └── psf_report.csv    每帧 PSF 参数
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
    get_configs,
)
from h5_loader import detect_h5_format, load_episode_h5, load_droid_h5, \
    EpisodeFrameReader, DroidFrameReader
from evaluate import evaluate
from csv_loader import load_joints_auto, find_nearest_joint, HAND_EYE_MAP

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


# ============================================================
# 结果保存（统一输出结构）
# ============================================================

def save_deblur_result(output_dir, frame_idx, gray, deblurred, psf_meta,
                       comp_writer=None, vid_writer=None):
    """
    保存单帧去模糊结果。
    psf_meta: like process_frame return (mode, psf, du, dv) or (mode, psf_map, ...)
    """
    blurred_dir = output_dir / "blurred"
    deblurred_dir = output_dir / "deblurred"
    comparison_dir = output_dir / "comparison"

    mode = psf_meta[0]
    if mode == "global":
        _, psf, du, dv = psf_meta
        psf_size = max(2 * int(abs(du)), 2 * int(abs(dv))) + 1
        psf_size = max(psf_size, 3)
        label_info = f"du={du:.1f} dv={dv:.1f}  psf={psf_size}"
    else:
        _, _, du_grid, dv_grid = psf_meta
        du_mean = float(np.abs(du_grid).mean())
        label_info = f"spatial |du|_avg={du_mean:.1f}"

    cv2.imwrite(str(blurred_dir / f"step_{frame_idx:04d}.jpg"), gray)
    cv2.imwrite(str(deblurred_dir / f"step_{frame_idx:04d}.jpg"), deblurred)

    # 对比图
    h, w = gray.shape
    gray_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    deb_color = cv2.cvtColor(deblurred, cv2.COLOR_GRAY2BGR)
    label_h = 28
    canvas = np.zeros((h + label_h, w * 2, 3), dtype=np.uint8)
    canvas[label_h:, :w] = gray_color
    cv2.putText(canvas, f"Blurred (step {frame_idx})",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    canvas[label_h:, w:] = deb_color
    cv2.putText(canvas, f"Deblurred (step {frame_idx})  {label_info}",
                (w + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(str(comparison_dir / f"compare_{frame_idx:04d}.jpg"), canvas)

    # 写入视频（不加标注）
    if vid_writer is not None:
        vid_writer.write(deblurred)
    if comp_writer is not None:
        comp = np.hstack([gray_color, deb_color])
        comp_writer.write(comp)


def setup_output_dirs(output_dir, W, H, fps=15.0):
    """创建输出目录和 video writers"""
    output_dir = Path(output_dir)
    for d in ["blurred", "deblurred", "comparison"]:
        (output_dir / d).mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    dv = cv2.VideoWriter(str(output_dir / "deblurred_video.mp4"),
                         fourcc, fps, (W, H), isColor=False)
    cv = cv2.VideoWriter(str(output_dir / "comparison_video.mp4"),
                         fourcc, fps, (W * 2, H), isColor=True)
    return dv, cv


# ============================================================
# CSV 模式：视频 / 图片帧
# ============================================================

def run_deblur_pipeline(joint_csv, output_dir,
                         video_path=None, frames_dir=None,
                         ground_truth_path=None,
                         fx=500., fy=500., depth=0.5, exposure=0.03,
                         method="wiener", K=0.01, rl_iters=30,
                         max_frames=None, hand_eye=None, robot=None,
                         spatial=False, grid_rows=4, grid_cols=4, overlap=0.25):
    if robot is None:
        robot = PANDA
    if hand_eye is None:
        hand_eye = PANDA_HAND_EYE_SIMPLE

    output_dir = Path(output_dir)
    handeye_name = {v: k for k, v in HAND_EYE_MAP.items()}.get(hand_eye, "custom")

    params = dict(fx=fx, fy=fy, depth=depth, exposure_time=exposure,
                  method=method, K=K, rl_iters=rl_iters)

    print("=" * 60)
    print("  Motion Deblur Pipeline (CSV mode)")
    print("=" * 60)
    print(f"  Camera: fx={fx:.1f}, fy={fy:.1f}, depth={depth}m, exposure={exposure}s")
    print(f"  Hand-eye: {handeye_name}")
    print(f"  Spatial:  {'yes (%dx%d, overlap=%.2f)' % (grid_rows, grid_cols, overlap) if spatial else 'no (global PSF)'}")
    print(f"  Output:   {output_dir}")
    print()

    joint_ts, q_all, qd_all = load_joints_auto(joint_csv)

    if frames_dir:
        paths = sorted(Path(frames_dir).glob("*.jpg")) + \
                sorted(Path(frames_dir).glob("*.png"))
        if max_frames:
            paths = paths[:max_frames]
        print(f"Loaded {len(paths)} frames from {frames_dir}")
        _process_frame_list(paths, joint_ts, q_all, qd_all,
                            params, hand_eye, robot, output_dir,
                            spatial, grid_rows, grid_cols, overlap)
    elif video_path:
        _process_video(video_path, joint_ts, q_all, qd_all,
                       params, hand_eye, robot, output_dir,
                       ground_truth_path, max_frames,
                       spatial, grid_rows, grid_cols, overlap)
    else:
        raise ValueError("Provide --video or --frames")


def _process_frame_list(paths, joint_ts, q_all, qd_all,
                         params, hand_eye, robot, output_dir,
                         spatial=False, grid_rows=4, grid_cols=4, overlap=0.25):
    first = cv2.imread(str(paths[0]), cv2.IMREAD_GRAYSCALE)
    h, w = first.shape[:2]
    fps = 15.0

    vid_writer, comp_writer = setup_output_dirs(output_dir, w, h, fps)
    psf_csv = output_dir / "psf_report.csv"

    n = len(paths)
    save_interval = max(1, n // 10)
    start = time.time()

    with open(psf_csv, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["step", "mode", "du_mean", "psf_size",
                         "blurred_mean", "deblurred_mean",
                         "blurred_std", "deblurred_std"])

        for fi, p in enumerate(paths):
            frame = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if frame is None:
                continue
            q, qd, _ = find_nearest_joint(fi, joint_ts, q_all, qd_all)
            deblurred, meta = process_frame(
                frame, q, qd, params, hand_eye, robot,
                spatial=spatial, grid_rows=grid_rows, grid_cols=grid_cols, overlap=overlap)

            save_deblur_result(output_dir, fi, frame, deblurred, meta,
                               comp_writer, vid_writer)

            mode = meta[0]
            if mode == "global":
                _, psf, du, dv = meta
            else:
                _, _, du_grid, dv_grid = meta
                du = float(np.abs(du_grid).mean())
                psf = None
            writer.writerow([
                fi, mode, f"{du:.3f}", psf.shape[0] if psf is not None else 0,
                f"{frame.mean():.1f}", f"{deblurred.mean():.1f}",
                f"{frame.std():.1f}", f"{deblurred.std():.1f}",
            ])

            if fi % save_interval == 0:
                elapsed = time.time() - start
                line = f"  [{fi}/{n}] {p.name}"
                if mode == "global":
                    line += f" | du={du:.2f} dv={dv:.2f} | psf={psf.shape[0]}×{psf.shape[1]}"
                else:
                    line += f" | spatial {grid_rows}×{grid_cols} | |du|_avg={du:.2f}"
                line += f" | {(fi+1)/elapsed:.1f}fps"
                print(line)

    vid_writer.release()
    comp_writer.release()
    total = time.time() - start
    print(f"Done. {n} frames in {total:.1f}s ({n/total:.1f}fps)")


def _process_video(video_path, joint_ts, q_all, qd_all,
                    params, hand_eye, robot, output_dir,
                    gt_path=None, max_frames=None,
                    spatial=False, grid_rows=4, grid_cols=4, overlap=0.25):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {fps:.1f}fps, {w}×{h}, {total} frames")

    gt_cap = cv2.VideoCapture(gt_path) if gt_path else None
    vid_writer, comp_writer = setup_output_dirs(output_dir, w, h, fps)
    psf_csv = output_dir / "psf_report.csv"

    save_interval = max(1, total // 10) if total > 0 else 10
    frame_idx = 0
    start = time.time()

    with open(psf_csv, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["step", "mode", "du_mean", "psf_size",
                         "blurred_mean", "deblurred_mean",
                         "blurred_std", "deblurred_std"])

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            if gray.std() < 2:
                frame_idx += 1
                continue

            t = frame_idx / fps
            q, qd, _ = find_nearest_joint(t, joint_ts, q_all, qd_all)
            deblurred, meta = process_frame(
                gray, q, qd, params, hand_eye, robot,
                spatial=spatial, grid_rows=grid_rows, grid_cols=grid_cols, overlap=overlap)

            save_deblur_result(output_dir, frame_idx, gray, deblurred, meta,
                               comp_writer, vid_writer)

            mode = meta[0]
            if mode == "global":
                _, psf, du, dv = meta
            else:
                _, _, du_grid, dv_grid = meta
                du = float(np.abs(du_grid).mean())
                psf = None
            writer.writerow([
                frame_idx, mode, f"{du:.3f}", psf.shape[0] if psf is not None else 0,
                f"{gray.mean():.1f}", f"{deblurred.mean():.1f}",
                f"{gray.std():.1f}", f"{deblurred.std():.1f}",
            ])

            if frame_idx % save_interval == 0:
                elapsed = time.time() - start
                if mode == "global":
                    info = f"  [{frame_idx}] du={du:.1f} dv={dv:.1f} | {(frame_idx+1)/elapsed:.1f}fps"
                else:
                    info = f"  [{frame_idx}] spatial {grid_rows}x{grid_cols} | |du|_avg={du:.1f} | {(frame_idx+1)/elapsed:.1f}fps"
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
    vid_writer.release()
    comp_writer.release()
    elapsed = time.time() - start
    print(f"Done. {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f}fps)")


# ============================================================
# H5 模式
# ============================================================

def run_h5_pipeline(h5_path, episode_dir, output_dir,
                     hand_eye="simple", camera_serial=None,
                     fx=733.37, fy=733.37, depth=0.5, exposure=0.03,
                     method="wiener", K=0.01, rl_iters=30,
                     max_frames=None, use_obs_joint=False):
    output_dir = Path(output_dir)
    params = dict(fx=fx, fy=fy, depth=depth, exposure_time=exposure,
                  method=method, K=K, rl_iters=rl_iters)

    print("=" * 60)
    print("  Motion Deblur Pipeline (H5 mode)")
    print("=" * 60)
    print(f"  H5:         {h5_path}")
    print(f"  Output:     {output_dir}")
    print(f"  Hand-eye:   {hand_eye}")
    print(f"  Method:     {method}", end="")
    if method == "wiener":
        print(f" (K={K})")
    else:
        print(f" (iters={rl_iters})")
    print(f"  Camera:     fx={fx:.1f}, fy={fy:.1f}, depth={depth}m, exposure={exposure}s")
    print()

    # 检测格式
    fmt = detect_h5_format(h5_path)
    print(f"[1/4] 检测 h5 格式: {fmt}")
    if fmt == "unknown":
        raise RuntimeError(f"无法识别的 h5 格式: {h5_path}")

    # 读取数据
    print("[2/4] 读取 h5 数据...")
    if fmt == "episode":
        meta = load_episode_h5(h5_path)
        from h5_loader import EpisodeFrameReader as EFR
        frame_reader = EFR(meta["rgb_bytes"])
        joint_pos = meta["joint_positions"]
        joint_vel = meta["joint_velocities"]
        sync_indices = meta["sync_indices"]
        num_frames = meta["num_frames"]
        H, W = meta["H"], meta["W"]
        video_fps = 15.0
    else:  # droid
        if episode_dir is None:
            episode_dir = str(Path(h5_path).parent)
        meta = load_droid_h5(h5_path, episode_dir)
        if camera_serial is None:
            camera_serial = meta["camera_serials"][0]
        joint_pos = meta["obs_joint_positions"] if use_obs_joint else meta["joint_positions"]
        joint_vel = meta["obs_joint_velocities"] if use_obs_joint else meta["joint_velocities"]
        frame_reader = DroidFrameReader(
            meta["video_paths"][camera_serial],
            meta["video_fps"],
            meta["camera_captures"].get(camera_serial, np.arange(len(joint_pos))),
            None)
        sync_indices = np.arange(len(joint_pos))
        num_frames = meta["num_frames"]
        H, W = meta["H"], meta["W"]
        video_fps = meta["video_fps"]

    hand_eye_params = HAND_EYE_MAP[hand_eye]

    # 输出目录
    print("[3/4] 创建输出目录...")
    vid_writer, comp_writer = setup_output_dirs(output_dir, W, H, video_fps)

    # 逐帧去模糊
    print("[4/4] 开始去模糊...")
    print()

    limit = min(num_frames, max_frames) if max_frames else num_frames
    save_interval = max(1, limit // 20)
    start_time = time.time()

    psf_csv = output_dir / "psf_report.csv"
    with open(psf_csv, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["step", "robot_idx", "du", "dv", "psf_size",
                         "blurred_mean", "deblurred_mean",
                         "blurred_std", "deblurred_std"])

        for i in range(limit):
            frame_bgr = frame_reader.read_frame(i)
            if frame_bgr is None:
                print(f"  [WARN] step {i}: 无法读取帧")
                continue

            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            if gray.std() < 2:
                continue

            ri = sync_indices[i] if fmt == "episode" else i
            q = joint_pos[ri]
            qd = joint_vel[ri]

            deblurred, meta = process_frame(
                gray, q, qd, params, hand_eye_params, PANDA)

            save_deblur_result(output_dir, i, gray, deblurred, meta,
                               comp_writer, vid_writer)

            mode = meta[0]
            if mode == "global":
                _, psf, du, dv = meta
            else:
                _, _, du_grid, dv_grid = meta
                du = float(np.abs(du_grid).mean())
                dv = float(np.abs(dv_grid).mean())
                psf = None
            writer.writerow([
                i, ri,
                f"{du:.3f}", f"{dv:.3f}", psf.shape[0] if psf is not None else 0,
                f"{gray.mean():.1f}", f"{deblurred.mean():.1f}",
                f"{gray.std():.1f}", f"{deblurred.std():.1f}",
            ])

            if i % save_interval == 0:
                elapsed = time.time() - start_time
                fps = (i + 1) / elapsed if elapsed > 0 else 0
                line = f"  [{i}/{limit}]  ri={ri}"
                if mode == "global":
                    line += f"  du={du:.2f} dv={dv:.2f}  psf={psf.shape[0]}×{psf.shape[1]}"
                else:
                    line += f"  spatial |du|_avg={du:.2f}"
                line += f"  {fps:.1f}fps"
                print(line)

    frame_reader.close()
    vid_writer.release()
    comp_writer.release()

    total_time = time.time() - start_time
    print()
    print(f"  ✅ 完成! 处理 {limit} 帧, 耗时 {total_time:.1f}s ({limit/total_time:.1f}fps)")
    print(f"  📁 输出: {output_dir}/")
    print(f"     ├── blurred/          — 原始模糊帧")
    print(f"     ├── deblurred/        — 去模糊帧")
    print(f"     ├── comparison/       — 左右对比图")
    print(f"     ├── deblurred_video.mp4")
    print(f"     ├── comparison_video.mp4")
    print(f"     └── psf_report.csv")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Kinematics-Guided Motion Deblurring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
输入方式（三选一）:
  1. --video blurry.mp4 --joints joints.csv
  2. --frames ./frames/ --joints actions.csv
  3. --h5 episode_0004.h5

空间变化 PSF:
  --spatial         启用空间变化 PSF（默认关闭，使用全局单一 PSF）
  --grid-rows 4     网格行数
  --grid-cols 4     网格列数
  --overlap 0.25    分块重叠比例

示例:
  # 视频 + CSV（全局 PSF）
  python main.py --video blurry.mp4 --joints joints.csv

  # 图片目录 + 空间变化 PSF
  python main.py --frames ./frames/ --joints actions.csv --spatial

  # h5 文件
  python main.py --h5 episode_0004.h5
  python main.py --h5 trajectory.h5 --hand-eye droid-left
        """)

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=str, help="输入视频路径")
    src.add_argument("--frames", type=str, help="输入图片目录")
    src.add_argument("--h5", type=str, help="h5 文件路径（自动检测格式）")

    parser.add_argument("--joints", type=str,
                        help="关节角 CSV（--video / --frames 模式必填）")
    parser.add_argument("--episode-dir", type=str, default=None,
                        help="episode 目录（DROID h5 模式，自动找 recordings/MP4）")
    parser.add_argument("--output", type=str, default="deblur_output",
                        help="输出目录")
    parser.add_argument("--gt", type=str, default=None,
                        help="Ground truth 视频（仅 --video 模式）")

    parser.add_argument("--hand-eye", type=str, default="simple",
                        choices=["simple", "droid-left", "droid-right"],
                        help="手眼标定预设")
    parser.add_argument("--camera", type=str, default=None,
                        help="摄像头 serial（仅 DROID h5 格式）")
    parser.add_argument("--fx", type=float, default=733.37,
                        help="焦距 x")
    parser.add_argument("--fy", type=float, default=733.37,
                        help="焦距 y")
    parser.add_argument("--depth", type=float, default=0.5,
                        help="物距 (米)")
    parser.add_argument("--exposure", type=float, default=0.03,
                        help="曝光时间 (秒)")

    parser.add_argument("--method", choices=["wiener", "rl"], default="wiener",
                        help="反卷积方法")
    parser.add_argument("--K", type=float, default=0.01,
                        help="Wiener K (越小去模糊越强)")
    parser.add_argument("--rl-iters", type=int, default=30,
                        help="RL 迭代次数")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="限制处理帧数")
    parser.add_argument("--use-obs-joint", action="store_true",
                        help="h5 模式使用 observation 关节角")

    # 空间变化 PSF
    parser.add_argument("--spatial", action="store_true",
                        help="使用空间变化 PSF (默认: 全局单一 PSF)")
    parser.add_argument("--grid-rows", type=int, default=4,
                        help="PSF 地图网格行数")
    parser.add_argument("--grid-cols", type=int, default=4,
                        help="PSF 地图网格列数")
    parser.add_argument("--overlap", type=float, default=0.25,
                        help="空间反卷积分块重叠比例")

    args = parser.parse_args()

    # 校验
    if args.video or args.frames:
        if not args.joints:
            parser.error("--video / --frames 模式需要 --joints")

    # 执行
    if args.h5:
        run_h5_pipeline(
            h5_path=args.h5,
            episode_dir=args.episode_dir or str(Path(args.h5).parent),
            output_dir=args.output,
            hand_eye=args.hand_eye,
            camera_serial=args.camera,
            fx=args.fx, fy=args.fy,
            depth=args.depth, exposure=args.exposure,
            method=args.method, K=args.K, rl_iters=args.rl_iters,
            max_frames=args.max_frames,
            use_obs_joint=args.use_obs_joint,
        )
    else:
        hand_eye = HAND_EYE_MAP[args.hand_eye]
        run_deblur_pipeline(
            joint_csv=args.joints,
            output_dir=args.output,
            video_path=args.video,
            frames_dir=args.frames,
            ground_truth_path=args.gt,
            fx=args.fx, fy=args.fy,
            depth=args.depth, exposure=args.exposure,
            method=args.method, K=args.K, rl_iters=args.rl_iters,
            max_frames=args.max_frames,
            hand_eye=hand_eye, robot=PANDA,
            spatial=args.spatial,
            grid_rows=args.grid_rows, grid_cols=args.grid_cols,
            overlap=args.overlap,
        )


if __name__ == "__main__":
    main()
