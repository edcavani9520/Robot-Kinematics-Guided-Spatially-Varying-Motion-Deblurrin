""")
main.py - 主函数：逐帧去模糊pipeline
=========================================
功能：
  1. 从视频/图片/h5读取帧 + 同步关节角数据
  2. 逐帧从关节角计算PSF → 去模糊（支持全局/空间变化PSF）
  3. 保存结果（含对比图、PSF报告）

输入方式：
  --video blurry.mp4 --joints joints.csv
  --frames ./frames/ --joints actions.csv
  --h5 episode_0004.h5

输出结果：
  output/
  ├── blurred/          原始模糊帧
  ├── deblurred/        去模糊帧
  ├── comparison/       左右对比图(blurred | deblurred)
  ├── deblurred_video.mp4
  ├── comparison_video.mp4
  └── psf_report.csv    每帧PSF参数
"""

import numpy as np
import cv2
import os
import sys
import csv
import time
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joint_deblur import (
    compute_psf, compute_psf_map, compute_psf_from_pose,
    wiener_deconvolution, richardson_lucy, tv_deconv,
    spatial_wiener_deconvolution, spatial_richardson_lucy,
    PANDA, PANDA_HAND_EYE_SIMPLE, euler_zyx_to_rotmat,
    get_configs,
)
from h5_loader import detect_h5_format, load_episode_h5, load_droid_h5, \
    EpisodeFrameReader, DroidFrameReader, load_kinova_h5, KinovaFrameReader
from evaluate import full_evaluate
from robot_configs import get_robot
from robot_configs import HAND_EYE_CONFIGS

# ============================================================
# 核心去模糊
# ============================================================
def save_deblur_result(output_dir, frame_idx, gray, deblurred, psf_meta,
                       comp_writer=None, vid_writer=None):
    """
    保存单帧去模糊结果
    psf_meta: 格式同process_frame返回值 (mode, psf, du, dv) 或 (mode, psf_map, ...)
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

    # 生成对比图
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

    # 写入视频（无标签）
    if vid_writer is not None:
        vid_writer.write(deblurred)
    if comp_writer is not None:
        comp = np.hstack([gray_color, deb_color])
        comp_writer.write(comp)

def setup_output_dirs(output_dir, W, H, fps=15.0):
    """创建输出目录和视频写入器"""
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
# CSV模式：视频/图片帧处理
# ============================================================
def run_h5_pipeline(h5_path, episode_dir, output_dir,
                     hand_eye="simple", camera_serial=None,
                     fx=733.37, fy=733.37, depth=0.5, exposure=0.03,
                     method="wiener", K=0.01, rl_iters=30, tv_lam=0.002,
                     max_frames=None, use_obs_joint=False,
                     robot_name="panda",
                     psf_sigma=0.0, adaptive_k=False,
):
    output_dir = Path(output_dir)
    params = dict(fx=fx, fy=fy, depth=depth, exposure_time=exposure,
                  method=method, K=K, rl_iters=rl_iters, tv_lam=tv_lam,
                  psf_sigma=0, adaptive_k=False)

    print("=" * 60)
    print("  运动模糊去除流水线 (H5模式)")
    print("=" * 60)
    print(f"  H5文件: {h5_path}")
    print(f"  输出目录: {output_dir}")
    print(f"  手眼标定: {hand_eye}")
    print(f"  机器人: {robot_name}")
    robot = get_robot(robot_name)
    print(f"  去模糊方法: {method}", end="")
    if method == "wiener":
        print(f" (K={K})")
    else:
        print(f" (迭代次数={rl_iters})")
    print(f"  相机参数: fx={fx:.1f}, fy={fy:.1f}, 深度={depth}m, 曝光时间={exposure}s")
    print()

    # 检测H5格式
    fmt = detect_h5_format(h5_path)
    print(f"[1/4] 检测h5格式: {fmt}")
    if fmt == "unknown":
        raise RuntimeError(f"无法识别的h5格式: {h5_path}")

    # 读取数据
    print("[2/4] 读取h5数据...")
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
    elif fmt == "kinova":
        meta = load_kinova_h5(h5_path)
        frame_reader = KinovaFrameReader(meta["images"])
        joint_pos = meta["joint_positions"]
        joint_vel = meta["joint_velocities"]
        sync_indices = np.arange(meta["num_frames"])
        num_frames = meta["num_frames"]
        H, W = meta["H"], meta["W"]
        video_fps = float(meta.get("camera_fps", 10.0))
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

    hand_eye_params = HAND_EYE_CONFIGS[hand_eye]

    # 创建输出目录
    print("[3/4] 创建输出目录...")
    vid_writer, comp_writer = setup_output_dirs(output_dir, W, H, video_fps)

    # 逐帧去模糊
    print("[4/4] 开始去模糊处理...")
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
                print(f"  [警告] 第{i}帧：无法读取")
                continue

            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            if gray.std() < 2:
                continue

            ri = sync_indices[i] if fmt == "episode" else i
            # 用 h5 真值 (tool_pose + tool_twist) 代替 FK+Jacobian
            p = meta["tool_pose"][ri]
            t = meta["tool_twist"][ri]
            R_ee_depth = euler_zyx_to_rotmat(p[3:])
            if hand_eye_params is not None:
                cam_pos = p[:3] + R_ee_depth @ hand_eye_params.t
                R_cam = R_ee_depth @ hand_eye_params.R
            else:
                cam_pos = p[:3]
                R_cam = R_ee_depth
            opt_axis = R_cam @ np.array([0, 0, 1])
            opt_z = opt_axis[2]
            depth_val = max(abs((0.0 - cam_pos[2]) / max(abs(opt_z), 0.01)), 0.02)
            psf, (du, dv) = compute_psf_from_pose(
                p, t, depth_val,
                fx=params.get("fx", 733.37), fy=params.get("fy", 733.37),
                cx=gray.shape[1] // 2, cy=gray.shape[0] // 2,
                exposure_time=params["exposure_time"],
                hand_eye=hand_eye_params)
            # PSF regularization
            if params.get("psf_sigma", 0) > 0:
                from scipy.ndimage import gaussian_filter
                psf = gaussian_filter(psf, sigma=params["psf_sigma"])
                psf /= psf.sum()
            K_eff = params["K"]
            if params.get("adaptive_k", False):
                K_eff = params["K"] * (psf.shape[0] / 17.0)
            if params.get("method") == "tv":
                deblurred = tv_deconv(gray, psf, lam=params.get("tv_lam", 0.04))
            else:
                deblurred = wiener_deconvolution(gray, psf, K=K_eff)
            meta_frame = ("global", psf, du, dv)

            save_deblur_result(output_dir, i, gray, deblurred, meta_frame,
                               comp_writer, vid_writer)

            mode = meta_frame[0]
            if mode == "global":
                _, psf, du, dv = meta_frame
            else:
                _, _, du_grid, dv_grid = meta_frame
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
                line = f"  [{i}/{limit}]  机器人索引={ri}"
                if mode == "global":
                    line += f"  du={du:.2f} dv={dv:.2f}  psf={psf.shape[0]}×{psf.shape[1]}"
                else:
                    line += f"  空间PSF |du|_avg={du:.2f}"
                line += f"  {fps:.1f}fps"
                print(line)

    frame_reader.close()
    vid_writer.release()
    comp_writer.release()

    total_time = time.time() - start_time
    print()
    print(f"  [OK] 处理完成！共处理 {limit} 帧，耗时 {total_time:.1f}s ({limit/total_time:.1f}fps)")
    print(f"  [DIR] 输出文件目录: {output_dir}/")
    print(f"     ├── blurred/          - 原始模糊帧")
    print(f"     ├── deblurred/        - 去模糊帧")
    print(f"     ├── comparison/       - 左右对比图")
    print(f"     ├── deblurred_video.mp4")
    print(f"     ├── comparison_video.mp4")
    print(f"     └── psf_report.csv")

# ============================================================
# 命令行参数解析
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="运动学引导的运动模糊去除工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
输入方式（三选一）：
""")

    # 输入源互斥参数
    parser.add_argument("--h5", type=str, required=True, help="H5 data file path")
    parser.add_argument("--episode-dir", type=str, default=None,
                        help="episode目录（DROID h5模式专用）")
    parser.add_argument("--output", type=str, default="deblur_output",
                        help="结果输出目录")
    parser.add_argument("--camera", type=str, default=None)
    parser.add_argument("--robot", type=str, default="panda",
                    help="机器人配置 (panda, kinova-gen3)")
    parser.add_argument("--hand-eye", type=str, default="simple",
                        choices=["simple", "droid-left", "droid-right", "kinova-gen3"],
                        help="手眼标定预设")
    parser.add_argument("--fx", type=float, default=733.37,
                        help="相机x方向焦距")
    parser.add_argument("--fy", type=float, default=733.37,
                        help="相机y方向焦距")
    parser.add_argument("--depth", type=float, default=0.5,
                        help="目标深度（米）")
    parser.add_argument("--exposure", type=float, default=0.03,
                        help="曝光时间（秒）")

    parser.add_argument("--psf-sigma", type=float, default=0.0,
                        help="PSF gaussian regularization sigma (0=off)")
    parser.add_argument("--adaptive-k", action="store_true",
                        help="Adaptive K scaling by PSF size")

    parser.add_argument("--method", choices=["wiener", "rl", "tv"], default="wiener",
                        help="反卷积算法")
    parser.add_argument("--K", type=float, default=0.01,
                         help="维纳滤波参数K（值越小去模糊越强）")
    parser.add_argument("--rl-iters", type=int, default=30,
                       help="RL算法迭代次数")
    parser.add_argument("--tv-lam", type=float, default=0.04,
                         help="全变分正则化强度（值越大去模糊越强）")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="最大处理帧数")

    # 空间PSF参数
    parser.add_argument("--spatial", action="store_true",
                        help="启用空间变化PSF（默认：全局统一PSF）")
    parser.add_argument("--grid-rows", type=int, default=4,
                        help="PSF网格行数")
    parser.add_argument("--grid-cols", type=int, default=4,
                        help="PSF网格列数")
    parser.add_argument("--overlap", type=float, default=0.25,
                        help="空间反卷积分块重叠比例")


    args = parser.parse_args()



    # 执行主逻辑
    if args.h5:
        run_h5_pipeline(
            h5_path=args.h5,
            episode_dir=args.episode_dir or str(Path(args.h5).parent),
            output_dir=args.output,
            hand_eye=args.hand_eye,
            camera_serial=args.camera,
            fx=args.fx, fy=args.fy,
            depth=args.depth, exposure=args.exposure,
            method=args.method, K=args.K, rl_iters=args.rl_iters, tv_lam=args.tv_lam,
            max_frames=args.max_frames,
            robot_name=args.robot,
            psf_sigma=args.psf_sigma,
            adaptive_k=args.adaptive_k)
if __name__ == "__main__":
    main()
