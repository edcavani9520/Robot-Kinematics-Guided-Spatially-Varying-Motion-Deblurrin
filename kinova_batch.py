"""
kinova_batch.py — 批量评估脚本 (Kinova h5 格式)
============================================
对 h5 所有帧执行去模糊并输出全面的锐度/相似度统计。

用法:
  python kinova_batch.py --h5 episode_0001.h5 --robot kinova-gen3 --hand-eye kinova-gen3 --K 0.01
  python kinova_batch.py --h5 episode_0001.h5 --method tv --tv-lam 0.04
  python kinova_batch.py --h5 episode_0001.h5 --method rl --rl-iters 50 --max-frames 20
"""

import sys, os, time, zipfile
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from h5_loader import load_kinova_h5, KinovaFrameReader
from joint_deblur import (
    compute_psf_from_pose,
    wiener_deconvolution,
    tv_deconv,
    richardson_lucy,
    euler_zyx_to_rotmat,
)
from evaluate import full_evaluate
from robot_configs import get_robot, HAND_EYE_CONFIGS


def make_comparison(gray, deblurred, fi, ch, label=""):
    """左右对比图 (原图 | 去模糊)"""
    h, w = gray.shape
    label_h = 60
    canvas = np.ones((h + label_h, w * 2, 3), dtype=np.uint8) * 240
    gc = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    dc = cv2.cvtColor(deblurred, cv2.COLOR_GRAY2BGR)
    canvas[label_h:, :w] = gc
    canvas[label_h:, w:] = dc
    cv2.putText(canvas, f"Original (frame {fi})", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(canvas, f"Deblurred {label}", (w + 10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    info = f"lap change: {ch:+.2f}"
    cv2.putText(canvas, info, (10, h + label_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
    return canvas


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Kinova h5 批量去模糊评估"
    )
    parser.add_argument("--h5", default="episode_0001.h5",
                        help="h5 文件路径")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="限制处理帧数")
    parser.add_argument("--method", choices=["wiener", "rl", "tv"],
                        default="wiener", help="反卷积算法")
    parser.add_argument("--K", type=float, default=0.01,
                        help="Wiener 参数 (越小去模糊越强)")
    parser.add_argument("--rl-iters", type=int, default=30,
                        help="RL 迭代次数")
    parser.add_argument("--tv-lam", type=float, default=0.04,
                        help="TV 正则化强度")
    parser.add_argument("--depth", type=float, default=0.5,
                        help="目标深度 (米)")
    parser.add_argument("--exposure", type=float, default=0.03,
                        help="曝光时间 (秒)")
    parser.add_argument("--fx", type=float, default=733.37,
                        help="相机 x 方向焦距")
    parser.add_argument("--fy", type=float, default=733.37,
                        help="相机 y 方向焦距")
    parser.add_argument("--robot", default="kinova-gen3",
                        help="机器人配置 (panda, kinova-gen3)")
    parser.add_argument("--hand-eye", default="kinova-gen3",
                        help="手眼标定预设")
    parser.add_argument("--psf-sigma", type=float, default=0.0,
                        help="PSF 高斯正则化 sigma (0=关闭)")
    parser.add_argument("--adaptive-k", action="store_true",
                        help="根据 PSF 大小自适应缩放 K")
    parser.add_argument("--out-dir", default="batch_output",
                        help="输出目录")
    parser.add_argument("--save-failed", action="store_true",
                        default=True, help="保存变差的帧对比图")
    parser.add_argument("--no-save-failed", action="store_false",
                        dest="save_failed", help="不保存变差帧")
    args = parser.parse_args()

    # 解析路径
    h5_path = os.path.join(ROOT, args.h5) if not os.path.isabs(args.h5) else args.h5
    if not os.path.exists(h5_path):
        # 也尝试直接当成绝对路径
        h5_path = args.h5

    # ---- 加载数据 ----
    print(f"Loading {h5_path} ...")
    meta = load_kinova_h5(h5_path)
    reader = KinovaFrameReader(meta["images"])
    total_frames = meta["num_frames"]
    if args.max_frames:
        total_frames = min(total_frames, args.max_frames)
    print(f"Will process {total_frames} frames")
    print()

    robot = get_robot(args.robot)
    hand_eye = HAND_EYE_CONFIGS.get(args.hand_eye, None)

    # ---- 遍历帧 ----
    lap_before, lap_after = [], []
    psnr_vals, ssim_vals = [], []
    psnr_matched_vals, ssim_matched_vals = [], []
    ten_before, ten_after = [], []
    times_ms = []
    failed_info = []   # (frame_idx, before, after, change, psnr, ssim)
    failed_imgs = []   # (frame_idx, gray, deblurred)

    out_dir = os.path.join(ROOT, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
    failed_dir = os.path.join(out_dir, "failed_frames")

    t_start = time.time()

    for fi in range(total_frames):
        t0 = time.time()

        # 读取帧
        frame_bgr = reader.read_frame(fi)
        if frame_bgr is None:
            continue
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 关节数据 (kinova 格式: 帧和机器人数据 1:1 对应, 无需 sync)
        q = meta["joint_positions"][fi]
        qd = meta["joint_velocities"][fi]

        # 从 h5 获取 tool_pose + tool_twist 计算 PSF
        pose = meta["tool_pose"][fi]
        twist = meta["tool_twist"][fi]
        R_ee_depth = euler_zyx_to_rotmat(pose[3:])
        if hand_eye is not None:
            cam_pos = pose[:3] + R_ee_depth @ hand_eye.t
            R_cam = R_ee_depth @ hand_eye.R
        else:
            cam_pos = pose[:3]
            R_cam = R_ee_depth
        opt_axis = R_cam @ np.array([0, 0, 1])
        opt_z = opt_axis[2]
        depth_val = abs((0.0 - cam_pos[2]) / max(abs(opt_z), 0.01)) if abs(opt_z) > 0.01 else abs(pose[2])
        depth_val = max(depth_val, 0.02)

        psf, (du, dv) = compute_psf_from_pose(
            pose, twist, depth_val,
            fx=args.fx, fy=args.fy,
            cx=w // 2, cy=h // 2,
            exposure_time=args.exposure,
            hand_eye=hand_eye,
        )

        # PSF 高斯平滑正则化
        if args.psf_sigma > 0:
            from scipy.ndimage import gaussian_filter
            psf = gaussian_filter(psf, sigma=args.psf_sigma)
            psf /= psf.sum()

        # 自适应 K
        K_eff = args.K
        if args.adaptive_k:
            psz = psf.shape[0]
            K_eff = args.K * (1.0 + 0.3 * np.log2(max(psz, 3) / 17.0))

        # 去模糊
        if args.method == "tv":
            deblurred = tv_deconv(gray, psf, lam=args.tv_lam)
        elif args.method == "rl":
            deblurred = richardson_lucy(gray, psf, iterations=args.rl_iters)
        else:
            deblurred = wiener_deconvolution(gray, psf, K=K_eff)

        # 评估
        ev = full_evaluate(gray, deblurred)
        lap_before.append(ev["laplacian_before"])
        lap_after.append(ev["laplacian_after"])
        improved = ev["laplacian_improved"]
        psnr_vals.append(ev["PSNR_raw"])
        ssim_vals.append(ev["SSIM_raw"])
        psnr_matched_vals.append(ev["PSNR_matched"])
        ssim_matched_vals.append(ev["SSIM_matched"])
        ten_before.append(ev["tenengrad_before"])
        ten_after.append(ev["tenengrad_after"])

        if args.save_failed and not improved:
            failed_info.append((fi, ev["laplacian_before"], ev["laplacian_after"],
                                ev["laplacian_change"], ev["PSNR_raw"], ev["SSIM_raw"]))
            failed_imgs.append((fi, gray, deblurred))

        elapsed = (time.time() - t0) * 1000
        times_ms.append(elapsed)

        if (fi + 1) % 50 == 0 or (fi + 1) == total_frames:
            print(f"  [{fi+1}/{total_frames}]  ({elapsed:.0f}ms/frame, "
                  f"lap: {ev['laplacian_before']:.1f} -> {ev['laplacian_after']:.1f})")

    dt = time.time() - t_start
    lap_b = np.array(lap_before)
    lap_a = np.array(lap_after)
    changed = lap_a - lap_b
    improved_count = int(np.sum(changed > 0))
    worsened_count = int(np.sum(changed < 0))
    same_count = int(np.sum(changed == 0))
    ten_b = np.array(ten_before)
    ten_a = np.array(ten_after)
    ten_changed = ten_a - ten_b
    ten_improved = int(np.sum(ten_changed > 0))
    ten_worsened = int(np.sum(ten_changed < 0))

    # ---- 保存变差帧对比图 ----
    if args.save_failed and failed_info:
        os.makedirs(failed_dir, exist_ok=True)
        print(f"\n  Saving {len(failed_info)} failed-frame comparisons to {failed_dir} ...")

        for fi, gray, deblurred in failed_imgs:
            ch = next((x[3] for x in failed_info if x[0] == fi), 0)
            canvas = make_comparison(gray, deblurred, fi, ch, "(worsened)")
            img_path = os.path.join(failed_dir, f"frame_{fi:04d}_failed.png")
            cv2.imwrite(img_path, canvas)

        # 文本报告
        txt_path = os.path.join(failed_dir, "failed_report.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Failed frames: {len(failed_info)} / {total_frames}\n")
            f.write(f"Parameters: K={args.K}, exposure={args.exposure}, method={args.method}\n")
            f.write(f"{'Frame':>6} {'Lap_before':>12} {'Lap_after':>12} {'Change':>10} {'PSNR':>8} {'SSIM':>7}\n")
            f.write("-" * 60 + "\n")
            for fi, b, a, ch, p, s in failed_info:
                f.write(f"{fi:>6d} {b:>12.2f} {a:>12.2f} {ch:>+10.2f} {p:>8.2f} {s:>7.4f}\n")

        # 打包 zip
        zip_path = os.path.join(out_dir, "failed_frames.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in sorted(os.listdir(failed_dir)):
                zf.write(os.path.join(failed_dir, fname), arcname=fname)
        print(f"  Zipped: {zip_path}")

    # ---- 打印总结 ----
    print()
    print("=" * 62)
    print("  Batch Evaluation Summary")
    print("=" * 62)
    print(f"  Frames processed:       {total_frames}")
    print(f"  Total time:             {dt:.1f}s  ({np.mean(times_ms):.0f}ms/frame)")
    print(f"  Method:                 {args.method} (K={args.K}, exposure={args.exposure}s)")
    print()
    print("  [1] Laplacian Variance (sharpness - higher = sharper)")
    print(f"  {'':>12} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}")
    print(f"  {'Before':>12} {lap_b.mean():>10.2f} {lap_b.std():>10.2f} {lap_b.min():>10.2f} {lap_b.max():>10.2f}")
    print(f"  {'After':>12} {lap_a.mean():>10.2f} {lap_a.std():>10.2f} {lap_a.min():>10.2f} {lap_a.max():>10.2f}")
    print(f"  {'Change':>12} {changed.mean():>+10.2f} {changed.std():>10.2f} {changed.min():>+10.2f} {changed.max():>+10.2f}")
    print()
    avg_change = changed.mean()
    pct_improved = improved_count / total_frames * 100
    pct_worsened = worsened_count / total_frames * 100
    print(f"  Improved: {improved_count}/{total_frames}  ({pct_improved:.1f}%)")
    print(f"  Worsened: {worsened_count}/{total_frames}  ({pct_worsened:.1f}%)")
    print(f"  Same:     {same_count}/{total_frames}    ({100-pct_improved-pct_worsened:.1f}%)")
    if pct_improved > pct_worsened:
        print(f"  >>> OVERALL: Deblurring IS effective (sharpness improved {avg_change:+.2f})")
    else:
        print(f"  >>> OVERALL: Deblurring is NOT effective (sharpness changed {avg_change:+.2f})")
    print()
    ten_avg = ten_changed.mean()
    ten_pct_imp = ten_improved / total_frames * 100
    ten_pct_wor = ten_worsened / total_frames * 100
    print("  [2] Tenengrad (gradient-based sharpness - higher = sharper)")
    print(f"  {'':>12} {'mean':>14} {'std':>14} {'min':>14} {'max':>14}")
    print(f"  {'Before':>12} {ten_b.mean():>14.2e} {ten_b.std():>14.2e} {ten_b.min():>14.2e} {ten_b.max():>14.2e}")
    print(f"  {'After':>12} {ten_a.mean():>14.2e} {ten_a.std():>14.2e} {ten_a.min():>14.2e} {ten_a.max():>14.2e}")
    print(f"  {'Change':>12} {ten_changed.mean():>+14.2e} {ten_changed.std():>14.2e} {ten_changed.min():>+14.2e} {ten_changed.max():>+14.2e}")
    print()
    print(f"  Improved: {ten_improved}/{total_frames}  ({ten_pct_imp:.1f}%)")
    print(f"  Worsened: {ten_worsened}/{total_frames}  ({ten_pct_wor:.1f}%)")
    print(f"  Same:     {total_frames-ten_improved-ten_worsened}/{total_frames}    ({100-ten_pct_imp-ten_pct_wor:.1f}%)")
    if ten_pct_imp > ten_pct_wor:
        print(f"  >>> OVERALL: Tenengrad SHARPENED ({ten_avg:+.2e})")
    else:
        print(f"  >>> OVERALL: Tenengrad BLURRED ({ten_avg:+.2e})")
    print()
    print("  [3] Pairwise (Blurry vs Deblurred)")
    print(f"  {'':>12} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}")
    print(f"  {'PSNR':>12} {np.mean(psnr_vals):>10.2f} {np.std(psnr_vals):>10.2f} {np.min(psnr_vals):>10.2f} {np.max(psnr_vals):>10.2f} dB")
    print(f"  {'SSIM':>12} {np.mean(ssim_vals):>10.4f} {np.std(ssim_vals):>10.4f} {np.min(ssim_vals):>10.4f} {np.max(ssim_vals):>10.4f}")
    print()
    print("  [4] Pairwise after Histogram Matching")
    print(f"  {'':>12} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}")
    print(f"  {'PSNR_matched':>12} {np.mean(psnr_matched_vals):>10.2f} {np.std(psnr_matched_vals):>10.2f} {np.min(psnr_matched_vals):>10.2f} {np.max(psnr_matched_vals):>10.2f} dB")
    print(f"  {'SSIM_matched':>12} {np.mean(ssim_matched_vals):>10.4f} {np.std(ssim_matched_vals):>10.4f} {np.min(ssim_matched_vals):>10.4f} {np.max(ssim_matched_vals):>10.4f}")
    if args.save_failed and failed_info:
        print()
        print("  Failed-frame details saved to:")
        print(f"    Images:  {failed_dir}/")
        print(f"    Report:  {failed_dir}/failed_report.txt")
        print(f"    Zip:     {zip_path}")
    print("=" * 62)


if __name__ == "__main__":
    main()
