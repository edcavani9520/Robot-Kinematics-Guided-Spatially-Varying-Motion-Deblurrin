import sys, os, time
import numpy as np
import cv2

ROOT = r"E:\Vital_document\CUHKSZ\课程文件\ECE4512\Final"
sys.path.insert(0, ROOT)

from h5_loader import load_episode_h5, EpisodeFrameReader
from joint_deblur import compute_psf_from_pose, wiener_deconvolution, tv_deconv, richardson_lucy, euler_zyx_to_rotmat, get_camera_velocity
from evaluate import full_evaluate
from robot_configs import get_robot, HAND_EYE_CONFIGS


# --- evaluation judgment helpers ---
def _judge_lap(change): return "SHARPENED" if change > 5 else "BLURRED" if change < -5 else "NO CHANGE"
def _judge_ten(change): return "SHARPENED" if change > 1e6 else "BLURRED" if change < -1e6 else "NO CHANGE"
def _judge_tv_ratio(tv_b, tv_a):
    r = (tv_a + 1e-10) / (tv_b + 1e-10)
    return f"x{r:.2f} (LOW)" if r < 1.5 else f"x{r:.2f} (MOD)" if r < 3.0 else f"x{r:.2f} (HIGH)"
def _judge_edge(er): return "GOOD" if er > 0.5 else "FAIR" if er > 0.2 else "RINGING DOMINANT"
def _judge_psnr(p): return "STRONG CHANGE" if p < 20 else "MODERATE" if p < 35 else "LOW CHANGE"
def _judge_ssim(s): return "NEAR IDENTICAL" if s > 0.95 else "GOOD" if s > 0.85 else "DAMAGED"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", default="episode_0002.h5")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--show-frame", type=int, default=None,
                        help="frame index for detailed evaluation (default: middle frame)")
    
    parser.add_argument("--exposure", type=float, default=0.03)
    parser.add_argument("--K", type=float, default=0.01)
    parser.add_argument("--method", type=str, default="wiener",
                        help="Deconv method: wiener / tv / rl")
    parser.add_argument("--tv-lam", type=float, default=0.002,
                        help="TV regularization strength")
    parser.add_argument("--lam", type=float, default=0.002,
                        help="TV regularization strength")
    parser.add_argument("--rl-iters", type=int, default=30,
                        help="RL iterations")
    parser.add_argument("--psf-sigma", type=float, default=0.0,
                        help="PSF gaussian regularization sigma (0=off)")
    parser.add_argument("--adaptive-k", action="store_true",
                        help="Adaptive K scaling by PSF size")
    parser.add_argument("--depth", type=float, default=0.5,
                        help="Fixed depth (used when tool_pose unavailable)")

    parser.add_argument("--robot", default="kinova-gen3")
    parser.add_argument("--hand-eye", default="kinova-gen3")
    args = parser.parse_args()

    h5_path = os.path.join(ROOT, args.h5)
    print(f"Loading {h5_path} ...")
    meta = load_episode_h5(h5_path)
    reader = EpisodeFrameReader(meta["rgb_bytes"])
    sync = meta["sync_indices"]
    total_frames = len(sync)
    if args.max_frames:
        total_frames = min(total_frames, args.max_frames)
    print(f"Will process {total_frames} frames")

    robot = get_robot(args.robot)
    hand_eye = HAND_EYE_CONFIGS.get(args.hand_eye, None)

    lap_before = []
    lap_after = []
    psnr_vals = []
    ssim_vals = []
    psnr_matched_vals = []
    ssim_matched_vals = []
    ten_before = []
    ten_after = []
    tv_before = []
    tv_after = []
    edge_ratios = []
    times_ms = []
    t_start = time.time()

    for fi in range(total_frames):
        t0 = time.time()

        frame_bgr = reader.read_frame(fi)
        if frame_bgr is None:
            continue
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        ri = sync[fi] if fi < len(sync) else fi
        q = meta["joint_positions"][ri]
        qd = meta["joint_velocities"][ri]

        if meta["tool_twist"] is not None:
            pose = meta["tool_pose"][ri]
            twist = meta["tool_twist"][ri]
            R_ee_depth = euler_zyx_to_rotmat(pose[3:])
            if hand_eye is not None:
                cam_pos = pose[:3] + R_ee_depth @ hand_eye.t
                R_cam = R_ee_depth @ hand_eye.R
            else:
                cam_pos = pose[:3]
                R_cam = R_ee_depth
            opt_axis = R_cam @ np.array([0, 0, 1])
            opt_z = opt_axis[2]
            depth_val = max(abs((0.0 - cam_pos[2]) / max(abs(opt_z), 0.01)), 0.02)
            psf, (du, dv) = compute_psf_from_pose(
                pose, twist, depth_val,
                fx=733.37, fy=733.37,
                cx=w // 2, cy=h // 2,
                exposure_time=args.exposure,
                hand_eye=hand_eye
            )
        else:
            # === 旧格式: 无 tool_twist，从关节角计算 ===
            q = meta["joint_positions"][ri]
            qd = meta["joint_velocities"][ri]
            v_cam_6d = get_camera_velocity(q, qd, hand_eye=hand_eye, robot=robot)
            depth_val = args.depth
            psf, (du, dv) = compute_psf_from_pose(
                depth=depth_val, fx=733.37, fy=733.37,
                cx=w // 2, cy=h // 2,
                exposure_time=args.exposure,
                v_cam_6d=v_cam_6d
            )
        # PSF regularization (gaussian smoothing)
        if args.psf_sigma > 0:
            from scipy.ndimage import gaussian_filter
            psf = gaussian_filter(psf, sigma=args.psf_sigma)
            psf /= psf.sum()
        # Adaptive K scaling
        K_eff = args.K
        if args.adaptive_k:
            psz = psf.shape[0]
            K_eff = args.K * (1.0 + 0.3 * np.log2(max(psz, 3) / 17.0))
        if args.method == "tv":
            deblurred = tv_deconv(gray, psf, lam=args.tv_lam)
        elif args.method == "rl":
            deblurred = richardson_lucy(gray, psf, iterations=args.rl_iters)
        else:
            deblurred = wiener_deconvolution(gray, psf, K=K_eff)

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
        tv_before.append(ev["tv_before"])
        tv_after.append(ev["tv_after"])
        edge_ratios.append(ev["edge_ratio"])


        elapsed = (time.time() - t0) * 1000
        times_ms.append(elapsed)

        if (fi + 1) % 50 == 0:
            print(f"  {fi+1}/{total_frames}  ({elapsed:.0f}ms/frame)")

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

    dt = time.time() - t_start
    lap_b = np.array(lap_before)
    lap_a = np.array(lap_after)
    lap_ch = lap_a - lap_b
    lap_imp = int(np.sum(lap_ch > 0))
    lap_wor = int(np.sum(lap_ch < 0))
    ten_b = np.array(ten_before)
    ten_a = np.array(ten_after)
    ten_ch = ten_a - ten_b
    ten_imp = int(np.sum(ten_ch > 0))
    ten_wor = int(np.sum(ten_ch < 0))
    tv_b_arr = np.array(tv_before)
    tv_a_arr = np.array(tv_after)
    tv_ch_arr = tv_a_arr - tv_b_arr
    tv_ratios = tv_a_arr / (tv_b_arr + 1e-10)
    er_arr = np.array(edge_ratios)

    print()
    print("=" * 86)
    print("  Batch Evaluation Summary")
    print("=" * 86)
    print(f"  Episode: {os.path.basename(args.h5)}  Method: {args.method.upper()}")
    k_str = f"K={args.K}" if args.method == "wiener" else f"lam={args.tv_lam}" if args.method == "tv" else f"iters={args.rl_iters}"
    print(f"  Params: {k_str}  Frames: {total_frames}  Time: {dt:.1f}s ({np.mean(times_ms):.0f}ms/frame)")
    print()
    print("  [1] No-Reference Sharpness Metrics")
    print(f"  {'':>55}  {'mean':>10} {'std':>10} {'min':>10} {'max':>10}")
    print(f"  {'Laplacian Before':>55} {lap_b.mean():>10.2f} {lap_b.std():>10.2f} {lap_b.min():>10.2f} {lap_b.max():>10.2f}")
    print(f"  {'After':>55} {lap_a.mean():>10.2f} {lap_a.std():>10.2f} {lap_a.min():>10.2f} {lap_a.max():>10.2f}")
    print(f"  {'Change':>55} {lap_ch.mean():>+10.2f} {lap_ch.std():>10.2f} {lap_ch.min():>+10.2f} {lap_ch.max():>+10.2f}")
    lap_judge = "SHARPENED" if lap_ch.mean() > 5 else "BLURRED" if lap_ch.mean() < -5 else "NO CHANGE"
    print(f"  {'>>> ' + lap_judge + f' ({lap_ch.mean():+.2f})':>75}")
    print(f"  {'Improved:':>55} {lap_imp}/{total_frames} ({lap_imp/total_frames*100:.1f}%)  Worsened: {lap_wor}/{total_frames} ({lap_wor/total_frames*100:.1f}%)")
    print()
    print(f"  {'Tenengrad Before':>55} {ten_b.mean():>10.2e} {ten_b.std():>10.2e} {ten_b.min():>10.2e} {ten_b.max():>10.2e}")
    print(f"  {'After':>55} {ten_a.mean():>10.2e} {ten_a.std():>10.2e} {ten_a.min():>10.2e} {ten_a.max():>10.2e}")
    print(f"  {'Change':>55} {ten_ch.mean():>+10.2e} {ten_ch.std():>10.2e} {ten_ch.min():>+10.2e} {ten_ch.max():>+10.2e}")
    ten_judge = "SHARPENED" if ten_ch.mean() > 1e6 else "BLURRED" if ten_ch.mean() < -1e6 else "NO CHANGE"
    print(f"  {'>>> ' + ten_judge + f' ({ten_ch.mean():+.2e})':>75}")
    print(f"  {'Improved:':>55} {ten_imp}/{total_frames} ({ten_imp/total_frames*100:.1f}%)  Worsened: {ten_wor}/{total_frames} ({ten_wor/total_frames*100:.1f}%)")
    print()
    print("  [2] Gradient Activity & Ringing")
    print(f"  {'':>55}  {'mean':>10} {'std':>10} {'min':>10} {'max':>10}")
    print(f"  {'TV (original)':>55} {tv_b_arr.mean():>10.2e} {tv_b_arr.std():>10.2e} {tv_b_arr.min():>10.2e} {tv_b_arr.max():>10.2e}")
    print(f"  {'TV (deblurred)':>55} {tv_a_arr.mean():>10.2e} {tv_a_arr.std():>10.2e} {tv_a_arr.min():>10.2e} {tv_a_arr.max():>10.2e}")
    print(f"  {'TV change':>55} {tv_ch_arr.mean():>+10.2e} {tv_ch_arr.std():>10.2e} {tv_ch_arr.min():>+10.2e} {tv_ch_arr.max():>+10.2e}")
    tv_ratio_mean = tv_ratios.mean()
    tv_judge = f"RINGING x{tv_ratio_mean:.2f}" if tv_ratio_mean > 2.0 else f"MODERATE x{tv_ratio_mean:.2f}" if tv_ratio_mean > 1.2 else f"LOW x{tv_ratio_mean:.2f}"
    print(f"  {'TV ratio (after/before)':>55} {tv_ratio_mean:.4f}  --- {tv_judge}")
    er_mean = er_arr.mean()
    er_judge = "GOOD" if er_mean > 0.5 else "FAIR" if er_mean > 0.2 else "RINGING DOMINANT"
    print(f"  {'Edge Ratio (avg)':>55} {er_mean:.4f}  --- {er_judge}  (>0.5=good <0.2=ringing)")
    print()
    print("  [3] Full-Reference (vs Original Blurry)")
    print(f"  {'':>55}  {'mean':>10} {'std':>10} {'min':>10} {'max':>10}")
    print(f"  {'PSNR_raw (dB)':>55} {np.mean(psnr_vals):>10.2f} {np.std(psnr_vals):>10.2f} {np.min(psnr_vals):>10.2f} {np.max(psnr_vals):>10.2f}")
    print(f"  {'SSIM_raw':>55} {np.mean(ssim_vals):>10.4f} {np.std(ssim_vals):>10.4f} {np.min(ssim_vals):>10.4f} {np.max(ssim_vals):>10.4f}")
    print(f"  {'PSNR_matched (dB)':>55} {np.mean(psnr_matched_vals):>10.2f} {np.std(psnr_matched_vals):>10.2f} {np.min(psnr_matched_vals):>10.2f} {np.max(psnr_matched_vals):>10.2f}")
    print(f"  {'SSIM_matched':>55} {np.mean(ssim_matched_vals):>10.4f} {np.std(ssim_matched_vals):>10.4f} {np.min(ssim_matched_vals):>10.4f} {np.max(ssim_matched_vals):>10.4f}")
    print()
    p_judge = "LOW CHANGE" if np.mean(psnr_vals) > 35 else "MODERATE" if np.mean(psnr_vals) > 20 else "STRONG CHANGE"
    s_judge = "NEAR IDENTICAL" if np.mean(ssim_vals) > 0.95 else "GOOD" if np.mean(ssim_vals) > 0.85 else "DAMAGED"
    print(f"  >>> PSNR {p_judge} ({np.mean(psnr_vals):.1f}dB)  SSIM {s_judge} ({np.mean(ssim_vals):.4f})")
    print()
    # Overall quality
    scores = []
    if lap_ch.mean() > 5: scores.append("sharp")
    if tv_ratio_mean < 1.5: scores.append("low-ringing")
    if er_mean > 0.5: scores.append("clean-edges")
    if np.mean(ssim_vals) > 0.85: scores.append("structure-preserved")
    quality = "GOOD" if len(scores) >= 3 else "FAIR" if len(scores) >= 2 else "POOR"
    print(f"  >>> OVERALL QUALITY: {quality}  ({', '.join(scores)})")
    print("=" * 86)
    # === Detailed evaluation for selected frame ===
    show_fi = args.show_frame if args.show_frame is not None else total_frames // 2
    if show_fi >= len(sync):
        show_fi = len(sync) // 2
    
    frame_bgr = reader.read_frame(show_fi)
    if frame_bgr is not None:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        ri = sync[show_fi] if show_fi < len(sync) else show_fi
        q = meta["joint_positions"][ri]
        qd = meta["joint_velocities"][ri]
        if meta["tool_twist"] is not None:
            pose = meta["tool_pose"][ri]
            twist = meta["tool_twist"][ri]
            R_ee_depth = euler_zyx_to_rotmat(pose[3:])
            if hand_eye is not None:
                cam_pos = pose[:3] + R_ee_depth @ hand_eye.t
                R_cam = R_ee_depth @ hand_eye.R
            else:
                cam_pos = pose[:3]
                R_cam = R_ee_depth
            opt_axis = R_cam @ np.array([0, 0, 1])
            opt_z = opt_axis[2]
            depth_val = max(abs((0.0 - cam_pos[2]) / max(abs(opt_z), 0.01)), 0.02)
            psf, (du, dv) = compute_psf_from_pose(
                pose, twist, depth_val,
                fx=733.37, fy=733.37,
                cx=w // 2, cy=h // 2,
                exposure_time=args.exposure,
                hand_eye=hand_eye
            )
        else:
            # === 旧格式: 无 tool_twist ===
            v_cam_6d = get_camera_velocity(q, qd, hand_eye=hand_eye, robot=robot)
            depth_val = args.depth
            psf, (du, dv) = compute_psf_from_pose(
                depth=depth_val, fx=733.37, fy=733.37,
                cx=w // 2, cy=h // 2,
                exposure_time=args.exposure,
                v_cam_6d=v_cam_6d
            )
        if args.psf_sigma > 0:
            from scipy.ndimage import gaussian_filter
            psf = gaussian_filter(psf, sigma=args.psf_sigma)
            psf /= psf.sum()
        K_eff = args.K
        if args.adaptive_k:
            psz = psf.shape[0]
            K_eff = args.K * (1.0 + 0.3 * np.log2(max(psz, 3) / 17.0))
        if args.method == "tv":
            deblurred = tv_deconv(gray, psf, lam=args.tv_lam)
        elif args.method == "rl":
            deblurred = richardson_lucy(gray, psf, iterations=args.rl_iters)
        else:
            deblurred = wiener_deconvolution(gray, psf, K=K_eff)

        ev = full_evaluate(gray, deblurred)
        print()
        print("=" * 86)
        print(f"  Detailed Evaluation: Frame {show_fi}  (depth={depth_val:.3f}m, du={du:.2f}, dv={dv:.2f}, psf={psf.shape[0]}x{psf.shape[1]})")
        print("=" * 86)
        s = ev["stats_before"]
        print(f"  Laplacian Variance       {ev['laplacian_before']:>10.2f}  ->  {ev['laplacian_after']:>10.2f}  (change: {ev['laplacian_change']:+>10.2f})")
        print(f"  Tenengrad (gradient)     {ev['tenengrad_before']:>10.2e}  ->  {ev['tenengrad_after']:>10.2e}  (change: {ev['tenengrad_change']:+>10.2e})")
        print(f"  PSNR (blurry vs deb)     {ev['PSNR_raw']:>10.2f} dB")
        print(f"  SSIM (blurry vs deb)     {ev['SSIM_raw']:>10.4f}")
        print(f"  Mean brightness          {s['mean']:>10.2f}  ->  {ev['stats_after']['mean']:>10.2f}")
        print(f"  Std dev                  {s['std']:>10.2f}  ->  {ev['stats_after']['std']:>10.2f}")
        print(f"  Pixel displacement       du={du:.2f}, dv={dv:.2f}")
        print(f"  PSF kernel size          {psf.shape[0]}x{psf.shape[1]}")
        print(f"  TV (original)            {ev["tv_before"]:>14.2e}")
        print(f"  TV (deblurred)           {ev["tv_after"]:>14.2e}  (change: {ev["tv_change"]:+>12.2e})")
        print(f"  Edge ratio (strong/weak) {ev["edge_ratio"]:>14.2f}  (>1.5=good <1.0=ringing)")
        print(f"  Edge ratio (all frames avg) {np.mean(edge_ratios):>14.2f}")
        print("=" * 86)

        # === Save detailed frame results ===
        method_name = "Wiener" if args.method == "wiener" else "TV-L2" if args.method == "tv" else "RL"
        param_str = f"K{args.K}" if args.method == "wiener" else f"lam{args.tv_lam}" if args.method == "tv" else f"iter{args.rl_iters}"
        # depth（旧格式）
        if args.method == "wiener":
            pass  # depth not in name by default
        # psf-sigma
        param_str += f"_e{args.exposure}"
        if args.psf_sigma > 0:
            param_str += f"_sig{args.psf_sigma}"
        # adaptive-k
        if args.adaptive_k:
            param_str += "_adptK"
        episode_name = os.path.splitext(os.path.basename(args.h5))[0]
        depth_prefix = "auto" if meta["tool_twist"] is not None else "fix"
        out_name = f"{episode_name}_{method_name}_{param_str}_{depth_prefix}_d{depth_val:.2f}_f{show_fi}"
        out_dir = os.path.join(ROOT, "batch_analyze", out_name)
        os.makedirs(out_dir, exist_ok=True)
        
        from PIL import Image
        hh, ww = gray.shape
        canvas = np.ones((hh + 60, ww * 2, 3), dtype=np.uint8) * 240
        gc = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        dc = cv2.cvtColor(deblurred, cv2.COLOR_GRAY2BGR)
        canvas[60:, :ww] = gc
        canvas[60:, ww:] = dc
        cv2.putText(canvas, f"Original frame {show_fi}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv2.putText(canvas, f"{method_name} {param_str}", (ww+10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).save(os.path.join(out_dir, "comparison.png"))
        
        psf_img = (psf / psf.max() * 255).astype(np.uint8)
        Image.fromarray(cv2.resize(psf_img, (200,200), interpolation=cv2.INTER_NEAREST)).save(os.path.join(out_dir, "psf.png"))
        
        with open(os.path.join(out_dir, "metrics.txt"), "w") as mf:
            mf.write(f"Frame: {show_fi}\nMethod: {method_name}\nParams: {param_str}\n")
            mf.write(f"Depth: {depth_val:.3f}m\nDisplacement: du={du:.2f} dv={dv:.2f}\nPSF: {psf.shape[0]}x{psf.shape[1]}\n")
            mf.write(f"Laplacian: {ev['laplacian_before']:.2f} -> {ev['laplacian_after']:.2f} ({ev['laplacian_change']:+>.2f})\n")
            mf.write(f"Tenengrad: {ev['tenengrad_before']:.2e} -> {ev['tenengrad_after']:.2e} ({ev['tenengrad_change']:+>.2e})\n")
            mf.write(f"PSNR: {ev['PSNR_raw']:.2f} dB\nSSIM: {ev['SSIM_raw']:.4f}\n")
        
        print(f"  [SAVED] Results -> {out_dir}/")

    print("=" * 62)

if __name__ == "__main__":
    main()
