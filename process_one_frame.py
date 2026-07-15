"""
process_one_frame.py — 单帧去模糊 + 对比图 + 评估

用法:
  python process_one_frame.py --frame 250 --depth 0.5 --K 0.01

支持中文路径（使用 PIL 保存图片）。
"""

import os, sys
import cv2
import numpy as np
from PIL import Image

FINAL_DIR = r"E:\Vital_document\CUHKSZ\课程文件\ECE4512\Final"
sys.path.insert(0, FINAL_DIR)

from h5_loader import load_episode_h5, EpisodeFrameReader
from joint_deblur import euler_zyx_to_rotmat, compute_psf_from_pose, wiener_deconvolution, tv_deconv
from evaluate import full_evaluate
from robot_configs import get_robot, HAND_EYE_CONFIGS


def imwrite_pil(path, img):
    """用 PIL 保存图像（支持中文路径）"""
    if len(img.shape) == 3:
        Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(path)
    else:
        Image.fromarray(img).save(path)


def main():
    # 默认参数
    frame_idx = 0
    depth = 0.5
    K = 0.01
    fx = fy = 733.37
    exposure = 0.03
    robot_name = "kinova-gen3"
    hand_eye_name = "kinova-gen3"
    table_z = 0.0
    psf_sigma = 0.0     # PSF regularization (gaussian), 0=off
    adaptive_k = False  # adaptive K scaling
    lam = 0.002         # TV regularization strength
    method = "wiener"   # deconv method (wiener/tv/rl)
    reverse_psf = False   # flip PSF 180 deg
    
    # 解析命令行
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--frame" and i + 1 < len(argv):
            frame_idx = int(argv[i + 1])
        if a == "--depth" and i + 1 < len(argv):
            depth = float(argv[i + 1])
        if a == "--K" and i + 1 < len(argv):
            K = float(argv[i + 1])
        if a == "--exposure" and i + 1 < len(argv):
            exposure = float(argv[i + 1])
        if a == "--robot" and i + 1 < len(argv):
            robot_name = argv[i + 1]
        if a == "--hand-eye" and i + 1 < len(argv):
            hand_eye_name = argv[i + 1]
        if a == "--psf-sigma" and i + 1 < len(argv):
            psf_sigma = float(argv[i + 1])
        if a == "--adaptive-k":
            adaptive_k = True
        if a == "--method" and i + 1 < len(argv):
            method = argv[i + 1]
        if a == "--tv-lam" and i + 1 < len(argv):
            lam = float(argv[i + 1])
        if a == "--reverse-psf":
            reverse_psf = True
        
    # 1. 加载 h5
    h5_path = os.path.join(FINAL_DIR, "episode_0002.h5")
    print(f"Loading {h5_path}...")
    meta = load_episode_h5(h5_path)
    reader = EpisodeFrameReader(meta["rgb_bytes"])
    sync = meta["sync_indices"]

    # 2. 读取帧
    frame_bgr = reader.read_frame(frame_idx)
    if frame_bgr is None:
        print(f"ERROR: cannot read frame {frame_idx}")
        return
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = frame_gray.shape
    print(f"Frame {frame_idx}: {w}x{h}")

    # 3. 获取关节角
    ri = sync[frame_idx] if frame_idx < len(sync) else frame_idx
    q = meta["joint_positions"][ri]
    qd = meta["joint_velocities"][ri]
    print(f"Robot time point: {ri}")

    # 4. 获取配置
    robot = get_robot(robot_name)
    hand_eye = HAND_EYE_CONFIGS.get(hand_eye_name, None)

    # 5. 用 tool_pose 计算深度（沿光轴到桌面）+ tool_twist 算 PSF
    pose = meta["tool_pose"][ri]
    twist = meta["tool_twist"][ri]
    # 计算深度：沿相机光轴到桌面（考虑手眼标定的平移）
    R_ee_depth = euler_zyx_to_rotmat(pose[3:])
    if hand_eye is not None:
        cam_pos = pose[:3] + R_ee_depth @ hand_eye.t
        R_cam = R_ee_depth @ hand_eye.R
    else:
        cam_pos = pose[:3]
        R_cam = R_ee_depth
    # distance along optical axis to table (user request)
    opt_axis = R_cam @ np.array([0, 0, 1])
    opt_z = opt_axis[2]
    depth = max(abs((table_z - cam_pos[2]) / max(abs(opt_z), 0.01)), 0.02)
    print(f"  depth={depth:.3f}m (camZ={cam_pos[2]:.3f}, opt_z={opt_z:+.3f})")
    psf, (du, dv) = compute_psf_from_pose(
        pose, twist, depth,
        fx=fx, fy=fy,
        cx=w // 2, cy=h // 2,
        exposure_time=exposure,
        hand_eye=hand_eye
    )
    # PSF regularization
    if psf_sigma > 0:
        from scipy.ndimage import gaussian_filter
        psf = gaussian_filter(psf, sigma=psf_sigma)
        psf /= psf.sum()
    # Adaptive K scaling
    K_eff = K
    if adaptive_k:
        psz = psf.shape[0]
        K_eff = K * (1.0 + 0.3 * np.log2(max(psz, 3) / 17.0))

    # Reverse PSF direction if requested
    if reverse_psf:
        psf = psf[::-1, ::-1]
        du, dv = -du, -dv
    print(f"Pixel displacement: du={du:.2f}, dv={dv:.2f}")
    print(f"PSF kernel: {psf.shape[0]}x{psf.shape[1]}")

   # 6. 去模糊
    if method == "tv":
        deblurred = tv_deconv(frame_gray, psf, lam=lam)
    else:
        deblurred = wiener_deconvolution(frame_gray, psf, K=K_eff)

    # 7. 评估原图 vs 处理后图像
    # 7. 完整评估（使用 evaluate.py 的 full_evaluate）
    eval_all = full_evaluate(frame_gray, deblurred)
    s_b = eval_all["stats_before"]
    s_a = eval_all["stats_after"]
    j1 = '++ SHARPER' if eval_all['laplacian_change'] > 0 else '-- BLURRIER'
    j2 = '++ SHARPER' if eval_all['tenengrad_change'] > 0 else '-- BLURRIER'

    print(f"\n{'='*86}")
    print('  Quantitative Evaluation: Original (blurry) vs Deblurred')
    print(f"{'='*86}")
    print(f"  {'[1] No-Ref Sharpness Metrics':<40}{'Original':>14}{'Deblurred':>14}{'Change':>14}{'Judgment':>12}")
    print(f"  {'-'*85}")
    eval_all['tenengrad_change'] = eval_all['tenengrad_after'] - eval_all['tenengrad_before']
    j2 = 'SHARPER' if eval_all['tenengrad_change'] > 0 else 'BLURRIER' if eval_all['tenengrad_change'] < 0 else 'SAME'
    print(f"  {'Laplacian Variance':<40}{eval_all['laplacian_before']:>14.2f}{eval_all['laplacian_after']:>14.2f}{eval_all['laplacian_change']:>+14.2f}{eval_all['laplacian_change']:>+14.2f}junk")
    print(f"  {'Tenengrad (gradient sum)':<40}{eval_all['tenengrad_before']:>14.2e}{eval_all['tenengrad_after']:>14.2e}{eval_all['tenengrad_change']:>+14.2e}{j2:>12}")
    print(f"  {'-'*85}")
    print(f"  {'[2] Image Statistics':<40}{'Original':>14}{'Deblurred':>14}{'Change':>14}{'Unit':>12}")
    print(f"  {'-'*85}")
    print(f"  {'Mean':<40}{s_b['mean']:>14.2f}{s_a['mean']:>14.2f}{(s_a['mean']-s_b['mean']):>+14.2f}{'gray level':>12}")
    print(f"  {'Std Dev':<40}{s_b['std']:>14.2f}{s_a['std']:>14.2f}{(s_a['std']-s_b['std']):>+14.2f}{'gray level':>12}")
    print(f"  {'Min':<40}{s_b['min']:>14d}{s_a['min']:>14d}{s_a['min']-s_b['min']:>+14d}{'gray level':>12}")
    print(f"  {'Max':<40}{s_b['max']:>14d}{s_a['max']:>14d}{s_a['max']-s_b['max']:>+14d}{'gray level':>12}")
    print(f"  {'Median':<40}{s_b['median']:>14.1f}{s_a['median']:>14.1f}{(s_a['median']-s_b['median']):>+14.1f}{'gray level':>12}")
    print(f"  {'Entropy':<40}{s_b['entropy']:>14.4f}{s_a['entropy']:>14.4f}{(s_a['entropy']-s_b['entropy']):>+14.4f}{'bits':>12}")
    print(f"  {'-'*85}")
    pq = 'EXCELLENT' if eval_all['PSNR_raw'] > 40 else 'GOOD' if eval_all['PSNR_raw'] > 30 else 'FAIR'
    sq = 'EXCELLENT' if eval_all['SSIM_raw'] > 0.99 else 'GOOD' if eval_all['SSIM_raw'] > 0.9 else 'FAIR'
    pmq = 'EXCELLENT' if eval_all['PSNR_matched'] > 50 else 'GOOD' if eval_all['PSNR_matched'] > 40 else 'FAIR'
    smq = 'EXCELLENT' if eval_all['SSIM_matched'] > 0.99 else 'GOOD' if eval_all['SSIM_matched'] > 0.9 else 'FAIR'
    print(f"  {'[3] Pairwise Metrics (blurry vs deblurred)':<40}{'Value':>23}{'Assessment':>12}")
    print(f"  {'-'*85}")
    print(f"  {'PSNR_raw':<40}{eval_all['PSNR_raw']:>14.2f} dB{'':>7}{pq:>12}")
    print(f"  {'SSIM_raw':<40}{eval_all['SSIM_raw']:>14.4f}{'':>19}{sq:>12}")
    print(f"  {'PSNR_matched (hist matched)':<40}{eval_all['PSNR_matched']:>14.2f} dB{'':>7}{pmq:>12}")
    print(f"  {'SSIM_matched (hist matched)':<40}{eval_all['SSIM_matched']:>14.4f}{'':>19}{smq:>12}")
    print(f"  {'-'*85}")
    print(f"  {'Pixel displacement (du, dv)':<40}({du:.1f}, {dv:.1f}){'':>27}pixels")
    print(f"  {'PSF kernel size':<40}{psf.shape[0]}x{psf.shape[1]}{'':>31}")
    print(f"{'='*86}")

    # 8. save results (PIL supports Chinese paths)
    out_dir = os.path.join(FINAL_DIR, 'single_frame_output')
    os.makedirs(out_dir, exist_ok=True)
    prefix = f'frame_{frame_idx:04d}'
    # --- 对比图 ---
    gray_color = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
    deb_color = cv2.cvtColor(deblurred, cv2.COLOR_GRAY2BGR)
    label_h = 60
    canvas = np.ones((h + label_h, w * 2, 3), dtype=np.uint8) * 240

    canvas[label_h:, :w] = gray_color
    canvas[label_h:, w:] = deb_color

    cv2.putText(canvas, f"Original (blurred) frame {frame_idx}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(canvas, f"Wiener Deblurred (K={K})", (w + 10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    info = (f"depth={depth}m  sig={psf_sigma}  exp={exposure}s  du={du:.1f}  dv={dv:.1f}  |  " +
            f"PSNR={eval_all['PSNR_raw']:.1f}dB  SSIM={eval_all['SSIM_raw']:.3f}  TV delta={eval_all['tv_change']:.2e}  EdgeR={eval_all['edge_ratio']:.2f}  |  "
            f"orig mean={s_b['mean']:.0f}  deb mean={s_a['mean']:.0f}")
    cv2.putText(canvas, info, (10, h + label_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)

    comp_path = os.path.join(out_dir, f"{prefix}_comparison.png")
    imwrite_pil(comp_path, canvas)
    print(f"\nSaved comparison: {comp_path}")

    # --- 单独保存 ---
    ori_path = os.path.join(out_dir, f"{prefix}_original.png")
    imwrite_pil(ori_path, frame_gray)
    print(f"Saved original:    {ori_path}")

    deb_path = os.path.join(out_dir, f"{prefix}_deblurred.png")
    imwrite_pil(deb_path, deblurred)
    print(f"Saved deblurred:   {deb_path}")

    # 9. 打印 PSF
    print(f"\nPSF kernel ({psf.shape[0]}x{psf.shape[1]}):")
    np.set_printoptions(precision=3, suppress=True, linewidth=80)
    for row in psf:
        line = " ".join(f"{v:.3f}" for v in row)
        if line.strip("0. "):
            print(f"  {line}")

    print(f"\nDone! Check '{out_dir}' for results.")


if __name__ == "__main__":
    main()
