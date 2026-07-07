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
from joint_deblur import compute_psf, wiener_deconvolution
from evaluate import evaluate, compare_sharpness
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

    # 5. 计算 PSF
    psf, (du, dv) = compute_psf(
        q, qd, depth,
        fx=fx, fy=fy,
        cx=w // 2, cy=h // 2,
        exposure_time=exposure,
        hand_eye=hand_eye, robot=robot
    )
    print(f"Pixel displacement: du={du:.2f}, dv={dv:.2f}")
    print(f"PSF kernel: {psf.shape[0]}x{psf.shape[1]}")

   # 6. 去模糊
    deblurred = wiener_deconvolution(frame_gray, psf, K=K)

    # 7. 评估原图 vs 处理后图像
    results, matched = evaluate(frame_gray, deblurred)

    # ---- sharpness metrics (no-reference) ----
    lap_res = compare_sharpness(frame_gray, deblurred)
    lap_orig = lap_res["before"]
    lap_deb  = lap_res["after"]
    lap_ch   = lap_res["change"]
    j1       = lap_res["judgment"].strip()

    sobelx_orig = cv2.Sobel(frame_gray, cv2.CV_64F, 1, 0)
    sobely_orig = cv2.Sobel(frame_gray, cv2.CV_64F, 0, 1)
    ten_orig = (sobelx_orig**2 + sobely_orig**2).sum()

    sobelx_deb = cv2.Sobel(deblurred, cv2.CV_64F, 1, 0)
    sobely_deb = cv2.Sobel(deblurred, cv2.CV_64F, 0, 1)
    ten_deb = (sobelx_deb**2 + sobely_deb**2).sum()

    # ---- statistics ----
    import math
    orig_hist, _ = np.histogram(frame_gray, bins=256, range=(0, 255), density=True)
    deb_hist, _ = np.histogram(deblurred, bins=256, range=(0, 255), density=True)
    orig_ent = -np.sum(orig_hist * np.log2(orig_hist + 1e-10))
    deb_ent  = -np.sum(deb_hist  * np.log2(deb_hist  + 1e-10))

    print(f"\n{'='*86}")
    print('  Quantitative Evaluation: Original (blurry) vs Deblurred')
    print(f"{'='*86}")
    print(f"  {'[1] No-Ref Sharpness Metrics':<40}{'Original':>14}{'Deblurred':>14}{'Change':>14}{'Judgment':>12}")
    print(f"  {'-'*85}")
    ten_ch = ten_deb - ten_orig
    j2 = 'SHARPER' if ten_ch > 0 else 'BLURRIER' if ten_ch < 0 else 'SAME'
    print(f"  {'Laplacian Variance':<40}{lap_orig:>14.2f}{lap_deb:>14.2f}{lap_ch:>+14.2f}{j1:>12}")
    print(f"  {'Tenengrad (gradient sum)':<40}{ten_orig:>14.2e}{ten_deb:>14.2e}{ten_ch:>+14.2e}{j2:>12}")
    print(f"  {'-'*85}")
    print(f"  {'[2] Image Statistics':<40}{'Original':>14}{'Deblurred':>14}{'Change':>14}{'Unit':>12}")
    print(f"  {'-'*85}")
    print(f"  {'Mean':<40}{frame_gray.mean():>14.2f}{deblurred.mean():>14.2f}{(deblurred.mean()-frame_gray.mean()):>+14.2f}{'gray level':>12}")
    print(f"  {'Std Dev':<40}{frame_gray.std():>14.2f}{deblurred.std():>14.2f}{(deblurred.std()-frame_gray.std()):>+14.2f}{'gray level':>12}")
    print(f"  {'Min':<40}{int(frame_gray.min()):>14d}{int(deblurred.min()):>14d}{int(deblurred.min())-int(frame_gray.min()):>+14d}{'gray level':>12}")
    print(f"  {'Max':<40}{int(frame_gray.max()):>14d}{int(deblurred.max()):>14d}{int(deblurred.max())-int(frame_gray.max()):>+14d}{'gray level':>12}")
    print(f"  {'Median':<40}{np.median(frame_gray):>14.1f}{np.median(deblurred):>14.1f}{(np.median(deblurred)-np.median(frame_gray)):>+14.1f}{'gray level':>12}")
    print(f"  {'Entropy':<40}{orig_ent:>14.4f}{deb_ent:>14.4f}{(deb_ent-orig_ent):>+14.4f}{'bits':>12}")
    print(f"  {'-'*85}")
    pq = 'EXCELLENT' if results['PSNR_raw'] > 40 else 'GOOD' if results['PSNR_raw'] > 30 else 'FAIR'
    sq = 'EXCELLENT' if results['SSIM_raw'] > 0.99 else 'GOOD' if results['SSIM_raw'] > 0.9 else 'FAIR'
    pmq = 'EXCELLENT' if results['PSNR_matched'] > 50 else 'GOOD' if results['PSNR_matched'] > 40 else 'FAIR'
    smq = 'EXCELLENT' if results['SSIM_matched'] > 0.99 else 'GOOD' if results['SSIM_matched'] > 0.9 else 'FAIR'
    print(f"  {'[3] Pairwise Metrics (blurry vs deblurred)':<40}{'Value':>23}{'Assessment':>12}")
    print(f"  {'-'*85}")
    print(f"  {'PSNR_raw':<40}{results['PSNR_raw']:>14.2f} dB{'':>7}{pq:>12}")
    print(f"  {'SSIM_raw':<40}{results['SSIM_raw']:>14.4f}{'':>19}{sq:>12}")
    print(f"  {'PSNR_matched (hist matched)':<40}{results['PSNR_matched']:>14.2f} dB{'':>7}{pmq:>12}")
    print(f"  {'SSIM_matched (hist matched)':<40}{results['SSIM_matched']:>14.4f}{'':>19}{smq:>12}")
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

    info = (f"depth={depth}m  exp={exposure}s  du={du:.1f}  dv={dv:.1f}  |  " +
            f"PSNR={results['PSNR_raw']:.1f}dB  SSIM={results['SSIM_raw']:.3f}  |  "
            f"orig mean={frame_gray.mean():.0f}  deb mean={deblurred.mean():.0f}")
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
