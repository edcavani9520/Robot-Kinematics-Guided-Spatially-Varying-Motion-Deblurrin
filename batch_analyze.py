import sys, os, time, zipfile
import numpy as np
import cv2

ROOT = r"E:\Vital_document\CUHKSZ\课程文件\ECE4512\Final"
sys.path.insert(0, ROOT)

from h5_loader import load_episode_h5, EpisodeFrameReader
from joint_deblur import compute_psf, wiener_deconvolution
from evaluate import compare_sharpness, psnr, ssim
from robot_configs import get_robot, HAND_EYE_CONFIGS


def make_comparison(gray, deblurred, fi, ch, label=""):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", default="episode_0002.h5")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--depth", type=float, default=0.5)
    parser.add_argument("--exposure", type=float, default=0.03)
    parser.add_argument("--K", type=float, default=0.01)
    parser.add_argument("--robot", default="kinova-gen3")
    parser.add_argument("--hand-eye", default="kinova-gen3")
    parser.add_argument("--out-dir", default="batch_output",
                        help="output directory for failed-frame images + zip")
    parser.add_argument("--save-failed", action="store_true", default=True,
                        help="save comparison images for worsened frames")
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
    lap_after  = []
    psnr_vals  = []
    ssim_vals  = []
    times_ms   = []
    failed_info = []   # (frame_idx, before, after, change, psnr, ssim)
    failed_imgs = []   # (frame_idx, gray_img, deblurred_img)

    out_dir = os.path.join(ROOT, args.out_dir)
    failed_dir = os.path.join(out_dir, "failed_frames")

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

        psf, (du, dv) = compute_psf(
            q, qd, args.depth,
            fx=733.37, fy=733.37,
            cx=w // 2, cy=h // 2,
            exposure_time=args.exposure,
            hand_eye=hand_eye, robot=robot
        )
        deblurred = wiener_deconvolution(gray, psf, K=args.K)

        lap_res = compare_sharpness(gray, deblurred)
        lap_before.append(lap_res["before"])
        lap_after.append(lap_res["after"])
        improved = lap_res["improved"]
        p = psnr(gray, deblurred)
        s = ssim(gray, deblurred)
        psnr_vals.append(p)
        ssim_vals.append(s)

        if args.save_failed and not improved:
            failed_info.append((fi, lap_res["before"], lap_res["after"],
                                lap_res["change"], p, s))
            failed_imgs.append((fi, gray, deblurred))

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
    same_count     = int(np.sum(changed == 0))

    # === Save failed-frame comparisons ===
    if args.save_failed and failed_info:
        from PIL import Image
        os.makedirs(failed_dir, exist_ok=True)
        print(f"\n  Saving {len(failed_info)} failed-frame comparisons to {failed_dir} ...")

        for fi, gray, deblurred in failed_imgs:
            ch = next((x[3] for x in failed_info if x[0] == fi), 0)
            canvas = make_comparison(gray, deblurred, fi, ch, "(worsened)")
            img_path = os.path.join(failed_dir, f"frame_{fi:04d}_failed.png")
            Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).save(img_path)

        # Text report
        txt_path = os.path.join(failed_dir, "failed_report.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Failed frames: {len(failed_info)} / {total_frames}\n")
            f.write(f"Parameters: depth={args.depth}, K={args.K}, exposure={args.exposure}\n")
            f.write(f"{'Frame':>6} {'Lap_before':>12} {'Lap_after':>12} {'Change':>10} {'PSNR':>8} {'SSIM':>7}\n")
            f.write("-" * 60 + "\n")
            for fi, b, a, ch, p, s in failed_info:
                f.write(f"{fi:>6d} {b:>12.2f} {a:>12.2f} {ch:>+10.2f} {p:>8.2f} {s:>7.4f}\n")

        # Zip
        zip_path = os.path.join(out_dir, "failed_frames.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in sorted(os.listdir(failed_dir)):
                zf.write(os.path.join(failed_dir, fname), arcname=fname)
        print(f"  Zipped: {zip_path}")

    # === Print summary ===
    print()
    print("=" * 62)
    print("  Batch Evaluation Summary")
    print("=" * 62)
    print(f"  Frames processed:       {total_frames}")
    print(f"  Total time:             {dt:.1f}s  ({np.mean(times_ms):.0f}ms/frame)")
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
    print("  [2] Pairwise (Blurry vs Deblurred)")
    print(f"  {'':>12} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}")
    print(f"  {'PSNR':>12} {np.mean(psnr_vals):>10.2f} {np.std(psnr_vals):>10.2f} {np.min(psnr_vals):>10.2f} {np.max(psnr_vals):>10.2f} dB")
    print(f"  {'SSIM':>12} {np.mean(ssim_vals):>10.4f} {np.std(ssim_vals):>10.4f} {np.min(ssim_vals):>10.4f} {np.max(ssim_vals):>10.4f}")
    if args.save_failed and failed_info:
        print()
        print("  Failed-frame details saved to:")
        print(f"    Images:  {failed_dir}/")
        print(f"    Report:  {failed_dir}/failed_report.txt")
        print(f"    Zip:     {zip_path}")
    print("=" * 62)


if __name__ == "__main__":
    main()
