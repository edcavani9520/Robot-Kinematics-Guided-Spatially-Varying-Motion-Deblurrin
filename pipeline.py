"""RGB motion-deblurring pipeline for the current Kinova episode format."""

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from cli_config import (
    RLConfig,
    TVConfig,
    WienerConfig,
    add_common_arguments,
    add_method_subcommands,
    build_run_name,
    config_from_args,
    method_config_dict,
    positive_int,
    prepare_run_directory,
    require_bool,
    require_optional_positive_int_value,
    resolve_psf_parameters,
    write_run_config,
)
from h5_loader import EpisodeFrameReader, load_episode_h5
from joint_deblur import (
    compute_psf_from_pose,
    get_camera_velocity,
    richardson_lucy,
    tv_deconv,
    wiener_deconvolution,
)
from robot_configs import HAND_EYE_CONFIGS, get_robot


def rgb_to_bgr(image):
    """Convert an RGB image to the channel order required by OpenCV output."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def build_comparison_canvas(original_rgb, deblurred_rgb, label_height=28, info=""):
    """Build an RGB comparison canvas without changing either image's colors."""
    if original_rgb.shape != deblurred_rgb.shape:
        raise ValueError("comparison images must have equal shapes")
    if original_rgb.ndim != 3 or original_rgb.shape[2] != 3:
        raise ValueError("comparison images must be RGB")
    height, width = original_rgb.shape[:2]
    canvas = np.zeros((height + label_height, width * 2, 3), dtype=np.uint8)
    canvas[label_height:, :width] = original_rgb
    canvas[label_height:, width:] = deblurred_rgb
    cv2.putText(
        canvas,
        "Original (blurred)",
        (8, min(20, label_height - 3)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        canvas,
        f"Deblurred {info}".rstrip(),
        (width + 8, min(20, label_height - 3)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    return canvas


def save_deblur_result(
    output_dir,
    frame_idx,
    original_rgb,
    deblurred_rgb,
    psf_meta,
    comp_writer=None,
    vid_writer=None,
):
    """Save RGB images, converting to BGR only at OpenCV boundaries."""
    output_dir = Path(output_dir)
    mode = psf_meta[0]
    if mode == "global":
        _, psf, du, dv = psf_meta
        info = f"du={du:.1f} dv={dv:.1f} psf={psf.shape[0]}"
    else:
        _, _, du_grid, _ = psf_meta
        info = f"spatial |du|={float(np.abs(du_grid).mean()):.1f}"

    cv2.imwrite(
        str(output_dir / "blurred" / f"step_{frame_idx:04d}.jpg"),
        rgb_to_bgr(original_rgb),
    )
    cv2.imwrite(
        str(output_dir / "deblurred" / f"step_{frame_idx:04d}.jpg"),
        rgb_to_bgr(deblurred_rgb),
    )
    canvas_rgb = build_comparison_canvas(original_rgb, deblurred_rgb, info=info)
    cv2.imwrite(
        str(output_dir / "comparison" / f"compare_{frame_idx:04d}.jpg"),
        rgb_to_bgr(canvas_rgb),
    )

    if vid_writer is not None:
        vid_writer.write(rgb_to_bgr(deblurred_rgb))
    if comp_writer is not None:
        comp_writer.write(rgb_to_bgr(np.hstack([original_rgb, deblurred_rgb])))


def setup_output_dirs(output_dir, width, height, fps=10.0):
    """Create output directories and color video writers."""
    output_dir = Path(output_dir)
    for name in ("blurred", "deblurred", "comparison"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    deblurred_writer = cv2.VideoWriter(
        str(output_dir / "deblurred_video.mp4"),
        fourcc,
        fps,
        (width, height),
        isColor=True,
    )
    comparison_writer = cv2.VideoWriter(
        str(output_dir / "comparison_video.mp4"),
        fourcc,
        fps,
        (width * 2, height),
        isColor=True,
    )
    if not deblurred_writer.isOpened() or not comparison_writer.isOpened():
        deblurred_writer.release()
        comparison_writer.release()
        raise RuntimeError(f"cannot open output video writers in {output_dir}")
    return deblurred_writer, comparison_writer


def compute_episode_psf(
    meta,
    frame_idx,
    *,
    width,
    height,
    robot,
    hand_eye,
    depth,
    fx,
    fy,
    exposure,
    psf_sigma=0.0,
):
    """Compute one PSF from current-format joint positions and velocities."""
    camera_velocity = get_camera_velocity(
        meta["joint_positions"][frame_idx],
        meta["joint_velocities"][frame_idx],
        hand_eye=hand_eye,
        robot=robot,
    )
    psf, (du, dv) = compute_psf_from_pose(
        depth=depth,
        fx=fx,
        fy=fy,
        cx=width // 2,
        cy=height // 2,
        exposure_time=exposure,
        v_cam_6d=camera_velocity,
    )
    if psf_sigma > 0:
        from scipy.ndimage import gaussian_filter

        psf = gaussian_filter(psf, sigma=psf_sigma)
        psf /= psf.sum()
    validate_psf(psf, du, dv, (height, width))
    return psf, du, dv


def validate_psf(psf, du, dv, image_shape):
    """Reject non-finite, empty, or physically unprocessable PSFs."""
    psf = np.asarray(psf)
    if not np.isfinite([du, dv]).all() or not np.isfinite(psf).all():
        raise ValueError("PSF and displacement must contain only finite values")
    if psf.ndim != 2 or psf.size == 0 or psf.sum() <= 0:
        raise ValueError("PSF must be a non-empty positive 2-D kernel")
    height, width = image_shape[:2]
    if psf.shape[0] > height or psf.shape[1] > width:
        raise ValueError(
            f"PSF shape {psf.shape} exceeds image shape {(height, width)}"
        )


def effective_wiener_k(config, psf):
    value = config.K
    if config.adaptive_k:
        value *= 1.0 + 0.3 * np.log2(max(psf.shape[0], 3) / 17.0)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"effective Wiener K must be finite and positive; got {value}")
    return float(value)


def deblur_rgb(image_rgb, psf, method_config):
    """Run exactly the algorithm represented by one method configuration."""
    if isinstance(method_config, TVConfig):
        return tv_deconv(image_rgb, psf, lam=method_config.lam), None
    if isinstance(method_config, RLConfig):
        return richardson_lucy(
            image_rgb, psf, iterations=method_config.iterations
        ), None
    if isinstance(method_config, WienerConfig):
        effective_k = effective_wiener_k(method_config, psf)
        return wiener_deconvolution(image_rgb, psf, K=effective_k), effective_k
    raise TypeError(f"unsupported method configuration: {type(method_config).__name__}")


def run_h5_pipeline(
    h5_path,
    output_root,
    *,
    method_config,
    fx=None,
    fy=None,
    depth=0.5,
    exposure=None,
    max_frames=None,
    psf_sigma=0.0,
    overwrite=False,
):
    """Process a current-format RGB HDF5 episode."""
    max_frames = require_optional_positive_int_value("max_frames", max_frames)
    overwrite = require_bool("overwrite", overwrite)
    meta = load_episode_h5(h5_path)
    physical = resolve_psf_parameters(
        meta,
        depth=depth,
        fx=fx,
        fy=fy,
        exposure=exposure,
        psf_sigma=psf_sigma,
    )
    reader = EpisodeFrameReader(meta["images"])
    robot = get_robot("kinova-gen3")
    hand_eye_params = HAND_EYE_CONFIGS["kinova-gen3"]
    limit = meta["num_frames"] if max_frames is None else min(meta["num_frames"], max_frames)
    run_name = build_run_name(
        h5_path, method_config, depth=physical.depth,
        exposure=physical.exposure, fx=physical.fx, fy=physical.fy,
        psf_sigma=physical.psf_sigma, max_frames=max_frames,
    )
    output_dir = prepare_run_directory(
        Path(output_root) / run_name,
        overwrite=overwrite,
        known_entries={"blurred", "deblurred", "comparison", "deblurred_video.mp4",
                       "comparison_video.mp4", "psf_report.csv", "run_config.json"},
    )
    write_run_config(output_dir, {
        "entry_point": "pipeline", "h5": str(Path(h5_path).resolve()),
        "method": method_config_dict(method_config), "depth": physical.depth,
        "exposure": physical.exposure, "fx": physical.fx, "fy": physical.fy,
        "psf_sigma": physical.psf_sigma,
        "intrinsics_source": physical.intrinsics_source,
        "max_frames": max_frames, "run_name": run_name,
    })
    video_writer, comparison_writer = setup_output_dirs(
        output_dir, meta["W"], meta["H"], meta["camera_fps"]
    )

    start = time.time()
    try:
        with (output_dir / "psf_report.csv").open("w", newline="", encoding="utf-8") as f:
            report = csv.writer(f)
            report.writerow(
                [
                    "step",
                    "du",
                    "dv",
                    "psf_size",
                    "effective_K",
                    "original_mean",
                    "deblurred_mean",
                    "original_std",
                    "deblurred_std",
                ]
            )
            for frame_idx in range(limit):
                image_rgb = reader.read_frame(frame_idx)
                if image_rgb is None:
                    raise IndexError(f"cannot read RGB frame {frame_idx}")
                psf, du, dv = compute_episode_psf(
                    meta,
                    frame_idx,
                    width=meta["W"],
                    height=meta["H"],
                    robot=robot,
                    hand_eye=hand_eye_params,
                    depth=physical.depth,
                    fx=physical.fx,
                    fy=physical.fy,
                    exposure=physical.exposure,
                    psf_sigma=physical.psf_sigma,
                )
                deblurred_rgb, effective_k = deblur_rgb(image_rgb, psf, method_config)
                save_deblur_result(
                    output_dir,
                    frame_idx,
                    image_rgb,
                    deblurred_rgb,
                    ("global", psf, du, dv),
                    comp_writer=comparison_writer,
                    vid_writer=video_writer,
                )
                report.writerow(
                    [
                        frame_idx,
                        f"{du:.3f}",
                        f"{dv:.3f}",
                        psf.shape[0],
                        "" if effective_k is None else f"{effective_k:.8g}",
                        f"{image_rgb.mean():.1f}",
                        f"{deblurred_rgb.mean():.1f}",
                        f"{image_rgb.std():.1f}",
                        f"{deblurred_rgb.std():.1f}",
                    ]
                )
                if frame_idx % max(1, limit // 20) == 0:
                    print(
                        f"  [{frame_idx + 1}/{limit}] du={du:.2f} dv={dv:.2f} "
                        f"psf={psf.shape[0]}x{psf.shape[1]}"
                    )
    finally:
        reader.close()
        video_writer.release()
        comparison_writer.release()

    elapsed = time.time() - start
    print(f"[OK] Processed {limit} RGB frames in {elapsed:.1f}s -> {output_dir}")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_common_arguments(parser, h5_required=True)
    parser.add_argument("--output", default="deblur_output")
    parser.add_argument("--max-frames", type=positive_int)
    parser.add_argument("--overwrite", action="store_true")
    add_method_subcommands(parser)
    args = parser.parse_args(argv)
    try:
        args.method_config = config_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main():
    args = _parse_args()
    run_h5_pipeline(
        args.h5,
        args.output,
        method_config=args.method_config,
        fx=args.fx,
        fy=args.fy,
        depth=args.depth,
        exposure=args.exposure,
        max_frames=args.max_frames,
        psf_sigma=args.psf_sigma,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
