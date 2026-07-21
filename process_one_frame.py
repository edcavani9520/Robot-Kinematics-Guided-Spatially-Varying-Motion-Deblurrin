"""Deblur and evaluate one RGB frame from a current-format episode."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from cli_config import (
    add_common_arguments, add_method_subcommands, build_run_name,
    config_from_args, method_config_dict, nonnegative_int,
    prepare_run_directory, write_run_config,
    require_bool, require_nonnegative_int_value,
    resolve_psf_parameters,
)
from evaluate import full_evaluate
from h5_loader import EpisodeFrameReader, load_episode_h5
from pipeline import build_comparison_canvas, compute_episode_psf, deblur_rgb
from robot_configs import HAND_EYE_CONFIGS, get_robot


def process_frame(
    h5_path,
    frame_idx,
    *,
    method_config,
    depth=0.5,
    exposure=None,
    fx=None,
    fy=None,
    psf_sigma=0.0,
    output_dir="single_frame_output",
    overwrite=False,
):
    frame_idx = require_nonnegative_int_value("frame_idx", frame_idx)
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
    if frame_idx < 0 or frame_idx >= meta["num_frames"]:
        raise IndexError(
            f"frame {frame_idx} outside episode range 0..{meta['num_frames'] - 1}"
        )
    reader = EpisodeFrameReader(meta["images"])
    image_rgb = reader.read_frame(frame_idx)
    robot = get_robot("kinova-gen3")
    hand_eye = HAND_EYE_CONFIGS["kinova-gen3"]
    psf, du, dv = compute_episode_psf(
        meta,
        frame_idx,
        width=meta["W"],
        height=meta["H"],
        robot=robot,
        hand_eye=hand_eye,
        depth=physical.depth,
        fx=physical.fx,
        fy=physical.fy,
        exposure=physical.exposure,
        psf_sigma=physical.psf_sigma,
    )
    deblurred_rgb, _ = deblur_rgb(image_rgb, psf, method_config)
    evaluation = full_evaluate(image_rgb, deblurred_rgb)
    reader.close()

    print(f"Frame {frame_idx}: {meta['W']}x{meta['H']} RGB")
    print(f"Pixel displacement: du={du:.2f}, dv={dv:.2f}")
    print(f"PSF: {psf.shape[0]}x{psf.shape[1]}")
    print(
        f"Laplacian: {evaluation['laplacian_before']:.2f} -> "
        f"{evaluation['laplacian_after']:.2f}"
    )
    print(f"PSNR: {evaluation['PSNR_raw']:.2f} dB")
    print(f"SSIM: {evaluation['SSIM_raw']:.4f}")

    run_name = build_run_name(
        h5_path, method_config, depth=physical.depth,
        exposure=physical.exposure, fx=physical.fx, fy=physical.fy,
        psf_sigma=physical.psf_sigma, frame=frame_idx,
    )
    output_dir = prepare_run_directory(
        Path(output_dir) / run_name, overwrite=overwrite,
        known_entries={f"frame_{frame_idx:04d}_original.png",
                       f"frame_{frame_idx:04d}_deblurred.png",
                       f"frame_{frame_idx:04d}_comparison.png",
                       f"frame_{frame_idx:04d}_psf.png", "run_config.json"},
    )
    write_run_config(output_dir, {
        "entry_point": "process_one_frame", "h5": str(Path(h5_path).resolve()),
        "method": method_config_dict(method_config), "depth": physical.depth,
        "exposure": physical.exposure, "fx": physical.fx, "fy": physical.fy,
        "psf_sigma": physical.psf_sigma, "frame": frame_idx,
        "intrinsics_source": physical.intrinsics_source,
        "run_name": run_name,
    })
    prefix = f"frame_{frame_idx:04d}"
    Image.fromarray(image_rgb).save(output_dir / f"{prefix}_original.png")
    Image.fromarray(deblurred_rgb).save(output_dir / f"{prefix}_deblurred.png")
    comparison = build_comparison_canvas(
        image_rgb,
        deblurred_rgb,
        label_height=60,
        info=f"{method_config.method} du={du:.1f} dv={dv:.1f}",
    )
    Image.fromarray(comparison).save(output_dir / f"{prefix}_comparison.png")
    psf_image = cv2.resize(
        (psf / psf.max() * 255).astype(np.uint8),
        (200, 200),
        interpolation=cv2.INTER_NEAREST,
    )
    Image.fromarray(psf_image).save(output_dir / f"{prefix}_psf.png")
    return image_rgb, deblurred_rgb, evaluation


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_common_arguments(parser, default_h5="episode_0001.h5")
    parser.add_argument("--frame", type=nonnegative_int, default=0)
    parser.add_argument("--output", default="single_frame_output")
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
    process_frame(
        args.h5,
        args.frame,
        method_config=args.method_config,
        depth=args.depth,
        exposure=args.exposure,
        fx=args.fx,
        fy=args.fy,
        psf_sigma=args.psf_sigma,
        output_dir=args.output,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
