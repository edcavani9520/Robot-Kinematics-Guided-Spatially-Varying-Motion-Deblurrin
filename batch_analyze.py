"""Batch evaluation for current-format RGB Kinova episodes."""

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from cli_config import (
    add_common_arguments, add_method_subcommands, build_run_name,
    config_from_args, method_config_dict, nonnegative_int, positive_int,
    prepare_run_directory, require_bool, require_nonnegative_int_value,
    require_optional_positive_int_value, resolve_psf_parameters, write_run_config,
)
from evaluate import full_evaluate
from h5_loader import EpisodeFrameReader, load_episode_h5
from pipeline import build_comparison_canvas, compute_episode_psf, deblur_rgb
from robot_configs import HAND_EYE_CONFIGS, get_robot


def analyze_episode(
    h5_path,
    *,
    method_config,
    depth=0.5,
    exposure=None,
    fx=None,
    fy=None,
    psf_sigma=0.0,
    max_frames=None,
    show_frame=None,
    output_root="batch_analyze",
    overwrite=False,
):
    """Evaluate RGB deconvolution over an episode and save one comparison."""
    max_frames = require_optional_positive_int_value("max_frames", max_frames)
    if show_frame is not None:
        show_frame = require_nonnegative_int_value("show_frame", show_frame)
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
    hand_eye = HAND_EYE_CONFIGS["kinova-gen3"]
    limit = meta["num_frames"] if max_frames is None else min(meta["num_frames"], max_frames)
    selected = limit // 2 if show_frame is None else show_frame
    if selected < 0 or selected >= limit:
        raise IndexError(
            f"selected frame {selected} outside processed range 0..{limit - 1}"
        )

    metric_names = (
        "PSNR_raw",
        "SSIM_raw",
        "PSNR_matched",
        "SSIM_matched",
        "laplacian_before",
        "laplacian_after",
        "tenengrad_before",
        "tenengrad_after",
        "tv_before",
        "tv_after",
        "edge_ratio",
    )
    values = {name: [] for name in metric_names}
    frame_rows = []
    selected_result = None
    start = time.perf_counter()

    for frame_idx in range(limit):
        image_rgb = reader.read_frame(frame_idx)
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
        for name in metric_names:
            values[name].append(evaluation[name])
        frame_rows.append({
            "frame": frame_idx,
            **{name: float(evaluation[name]) for name in metric_names},
        })
        if frame_idx == selected:
            selected_result = (image_rgb, deblurred_rgb, evaluation, psf, du, dv)
        if (frame_idx + 1) % 25 == 0 or frame_idx + 1 == limit:
            print(f"  [{frame_idx + 1}/{limit}] RGB frames")

    reader.close()
    elapsed = time.perf_counter() - start
    metric_summary = {
        name: {
            "mean": float(np.mean(series)),
            "std": float(np.std(series)),
            "median": float(np.median(series)),
        }
        for name, series in values.items()
    }
    summary = {name: stats["mean"] for name, stats in metric_summary.items()}
    seconds_per_frame = elapsed / limit
    processing_fps = limit / elapsed
    print(f"Processed {limit} RGB frames in {elapsed:.1f}s")
    print(
        f"Laplacian: {summary['laplacian_before']:.2f} -> "
        f"{summary['laplacian_after']:.2f}"
    )
    print(
        f"Tenengrad: {summary['tenengrad_before']:.2e} -> "
        f"{summary['tenengrad_after']:.2e}"
    )
    print(f"PSNR: {summary['PSNR_raw']:.2f} dB  SSIM: {summary['SSIM_raw']:.4f}")
    print(f"Edge ratio: {summary['edge_ratio']:.3f}")

    if selected_result is not None:
        image_rgb, deblurred_rgb, evaluation, psf, du, dv = selected_result
        run_name = build_run_name(
            h5_path, method_config, depth=physical.depth,
            exposure=physical.exposure, fx=physical.fx, fy=physical.fy,
            psf_sigma=physical.psf_sigma,
            max_frames=max_frames, frame=selected,
        )
        output_dir = prepare_run_directory(
            Path(output_root) / run_name, overwrite=overwrite,
            known_entries={
                "comparison.png", "psf.png", "metrics.txt", "run_config.json",
                "frame_metrics.csv", "summary.json",
            },
        )
        write_run_config(output_dir, {
            "entry_point": "batch_analyze", "h5": str(Path(h5_path).resolve()),
            "method": method_config_dict(method_config), "depth": physical.depth,
            "exposure": physical.exposure, "fx": physical.fx, "fy": physical.fy,
            "psf_sigma": physical.psf_sigma, "max_frames": max_frames,
            "intrinsics_source": physical.intrinsics_source,
            "show_frame": selected, "run_name": run_name,
        })
        parameter = str(method_config)
        comparison = build_comparison_canvas(
            image_rgb,
            deblurred_rgb,
            label_height=60,
            info=f"{method_config.method} {parameter}",
        )
        Image.fromarray(comparison).save(output_dir / "comparison.png")
        psf_image = cv2.resize(
            (psf / psf.max() * 255).astype(np.uint8),
            (200, 200),
            interpolation=cv2.INTER_NEAREST,
        )
        Image.fromarray(psf_image).save(output_dir / "psf.png")
        with (output_dir / "frame_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=("frame", *metric_names))
            writer.writeheader()
            writer.writerows(frame_rows)
        structured_summary = {
            "run_name": run_name,
            "h5": str(Path(h5_path).resolve()),
            "num_frames": limit,
            "method": method_config_dict(method_config),
            "physical": {
                "depth": physical.depth,
                "exposure": physical.exposure,
                "fx": physical.fx,
                "fy": physical.fy,
                "psf_sigma": physical.psf_sigma,
                "intrinsics_source": physical.intrinsics_source,
            },
            "elapsed_seconds": elapsed,
            "seconds_per_frame": seconds_per_frame,
            "processing_fps": processing_fps,
            "metrics": metric_summary,
        }
        with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(structured_summary, f, ensure_ascii=False, indent=2)
        with (output_dir / "metrics.txt").open("w", encoding="utf-8") as f:
            f.write(f"Frame: {selected}\nMethod: {method_config.method}\nParameter: {parameter}\n")
            f.write(
                f"Depth: {physical.depth:.3f} m\n"
                f"Displacement: du={du:.2f}, dv={dv:.2f}\n"
            )
            f.write(f"PSF: {psf.shape[0]}x{psf.shape[1]}\n")
            f.write(
                f"Laplacian: {evaluation['laplacian_before']:.2f} -> "
                f"{evaluation['laplacian_after']:.2f}\n"
            )
            f.write(f"PSNR: {evaluation['PSNR_raw']:.2f} dB\n")
            f.write(f"SSIM: {evaluation['SSIM_raw']:.4f}\n")
        print(f"Saved RGB comparison to {output_dir}")
    return {
        "run_dir": str(output_dir.resolve()),
        **structured_summary,
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_common_arguments(parser, default_h5="episode_0001.h5")
    parser.add_argument("--max-frames", type=positive_int)
    parser.add_argument("--show-frame", type=nonnegative_int)
    parser.add_argument("--output-root", default="batch_analyze")
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
    analyze_episode(
        args.h5,
        method_config=args.method_config,
        depth=args.depth,
        exposure=args.exposure,
        fx=args.fx,
        fy=args.fy,
        psf_sigma=args.psf_sigma,
        max_frames=args.max_frames,
        show_frame=args.show_frame,
        output_root=args.output_root,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
