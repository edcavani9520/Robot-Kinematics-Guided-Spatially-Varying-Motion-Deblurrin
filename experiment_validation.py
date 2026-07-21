"""Validate generated deblurring experiment artifacts and freeze parameters."""

import argparse
import csv
import json
import math
from pathlib import Path


REQUIRED_RUN_ASSETS = {
    "summary.json",
    "frame_metrics.csv",
    "run_config.json",
    "comparison.png",
    "psf.png",
    "metrics.txt",
}


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _selection(index_rows, aggregate_by_name, experiment, axis, value):
    row = next(
        row for row in index_rows
        if row["experiment"] == experiment
        and row["axis"] == axis
        and row["value"] == value
    )
    aggregate = aggregate_by_name[row["run_name"]]
    laplacian_ratio = (
        float(aggregate["laplacian_after_mean"])
        / float(aggregate["laplacian_before_mean"])
    )
    tv_ratio = (
        float(aggregate["tv_after_mean"])
        / float(aggregate["tv_before_mean"])
    )
    return {
        "value": float(value),
        "run_name": row["run_name"],
        "evidence": {
            "laplacian_ratio": laplacian_ratio,
            "tv_ratio": tv_ratio,
            "edge_ratio": float(aggregate["edge_ratio_mean"]),
            "input_output_ssim": float(aggregate["SSIM_raw_mean"]),
            "processing_fps": float(aggregate["processing_fps"]),
        },
    }


def _write_csv(path, rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_experiment_results(output_root, *, expected_runs=45, expected_labels=51):
    """Validate completeness/validity and write inspectable JSON reports."""
    output_root = Path(output_root)
    index_rows = _read_csv(output_root / "experiment_index.csv")
    aggregate_rows = _read_csv(output_root / "aggregate_metrics.csv")
    audit_rows = _read_csv(output_root / "data_audit.csv")
    manifest = json.loads(
        (output_root / "suite_manifest.json").read_text(encoding="utf-8")
    )
    if len(index_rows) != expected_labels:
        raise ValueError(
            f"expected {expected_labels} experiment labels, found {len(index_rows)}"
        )
    if len(aggregate_rows) != expected_runs:
        raise ValueError(
            f"expected {expected_runs} aggregate runs, found {len(aggregate_rows)}"
        )
    if len({row["run_name"] for row in aggregate_rows}) != expected_runs:
        raise ValueError("aggregate run_name values must be unique")
    if manifest.get("completed_runs") != expected_runs:
        raise ValueError("manifest completed_runs does not match expected runs")

    aggregate_by_name = {row["run_name"]: row for row in aggregate_rows}
    indexed_names = {row["run_name"] for row in index_rows}
    if indexed_names != set(aggregate_by_name):
        raise ValueError("experiment index and aggregate run names do not match")

    complete_assets = 0
    frame_rows_total = 0
    for run_name in sorted(indexed_names):
        index_row = next(row for row in index_rows if row["run_name"] == run_name)
        run_dir = Path(index_row["run_dir"])
        existing = {path.name for path in run_dir.iterdir()} if run_dir.is_dir() else set()
        missing = REQUIRED_RUN_ASSETS - existing
        if missing:
            raise ValueError(
                f"run {run_name} missing required assets: {sorted(missing)}"
            )
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        frame_rows = _read_csv(run_dir / "frame_metrics.csv")
        if len(frame_rows) != int(summary["num_frames"]):
            raise ValueError(f"run {run_name} frame row count does not match summary")
        if int(aggregate_by_name[run_name]["num_frames"]) != len(frame_rows):
            raise ValueError(f"run {run_name} aggregate frame count does not match")
        complete_assets += 1
        frame_rows_total += len(frame_rows)

    nonfinite = []
    for row in aggregate_rows:
        for column, text in row.items():
            if text in (None, ""):
                continue
            try:
                value = float(text)
            except ValueError:
                continue
            if not math.isfinite(value):
                nonfinite.append({"run_name": row["run_name"], "column": column})
    if nonfinite:
        raise ValueError(f"non-finite aggregate values: {nonfinite[:5]}")

    labeled_rows = []
    for index_row in index_rows:
        labeled_rows.append({
            "experiment": index_row["experiment"],
            "axis": index_row["axis"],
            "value": index_row["value"],
            **aggregate_by_name[index_row["run_name"]],
        })
    _write_csv(output_root / "labeled_metrics.csv", labeled_rows)

    selected = {
        "physical_defaults": {
            "depth_m": 0.5,
            "exposure_seconds": 0.01,
            "fx_px": 260.6450745682411,
            "fy_px": 260.6450745682411,
            "psf_sigma": 0.0,
            "adaptive_k": False,
        },
        "wiener_K": _selection(
            index_rows, aggregate_by_name, "E2", "K", "0.03"
        ),
        "rl_iterations": _selection(
            index_rows, aggregate_by_name, "E3", "iterations", "5"
        ),
        "tv_lambda": _selection(
            index_rows, aggregate_by_name, "E4", "lambda", "0.001"
        ),
        "selection_basis": (
            "Episode 0003 only; combined sharpness, TV growth, edge consistency, "
            "input preservation, runtime, and representative-frame inspection."
        ),
        "downstream_preprocessing": "Wiener K=0.03 with the physical defaults above",
    }
    with (output_root / "selected_parameters.json").open("w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)

    report = {
        "status": "pass",
        "grain": "one aggregate row per unique computational run",
        "checks": {
            "experiment_labels": len(index_rows),
            "unique_aggregate_runs": len(aggregate_rows),
            "complete_run_assets": complete_assets,
            "per_frame_metric_rows": frame_rows_total,
            "input_episode_rows": len(audit_rows),
            "input_frames": sum(int(row["frames"]) for row in audit_rows),
            "nonfinite_aggregate_values": len(nonfinite),
        },
        "manifest": {
            "labels": manifest.get("labels"),
            "unique_runs": manifest.get("unique_runs"),
            "completed_runs": manifest.get("completed_runs"),
            "corrected_exposure_seconds": manifest.get("corrected_exposure_seconds"),
        },
        "known_analytical_limits": [
            "Input-to-output PSNR/SSIM are content-preservation metrics, not sharp-ground-truth accuracy.",
            "Laplacian and Tenengrad can reward noise or ringing and require artifact guardrails.",
            "The five H5 files declare 10 fps but have measured median rates near 2.6-3.1 fps.",
            "Camera intrinsics, hand-eye transform, and 0.5 m depth are estimates.",
        ],
    }
    with (output_root / "validation_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args(argv)
    report = validate_experiment_results(args.output_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
