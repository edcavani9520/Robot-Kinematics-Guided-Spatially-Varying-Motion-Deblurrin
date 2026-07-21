"""Run the corrected E1-E7 deblurring experiment matrix."""

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from batch_analyze import analyze_episode
from cli_config import (
    DEFAULT_FOCAL_LENGTH_PX,
    MethodConfig,
    RLConfig,
    TVConfig,
    WienerConfig,
    method_config_dict,
    positive_int,
)
from h5_loader import load_episode_h5


@dataclass(frozen=True)
class ExperimentSpec:
    experiment: str
    axis: str
    value: str
    h5_path: Path
    method_config: MethodConfig
    depth: float = 0.5
    exposure: float = 0.01
    fx: float = DEFAULT_FOCAL_LENGTH_PX
    fy: float = DEFAULT_FOCAL_LENGTH_PX
    psf_sigma: float = 0.0

    @property
    def run_key(self):
        method = json.dumps(method_config_dict(self.method_config), sort_keys=True)
        return (
            str(self.h5_path.resolve()),
            method,
            self.depth,
            self.exposure,
            self.fx,
            self.fy,
            self.psf_sigma,
        )


def build_experiment_specs(root: Path):
    """Build all approved E1-E7 labels before computational deduplication."""
    root = Path(root)
    development = root / "episode_0003.h5"
    specs = [
        ExperimentSpec("E1", "method", "wiener", development, WienerConfig(K=0.03)),
        ExperimentSpec("E1", "method", "rl", development, RLConfig(iterations=5)),
        ExperimentSpec("E1", "method", "tv", development, TVConfig(lam=0.002)),
    ]

    for value in (
        0.0001, 0.0005, 0.001, 0.003, 0.005, 0.007,
        0.01, 0.02, 0.03, 0.15, 0.2, 0.5, 1.0,
    ):
        specs.append(ExperimentSpec(
            "E2", "K", str(value), development, WienerConfig(K=value)
        ))

    for value in (5, 10, 15, 20, 30, 50):
        specs.append(ExperimentSpec(
            "E3", "iterations", str(value), development,
            RLConfig(iterations=value),
        ))

    for value in (0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.003, 0.004):
        specs.append(ExperimentSpec(
            "E4", "lambda", str(value), development, TVConfig(lam=value)
        ))

    for value in (0.3, 0.5, 0.8):
        specs.append(ExperimentSpec(
            "E5", "depth", str(value), development,
            WienerConfig(K=0.03), depth=value,
        ))
    for value in (0.005, 0.01, 0.02):
        specs.append(ExperimentSpec(
            "E5", "exposure", str(value), development,
            WienerConfig(K=0.03), exposure=value,
        ))

    for value, adaptive, sigma in (
        ("base", False, 0.0),
        ("adaptive", True, 0.0),
        ("smoothing", False, 1.0),
        ("adaptive+smoothing", True, 1.0),
    ):
        specs.append(ExperimentSpec(
            "E6", "optional", value, development,
            WienerConfig(K=0.03, adaptive_k=adaptive), psf_sigma=sigma,
        ))

    held_out = ("episode_0001.h5", "episode_0002.h5", "episode_0004.h5", "episode_0005.h5")
    methods = (
        ("wiener", WienerConfig(K=0.03)),
        ("rl", RLConfig(iterations=5)),
        ("tv", TVConfig(lam=0.001)),
    )
    for filename in held_out:
        for name, method in methods:
            specs.append(ExperimentSpec(
                "E7", "episode_method", f"{Path(filename).stem}:{name}",
                root / filename, method,
            ))
    return specs


def deduplicate_specs(specs):
    """Group experiment labels that share one exact computational run."""
    grouped = {}
    for spec in specs:
        grouped.setdefault(spec.run_key, []).append(spec)
    return grouped


def _spec_record(spec):
    return {
        "experiment": spec.experiment,
        "axis": spec.axis,
        "value": spec.value,
        "h5": str(spec.h5_path.resolve()),
        "method": method_config_dict(spec.method_config),
        "depth": spec.depth,
        "exposure": spec.exposure,
        "fx": spec.fx,
        "fy": spec.fy,
        "psf_sigma": spec.psf_sigma,
    }


def audit_episodes(root, output_root):
    """Validate all five current episodes and persist a compact audit table."""
    rows = []
    for episode in (1, 2, 3, 4, 5):
        path = Path(root) / f"episode_{episode:04d}.h5"
        if not path.is_file():
            raise FileNotFoundError(f"missing experiment episode: {path}")
        meta = load_episode_h5(path)
        timestamps = meta["timestamps"]
        rows.append({
            "filename": path.name,
            "frames": meta["num_frames"],
            "height": meta["H"],
            "width": meta["W"],
            "image_dtype": str(meta["images"].dtype),
            "proprio_shape": json.dumps(list(meta["proprio"].shape)),
            "action_shape": json.dumps(list(meta["actions"].shape)),
            "timestamps_strictly_increasing": bool(
                len(timestamps) < 2 or (timestamps[1:] > timestamps[:-1]).all()
            ),
            "duration_seconds": float(timestamps[-1] - timestamps[0])
            if len(timestamps) > 1 else 0.0,
            "measured_fps": meta["camera_fps"],
            "declared_fps": meta["camera_fps_declared"],
            "h5_exposure_seconds": meta["exposure_seconds"],
            "intrinsics_source": meta["camera_intrinsics_source"] or "fallback_fov",
        })
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "data_audit.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _flatten_result(result):
    row = {
        "run_name": result["run_name"],
        "run_dir": result["run_dir"],
        "h5": result["h5"],
        "num_frames": result["num_frames"],
        "elapsed_seconds": result["elapsed_seconds"],
        "seconds_per_frame": result["seconds_per_frame"],
        "processing_fps": result["processing_fps"],
    }
    for key, value in result["method"].items():
        row[f"method_{key}"] = value
    for key, value in result["physical"].items():
        row[f"physical_{key}"] = value
    for metric, stats in result["metrics"].items():
        for statistic, value in stats.items():
            row[f"{metric}_{statistic}"] = value
    return row


def find_existing_result(spec, runs_root, max_frames=None):
    """Return an exact completed result so interrupted suites can resume safely."""
    runs_root = Path(runs_root)
    if not runs_root.is_dir():
        return None
    expected_method = method_config_dict(spec.method_config)
    for summary_path in runs_root.glob("*/summary.json"):
        try:
            result = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        physical = result.get("physical", {})
        if (
            Path(result.get("h5", "")).resolve() == spec.h5_path.resolve()
            and result.get("method") == expected_method
            and physical.get("depth") == spec.depth
            and physical.get("exposure") == spec.exposure
            and physical.get("fx") == spec.fx
            and physical.get("fy") == spec.fy
            and physical.get("psf_sigma") == spec.psf_sigma
            and (max_frames is None or result.get("num_frames") <= max_frames)
        ):
            required = {
                "summary.json", "frame_metrics.csv", "run_config.json",
                "comparison.png", "psf.png", "metrics.txt",
            }
            if required <= {path.name for path in summary_path.parent.iterdir()}:
                result["run_dir"] = str(summary_path.parent.resolve())
                return result
    return None


def _write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_suite(
    root,
    output_root,
    *,
    experiments=None,
    max_frames=None,
    overwrite=False,
    analyzer=analyze_episode,
    perform_audit=False,
):
    """Run a selected experiment subset and write deduplicated indexes."""
    allowed = {f"E{number}" for number in range(1, 8)}
    selected = allowed if experiments is None else set(experiments)
    unknown = selected - allowed
    if unknown:
        raise ValueError(f"unknown experiment: {sorted(unknown)}")

    root = Path(root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if perform_audit:
        audit_episodes(root, output_root)

    specs = [
        spec for spec in build_experiment_specs(root)
        if spec.experiment in selected
    ]
    missing = sorted({str(spec.h5_path) for spec in specs if not spec.h5_path.is_file()})
    if missing:
        raise FileNotFoundError(f"missing experiment episodes: {missing}")
    grouped = deduplicate_specs(specs)
    results = {}
    failures = []
    runs_root = output_root / "runs"

    for run_number, (key, labels) in enumerate(grouped.items(), start=1):
        spec = labels[0]
        print(
            f"[suite {run_number}/{len(grouped)}] {spec.h5_path.name} "
            f"{method_config_dict(spec.method_config)}"
        )
        try:
            existing = None if overwrite else find_existing_result(
                spec, runs_root, max_frames=max_frames
            )
            if existing is not None:
                print(f"  [resume] {existing['run_name']}")
                results[key] = existing
                continue
            results[key] = analyzer(
                spec.h5_path,
                method_config=spec.method_config,
                depth=spec.depth,
                exposure=spec.exposure,
                fx=spec.fx,
                fy=spec.fy,
                psf_sigma=spec.psf_sigma,
                max_frames=max_frames,
                output_root=runs_root,
                overwrite=overwrite,
            )
        except Exception as exc:
            failures.append({
                "spec": _spec_record(spec),
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            break

    index_rows = []
    for key, labels in grouped.items():
        if key not in results:
            continue
        for label in labels:
            index_rows.append({
                "experiment": label.experiment,
                "axis": label.axis,
                "value": label.value,
                "run_name": results[key]["run_name"],
                "run_dir": results[key]["run_dir"],
            })
    aggregate_rows = [_flatten_result(result) for result in results.values()]
    _write_csv(output_root / "experiment_index.csv", index_rows)
    _write_csv(output_root / "aggregate_metrics.csv", aggregate_rows)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root.resolve()),
        "output_root": str(output_root.resolve()),
        "experiments": sorted(selected),
        "corrected_exposure_seconds": 0.01,
        "max_frames": max_frames,
        "labels": len(specs),
        "unique_runs": len(grouped),
        "completed_runs": len(results),
        "specs": [_spec_record(spec) for spec in specs],
    }
    with (output_root / "suite_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    failure_path = output_root / "failures.json"
    if failures:
        with failure_path.open("w", encoding="utf-8") as f:
            json.dump(failures, f, ensure_ascii=False, indent=2)
        raise RuntimeError(f"experiment suite failed: {failures[0]}")
    if failure_path.exists():
        failure_path.unlink()
    return {
        "labels": len(specs),
        "unique_runs": len(grouped),
        "completed_runs": len(results),
        "output_root": str(output_root.resolve()),
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-root", type=Path, default=Path("experiment_results_20260719")
    )
    parser.add_argument("--experiments", default="E1,E2,E3,E4,E5,E6,E7")
    parser.add_argument("--max-frames", type=positive_int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    experiments = {item.strip().upper() for item in args.experiments.split(",") if item.strip()}
    specs = [
        spec for spec in build_experiment_specs(args.root)
        if spec.experiment in experiments
    ]
    if args.dry_run:
        print(f"labels={len(specs)} unique_runs={len(deduplicate_specs(specs))}")
        return 0
    report = run_suite(
        args.root,
        args.output_root,
        experiments=experiments,
        max_frames=args.max_frames,
        overwrite=args.overwrite,
        perform_audit=True,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
