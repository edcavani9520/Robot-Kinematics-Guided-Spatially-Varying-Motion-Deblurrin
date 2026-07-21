import csv
import json

import pytest

from experiment_validation import validate_experiment_results


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_result_tree(tmp_path):
    root = tmp_path / "results"
    runs = root / "runs"
    runs.mkdir(parents=True)
    index = []
    aggregate = []
    selections = [("E2", "K", "0.03"), ("E3", "iterations", "5"),
                  ("E4", "lambda", "0.001")]
    for number, (experiment, axis, value) in enumerate(selections, start=1):
        name = f"run_{number}"
        run_dir = runs / name
        run_dir.mkdir()
        summary = {
            "run_name": name,
            "num_frames": 2,
            "metrics": {"SSIM_raw": {"mean": 0.99, "std": 0.01, "median": 0.99}},
        }
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        _write_csv(run_dir / "frame_metrics.csv", [
            {"frame": 0, "SSIM_raw": 0.98},
            {"frame": 1, "SSIM_raw": 1.0},
        ])
        for filename in ("run_config.json", "comparison.png", "psf.png", "metrics.txt"):
            (run_dir / filename).touch()
        index.append({
            "experiment": experiment, "axis": axis, "value": value,
            "run_name": name, "run_dir": str(run_dir),
        })
        aggregate.append({
            "run_name": name, "num_frames": 2, "processing_fps": 5.0,
            "laplacian_before_mean": 10.0,
            "laplacian_after_mean": 12.0,
            "tv_before_mean": 10.0, "tv_after_mean": 11.0,
            "edge_ratio_mean": 0.95, "SSIM_raw_mean": 0.99,
        })
    _write_csv(root / "experiment_index.csv", index)
    _write_csv(root / "aggregate_metrics.csv", aggregate)
    _write_csv(root / "data_audit.csv", [{"filename": "episode.h5", "frames": 6}])
    (root / "suite_manifest.json").write_text(json.dumps({
        "labels": 3, "unique_runs": 3, "completed_runs": 3,
    }), encoding="utf-8")
    return root


def test_validates_assets_and_writes_selected_parameters(tmp_path):
    root = _make_result_tree(tmp_path)

    report = validate_experiment_results(root, expected_runs=3, expected_labels=3)

    assert report["status"] == "pass"
    assert report["checks"]["complete_run_assets"] == 3
    selected = json.loads((root / "selected_parameters.json").read_text())
    assert selected["wiener_K"]["value"] == 0.03
    assert selected["rl_iterations"]["value"] == 5
    assert selected["tv_lambda"]["value"] == 0.001
    assert (root / "validation_report.json").is_file()
    with (root / "labeled_metrics.csv").open(newline="", encoding="utf-8") as handle:
        labeled = list(csv.DictReader(handle))
    assert len(labeled) == 3
    assert labeled[0]["experiment"] == "E2"
    assert "SSIM_raw_mean" in labeled[0]


def test_rejects_missing_run_asset(tmp_path):
    root = _make_result_tree(tmp_path)
    (root / "runs" / "run_1" / "psf.png").unlink()

    with pytest.raises(ValueError, match="missing required assets"):
        validate_experiment_results(root, expected_runs=3, expected_labels=3)
