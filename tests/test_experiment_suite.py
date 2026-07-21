from pathlib import Path
import csv
import json

from cli_config import RLConfig, TVConfig, WienerConfig
from experiment_suite import (
    build_experiment_specs,
    deduplicate_specs,
    find_existing_result,
    run_suite,
)


def _for(specs, experiment):
    return [spec for spec in specs if spec.experiment == experiment]


def test_builds_complete_approved_experiment_matrix():
    specs = build_experiment_specs(Path("."))

    assert len(_for(specs, "E1")) == 3
    assert [spec.method_config.K for spec in _for(specs, "E2")] == [
        0.0001, 0.0005, 0.001, 0.003, 0.005, 0.007,
        0.01, 0.02, 0.03, 0.15, 0.2, 0.5, 1.0,
    ]
    assert [spec.method_config.iterations for spec in _for(specs, "E3")] == [
        5, 10, 15, 20, 30, 50,
    ]
    assert [spec.method_config.lam for spec in _for(specs, "E4")] == [
        0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.003, 0.004,
    ]
    assert len(_for(specs, "E5")) == 6
    assert len(_for(specs, "E6")) == 4
    assert len(_for(specs, "E7")) == 12


def test_uses_corrected_exposure_and_current_episode_allocation():
    specs = build_experiment_specs(Path("E:/dataset"))

    for spec in specs:
        if not (spec.experiment == "E5" and spec.axis == "exposure"):
            assert spec.exposure == 0.01

    assert {spec.h5_path.name for spec in specs if spec.experiment != "E7"} == {
        "episode_0003.h5"
    }
    assert {spec.h5_path.name for spec in _for(specs, "E7")} == {
        "episode_0001.h5",
        "episode_0002.h5",
        "episode_0004.h5",
        "episode_0005.h5",
    }


def test_physical_and_optional_ablation_values_are_exact():
    specs = build_experiment_specs(Path("."))
    physical = _for(specs, "E5")
    optional = _for(specs, "E6")

    assert [spec.depth for spec in physical if spec.axis == "depth"] == [0.3, 0.5, 0.8]
    assert [spec.exposure for spec in physical if spec.axis == "exposure"] == [
        0.005, 0.01, 0.02
    ]
    assert [
        (spec.method_config.adaptive_k, spec.psf_sigma) for spec in optional
    ] == [(False, 0.0), (True, 0.0), (False, 1.0), (True, 1.0)]


def test_deduplicates_to_45_computational_runs_without_losing_labels():
    specs = build_experiment_specs(Path("."))
    grouped = deduplicate_specs(specs)

    assert len(grouped) == 45
    assert sum(len(labels) for labels in grouped.values()) == len(specs) == 51
    assert all(labels for labels in grouped.values())

    baseline_key = next(
        key
        for key, labels in grouped.items()
        if any(label.experiment == "E2" and label.value == "0.03" for label in labels)
    )
    baseline_labels = {(label.experiment, label.axis) for label in grouped[baseline_key]}
    assert {("E1", "method"), ("E2", "K"), ("E5", "depth"),
            ("E5", "exposure"), ("E6", "optional")} <= baseline_labels


def test_method_types_match_each_experiment():
    specs = build_experiment_specs(Path("."))

    assert all(isinstance(spec.method_config, WienerConfig) for spec in _for(specs, "E2"))
    assert all(isinstance(spec.method_config, RLConfig) for spec in _for(specs, "E3"))
    assert all(isinstance(spec.method_config, TVConfig) for spec in _for(specs, "E4"))
    held_out_tv = [
        spec for spec in _for(specs, "E7")
        if isinstance(spec.method_config, TVConfig)
    ]
    assert {spec.method_config.lam for spec in held_out_tv} == {0.001}


def test_run_suite_writes_deduplicated_index_and_aggregate(tmp_path):
    for episode in (1, 2, 3, 4, 5):
        (tmp_path / f"episode_{episode:04d}.h5").touch()
    output = tmp_path / "results"
    calls = []

    def fake_analyze(h5_path, **kwargs):
        calls.append((Path(h5_path).name, kwargs))
        run_dir = output / "runs" / f"run_{len(calls)}"
        run_dir.mkdir(parents=True)
        result = {
            "run_dir": str(run_dir),
            "run_name": run_dir.name,
            "h5": str(Path(h5_path).resolve()),
            "num_frames": 5,
            "method": {"method": kwargs["method_config"].method},
            "physical": {
                "depth": kwargs["depth"],
                "exposure": kwargs["exposure"],
                "fx": kwargs["fx"],
                "fy": kwargs["fy"],
                "psf_sigma": kwargs["psf_sigma"],
            },
            "elapsed_seconds": 1.0,
            "seconds_per_frame": 0.2,
            "processing_fps": 5.0,
            "metrics": {"SSIM_raw": {"mean": 0.9, "std": 0.1, "median": 0.92}},
        }
        (run_dir / "summary.json").write_text(json.dumps(result), encoding="utf-8")
        return result

    report = run_suite(
        root=tmp_path,
        output_root=output,
        experiments={"E1"},
        max_frames=5,
        analyzer=fake_analyze,
    )

    assert report["unique_runs"] == 3
    assert report["labels"] == 3
    assert len(calls) == 3
    assert all(call[1]["exposure"] == 0.01 for call in calls)
    assert all(call[1]["max_frames"] == 5 for call in calls)

    with (output / "experiment_index.csv").open(newline="", encoding="utf-8") as handle:
        index_rows = list(csv.DictReader(handle))
    with (output / "aggregate_metrics.csv").open(newline="", encoding="utf-8") as handle:
        aggregate_rows = list(csv.DictReader(handle))
    manifest = json.loads((output / "suite_manifest.json").read_text(encoding="utf-8"))

    assert len(index_rows) == 3
    assert len(aggregate_rows) == 3
    assert "SSIM_raw_mean" in aggregate_rows[0]
    assert manifest["experiments"] == ["E1"]
    assert manifest["corrected_exposure_seconds"] == 0.01


def test_run_suite_rejects_unknown_experiment(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="unknown experiment"):
        run_suite(tmp_path, tmp_path / "out", experiments={"E8"}, analyzer=lambda: None)


def test_finds_only_exact_completed_result_for_safe_resume(tmp_path):
    spec = _for(build_experiment_specs(tmp_path), "E2")[0]
    run_dir = tmp_path / "runs" / "finished"
    run_dir.mkdir(parents=True)
    result = {
        "run_name": "finished",
        "h5": str(spec.h5_path.resolve()),
        "num_frames": 104,
        "method": {"K": spec.method_config.K, "adaptive_k": False, "method": "wiener"},
        "physical": {
            "depth": spec.depth, "exposure": spec.exposure,
            "fx": spec.fx, "fy": spec.fy, "psf_sigma": spec.psf_sigma,
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(result), encoding="utf-8")
    for filename in ("frame_metrics.csv", "run_config.json", "comparison.png",
                     "psf.png", "metrics.txt"):
        (run_dir / filename).touch()

    resumed = find_existing_result(spec, tmp_path / "runs")

    assert resumed["run_name"] == "finished"
    assert Path(resumed["run_dir"]) == run_dir.resolve()
