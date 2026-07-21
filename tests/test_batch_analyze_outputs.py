import csv
import json
from pathlib import Path

import numpy as np
import pytest

import batch_analyze
from cli_config import WienerConfig


METRICS = {
    "PSNR_raw": 30.0,
    "SSIM_raw": 0.9,
    "PSNR_matched": 31.0,
    "SSIM_matched": 0.91,
    "laplacian_before": 10.0,
    "laplacian_after": 12.0,
    "tenengrad_before": 20.0,
    "tenengrad_after": 24.0,
    "tv_before": 30.0,
    "tv_after": 33.0,
    "edge_ratio": 0.8,
}


def test_analyze_episode_persists_whole_episode_metrics(tmp_path, monkeypatch):
    images = np.zeros((2, 3, 4, 3), dtype=np.uint8)
    images[1] = 20
    meta = {
        "images": images,
        "joint_positions": np.zeros((2, 7)),
        "joint_velocities": np.zeros((2, 7)),
        "num_frames": 2,
        "H": 3,
        "W": 4,
        "camera_intrinsics": None,
        "exposure_seconds": None,
    }
    calls = {"evaluation": 0}

    monkeypatch.setattr(batch_analyze, "load_episode_h5", lambda _: meta)
    monkeypatch.setattr(
        batch_analyze,
        "compute_episode_psf",
        lambda *args, **kwargs: (np.array([[1.0]]), 0.25, -0.5),
    )
    monkeypatch.setattr(
        batch_analyze,
        "deblur_rgb",
        lambda image, psf, method: (image.copy(), None),
    )

    def fake_evaluate(original, processed):
        offset = calls["evaluation"]
        calls["evaluation"] += 1
        return {key: value + offset for key, value in METRICS.items()}

    monkeypatch.setattr(batch_analyze, "full_evaluate", fake_evaluate)

    result = batch_analyze.analyze_episode(
        "episode_current.h5",
        method_config=WienerConfig(K=0.03),
        depth=0.5,
        exposure=0.01,
        fx=260.65,
        fy=260.65,
        output_root=tmp_path / "runs",
    )

    run_dir = Path(result["run_dir"])
    expected = {
        "frame_metrics.csv",
        "summary.json",
        "metrics.txt",
        "comparison.png",
        "psf.png",
        "run_config.json",
    }
    assert expected <= {path.name for path in run_dir.iterdir()}

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["num_frames"] == 2
    assert summary["method"] == {"method": "wiener", "K": 0.03, "adaptive_k": False}
    assert summary["physical"]["exposure"] == 0.01
    assert summary["metrics"]["SSIM_raw"]["mean"] == pytest.approx(1.4)
    assert summary["metrics"]["SSIM_raw"]["std"] == pytest.approx(0.5)
    assert summary["metrics"]["SSIM_raw"]["median"] == pytest.approx(1.4)
    assert summary["elapsed_seconds"] >= 0
    assert summary["seconds_per_frame"] >= 0
    assert summary["processing_fps"] > 0

    with (run_dir / "frame_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["frame"]) for row in rows] == [0, 1]
    assert float(rows[0]["SSIM_raw"]) == 0.9
    assert float(rows[1]["SSIM_raw"]) == 1.9
