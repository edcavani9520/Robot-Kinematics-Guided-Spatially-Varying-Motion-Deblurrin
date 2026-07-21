# Deblurring Experiment Data Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate reproducible, whole-episode E1-E7 deblurring experiment data from the five current-format H5 episodes using the corrected 0.01 s exposure model.

**Architecture:** Extend `batch_analyze.py` so each run persists per-frame metrics and aggregate statistics, then add a matrix runner that defines, deduplicates, executes, and indexes the approved experiments. Store every new artifact below `experiment_results_20260719/` so deleted historical outputs and unrelated working-tree changes remain untouched.

**Tech Stack:** Python 3, NumPy, h5py, OpenCV, Pillow, csv/json, pytest.

---

### Task 1: Persist Whole-Episode Metrics

**Files:**
- Modify: `batch_analyze.py`
- Create: `tests/test_batch_analyze_outputs.py`

- [ ] **Step 1: Write a failing focused test**

Create a small two-frame H5 episode in `tmp_path`, monkeypatch the expensive PSF/deconvolution calls, run `analyze_episode`, and assert that the run directory contains `frame_metrics.csv`, `summary.json`, `metrics.txt`, `comparison.png`, `psf.png`, and `run_config.json`. Assert that `summary.json` contains `count`, `mean`, `std`, and `median` for every metric plus elapsed time and time per frame.

```python
def test_analyze_episode_persists_whole_episode_metrics(tmp_path, monkeypatch):
    result = analyze_episode(
        h5_path,
        method_config=WienerConfig(K=0.03),
        depth=0.5,
        exposure=0.01,
        fx=260.65,
        fy=260.65,
        output_root=tmp_path / "runs",
    )
    run_dir = Path(result["run_dir"])
    assert (run_dir / "frame_metrics.csv").is_file()
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["num_frames"] == 2
    assert set(summary["metrics"]["SSIM_raw"]) == {"mean", "std", "median"}
```

- [ ] **Step 2: Run the test and confirm it fails**

Run:

```powershell
python -m pytest tests/test_batch_analyze_outputs.py -v
```

Expected: failure because `frame_metrics.csv`, `summary.json`, and `run_dir` are not yet produced.

- [ ] **Step 3: Implement structured output**

In `analyze_episode`:

- retain every frame index and metric value;
- compute mean, population standard deviation, and median per metric;
- add `elapsed_seconds`, `seconds_per_frame`, and `processing_fps`;
- write `frame_metrics.csv` with one row per frame;
- write `summary.json` with physical parameters, method config, frame count, timing, and aggregate metrics;
- include the new filenames in `known_entries`;
- return a dictionary containing `run_dir`, `run_name`, and the structured summary.

Use standard-library `csv` and `json`; convert NumPy scalars to Python floats before serialization.

- [ ] **Step 4: Run focused and existing tests**

```powershell
python -m pytest tests/test_batch_analyze_outputs.py tests/test_color_evaluate.py tests/test_rgb_deblur.py -v
```

Expected: all selected tests pass.

### Task 2: Define and Index the E1-E7 Matrix

**Files:**
- Create: `experiment_suite.py`
- Create: `tests/test_experiment_suite.py`

- [ ] **Step 1: Write failing matrix tests**

Test that `build_experiment_specs()`:

- uses episode_0003 for E1-E6;
- uses episodes 0001, 0002, 0004, and 0005 for E7;
- contains the 13 Wiener K values, six RL iteration values, seven TV lambda values, three depths, three exposures, and four optional-component combinations;
- uses exposure 0.01 except in the exposure sensitivity runs;
- has no duplicate computational run keys after deduplication;
- never references the removed old H5 schema.

```python
def test_suite_uses_corrected_exposure_and_current_episodes():
    specs = build_experiment_specs(Path("."))
    assert {p.name for p in held_out_episode_paths(specs)} == {
        "episode_0001.h5", "episode_0002.h5",
        "episode_0004.h5", "episode_0005.h5",
    }
    assert all(s.exposure == 0.01 for s in specs if s.axis != "exposure")
```

- [ ] **Step 2: Run the test and confirm it fails**

```powershell
python -m pytest tests/test_experiment_suite.py -v
```

Expected: import failure because `experiment_suite.py` does not exist.

- [ ] **Step 3: Implement the matrix runner**

Create immutable `ExperimentSpec` values with:

```python
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
```

Define E1-E7 exactly as approved in the experiment design. Deduplicate identical computational configurations while retaining all experiment labels in the final index. Provide CLI arguments:

```text
--root
--output-root (default experiment_results_20260719)
--overwrite
--max-frames
--dry-run
--experiments (comma-separated subset of E1 through E7; default all)
```

For every unique run call `analyze_episode`, then write:

- `experiment_index.csv`: one row for each E1-E7 label linked to its run directory;
- `aggregate_metrics.csv`: flattened mean/std/median metrics and timing;
- `suite_manifest.json`: fixed physical defaults, episode identities, execution time, and all run specifications;
- `failures.json`: failed run and error details, omitted when there are no failures.

Exit nonzero if any H5 file is missing or any run fails. Do not delete arbitrary output directories. With `--overwrite`, rely on the known-entry safety in `analyze_episode`.

- [ ] **Step 4: Run the matrix tests**

```powershell
python -m pytest tests/test_experiment_suite.py -v
```

Expected: all matrix and serialization tests pass.

### Task 3: Validate the Five Input Episodes

**Files:**
- Generate: `experiment_results_20260719/data_audit.csv`

- [ ] **Step 1: Run schema and timing validation**

Load all five episodes through `load_episode_h5`. Record filename, frame count, image shape/dtype, state/action shapes, timestamp monotonicity, measured median FPS, declared FPS, physical metadata source, and duration.

- [ ] **Step 2: Reject incompatible inputs before computation**

Expected validated totals:

```text
episode_0001: 107 frames
episode_0002: 126 frames
episode_0003: 104 frames
episode_0004:  69 frames
episode_0005: 122 frames
total: 528 frames
```

Every image must be `(N, 240, 320, 3)` RGB uint8, proprio `(N, 8)`, action `(N, 7)`, with strictly increasing timestamps.

### Task 4: Smoke-Test the Corrected Pipeline

**Files:**
- Generate: `experiment_results_20260719/smoke/`

- [ ] **Step 1: Run three-method smoke tests**

Run the three E1 method configurations on five frames of episode_0003.

```powershell
python experiment_suite.py --experiments E1 --output-root experiment_results_20260719/smoke --max-frames 5
```

- [ ] **Step 2: Inspect structured outputs**

Confirm frame count 5, finite metrics, normalized finite PSFs, RGB image dimensions, valid JSON/CSV, and positive timings. Stop before the full run if any check fails.

### Task 5: Generate the Full Experiment Dataset

**Files:**
- Generate: `experiment_results_20260719/runs/`
- Generate: `experiment_results_20260719/experiment_index.csv`
- Generate: `experiment_results_20260719/aggregate_metrics.csv`
- Generate: `experiment_results_20260719/suite_manifest.json`

- [ ] **Step 1: Run all deduplicated E1-E7 configurations**

```powershell
python experiment_suite.py --output-root experiment_results_20260719
```

Expected: 45 unique computational runs covering every approved label, with all episode frames processed.

- [ ] **Step 2: Preserve progress and report failures**

The runner writes each completed run immediately. If interrupted, rerun without deleting completed results and use safe overwrite only for the exact known run directories that need regeneration.

### Task 6: Validate and Summarize Generated Data

**Files:**
- Generate: `experiment_results_20260719/validation_report.json`
- Generate: `experiment_results_20260719/selected_parameters.json`

- [ ] **Step 1: Validate completeness**

Check that every experiment label has an index row, every index target has `summary.json`, `frame_metrics.csv`, `run_config.json`, `comparison.png`, and `psf.png`, all expected frame counts match, and all numeric metrics are finite.

- [ ] **Step 2: Select development-episode parameters without held-out tuning**

Use episode_0003 only. Record the best conservative Wiener K, RL iteration count, and TV lambda using sharpness improvement constrained by artifact/content-preservation metrics and runtime. If K=0.03 remains acceptable, freeze it for paired LoRA data generation.

- [ ] **Step 3: Run the full regression suite**

```powershell
python -m pytest tests -v
```

Expected: all tests pass.

- [ ] **Step 4: Report output locations and caveats**

Report the selected parameters, total runs/frames, elapsed time, any failures, and links to the aggregate CSV, manifest, validation report, and representative comparisons. Do not claim sharp-ground-truth accuracy from input-to-output PSNR/SSIM.
