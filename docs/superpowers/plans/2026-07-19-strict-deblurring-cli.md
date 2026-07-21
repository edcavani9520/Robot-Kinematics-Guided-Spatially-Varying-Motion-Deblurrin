# Strict Deblurring CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ambiguous deblurring arguments with method-specific subcommands, enforce numerical and runtime feasibility boundaries, and produce collision-safe, fully reproducible experiment outputs.

**Architecture:** A new `cli_config.py` owns immutable method configurations, argparse converters, parser construction, run naming, configuration manifests, and safe output preparation. `pipeline.py`, `batch_analyze.py`, and `process_one_frame.py` consume one method configuration instead of simultaneously accepting every algorithm parameter. Runtime PSF checks remain close to PSF construction, while entry-point-specific frame validation remains in each workflow.

**Tech Stack:** Python 3, argparse, dataclasses, pathlib, JSON, NumPy, OpenCV, pytest.

---

## File map

- Create `cli_config.py`: CLI types, method configs, parser helpers, run names, manifests, output safety.
- Create `tests/test_cli_config.py`: converter, subcommand, naming, manifest, and collision tests.
- Create `tests/test_cli_entrypoints.py`: parser-to-workflow propagation and method rejection tests.
- Modify `pipeline.py`: config-based dispatch, PSF/runtime checks, strict parser, manifest and CSV metadata.
- Modify `batch_analyze.py`: strict batch parser, selected-frame validation, reproducible output directory.
- Modify `process_one_frame.py`: strict one-frame parser, reproducible output directory.
- Create `kinova_batch.py`: compatibility alias to strict batch entry point.
- Modify `README.md`: new commands and parameter domains.
- Modify `实验参数与组合清单.md`: historical-version notice.

### Task 1: Immutable method configurations and numeric boundaries

**Files:**
- Create: `cli_config.py`
- Create: `tests/test_cli_config.py`

- [ ] **Step 1: Write failing converter and configuration tests**

```python
# tests/test_cli_config.py
import argparse
import math

import pytest

from cli_config import (
    RLConfig,
    TVConfig,
    WienerConfig,
    finite_nonnegative_float,
    finite_positive_float,
    nonnegative_int,
    positive_int,
)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_finite_positive_float_rejects_invalid_domain(value):
    with pytest.raises(argparse.ArgumentTypeError):
        finite_positive_float(value)


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "-inf"])
def test_finite_nonnegative_float_rejects_invalid_domain(value):
    with pytest.raises(argparse.ArgumentTypeError):
        finite_nonnegative_float(value)


@pytest.mark.parametrize("value", ["0", "-1", "1.5"])
def test_positive_int_rejects_nonpositive_or_noninteger(value):
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int(value)


def test_nonnegative_int_accepts_zero_and_rejects_negative():
    assert nonnegative_int("0") == 0
    with pytest.raises(argparse.ArgumentTypeError):
        nonnegative_int("-1")


def test_method_configs_have_only_relevant_fields():
    assert WienerConfig(K=0.03, adaptive_k=True).method == "wiener"
    assert RLConfig(iterations=5).method == "rl"
    assert TVConfig(lam=0.002).method == "tv"
    assert not hasattr(RLConfig(iterations=5), "K")
    assert not hasattr(TVConfig(lam=0.002), "iterations")
```

- [ ] **Step 2: Run the tests and confirm the missing-module failure**

Run: `python -m pytest tests/test_cli_config.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'cli_config'`.

- [ ] **Step 3: Implement converters and immutable method configs**

```python
# cli_config.py
import argparse
import math
from dataclasses import asdict, dataclass
from typing import Literal, Union


def finite_positive_float(text):
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return value


def finite_nonnegative_float(text):
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return value


def positive_int(text):
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def nonnegative_int(text):
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


@dataclass(frozen=True)
class WienerConfig:
    K: float = 0.01
    adaptive_k: bool = False
    method: Literal["wiener"] = "wiener"


@dataclass(frozen=True)
class RLConfig:
    iterations: int = 30
    method: Literal["rl"] = "rl"


@dataclass(frozen=True)
class TVConfig:
    lam: float = 0.002
    method: Literal["tv"] = "tv"


MethodConfig = Union[WienerConfig, RLConfig, TVConfig]
```

- [ ] **Step 4: Run the focused tests**

Run: `python -m pytest tests/test_cli_config.py -q`

Expected: all tests in this file pass.

- [ ] **Step 5: Review ownership before staging**

Run: `git diff -- cli_config.py tests/test_cli_config.py`

Expected: only the new strict-CLI implementation and tests appear. Stage only these files if the user requests implementation commits; otherwise leave them unstaged.

### Task 2: Method subparsers and configuration construction

**Files:**
- Modify: `cli_config.py`
- Modify: `tests/test_cli_config.py`

- [ ] **Step 1: Add failing parser tests**

```python
from cli_config import add_common_arguments, add_method_subcommands, config_from_args


def build_test_parser():
    parser = argparse.ArgumentParser()
    add_common_arguments(parser, h5_required=True)
    add_method_subcommands(parser)
    return parser


def test_wiener_subcommand_builds_wiener_config():
    args = build_test_parser().parse_args(
        ["--h5", "episode.h5", "wiener", "--K", "0.03", "--adaptive-k"]
    )
    assert config_from_args(args) == WienerConfig(K=0.03, adaptive_k=True)


def test_rl_rejects_wiener_parameter():
    with pytest.raises(SystemExit):
        build_test_parser().parse_args(
            ["--h5", "episode.h5", "rl", "--rl-iters", "5", "--K", "0.03"]
        )


def test_tv_rejects_rl_parameter():
    with pytest.raises(SystemExit):
        build_test_parser().parse_args(
            ["--h5", "episode.h5", "tv", "--tv-lam", "0.002", "--rl-iters", "5"]
        )
```

- [ ] **Step 2: Run and confirm missing helper failures**

Run: `python -m pytest tests/test_cli_config.py -q`

Expected: import failure naming `add_common_arguments`.

- [ ] **Step 3: Implement common arguments and required method subcommands**

```python
def add_common_arguments(parser, *, h5_required=False, default_h5=None):
    parser.add_argument("--h5", required=h5_required, default=default_h5)
    parser.add_argument("--depth", type=finite_positive_float, default=0.5)
    parser.add_argument("--exposure", type=finite_positive_float, default=0.03)
    parser.add_argument("--fx", type=finite_positive_float, default=733.37)
    parser.add_argument("--fy", type=finite_positive_float, default=733.37)
    parser.add_argument("--psf-sigma", type=finite_nonnegative_float, default=0.0)


def add_method_subcommands(parser):
    methods = parser.add_subparsers(dest="method", required=True)
    wiener = methods.add_parser("wiener")
    wiener.add_argument("--K", type=finite_positive_float, default=0.01)
    wiener.add_argument("--adaptive-k", action="store_true")
    rl = methods.add_parser("rl")
    rl.add_argument("--rl-iters", type=positive_int, default=30)
    tv = methods.add_parser("tv")
    tv.add_argument("--tv-lam", type=finite_positive_float, default=0.002)


def config_from_args(args):
    if args.method == "wiener":
        return WienerConfig(K=args.K, adaptive_k=args.adaptive_k)
    if args.method == "rl":
        return RLConfig(iterations=args.rl_iters)
    if args.method == "tv":
        return TVConfig(lam=args.tv_lam)
    raise ValueError(f"unsupported method: {args.method}")
```

- [ ] **Step 4: Run parser tests**

Run: `python -m pytest tests/test_cli_config.py -q`

Expected: all parser tests pass and incompatible parameters exit with argparse status 2.

### Task 3: Output naming, manifest, and safe collision handling

**Files:**
- Modify: `cli_config.py`
- Modify: `tests/test_cli_config.py`

- [ ] **Step 1: Add failing output-safety tests**

```python
import json
from pathlib import Path

from cli_config import build_run_name, prepare_run_directory, write_run_config


def test_run_name_contains_every_effective_parameter():
    name = build_run_name(
        "episode_0001.h5",
        WienerConfig(K=0.03, adaptive_k=True),
        depth=0.5,
        exposure=0.03,
        fx=733.37,
        fy=733.37,
        psf_sigma=0.0,
        max_frames=50,
        frame=12,
    )
    for token in ("wiener", "K0.03", "adpt1", "d0.5", "e0.03", "fx733.37", "fy733.37", "sig0", "n50", "f12"):
        assert token in name


def test_nonempty_run_directory_requires_overwrite(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "comparison.png").write_bytes(b"old")
    with pytest.raises(FileExistsError):
        prepare_run_directory(run_dir, overwrite=False, known_entries={"comparison.png"})


def test_overwrite_removes_only_known_entries(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "comparison.png").write_bytes(b"old")
    (run_dir / "user-note.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="unknown entries"):
        prepare_run_directory(run_dir, overwrite=True, known_entries={"comparison.png"})
    assert (run_dir / "user-note.txt").exists()


def test_manifest_round_trip(tmp_path):
    config = {"method": {"method": "rl", "iterations": 5}, "depth": 0.5}
    write_run_config(tmp_path, config)
    assert json.loads((tmp_path / "run_config.json").read_text(encoding="utf-8")) == config
```

- [ ] **Step 2: Run and confirm helper failures**

Run: `python -m pytest tests/test_cli_config.py -q`

Expected: imports fail for the new output helpers.

- [ ] **Step 3: Implement deterministic naming and conservative cleanup**

```python
import json
import shutil
from pathlib import Path


def _number(value):
    return format(float(value), ".12g")


def build_run_name(h5_path, method_config, *, depth, exposure, fx, fy,
                   psf_sigma, max_frames=None, frame=None):
    method = method_config.method
    if isinstance(method_config, WienerConfig):
        method_token = f"wiener_K{_number(method_config.K)}_adpt{int(method_config.adaptive_k)}"
    elif isinstance(method_config, RLConfig):
        method_token = f"rl_iter{method_config.iterations}"
    else:
        method_token = f"tv_lam{_number(method_config.lam)}"
    tokens = [
        Path(h5_path).stem, method_token, f"d{_number(depth)}",
        f"e{_number(exposure)}", f"fx{_number(fx)}", f"fy{_number(fy)}",
        f"sig{_number(psf_sigma)}",
    ]
    if max_frames is not None:
        tokens.append(f"n{max_frames}")
    if frame is not None:
        tokens.append(f"f{frame}")
    return "_".join(tokens)


def prepare_run_directory(run_dir, *, overwrite, known_entries):
    run_dir = Path(run_dir)
    if run_dir.exists():
        entries = {entry.name for entry in run_dir.iterdir()}
        if entries and not overwrite:
            raise FileExistsError(f"output directory is not empty: {run_dir}")
        unknown = entries - set(known_entries)
        if overwrite and unknown:
            raise FileExistsError(f"refusing overwrite; unknown entries: {sorted(unknown)}")
        if overwrite:
            for name in entries:
                target = run_dir / name
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_run_config(run_dir, config):
    path = Path(run_dir) / "run_config.json"
    path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")


def method_config_dict(method_config):
    return asdict(method_config)
```

- [ ] **Step 4: Run output-safety tests**

Run: `python -m pytest tests/test_cli_config.py -q`

Expected: all tests pass; unknown user files are never removed.

### Task 4: Config-based deconvolution and PSF feasibility checks

**Files:**
- Modify: `pipeline.py:121-168`
- Create: `tests/test_pipeline_config.py`

- [ ] **Step 1: Add failing dispatch and PSF validation tests**

```python
# tests/test_pipeline_config.py
import numpy as np
import pytest

import pipeline
from cli_config import RLConfig, TVConfig, WienerConfig


def test_deblur_rgb_dispatches_only_configured_method(monkeypatch):
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    psf = np.ones((3, 3), dtype=float) / 9
    monkeypatch.setattr(pipeline, "richardson_lucy", lambda image, psf, iterations: image + iterations)
    result, effective_k = pipeline.deblur_rgb(image, psf, RLConfig(iterations=5))
    assert np.all(result == 5)
    assert effective_k is None


def test_validate_psf_rejects_nonfinite_displacement():
    with pytest.raises(ValueError, match="finite"):
        pipeline.validate_psf(np.ones((3, 3)) / 9, float("nan"), 1.0, (240, 320))


def test_validate_psf_rejects_kernel_larger_than_image():
    with pytest.raises(ValueError, match="exceeds image"):
        pipeline.validate_psf(np.ones((11, 11)) / 121, 1.0, 1.0, (10, 10))
```

- [ ] **Step 2: Run and confirm signature/helper failures**

Run: `python -m pytest tests/test_pipeline_config.py -q`

Expected: failures show the old `deblur_rgb` signature and missing `validate_psf`.

- [ ] **Step 3: Replace multi-method parameters with one config**

```python
from cli_config import RLConfig, TVConfig, WienerConfig


def effective_wiener_k(config, psf):
    value = config.K
    if config.adaptive_k:
        value *= 1.0 + 0.3 * np.log2(max(psf.shape[0], 3) / 17.0)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"effective Wiener K must be finite and positive; got {value}")
    return float(value)


def validate_psf(psf, du, dv, image_shape):
    psf = np.asarray(psf)
    if not np.isfinite([du, dv]).all() or not np.isfinite(psf).all():
        raise ValueError("PSF and displacement must contain only finite values")
    if psf.ndim != 2 or psf.size == 0 or psf.sum() <= 0:
        raise ValueError("PSF must be a non-empty normalized 2-D kernel")
    height, width = image_shape[:2]
    if psf.shape[0] > height or psf.shape[1] > width:
        raise ValueError(f"PSF shape {psf.shape} exceeds image shape {(height, width)}")


def deblur_rgb(image_rgb, psf, method_config):
    if isinstance(method_config, TVConfig):
        return tv_deconv(image_rgb, psf, lam=method_config.lam), None
    if isinstance(method_config, RLConfig):
        return richardson_lucy(image_rgb, psf, iterations=method_config.iterations), None
    if isinstance(method_config, WienerConfig):
        effective_k = effective_wiener_k(method_config, psf)
        return wiener_deconvolution(image_rgb, psf, K=effective_k), effective_k
    raise TypeError(f"unsupported method configuration: {type(method_config).__name__}")
```

Call `validate_psf(psf, du, dv, (height, width))` at the end of `compute_episode_psf` and return the validated kernel.

- [ ] **Step 4: Run focused and existing deconvolution tests**

Run: `python -m pytest tests/test_pipeline_config.py tests/test_rgb_deblur.py -q`

Expected: all tests pass after updating existing direct calls to construct the appropriate config and unpack `(image, effective_k)`.

### Task 5: Strict pipeline CLI and reproducible pipeline output

**Files:**
- Modify: `pipeline.py:171-320`
- Create: `tests/test_cli_entrypoints.py`

- [ ] **Step 1: Add failing pipeline parser propagation tests**

```python
# tests/test_cli_entrypoints.py
import pytest

import pipeline
from cli_config import RLConfig


def test_pipeline_parser_builds_rl_configuration():
    args = pipeline._parse_args([
        "--h5", "episode.h5", "--output", "out", "--max-frames", "5",
        "rl", "--rl-iters", "7",
    ])
    assert args.method_config == RLConfig(iterations=7)
    assert args.max_frames == 5


def test_pipeline_rejects_zero_max_frames():
    with pytest.raises(SystemExit):
        pipeline._parse_args([
            "--h5", "episode.h5", "--max-frames", "0", "wiener"
        ])
```

- [ ] **Step 2: Run and confirm old parser failure**

Run: `python -m pytest tests/test_cli_entrypoints.py::test_pipeline_parser_builds_rl_configuration -q`

Expected: failure because `_parse_args` does not accept an argument list or build `method_config`.

- [ ] **Step 3: Implement strict parser and config-based workflow signature**

Use `add_common_arguments`, `add_method_subcommands`, `config_from_args`, `positive_int`, `build_run_name`, `prepare_run_directory`, and `write_run_config`. Define `_parse_args(argv=None)` and call `parser.parse_args(argv)`. Remove `--robot` and `--hand-eye`. Change `run_h5_pipeline` to accept `method_config` and `overwrite` rather than `method`, `K`, `rl_iters`, `tv_lam`, and `adaptive_k`.

```python
def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, h5_required=True)
    parser.add_argument("--output", default="deblur_output")
    parser.add_argument("--max-frames", type=positive_int)
    parser.add_argument("--overwrite", action="store_true")
    add_method_subcommands(parser)
    args = parser.parse_args(argv)
    args.method_config = config_from_args(args)
    return args
```

Build a configuration-derived child directory under `--output`, prepare it before opening writers, and write `run_config.json`. Add `effective_K` to `psf_report.csv`; write an empty value for RL and TV. Replace the truthiness limit expression with:

```python
limit = meta["num_frames"] if max_frames is None else min(meta["num_frames"], max_frames)
```

Check both video writers with `isOpened()` and raise `RuntimeError` after releasing either opened writer if initialization fails.

- [ ] **Step 4: Run pipeline parser and output tests**

Run: `python -m pytest tests/test_cli_entrypoints.py tests/test_rgb_output.py -q`

Expected: all tests pass.

### Task 6: Strict batch and one-frame entry points

**Files:**
- Modify: `batch_analyze.py`
- Modify: `process_one_frame.py`
- Modify: `tests/test_cli_entrypoints.py`

- [ ] **Step 1: Add failing entry-point tests**

```python
import batch_analyze
import process_one_frame
from cli_config import TVConfig, WienerConfig


def test_batch_parser_builds_tv_config_and_valid_indices():
    args = batch_analyze._parse_args([
        "--h5", "episode.h5", "--max-frames", "20", "--show-frame", "4",
        "tv", "--tv-lam", "0.003",
    ])
    assert args.method_config == TVConfig(lam=0.003)


def test_single_parser_builds_wiener_config():
    args = process_one_frame._parse_args([
        "--h5", "episode.h5", "--frame", "3", "--reverse-psf",
        "wiener", "--K", "0.03", "--adaptive-k",
    ])
    assert args.method_config == WienerConfig(K=0.03, adaptive_k=True)


def test_selected_frame_is_not_silently_clamped():
    with pytest.raises(IndexError, match="outside processed range"):
        batch_analyze.validate_selected_frame(20, 20)
```

- [ ] **Step 2: Run and confirm old parser failures**

Run: `python -m pytest tests/test_cli_entrypoints.py -q`

Expected: failures identify missing argv support, method configs, and selected-frame validator.

- [ ] **Step 3: Refactor batch workflow**

Change `analyze_episode` to accept `method_config` and `overwrite`. Validate selection exactly:

```python
def validate_selected_frame(selected, limit):
    if selected < 0 or selected >= limit:
        raise IndexError(f"selected frame {selected} outside processed range 0..{limit - 1}")
    return selected
```

Default selection remains `limit // 2`; explicitly supplied `--show-frame` must pass validation. Use the shared run name, safe directory preparation, and manifest. Remove the old partial folder-name logic and include all effective parameters.

- [ ] **Step 4: Refactor one-frame workflow**

Change `process_frame` to accept `method_config` and `overwrite`. Preserve the existing frame range check and reverse-PSF behavior. Use the shared run name under `--output`, safe directory preparation, and manifest. Its known generated entries are the four PNG files plus `run_config.json`.

- [ ] **Step 5: Run focused entry-point tests**

Run: `python -m pytest tests/test_cli_entrypoints.py -q`

Expected: all tests pass and incompatible method parameters exit during parsing.

### Task 7: Compatibility alias and documentation

**Files:**
- Create: `kinova_batch.py`
- Modify: `README.md:35-76`
- Modify: `实验参数与组合清单.md:1`

- [ ] **Step 1: Recreate the compatibility alias**

```python
"""Compatibility command for the strict RGB batch analyzer."""

from batch_analyze import main


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Replace README commands with subcommand syntax**

Document commands in the parser order, for example:

```powershell
python process_one_frame.py --h5 episode_0001.h5 --frame 50 wiener --K 0.03
python batch_analyze.py --h5 episode_0001.h5 --max-frames 50 rl --rl-iters 5
python pipeline.py --h5 episode_0001.h5 --output runs tv --tv-lam 0.002
```

Document the numeric domains, method-only parameters, run-specific child directory, `run_config.json`, and `--overwrite` refusal rules. Remove `--robot`, `--hand-eye`, and old `--method` examples.

- [ ] **Step 3: Mark the experiment inventory as historical**

Add this notice immediately after its title:

```markdown
> 历史版本说明：本清单记录统一 RGB/严格 CLI 改造之前的实验。数据帧数、分辨率、输出命名和 `--method` 命令形式不代表当前代码；当前用法以 `README.md` 为准。
```

- [ ] **Step 4: Verify all help surfaces**

Run:

```powershell
python pipeline.py --help
python pipeline.py wiener --help
python pipeline.py rl --help
python pipeline.py tv --help
python batch_analyze.py --help
python process_one_frame.py --help
python kinova_batch.py --help
```

Expected: top-level help lists required `wiener`, `rl`, and `tv` subcommands; method help displays only its relevant parameters; robot and hand-eye arguments are absent.

### Task 8: Full verification and ownership-safe handoff

**Files:**
- Verify all Python files, tests, and documentation touched above.

- [ ] **Step 1: Run syntax and full automated tests**

Run:

```powershell
python -m compileall -q cli_config.py pipeline.py batch_analyze.py process_one_frame.py kinova_batch.py tests
python -m pytest -q
```

Expected: compilation exits 0 and the complete suite passes, including the original 24 tests and all new strict-CLI tests.

- [ ] **Step 2: Run read-only parser rejection checks**

Run:

```powershell
python pipeline.py --h5 episode_0001.h5 --max-frames 0 wiener
python batch_analyze.py --h5 episode_0001.h5 rl --rl-iters 5 --K 0.03
python process_one_frame.py --h5 episode_0001.h5 tv --tv-lam 0
```

Expected: every command exits with status 2 before loading H5 or creating output.

- [ ] **Step 3: Run one valid smoke command in a temporary output root**

Run:

```powershell
$strictCliTmp = Join-Path $env:TEMP 'ece4512-strict-cli-smoke'
python process_one_frame.py --h5 episode_0001.h5 --frame 0 --output $strictCliTmp wiener --K 0.03
```

Expected: command exits 0 and creates a configuration-derived child directory containing four PNG files and `run_config.json`.

- [ ] **Step 4: Verify overwrite protection**

Repeat the smoke command without `--overwrite` and expect `FileExistsError`. Repeat with `--overwrite` and expect success. Confirm `run_config.json` contains the exact command configuration.

- [ ] **Step 5: Review the final diff without absorbing unrelated work**

Run:

```powershell
git -c safe.directory='E:/Vital_document/CUHKSZ/课程文件/ECE4512/Final' status --short
git -c safe.directory='E:/Vital_document/CUHKSZ/课程文件/ECE4512/Final' diff -- cli_config.py pipeline.py batch_analyze.py process_one_frame.py kinova_batch.py README.md 实验参数与组合清单.md tests
```

Expected: strict-CLI changes are identifiable; pre-existing H5 and unrelated document changes remain untouched. Do not create an implementation commit that would absorb pre-existing user edits unless the user explicitly requests that commit scope.
