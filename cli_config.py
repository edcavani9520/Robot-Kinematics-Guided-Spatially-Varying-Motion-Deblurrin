"""Strict CLI validation and method-specific deblurring configurations."""

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Union


def finite_positive_float(text: str) -> float:
    """Parse a finite floating-point value strictly greater than zero."""
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return value


def finite_nonnegative_float(text: str) -> float:
    """Parse a finite floating-point value greater than or equal to zero."""
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return value


def positive_int(text: str) -> int:
    """Parse an integer that is at least one."""
    try:
        value = int(text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def nonnegative_int(text: str) -> int:
    """Parse an integer that is at least zero."""
    try:
        value = int(text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def require_bool(name: str, value: object) -> bool:
    """Validate a programmatic boolean without accepting integers as booleans."""
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")
    return value


def require_optional_positive_int_value(name: str, value: object):
    """Validate a direct-call optional positive integer."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer or None")
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def require_nonnegative_int_value(name: str, value: object) -> int:
    """Validate a direct-call non-negative integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True)
class WienerConfig:
    K: float = 0.01
    adaptive_k: bool = False
    method: Literal["wiener"] = field(default="wiener", init=False)

    def __post_init__(self) -> None:
        if type(self.adaptive_k) is not bool:
            raise TypeError("adaptive_k must be a boolean")
        if isinstance(self.K, bool) or not _is_finite_positive(self.K):
            raise ValueError("K must be finite and greater than zero")


@dataclass(frozen=True)
class RLConfig:
    iterations: int = 30
    method: Literal["rl"] = field(default="rl", init=False)

    def __post_init__(self) -> None:
        if isinstance(self.iterations, bool) or not isinstance(self.iterations, int):
            raise TypeError("iterations must be an integer")
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1")


@dataclass(frozen=True)
class TVConfig:
    lam: float = 0.002
    method: Literal["tv"] = field(default="tv", init=False)

    def __post_init__(self) -> None:
        if isinstance(self.lam, bool) or not _is_finite_positive(self.lam):
            raise ValueError("lam must be finite and greater than zero")


MethodConfig = Union[WienerConfig, RLConfig, TVConfig]
DEFAULT_EXPOSURE_SECONDS = 0.01
DEFAULT_DIAGONAL_FOV_DEG = 75.0
DEFAULT_IMAGE_WIDTH = 320
DEFAULT_IMAGE_HEIGHT = 240


def estimate_focal_length_from_diagonal_fov(width, height, diagonal_fov_deg):
    """Estimate square-pixel focal length from image diagonal and diagonal FOV."""
    for name, value in (("width", width), ("height", height)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and greater than zero")
    if (
        isinstance(diagonal_fov_deg, bool)
        or not isinstance(diagonal_fov_deg, (int, float))
        or not math.isfinite(diagonal_fov_deg)
        or not 0 < diagonal_fov_deg < 180
    ):
        raise ValueError("diagonal_fov_deg must be finite and between 0 and 180")
    diagonal_pixels = math.hypot(width, height)
    return diagonal_pixels / (
        2.0 * math.tan(math.radians(diagonal_fov_deg) / 2.0)
    )


DEFAULT_FOCAL_LENGTH_PX = estimate_focal_length_from_diagonal_fov(
    DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT, DEFAULT_DIAGONAL_FOV_DEG
)


@dataclass(frozen=True)
class PSFParameters:
    """Validated physical parameters used by every PSF entry point."""

    depth: float
    fx: float
    fy: float
    exposure: float = DEFAULT_EXPOSURE_SECONDS
    psf_sigma: float = 0.0
    intrinsics_source: str = "explicit"

    def __post_init__(self) -> None:
        for name in ("depth", "fx", "fy", "exposure"):
            value = getattr(self, name)
            if isinstance(value, bool) or not _is_finite_positive(value):
                raise ValueError(f"{name} must be finite and greater than zero")
        if isinstance(self.psf_sigma, bool):
            raise ValueError("psf_sigma must be finite and non-negative")
        try:
            valid_sigma = math.isfinite(self.psf_sigma) and self.psf_sigma >= 0
        except TypeError:
            valid_sigma = False
        if not valid_sigma:
            raise ValueError("psf_sigma must be finite and non-negative")
        if not isinstance(self.intrinsics_source, str) or not self.intrinsics_source:
            raise ValueError("intrinsics_source must be a non-empty string")


def add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    h5_required: bool = False,
    default_h5: str | None = None,
) -> None:
    """Register input and camera arguments shared by every method."""
    parser.allow_abbrev = False
    parser.add_argument(
        "--h5", required=h5_required and default_h5 is None, default=default_h5
    )
    parser.add_argument("--depth", type=finite_positive_float, default=0.5)
    parser.add_argument(
        "--exposure",
        type=finite_positive_float,
        help="exposure in seconds; defaults to H5 metadata, then 0.01 for legacy data",
    )
    parser.add_argument("--fx", type=finite_positive_float)
    parser.add_argument("--fy", type=finite_positive_float)
    parser.add_argument("--psf-sigma", type=finite_nonnegative_float, default=0.0)


def add_method_subcommands(parser: argparse.ArgumentParser) -> None:
    """Register required methods; common options must precede the chosen method."""
    methods = parser.add_subparsers(dest="method", required=True)

    wiener = methods.add_parser("wiener", allow_abbrev=False)
    wiener.add_argument("--K", type=finite_positive_float, default=0.01)
    wiener.add_argument("--adaptive-k", action="store_true")

    rl = methods.add_parser("rl", allow_abbrev=False)
    rl.add_argument("--rl-iters", type=positive_int, default=30)

    tv = methods.add_parser("tv", allow_abbrev=False)
    tv.add_argument("--tv-lam", type=finite_positive_float, default=0.002)


def config_from_args(args: argparse.Namespace) -> MethodConfig:
    """Construct the immutable method configuration selected by parsed arguments."""
    if (getattr(args, "fx", None) is None) != (getattr(args, "fy", None) is None):
        raise ValueError("--fx and --fy must be provided together")
    method = getattr(args, "method", None)
    if method == "wiener":
        return WienerConfig(K=args.K, adaptive_k=args.adaptive_k)
    if method == "rl":
        return RLConfig(iterations=args.rl_iters)
    if method == "tv":
        return TVConfig(lam=args.tv_lam)
    raise ValueError(f"Unsupported method: {method!r}")


def resolve_psf_parameters(
    meta,
    *,
    depth,
    fx=None,
    fy=None,
    exposure=None,
    psf_sigma=0.0,
) -> PSFParameters:
    """Resolve explicit values over H5 metadata without inventing focal lengths."""
    if (fx is None) != (fy is None):
        raise ValueError("fx/fy must be provided together")
    intrinsics = meta.get("camera_intrinsics") or {}
    if fx is not None:
        resolved_fx, resolved_fy = fx, fy
        intrinsics_source = "explicit_cli_or_python"
    elif intrinsics:
        resolved_fx, resolved_fy = intrinsics["fx"], intrinsics["fy"]
        intrinsics_source = meta.get("camera_intrinsics_source") or "h5_metadata"
    else:
        resolved_fx = resolved_fy = estimate_focal_length_from_diagonal_fov(
            meta["W"], meta["H"], DEFAULT_DIAGONAL_FOV_DEG
        )
        intrinsics_source = "estimated_diagonal_fov_75deg"
    if exposure is None:
        exposure = meta.get("exposure_seconds")
    if exposure is None:
        exposure = DEFAULT_EXPOSURE_SECONDS
    return PSFParameters(
        depth=depth,
        fx=resolved_fx,
        fy=resolved_fy,
        exposure=exposure,
        psf_sigma=psf_sigma,
        intrinsics_source=intrinsics_source,
    )


def _is_finite_positive(value: object) -> bool:
    try:
        return math.isfinite(value) and value > 0
    except TypeError:
        return False


def method_config_dict(config: MethodConfig) -> dict:
    """Return a JSON-serializable method configuration."""
    return asdict(config)


def _number(value: float) -> str:
    return format(float(value), ".12g")


def build_run_name(
    h5_path,
    method_config: MethodConfig,
    *,
    depth,
    exposure,
    fx,
    fy,
    psf_sigma,
    max_frames=None,
    frame=None,
) -> str:
    """Build a stable directory name containing every effective parameter."""
    if isinstance(method_config, WienerConfig):
        method_token = (
            f"wiener_K{_number(method_config.K)}_"
            f"adpt{int(method_config.adaptive_k)}"
        )
    elif isinstance(method_config, RLConfig):
        method_token = f"rl_iter{method_config.iterations}"
    elif isinstance(method_config, TVConfig):
        method_token = f"tv_lam{_number(method_config.lam)}"
    else:
        raise TypeError(f"unsupported method configuration: {type(method_config).__name__}")
    tokens = [
        Path(h5_path).stem,
        method_token,
        f"d{_number(depth)}",
        f"e{_number(exposure)}",
        f"fx{_number(fx)}",
        f"fy{_number(fy)}",
        f"sig{_number(psf_sigma)}",
    ]
    if max_frames is not None:
        tokens.append(f"n{max_frames}")
    if frame is not None:
        tokens.append(f"f{frame}")
    return "_".join(tokens)


def prepare_run_directory(run_dir, *, overwrite: bool, known_entries) -> Path:
    """Prepare one exact run directory without deleting unknown user content."""
    run_dir = Path(run_dir)
    known_entries = set(known_entries)
    if run_dir.exists():
        entries = {entry.name for entry in run_dir.iterdir()}
        if entries and not overwrite:
            raise FileExistsError(
                f"output directory is not empty: {run_dir}; use --overwrite"
            )
        unknown = entries - known_entries
        if overwrite and unknown:
            raise FileExistsError(
                f"refusing overwrite because output contains unknown entries: "
                f"{sorted(unknown)}"
            )
        if overwrite:
            for name in entries:
                target = run_dir / name
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_run_config(run_dir, config: dict) -> Path:
    """Write a complete, deterministic JSON run manifest."""
    path = Path(run_dir) / "run_config.json"
    path.write_text(
        json.dumps(config, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    return path
