"""Hardware-independent real-time RGB Wiener deblurring."""

from dataclasses import dataclass
import math
import time

import numpy as np

from cli_config import DEFAULT_FOCAL_LENGTH_PX

from joint_deblur import (
    compute_psf_from_pose,
    get_camera_velocity,
    wiener_deconvolution,
)
from robot_configs import HAND_EYE_CONFIGS, get_robot


def _finite_positive(name, value):
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and greater than zero")


def _finite_nonnegative(name, value):
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class RealtimeDeblurConfig:
    """Configuration for the live shared-PSF RGB Wiener path."""

    fx: float = DEFAULT_FOCAL_LENGTH_PX
    fy: float = DEFAULT_FOCAL_LENGTH_PX
    K: float = 0.01
    depth: float = 0.5
    exposure: float = 0.01
    psf_sigma: float = 0.0
    adaptive_k: bool = False
    min_motion_px: float = 0.25

    def __post_init__(self):
        _finite_positive("K", self.K)
        _finite_positive("depth", self.depth)
        _finite_positive("exposure", self.exposure)
        _finite_positive("fx", self.fx)
        _finite_positive("fy", self.fy)
        _finite_nonnegative("psf_sigma", self.psf_sigma)
        _finite_nonnegative("min_motion_px", self.min_motion_px)
        if type(self.adaptive_k) is not bool:
            raise TypeError("adaptive_k must be a boolean")


@dataclass(frozen=True)
class DeblurDiagnostics:
    """Per-frame timing and PSF information for live diagnostics."""

    applied: bool
    velocity_source: str
    du: float
    dv: float
    motion_px: float
    psf_size: int
    elapsed_ms: float


class RealTimeRGBDeblurrer:
    """Estimate live camera motion and Wiener-deblur one RGB frame at a time."""

    def __init__(self, config=None):
        self.config = config or RealtimeDeblurConfig()
        self.robot = get_robot("kinova-gen3")
        self.hand_eye = HAND_EYE_CONFIGS["kinova-gen3"]
        self._previous_joints_deg = None
        self._previous_timestamp = None

    @staticmethod
    def _validate_image(image_rgb):
        image_rgb = np.asarray(image_rgb)
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(
                f"live image must have RGB shape (H, W, 3); got {image_rgb.shape}"
            )
        if image_rgb.dtype != np.uint8:
            raise ValueError(f"live RGB image must use uint8; got {image_rgb.dtype}")
        if image_rgb.shape[0] == 0 or image_rgb.shape[1] == 0:
            raise ValueError("live RGB image must not be empty")
        return image_rgb

    @staticmethod
    def _validate_joints(values, label):
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (7,):
            raise ValueError(f"{label} must contain exactly seven values")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{label} must contain only finite values")
        return values

    def _resolve_velocity(self, joints_deg, measured_velocity_deg_s, timestamp):
        previous_joints = self._previous_joints_deg
        previous_timestamp = self._previous_timestamp
        self._previous_joints_deg = joints_deg.copy()
        self._previous_timestamp = timestamp

        if measured_velocity_deg_s is not None:
            measured = self._validate_joints(
                measured_velocity_deg_s, "joint velocities"
            )
            return measured, "measured"
        if previous_joints is None:
            return None, "initializing"

        dt = timestamp - previous_timestamp
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("timestamps must increase between live frames")
        delta_deg = (joints_deg - previous_joints + 180.0) % 360.0 - 180.0
        return delta_deg / dt, "estimated"

    def _diagnostics(
        self,
        start,
        *,
        applied,
        velocity_source,
        du=0.0,
        dv=0.0,
        motion_px=0.0,
        psf_size=0,
    ):
        return DeblurDiagnostics(
            applied=applied,
            velocity_source=velocity_source,
            du=float(du),
            dv=float(dv),
            motion_px=float(motion_px),
            psf_size=int(psf_size),
            elapsed_ms=(time.perf_counter() - start) * 1000.0,
        )

    def process(
        self,
        image_rgb,
        joint_positions_deg,
        joint_velocities_deg_s=None,
        timestamp=None,
    ):
        """Return the RGB policy frame and live deblurring diagnostics."""
        start = time.perf_counter()
        image_rgb = self._validate_image(image_rgb)
        joints_deg = self._validate_joints(joint_positions_deg, "joint positions")
        timestamp = time.monotonic() if timestamp is None else float(timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")

        velocity_deg_s, velocity_source = self._resolve_velocity(
            joints_deg, joint_velocities_deg_s, timestamp
        )
        if velocity_deg_s is None:
            return image_rgb.copy(), self._diagnostics(
                start, applied=False, velocity_source=velocity_source
            )

        camera_velocity = get_camera_velocity(
            np.deg2rad(joints_deg),
            np.deg2rad(velocity_deg_s),
            hand_eye=self.hand_eye,
            robot=self.robot,
        )
        height, width = image_rgb.shape[:2]
        psf, (du, dv) = compute_psf_from_pose(
            depth=self.config.depth,
            fx=self.config.fx,
            fy=self.config.fy,
            cx=width // 2,
            cy=height // 2,
            exposure_time=self.config.exposure,
            v_cam_6d=camera_velocity,
        )
        psf = np.asarray(psf, dtype=np.float64)
        if psf.ndim != 2 or psf.size == 0 or not np.all(np.isfinite(psf)):
            raise ValueError("live PSF must be a finite non-empty 2-D array")
        if psf.sum() <= 0:
            raise ValueError("live PSF must have positive mass")

        if self.config.psf_sigma > 0:
            from scipy.ndimage import gaussian_filter

            psf = gaussian_filter(psf, sigma=self.config.psf_sigma)
        psf /= psf.sum()

        motion_px = math.hypot(du, dv)
        psf_size = max(psf.shape)
        if motion_px < self.config.min_motion_px:
            return image_rgb.copy(), self._diagnostics(
                start,
                applied=False,
                velocity_source=velocity_source,
                du=du,
                dv=dv,
                motion_px=motion_px,
                psf_size=psf_size,
            )

        effective_k = self.config.K
        if self.config.adaptive_k:
            effective_k *= 1.0 + 0.3 * np.log2(max(psf_size, 3) / 17.0)
        _finite_positive("effective Wiener K", effective_k)
        result_rgb = wiener_deconvolution(image_rgb, psf, K=effective_k)
        if result_rgb.shape != image_rgb.shape or result_rgb.dtype != np.uint8:
            raise RuntimeError(
                "RGB Wiener returned an invalid live frame: "
                f"{result_rgb.shape} {result_rgb.dtype}"
            )
        return result_rgb, self._diagnostics(
            start,
            applied=True,
            velocity_source=velocity_source,
            du=du,
            dv=dv,
            motion_px=motion_px,
            psf_size=psf_size,
        )
