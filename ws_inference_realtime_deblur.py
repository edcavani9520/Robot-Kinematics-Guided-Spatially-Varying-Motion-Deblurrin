#!/usr/bin/env python3
"""Launch the external Pi05 WebSocket controller with RGB Wiener deblurring.

Gen3 controller repository:
https://github.com/edcavani9520/fnii-gen3-controller.git
"""

import argparse
import importlib.util
import math
from pathlib import Path
import sys
import time

import numpy as np

from cli_config import DEFAULT_FOCAL_LENGTH_PX
from realtime_deblur import RealTimeRGBDeblurrer, RealtimeDeblurConfig


DEBLUR_REPOSITORY_URL = (
    "https://github.com/edcavani9520/"
    "Robot-Kinematics-Guided-Spatially-Varying-Motion-Deblurrin.git"
)
CONTROLLER_REPOSITORY_URL = (
    "https://github.com/edcavani9520/fnii-gen3-controller.git"
)


def load_parent_controller(controller_root):
    """Load Pi05WebSocketControl only after the external checkout is selected."""
    controller_root = Path(controller_root).expanduser().resolve()
    source = controller_root / "pi05_ws_control.py"
    if not source.is_file():
        raise FileNotFoundError(
            f"Pi05 controller source not found: {source}\n"
            f"Download it with:\n  git clone {CONTROLLER_REPOSITORY_URL}\n"
            "Then pass the checkout using --controller-root."
        )

    root_text = str(controller_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    spec = importlib.util.spec_from_file_location("_fnii_pi05_ws_control", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create an import spec for {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return module.Pi05WebSocketControl
    except AttributeError as exc:
        raise ImportError(f"{source} does not define Pi05WebSocketControl") from exc


def _measured_actuator_velocities(full_status):
    """Return seven finite actuator velocities in degrees/second when available."""
    actuators = getattr(full_status, "actuators", None)
    if actuators is None or len(actuators) != 7:
        return None
    try:
        velocities = np.asarray(
            [actuator.velocity for actuator in actuators], dtype=np.float64
        )
    except (AttributeError, TypeError, ValueError):
        return None
    return velocities if np.all(np.isfinite(velocities)) else None


def build_deblur_controller_class(parent_class):
    """Build a parent-compatible controller overriding only camera acquisition."""

    class Pi05DeblurWebSocketControl(parent_class):
        def __init__(
            self,
            *args,
            deblurrer,
            deblur_log_every=10,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.realtime_deblurrer = deblurrer
            self.deblur_log_every = max(1, int(deblur_log_every))
            self.deblur_frame_count = 0

        def get_camera_image(self):
            image_rgb = super().get_camera_image()
            joints_deg = self.get_joint_positions()
            measured_velocity = _measured_actuator_velocities(
                getattr(self, "full_status", None)
            )
            processed_rgb, diagnostics = self.realtime_deblurrer.process(
                image_rgb,
                joints_deg,
                joint_velocities_deg_s=measured_velocity,
                timestamp=time.monotonic(),
            )
            self.deblur_frame_count += 1
            if self.deblur_frame_count % self.deblur_log_every == 0:
                state = "applied" if diagnostics.applied else "bypass"
                print(
                    f"[deblur] frame={self.deblur_frame_count} {state} "
                    f"source={diagnostics.velocity_source} "
                    f"du={diagnostics.du:+.2f} dv={diagnostics.dv:+.2f} "
                    f"psf={diagnostics.psf_size} "
                    f"time={diagnostics.elapsed_ms:.1f}ms"
                )
            return processed_rgb

    Pi05DeblurWebSocketControl.__name__ = "Pi05DeblurWebSocketControl"
    return Pi05DeblurWebSocketControl


def _finite_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be a finite number")
    return parsed


def _positive_float(value):
    parsed = _finite_float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_float(value):
    parsed = _finite_float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _positive_int(value):
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_int(value):
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _build_parser():
    repository_help = f"""
Repositories:
  RGB deblurring: {DEBLUR_REPOSITORY_URL}
  Gen3 controller: {CONTROLLER_REPOSITORY_URL}

Clone both repositories as siblings, then run:
  git clone {DEBLUR_REPOSITORY_URL}
  git clone {CONTROLLER_REPOSITORY_URL}
  python ws_inference_realtime_deblur.py --controller-root ../fnii-gen3-controller
"""
    parser = argparse.ArgumentParser(
        description=(
            "Run Pi05 WebSocket inference while replacing observation/image "
            "with a real-time RGB Wiener-deblurred frame."
        ),
        epilog=repository_help,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    external = parser.add_argument_group("external controller")
    external.add_argument(
        "--controller-root",
        type=Path,
        default=Path("../fnii-gen3-controller"),
        help="checkout containing pi05_ws_control.py",
    )
    external.add_argument("--ws-host", default="localhost")
    external.add_argument("--ws-port", type=int, default=8000)
    external.add_argument("--robot-ip", default="192.168.8.10")
    external.add_argument("--camera-id", type=int, default=0)
    external.add_argument("--prompt", default="Put the cube into the bowl")
    external.add_argument("--dry-run", action="store_true")
    external.add_argument("--observe-only", action="store_true")
    external.add_argument("--freq", type=_positive_float, default=10.0)
    external.add_argument("--action-steps", type=_positive_int, default=1)
    external.add_argument("--max-pos-step", type=_positive_float, default=0.015)
    external.add_argument("--max-rot-step", type=_positive_float, default=1.0)
    external.add_argument("--max-joint-speed", type=_positive_float, default=10.0)
    external.add_argument("--action-scale", type=_positive_float, default=1.0)
    external.add_argument("--control-mode", choices=("twist", "ik"), default="twist")
    external.add_argument("--max-linear-speed", type=_positive_float, default=0.05)
    external.add_argument("--max-angular-speed", type=_positive_float, default=3.0)
    external.add_argument("--log-every", type=_positive_int, default=5)
    external.add_argument("--skip-start-position", action="store_true")
    external.add_argument("--start-pose-path")
    external.add_argument("--start-tolerance-deg", type=_positive_float, default=3.0)
    external.add_argument("--min-ee-z", type=_finite_float)
    external.add_argument("--max-down-step", type=_positive_float)
    external.add_argument("--camera-drain-frames", type=_nonnegative_int, default=0)

    deblur = parser.add_argument_group("real-time RGB Wiener deblurring")
    deblur.add_argument("--K", type=_positive_float, default=0.01)
    deblur.add_argument("--depth", type=_positive_float, default=0.5)
    deblur.add_argument("--exposure", type=_positive_float, default=0.01)
    deblur.add_argument("--fx", type=_positive_float, default=DEFAULT_FOCAL_LENGTH_PX)
    deblur.add_argument("--fy", type=_positive_float, default=DEFAULT_FOCAL_LENGTH_PX)
    deblur.add_argument("--psf-sigma", type=_nonnegative_float, default=0.0)
    deblur.add_argument("--adaptive-k", action="store_true")
    deblur.add_argument("--min-motion-px", type=_nonnegative_float, default=0.25)
    deblur.add_argument("--deblur-log-every", type=_positive_int, default=10)
    return parser


def _parse_args(argv=None):
    return _build_parser().parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    parent_class = load_parent_controller(args.controller_root)
    controller_class = build_deblur_controller_class(parent_class)
    deblurrer = RealTimeRGBDeblurrer(
        RealtimeDeblurConfig(
            K=args.K,
            depth=args.depth,
            exposure=args.exposure,
            fx=args.fx,
            fy=args.fy,
            psf_sigma=args.psf_sigma,
            adaptive_k=args.adaptive_k,
            min_motion_px=args.min_motion_px,
        )
    )
    controller = controller_class(
        ws_host=args.ws_host,
        ws_port=args.ws_port,
        robot_ip=args.robot_ip,
        camera_id=args.camera_id,
        prompt=args.prompt,
        dry_run=args.dry_run,
        control_freq=args.freq,
        action_steps=args.action_steps,
        max_pos_step=args.max_pos_step,
        max_rot_step=args.max_rot_step,
        max_joint_speed=args.max_joint_speed,
        action_scale=args.action_scale,
        observe_only=args.observe_only,
        control_mode=args.control_mode,
        max_linear_speed=args.max_linear_speed,
        max_angular_speed=args.max_angular_speed,
        log_every=args.log_every,
        auto_start=not args.skip_start_position,
        start_pose_path=args.start_pose_path,
        start_tolerance_deg=args.start_tolerance_deg,
        min_ee_z=args.min_ee_z,
        max_down_step=args.max_down_step,
        camera_drain_frames=args.camera_drain_frames,
        deblurrer=deblurrer,
        deblur_log_every=args.deblur_log_every,
    )
    print(f"[deblur] source: {DEBLUR_REPOSITORY_URL}")
    print(f"[controller] source: {CONTROLLER_REPOSITORY_URL}")
    controller.run()


if __name__ == "__main__":
    main()
