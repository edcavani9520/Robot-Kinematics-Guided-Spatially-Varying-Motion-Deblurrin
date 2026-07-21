from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from realtime_deblur import DeblurDiagnostics
import ws_inference_realtime_deblur as ws_wrapper


def test_loader_error_includes_public_clone_command(tmp_path):
    with pytest.raises(FileNotFoundError) as exc_info:
        ws_wrapper.load_parent_controller(tmp_path)

    message = str(exc_info.value)
    assert str(tmp_path / "pi05_ws_control.py") in message
    assert "https://github.com/edcavani9520/fnii-gen3-controller.git" in message
    assert "git clone https://" in message


def test_wrapper_returns_processed_frame_and_passes_measured_velocity():
    raw = np.zeros((4, 5, 3), dtype=np.uint8)
    raw[:] = [200, 20, 4]
    processed = np.zeros_like(raw)
    processed[:] = [1, 2, 240]

    class FakeParent:
        def __init__(self, *args, **kwargs):
            self.parent_args = args
            self.parent_kwargs = kwargs
            self.full_status = SimpleNamespace(
                actuators=[SimpleNamespace(velocity=float(i + 1)) for i in range(7)]
            )

        def get_camera_image(self):
            return raw.copy()

        def get_joint_positions(self):
            return np.arange(7, dtype=np.float32) + 10

    class FakeDeblurrer:
        def __init__(self):
            self.calls = []

        def process(self, image, joints, joint_velocities_deg_s, timestamp):
            self.calls.append(
                (image.copy(), joints.copy(), joint_velocities_deg_s.copy(), timestamp)
            )
            diagnostics = DeblurDiagnostics(
                applied=True,
                velocity_source="measured",
                du=2.0,
                dv=-1.0,
                motion_px=np.hypot(2.0, 1.0),
                psf_size=5,
                elapsed_ms=3.5,
            )
            return processed.copy(), diagnostics

    fake_deblurrer = FakeDeblurrer()
    Controller = ws_wrapper.build_deblur_controller_class(FakeParent)
    controller = Controller(
        "parent-positional",
        parent_option=7,
        deblurrer=fake_deblurrer,
        deblur_log_every=100,
    )

    output = controller.get_camera_image()

    assert controller.parent_args == ("parent-positional",)
    assert controller.parent_kwargs == {"parent_option": 7}
    np.testing.assert_array_equal(output, processed)
    sent_image, joints, velocities, timestamp = fake_deblurrer.calls[0]
    np.testing.assert_array_equal(sent_image, raw)
    np.testing.assert_array_equal(joints, np.arange(7) + 10)
    np.testing.assert_array_equal(velocities, np.arange(7) + 1)
    assert np.isfinite(timestamp)


def test_wrapper_falls_back_when_measured_velocity_is_unavailable():
    raw = np.zeros((3, 4, 3), dtype=np.uint8)

    class FakeParent:
        def __init__(self):
            self.full_status = None

        def get_camera_image(self):
            return raw

        def get_joint_positions(self):
            return np.zeros(7)

    class FakeDeblurrer:
        def process(self, image, joints, joint_velocities_deg_s, timestamp):
            assert joint_velocities_deg_s is None
            return image.copy(), DeblurDiagnostics(
                False, "initializing", 0.0, 0.0, 0.0, 0, 0.1
            )

    Controller = ws_wrapper.build_deblur_controller_class(FakeParent)
    output = Controller(deblurrer=FakeDeblurrer()).get_camera_image()

    np.testing.assert_array_equal(output, raw)


def test_cli_defaults_match_parent_and_live_wiener_defaults():
    args = ws_wrapper._parse_args(
        ["--controller-root", "../fnii-gen3-controller"]
    )

    assert args.controller_root == Path("../fnii-gen3-controller")
    assert args.freq == 10.0
    assert args.control_mode == "twist"
    assert args.K == 0.01
    assert args.depth == 0.5
    assert args.exposure == 0.01
    assert args.fx == pytest.approx(260.6450745682411)
    assert args.fy == pytest.approx(260.6450745682411)
    assert args.min_motion_px == 0.25


def test_cli_help_contains_public_repositories_and_run_command():
    help_text = ws_wrapper._build_parser().format_help()

    assert ws_wrapper.DEBLUR_REPOSITORY_URL in help_text
    assert ws_wrapper.CONTROLLER_REPOSITORY_URL in help_text
    assert "git clone https://" in help_text
    assert "--controller-root ../fnii-gen3-controller" in help_text


@pytest.mark.parametrize(
    "argv",
    [
        ["--K", "0"],
        ["--depth", "nan"],
        ["--exposure", "-1"],
        ["--min-motion-px", "-0.1"],
        ["--freq", "0"],
    ],
)
def test_cli_rejects_invalid_realtime_domains(argv):
    with pytest.raises(SystemExit):
        ws_wrapper._parse_args(["--fx", "300", "--fy", "301", *argv])


def test_cli_allows_overriding_estimated_focal_lengths():
    args = ws_wrapper._parse_args(["--fx", "300", "--fy", "301"])

    assert args.fx == 300.0
    assert args.fy == 301.0
