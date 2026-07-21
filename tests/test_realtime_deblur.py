import numpy as np
import pytest

import realtime_deblur
from realtime_deblur import RealTimeRGBDeblurrer, RealtimeDeblurConfig


def _rgb_frame(value=0):
    frame = np.zeros((12, 16, 3), dtype=np.uint8)
    frame[..., 0] = value
    frame[..., 1] = value + 1
    frame[..., 2] = value + 2
    return frame


def _config(**overrides):
    values = {"fx": 300.0, "fy": 301.0}
    values.update(overrides)
    return RealtimeDeblurConfig(**values)


def _processor(**overrides):
    return RealTimeRGBDeblurrer(_config(**overrides))


def test_realtime_config_defaults_to_estimated_75_degree_focal_length():
    config = RealtimeDeblurConfig()

    assert config.fx == pytest.approx(260.6450745682411)
    assert config.fy == pytest.approx(260.6450745682411)


def test_first_frame_without_measured_velocity_initializes_and_bypasses():
    processor = _processor()
    image = _rgb_frame(10)

    output, diagnostics = processor.process(
        image, np.zeros(7), timestamp=1.0
    )

    np.testing.assert_array_equal(output, image)
    assert diagnostics.applied is False
    assert diagnostics.velocity_source == "initializing"
    assert diagnostics.psf_size == 0


@pytest.mark.parametrize(
    ("image", "joints", "message"),
    [
        (np.zeros((12, 16), dtype=np.uint8), np.zeros(7), "RGB"),
        (np.zeros((12, 16, 3), dtype=np.float32), np.zeros(7), "uint8"),
        (_rgb_frame(), np.zeros(6), "seven"),
        (_rgb_frame(), np.array([0, 0, 0, 0, 0, 0, np.nan]), "finite"),
    ],
)
def test_process_rejects_invalid_image_or_joint_state(image, joints, message):
    with pytest.raises(ValueError, match=message):
        _processor().process(image, joints, timestamp=1.0)


def test_wrapped_finite_difference_velocity_crosses_359_to_zero(monkeypatch):
    captured = {}

    def fake_camera_velocity(q, qd, **kwargs):
        captured["qd"] = qd.copy()
        return np.zeros(6)

    monkeypatch.setattr(realtime_deblur, "get_camera_velocity", fake_camera_velocity)
    monkeypatch.setattr(
        realtime_deblur,
        "compute_psf_from_pose",
        lambda **kwargs: (np.array([[1.0]]), (0.0, 0.0)),
    )
    processor = _processor()
    q0 = np.zeros(7)
    q0[0] = 359.0
    q1 = np.zeros(7)

    processor.process(_rgb_frame(), q0, timestamp=1.0)
    _, diagnostics = processor.process(_rgb_frame(), q1, timestamp=1.1)

    np.testing.assert_allclose(captured["qd"][0], np.deg2rad(10.0))
    assert diagnostics.velocity_source == "estimated"


def test_measured_velocity_takes_precedence(monkeypatch):
    captured = {}

    def fake_camera_velocity(q, qd, **kwargs):
        captured["qd"] = qd.copy()
        return np.zeros(6)

    monkeypatch.setattr(realtime_deblur, "get_camera_velocity", fake_camera_velocity)
    monkeypatch.setattr(
        realtime_deblur,
        "compute_psf_from_pose",
        lambda **kwargs: (np.array([[1.0]]), (0.0, 0.0)),
    )

    _, diagnostics = _processor().process(
        _rgb_frame(),
        np.zeros(7),
        joint_velocities_deg_s=np.array([20.0, 0, 0, 0, 0, 0, 0]),
        timestamp=1.0,
    )

    np.testing.assert_allclose(captured["qd"][0], np.deg2rad(20.0))
    assert diagnostics.velocity_source == "measured"


def test_motion_below_threshold_skips_wiener(monkeypatch):
    monkeypatch.setattr(
        realtime_deblur, "get_camera_velocity", lambda *args, **kwargs: np.zeros(6)
    )
    monkeypatch.setattr(
        realtime_deblur,
        "compute_psf_from_pose",
        lambda **kwargs: (np.array([[1.0]]), (0.1, 0.1)),
    )
    monkeypatch.setattr(
        realtime_deblur,
        "wiener_deconvolution",
        lambda *args, **kwargs: pytest.fail("Wiener must be bypassed"),
    )
    image = _rgb_frame(20)

    output, diagnostics = _processor(min_motion_px=0.5).process(
        image,
        np.zeros(7),
        joint_velocities_deg_s=np.zeros(7),
        timestamp=1.0,
    )

    np.testing.assert_array_equal(output, image)
    assert diagnostics.applied is False
    assert diagnostics.motion_px == pytest.approx(np.hypot(0.1, 0.1))


def test_nonzero_motion_runs_wiener_on_complete_rgb_image(monkeypatch):
    captured = {}
    expected = _rgb_frame(30)
    monkeypatch.setattr(
        realtime_deblur, "get_camera_velocity", lambda *args, **kwargs: np.zeros(6)
    )
    monkeypatch.setattr(
        realtime_deblur,
        "compute_psf_from_pose",
        lambda **kwargs: (np.ones((5, 5)) / 25.0, (3.0, 4.0)),
    )

    def fake_wiener(image, psf, K):
        captured.update(image=image.copy(), psf=psf.copy(), K=K)
        return expected.copy()

    monkeypatch.setattr(realtime_deblur, "wiener_deconvolution", fake_wiener)
    image = _rgb_frame(5)
    config = _config(K=0.03, min_motion_px=0.0)

    output, diagnostics = RealTimeRGBDeblurrer(config).process(
        image,
        np.zeros(7),
        joint_velocities_deg_s=np.ones(7),
        timestamp=1.0,
    )

    np.testing.assert_array_equal(captured["image"], image)
    assert captured["image"].shape == image.shape
    assert captured["psf"].shape == (5, 5)
    assert captured["K"] == 0.03
    np.testing.assert_array_equal(output, expected)
    assert diagnostics.applied is True
    assert diagnostics.motion_px == 5.0
    assert diagnostics.psf_size == 5


def test_psf_smoothing_remains_normalized(monkeypatch):
    captured = {}
    impulse = np.zeros((5, 5), dtype=np.float64)
    impulse[2, 2] = 1.0
    monkeypatch.setattr(
        realtime_deblur, "get_camera_velocity", lambda *args, **kwargs: np.zeros(6)
    )
    monkeypatch.setattr(
        realtime_deblur,
        "compute_psf_from_pose",
        lambda **kwargs: (impulse.copy(), (2.0, 0.0)),
    )

    def fake_wiener(image, psf, K):
        captured["psf"] = psf.copy()
        return image.copy()

    monkeypatch.setattr(realtime_deblur, "wiener_deconvolution", fake_wiener)

    _processor(psf_sigma=1.0, min_motion_px=0.0).process(
        _rgb_frame(),
        np.zeros(7),
        joint_velocities_deg_s=np.ones(7),
        timestamp=1.0,
    )

    assert captured["psf"].sum() == pytest.approx(1.0)
    assert np.count_nonzero(captured["psf"]) > 1
