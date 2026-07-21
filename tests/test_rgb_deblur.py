import numpy as np
import pytest

from joint_deblur import (
    compute_interaction_matrix,
    compute_psf_from_pose,
    create_motion_psf,
    richardson_lucy,
    spatial_richardson_lucy,
    spatial_wiener_deconvolution,
    tv_deconv,
    wiener_deconvolution,
)


def _rgb_fixture():
    yy, xx = np.mgrid[:8, :10]
    return np.stack(
        [
            (xx * 17 + yy * 3) % 256,
            (yy * 29 + 11) % 256,
            ((xx + yy) * 13 + 7) % 256,
        ],
        axis=2,
    ).astype(np.uint8)


@pytest.mark.parametrize(
    ("deconvolve", "kwargs"),
    [
        (wiener_deconvolution, {"K": 0.02}),
        (richardson_lucy, {"iterations": 2}),
        (tv_deconv, {"lam": 0.002, "max_iter": 2}),
    ],
)
def test_global_deconvolution_processes_each_rgb_channel_independently(
    deconvolve, kwargs
):
    rgb = _rgb_fixture()
    psf = np.array([[0.0, 0.25, 0.0], [0.25, 0.0, 0.25], [0.0, 0.25, 0.0]])

    actual = deconvolve(rgb, psf, **kwargs)
    expected = np.stack(
        [deconvolve(rgb[..., channel], psf, **kwargs) for channel in range(3)],
        axis=2,
    )

    assert actual.shape == rgb.shape
    assert actual.dtype == np.uint8
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("deconvolve", "kwargs"),
    [
        (spatial_wiener_deconvolution, {"K": 0.02}),
        (spatial_richardson_lucy, {"iterations": 2}),
    ],
)
def test_spatial_deconvolution_preserves_rgb_channel_independence(
    deconvolve, kwargs
):
    rgb = _rgb_fixture()
    psf_map = [[np.array([[1.0]])]]

    actual = deconvolve(rgb, psf_map, 1, 1, overlap=0.0, **kwargs)
    expected = np.stack(
        [
            deconvolve(
                rgb[..., channel], psf_map, 1, 1, overlap=0.0, **kwargs
            )
            for channel in range(3)
        ],
        axis=2,
    )

    assert actual.shape == rgb.shape
    assert actual.dtype == np.uint8
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("shape", [(4,), (4, 5, 1), (4, 5, 4)])
def test_global_deconvolution_rejects_non_image_shapes(shape):
    image = np.zeros(shape, dtype=np.uint8)
    with pytest.raises(ValueError, match="RGB"):
        wiener_deconvolution(image, np.array([[1.0]]))


def test_subpixel_motion_psf_preserves_fractional_displacement():
    psf = create_motion_psf(0.25, 0.0)

    assert np.count_nonzero(psf) > 1
    assert psf.sum() == pytest.approx(1.0)
    yy, xx = np.indices(psf.shape)
    assert (psf * xx).sum() == pytest.approx(psf.shape[1] // 2)
    assert (psf * yy).sum() == pytest.approx(psf.shape[0] // 2)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((0, 0, 0, 300, 300, 160, 120), "depth"),
        ((160, 120, 0.5, 0, 300, 160, 120), "fx"),
        ((160, 120, 0.5, 300, float("nan"), 160, 120), "fy"),
    ],
)
def test_interaction_matrix_rejects_invalid_direct_python_parameters(args, message):
    with pytest.raises(ValueError, match=message):
        compute_interaction_matrix(*args)


@pytest.mark.parametrize("exposure", [0, -0.01, float("nan")])
def test_pose_psf_rejects_invalid_direct_python_exposure(exposure):
    with pytest.raises(ValueError, match="exposure"):
        compute_psf_from_pose(
            depth=0.5,
            fx=300,
            fy=300,
            cx=160,
            cy=120,
            exposure_time=exposure,
            v_cam_6d=np.zeros(6),
        )


@pytest.mark.parametrize("K", [0, -0.1, float("nan")])
def test_wiener_rejects_invalid_direct_python_k(K):
    with pytest.raises(ValueError, match="K"):
        wiener_deconvolution(np.zeros((3, 3), dtype=np.uint8), np.array([[1.0]]), K=K)


@pytest.mark.parametrize("iterations", [0, -1, 1.5, True])
def test_rl_rejects_invalid_direct_python_iterations(iterations):
    with pytest.raises(ValueError, match="iterations"):
        richardson_lucy(
            np.zeros((3, 3), dtype=np.uint8),
            np.array([[1.0]]),
            iterations=iterations,
        )


@pytest.mark.parametrize("lam", [0, -0.1, float("nan")])
def test_tv_rejects_invalid_direct_python_lambda(lam):
    with pytest.raises(ValueError, match="lam"):
        tv_deconv(
            np.zeros((3, 3), dtype=np.uint8),
            np.array([[1.0]]),
            lam=lam,
            max_iter=1,
        )


def test_deconvolution_rejects_psf_larger_than_image():
    with pytest.raises(ValueError, match="exceeds image"):
        wiener_deconvolution(
            np.zeros((3, 3), dtype=np.uint8),
            np.ones((5, 5), dtype=np.float64),
        )


def test_motion_psf_rejects_unprocessable_kernel_size():
    with pytest.raises(ValueError, match="kernel"):
        create_motion_psf(10000.0, 0.0)
