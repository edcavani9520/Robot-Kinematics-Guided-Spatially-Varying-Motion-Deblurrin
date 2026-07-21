import cv2
import numpy as np

from evaluate import (
    full_evaluate,
    laplacian_sharpness,
    match_histogram,
    ssim,
    tenengrad,
    total_variation,
)


def _color_pair():
    yy, xx = np.mgrid[:12, :14]
    original = np.stack(
        [
            (xx * 13 + yy * 3) % 256,
            (yy * 19 + 40) % 256,
            ((xx + yy) * 7 + 90) % 256,
        ],
        axis=2,
    ).astype(np.uint8)
    processed = original.copy()
    processed[..., 0] = 255 - processed[..., 0]
    processed[..., 1] = np.roll(processed[..., 1], 2, axis=1)
    return original, processed


def test_rgb_ssim_is_mean_of_per_channel_ssim():
    original, processed = _color_pair()
    expected = np.mean(
        [ssim(original[..., c], processed[..., c]) for c in range(3)]
    )

    assert ssim(original, processed) == expected


def test_rgb_histogram_matching_is_independent_per_channel():
    source, reference = _color_pair()
    expected = np.stack(
        [
            match_histogram(source[..., c], reference[..., c])
            for c in range(3)
        ],
        axis=2,
    )

    np.testing.assert_array_equal(match_histogram(source, reference), expected)


def test_sharpness_and_variation_metrics_use_rgb_luminance():
    original, _ = _color_pair()
    luminance = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)

    assert laplacian_sharpness(original) == laplacian_sharpness(luminance)
    assert tenengrad(original) == tenengrad(luminance)
    assert total_variation(original) == total_variation(luminance)


def test_full_evaluate_keeps_color_matched_image_and_channel_stats():
    original, processed = _color_pair()

    result = full_evaluate(original, processed)

    assert result["matched_image"].shape == original.shape
    assert set(result["stats_before"]["channels"]) == {"R", "G", "B"}
    assert result["stats_before"]["channels"]["R"]["mean"] == float(
        original[..., 0].mean()
    )

