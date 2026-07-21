from pathlib import Path

import numpy as np

import pipeline


def _sentinel_rgb():
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    image[:] = [250, 20, 5]
    return image


def test_rgb_to_bgr_swaps_only_red_and_blue():
    rgb = np.array([[[250, 20, 5]]], dtype=np.uint8)
    np.testing.assert_array_equal(pipeline.rgb_to_bgr(rgb), [[[5, 20, 250]]])


def test_comparison_canvas_keeps_rgb_pixels():
    original = _sentinel_rgb()
    deblurred = np.full_like(original, [1, 2, 240])

    canvas = pipeline.build_comparison_canvas(original, deblurred, label_height=8)

    np.testing.assert_array_equal(canvas[8, 0], [250, 20, 5])
    np.testing.assert_array_equal(canvas[8, original.shape[1]], [1, 2, 240])


def test_setup_output_dirs_creates_two_color_video_writers(tmp_path, monkeypatch):
    calls = []

    class FakeWriter:
        def isOpened(self):
            return True

        def write(self, frame):
            pass

        def release(self):
            pass

    def fake_writer(*args, **kwargs):
        calls.append((args, kwargs))
        return FakeWriter()

    monkeypatch.setattr(pipeline.cv2, "VideoWriter", fake_writer)
    monkeypatch.setattr(pipeline.cv2, "VideoWriter_fourcc", lambda *args: 7)

    pipeline.setup_output_dirs(tmp_path, 4, 3, fps=10.0)

    assert len(calls) == 2
    assert all(call_kwargs["isColor"] is True for _, call_kwargs in calls)
    assert (tmp_path / "blurred").is_dir()
    assert (tmp_path / "deblurred").is_dir()
    assert (tmp_path / "comparison").is_dir()


def test_pipeline_keeps_low_texture_frames(tmp_path, monkeypatch):
    image = np.zeros((1, 3, 4, 3), dtype=np.uint8)
    meta = {
        "images": image,
        "joint_positions": np.zeros((1, 7)),
        "joint_velocities": np.zeros((1, 7)),
        "num_frames": 1,
        "H": 3,
        "W": 4,
        "camera_fps": 10.0,
        "camera_intrinsics": None,
        "exposure_seconds": None,
    }
    saved = []

    class FakeWriter:
        def release(self):
            pass

    monkeypatch.setattr(pipeline, "load_episode_h5", lambda _: meta)
    monkeypatch.setattr(
        pipeline,
        "compute_episode_psf",
        lambda *args, **kwargs: (np.array([[1.0]]), 0.0, 0.0),
    )
    monkeypatch.setattr(
        pipeline,
        "deblur_rgb",
        lambda frame, psf, config: (frame.copy(), config.K),
    )
    monkeypatch.setattr(
        pipeline, "setup_output_dirs", lambda *args, **kwargs: (FakeWriter(), FakeWriter())
    )
    monkeypatch.setattr(
        pipeline,
        "save_deblur_result",
        lambda *args, **kwargs: saved.append(args[1]),
    )

    pipeline.run_h5_pipeline(
        "episode.h5",
        tmp_path,
        method_config=pipeline.WienerConfig(),
        fx=300.0,
        fy=300.0,
        exposure=0.01,
    )

    assert saved == [0]


def test_save_result_converts_rgb_to_bgr_for_opencv(tmp_path, monkeypatch):
    original = _sentinel_rgb()
    deblurred = np.full_like(original, [1, 2, 240])
    written_images = []

    class CapturingWriter:
        def __init__(self):
            self.frames = []

        def write(self, frame):
            self.frames.append(frame.copy())

    monkeypatch.setattr(
        pipeline.cv2,
        "imwrite",
        lambda path, image: written_images.append((Path(path), image.copy())) or True,
    )
    for name in ("blurred", "deblurred", "comparison"):
        (tmp_path / name).mkdir()
    video = CapturingWriter()
    comparison = CapturingWriter()

    pipeline.save_deblur_result(
        tmp_path,
        0,
        original,
        deblurred,
        ("global", np.array([[1.0]]), 0.0, 0.0),
        comp_writer=comparison,
        vid_writer=video,
    )

    np.testing.assert_array_equal(written_images[0][1][0, 0], [5, 20, 250])
    np.testing.assert_array_equal(written_images[1][1][0, 0], [240, 2, 1])
    np.testing.assert_array_equal(video.frames[0][0, 0], [240, 2, 1])
    np.testing.assert_array_equal(comparison.frames[0][0, 0], [5, 20, 250])
