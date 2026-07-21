import h5py
import numpy as np
import pytest

from h5_loader import EpisodeFrameReader, load_episode_h5


def _write_episode(
    path, *, images=None, proprio=None, actions=None, timestamps=None, attrs=None
):
    if images is None:
        images = np.zeros((3, 2, 2, 3), dtype=np.uint8)
        images[0, 0, 0] = [255, 0, 17]
    if proprio is None:
        proprio = np.zeros((len(images), 8), dtype=np.float64)
        proprio[:, 0] = [359.0, 0.0, 1.0]
    if actions is None:
        actions = np.arange(len(images) * 7, dtype=np.float64).reshape(len(images), 7)
    if timestamps is None:
        timestamps = np.arange(len(images), dtype=np.float64)

    with h5py.File(path, "w") as f:
        obs = f.create_group("obs")
        obs.create_dataset("image", data=images)
        obs.create_dataset("proprio", data=proprio)
        f.create_dataset("action", data=actions)
        f.create_dataset("timestamps", data=timestamps)
        f.attrs["camera_fps"] = 10
        for key, value in (attrs or {}).items():
            f.attrs[key] = value
    return images, proprio, actions, timestamps


def test_load_current_rgb_episode_preserves_rgb_and_metadata(tmp_path):
    path = tmp_path / "episode.h5"
    images, proprio, actions, timestamps = _write_episode(path)

    meta = load_episode_h5(path)
    reader = EpisodeFrameReader(meta["images"])

    assert meta["num_frames"] == 3
    assert (meta["H"], meta["W"]) == (2, 2)
    assert meta["camera_fps_declared"] == 10.0
    assert meta["camera_fps"] == 1.0
    np.testing.assert_array_equal(reader.read_frame(0)[0, 0], [255, 0, 17])
    np.testing.assert_allclose(meta["joint_positions"], np.deg2rad(proprio[:, :7]))
    np.testing.assert_array_equal(meta["actions"], actions)
    np.testing.assert_array_equal(meta["timestamps"], timestamps)


def test_loads_calibrated_camera_metadata_and_exposure(tmp_path):
    path = tmp_path / "calibrated.h5"
    _write_episode(
        path,
        attrs={
            "fx": 301.0,
            "fy": 302.0,
            "cx": 159.5,
            "cy": 119.5,
            "exposure_seconds": 0.01,
            "camera_backend": "V4L2",
            "camera_intrinsics_source": "estimated_diagonal_fov_75deg",
        },
    )

    meta = load_episode_h5(path)

    assert meta["camera_intrinsics"] == {
        "fx": 301.0,
        "fy": 302.0,
        "cx": 159.5,
        "cy": 119.5,
    }
    assert meta["exposure_seconds"] == 0.01
    assert meta["camera_backend"] == "V4L2"
    assert meta["camera_intrinsics_source"] == "estimated_diagonal_fov_75deg"


def test_joint_velocity_uses_wrapped_centered_differences(tmp_path):
    path = tmp_path / "episode.h5"
    _write_episode(path)

    velocity = load_episode_h5(path)["joint_velocities"]

    np.testing.assert_allclose(velocity[:, 0], np.deg2rad([1.0, 1.0, 1.0]))
    np.testing.assert_allclose(velocity[:, 1:], 0.0)


@pytest.mark.parametrize(
    ("images", "proprio", "actions", "timestamps", "message"),
    [
        (np.zeros((3, 2, 2), np.uint8), None, None, None, "obs/image"),
        (np.zeros((3, 2, 2, 3), np.float32), None, None, None, "uint8"),
        (None, np.zeros((2, 8)), None, None, "same number"),
        (None, None, None, np.array([0.0, 1.0, 1.0]), "strictly increasing"),
    ],
)
def test_rejects_invalid_current_format(
    tmp_path, images, proprio, actions, timestamps, message
):
    path = tmp_path / "invalid.h5"
    _write_episode(
        path,
        images=images,
        proprio=proprio,
        actions=actions,
        timestamps=timestamps,
    )

    with pytest.raises(ValueError, match=message):
        load_episode_h5(path)


def test_rejects_missing_required_dataset(tmp_path):
    path = tmp_path / "missing.h5"
    with h5py.File(path, "w") as f:
        f.create_group("obs")

    with pytest.raises(ValueError, match="obs/image"):
        load_episode_h5(path)


def test_frame_reader_rejects_non_rgb_arrays():
    with pytest.raises(ValueError, match="RGB"):
        EpisodeFrameReader(np.zeros((2, 3, 4), dtype=np.uint8))
