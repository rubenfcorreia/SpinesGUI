import numpy as np
import pytest

from registration_viewer import (
    RegistrationData, compose_pc_images, normalize_pc_image,
    registration_pc_curves, resample_trace, selected_pc_values,
)


def make_plane(tmp_path, *, with_reg_outputs=True, reported_frames=4):
    plane = tmp_path / "plane0"
    plane.mkdir()
    movie = np.arange(3 * 4 * 5, dtype=np.int16).reshape(3, 4, 5)
    movie.tofile(plane / "data.bin")
    ops = {
        "Ly": 4,
        "Lx": 5,
        "nframes": reported_frames,
        "datatype": "int16",
        "reg_file": str(plane / "data.bin"),
    }
    np.save(plane / "ops.npy", ops, allow_pickle=True)
    if with_reg_outputs:
        np.save(plane / "reg_outputs.npy", {
            "xoff": np.array([0, 1, 2]),
            "yoff": np.array([3, 4, 5]),
            "corrXY": np.array([0.1, 0.2, 0.3]),
            "badframes": np.array([False, True, False]),
            "badframes0": np.array([False, False, False]),
            "regPC": np.zeros((2, 2, 2, 2), dtype=np.float32),
            "tPC": np.zeros((3, 2), dtype=np.float32),
        }, allow_pickle=True)
    return {"folder": str(plane), "ops_path": str(plane / "ops.npy")}


def test_registration_only_folder_reads_frames_lazily_and_clamps_count(tmp_path):
    data = RegistrationData(make_plane(tmp_path, reported_frames=4))
    assert data.nframes == 3
    assert "using 3" in data.frame_count_note
    assert np.array_equal(data.read_frame(2), np.arange(40, 60, dtype=np.int16).reshape(4, 5))
    assert data.metric("xoff").tolist() == [0, 1, 2]
    assert data.metric("badframes").tolist() == [False, True, False]
    assert data.pc(1)[0].shape == (2, 2)
    assert data.tpc(1).shape == (3,)


def test_missing_reg_outputs_preserves_movie_access(tmp_path):
    data = RegistrationData(make_plane(tmp_path, with_reg_outputs=False))
    assert data.read_frame(0).shape == (4, 5)
    assert data.metric("corrXY") is None
    assert data.pc(0) is None


def test_missing_registered_binary_is_actionable(tmp_path):
    plane = make_plane(tmp_path)
    (tmp_path / "plane0" / "data.bin").unlink()
    with pytest.raises(FileNotFoundError, match="Registered binary not found"):
        RegistrationData(plane)


def test_pc_composition_and_safe_normalization():
    top = np.zeros((2, 2), dtype=np.float32)
    bottom = np.ones((2, 2), dtype=np.float32)
    difference, merged, animated = compose_pc_images(top, bottom)
    assert difference.shape == (2, 2)
    assert merged.shape == (2, 2, 3)
    assert animated.shape == (2, 2, 2)
    assert np.all(np.isfinite(normalize_pc_image(top)))


def test_registration_shift_curves_and_selected_pc_values():
    reg_dx = np.arange(15, dtype=np.float32).reshape(5, 3)
    assert registration_pc_curves(reg_dx).shape == (5, 3)
    assert np.array_equal(selected_pc_values(reg_dx, 2), reg_dx[2])
    assert selected_pc_values(reg_dx, 5) is None


def test_resample_trace_aligns_registration_shifts_to_tpc_samples():
    result = resample_trace(np.array([0.0, 10.0, 20.0, 30.0]), 3)
    assert result.shape == (3,)
    assert np.allclose(result, [0.0, 15.0, 30.0])
    assert np.array_equal(resample_trace([1, 2, 3], 3), [1, 2, 3])
