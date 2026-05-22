"""Tests for corner_maze_rl.encoders.grid_cells.

Math correctness for circular_gaussian and encode_pose_to_vector, plus the
full GridCellEncoder against the bundled prebuilt dictionary.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from corner_maze_rl.encoders.base import CompositeEncoder
from corner_maze_rl.encoders.grid_cells import (
    D_RANGE,
    GRID_ANGLES,
    HEADING_MAP,
    NUM_MODULES,
    OUTPUT_DIM,
    VM_SIGMA,
    X_RANGE,
    Y_RANGE,
    GridCellEncoder,
    circular_gaussian,
    encode_pose_to_vector,
    make_pose_vector_dict,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PREBUILT_DICT = REPO_ROOT / "data" / "lookups" / "grid_cells_60d.npz"


# ---------------------------------------------------------------------------
# circular_gaussian
# ---------------------------------------------------------------------------

def test_circular_gaussian_peaks_at_target():
    assert circular_gaussian(45.0, 45.0, 45.0) == pytest.approx(1.0)


def test_circular_gaussian_decays_with_distance():
    peak = circular_gaussian(45.0, 45.0, 45.0)
    near = circular_gaussian(50.0, 45.0, 45.0)
    far = circular_gaussian(120.0, 45.0, 45.0)
    assert peak > near > far > 0.0


def test_circular_gaussian_wraps_around():
    """Should treat 350° and 10° as 20° apart, not 340° apart."""
    near_clockwise = circular_gaussian(350.0, 10.0, 30.0)
    near_ccw = circular_gaussian(10.0, 350.0, 30.0)
    same_angle = circular_gaussian(180.0, 200.0, 30.0)
    assert near_clockwise == pytest.approx(near_ccw)
    # 20° apart with sigma=30 gives a value clearly above zero
    assert near_clockwise > 0.5


def test_circular_gaussian_zero_far_away():
    """At ~5 sigma the value is essentially zero."""
    assert circular_gaussian(180.0, 0.0, 30.0) < 1e-5


# ---------------------------------------------------------------------------
# encode_pose_to_vector — math
# ---------------------------------------------------------------------------

@pytest.fixture
def synth_grid_stack():
    """Synthetic grid stack: each cell holds a unique signature.

    Module m at (y, x) cell -> phases [m + 0.1*x, m + 0.1*y, m + 0.1*(x+y)]
    """
    g = np.zeros((NUM_MODULES, 11, 11, 3), dtype=np.float32)
    for m in range(NUM_MODULES):
        for y in range(11):
            for x in range(11):
                g[m, y, x, 0] = m + 0.1 * x
                g[m, y, x, 1] = m + 0.1 * y
                g[m, y, x, 2] = m + 0.1 * (x + y)
    return g


def test_encode_pose_returns_60d_float32(synth_grid_stack):
    v = encode_pose_to_vector(5, 6, 0, synth_grid_stack)
    assert v.shape == (60,)
    assert v.dtype == np.float32


def test_encode_pose_first_15_are_position_only(synth_grid_stack):
    """Group 1 (positions 0..14) is direction-invariant."""
    v_east = encode_pose_to_vector(5, 6, 0, synth_grid_stack)
    v_west = encode_pose_to_vector(5, 6, 2, synth_grid_stack)
    assert np.allclose(v_east[:15], v_west[:15])


def test_encode_pose_direction_changes_groups_2_3_4(synth_grid_stack):
    v_east = encode_pose_to_vector(5, 6, 0, synth_grid_stack)
    v_north = encode_pose_to_vector(5, 6, 3, synth_grid_stack)
    # Same position, different heading: groups 2-4 must differ.
    assert not np.allclose(v_east[15:], v_north[15:])


def test_encode_pose_clamps_out_of_range(synth_grid_stack):
    """Values outside 1..11 should be clamped, not error."""
    v_in = encode_pose_to_vector(11, 11, 0, synth_grid_stack)
    v_clamped = encode_pose_to_vector(99, 99, 0, synth_grid_stack)
    assert np.allclose(v_in, v_clamped)


def test_encode_pose_legacy_yx_indexing_convention(synth_grid_stack):
    """Legacy uses tensor[y_idx, x_idx]; positions (5,6) and (6,5) differ."""
    v_56 = encode_pose_to_vector(5, 6, 0, synth_grid_stack)
    v_65 = encode_pose_to_vector(6, 5, 0, synth_grid_stack)
    assert not np.allclose(v_56[:15], v_65[:15])


def test_make_pose_vector_dict_size_and_shape(synth_grid_stack):
    d = make_pose_vector_dict(synth_grid_stack)
    assert len(d) == 11 * 11 * 4  # 484
    sample = next(iter(d.values()))
    assert sample.shape == (60,) and sample.dtype == np.float32


# ---------------------------------------------------------------------------
# GridCellEncoder
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not PREBUILT_DICT.is_file(), reason="prebuilt dict not bundled")
def test_grid_cell_encoder_loads_default():
    enc = GridCellEncoder()
    assert enc.output_dim == 60
    assert enc.n_poses == 11 * 11 * 4


@pytest.mark.skipif(not PREBUILT_DICT.is_file(), reason="prebuilt dict not bundled")
def test_grid_cell_encoder_lookup_shape_and_dtype():
    enc = GridCellEncoder()
    v = enc.encode(5, 6, 0)
    assert v.shape == (60,) and v.dtype == np.float32


@pytest.mark.skipif(not PREBUILT_DICT.is_file(), reason="prebuilt dict not bundled")
def test_grid_cell_encoder_layout_kwarg_ignored():
    enc = GridCellEncoder()
    v1 = enc.encode(5, 6, 0)
    v2 = enc.encode(5, 6, 0, layout="anything")
    assert np.array_equal(v1, v2)


@pytest.mark.skipif(not PREBUILT_DICT.is_file(), reason="prebuilt dict not bundled")
def test_grid_cell_encoder_unknown_pose_raises():
    enc = GridCellEncoder()
    with pytest.raises(KeyError):
        enc.encode(99, 99, 0)


def test_grid_cell_encoder_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        GridCellEncoder(tmp_path / "does_not_exist.pkl")


# ---------------------------------------------------------------------------
# CompositeEncoder
# ---------------------------------------------------------------------------

class _ConstEncoder:
    def __init__(self, dim: int, value: float):
        self.output_dim = dim
        self._v = np.full(dim, value, dtype=np.float32)

    def encode(self, x, y, direction, layout=None):
        return self._v


def test_composite_concatenates_in_order():
    a = _ConstEncoder(3, 1.0)
    b = _ConstEncoder(2, 2.0)
    comp = CompositeEncoder([a, b])
    assert comp.output_dim == 5
    out = comp.encode(0, 0, 0)
    assert out.tolist() == [1.0, 1.0, 1.0, 2.0, 2.0]


def test_composite_rejects_empty():
    with pytest.raises(ValueError):
        CompositeEncoder([])
