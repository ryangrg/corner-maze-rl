"""60-D grid-cell pose encoder.

Ported from legacy ``notebooks/DatasetBuilder.ipynb`` cell 1 with the Drive /
Colab plumbing stripped out. Pure-numpy implementation.

Layout of the 60-D vector (5 modules × 3 phases per group, 4 groups):

  * Group 1 (base, no directional weighting): 5*3 = 15 values
  * Group 2 (directional, peak at module_angle):           15 values
  * Group 3 (directional, peak at module_angle + 120):     15 values
  * Group 4 (directional, peak at module_angle + 240):     15 values
  * Total: 60 values, float32.

The "module" tensors (shape ``(11, 11, 3)`` per module) come from the legacy
asset pipeline; they aren't reproduced here. We ship the prebuilt
``data/lookups/grid_cells_60d.npz`` as the canonical encoder, and provide
``make_pose_vector_dict`` for users who want to regenerate from their own
modules.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


# ---------------------------------------------------------------------------
# Defaults from legacy DatasetBuilder.ipynb cell 1
# ---------------------------------------------------------------------------

NUM_MODULES: int = 5
GRID_ANGLES: tuple[int, ...] = (45, 45, 45, 45, 45)  # orientation A per module
VM_SIGMA: float = 45.0   # degrees
VM_PEAK_VAL: float = 1.0

# heading int -> degrees. 0=East, 1=South, 2=West, 3=North.
HEADING_MAP: dict[int, float] = {0: 0.0, 1: 270.0, 2: 180.0, 3: 90.0}

# Default pose iteration ranges (matches the prebuilt dict).
X_RANGE: range = range(1, 12)
Y_RANGE: range = range(1, 12)
D_RANGE: range = range(0, 4)

OUTPUT_DIM: int = 60


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------

def circular_gaussian(angle_deg: float, peak_deg: float, sigma_deg: float) -> float:
    """Von-Mises-style Gaussian on the unit circle.

    Returns 1.0 at ``angle_deg == peak_deg`` and decays with sigma_deg
    measured along the *shortest* arc.
    """
    diff = abs(angle_deg - peak_deg) % 360.0
    shortest = min(diff, 360.0 - diff)
    return float(np.exp(-0.5 * (shortest / sigma_deg) ** 2))


def encode_pose_to_vector(
    x: int,
    y: int,
    direction: int,
    grid_stack: np.ndarray,
    *,
    grid_angles: Iterable[int] = GRID_ANGLES,
    heading_map: Mapping[int, float] = HEADING_MAP,
    vm_sigma: float = VM_SIGMA,
) -> np.ndarray:
    """Compute the 60-D vector for one pose given the grid-module stack.

    Parameters
    ----------
    x, y : 1-indexed maze coordinates (1..11 inclusive). Clamped to that range.
    direction : 0..3 heading int.
    grid_stack : ``(num_modules, 11, 11, 3)`` ndarray of module activations
        — the legacy convention is ``tensor[y_idx, x_idx]``.
    grid_angles : per-module orientation A in degrees.
    heading_map : direction-int → degrees.
    vm_sigma : sigma for the von-Mises-style directional gaussian.

    Returns
    -------
    np.ndarray, shape ``(60,)``, dtype float32.
    """
    num_modules = grid_stack.shape[0]
    grid_angles = list(grid_angles)
    if len(grid_angles) != num_modules:
        raise ValueError(
            f"grid_angles length {len(grid_angles)} != num_modules {num_modules}"
        )

    idx_x = max(0, min(10, x - 1))
    idx_y = max(0, min(10, y - 1))

    # Group 1: base. Shape (num_modules, 3) -> flat (num_modules*3,)
    base = grid_stack[:, idx_y, idx_x, :].reshape(num_modules, 3)
    base_vals = base.flatten()

    heading_deg = heading_map.get(direction, 0.0)

    g2: list[float] = []
    g3: list[float] = []
    g4: list[float] = []
    for m in range(num_modules):
        a = grid_angles[m]
        c2 = circular_gaussian(heading_deg, a, vm_sigma)
        c3 = circular_gaussian(heading_deg, a + 120, vm_sigma)
        c4 = circular_gaussian(heading_deg, a + 240, vm_sigma)
        mod_base = base[m]
        g2.extend(mod_base * c2)
        g3.extend(mod_base * c3)
        g4.extend(mod_base * c4)

    return np.concatenate(
        [base_vals, np.asarray(g2), np.asarray(g3), np.asarray(g4)]
    ).astype(np.float32, copy=False)


def make_pose_vector_dict(
    grid_stack: np.ndarray,
    *,
    x_range: Iterable[int] = X_RANGE,
    y_range: Iterable[int] = Y_RANGE,
    d_range: Iterable[int] = D_RANGE,
    grid_angles: Iterable[int] = GRID_ANGLES,
    heading_map: Mapping[int, float] = HEADING_MAP,
    vm_sigma: float = VM_SIGMA,
) -> dict[tuple[int, int, int], np.ndarray]:
    """Build the full ``(x, y, d) -> 60-D vector`` dict for a module stack."""
    out: dict[tuple[int, int, int], np.ndarray] = {}
    for x in x_range:
        for y in y_range:
            for d in d_range:
                out[(x, y, d)] = encode_pose_to_vector(
                    x, y, d, grid_stack,
                    grid_angles=grid_angles,
                    heading_map=heading_map,
                    vm_sigma=vm_sigma,
                )
    return out


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class GridCellEncoder:
    """StateEncoder using a precomputed pose -> 60-D vector dictionary.

    The default lookup path points at ``data/lookups/grid_cells_60d.npz``
    (484 poses; 11x11x4). The npz has two arrays:
      keys:    (N, 3) int16  — (x, y, direction)
      vectors: (N, 60) float32

    Pass ``dict_path`` to use a different file (e.g. one regenerated via
    ``make_pose_vector_dict``).
    """

    output_dim: int = OUTPUT_DIM

    def __init__(self, dict_path: str | Path | None = None):
        path = Path(dict_path) if dict_path is not None else _default_dict_path()
        if not path.is_file():
            raise FileNotFoundError(
                f"GridCellEncoder lookup not found at {path}. "
                "Either ship grid_cells_60d.npz into data/lookups/ "
                "or build it from grid modules via make_pose_vector_dict()."
            )
        with np.load(path, allow_pickle=False) as z:
            keys = z["keys"]
            vectors = z["vectors"]
        self._dict: dict[tuple[int, int, int], np.ndarray] = {
            (int(k[0]), int(k[1]), int(k[2])): vectors[i]
            for i, k in enumerate(keys)
        }
        self._dict_path = path

    @property
    def dict_path(self) -> Path:
        return self._dict_path

    @property
    def n_poses(self) -> int:
        return len(self._dict)

    def encode(
        self,
        x: int,
        y: int,
        direction: int,
        layout: str | None = None,  # ignored; grid_cells is layout-invariant
    ) -> np.ndarray:
        try:
            return self._dict[(int(x), int(y), int(direction))]
        except KeyError:
            raise KeyError(
                f"GridCellEncoder: pose ({x}, {y}, {direction}) not in dict"
            ) from None


def _default_dict_path() -> Path:
    """Locate the bundled pose-vector lookup.

    Looks in ``<repo>/data/lookups/`` relative to the package install.
    """
    return Path(__file__).resolve().parents[3] / "data" / "lookups" / "grid_cells_60d.npz"
