"""Pin the zone -> grid mapping to the rig's zone geometry.

The rig's 240x240 tracking frame (x: 0 = south .. 239 = north; y: 0 = west ..
239 = east) and its 21 zones are the ground truth (zone_pixel_map, ported from
the Pi tracker). These tests derive what the mapping must satisfy from that
geometry alone, so a compass slip in a hand-written table (such as the zone 3 /
zone 9 swap fixed on 2026-10-09) fails here rather than in the yoked data.
"""
import pytest

from corner_maze_rl.yoking.map_to_minigrid import (
    INTERSECTION_ZONES, SEGMENT_ZONES, WELL_ZONES, ZONE0_CORNERS, zone_to_grid,
)
from corner_maze_rl.yoking.zone_pixel_map import ZONE_PIXEL_BOUNDS


def _box(zone):
    b = ZONE_PIXEL_BOUNDS[zone]
    return b['x'][0], b['x'][1], b['y'][0], b['y'][1]


# The walkable zone graph of the maze, by zone id only (no compass words), so a
# compass slip in the mapping cannot hide here. Ring segments meet the wells
# through the zone-0 perimeter corners (ZONE0_CORNERS), listed separately.
WALKABLE_EDGES = [
    (11, 7), (11, 12), (11, 15), (11, 10),          # centre to the four arms
    (7, 3), (12, 13), (15, 19), (10, 9),            # arm to its T-junction
    (3, 2), (3, 4), (13, 8), (13, 16),              # junction to the two ring segments
    (19, 18), (19, 20), (9, 6), (9, 14),
]
# (ring segment, well) pairs that meet at a zone-0 corner
CORNER_PAIRS = [(2, 1), (6, 1), (4, 5), (8, 5), (14, 17), (18, 17), (16, 21), (20, 21)]


def _cells_of(zone):
    """All grid cells a zone can map to (segments map to three)."""
    if zone in WELL_ZONES:
        return {WELL_ZONES[zone]}
    if zone in INTERSECTION_ZONES:
        return {INTERSECTION_ZONES[zone]}
    if zone in SEGMENT_ZONES:
        return set(SEGMENT_ZONES[zone][3])
    raise KeyError(zone)


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


NON_WELL = [z for z in range(1, 22) if z not in WELL_ZONES]


@pytest.mark.parametrize('zone', NON_WELL)
def test_compass_of_each_zone_matches_its_box(zone):
    """West boxes (small y) must land in the west column, and so on."""
    x0, x1, y0, y1 = _box(zone)
    for gx, gy in _cells_of(zone):
        if y1 <= 40:            # west edge of the frame
            assert gx == 2, (zone, gx, gy)
        if y0 >= 200:           # east edge
            assert gx == 10, (zone, gx, gy)
        if x1 <= 40:            # south edge
            assert gy == 10, (zone, gx, gy)
        if x0 >= 200:           # north edge
            assert gy == 2, (zone, gx, gy)
        if 90 <= x0 and x1 <= 145 and 35 <= y0 and y1 <= 205:   # the plus-maze column
            assert gx == 6 or gy == 6, (zone, gx, gy)


@pytest.mark.parametrize('a,b', WALKABLE_EDGES)
def test_walkable_neighbours_map_to_adjacent_cells(a, b):
    """Zones the rat can step between directly must map to cells that are one
    step apart (for some pair of their cells)."""
    best = min(_manhattan(ca, cb) for ca in _cells_of(a) for cb in _cells_of(b))
    assert best == 1, (a, b, _cells_of(a), _cells_of(b), best)


@pytest.mark.parametrize('segment,well', CORNER_PAIRS)
def test_ring_segments_reach_their_wells_through_a_corner(segment, well):
    """The ring segment's end cell is one step from a zone-0 corner cell that is
    itself diagonal to the well cell."""
    well_cell = WELL_ZONES[well]
    corners = {(gx, gy) for *_, gx, gy in ZONE0_CORNERS}
    corner = min(corners, key=lambda c: _manhattan(c, well_cell))
    assert _manhattan(corner, well_cell) == 2, (well, corner, well_cell)
    assert min(_manhattan(corner, c) for c in _cells_of(segment)) == 1, (segment, corner)


def test_junctions_sit_between_their_ring_and_arm():
    """Each T-junction cell is adjacent to the end cells of both ring
    segments beside it and to the outer cell of its arm."""
    expected = {
        3:  {2, 4, 7},     # W junction: W ring (south half, north half), W arm
        9:  {6, 14, 10},   # S junction
        13: {8, 16, 12},   # N junction
        19: {18, 20, 15},  # E junction
    }
    for junction, neighbours in expected.items():
        cell = INTERSECTION_ZONES[junction]
        for z in neighbours:
            assert min(_manhattan(cell, c) for c in _cells_of(z)) == 1, (junction, z)


def test_zone_to_grid_uses_the_table():
    assert zone_to_grid(3, 118, 10) == (2, 6)
    assert zone_to_grid(9, 10, 120) == (6, 10)
    assert zone_to_grid(13, 220, 120) == (6, 2)
    assert zone_to_grid(19, 118, 220) == (10, 6)
    assert zone_to_grid(11, 120, 120) == (6, 6)
