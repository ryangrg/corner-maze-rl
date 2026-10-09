"""The vendored task-context constants (doors.json, zones.json) and the env.

The files under src/corner_maze_rl/data/ are copies of
corner-maze-analysis/docs/context/constants/. context_constants.json records
their SHA-256 so a copy cannot drift silently: changing a file without
refreshing the manifest fails here. The rig's door order is the ground truth;
the env's BARRIER_LOCATIONS and the yoking's zone-to-grid table must agree with
the registry.
"""
import hashlib
import json
from pathlib import Path

from corner_maze_rl.env.constants import BARRIER_LOCATIONS
from corner_maze_rl.yoking.map_to_minigrid import INTERSECTION_ZONES, SEGMENT_ZONES, WELL_ZONES

DATA = Path(__file__).resolve().parents[1] / "src" / "corner_maze_rl" / "data"
MANIFEST = json.loads((DATA / "context_constants.json").read_text())


def test_vendored_files_match_their_recorded_hashes():
    for name, entry in MANIFEST.items():
        digest = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
        assert digest == entry["sha256"], (
            f"{name} differs from the recorded copy; update it from {entry['source']} "
            "and refresh context_constants.json deliberately"
        )


def test_barrier_locations_follow_the_registry():
    reg = json.loads((DATA / "doors.json").read_text())
    assert [list(b) for b in BARRIER_LOCATIONS] == reg["env_barrier_locations"]
    for d in reg["doors"]:
        assert list(BARRIER_LOCATIONS[d["env_index"]]) == d["grid_cell"], d["id"]


def test_zone_to_grid_table_follows_the_zone_constants():
    zones = {z["id"]: z for z in json.loads((DATA / "zones.json").read_text())["zones"]}
    for zid, cell in WELL_ZONES.items():
        assert [list(cell)] == zones[zid]["grid_cells"], zid
    for zid, cell in INTERSECTION_ZONES.items():
        assert [list(cell)] == zones[zid]["grid_cells"], zid
    for zid, (_axis, _lo, _hi, cells) in SEGMENT_ZONES.items():
        assert [list(c) for c in cells] == zones[zid]["grid_cells"], zid


def test_registry_permutation_is_a_bijection():
    reg = json.loads((DATA / "doors.json").read_text())
    assert sorted(reg["env_index_to_rig_door"]) == sorted(f"B{i}" for i in range(1, 17))
