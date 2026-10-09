"""Map session cord tracking data (x, y, zone) to minigrid grid cells.

Provides zone_to_grid() for single-point mapping and map_session_to_grid()
for batch-mapping a coordinate DataFrame.
"""
import os
import sys

import numpy as np
import pandas as pd

# ── Zone-to-grid mapping tables ───────────────

# Wells: zone -> fixed (grid_x, grid_y)
WELL_ZONES = {
    1:  (1, 11),   # SW
    5:  (1, 1),    # NW
    21: (11, 1),   # NE
    17: (11, 11),  # SE
}

# Intersections: zone -> fixed (grid_x, grid_y)
#
# Compass convention (the rig's zone map is the ground truth; see
# corner-maze-analysis/docs/context/07_pitfalls.md A7/B13): tracking x runs
# south (0) -> north (239) and tracking y runs west (0) -> east (239); grid_x
# increases eastward (tracking y) and grid_y increases southward (tracking x
# inverted). Zone 3 is the WEST T-junction (x 98-138, y 0-35), zone 9 the SOUTH
# (x 0-34, y 101-139), zone 13 the NORTH (x 205-239), zone 19 the EAST
# (y 205-239). Until 2026-10-09 zones 3 and 9 were assigned to each other's
# cells; tests/test_zone_to_grid.py pins the mapping to the zone geometry.
INTERSECTION_ZONES = {
    3:  (2, 6),    # W junction
    9:  (6, 10),   # S junction
    11: (6, 6),    # centre
    13: (6, 2),    # N junction
    19: (10, 6),   # E junction
}

# Straight segments: zone -> (axis, min_val, max_val, [cell_low, cell_mid, cell_high])
# 'y' axis: divide by phys_y, increasing y -> increasing grid_x
# 'x' axis: divide by phys_x, increasing x -> decreasing grid_y
SEGMENT_ZONES = {
    # Divide by phys_y
    8:  ('y', 41,  97,  [(3, 2),  (4, 2),  (5, 2)]),
    7:  ('y', 36,  96,  [(3, 6),  (4, 6),  (5, 6)]),
    6:  ('y', 45,  100, [(3, 10), (4, 10), (5, 10)]),
    16: ('y', 140, 195, [(7, 2),  (8, 2),  (9, 2)]),
    15: ('y', 143, 204, [(7, 6),  (8, 6),  (9, 6)]),
    14: ('y', 140, 196, [(7, 10), (8, 10), (9, 10)]),
    # Divide by phys_x (inverted: low x -> high grid_y)
    4:  ('x', 139, 194, [(2, 5),  (2, 4),  (2, 3)]),
    12: ('x', 144, 204, [(6, 5),  (6, 4),  (6, 3)]),
    20: ('x', 140, 195, [(10, 5), (10, 4), (10, 3)]),
    2:  ('x', 43,  97,  [(2, 9),  (2, 8),  (2, 7)]),
    10: ('x', 35,  95,  [(6, 9),  (6, 8),  (6, 7)]),
    18: ('x', 43,  97,  [(10, 9), (10, 8), (10, 7)]),
}

# Perimeter corners: zone-0 gap regions -> fixed (grid_x, grid_y)
# (x_min, x_max, y_min, y_max, grid_x, grid_y)
ZONE0_CORNERS = [
    (150, 230, 0,   80,  2,  2),    # NW: between zones 4, 5, 8
    (150, 230, 160, 239, 10, 2),    # NE: between zones 16, 20, 21
    (0,   80,  160, 239, 10, 10),   # SE: between zones 14, 17, 18
    (0,   80,  0,   80,  2,  10),   # SW: between zones 1, 2, 6
]


# ── Mapping functions ────────────────────────

def zone_to_grid(zone: int, phys_x: int, phys_y: int) -> tuple[int, int]:
    """Map a single (zone, phys_x, phys_y) to a minigrid (grid_x, grid_y)."""
    # Wells
    if zone in WELL_ZONES:
        return WELL_ZONES[zone]

    # Intersections
    if zone in INTERSECTION_ZONES:
        return INTERSECTION_ZONES[zone]

    # Straight segments: subdivide into thirds
    if zone in SEGMENT_ZONES:
        axis, min_val, max_val, cells = SEGMENT_ZONES[zone]
        val = phys_y if axis == 'y' else phys_x
        span = max_val - min_val
        idx = int((val - min_val) / span * 3)
        idx = max(0, min(2, idx))
        return cells[idx]

    # Zone 0: check perimeter corner bounding boxes
    if zone == 0:
        for x_min, x_max, y_min, y_max, gx, gy in ZONE0_CORNERS:
            if x_min <= phys_x <= x_max and y_min <= phys_y <= y_max:
                return (gx, gy)
        return (0, 0)  # unmapped zone-0 point

    return (0, 0)


def map_session_to_grid(coord_df: pd.DataFrame) -> pd.DataFrame:
    """Map a coordinate DataFrame to minigrid grid cells.

    Args:
        coord_df: DataFrame with columns (t_ms, x, y, zone).
            Extra columns (e.g., phase, trial_number) are passed through.

    Returns:
        DataFrame with columns (t_ms, x, y, zone, grid_x, grid_y)
        plus any extra columns from the input.
    """
    zones = coord_df['zone'].values.astype(int)
    xs = coord_df['x'].values.astype(int)
    ys = coord_df['y'].values.astype(int)

    n = len(zones)
    grid_xs = np.zeros(n, dtype=int)
    grid_ys = np.zeros(n, dtype=int)

    for i in range(n):
        gx, gy = zone_to_grid(int(zones[i]), int(xs[i]), int(ys[i]))
        grid_xs[i] = gx
        grid_ys[i] = gy

    result = pd.DataFrame({
        't_ms': coord_df['t_ms'].values,
        'x': xs,
        'y': ys,
        'zone': zones,
        'grid_x': grid_xs,
        'grid_y': grid_ys,
    })

    # Pass through extra columns (phase, trial_number, etc.)
    for col in coord_df.columns:
        if col not in result.columns:
            result[col] = coord_df[col].values

    return result


# ── Standalone script ────────────────────────

if __name__ == '__main__':
    from .zone_pixel_map import ZONE_PIXEL_BOUNDS  # noqa: F401

    SUBJECT = 'CM005'
    SESSION_NUMBER = '1e'
    PARQUET_PATH = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'dataframes', 'all_sessions.parquet'
    )
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'csv')

    df = pd.read_parquet(PARQUET_PATH)
    mask = (df['subject'] == SUBJECT) & (df['session_number'] == SESSION_NUMBER)
    matches = df[mask]

    if len(matches) == 0:
        print(f"No session found for subject={SUBJECT}, session_number={SESSION_NUMBER}")
        sys.exit(1)

    if len(matches) > 1:
        print(f"Found {len(matches)} matching sessions, using the first one.")

    row = matches.iloc[0]
    cord_x = np.array(row['cord_cord_x'], dtype=int)
    cord_y = np.array(row['cord_cord_y'], dtype=int)
    cord_zone = np.array(row['cord_zone'], dtype=int)
    cord_ts = np.array(row['cord_time_stamp'], dtype=float)

    coord_df = pd.DataFrame({
        't_ms': cord_ts, 'x': cord_x, 'y': cord_y, 'zone': cord_zone,
    })
    grid_df = map_session_to_grid(coord_df)

    # Rename columns for legacy CSV format
    out_df = grid_df.rename(columns={'t_ms': 'time_stamp', 'x': 'cord_x', 'y': 'cord_y'})

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{SUBJECT}_{SESSION_NUMBER}.csv")
    out_df.to_csv(out_path, index=False)

    session_type = row.get('session_type', '')
    n = len(out_df)
    print(f"Wrote {n} rows to {out_path}")
    print(f"  Subject: {SUBJECT}, Session: {SESSION_NUMBER} ({session_type})")
    print(f"  Unmapped (0,0) points: {((grid_df['grid_x'] == 0) & (grid_df['grid_y'] == 0)).sum()}")
