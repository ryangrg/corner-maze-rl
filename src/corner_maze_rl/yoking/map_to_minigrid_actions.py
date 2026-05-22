"""Convert grid-mapped session data to MiniGrid action sequences.

Provides build_action_sequence() for pipeline use and standalone script mode
for legacy CSV-based processing.

Direction is inferred from consecutive grid position differences (no BFS).
Well visits and pretrials use hard-coded forced action blocks.
"""
import os
from collections import deque

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────
# Actions: 0=left, 1=right, 2=forward, 3=pickup(well entry), 4=pause
ACT_LEFT = 0
ACT_RIGHT = 1
ACT_FORWARD = 2
ACT_PICKUP = 3
ACT_PAUSE = 4

# Directions: 0=East(+x), 1=South(+y), 2=West(-x), 3=North(-y)
DELTA_TO_DIR = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}

WELL_POSITIONS = {(1, 1), (11, 1), (1, 11), (11, 11)}
CORNER_TO_WELL = {(2, 2): (1, 1), (10, 2): (11, 1), (2, 10): (1, 11), (10, 10): (11, 11)}
WELL_TO_CORNER = {v: k for k, v in CORNER_TO_WELL.items()}

# After Pickup: agent facing direction at each well
WELL_ENTRY_DIR = {(1, 1): 3, (11, 1): 0, (1, 11): 2, (11, 11): 1}
# After Forward exit: (corner_pos, corner_dir)
WELL_EXIT_RESULT = {
    (11, 1): ((10, 2), 2), (11, 11): ((10, 10), 3),
    (1, 11): ((2, 10), 0), (1, 1): ((2, 2), 1),
}

# Zone-to-well mapping for reward event alignment
ZONE_TO_WELL_POS = {21: (11, 1), 17: (11, 11), 1: (1, 11), 5: (1, 1)}

# ── Walkable grid cells (from _gen_grid construction) ─
# Three horizontal rows at y=2,6,10: x=2..10
# Three vertical columns at x=2,6,10: y=3..5, y=7..9
WALKABLE_CELLS = set()
for _i in range(3):
    for _j in range(9):
        WALKABLE_CELLS.add((_j + 2, 4 * _i + 2))
for _i in range(3):
    for _j in range(3):
        WALKABLE_CELLS.add((4 * _i + 2, _j + 3))
        WALKABLE_CELLS.add((4 * _i + 2, _j + 7))


def find_path(start, end, blocked=None):
    """BFS shortest path on the walkable grid. Returns list of positions
    including start and end, or empty list if no path exists.

    Args:
        start: Starting position (x, y).
        end: Target position (x, y).
        blocked: Optional set of positions to avoid (active barriers).
    """
    if start == end:
        return [start]
    visited = {start}
    queue = deque([(start, [start])])
    while queue:
        pos, path = queue.popleft()
        for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt in WALKABLE_CELLS and nxt not in visited:
                if blocked and nxt in blocked:
                    continue
                new_path = path + [nxt]
                if nxt == end:
                    return new_path
                visited.add(nxt)
                queue.append((nxt, new_path))
    return []


# ── Grid position consolidation ──────────────
def consolidate_grid(grid_x, grid_y, max_gap=2):
    """Deduplicate and merge grid position runs, handling flicker.

    Returns list of (gx, gy, start_idx, end_idx) tuples.
    """
    n = len(grid_x)
    if n == 0:
        return []

    # Step 1: deduplicate consecutive identical positions into runs
    runs = []
    i = 0
    while i < n:
        gx, gy = int(grid_x[i]), int(grid_y[i])
        j = i + 1
        while j < n and int(grid_x[j]) == gx and int(grid_y[j]) == gy:
            j += 1
        runs.append([gx, gy, i, j])
        i = j

    # Step 2: iteratively merge same-position runs separated by brief gaps
    changed = True
    while changed:
        changed = False
        new_runs = []
        i = 0
        while i < len(runs):
            if (i + 2 < len(runs)
                    and runs[i][0] == runs[i + 2][0]
                    and runs[i][1] == runs[i + 2][1]
                    and (runs[i + 1][3] - runs[i + 1][2]) <= max_gap):
                new_runs.append([runs[i][0], runs[i][1], runs[i][2], runs[i + 2][3]])
                i += 3
                changed = True
            else:
                new_runs.append(runs[i])
                i += 1
        runs = new_runs

    return [(gx, gy, s, e) for gx, gy, s, e in runs]


def filter_phantom_jumps(runs, jump_threshold=4):
    """Remove phantom position jumps caused by zone boundary artifacts.

    A run is considered a phantom if:
    - It's far from its predecessor (manhattan > threshold)
    - The next run returns close to the predecessor (manhattan <= threshold)
    This pattern indicates a brief zone misclassification, not real movement.

    Args:
        runs: List of (gx, gy, start_idx, end_idx) tuples from consolidate_grid.
        jump_threshold: Max manhattan distance for "reachable in one step".
            Positions beyond this from the previous run are suspicious.

    Returns:
        Filtered list of runs with phantoms removed.
    """
    if len(runs) <= 2:
        return runs

    filtered = [runs[0]]
    i = 1
    while i < len(runs) - 1:
        prev = filtered[-1]
        curr = runs[i]
        nxt = runs[i + 1]

        dist_prev_curr = abs(curr[0] - prev[0]) + abs(curr[1] - prev[1])
        dist_prev_nxt = abs(nxt[0] - prev[0]) + abs(nxt[1] - prev[1])

        if dist_prev_curr > jump_threshold and dist_prev_nxt <= jump_threshold:
            # Phantom: far jump that snaps back — skip it
            i += 1
            continue

        filtered.append(curr)
        i += 1

    # Always keep the last run
    if len(runs) > 1:
        filtered.append(runs[-1])

    return filtered


# ── Barrier remapping ────────────────────────
def _remap_blocked_to_neighbor(grid_x, grid_y, blocked, valid=None):
    """Remap positions at blocked/invalid cells to nearest valid neighbor.

    For each frame where (grid_x[i], grid_y[i]) is in `blocked` (or not in
    `valid` when provided), replaces the position with the nearest cell that
    is walkable and not blocked, by manhattan distance. Modifies arrays
    in-place.

    Args:
        grid_x: Array of grid x coordinates (modified in-place).
        grid_y: Array of grid y coordinates (modified in-place).
        blocked: Set of blocked positions to remap. Ignored if valid is set.
        valid: If provided, whitelist of valid positions. Any position NOT
            in this set is remapped. Takes precedence over blocked.
    """
    n = len(grid_x)
    if valid is not None:
        # Whitelist mode: remap anything not in valid
        candidates = sorted(valid)
    else:
        # Blocked mode: remap anything in blocked
        candidates = sorted(WALKABLE_CELLS - blocked)

    for i in range(n):
        pos = (int(grid_x[i]), int(grid_y[i]))
        if valid is not None:
            if pos in valid:
                continue
        else:
            if pos not in blocked:
                continue
        # Find nearest valid cell by manhattan distance
        best = None
        best_dist = float('inf')
        for c in candidates:
            d = abs(c[0] - pos[0]) + abs(c[1] - pos[1])
            if d < best_dist:
                best_dist = d
                best = c
        if best is not None:
            grid_x[i] = best[0]
            grid_y[i] = best[1]


def _simulate_iti_barriers_per_frame(grid_x, grid_y, iti_configs, initial_idx, BL, TL):
    """Walk the rat's mapped frames and replicate the env's ITI sub-config
    flips along the path. Return a list of barrier sets (one per frame)
    representing the env's active barriers at that frame.

    The env flips `_iti_config_idx` when the agent steps on an A or B trigger
    cell. After flipping, the new sub-config typically has only S triggers
    (no further A/B), so at most one flip per ITI in practice — but this
    function handles any chain.
    """
    n = len(grid_x)
    if n == 0:
        return []
    current_layout = iti_configs[initial_idx]
    current_idx = initial_idx
    out = []
    for i in range(n):
        barriers = {BL[k] for k in range(16) if current_layout[1 + k] == 1}
        out.append(barriers)
        pos = (int(grid_x[i]), int(grid_y[i]))
        # Trigger fire: agent on a cell whose trigger slot is A (1) or B (2)
        for k in range(12):
            if TL[k] == pos:
                tval = current_layout[25 + k]
                if tval == 1 and current_idx != 1:
                    current_idx = 1
                    current_layout = iti_configs[1]
                elif tval == 2 and current_idx != 2:
                    current_idx = 2
                    current_layout = iti_configs[2]
                break
    return out


def _remap_with_per_frame_barriers(grid_x, grid_y, blocked_per_frame, anchor):
    """Per-frame variant of barrier+connectivity remap.

    For each frame i, if (gx[i], gy[i]) is in `blocked_per_frame[i]` or is
    walkable-but-unreachable from the most recently validated position
    given those barriers, remap to the nearest reachable cell. The anchor
    advances to the last validated reachable position so reachability
    tracks the rat's actual progress through the segment.

    Modifies arrays in-place.
    """
    n = len(grid_x)
    if n == 0 or anchor is None:
        return
    cur_anchor = anchor
    cached_reach = None
    cached_blocked = None
    for i in range(n):
        blocked = blocked_per_frame[i]
        if cached_blocked is not blocked:
            cached_reach = _reachable_from(cur_anchor, blocked)
            cached_blocked = blocked
        pos = (int(grid_x[i]), int(grid_y[i]))
        if pos in WELL_POSITIONS:
            continue
        if pos in cached_reach and pos not in blocked:
            cur_anchor = pos
            continue
        candidates = sorted(cached_reach - WELL_POSITIONS)
        if not candidates:
            continue
        best = None
        best_dist = float('inf')
        for c in candidates:
            d = abs(c[0] - pos[0]) + abs(c[1] - pos[1])
            if d < best_dist:
                best_dist = d
                best = c
        if best is not None:
            grid_x[i] = best[0]
            grid_y[i] = best[1]
            cur_anchor = best


def _reachable_from(anchor, blocked):
    """BFS-reachable walkable cells from `anchor` given the active barrier set.

    Treats well positions as reachable when their corner is reachable
    (rats reach wells via corners, and corners are walkable cells).
    """
    if anchor not in WALKABLE_CELLS:
        return set()
    reachable = {anchor}
    queue = deque([anchor])
    while queue:
        pos = queue.popleft()
        for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt in reachable:
                continue
            if nxt not in WALKABLE_CELLS:
                continue
            if blocked and nxt in blocked:
                continue
            reachable.add(nxt)
            queue.append(nxt)
    for corner, well in CORNER_TO_WELL.items():
        if corner in reachable:
            reachable.add(well)
    return reachable


def _remap_to_reachable_one_pass(grid_x, grid_y, blocked, anchor):
    """Combined barrier + isolated-pocket remap that uses reachability and
    last-known-side tiebreaking.

    Per-frame: if (gx, gy) is in the reachable set from `anchor` under
    `blocked`, keep it and update last_reachable. Otherwise remap to the
    nearest reachable cell, breaking distance ties by minimum distance to
    `last_reachable` (the cell the rat was at on the previous reachable
    frame). This keeps rearings-on-barrier (which the tracker briefly
    maps to the opposite-side isolated pocket) on the rat's actual side
    rather than letting BFS-bridging carry the agent the long way
    around through the opposite half of the maze.

    Replaces the older `_remap_blocked_to_neighbor` +
    `_remap_unreachable_to_reachable` two-stage pipeline for trial-phase
    and ITI processing. Modifies arrays in-place. Skips well positions.
    """
    if anchor is None:
        return
    reachable = _reachable_from(anchor, blocked)
    if not reachable:
        return
    candidates = sorted(reachable - WELL_POSITIONS)
    if not candidates:
        return
    n = len(grid_x)

    # Initialize last_reachable by looking ahead for the first frame that
    # IS reachable. The anchor (pretrial-end position) is often equidistant
    # from both sides of a center barrier, which leaves the tiebreak
    # degenerate for any unreachable frames that occur BEFORE the rat first
    # touches a reachable cell. Look-ahead picks the rat's *actual* side
    # for the initial frames too.
    last_reachable = anchor
    for i in range(n):
        pos = (int(grid_x[i]), int(grid_y[i]))
        if pos in reachable:
            last_reachable = pos
            break

    for i in range(n):
        pos = (int(grid_x[i]), int(grid_y[i]))
        if pos in WELL_POSITIONS:
            continue
        if pos in reachable:
            last_reachable = pos
            continue
        # Snap unreachable position to last_reachable. The unreachable
        # cell is never a real rat position — it's a tracker phantom
        # from rearing on a barrier — so "geometric proximity to the
        # phantom" isn't meaningful. The rat is by definition still
        # near where they last were on a reachable frame, and
        # last_reachable is itself a candidate (d_side=0 wins), so this
        # is equivalent to "stay put while rearing." This collapses
        # opposite-pocket cases (e.g. (6,10) in trl_n_*_xx, (2,6) in
        # trl_e_n_xx) without needing BFS to bridge through the wrong
        # half of the maze.
        best = None
        best_side_d = float('inf')
        for c in candidates:
            d_side = abs(c[0] - last_reachable[0]) + abs(c[1] - last_reachable[1])
            if d_side < best_side_d:
                best = c
                best_side_d = d_side
        if best is not None:
            grid_x[i] = best[0]
            grid_y[i] = best[1]


def _remap_unreachable_to_reachable(grid_x, grid_y, blocked, anchor):
    """Remap frames mapped to barrier-isolated pockets back to reachable cells.

    Same shape as `_remap_blocked_to_neighbor` but the criterion is BFS
    reachability from `anchor` given `blocked`, not membership in the barrier
    set. Catches frames that landed on technically-walkable cells which are
    cut off by active barriers (e.g. (10,6) in `trl_w_s_sw` where (10,5) and
    (10,7) are barriers).

    Critical detail — the unreachable cell is picked to minimize distance
    from the LAST REACHABLE POSITION the rat was at, not from the
    unreachable cell itself. Reason: when the rat rears on a barrier
    (e.g. (2,7) in trl_e_n_xx), the tracker can briefly map to the
    isolated middle pocket at (2,6). The naive nearest-to-pos remap
    would tie-break lexicographically between equidistant cells on
    opposite sides of the barrier — picking (2,4) in the north segment
    when the rat was actually at (2,8) on the south side, then BFS
    bridges through long corridor detours that the rat never walked.
    Anchoring the remap to last_reachable collapses these rearings into
    "stay on the rat's actual side" — equivalent to a forward-into-
    barrier (agent doesn't move), matching reality.

    Modifies arrays in-place. Skips well positions.
    """
    if anchor is None:
        return
    reachable = _reachable_from(anchor, blocked)
    if not reachable:
        return
    candidates = sorted(reachable - WELL_POSITIONS)
    if not candidates:
        return
    n = len(grid_x)
    last_reachable = anchor
    for i in range(n):
        pos = (int(grid_x[i]), int(grid_y[i]))
        if pos in reachable or pos in WELL_POSITIONS:
            if pos in reachable:
                last_reachable = pos
            continue
        best = None
        best_dist = float('inf')
        for c in candidates:
            d = abs(c[0] - last_reachable[0]) + abs(c[1] - last_reachable[1])
            if d < best_dist:
                best_dist = d
                best = c
        if best is not None:
            grid_x[i] = best[0]
            grid_y[i] = best[1]


# ── Pause action generation ───────────────────
def _pause_count(dwell_ms, pause_threshold_ms, consolidate_pauses):
    """Return number of PAUSE actions for a dwell duration.

    When consolidate_pauses is True (default), any dwell exceeding the
    threshold produces exactly 1 PAUSE.  When False, the dwell is divided
    by the threshold to produce proportional PAUSEs (min 1).
    """
    if dwell_ms < pause_threshold_ms:
        return 0
    if consolidate_pauses:
        return 1
    return max(1, int(dwell_ms // pause_threshold_ms))


# ── Turn action generation ────────────────────
def turn_actions(current_dir, target_dir, rng):
    """Generate turn actions to face target_dir from current_dir.

    Returns list of action integers. Uses 50/50 roll for 180 turns.
    """
    diff = (target_dir - current_dir) % 4
    if diff == 0:
        return []
    elif diff == 1:
        return [ACT_RIGHT]
    elif diff == 3:
        return [ACT_LEFT]
    else:  # diff == 2: 180 turnaround
        if rng.random() < 0.5:
            return [ACT_LEFT, ACT_LEFT]
        else:
            return [ACT_RIGHT, ACT_RIGHT]


# ── Forced well visit block ───────────────────
def well_visit_actions(well_pos, rng):
    """Generate the hard-coded action block for a well visit.

    Returns (actions, final_pos, final_dir) where actions is a list of
    action integers and final_pos/dir is the agent state after the block.
    Well entry is atomic: pickup -> turnaround -> exit (no pause).
    """
    entry_dir = WELL_ENTRY_DIR[well_pos]
    corner_pos, corner_dir = WELL_EXIT_RESULT[well_pos]

    actions = []

    # 1. Pickup (enter well)
    actions.append(ACT_PICKUP)

    # 2. Turnaround inside well: 50/50 LL or RR
    if rng.random() < 0.5:
        actions.extend([ACT_LEFT, ACT_LEFT])
    else:
        actions.extend([ACT_RIGHT, ACT_RIGHT])

    # 3. Forward exit
    actions.append(ACT_FORWARD)

    return actions, corner_pos, corner_dir


# ── Forced pretrial block ─────────────────────
def pretrial_actions(rng):
    """Generate hard-coded pretrial sequence with 50/50 turnaround roll.

    Returns list of action integers: F,[LL or RR],F,F,[LL or RR]
    """
    turn1 = [ACT_LEFT, ACT_LEFT] if rng.random() < 0.5 else [ACT_RIGHT, ACT_RIGHT]
    turn2 = [ACT_LEFT, ACT_LEFT] if rng.random() < 0.5 else [ACT_RIGHT, ACT_RIGHT]
    return [ACT_FORWARD] + turn1 + [ACT_FORWARD, ACT_FORWARD] + turn2


# ── Real pretrial action generation ──────────
# Valid grid cells per arm during pretrial (corridor + trigger)
_PRETRIAL_VALID_CELLS = {
    0: {(6, 4), (6, 3), (6, 2)},
    1: {(8, 6), (9, 6), (10, 6)},
    2: {(6, 8), (6, 9), (6, 10)},
    3: {(4, 6), (3, 6), (2, 6)},
}

# Trigger grid position per arm (matches PRETRIAL_TRIGGER_POSITIONS in constants.py)
_PRETRIAL_TRIGGER_GRID = {
    0: (6, 2),
    1: (10, 6),
    2: (6, 10),
    3: (2, 6),
}

# Physical trigger zone per arm. zone_to_grid returns incorrect positions
# for arms 2/3 — override with _PRETRIAL_TRIGGER_GRID[arm] when zone matches.
_PRETRIAL_TRIGGER_ZONE = {0: 13, 1: 19, 2: 9, 3: 3}

_TRIGGER_WAIT_MS = 10_000


def generate_real_pretrial_actions(pretrial_df, arm, rng,
                                   build_pause=True,
                                   pause_threshold_ms=1500,
                                   consolidate_pauses=True):
    """Generate pretrial actions from real tracking data.

    Converts a pretrial coordinate slice into MiniGrid actions by:
    1. Mapping zones to grid with trigger zone override
    2. Filtering to valid pretrial cells (whitelist)
    3. Consolidating with max_gap=0 (no flicker merging)
    4. Walking runs to emit turns, forwards, and pauses
    5. Stopping when trigger fires (trigger position reached after 10s)

    Args:
        pretrial_df: DataFrame with (t_ms, x, y, zone) for this pretrial.
        arm: Arm index (0-3).
        rng: numpy random generator for stochastic turnarounds.
        build_pause: Whether to emit pause actions for dwelling.
        pause_threshold_ms: Dwell threshold for pause.

    Returns:
        (output, current_pos, current_dir) where output is a list of
        (action, gx, gy, direction, rewarded) tuples.
    """
    from .map_to_minigrid import zone_to_grid

    valid_cells = _PRETRIAL_VALID_CELLS[arm]
    trigger_pos = _PRETRIAL_TRIGGER_GRID[arm]
    trigger_zone = _PRETRIAL_TRIGGER_ZONE[arm]
    start_pos, start_dir = _ARM_START_POSE[arm]

    if len(pretrial_df) < 2:
        # Fall back to synthetic if insufficient data
        return [], start_pos, start_dir

    zones = pretrial_df['zone'].values.astype(int)
    xs = pretrial_df['x'].values.astype(int)
    ys = pretrial_df['y'].values.astype(int)
    t_ms = pretrial_df['t_ms'].values.astype(float)

    # Step 1: Map zones to grid with trigger zone override
    n = len(zones)
    grid_x = np.zeros(n, dtype=int)
    grid_y = np.zeros(n, dtype=int)
    for i in range(n):
        if int(zones[i]) == trigger_zone:
            grid_x[i], grid_y[i] = trigger_pos
        else:
            grid_x[i], grid_y[i] = zone_to_grid(int(zones[i]), int(xs[i]),
                                                  int(ys[i]))

    # Step 2: Remap invalid cells to nearest valid pretrial cell
    _remap_blocked_to_neighbor(grid_x, grid_y, blocked=set(), valid=valid_cells)

    # Step 3: Consolidate with max_gap=0 (dedup only, no flicker merging)
    runs = consolidate_grid(grid_x, grid_y, max_gap=0)

    if not runs:
        return [], start_pos, start_dir

    # Step 4: Walk runs and generate actions
    output = []
    current_pos = start_pos
    current_dir = start_dir

    def _walk_to(target, cur_pos, cur_dir):
        """Emit turn+forward actions to walk from cur_pos to target.

        Handles non-adjacent positions by stepping through intermediates
        along the arm (straight line in valid cells).
        """
        steps = []
        while cur_pos != target:
            dx = target[0] - cur_pos[0]
            dy = target[1] - cur_pos[1]
            # Clamp to single step
            step_dx = max(-1, min(1, dx))
            step_dy = max(-1, min(1, dy))
            # Only move along one axis at a time
            if step_dx != 0:
                step_dy = 0
            next_pos = (cur_pos[0] + step_dx, cur_pos[1] + step_dy)
            target_dir = DELTA_TO_DIR[(step_dx, step_dy)]
            turns = turn_actions(cur_dir, target_dir, rng)
            for act in turns:
                steps.append((act, cur_pos[0], cur_pos[1], cur_dir, 0))
                if act == ACT_LEFT:
                    cur_dir = (cur_dir - 1) % 4
                elif act == ACT_RIGHT:
                    cur_dir = (cur_dir + 1) % 4
            steps.append((ACT_FORWARD, cur_pos[0], cur_pos[1], cur_dir, 0))
            cur_pos = next_pos
        return steps, cur_pos, cur_dir

    from corner_maze_rl.env.constants import PRETRIAL_MIN_STEPS

    for ri, (gx, gy, s_idx, e_idx) in enumerate(runs):
        pos = (gx, gy)
        run_start_t = t_ms[s_idx]
        run_end_t = t_ms[min(e_idx - 1, len(t_ms) - 1)]

        # Normal movement (yoked data drives the agent naturally)
        if pos != current_pos:
            steps, current_pos, current_dir = _walk_to(
                pos, current_pos, current_dir)
            output.extend(steps)

        # Trigger fire: agent at trigger via natural movement + enough steps
        if pos == trigger_pos and len(output) >= PRETRIAL_MIN_STEPS:
            break  # Pretrial ends — trial bridge takes over

        # Dwell at position — emit pause(s) for time spent here
        if build_pause:
            dwell_ms = run_end_t - run_start_t
            for _ in range(_pause_count(dwell_ms, pause_threshold_ms,
                                        consolidate_pauses)):
                output.append((ACT_PAUSE, current_pos[0], current_pos[1],
                               current_dir, 0))

    # Bridge to the env's pretrial trigger if the rat's coord-derived path
    # didn't naturally end at it. The env's pretrial→trial transition fires
    # only when the agent is on the trigger cell AND pretrial_step_count
    # has reached PRETRIAL_MIN_STEPS. Some rats drive through the trigger
    # cell too early (before the count is satisfied) and then wander to
    # the arm interior, leaving the env stuck in pretrial layout. Without
    # this bridge, subsequent trial-phase actions collide with pretrial
    # barriers (e.g. the center wall at (6,7) for South-arm pretrial that
    # only lifts after the transition). The bridge appends a few F/L/R
    # actions to walk the agent back to the trigger, plus PAUSE-padding
    # if the MIN_STEPS gate isn't yet satisfied.
    if current_pos != trigger_pos:
        bridge_steps, current_pos, current_dir = _walk_to(
            trigger_pos, current_pos, current_dir
        )
        output.extend(bridge_steps)
    while len(output) < PRETRIAL_MIN_STEPS:
        output.append((ACT_PAUSE, current_pos[0], current_pos[1],
                       current_dir, 0))

    return output, current_pos, current_dir


# ── Match well visits to reward events ────────
def match_well_rewards(runs, timestamps, reward_events):
    """For each well visit run, check if it aligns with a reward event.

    Uses non-sequential matching: each reward event is independently matched
    to the well run that contains its timestamp, allowing unrewarded well
    visits to appear between rewarded ones without breaking alignment.

    Returns a set of run indices that are rewarded.
    """
    rewarded_indices = set()

    # Build index of well runs by position for fast lookup
    well_runs_by_pos = {}
    for ri, (gx, gy, start, end) in enumerate(runs):
        pos = (gx, gy)
        if pos in WELL_POSITIONS:
            run_start = timestamps[start]
            run_end = timestamps[min(end - 1, len(timestamps) - 1)]
            well_runs_by_pos.setdefault(pos, []).append((ri, run_start, run_end))

    for reward_ts, reward_pos in reward_events:
        candidates = well_runs_by_pos.get(reward_pos, [])
        for ri, run_start, run_end in candidates:
            if run_start <= reward_ts <= run_end + 1000:
                rewarded_indices.add(ri)
                break

    return rewarded_indices


# ── Main action sequence generation ───────────
def generate_actions(runs, timestamps, reward_events, pretrial_ts, rng,
                     build_pause, pause_threshold_ms,
                     consolidate_pauses=True):
    """Convert consolidated grid runs to an action sequence.

    Returns list of (action, grid_x, grid_y, direction, rewarded) tuples.
    """
    if len(runs) < 2:
        return []

    # Skip pre-pretrial runs: for acquisition sessions, discard all runs
    # before the first pretrial boundary. The session starts with the
    # pretrial block, not with pre-session exploration.
    if pretrial_ts:
        first_pt_start = pretrial_ts[0][0]
        start_ri = 0
        for ri_check in range(len(runs)):
            _, _, s, _ = runs[ri_check]
            if timestamps[s] >= first_pt_start - 2000:
                start_ri = ri_check
                break
        if start_ri > 0:
            runs = runs[start_ri:]

    # Match well visits to reward events
    rewarded_runs = match_well_rewards(runs, timestamps, reward_events)

    output = []  # list of (action, gx, gy, direction, rewarded)

    # Infer initial direction from first movement
    first_pos = (runs[0][0], runs[0][1])

    init_dir = None
    for ri in range(len(runs) - 1):
        dx = runs[ri + 1][0] - runs[ri][0]
        dy = runs[ri + 1][1] - runs[ri][1]
        if (dx, dy) in DELTA_TO_DIR:
            init_dir = DELTA_TO_DIR[(dx, dy)]
            break
        # For non-adjacent moves, use BFS first step direction
        path = find_path((runs[ri][0], runs[ri][1]),
                         (runs[ri + 1][0], runs[ri + 1][1]))
        if path and len(path) >= 2:
            pdx = path[1][0] - path[0][0]
            pdy = path[1][1] - path[0][1]
            if (pdx, pdy) in DELTA_TO_DIR:
                init_dir = DELTA_TO_DIR[(pdx, pdy)]
                break

    if init_dir is None:
        init_dir = 1  # default: South

    current_pos = first_pos
    current_dir = init_dir
    _CORNERS = {(10, 10), (2, 10), (2, 2), (10, 2)}
    at_well_exit = False  # True when at diagonal corner position after well exit

    ri = 0
    while ri < len(runs):
        gx, gy, start, end = runs[ri]
        pos = (gx, gy)

        # --- Well visit: forced override ---
        # Skip very brief well-zone appearances (< 3 rows) — these are
        # tracking artifacts where the rat briefly maps to the well zone
        # from the corner, not actual well entries.
        if pos in WELL_POSITIONS and ri > 0 and (end - start) >= 3:
            prev_pos = (runs[ri - 1][0], runs[ri - 1][1])
            # Verify we came from the correct corner
            if CORNER_TO_WELL.get(prev_pos) == pos:
                is_rewarded = ri in rewarded_runs
                corner = WELL_TO_CORNER[pos]

                # Navigate to the corner if not already there.
                # PICKUP must be emitted from the corner position.
                if current_pos != corner:
                    path = find_path(current_pos, corner)
                    if path and len(path) >= 2:
                        for step_i in range(len(path) - 1):
                            sdx = path[step_i + 1][0] - path[step_i][0]
                            sdy = path[step_i + 1][1] - path[step_i][1]
                            target_dir = DELTA_TO_DIR[(sdx, sdy)]
                            for act in turn_actions(current_dir, target_dir, rng):
                                output.append((act, current_pos[0], current_pos[1],
                                               current_dir, 0))
                                if act == ACT_LEFT:
                                    current_dir = (current_dir - 1) % 4
                                elif act == ACT_RIGHT:
                                    current_dir = (current_dir + 1) % 4
                            output.append((ACT_FORWARD, current_pos[0], current_pos[1],
                                           current_dir, 0))
                            current_pos = path[step_i + 1]

                # Generate well visit forced block
                well_acts, exit_pos, exit_dir = well_visit_actions(pos, rng)
                for act in well_acts:
                    rewarded_flag = 1 if (act == ACT_PICKUP and is_rewarded) else 0
                    output.append((act, current_pos[0], current_pos[1],
                                   current_dir, rewarded_flag))
                    # Update position/direction after each action
                    if act == ACT_PICKUP:
                        current_pos = pos
                        current_dir = WELL_ENTRY_DIR[pos]
                    elif act == ACT_FORWARD and current_pos in WELL_POSITIONS:
                        current_pos = exit_pos
                        current_dir = exit_dir
                    elif act == ACT_LEFT:
                        current_dir = (current_dir - 1) % 4
                    elif act == ACT_RIGHT:
                        current_dir = (current_dir + 1) % 4
                    # ACT_PAUSE: no state change

                at_well_exit = True
                ri += 1
                continue

        # --- Pretrial: forced override (trial sessions) ---
        # pretrial_ts is a list of (pretrial_ts, trial_start_ts) tuples
        handled_pretrial = False
        if pretrial_ts and ri < len(runs) - 1:
            run_ts = timestamps[start]
            for pt_pair in list(pretrial_ts):
                pt_start, pt_trial_start = pt_pair
                if abs(run_ts - pt_start) < 2000:  # within 2s of pretrial event
                    # Emit forced pretrial block, tracking position
                    # through each action (F,TT,F,F,TT navigates the
                    # arm dead-end and back to the trigger position).
                    DIR_TO_DELTA = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
                    pt_acts = pretrial_actions(rng)
                    for act in pt_acts:
                        output.append((act, current_pos[0], current_pos[1],
                                       current_dir, 0))
                        if act == ACT_FORWARD:
                            dx, dy = DIR_TO_DELTA[current_dir]
                            new_pos = (current_pos[0] + dx, current_pos[1] + dy)
                            if new_pos in WALKABLE_CELLS:
                                current_pos = new_pos
                        elif act == ACT_LEFT:
                            current_dir = (current_dir - 1) % 4
                        elif act == ACT_RIGHT:
                            current_dir = (current_dir + 1) % 4
                    pretrial_ts.remove(pt_pair)

                    # Skip all runs until Trial Start timestamp
                    ri += 1
                    while ri < len(runs):
                        _, _, s, _ = runs[ri]
                        if timestamps[s] >= pt_trial_start:
                            break
                        ri += 1
                    handled_pretrial = True
                    break

        if handled_pretrial:
            continue  # re-enter while loop at new ri

        # --- Pause action(s) for cell dwell ---
        if build_pause and pos not in WELL_POSITIONS:
            dwell_ms = timestamps[min(end - 1, len(timestamps) - 1)] - timestamps[start]
            for _ in range(_pause_count(dwell_ms, pause_threshold_ms,
                                        consolidate_pauses)):
                output.append((ACT_PAUSE, current_pos[0], current_pos[1],
                               current_dir, 0))

        # --- Well-exit resolution ─────────────────────────────────
        # After a well exit the env is in WELL_EXIT masked state at the
        # corner.  Only LEFT (combined move+turn), FORWARD (move), and
        # PAUSE are valid.  We scan ahead through the runs to find the
        # first real movement away from the corner, emit PAUSEs for
        # dwells, skip tracking flickers, handle well re-entries, and
        # commit LEFT or FORWARD based on the destination.
        if at_well_exit and current_pos in _WELL_EXIT_LEFT:
            left_dest = _WELL_EXIT_LEFT[current_pos]
            fwd_dest = _WELL_EXIT_FORWARD[current_pos]
            corner = current_pos
            well_pos = CORNER_TO_WELL[corner]

            # Scan ahead from current ri to find the resolution action.
            scan = ri
            resolved = False
            while scan < len(runs) and not resolved:
                sgx, sgy, ss, se = runs[scan]
                spos = (sgx, sgy)

                # Dwell at corner → emit PAUSE(s), advance
                if spos == corner:
                    if build_pause:
                        dwell_ms = timestamps[min(se - 1, len(timestamps) - 1)] - timestamps[ss]
                        for _ in range(_pause_count(dwell_ms, pause_threshold_ms,
                                                    consolidate_pauses)):
                            output.append((ACT_PAUSE, corner[0], corner[1],
                                           current_dir, 0))
                    scan += 1
                    continue

                # Brief well-zone flicker → skip silently
                if spos == well_pos and (se - ss) < 3:
                    scan += 1
                    continue

                # Real well re-entry → resolve exit, turnaround, return
                if spos == well_pos and (se - ss) >= 3:
                    fwd = _WELL_EXIT_FORWARD[corner]
                    output.append((ACT_FORWARD, corner[0], corner[1],
                                   current_dir, 0))
                    current_pos = fwd[0]
                    current_dir = fwd[1]
                    for _ in range(2):
                        output.append((ACT_RIGHT, current_pos[0], current_pos[1],
                                       current_dir, 0))
                        current_dir = (current_dir + 1) % 4
                    output.append((ACT_FORWARD, current_pos[0], current_pos[1],
                                   current_dir, 0))
                    current_pos = corner
                    at_well_exit = False
                    # Let the main loop process this well run as a visit
                    ri = scan
                    resolved = True
                    continue

                # Real movement to a non-corner, non-well position →
                # determine LEFT or FORWARD.
                if spos == left_dest[0]:
                    output.append((ACT_LEFT, corner[0], corner[1],
                                   current_dir, 0))
                    current_pos = left_dest[0]
                    current_dir = left_dest[1]
                elif spos == fwd_dest[0]:
                    output.append((ACT_FORWARD, corner[0], corner[1],
                                   current_dir, 0))
                    current_pos = fwd_dest[0]
                    current_dir = fwd_dest[1]
                else:
                    # Destination doesn't match either exit directly —
                    # choose whichever corridor is closer, then let the
                    # main loop navigate from the L/F exit position.
                    ld = abs(spos[0] - left_dest[0][0]) + abs(spos[1] - left_dest[0][1])
                    fd = abs(spos[0] - fwd_dest[0][0]) + abs(spos[1] - fwd_dest[0][1])
                    if ld <= fd:
                        output.append((ACT_LEFT, corner[0], corner[1],
                                       current_dir, 0))
                        current_pos = left_dest[0]
                        current_dir = left_dest[1]
                    else:
                        output.append((ACT_FORWARD, corner[0], corner[1],
                                       current_dir, 0))
                        current_pos = fwd_dest[0]
                        current_dir = fwd_dest[1]

                at_well_exit = False
                ri = scan  # resume from the movement run
                resolved = True

            if not resolved:
                # Ran out of runs while in well-exit state
                at_well_exit = False
                ri = scan
            # Re-enter main loop at the new ri (don't increment)
            continue

        # --- Normal move to next position ---
        if ri + 1 < len(runs):
            next_pos = (runs[ri + 1][0], runs[ri + 1][1])

            # Skip if next is a well and we're at its corner
            if next_pos in WELL_POSITIONS and CORNER_TO_WELL.get(pos) == next_pos:
                ri += 1
                continue

            # Use current_pos (not pos) for movement — they may differ
            # after well-exit resolution or BFS bridging.
            move_from = current_pos if current_pos != pos else pos
            dx = next_pos[0] - move_from[0]
            dy = next_pos[1] - move_from[1]

            if (dx, dy) in DELTA_TO_DIR:
                # Skip if target is off the walkable grid
                if next_pos not in WALKABLE_CELLS and next_pos not in WELL_POSITIONS:
                    ri += 1
                    continue
                target_dir = DELTA_TO_DIR[(dx, dy)]
                turns = turn_actions(current_dir, target_dir, rng)
                for act in turns:
                    output.append((act, current_pos[0], current_pos[1],
                                   current_dir, 0))
                    if act == ACT_LEFT:
                        current_dir = (current_dir - 1) % 4
                    elif act == ACT_RIGHT:
                        current_dir = (current_dir + 1) % 4
                output.append((ACT_FORWARD, current_pos[0], current_pos[1],
                               current_dir, 0))
                current_pos = next_pos

            elif dx == 0 and dy == 0:
                # Same position (shouldn't happen after consolidation)
                pass
            else:
                # Non-adjacent: BFS for shortest path through maze
                path = find_path(current_pos, next_pos)
                if not path:
                    # Fallback: try from pos (run position) if current_pos drifted
                    path = find_path(pos, next_pos)
                    if path:
                        current_pos = pos

                if path and len(path) >= 2:
                    for step_i in range(len(path) - 1):
                        sdx = path[step_i + 1][0] - path[step_i][0]
                        sdy = path[step_i + 1][1] - path[step_i][1]
                        target_dir = DELTA_TO_DIR[(sdx, sdy)]
                        turns = turn_actions(current_dir, target_dir, rng)
                        for act in turns:
                            output.append((act, current_pos[0], current_pos[1],
                                           current_dir, 0))
                            if act == ACT_LEFT:
                                current_dir = (current_dir - 1) % 4
                            elif act == ACT_RIGHT:
                                current_dir = (current_dir + 1) % 4
                        output.append((ACT_FORWARD, current_pos[0], current_pos[1],
                                       current_dir, 0))
                        current_pos = path[step_i + 1]

        ri += 1

    return output


# ── Pretrial start poses (from CornerMazeEnv.gen_start_pose) ─
_ARM_START_POSE = {
    0: ((6, 3), 1),   # North arm: pos, dir (facing South)
    1: ((9, 6), 2),   # East arm: pos, dir (facing West)
    2: ((6, 9), 3),   # South arm: pos, dir (facing North)
    3: ((3, 6), 0),   # West arm: pos, dir (facing East)
}

DIR_TO_DELTA = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}

# Goal index (GOAL_LOCATION_MAP ordering) → well position
_GOAL_TO_WELL = {0: (11, 1), 1: (11, 11), 2: (1, 11), 3: (1, 1)}

# Well-exit corner choices: after exiting a well, the agent is at a corner
# with two movement options (LEFT or FORWARD). Maps corner → {dest: (action, new_pos, new_dir)}.
_WELL_EXIT_FORWARD = {
    (10, 2):  ((9, 2),   2),  # West along north corridor
    (10, 10): ((10, 9),  3),  # North along east corridor
    (2, 10):  ((3, 10),  0),  # East along south corridor
    (2, 2):   ((2, 3),   1),  # South along west corridor
}
_WELL_EXIT_LEFT = {
    (10, 2):  ((10, 3),  1),  # South along east corridor
    (10, 10): ((9, 10),  2),  # West along south corridor
    (2, 10):  ((2, 9),   3),  # North along west corridor
    (2, 2):   ((3, 2),   0),  # East along north corridor
}


def _find_registered_visit(trial_well_visits, well_pos, t_start, t_end, tol_ms=200.0):
    """Look up an upstream-registered visit overlapping the given run window.

    `trial_well_visits` is the per-trial subset of get_trial_well_visits output
    (errors + rewards that MazeControl logged within this trial's phase). A
    coord-derived well-visit run is "registered" iff one of these visits is
    at the same well and its [t_entry_ms, t_exit_ms] window overlaps the
    run's [t_start, t_end] window within `tol_ms` slack.

    Returns the matching visit dict (with `is_reward` key) or None.
    """
    if not trial_well_visits:
        return None
    for v in trial_well_visits:
        if v['well_grid_pos'] != well_pos:
            continue
        if v['t_entry_ms'] <= t_end + tol_ms and v['t_exit_ms'] >= t_start - tol_ms:
            return v
    return None


def _generate_segment_actions(runs, timestamps, current_pos, current_dir,
                              rng, build_pause, pause_threshold_ms,
                              reward_events=None, stop_at_well=False,
                              trial_rewarded=True, last_trial=False,
                              blocked=None, goal_well=None,
                              after_well_exit=False, stop_at_pos=None,
                              consolidate_pauses=True,
                              trial_well_visits=None):
    """Generate navigation actions for a trial or ITI segment.

    Processes consolidated grid runs and produces turn/forward/pause actions.
    Optionally stops when the agent reaches a well corner (for trial segments)
    or a specific grid position (for ITI segments approaching an S trigger).

    Args:
        runs: Consolidated grid runs [(gx, gy, start_idx, end_idx), ...].
        timestamps: Full timestamp array for index lookups.
        current_pos: Starting (gx, gy) position.
        current_dir: Starting direction (0-3).
        rng: numpy random generator.
        build_pause: Whether to emit pause actions.
        pause_threshold_ms: Dwell threshold for pause.
        reward_events: List of (t_ms, well_pos) for reward matching (exposure).
        stop_at_well: If True, stop and emit well_visit when reaching a well corner.
        trial_rewarded: If stop_at_well, whether the well visit is rewarded.

    Returns:
        (output, current_pos, current_dir, well_rewarded) where output is a list
        of (action, gx, gy, direction, rewarded) tuples and well_rewarded indicates
        if the well visit was rewarded (None if no well visit).
    """
    output = []
    well_rewarded = None

    # Remap runs at blocked barrier positions to nearest walkable neighbor.
    # The agent was pushing against the barrier, so the time at the blocked
    # position becomes dwell time at the adjacent valid cell.
    if blocked:
        remapped = []
        for gx, gy, s, e in runs:
            if (gx, gy) in blocked:
                # Find nearest walkable non-blocked neighbor
                best = None
                best_dist = float('inf')
                for c in WALKABLE_CELLS:
                    if c not in blocked:
                        d = abs(c[0] - gx) + abs(c[1] - gy)
                        if d < best_dist:
                            best_dist = d
                            best = c
                if best is not None:
                    remapped.append((best[0], best[1], s, e))
            else:
                remapped.append((gx, gy, s, e))
        # Merge consecutive runs at the same position after remapping
        merged = []
        for r in remapped:
            if merged and merged[-1][0] == r[0] and merged[-1][1] == r[1]:
                merged[-1] = (r[0], r[1], merged[-1][2], r[3])
            else:
                merged.append(r)
        runs = merged

    _CORNERS = {(10, 10), (2, 10), (2, 2), (10, 2)}
    at_well_exit = after_well_exit  # True when at diagonal corner after well exit

    # Build reward lookup if needed (for exposure-style matching)
    rewarded_runs = set()
    if reward_events and not stop_at_well:
        rewarded_runs = match_well_rewards(runs, timestamps, reward_events)

    ri = 0
    while ri < len(runs):
        # Stop if we've reached the target position (e.g. S trigger for ITI)
        if stop_at_pos is not None and current_pos == stop_at_pos:
            break

        gx, gy, start, end = runs[ri]
        pos = (gx, gy)

        # Well visit: emit visit block when the rat actually enters the well
        # (next run is at the well position). Gating depends on caller:
        #   - stop_at_well=True (trial-phase): gate on trial_well_visits
        #     (MazeControl-logged visits) so the action stream matches the
        #     experimental record and the env's trial counter doesn't
        #     over-count goal-well entries.
        #   - stop_at_well=False (ITI): apply our own 250 ms dwell threshold.
        #     ITI well visits are real rat behavior (quick reward-checks);
        #     filter sub-threshold tracking flicker, emit the rest. Env
        #     doesn't fire trial transitions on these because ITI layouts
        #     have no goal-marked wells (rewarded=0 throughout).
        if (pos in CORNER_TO_WELL and ri + 1 < len(runs)
                and (runs[ri + 1][0], runs[ri + 1][1]) == CORNER_TO_WELL[pos]):
            well_pos = CORNER_TO_WELL[pos]
            is_goal = (goal_well is None or well_pos == goal_well)
            well_run = runs[ri + 1]
            w_start = float(timestamps[well_run[2]])
            w_end = float(timestamps[min(well_run[3] - 1, len(timestamps) - 1)])

            if stop_at_well:
                # Trial-phase path
                if trial_well_visits is not None:
                    matched = _find_registered_visit(
                        trial_well_visits, well_pos, w_start, w_end
                    )
                    if matched is None:
                        ri += 2
                        continue
                    is_rewarded = bool(matched['is_reward'])
                else:
                    is_rewarded = trial_rewarded and is_goal
            else:
                # ITI path: dwell-threshold gate (matches MazeControl's
                # reward-dwell criterion). Sub-threshold dips are tracking
                # flicker / inertia, not deliberate well checks.
                ITI_WELL_DWELL_MS = 250
                if (w_end - w_start) < ITI_WELL_DWELL_MS:
                    ri += 2
                    continue
                # ITI well visits never count toward trial reward.
                is_rewarded = False

            if last_trial and is_goal:
                # Final trial, correct well: just PICKUP (env terminates)
                output.append((ACT_PICKUP, current_pos[0], current_pos[1],
                               current_dir, 1 if is_rewarded else 0))
                current_pos = well_pos
                current_dir = WELL_ENTRY_DIR[well_pos]
            else:
                well_acts, exit_pos, exit_dir = well_visit_actions(well_pos, rng)
                for act in well_acts:
                    rewarded_flag = 1 if (act == ACT_PICKUP and is_rewarded) else 0
                    output.append((act, current_pos[0], current_pos[1],
                                   current_dir, rewarded_flag))
                    if act == ACT_PICKUP:
                        current_pos = well_pos
                        current_dir = WELL_ENTRY_DIR[well_pos]
                    elif act == ACT_FORWARD and current_pos in WELL_POSITIONS:
                        current_pos = exit_pos
                        current_dir = exit_dir
                    elif act == ACT_LEFT:
                        current_dir = (current_dir - 1) % 4
                    elif act == ACT_RIGHT:
                        current_dir = (current_dir + 1) % 4

            # Stop the segment only on a trial-phase goal-well visit.
            # ITI calls pass stop_at_well=False (and goal_well=None, which
            # would otherwise make is_goal=True trivially); treat ITI
            # visits as transient checks and keep navigating.
            if stop_at_well and is_goal:
                well_rewarded = is_rewarded
                break  # stop at goal well
            else:
                # Wrong well (trial) or ITI check — continue navigating
                # from well-exit state. Skip past the well run.
                at_well_exit = True
                ri += 2
                continue

        # Pause action(s) for cell dwell
        if build_pause and pos not in WELL_POSITIONS:
            dwell_ms = timestamps[min(end - 1, len(timestamps) - 1)] - timestamps[start]
            for _ in range(_pause_count(dwell_ms, pause_threshold_ms,
                                        consolidate_pauses)):
                output.append((ACT_PAUSE, current_pos[0], current_pos[1],
                               current_dir, 0))

        # Move to next position
        if ri + 1 < len(runs):
            next_pos = (runs[ri + 1][0], runs[ri + 1][1])

            # If next is a well and we're at its corner, handle on next iteration
            if next_pos in WELL_POSITIONS and CORNER_TO_WELL.get(pos) == next_pos:
                ri += 1
                continue

            # Well-exit corner: choose LEFT or FORWARD based on destination.
            # Destination is next_pos when dwelling at corner (pos==current_pos),
            # or pos itself when the rat left the corner directly.
            if at_well_exit and current_pos in _WELL_EXIT_LEFT:
                dest = next_pos if pos == current_pos else pos
                left_dest = _WELL_EXIT_LEFT[current_pos]
                fwd_dest = _WELL_EXIT_FORWARD[current_pos]
                if dest == left_dest[0]:
                    output.append((ACT_LEFT, current_pos[0], current_pos[1],
                                   current_dir, 0))
                    current_pos = left_dest[0]
                    current_dir = left_dest[1]
                    at_well_exit = False
                    ri += 1
                    continue
                elif dest == fwd_dest[0]:
                    output.append((ACT_FORWARD, current_pos[0], current_pos[1],
                                   current_dir, 0))
                    current_pos = fwd_dest[0]
                    current_dir = fwd_dest[1]
                    at_well_exit = False
                    ri += 1
                    continue
                else:
                    # Dest doesn't directly match L_dest or F_dest, but if
                    # the BFS path from corner to dest passes through one of
                    # them as its first hop, emit the corner-trick action
                    # (env collapses turn-and-step into one step) and fall
                    # through to the regular BFS bridge below to walk the
                    # remaining cells from the new position. This handles
                    # tracker phantom jumps that skip past the corner-trick
                    # destination, e.g. (10,2)→(10,6) skipping (10,3).
                    trick_path = find_path(current_pos, dest, blocked=blocked)
                    if trick_path and len(trick_path) >= 2:
                        if trick_path[1] == left_dest[0]:
                            output.append((ACT_LEFT, current_pos[0], current_pos[1],
                                           current_dir, 0))
                            current_pos = left_dest[0]
                            current_dir = left_dest[1]
                        elif trick_path[1] == fwd_dest[0]:
                            output.append((ACT_FORWARD, current_pos[0], current_pos[1],
                                           current_dir, 0))
                            current_pos = fwd_dest[0]
                            current_dir = fwd_dest[1]
                    at_well_exit = False

            # Use current_pos (not pos) for movement — they may differ
            # after well-exit resolution or BFS bridging.
            move_from = current_pos if current_pos != pos else pos
            dx = next_pos[0] - move_from[0]
            dy = next_pos[1] - move_from[1]

            if (dx, dy) in DELTA_TO_DIR:
                # Skip if target is off the walkable grid or blocked by barrier
                if next_pos not in WALKABLE_CELLS and next_pos not in WELL_POSITIONS:
                    ri += 1
                    continue
                if blocked and next_pos in blocked:
                    ri += 1
                    continue
                target_dir = DELTA_TO_DIR[(dx, dy)]
                turns = turn_actions(current_dir, target_dir, rng)
                for act in turns:
                    output.append((act, current_pos[0], current_pos[1],
                                   current_dir, 0))
                    if act == ACT_LEFT:
                        current_dir = (current_dir - 1) % 4
                    elif act == ACT_RIGHT:
                        current_dir = (current_dir + 1) % 4
                output.append((ACT_FORWARD, current_pos[0], current_pos[1],
                               current_dir, 0))
                current_pos = next_pos
                if stop_at_pos is not None and current_pos == stop_at_pos:
                    ri = len(runs)  # force outer loop exit
            elif dx == 0 and dy == 0:
                pass
            else:
                # Non-adjacent: BFS (barrier-aware if blocked set provided)
                path = find_path(current_pos, next_pos, blocked=blocked)
                if not path:
                    path = find_path(pos, next_pos, blocked=blocked)
                    if path:
                        current_pos = pos
                if path and len(path) >= 2:
                    for step_i in range(len(path) - 1):
                        sdx = path[step_i + 1][0] - path[step_i][0]
                        sdy = path[step_i + 1][1] - path[step_i][1]
                        target_dir = DELTA_TO_DIR[(sdx, sdy)]
                        turns = turn_actions(current_dir, target_dir, rng)
                        for act in turns:
                            output.append((act, current_pos[0], current_pos[1],
                                           current_dir, 0))
                            if act == ACT_LEFT:
                                current_dir = (current_dir - 1) % 4
                            elif act == ACT_RIGHT:
                                current_dir = (current_dir + 1) % 4
                        output.append((ACT_FORWARD, current_pos[0], current_pos[1],
                                       current_dir, 0))
                        current_pos = path[step_i + 1]
                        if stop_at_pos is not None and current_pos == stop_at_pos:
                            ri = len(runs)  # force outer loop exit
                            break

        ri += 1

    return output, current_pos, current_dir, well_rewarded, ri


def generate_acquisition_actions(
    phase_grid_df,
    trial_configs: list,
    reward_events: list,
    rng,
    build_pause: bool,
    pause_threshold_ms: int,
    max_gap: int = 2,
    trial_barrier_sets: list | None = None,
    iti_barrier_sets: list | None = None,
    iti_sim_configs: list | None = None,
    use_real_pretrial: bool = False,
    consolidate_pauses: bool = True,
    registered_well_visits: list | None = None,
) -> list:
    """Generate actions for an acquisition session using phase-joined coordinates.

    Processes each trial cycle: pretrial → real trial → real ITI.
    The pretrial block is either synthetic (default) or real tracking data
    (use_real_pretrial=True). The trial segment uses real tracking data from
    the trigger to the goal well, and the ITI segment uses real tracking data
    navigating to the next arm.

    Args:
        phase_grid_df: DataFrame with (t_ms, x, y, zone, grid_x, grid_y, phase, trial_number).
        trial_configs: List of [start_arm, cue, goal, tag] per trial.
        reward_events: List of (t_ms, well_pos) for reward detection.
        rng: numpy random generator.
        build_pause: Whether to include pause actions.
        pause_threshold_ms: Dwell threshold for pause.
        max_gap: Flicker sensitivity for consolidation.

    Returns:
        List of (action, grid_x, grid_y, direction, rewarded) tuples.
    """
    output = []
    trial_numbers = sorted(phase_grid_df['trial_number'].unique())

    # Build per-trial reward lookup: check if any reward event falls
    # within each trial's time range. Used only as a legacy fallback
    # when registered_well_visits is None; the gated path uses the
    # per-visit is_reward flag instead.
    trial_rewarded_map = {}
    for trial_num in trial_numbers:
        trial_df = phase_grid_df[
            (phase_grid_df['trial_number'] == trial_num) &
            (phase_grid_df['phase'] == 'trial')
        ]
        if len(trial_df) > 0:
            t_start = float(trial_df['t_ms'].iloc[0])
            t_end = float(trial_df['t_ms'].iloc[-1])
            trial_rewarded_map[trial_num] = any(
                t_start <= t <= t_end + 1000 for t, _ in reward_events
            )
        else:
            trial_rewarded_map[trial_num] = False

    # Bucket upstream-registered well visits by trial_number so each trial
    # segment only sees its own visits during the gating overlap check.
    trial_visits_map: dict[int, list[dict]] = {}
    if registered_well_visits is not None:
        for v in registered_well_visits:
            trial_visits_map.setdefault(v['trial_number'], []).append(v)

    for trial_idx, trial_num in enumerate(trial_numbers):
        if trial_idx >= len(trial_configs):
            break

        arm = trial_configs[trial_idx][0]
        start_pos, start_dir = _ARM_START_POSE[arm]

        # ── Pretrial block ────────────────────────────
        if use_real_pretrial:
            # Real pretrial from tracking data
            pretrial_df = phase_grid_df[
                (phase_grid_df['trial_number'] == trial_num) &
                (phase_grid_df['phase'] == 'pretrial')
            ]
            if len(pretrial_df) >= 2:
                pt_output, current_pos, current_dir = generate_real_pretrial_actions(
                    pretrial_df, arm, rng, build_pause, pause_threshold_ms,
                    consolidate_pauses)
                output.extend(pt_output)
            else:
                # Insufficient pretrial data — fall back to synthetic
                pt_acts = pretrial_actions(rng)
                current_pos = start_pos
                current_dir = start_dir
                for act in pt_acts:
                    output.append((act, current_pos[0], current_pos[1],
                                   current_dir, 0))
                    if act == ACT_FORWARD:
                        dx, dy = DIR_TO_DELTA[current_dir]
                        new_pos = (current_pos[0] + dx, current_pos[1] + dy)
                        if new_pos in WALKABLE_CELLS:
                            current_pos = new_pos
                    elif act == ACT_LEFT:
                        current_dir = (current_dir - 1) % 4
                    elif act == ACT_RIGHT:
                        current_dir = (current_dir + 1) % 4
        else:
            # Synthetic pretrial: F, TT, F, F, TT
            pt_acts = pretrial_actions(rng)
            current_pos = start_pos
            current_dir = start_dir
            for act in pt_acts:
                output.append((act, current_pos[0], current_pos[1],
                               current_dir, 0))
                if act == ACT_FORWARD:
                    dx, dy = DIR_TO_DELTA[current_dir]
                    new_pos = (current_pos[0] + dx, current_pos[1] + dy)
                    if new_pos in WALKABLE_CELLS:
                        current_pos = new_pos
                elif act == ACT_LEFT:
                    current_dir = (current_dir - 1) % 4
                elif act == ACT_RIGHT:
                    current_dir = (current_dir + 1) % 4

        # ── Trial segment (real tracking data) ────────
        trial_df = phase_grid_df[
            (phase_grid_df['trial_number'] == trial_num) &
            (phase_grid_df['phase'] == 'trial')
        ]

        if len(trial_df) > 0:
            # Filter off-grid points
            trial_df = trial_df[
                ~((trial_df['grid_x'] == 0) & (trial_df['grid_y'] == 0))
            ].reset_index(drop=True)

            if len(trial_df) > 0:
                # Remap barrier positions to nearest walkable neighbor, then
                # remap any remaining frames that landed in barrier-isolated
                # pockets (walkable cells unreachable from `current_pos` given
                # the active barrier set).
                if trial_barrier_sets and trial_idx < len(trial_barrier_sets):
                    barriers = trial_barrier_sets[trial_idx]
                    gx_arr = trial_df['grid_x'].values.copy()
                    gy_arr = trial_df['grid_y'].values.copy()
                    _remap_to_reachable_one_pass(gx_arr, gy_arr, barriers, current_pos)
                    trial_df = trial_df.copy()
                    trial_df['grid_x'] = gx_arr
                    trial_df['grid_y'] = gy_arr

                grid_x = trial_df['grid_x'].values
                grid_y = trial_df['grid_y'].values
                timestamps = trial_df['t_ms'].values.astype(float)
                runs = consolidate_grid(grid_x, grid_y, max_gap=max_gap)
                runs = filter_phantom_jumps(runs)

                if len(runs) > 0:
                    first_run_pos = (int(runs[0][0]), int(runs[0][1]))

                    # Bridge from pretrial end to first tracked trial position
                    trial_blocked = trial_barrier_sets[trial_idx] if trial_barrier_sets else None
                    if first_run_pos != current_pos:
                        path = find_path(current_pos, first_run_pos, blocked=trial_blocked)
                        if path and len(path) >= 2:
                            for step_i in range(len(path) - 1):
                                sdx = path[step_i + 1][0] - path[step_i][0]
                                sdy = path[step_i + 1][1] - path[step_i][1]
                                target_dir = DELTA_TO_DIR[(sdx, sdy)]
                                turns = turn_actions(current_dir, target_dir, rng)
                                for act in turns:
                                    output.append((act, current_pos[0], current_pos[1],
                                                   current_dir, 0))
                                    if act == ACT_LEFT:
                                        current_dir = (current_dir - 1) % 4
                                    elif act == ACT_RIGHT:
                                        current_dir = (current_dir + 1) % 4
                                output.append((ACT_FORWARD, current_pos[0], current_pos[1],
                                               current_dir, 0))
                                current_pos = path[step_i + 1]

                    # Infer direction from first movement in trial segment
                    seg_dir = current_dir
                    for ri in range(min(5, len(runs) - 1)):
                        dx = runs[ri + 1][0] - runs[ri][0]
                        dy = runs[ri + 1][1] - runs[ri][1]
                        if (dx, dy) in DELTA_TO_DIR:
                            seg_dir = DELTA_TO_DIR[(dx, dy)]
                            break
                        path = find_path((runs[ri][0], runs[ri][1]),
                                         (runs[ri + 1][0], runs[ri + 1][1]),
                                         blocked=trial_blocked)
                        if path and len(path) >= 2:
                            pdx = path[1][0] - path[0][0]
                            pdy = path[1][1] - path[0][1]
                            if (pdx, pdy) in DELTA_TO_DIR:
                                seg_dir = DELTA_TO_DIR[(pdx, pdy)]
                                break

                    # Bridge direction from pretrial end to trial movement
                    if seg_dir != current_dir:
                        dir_turns = turn_actions(current_dir, seg_dir, rng)
                        for act in dir_turns:
                            output.append((act, current_pos[0], current_pos[1],
                                           current_dir, 0))
                            if act == ACT_LEFT:
                                current_dir = (current_dir - 1) % 4
                            elif act == ACT_RIGHT:
                                current_dir = (current_dir + 1) % 4

                    current_pos = first_run_pos

                    is_last = (trial_idx == len(trial_numbers) - 1)
                    trial_blocked = trial_barrier_sets[trial_idx] if trial_barrier_sets else None
                    goal_idx = trial_configs[trial_idx][2]
                    seg_out, current_pos, current_dir, well_result, _ = _generate_segment_actions(
                        runs, timestamps, current_pos, current_dir, rng,
                        build_pause, pause_threshold_ms,
                        stop_at_well=True,
                        trial_rewarded=trial_rewarded_map.get(trial_num, False),
                        last_trial=is_last,
                        blocked=trial_blocked,
                        goal_well=_GOAL_TO_WELL.get(goal_idx),
                        consolidate_pauses=consolidate_pauses,
                        trial_well_visits=trial_visits_map.get(trial_num) if registered_well_visits is not None else None,
                    )
                    output.extend(seg_out)
                    trial_reached_goal = well_result is not None
                else:
                    trial_reached_goal = False
            else:
                trial_reached_goal = False
        else:
            trial_reached_goal = False

        # ── ITI segment (real tracking data) ──────────
        iti_df = phase_grid_df[
            (phase_grid_df['trial_number'] == trial_num) &
            (phase_grid_df['phase'] == 'iti')
        ]

        if len(iti_df) > 0:
            iti_df = iti_df[
                ~((iti_df['grid_x'] == 0) & (iti_df['grid_y'] == 0))
            ].reset_index(drop=True)

            if len(iti_df) > 0:
                # Apply barrier+connectivity remap using the env's *currently
                # active* barrier set per frame. The env flips
                # `_iti_config_idx` when the agent steps on an A/B trigger,
                # so the active barrier set is not constant across the ITI.
                # Co-simulate the layout swap along the rat's mapped path
                # and remap each frame against the barriers active at that
                # frame. Falls back to the static initial set if sim configs
                # weren't supplied.
                gx_arr = iti_df['grid_x'].values.copy()
                gy_arr = iti_df['grid_y'].values.copy()
                used_per_frame = False
                if (iti_sim_configs and trial_idx < len(iti_sim_configs)
                        and iti_sim_configs[trial_idx] is not None):
                    from corner_maze_rl.env.constants import (
                        BARRIER_LOCATIONS as _SIM_BL,
                        TRIGGER_LOCATIONS as _SIM_TL,
                    )
                    sim_iti_configs, sim_initial_idx = iti_sim_configs[trial_idx]
                    blocked_per_frame = _simulate_iti_barriers_per_frame(
                        gx_arr, gy_arr, sim_iti_configs, sim_initial_idx,
                        _SIM_BL, _SIM_TL,
                    )
                    _remap_with_per_frame_barriers(
                        gx_arr, gy_arr, blocked_per_frame, current_pos
                    )
                    used_per_frame = True
                elif iti_barrier_sets and trial_idx < len(iti_barrier_sets):
                    iti_barriers = iti_barrier_sets[trial_idx]
                    if iti_barriers:
                        _remap_to_reachable_one_pass(gx_arr, gy_arr, iti_barriers, current_pos)
                if used_per_frame or (iti_barrier_sets and trial_idx < len(iti_barrier_sets)
                                      and iti_barrier_sets[trial_idx]):
                    iti_df = iti_df.copy()
                    iti_df['grid_x'] = gx_arr
                    iti_df['grid_y'] = gy_arr

                grid_x = iti_df['grid_x'].values
                grid_y = iti_df['grid_y'].values
                timestamps = iti_df['t_ms'].values.astype(float)
                runs = consolidate_grid(grid_x, grid_y, max_gap=max_gap)
                runs = filter_phantom_jumps(runs)

                # ── Trial continuation in ITI data ──────────
                # When the trial segment didn't reach the goal well (phase
                # boundary mismatch — the real experiment ended the trial
                # phase before the rat found the correct well), the ITI
                # tracking data contains the remaining trial navigation.
                # Process it with trial barriers until the goal is reached,
                # then fall through to normal ITI processing for the rest.
                if not trial_reached_goal and runs:
                    trial_blocked = trial_barrier_sets[trial_idx] if trial_barrier_sets else None
                    goal_idx = trial_configs[trial_idx][2]
                    is_last = (trial_idx == len(trial_numbers) - 1)
                    goal_well_pos = _GOAL_TO_WELL.get(goal_idx)

                    # Goal-well-as-first-run synthesis. When the ITI runs start
                    # *at* the goal well (the rat's tracker placed the rat in
                    # the well at the start of ITI-phase data, often because
                    # the trial phase boundary closed at the well visit), the
                    # segment generator's well-visit detection won't fire — it
                    # requires a (corner, well) run pair, but here there's no
                    # preceding corner run. Synthesize the last-mile path from
                    # current_pos to the goal corner, emit the well-visit
                    # block (PICKUP+RR/LL+F), and slice off the goal-well run
                    # so the rest of the runs can be processed as normal ITI.
                    if (goal_well_pos and runs
                            and (runs[0][0], runs[0][1]) == goal_well_pos):
                        goal_corner = WELL_TO_CORNER.get(goal_well_pos)
                        # No gating here: the trial-continuation-in-ITI path
                        # handles miss trials where upstream's trial phase
                        # ended before the rat reached the goal but the rat
                        # later reached it during the ITI window (all 40 such
                        # cases in the in-scope dataset have substantial
                        # ITI-window goal-well dwell — see investigation log).
                        # Upstream's trial_well_visits is structurally blind
                        # to ITI-window visits, so gating against it would
                        # incorrectly suppress these legitimate completions.
                        is_rewarded = trial_rewarded_map.get(trial_num, False)
                        if goal_corner and current_pos != goal_corner:
                            path = find_path(current_pos, goal_corner, blocked=trial_blocked)
                            if path and len(path) >= 2:
                                for step_i in range(len(path) - 1):
                                    sdx = path[step_i + 1][0] - path[step_i][0]
                                    sdy = path[step_i + 1][1] - path[step_i][1]
                                    target_dir = DELTA_TO_DIR[(sdx, sdy)]
                                    for act in turn_actions(current_dir, target_dir, rng):
                                        output.append((act, current_pos[0], current_pos[1],
                                                       current_dir, 0))
                                        if act == ACT_LEFT:
                                            current_dir = (current_dir - 1) % 4
                                        elif act == ACT_RIGHT:
                                            current_dir = (current_dir + 1) % 4
                                    output.append((ACT_FORWARD, current_pos[0], current_pos[1],
                                                   current_dir, 0))
                                    current_pos = path[step_i + 1]
                        if current_pos == goal_corner:
                            well_acts, exit_pos, exit_dir = well_visit_actions(goal_well_pos, rng)
                            for act in well_acts:
                                rew_flag = 1 if (act == ACT_PICKUP and is_rewarded) else 0
                                output.append((act, current_pos[0], current_pos[1],
                                               current_dir, rew_flag))
                                if act == ACT_PICKUP:
                                    current_pos = goal_well_pos
                                    current_dir = WELL_ENTRY_DIR[goal_well_pos]
                                elif act == ACT_FORWARD and current_pos in WELL_POSITIONS:
                                    current_pos = exit_pos
                                    current_dir = exit_dir
                                elif act == ACT_LEFT:
                                    current_dir = (current_dir - 1) % 4
                                elif act == ACT_RIGHT:
                                    current_dir = (current_dir + 1) % 4
                            trial_reached_goal = True
                            runs = runs[1:]  # drop the goal-well run

                if not trial_reached_goal and runs:
                    trial_blocked = trial_barrier_sets[trial_idx] if trial_barrier_sets else None
                    goal_idx = trial_configs[trial_idx][2]
                    is_last = (trial_idx == len(trial_numbers) - 1)

                    # Bridge from trial end to first ITI position
                    first_run_pos = (int(runs[0][0]), int(runs[0][1]))
                    if first_run_pos != current_pos:
                        path = find_path(current_pos, first_run_pos, blocked=trial_blocked)
                        if path and len(path) >= 2:
                            for step_i in range(len(path) - 1):
                                sdx = path[step_i + 1][0] - path[step_i][0]
                                sdy = path[step_i + 1][1] - path[step_i][1]
                                target_dir = DELTA_TO_DIR[(sdx, sdy)]
                                for act in turn_actions(current_dir, target_dir, rng):
                                    output.append((act, current_pos[0], current_pos[1],
                                                   current_dir, 0))
                                    if act == ACT_LEFT:
                                        current_dir = (current_dir - 1) % 4
                                    elif act == ACT_RIGHT:
                                        current_dir = (current_dir + 1) % 4
                                output.append((ACT_FORWARD, current_pos[0], current_pos[1],
                                               current_dir, 0))
                                current_pos = path[step_i + 1]

                    # No trial_well_visits here: this is the ITI-fallback
                    # path (upstream's trial-phase boundary closed before
                    # the rat reached the goal, but the rat completed the
                    # trial within the ITI window). trial_well_visits is
                    # blind to ITI-window events, so emit per coord-derived
                    # runs and let the env's goal-well detection do its job.
                    seg_out, current_pos, current_dir, well_result, consumed_ri = _generate_segment_actions(
                        runs, timestamps, current_pos, current_dir, rng,
                        build_pause, pause_threshold_ms,
                        stop_at_well=True,
                        trial_rewarded=trial_rewarded_map.get(trial_num, False),
                        last_trial=is_last,
                        blocked=trial_blocked,
                        goal_well=_GOAL_TO_WELL.get(goal_idx),
                        consolidate_pauses=consolidate_pauses,
                    )
                    output.extend(seg_out)
                    # Slice off consumed runs so the normal ITI path below
                    # only processes the remaining post-goal-well runs.
                    runs = runs[consumed_ri:]

                # Drop *leading* well-position runs (the rat lingering at
                # the trial-end goal well at the very start of ITI before
                # exiting). The trial-end well_visit_actions block already
                # exited the agent to the corner; those lingering frames
                # would re-trigger emission and conflict with the corner-
                # trick / bridge logic below. *Mid-ITI* well-position runs
                # are kept — they're genuine quick-check visits at corners,
                # which the segment generator emits as PICKUP+exit blocks
                # (with a 250 ms dwell gate). Env doesn't fire trial
                # transitions on these (ITI layouts have no rewarded
                # wells), so they're cosmetic for env state but match
                # real rat behavior in the action stream.
                # Remap trial barrier overshoots to nearest walkable neighbor
                # when the trial didn't reach the goal (env still has trial barriers).
                barrier_filter = set()
                if not trial_reached_goal and trial_barrier_sets:
                    barrier_filter = trial_barrier_sets[trial_idx]
                remapped = []
                seen_non_well = False
                for gx, gy, s, e in runs:
                    if (gx, gy) in WELL_POSITIONS:
                        if not seen_non_well:
                            # Leading well — rat hadn't left the goal well
                            # yet. Drop.
                            continue
                        # Mid-ITI well: keep for emission.
                        remapped.append((gx, gy, s, e))
                        continue
                    seen_non_well = True
                    if (gx, gy) in barrier_filter:
                        # Remap to nearest walkable non-blocked neighbor
                        best = None
                        best_dist = float('inf')
                        for c in WALKABLE_CELLS:
                            if c not in barrier_filter and c not in WELL_POSITIONS:
                                d = abs(c[0] - gx) + abs(c[1] - gy)
                                if d < best_dist:
                                    best_dist = d
                                    best = c
                        if best is not None:
                            remapped.append((best[0], best[1], s, e))
                    else:
                        remapped.append((gx, gy, s, e))
                runs = []
                for r in remapped:
                    if runs and runs[-1][0] == r[0] and runs[-1][1] == r[1]:
                        runs[-1] = (r[0], r[1], runs[-1][2], r[3])  # merge
                    else:
                        runs.append(r)

                if len(runs) > 0:
                    first_run_pos = (int(runs[0][0]), int(runs[0][1]))

                    iti_blocked = iti_barrier_sets[trial_idx] if iti_barrier_sets else None

                    # Well-exit corner L/F: when the rat just exited a well
                    # and the first ITI run is the L-trick or F-trick
                    # destination, emit a single corner-trick action. The
                    # env collapses L (or F) at a CORNER+WELL_EXIT_POSE into
                    # a combined turn-and-step, so a regular bridge that
                    # emits L+F here would double-count the move and create
                    # a phantom (corner, post-L-dir) frame. Run BEFORE the
                    # bridge so the corner-trick wins when the tracker
                    # skipped the corner frame entirely (e.g. went directly
                    # from well to next corridor cell).
                    well_exit_handled = False
                    if current_pos in _WELL_EXIT_LEFT and first_run_pos != current_pos:
                        left_dest = _WELL_EXIT_LEFT[current_pos]
                        fwd_dest = _WELL_EXIT_FORWARD[current_pos]
                        if first_run_pos == left_dest[0]:
                            output.append((ACT_LEFT, current_pos[0], current_pos[1],
                                           current_dir, 0))
                            current_pos = left_dest[0]
                            current_dir = left_dest[1]
                            well_exit_handled = True
                        elif first_run_pos == fwd_dest[0]:
                            output.append((ACT_FORWARD, current_pos[0], current_pos[1],
                                           current_dir, 0))
                            current_pos = fwd_dest[0]
                            current_dir = fwd_dest[1]
                            well_exit_handled = True
                        else:
                            # first_run_pos is further than the corner-trick
                            # destination, but if the BFS path's first hop is
                            # left_dest or fwd_dest, emit the corner-trick
                            # action and let the bridge below walk the rest.
                            tp = find_path(current_pos, first_run_pos, blocked=iti_blocked)
                            if tp and len(tp) >= 2:
                                if tp[1] == left_dest[0]:
                                    output.append((ACT_LEFT, current_pos[0], current_pos[1],
                                                   current_dir, 0))
                                    current_pos = left_dest[0]
                                    current_dir = left_dest[1]
                                elif tp[1] == fwd_dest[0]:
                                    output.append((ACT_FORWARD, current_pos[0], current_pos[1],
                                                   current_dir, 0))
                                    current_pos = fwd_dest[0]
                                    current_dir = fwd_dest[1]

                    bridged_to_first = False
                    if first_run_pos != current_pos:
                        path = find_path(current_pos, first_run_pos, blocked=iti_blocked)
                        if path and len(path) >= 2:
                            for step_i in range(len(path) - 1):
                                sdx = path[step_i + 1][0] - path[step_i][0]
                                sdy = path[step_i + 1][1] - path[step_i][1]
                                target_dir = DELTA_TO_DIR[(sdx, sdy)]
                                turns = turn_actions(current_dir, target_dir, rng)
                                for act in turns:
                                    output.append((act, current_pos[0], current_pos[1],
                                                   current_dir, 0))
                                    if act == ACT_LEFT:
                                        current_dir = (current_dir - 1) % 4
                                    elif act == ACT_RIGHT:
                                        current_dir = (current_dir + 1) % 4
                                output.append((ACT_FORWARD, current_pos[0], current_pos[1],
                                               current_dir, 0))
                                current_pos = path[step_i + 1]
                            bridged_to_first = True

                    # Infer direction from first movement in ITI segment
                    seg_dir = current_dir
                    for ri in range(min(5, len(runs) - 1)):
                        dx = runs[ri + 1][0] - runs[ri][0]
                        dy = runs[ri + 1][1] - runs[ri][1]
                        if (dx, dy) in DELTA_TO_DIR:
                            seg_dir = DELTA_TO_DIR[(dx, dy)]
                            break
                        path = find_path((runs[ri][0], runs[ri][1]),
                                         (runs[ri + 1][0], runs[ri + 1][1]),
                                         blocked=iti_blocked)
                        if path and len(path) >= 2:
                            pdx = path[1][0] - path[0][0]
                            pdy = path[1][1] - path[0][1]
                            if (pdx, pdy) in DELTA_TO_DIR:
                                seg_dir = DELTA_TO_DIR[(pdx, pdy)]
                                break

                    if not well_exit_handled and not bridged_to_first:
                        current_pos = first_run_pos

                    # Pass well-exit state if still at corner (L/F choice pending).
                    # Only set if the agent actually just exited a well
                    # (not if it merely navigated to a corner via bridge).
                    iti_at_well_exit = (not well_exit_handled
                                        and not bridged_to_first
                                        and current_pos in _WELL_EXIT_LEFT)
                    # Stop ITI at the S trigger position (= next arm's start
                    # pose). When the agent reaches this position, the env
                    # fires the S trigger and transitions to PRETRIAL, changing
                    # the barrier layout. Any actions past this point would
                    # face the wrong barriers.
                    iti_stop = None
                    if trial_idx + 1 < len(trial_configs):
                        next_arm = trial_configs[trial_idx + 1][0]
                        iti_stop = _ARM_START_POSE[next_arm][0]
                    seg_out, current_pos, current_dir, _, _ = _generate_segment_actions(
                        runs, timestamps, current_pos, current_dir, rng,
                        build_pause, pause_threshold_ms,
                        stop_at_well=False,
                        blocked=iti_blocked,
                        after_well_exit=iti_at_well_exit,
                        stop_at_pos=iti_stop,
                        consolidate_pauses=consolidate_pauses,
                    )
                    output.extend(seg_out)

                    # Bridge to the S trigger if ITI ended before reaching it.
                    # Try with ITI barriers; if no path, try with the
                    # intersection (barriers in BOTH ITI and trial —
                    # positions that are always blocked regardless of
                    # config). The env's A/B triggers dynamically open
                    # config-specific barriers as the agent walks through.
                    if iti_stop and current_pos != iti_stop:
                        path = find_path(current_pos, iti_stop, blocked=iti_blocked)
                        if not path and trial_barrier_sets:
                            always_blocked = (iti_blocked & trial_barrier_sets[trial_idx]
                                              if iti_blocked else set())
                            path = find_path(current_pos, iti_stop, blocked=always_blocked)
                        if not path:
                            path = find_path(current_pos, iti_stop)
                        if path and len(path) >= 2:
                            for step_i in range(len(path) - 1):
                                sdx = path[step_i + 1][0] - path[step_i][0]
                                sdy = path[step_i + 1][1] - path[step_i][1]
                                target_dir = DELTA_TO_DIR[(sdx, sdy)]
                                for act in turn_actions(current_dir, target_dir, rng):
                                    output.append((act, current_pos[0], current_pos[1],
                                                   current_dir, 0))
                                    if act == ACT_LEFT:
                                        current_dir = (current_dir - 1) % 4
                                    elif act == ACT_RIGHT:
                                        current_dir = (current_dir + 1) % 4
                                output.append((ACT_FORWARD, current_pos[0], current_pos[1],
                                               current_dir, 0))
                                current_pos = path[step_i + 1]

    return output


# ── Exposure B barrier-drop sequence ─────────

# Zone trigger positions from EXPB_BARRIER_SEQUENCE (stages 2-5)
_EXPB_ZONE_TARGETS = [(8, 6), (6, 4), (4, 6), (6, 8)]

# Steps needed for timed stages (acclimation + CE drop)
_EXPB_ACCLIM_STEPS = 60   # EXPB_ACCLIMATION_STEPS
_EXPB_TIMED_STEPS = 30    # EXPB_BARRIER_DELAY_STEPS

# Reachable positions at each expb barrier stage (flood-fill from center
# through opened cross barriers, blocked by arm entry barriers).
_EXPB_REACHABLE = {
    0: {(6, 6)},
    1: {(6, 6), (7, 6), (8, 6)},
    2: {(6, 4), (6, 5), (6, 6), (7, 6), (8, 6)},
    3: {(4, 6), (5, 6), (6, 4), (6, 5), (6, 6), (7, 6), (8, 6)},
    4: {(4, 6), (5, 6), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (7, 6), (8, 6)},
}
# Zone trigger positions that advance to the next stage
_EXPB_ZONE_ADVANCE = {(8, 6): 2, (6, 4): 3, (4, 6): 4, (6, 8): 5}


def generate_exposure_b_actions(
    grid_df,
    reward_events: list,
    timed_phase_end_t_ms: float,
    rng,
    build_pause: bool,
    pause_threshold_ms: int,
    max_gap: int = 2,
    consolidate_pauses: bool = True,
) -> list:
    """Generate actions for an Exposure B session.

    Timed phases (stages 0-1): stochastic turns + pauses at center while the
    agent is confined and the env's delay counters tick.
    After timed phases: real coordinate data with arm entry barrier positions
    filtered out. The rat's corridor exploration naturally hits zone trigger
    positions, progressing the barrier state machine. Once all barriers drop,
    full maze exploration + well visits use real data.

    Args:
        grid_df: Full session grid-mapped DataFrame.
        reward_events: List of (t_ms, well_pos) for reward matching.
        timed_phase_end_t_ms: Timestamp after timed barrier stages end.
        rng: numpy random generator.
        build_pause: Whether to include pause actions.
        pause_threshold_ms: Dwell threshold for pause.
        max_gap: Flicker sensitivity.

    Returns:
        List of (action, grid_x, grid_y, direction, rewarded) tuples.
    """
    output = []
    current_pos = (6, 6)
    current_dir = rng.integers(0, 4)  # random initial direction

    # ── Timed phases: stochastic turns + pauses at center ──────
    total_timed_steps = _EXPB_ACCLIM_STEPS + _EXPB_TIMED_STEPS
    for _ in range(total_timed_steps):
        if rng.random() < 0.3:
            act = ACT_RIGHT if rng.random() < 0.5 else ACT_LEFT
            output.append((act, current_pos[0], current_pos[1], current_dir, 0))
            if act == ACT_LEFT:
                current_dir = (current_dir - 1) % 4
            else:
                current_dir = (current_dir + 1) % 4
        else:
            output.append((ACT_PAUSE, current_pos[0], current_pos[1],
                           current_dir, 0))

    # ── Real coordinate data: barrier exploration + reward phase ──────
    real_df = grid_df[grid_df['t_ms'] >= timed_phase_end_t_ms].reset_index(drop=True)
    real_df = real_df[
        ~((real_df['grid_x'] == 0) & (real_df['grid_y'] == 0))
    ].reset_index(drop=True)

    if len(real_df) > 0:
        grid_x = real_df['grid_x'].values
        grid_y = real_df['grid_y'].values
        timestamps = real_df['t_ms'].values.astype(float)
        runs = consolidate_grid(grid_x, grid_y, max_gap=max_gap)
        runs = filter_phantom_jumps(runs)

        # Split into barrier phase and reward phase.
        # During barrier phase: filter by reachable set, use barrier-aware
        # segment generation. After stage 5: normal exposure actions.
        barrier_stage = 1  # start after timed phases (CE already dropped)
        barrier_runs = []
        reward_phase_start_idx = len(runs)  # default: all barrier

        # Sustained-and-sequential filter: only advance barrier_stage one
        # stage at a time (no skipping), and only when the run at the
        # zone-advance cell is sustained (≥5 frames after consolidation).
        # The env's trigger system requires sequential crossings: visiting
        # (6,8) in stage 2 isn't a trigger event because the (6,8)-→stage-5
        # trigger is only armed in stage 4. Without this, brief phantom
        # visits at later-stage zones cause yoking to jump ahead of env.
        _ZONE_ADVANCE_MIN_FRAMES = 5
        for ri, (gx, gy, s, e) in enumerate(runs):
            pos = (gx, gy)
            if pos in _EXPB_ZONE_ADVANCE:
                new_stage = _EXPB_ZONE_ADVANCE[pos]
                run_len = e - s
                if (new_stage == barrier_stage + 1
                        and run_len >= _ZONE_ADVANCE_MIN_FRAMES):
                    barrier_stage = new_stage
                    # Mark that delay pauses are needed after this run
                    barrier_runs.append((gx, gy, s, e))
                    barrier_runs.append('DELAY')  # sentinel for delay insert
                    continue
                # Out-of-order or brief-flicker zone-advance hit: don't
                # advance stage. Fall through to reachable-set check —
                # the cell will be emitted only if it's in the current
                # stage's reachable set.

            if barrier_stage >= 5:
                reward_phase_start_idx = ri
                break

            if pos in _EXPB_REACHABLE.get(barrier_stage, set()):
                barrier_runs.append((gx, gy, s, e))

        # ── Barrier phase: navigate filtered runs within reachable set ──
        # Track navigation stage separately from the collection stage.
        # nav_stage advances only when we process a DELAY sentinel,
        # keeping the BFS reachable set in sync with the env's actual
        # barrier state.
        nav_stage = 1
        for entry in barrier_runs:
            # Insert delay pauses after zone triggers (env needs 30 steps
            # before the next zone trigger activates)
            if entry == 'DELAY':
                for _ in range(_EXPB_TIMED_STEPS):
                    output.append((ACT_PAUSE, current_pos[0], current_pos[1],
                                   current_dir, 0))
                nav_stage += 1
                continue

            gx, gy, s, e = entry
            target = (gx, gy)
            if target == current_pos:
                # Dwell — emit pause(s)
                if build_pause:
                    dwell = timestamps[min(e-1, len(timestamps)-1)] - timestamps[s]
                    for _ in range(_pause_count(dwell, pause_threshold_ms,
                                                consolidate_pauses)):
                        output.append((ACT_PAUSE, current_pos[0], current_pos[1],
                                       current_dir, 0))
                continue

            # Navigate from current_pos to target within reachable set
            reachable = _EXPB_REACHABLE.get(nav_stage, WALKABLE_CELLS)
            if target not in reachable:
                # Target is outside the current reachable set — skip it.
                # The env hasn't opened this corridor yet.
                continue

            dx = target[0] - current_pos[0]
            dy = target[1] - current_pos[1]
            if (dx, dy) in DELTA_TO_DIR:
                # Adjacent — turn and forward
                target_dir = DELTA_TO_DIR[(dx, dy)]
                for act in turn_actions(current_dir, target_dir, rng):
                    output.append((act, current_pos[0], current_pos[1],
                                   current_dir, 0))
                    if act == ACT_LEFT:
                        current_dir = (current_dir - 1) % 4
                    elif act == ACT_RIGHT:
                        current_dir = (current_dir + 1) % 4
                output.append((ACT_FORWARD, current_pos[0], current_pos[1],
                               current_dir, 0))
                current_pos = target
            else:
                # Non-adjacent — BFS within current reachable set
                path = []
                if current_pos != target:
                    visited = {current_pos}
                    queue = deque([(current_pos, [current_pos])])
                    while queue:
                        p, pth = queue.popleft()
                        for ddx, ddy in [(1,0),(0,1),(-1,0),(0,-1)]:
                            nxt = (p[0]+ddx, p[1]+ddy)
                            if nxt in reachable and nxt not in visited:
                                new_pth = pth + [nxt]
                                if nxt == target:
                                    path = new_pth
                                    queue.clear()
                                    break
                                visited.add(nxt)
                                queue.append((nxt, new_pth))
                if path and len(path) >= 2:
                    for si in range(len(path) - 1):
                        sdx = path[si+1][0] - path[si][0]
                        sdy = path[si+1][1] - path[si][1]
                        td = DELTA_TO_DIR[(sdx, sdy)]
                        for act in turn_actions(current_dir, td, rng):
                            output.append((act, current_pos[0], current_pos[1],
                                           current_dir, 0))
                            if act == ACT_LEFT:
                                current_dir = (current_dir - 1) % 4
                            elif act == ACT_RIGHT:
                                current_dir = (current_dir + 1) % 4
                        output.append((ACT_FORWARD, current_pos[0], current_pos[1],
                                       current_dir, 0))
                        current_pos = path[si + 1]

        # ── Reward phase: full maze open, use normal exposure logic ──
        reward_runs = runs[reward_phase_start_idx:]
        if len(reward_runs) > 0:
            first_run_pos = (int(reward_runs[0][0]), int(reward_runs[0][1]))

            # Bridge position
            if first_run_pos != current_pos:
                path = find_path(current_pos, first_run_pos)
                if path and len(path) >= 2:
                    for step_i in range(len(path) - 1):
                        sdx = path[step_i+1][0] - path[step_i][0]
                        sdy = path[step_i+1][1] - path[step_i][1]
                        target_dir = DELTA_TO_DIR[(sdx, sdy)]
                        for act in turn_actions(current_dir, target_dir, rng):
                            output.append((act, current_pos[0], current_pos[1],
                                           current_dir, 0))
                            if act == ACT_LEFT:
                                current_dir = (current_dir - 1) % 4
                            elif act == ACT_RIGHT:
                                current_dir = (current_dir + 1) % 4
                        output.append((ACT_FORWARD, current_pos[0], current_pos[1],
                                       current_dir, 0))
                        current_pos = path[step_i + 1]

            seg_out = generate_actions(
                reward_runs, timestamps, reward_events, [], rng,
                build_pause, pause_threshold_ms, consolidate_pauses,
            )

            # Bridge direction
            if seg_out:
                first_dir = seg_out[0][3]
                if first_dir != current_dir:
                    _CORNERS = {(10, 10), (2, 10), (2, 2), (10, 2)}
                    if current_pos in _CORNERS:
                        while current_dir != first_dir:
                            output.append((ACT_RIGHT, current_pos[0],
                                           current_pos[1], current_dir, 0))
                            current_dir = (current_dir + 1) % 4
                    else:
                        for act in turn_actions(current_dir, first_dir, rng):
                            output.append((act, current_pos[0], current_pos[1],
                                           current_dir, 0))
                            if act == ACT_LEFT:
                                current_dir = (current_dir - 1) % 4
                            elif act == ACT_RIGHT:
                                current_dir = (current_dir + 1) % 4

            output.extend(seg_out)

    return output


# ── Pipeline entry point ─────────────────────

def build_action_sequence(
    grid_df: pd.DataFrame,
    reward_events: list[tuple[float, tuple[int, int]]],
    pretrial_boundaries: list[tuple[float, float]],
    seed: int = 42,
    build_pause: bool = True,
    pause_threshold_ms: int = 1500,
    max_gap: int = 2,
    trial_configs: list | None = None,
    session_number: str | None = None,
    use_real_pretrial: bool = False,
    consolidate_pauses: bool = True,
    registered_well_visits: list | None = None,
) -> pd.DataFrame:
    """Convert a grid-mapped coordinate DataFrame to an action sequence.

    For acquisition sessions (when trial_configs is provided and grid_df has
    phase/trial_number columns), uses phase-aware generation. For exposure
    sessions, uses the original single-stream approach.

    Args:
        grid_df: DataFrame with columns (t_ms, x, y, zone, grid_x, grid_y).
            For acquisition: also has (phase, trial_number) from phase join.
        reward_events: List of (t_ms, well_pos) tuples for rewarded well visits.
        pretrial_boundaries: List of (pretrial_t_ms, trial_start_t_ms) tuples.
            Ignored when trial_configs is provided (phase-aware path).
        seed: Random seed for 50/50 turnaround rolls.
        build_pause: Whether to include pause actions.
        pause_threshold_ms: Cell dwell >= this triggers a pause action.
        max_gap: Flicker sensitivity for grid consolidation.
        trial_configs: List of [start_arm, cue, goal, tag] per trial.
            When provided, uses phase-aware generation.

    Returns:
        DataFrame with columns (step, action, grid_x, grid_y, direction, rewarded).
    """
    rng = np.random.default_rng(seed)

    # Compute per-trial barrier positions for acquisition sessions
    trial_barrier_sets = None
    iti_sim_configs = None
    if trial_configs is not None and 'phase' in grid_df.columns:
        from corner_maze_rl.env.corner_maze_env import CornerMazeEnv as _Env
        from corner_maze_rl.env.constants import BARRIER_LOCATIONS as _BL
        _tmp_env = _Env(render_mode='rgb_array', session_type='exposure')
        _tmp_env.reset()
        trial_barrier_sets = []
        iti_barrier_sets = []
        iti_sim_configs = []  # (iti_configs_tuple3, initial_idx) per ITI for co-sim
        for ti, cfg in enumerate(trial_configs):
            # Trial barriers
            layout = _tmp_env.maze_config_trl_list[cfg[0]][cfg[1]][cfg[2]]
            barriers = {_BL[i] for i in range(16) if layout[1 + i] == 1}
            trial_barrier_sets.append(barriers)

            # ITI barriers from the initial sub-config selected by proximity.
            # _select_iti_configuration picks: proximal(2), proximal_adj(1), distal(0)
            if ti < len(trial_configs) - 1:
                next_arm = trial_configs[ti + 1][0]
                iti_configs = _tmp_env.maze_config_iti_list[next_arm]
                # Determine goal_loc from current trial's well flags
                goal_loc = cfg[2]  # goal index from trial config
                start_arm_loc = next_arm
                if start_arm_loc == goal_loc:
                    config_idx = 2  # proximal
                elif start_arm_loc == (goal_loc + 1) % 4:
                    config_idx = 1  # proximal_adj
                else:
                    config_idx = 0  # distal
                iti_layout = iti_configs[config_idx]
                iti_barriers = {_BL[i] for i in range(16) if iti_layout[1 + i] == 1}
                iti_barrier_sets.append(iti_barriers)
                iti_sim_configs.append((iti_configs, config_idx))
            else:
                iti_barrier_sets.append(set())
                iti_sim_configs.append(None)
        _tmp_env.close()

    # Phase-aware path for acquisition sessions
    if trial_configs is not None and 'phase' in grid_df.columns:
        output = generate_acquisition_actions(
            grid_df, trial_configs, reward_events, rng,
            build_pause, pause_threshold_ms, max_gap,
            trial_barrier_sets=trial_barrier_sets,
            iti_barrier_sets=iti_barrier_sets,
            iti_sim_configs=iti_sim_configs,
            use_real_pretrial=use_real_pretrial,
            consolidate_pauses=consolidate_pauses,
            registered_well_visits=registered_well_visits,
        )
    elif session_number == '2e' and reward_events:
        # Exposure B: stochastic turns+pauses for timed phases,
        # then real coordinate data for zone triggers + rewards.
        # timed_phase_end_t_ms is passed via pretrial_boundaries[0]
        # when called from build_yoked for 2e sessions.
        timed_end = pretrial_boundaries[0] if pretrial_boundaries else reward_events[0][0] - 30000
        output = generate_exposure_b_actions(
            grid_df, reward_events, timed_end, rng,
            build_pause, pause_threshold_ms, max_gap,
            consolidate_pauses,
        )
    else:
        # Exposure A / legacy path: single continuous stream
        # Filter leading zone-0 rows (placement noise)
        zones = grid_df['zone'].values
        first_valid = 0
        for i in range(len(zones)):
            if int(zones[i]) != 0:
                first_valid = i
                break
        if first_valid > 0:
            grid_df = grid_df.iloc[first_valid:].reset_index(drop=True)

        # Remove off-grid (0,0) positions
        off_grid = (grid_df['grid_x'] == 0) & (grid_df['grid_y'] == 0)
        if off_grid.any():
            grid_df = grid_df[~off_grid].reset_index(drop=True)

        grid_x = grid_df['grid_x'].values
        grid_y = grid_df['grid_y'].values
        timestamps = grid_df['t_ms'].values.astype(float)

        runs = consolidate_grid(grid_x, grid_y, max_gap=max_gap)
        output = generate_actions(
            runs, timestamps, reward_events, list(pretrial_boundaries), rng,
            build_pause, pause_threshold_ms, consolidate_pauses,
        )

    if not output:
        return pd.DataFrame(columns=['step', 'action', 'grid_x', 'grid_y', 'direction', 'rewarded'])

    # Truncate at the last rewarded PICKUP. DT per-trial RTG requires every
    # session to end on a real reward — no dangling pretrial, aborted trial
    # nav, wrong-well-only final trial, or post-final-reward ITI. If no
    # rewarded action exists, the session has no usable signal and is dropped.
    last_reward_idx = None
    for idx in range(len(output) - 1, -1, -1):
        if output[idx][4] == 1:
            last_reward_idx = idx
            break
    if last_reward_idx is None:
        return pd.DataFrame(columns=['step', 'action', 'grid_x', 'grid_y', 'direction', 'rewarded'])
    output = output[:last_reward_idx + 1]

    out_df = pd.DataFrame(output, columns=['action', 'grid_x', 'grid_y', 'direction', 'rewarded'])
    out_df.insert(0, 'step', range(len(out_df)))
    return out_df


# ── Standalone script (legacy) ───────────────

if __name__ == '__main__':
    SUBJECT = 'CM005'
    SESSION_NUMBER = '1e'
    MAX_GAP = 2
    BUILD_WITH_PAUSE = True
    PAUSE_THRESHOLD_MS = 1500
    RANDOM_SEED = 42

    INPUT_CSV = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'csv', f'{SUBJECT}_{SESSION_NUMBER}.csv'
    )
    PARQUET_PATH = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'dataframes', 'all_sessions.parquet'
    )
    REWARD_PARQUET_PATH = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'dataframes', 'exposure_reward_times.parquet'
    )
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'csv')

    # Legacy reward/boundary loaders for standalone mode
    def _load_exposure_rewards(subject, session_number):
        df = pd.read_parquet(REWARD_PARQUET_PATH)
        mask = (df['subject'] == subject) & (df['session_number'] == session_number)
        matches = df[mask]
        if len(matches) == 0:
            return []
        row = matches.iloc[0]
        rewards = []
        for i in range(1, 34):
            col = f'reward_{i}'
            val = row.get(col)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                continue
            if isinstance(val, str):
                import ast
                try:
                    val = ast.literal_eval(val)
                except (ValueError, SyntaxError):
                    continue
            if isinstance(val, tuple) and len(val) >= 4:
                ts_ms = val[0]
                well_zone = int(val[3])
                well_pos = ZONE_TO_WELL_POS.get(well_zone)
                if well_pos:
                    rewards.append((ts_ms, well_pos))
        return rewards

    def _load_pretrial_boundaries(subject, session_number):
        df = pd.read_parquet(PARQUET_PATH)
        mask = (df['subject'] == subject) & (df['session_number'] == session_number)
        matches = df[mask]
        if len(matches) == 0:
            return []
        row = matches.iloc[0]
        act_types = np.array(row['sess_act_type'])
        sess_ts = np.array(row['sess_time_stamp'], dtype=float)
        boundaries = []
        for i, at in enumerate(act_types):
            if at == 'Pretrial':
                trial_start_ts = None
                for j in range(i + 1, len(act_types)):
                    if act_types[j] == 'Trial Start':
                        trial_start_ts = sess_ts[j]
                        break
                if trial_start_ts is not None:
                    boundaries.append((sess_ts[i], trial_start_ts))
        return boundaries

    # Load input CSV
    csv_df = pd.read_csv(INPUT_CSV)
    n_raw = len(csv_df)

    # Filter leading zone-0 rows
    zones = csv_df['zone'].values
    first_valid = 0
    for i in range(len(zones)):
        if int(zones[i]) != 0:
            first_valid = i
            break
    if first_valid > 0:
        csv_df = csv_df.iloc[first_valid:].reset_index(drop=True)
        print(f"Trimmed {first_valid} leading zone-0 rows (placement noise)")

    # Remove off-grid (0,0) positions
    off_grid = (csv_df['grid_x'] == 0) & (csv_df['grid_y'] == 0)
    n_off = off_grid.sum()
    if n_off > 0:
        csv_df = csv_df[~off_grid].reset_index(drop=True)
        print(f"Removed {n_off} off-grid (0,0) rows (off-maze noise)")

    print(f"Kept {len(csv_df)} of {n_raw} raw rows")

    grid_x = csv_df['grid_x'].values
    grid_y = csv_df['grid_y'].values
    timestamps = csv_df['time_stamp'].values

    # Determine session type
    pq_df = pd.read_parquet(PARQUET_PATH)
    pq_mask = (pq_df['subject'] == SUBJECT) & (pq_df['session_number'] == SESSION_NUMBER)
    pq_row = pq_df[pq_mask].iloc[0]
    session_type = pq_row.get('session_type', '')

    # Load boundary data based on session type
    reward_events = []
    pretrial_ts = []

    if session_type == 'Exposure':
        reward_events = _load_exposure_rewards(SUBJECT, SESSION_NUMBER)
        print(f"Loaded {len(reward_events)} reward events from exposure_reward_times.parquet")
    else:
        pretrial_ts = _load_pretrial_boundaries(SUBJECT, SESSION_NUMBER)
        print(f"Loaded {len(pretrial_ts)} pretrial boundaries from all_sessions.parquet")

    # Consolidate grid positions
    runs = consolidate_grid(grid_x, grid_y, max_gap=MAX_GAP)
    print(f"Consolidated {len(grid_x)} timesteps into {len(runs)} distinct position runs")

    # Count well visits
    well_visit_count = sum(1 for gx, gy, _, _ in runs if (gx, gy) in WELL_POSITIONS)
    print(f"Well visits detected: {well_visit_count}")

    # Generate actions
    rng = np.random.default_rng(RANDOM_SEED)
    output = generate_actions(
        runs, timestamps, reward_events, pretrial_ts, rng,
        BUILD_WITH_PAUSE, PAUSE_THRESHOLD_MS,
    )

    # Write output CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f'{SUBJECT}_{SESSION_NUMBER}_actions.csv')

    out_df = pd.DataFrame(output, columns=['action', 'grid_x', 'grid_y', 'direction', 'rewarded'])
    out_df.insert(0, 'step', range(len(out_df)))
    out_df.to_csv(out_path, index=False)

    # Summary
    action_names = {0: 'left', 1: 'right', 2: 'forward', 3: 'pickup', 4: 'pause'}
    print(f"\nWrote {len(out_df)} actions to {out_path}")
    print(f"  Subject: {SUBJECT}, Session: {SESSION_NUMBER} ({session_type})")
    print(f"  Action counts:")
    for act_id in sorted(out_df['action'].unique()):
        count = (out_df['action'] == act_id).sum()
        print(f"    {action_names.get(act_id, act_id)}: {count}")
    rewarded_count = (out_df['rewarded'] == 1).sum()
    print(f"  Rewarded well entries: {rewarded_count}")
