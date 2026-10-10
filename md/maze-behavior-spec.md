# 2S2C Maze: the MiniGrid mapping

This file holds only what is specific to the MiniGrid environment in this repository: how the
real maze's arms, wells and corners map onto the 13 × 13 grid, how the environment runs the
phases of a trial, its reward constants, and the modelling implications. The task itself
(apparatus, protocol, cohorts, scoring, results) is described once, in the analysis
repository's context package (`corner-maze-analysis/docs/context/`, private; start at its
`00_START_HERE.md`). Its door and zone registry is vendored here under
`src/corner_maze_rl/data/` with a hash manifest. Where an older copy of this file or any note in
this repository restates a task fact, the package wins. (Trimmed 2026-10-10; the previous
version restated the task rules and carried a stale roster and a wrong account of Exposure.)

## Subject roster

See the README's "Manuscript subject roster": 47 rats are in the manuscript (PI+VC 17, PI 12,
VC 12, PI+VC_f1 6); the dataset ships 56 yoked subjects, 9 of them outside the manuscript.
`training_group`, `novel_route_probe` and `reversal_probe` come from the upstream
`subjects.parquet`; CM identifiers are the names, `subject_id` is the join key.

## Arm structure

The plus maze has four arms connecting the center intersection to the perimeter. Each arm spans
two zones in the rig's tracking vocabulary: an inner zone adjacent to the center and an outer
(junction) zone toward the perimeter.

### Real maze (zone-based)

| Arm | Inner zone | Outer zone | Direction |
|-----|-----------|-----------|-----------|
| North | 12 | 13 | increasing x |
| South | 10 | 9 | decreasing x |
| East | 15 | 19 | increasing y |
| West | 7 | 3 | decreasing y |

The start arm for each trial is identified by the inner zone: North=12, South=10, East=15, West=7.

### MiniGrid mapping

| Arm | Grid coordinates (approx.) | Start pose (x, y, dir) |
|-----|---------------------------|----------------------|
| North | (6, 2)-(6, 5) | (6, 3, 0) facing East |
| East | (7, 6)-(10, 6) | (9, 6, 1) facing South |
| South | (6, 7)-(6, 10) | (6, 9, 2) facing West |
| West | (2, 6)-(5, 6) | (3, 6, 3) facing North |

Center intersection: (6, 6). Grid size: 13x13. The full zone → grid-cell table, including the
ring segments and the junctions (zone 3 = west junction → (2, 6); zone 9 = south junction →
(6, 10), fixed 2026-10-09), is `src/corner_maze_rl/data/zones.json` and
`yoking/map_to_minigrid.py`.

## Goal well structure

| Corner | Real zone | MiniGrid well pos | MiniGrid corner pos |
|--------|----------|-------------------|-------------------|
| SW | 1 | (1, 1) | (2, 2) |
| NW | 5 | (1, 11) | (2, 10) |
| SE | 17 | (11, 1) | (10, 2) |
| NE | 21 | (11, 11) | (10, 10) |

In MiniGrid, the agent navigates to the corner position (CORNER_POSES), then uses the ENTER_WELL
action (action 3) to visit the adjacent well position and receive reward.

### Cue-goal orientation

Each subject has a fixed cue-goal orientation (e.g., "N/NE") that determines which goal corner
goes with the lit monitor. Within a session the goal corner is fixed for fixed-frame session
types and rotates with the cue every trial for rotating types; across sessions the task frame
is rotated against the room. `data/canonical.py` rotates fixed-frame sessions to a canonical
cue-north frame at consume time (stored tables are in the room frame).

### Trial reward

- Correct well entered during trial: R = +1 (WELL_REWARD_SCR)
- Wrong well entered during trial: R = small negative (step cost)
- Exposure well: R = +1 (but this is exposure reward, NOT trial reward)

## Phases in the environment

- **Pretrial**: the agent is confined to the start arm by the pretrial layout; a minimum of
  PRETRIAL_MIN_STEPS = 7 steps is enforced before the trigger activates; the cue is shown.
- **Trial**: the trial layout drops the start arm's centre door; the trial ends when the
  correct well is entered.
- **ITI**: the `iti_*` layouts route the agent back to the next start arm; sub-phases are
  controlled by the A/B triggers. Since 2026-10-09 the ITI layouts raise the centre door
  opposite the start arm, as the rig does.
- **Exposure**: the `expa_*` and `expb_*` layouts correspond to the rig's 1e and 2e sessions.
  Upstream these are two separate days before acquisition, not phases inside an acquisition
  session, and all four wells pay.

The doors behind every layout are `BARRIER_LOCATIONS` in `env/constants.py`; their
correspondence to the rig's door columns and the rig's layout per phase and start arm are in
the vendored `data/doors.json` (checked by `tests/test_context_constants.py`).

## Key constants (MiniGrid)

| Constant | Value | Notes |
|----------|-------|-------|
| PRETRIAL_MIN_STEPS | 7 | Minimum steps in pretrial before trigger activates |
| STEP_FORWARD_COST | -0.0005 | Default (overridden to 0.0 for SR) |
| STEP_TURN_COST | -0.001 | Default (overridden to 0.0 for SR) |
| WELL_REWARD_SCR | 1.061 | Default (overridden to 1.0 for SR) |
| MAX_STEPS | 6000 | Typical episode step limit |
| Grid size | 13×13 | With walls on perimeter |
| Valid floor positions | 49 | Reachable maze positions |
| Directions | 4 | 0=right, 1=down, 2=left, 3=up |

The rig's own constants (dwell times, debounce, pauses, resolutions) are in the package's
`constants/rig_constants.json`, not here.

## Implications for SR modeling

1. **Exposure vs trial reward:** w should ONLY learn from trial-phase rewards. Exposure rewards are non-discriminative (all 4 wells rewarded).
2. **Goal rotation:** The correct well depends on the trial's cue. A position-only SR without cue information can only learn for a single fixed goal. To handle cue-dependent goals, cue must be part of the state representation.
3. **Session structure:** The full pretrial→trial→ITI loop is ~21-33 steps (optimal path). SR with teleports simplifies to just the trial decision; without teleports, the agent must navigate all phases.
4. **Barrier changes:** Barriers differ between pretrial (dead-end), trial (choice corridors), and ITI (open). A single SR M matrix across all phases faces non-stationary transitions (Momennejad 2017, Russek 2017).
