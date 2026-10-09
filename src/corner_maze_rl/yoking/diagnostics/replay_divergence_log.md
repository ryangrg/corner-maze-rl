# Replay Divergence Log — CM016

Tracking misalignments between yoked action data and env agent state.
Each entry: find divergence → diagnose → fix → rebuild → re-run.

---

## Divergence #1 — Step 38 (ITI start after first reward)

**Symptom**: At step 38, expected pos=(11,1) dir=2, actual pos=(10,2) dir=2.

**Context**: Steps 34-37 are the well visit block (PICKUP, R, R, F). The trial segment ends after the well visit. Step 38 is the first action of the ITI segment.

**Cause**: Same gap pattern as the pretrial→trial boundary. After the well visit block, the generator ends at current_pos=(10,2) (well exit). The ITI segment then resets `current_pos = (int(runs[0][0]), int(runs[0][1]))` to (11,1) — the first ITI tracking coordinate (rat still in the well zone). The generator records actions from (11,1), but the env agent is at (10,2).

**Pattern**: At EVERY segment boundary (pretrial→trial, trial→ITI, ITI→pretrial), there's a potential gap between where the previous segment left the agent and where the new segment's tracking data starts.

**Fix**: Applied bridge logic + skip leading well-position runs in ITI segment. ✓ Resolved.

---

## Divergence #2 — Step 59 (ITI barrier blocks rat's actual path)

**Symptom**: At step 59, agent can't move from (2,6) to (3,6) — Barrier at (3,6).

**Context**: During ITI after trial 1. Trial 1: arm=0/North, goal=NE. Next trial: arm=2/South. The env's ITI layout (`maze_config_iti_list[2]`, config_idx=0 "distal") has a barrier at (3,6) that blocks the path the rat actually took.

**Cause**: The env's ITI system uses 3 sub-configs with A/B/S triggers to dynamically open paths as the rat navigates. The INITIAL config may block certain paths that only become available after the agent hits a trigger to switch configs. The yoked action data follows the rat's actual path, which passed through triggers that opened barriers. But in the replay, the initial config blocks the agent before it can reach those triggers.

**Pattern**: ITI barriers don't match the rat's actual free movement during ITI. The real maze's ITI barrier state at any moment depends on which triggers the rat has already passed through — a stateful process that the flat action sequence can't replicate.

**Fix**: Added `filter_phantom_jumps()` to remove zone-boundary artifacts where manhattan(prev,curr) > threshold AND next run returns close to prev. Also added ITI well-position skip and bridge. ✓ Resolved — advanced to step 209.

---

## Divergence #3 — Step 209 (ITI direction mismatch after well exit)

**Symptom**: At step 209, position matches (10,2) but direction differs: expected dir=1 (South), actual dir=2 (West). Trial count=4, ITI phase.

**Context**: Steps 205-208 are the well visit block (PICKUP, L, L, F). After exit, env agent is at (10,2) facing West (dir=2) — this is correct per `WELL_EXIT_RESULT[(11,1)] = ((10,2), 2)`. But the generator's ITI segment overrides `current_dir` with the direction inferred from the first ITI movement (South), and records dir=1.

**Cause**: At ITI segment start, the code does:
```python
seg_dir = current_dir  # starts as well exit dir (2=West)
for ri in range(...):  # infer from first movement
    seg_dir = DELTA_TO_DIR[...]  # finds South (1)
current_dir = seg_dir  # overrides to 1
```

This sets the RECORDED direction to match the tracking data's first movement, but the ENV agent is actually facing West (2) from the well exit. No turn is generated to bridge the gap.

**Pattern**: Direction mismatch at segment boundaries. The generator infers direction from tracking data independently of where the previous segment left the agent facing. Same concept as the position bridge but for direction.

**Fix**: Generate right-only turn bridge at corner positions (avoids CORNER_LEFT_TURN_WELL_EXIT special case). ✓ Resolved.

---

## Result: CM016 Session 5 (Acquisition) — FULL SESSION ALIGNED

After fixes #1–#3, CM016 session 5 (32 trials) runs to completion with **zero divergence**. Session ends naturally at step 1519/1520 when all 32 trials complete.

---

## Divergence #4 — CM016 1e (Exposure), Step 0 (initial direction)

**Symptom**: Expected dir=0 (East), actual dir=1 (South) at step 0.

**Cause**: Exposure sessions don't call `_inject_trial_configs` which sets initial direction for acquisition. The env defaults to `DEFAULT_AGENT_START_DIR=1` (South), but the yoked data has dir=0 (East, inferred from first movement).

**Fix**: Set `env.agent_dir` from first action's direction column after reset for exposure sessions. Applied in both `replay_session.py` main and `ReplayController._reset_env()`. ✓ Resolved.

---

## Divergence #5 — CM016 1e (Exposure), Step 194 (corner-left after well exit)

**Symptom**: Expected pos=(10,10), actual pos=(9,10) after a LEFT turn at corner (10,10).

**Cause**: Same as divergence #3 — `CORNER_LEFT_TURN_WELL_EXIT` env special case. This time in the exposure code path (`generate_actions`), not the acquisition path. After well exit, the next navigation action uses `turn_actions()` which can produce LEFT at the corner.

**Fix**: Added `just_exited_well` flag to `generate_actions()`. When set and agent is at a corner, use RIGHT-only turns. Cleared after the first forward. ✓ Resolved.

---

## Result: CM016 1e (Exposure) — FULL SESSION ALIGNED

After fixes #4–#5, CM016 1e (2593 actions, 25 rewards) runs with **zero divergence**.

---

---

## Divergence #6 — CM016 Sessions 1-4 (barrier overshoot in trial/ITI)

**Symptom**: Agent bumps into trial layout barriers — tracking data maps rat to barrier positions (1-cell overshoot). Session 5 (32/32 perfect) unaffected because the rat never approached wrong-direction barriers.

**Cause**: Same proximity artifact as exposure_b, but now in acquisition trial and ITI layouts. The rat's tracking coordinate overshoots into barrier cells, and the action generator (via adjacent moves or BFS) routes through them.

**Fix in progress**:
- Added `blocked` parameter to `find_path()` for barrier-aware BFS
- Added per-trial barrier set computation from layout tables
- Added run-level barrier filtering in `_generate_segment_actions`
- **Fixes applied**:
- Per-trial barrier sets computed from layout tables
- Per-ITI barrier sets computed from initial sub-config (proximity-based selection)
- Run filtering + barrier-aware BFS in `_generate_segment_actions`
- Goal-aware well visits: wrong wells emit visit block but don't stop; only goal well ends the trial
- Right-only turns at corners after wrong well exit

**Resolved.** Additional fixes required for error trials:
- Well visit detection: only trigger when NEXT run is at well position (rat actually entered), not just at corner
- Goal-aware wells: only stop at the CORRECT goal well; wrong well visits emit the block and continue
- LEFT special case handling: use LEFT=combined turn+forward at corners after well exit (when target matches left direction); use RIGHT-only otherwise
- `after_well_exit=True` flag passed to ITI `_generate_segment_actions` (env's WELL_EXIT state persists)
- Clear `just_exited_well` after PAUSE actions in both `generate_actions` and `_generate_segment_actions`

---

## FINAL RESULT: ALL CM016 SESSIONS ALIGNED

| Session | Type | Trials | Status |
|---------|------|--------|--------|
| 1e | Exposure A | — | all 2510 steps ✓ |
| 2e | Exposure B | — | all 1938 steps ✓ |
| 1 | Acquisition | 18 (8 perfect) | ended 1859/1860 ✓ |
| 2 | Acquisition | 26 (10 perfect) | ended 2182/2183 ✓ |
| 3 | Acquisition | 25 (18 perfect) | ended 1857/1858 ✓ |
| 4 | Acquisition | 32 (30 perfect) | ended 1734/1735 ✓ |
| 5 | Acquisition | 32 (32 perfect) | ended 1519/1520 ✓ |

---

## Batch Run: 16 Subjects (150 sessions)

**Result: 139 ok, 11 failed (92.7%)**

### Fixes applied in this batch:
- 2e sessions without reward data: fall through to exposure_a mode (no barrier sync possible without reward timestamps)
- `fwd_pos`/`fwd_cell` updated after setting `agent_dir` for exposure sessions (was stale from reset)
- Initial position set from action data for exposure sessions (stitched sessions start away from center)
- Trial segment direction bridge: generate turns from pretrial end direction to first trial movement direction (East/West arms end pretrial facing opposite to trial movement)

### Remaining 11 failures:
**Exposure sessions (8):** CM000_1e/2e, CM003_2e, CM009_1e, CM011_1e, CM015_2e, CM017_2e, CM018_2e — all diverge at the first well visit exit where the `just_exited_well` + CORNER_LEFT_TURN_WELL_EXIT interaction isn't handled correctly. The generator uses right-only turns but the env may or may not have the special case active depending on whether a dwell run at the same position counts as "no action emitted."

**Acquisition sessions (3):** CM000_7 (step 1596, tc=27), CM004_8 (step 549, tc=11), CM008_2 (step 94), CM011_4 (step 1373, tc=23) — error trial edge cases late in sessions.

### Root cause of remaining exposure failures:
After a well exit, the agent is at a corner. The env's `CORNER_LEFT_TURN_WELL_EXIT` fires on the FIRST action at that corner IF `last_pose` is a WELL_EXIT_POSE. The generator tracks this with `just_exited_well`. The mismatch happens when:
1. A dwell run at the same position (corner) is processed WITHOUT emitting a pause (dwell < threshold)
2. No action is emitted to the env, so the env's WELL_EXIT state persists
3. But the generator sees "same position, no movement" and the flag handling is ambiguous

A precise fix needs to track whether ANY action was emitted since the well exit, not just whether a forward or pause happened.

---

## Summary of all fixes:
1. **Segment boundary position bridge** — BFS bridge from carried position to first tracked position at pretrial→trial and trial→ITI boundaries. Skip leading well-position runs in ITI.
2. **Phantom jump filter** — `filter_phantom_jumps(runs, threshold=4)` removes zone-boundary artifacts (brief grid position jumps caused by 1-2 pixel shifts crossing zone boundaries).
3. **Right-only turns at corners after well exit** — Avoids `CORNER_LEFT_TURN_WELL_EXIT` env special case. Applied in both acquisition (`generate_acquisition_actions`) and exposure (`generate_actions`) code paths.
4. **Initial direction from action data** — Set agent direction from first action's direction column for exposure sessions.
5. **Last trial pickup-only** — Final trial emits only PICKUP (no turnaround/exit) since env terminates on last reward.



---

## 2026-10-09 — Zone map fix (W/S junctions) and Divergence #7 (CM063 s4)

**Zone map.** `map_to_minigrid.INTERSECTION_ZONES` mapped zone 3 → (6, 10) and zone 9 → (2, 6). The rig's
zone map (ground truth; `corner-maze-analysis/docs/context/07_pitfalls.md` A7/B13) has zone 3 as the WEST
junction (x 98–138, y 0–35; trace neighbours 7, 4, 2) and zone 9 as the SOUTH junction (x 0–34, y 101–139;
neighbours 10, 14, 6). Fixed to 3 → (2, 6), 9 → (6, 10). Over 30 acquisition sessions the raw zone→grid
trace had 20.8 % non-adjacent cell transitions with the old table and 11.6 % with the fix (the rest are the
well ↔ zone-0 corner gaps). Full rebuild, baseline (old code, same upstream) vs fixed: junction steps
S 24,594 → 39,242, W 24,796 → 37,646, N +1,025, E +923 (synthetic); exposure 161,490 → 129,280 steps;
rewards identical in all 552 sessions. Heuristics over one synthetic+exposure pass, old table → fixed:
phantom runs dropped 21,443 → 120; BFS bridged paths 31,812 → 1,630; bridge steps 105,879 → 4,358;
per-frame barrier remaps 1,266,978 → 27,867. `tests/test_zone_to_grid.py` now derives the required
mapping from the zone boxes and the walkable zone graph.

**Divergence #7 — CM063 s4 (PI+VC_f1, Fixed Cue 1 Twist), step 1178/1397 (synthetic) and 1349/1582 (real).**

*Symptom*: expected pos=(10,10) dir=2, env pos=(9,10) dir=2, trial 16, ITI.

*Context*: the rat made an unrewarded visit to the SE well during the ITI, exited to the corner (10,10)
facing north, dwelt (PAUSE), then left west along the south corridor. The ITI segment generator ended
there (its `stop_at_pos` logic), and `generate_acquisition_actions`' bridge to the next start arm emitted
LEFT (turn in place) + FORWARD. The env, however, was still in its WELL_EXIT masked state at the corner:
`CornerMazeEnv.step` does not update `last_pose` on PAUSE, and a LEFT at a corner in that state is the
corner trick (turn and step to (9,10)). So the env was one cell ahead of the stream.

*Why new*: with the junction fix the S-junction frames of this session are mapped correctly, which changed
where the ITI segment stopped; previously those frames were phantom-filtered and the bridge never started
from a well-exit corner.

*Fix*: `_env_in_well_exit_state(output, current_pos)` (last non-PAUSE action was the FORWARD recorded at a
well position, and current_pos is a corner) guards the S-trigger bridge; the first hop is then the single
trick action (LEFT → left-trick cell, FORWARD → forward-trick cell). Counted as `well_exit_bridge_trick`
(fired once in the whole rebuild). ✓ Resolved: 552/552 synthetic and 422/422 real sessions replay with
0 divergence; contiguity 552/552; path fidelity: no trials flagged.
