# Environment Architecture

Authoritative spec of the corner maze MiniGrid environment as implemented in the legacy repo (`corner-maze-rl-legacy/src/env/corner_maze_env.py`). The new repo's env will be built against this spec; behavior must not diverge without explicit decision.

Source extracted from legacy MEMORY.md, 2026-05-05. Cross-reference with [`maze-behavior-spec.md`](maze-behavior-spec.md) for task-level rules and [`reward-structure-analysis.md`](reward-structure-analysis.md) for reward shaping rationale.

---

## Grid & spatial

13×13 MiniGrid.

| Element | Positions |
|---------|-----------|
| Corner positions (4) | (10,10), (2,10), (2,2), (10,2) |
| Wells (4) | (11,11), (1,11), (1,1), (11,1) |
| Cues (4) | (11,6), (6,11), (1,6), (6,1) |
| Barriers | 12 |
| Triggers | 12 |
| Pretrial trigger positions | 8 |

## Actions

| ID | Action | Notes |
|----|--------|-------|
| 0 | left turn | |
| 1 | right turn | |
| 2 | forward | |
| 3 | enter well | only legal at `CORNER_POSES` |
| 4 | pause | disabled by default |

## Observations

The env exposes three obs keys; downstream code picks one (or wraps).

| Key | Shape | Use case |
|-----|-------|----------|
| `view` | 21×21×3 RGB | MiniGrid default. Used with `ImgObsWrapper` + `CnnPolicy`. |
| `embedding` | 60-dim vector | From parquet, used with `EmbeddingObsWrapper` + `MlpPolicy`. |
| `stereo` | 96×96×2 | Stereo eye images, used with `StereoObsWrapper` + `MlpPolicy` + `StereoFeaturesExtractor`. |

Eye images (128×128) loaded from parquet for rendering only.

## Layout system

- **Layout tuples**: 37 elements — `[state_type, 12 barriers, 4 cues, 4 wells, 12 triggers]`.
- `layouts` dict maps name → tuple; `layout_name_lookup` is the reverse.
- Lookup tables:
  - `maze_config_trl_list[arm][cue][goal]`
  - `maze_config_pre_list[arm][cue]`
  - `maze_config_iti_list[arm]`
- `grid_configuration_sequence`: ordered list of `(state_type, layout_tuple)` that drives the session.

## Session phases

| Phase | Description |
|-------|-------------|
| **Exposure** (`expa`/`expb`) | Free exploration, 4 reward wells, cycle-alternation. |
| **Pretrial** | Dead-end entry, cue presentation. |
| **Trial** | Goal-directed with cues and barriers. |
| **ITI** | Inter-trial interval; 3 sub-configs switched via A/B triggers (`_iti_config_idx`). Both centre doors on the start arm's axis are up throughout (the start arm's own and the one opposite), as on the rig; before 2026-10-09 only the start arm's was, see corner-maze-analysis `docs/context/constants/doors.json`. |

## State types

| Constant | Value |
|----------|-------|
| `STATE_BASE` | 0 |
| `STATE_EXPA` | 1 |
| `STATE_EXPB` | 2 |
| `STATE_PRETRIAL` | 3 |
| `STATE_TRIAL` | 4 |
| `STATE_ITI` | 5 |

## Session types (16+)

Enumerated paradigms exposed by the env:

- Exposure
- PI+VC f2 / PI+VC f1: acquisition, novel route, no cue, reversal, rotate
- PI: novel route no cue, reversal no cue
- VC: acquisition, novel route rotate, reversal rotate
- Single-trial variants

## Reward structure

| Constant | Value | Meaning |
|----------|-------|---------|
| `STEP_FORWARD_COST` | -0.0005 | Per forward step. |
| `STEP_TURN_COST` | -0.001 | Per turn. 2× forward cost to discourage spinning. |
| `WELL_REWARD_SCR` | 1.061 | Correct well visit. |
| Empty well | -0.005 | Wrong / non-rewarded well. |

`_compute_reward(well_event, action)` selects the per-step cost by action type and applies the well event delta when applicable.

**Note:** the env's per-step reward is *not* the per-trial `+1` score (max +32/session) used for human-readable evaluation. Agents see the env reward; humans grade with the `+1`/trial score. See [`reward-structure-analysis.md`](reward-structure-analysis.md) for rationale.

## Pose labels & embeddings

- **Format**: `{prefix}_{x}_{y}_{dir}` — e.g. `trl_e_s_xx_8_2_0`.
- `_get_pose_label()` constructs the key; uses `_iti_config_idx` for ITI list entries.
- `_load_embeddings()` loads the parquet once and caches embeddings + eye images.

## Data serialization conventions

(For the new repo's run-saving layer to mirror.)

- Parquet columns with mixed-type nested lists (`trajectory`, `trial_scores`, `turn_scores`, `session_scores`, `trial_tags`, `trial_configs`, `sequence_labels`) are serialized as JSON strings — pyarrow can't store mixed-type nested lists natively.
- Custom `_default` handler in `save_episode_dataframe()` (legacy `session_runner.py`) handles numpy `int64`/`float64`.
- Parse back with `json.loads()`.

## Training-loop conventions (carry-over from legacy)

- All training scripts wrap the loop in `try/except KeyboardInterrupt` with `finally` for graceful Ctrl+C saves.
- For SB3 path: `SB3 DummyVecEnv` auto-resets *before* `_on_step()` — read terminal data via `self.locals['dones']` and `self.locals['infos'][0]`.
- `self.logger.record()` must be called every step to persist values.

## Hyperparameter reference values (legacy `train_maskable_ppo.py`)

Anchor values; new repo's PPO defaults should start here unless we have reason to deviate.

| Param | Value |
|-------|-------|
| `N_STEPS` | 64 |
| `BATCH_SIZE` | 16 |
| `MAX_STEPS` | 6144 |
| `n_epochs` | 4 |
| `gamma` | 0.999 |
| `gae_lambda` | 0.95 |
| `clip_range` | 0.2 |
| `max_grad_norm` | 0.5 |
| `lr` (CNN) | 1e-5 |
| `lr` (embedding MLP) | 1e-4 |
| `ent_coef` | 0.01 |
| `vf_coef` | 0.5 |
