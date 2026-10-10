# Implementation Plan — Yoke Novel Route + Reversal probe sessions

**Status:** proposed (2026-07-20). Awaiting review before any code change.

**Mission.** Extend the yoking pipeline (`src/corner_maze_rl/yoking/`) to build **Novel Route** and **Reversal** probe sessions into the shipped dataset (`data/yoked/dataset/`), for a fixed roster of **41 subjects**. Today only Acquisition + Exposure are yoked; these two probe phases exist upstream but are gated out.

**Headline finding — this is a yoking-config task, not env work.** `CornerMazeEnv` already implements every Novel Route and Reversal paradigm these sessions need. `SESSION_GENERATORS` (`env/corner_maze_env.py:1143`) contains `gen_*_novel_route` and `gen_*_reversal` for all four groups, and the yoked-replay path already *skips* the random generator shuffle and consumes the rat's recorded `trial_configs` verbatim (`corner_maze_env.py:1174`). So no env change is required — we only add mapping entries and un-gate the phase filters. **Do not soften the env to fit the yoked side** (per the standing divergence-check rule); if a probe session diverges, fix the yoking pipeline.

**Non-destructive guarantee (hard constraint).** Nothing already built is modified. The work is **purely additive**: new PARADIGM_MAP keys (existing keys untouched), two brand-new action tables, and 82 appended `sessions.parquet` rows. No existing code path (Acquisition, Exposure), no existing per-session parquet, and no downstream consumer (training runner, notebooks, `data/canonical.py`, loaders) is edited. The one place this needs *proof* rather than assertion is Stage 2 rewrite behavior — see §4a. The env is not touched at all.

---

## 0. Scope — locked

- **Roster: exactly these 41 subjects** (confirmed 2026-07-20). This is the manuscript-48 minus **CM042** (VC) and the entire **PI+VC_f1** cohort (CM057, CM058, CM060, CM061, CM063, CM064). f1's probe session_types (`Fixed Cue Novel Route Twist`, `Fixed Cue Switch Twist`) are therefore **out of scope** and get no PARADIGM_MAP entry in this pass.

  PI+VC (17): CM000 CM001 CM002 CM003 CM004 CM005 CM006 CM007 CM008 CM009 CM010 CM011 CM014 CM015 CM016 CM017 CM018
  PI (12): CM023 CM024 CM027 CM036 CM037 CM046 CM049 CM050 CM051 CM052 CM053 CM054
  VC (12): CM025 CM026 CM031 CM035 CM038 CM040 CM041 CM043 CM044 CM045 CM055 CM056

- **Session inventory (verified upstream):** every one of the 41 rats has exactly **1 Novel Route + 1 Reversal** session = **82 new sessions** (41 + 41). All 82 have complete tracking — `session_manifest.match_status = 'found'` for all, coordinate row counts 31k–199k, `trials.parquet` present with 32–80 trials each. **Zero `no_data` gaps** (unlike the CM008 exposure hole, which is a different session).

- **Deliverable of this document:** plan only. Implementation is a separate, subsequent pass.

---

## 1. The session_type → env-paradigm map (the crux artifact)

`PARADIGM_MAP` (`src/corner_maze_rl/data/session_types.py:18`) is keyed on `(training_group, session_type)` — **not** phase — and its right-hand side must be a valid `SESSION_GENERATORS` key. All nine env paradigms below **already exist** in the registry. The mapping is fully determined by `(group, session_type)` and independently **validated** against upstream `subjects.parquet` probe columns, whose suffix encodes the probe cue condition (`PI` = no cue, `PI+VC` = fixed cue, `VC` = rotating cue):

### Novel Route (5 new entries)

| training_group | session_type | → env paradigm | probe col (`novel_route_probe`) validates |
|---|---|---|---|
| `PI+VC` | `Fixed Cue 2a` | `PI+VC f2 novel route` | (n/a; group is fixed-cue) — 17 rats |
| `PI` | `Dark Detour` | `PI novel route cue` | `PI_PI+VC` → probe = fixed cue — 6 rats |
| `PI` | `Dark Detour No Cue` | `PI novel route no cue` | `PI_PI` → probe = no cue — 6 rats |
| `VC` | `Rotate Detour` | `VC novel route fixed` | `VC_PI+VC` → probe = fixed cue — 6 rats |
| `VC` | `Rotate Detour Moving` | `VC novel route rotate` | `VC_VC` → probe = rotating cue — 6 rats |

### Reversal (4 new entries)

| training_group | session_type | → env paradigm | probe col (`reversal_probe`) validates |
|---|---|---|---|
| `PI+VC` | `Fixed Cue Switch` | `PI+VC f2 reversal` | (n/a; group is fixed-cue) — 17 rats |
| `PI` | `Dark Reverse` | `PI reversal no cue` | `PI^PI` → probe = no cue (all 12) — 12 rats |
| `VC` | `Fixed Cue Switch` | `VC reversal fixed` | `VC^PI+VC` → probe = fixed cue — 6 rats |
| `VC` | `Rotate Reverse` | `VC reversal rotate` | `VC^VC` → probe = rotating cue — 6 rats |

Notes:
- No key collision: none of these nine session_type strings is reused by any Acquisition session_type in the same group (Acquisition uses `Fixed Cue 1`, `Dark Train`, `Rotate Train`, `Fixed Cue 1 Twist`). `Fixed Cue Switch` appears under both `PI+VC` and `VC` but the group disambiguates.
- **The one row to watch during validation:** the `Rotate Detour` (→ *fixed*) vs `Rotate Detour Moving` (→ *rotate*) split. The probe column supports it (`VC_PI+VC` vs `VC_VC`), but the env's fixed-vs-rotate cue rendering for these two is the most likely place a path-fidelity divergence would surface. Confirm via the divergence checker on one rat of each before the full build.

---

## 2. Dataset layout decision — two new action tables

Novel Route and Reversal have **no synthetic/real pretrial split** (that concept is Acquisition-only). Mirroring the Exposure precedent (`actions_exposure.parquet` — single table, no pretrial variant), add **two new single action tables**:

- `actions_novel_route.parquet`
- `actions_reversal.parquet`

Same schema as the existing action tables: `session_id, step, action, grid_x, grid_y, direction, rewarded, actions_to_reward, pose_label`. The 82 new sessions also append to the shared **`sessions.parquet`** (one row each, `session_phase` = `Novel Route` / `Reversal`). `subjects.parquet` is unchanged (all 41 rats already present from Acquisition).

Loader side is already partly wired: `data_loader.py:_PHASE_TO_TAG` (`:28`) already maps `'Novel Route' → 'novel'` and `'Reversal' → 'reversal'`, so `get_trial_configs` stamps the correct probe tag on each trial tuple automatically.

---

## 3. Files to touch (exact edit sites)

### 3a. `src/corner_maze_rl/data/session_types.py`
- Add the **9 PARADIGM_MAP entries** from §1 to the dict at `:18`. Group them under `# Novel Route` / `# Reversal` comment blocks. No change to `map_session_to_env_kwargs` logic — it already returns kwargs for any mapped key and `None` for unmapped.

### 3b. `src/corner_maze_rl/yoking/build_yoked.py` (Stage 1, `corner-maze-build-yoked`)
- `:369` — extend `--phase` argparse `choices` to `['Exposure', 'Acquisition', 'Novel Route', 'Reversal']`.
- `:403-406` — extend the default-scope filter `isin([...])` to include the two new phases.
- `:302` — filename-variant logic: keep the "no suffix" branch for the new phases (they are neither Acquisition synthetic/real nor Exposure). Confirm the else-branch writes a plain per-session parquet.
- `:64` `is_exposure` master branch — the new phases fall into the **non-exposure else-branch** (`:222-256`), which uses `get_phase_coordinates` + `get_trial_rewards` + `get_trial_well_visits` + `get_trial_configs`. Verify that path handles the probe trial structure (pre-probe trials 1–16 + probe trials). No new branch expected, but this is the second place to watch.

### 3c. `src/corner_maze_rl/yoking/build_dataset.py` (Stage 2, `corner-maze-build-dataset`)
- `:108` — add `novel_route` and `reversal` buckets to the `buckets` dict.
- `:73-89` `_bucket_for` — route `phase == 'Novel Route'` → `'novel_route'`, `phase == 'Reversal'` → `'reversal'` (currently anything non-Exposure/Acquisition returns `None` → skipped).
- `:276-280` `bucket_filename` — add `'novel_route': 'actions_novel_route.parquet'`, `'reversal': 'actions_reversal.parquet'`.
- `:216-229` — **generalize the binary phase-vs-bucket assertion** (`expected = 'Exposure' if bucket=='exposure' else 'Acquisition'`) into a bucket→expected-phase lookup so it accepts the two new buckets. This is the one assertion that will otherwise throw.
- `:248-266` — the `n_trials == n_rewards == len(trial_configs)` count check is **Acquisition-only** and already skips other phases; confirm it stays skipped for the probes (probe sessions deliberately don't satisfy it — pre-probe + probe trial counts differ).
- `:231-246` — terminal-rewarded-PICKUP invariant (`last row action==3 & rewarded==1`) **must still hold** for probe sessions. If a rat's final trial ends on a wrong well, `build_action_sequence` truncation (`map_to_minigrid_actions.py:2367`) drops the tail; watch for sessions that shrink.

### 3d. `src/corner_maze_rl/yoking/compute_pose_labels.py`
- `_build_replay_env` (`:26-90`) — the new phases are **not** exposure, so they flow through the PARADIGM_MAP branch (`:72-90`) automatically once §3a lands. No hardcoded branch needed. Verify `goal_location` derivation from `trial_configs[0][2]` is valid for probe sessions.

### 3e. Diagnostics (`src/corner_maze_rl/yoking/diagnostics/`)
- `check_divergence.py`, `check_path_fidelity.py`, `check_well_visits.py` branch on `session_phase == 'Exposure'`. Confirm the non-exposure path applies cleanly to probes (it should — they use the same PARADIGM_MAP resolution at `check_divergence.py:107`). `check_well_visits.py:208` currently restricts to Acquisition — decide whether to extend it to probes or leave probe well-visit checking out of scope.

---

## 4. Build & validation sequence

### 4a. Non-destructive Stage-2 verification (do this first, before trusting any full build)

`corner-maze-build-dataset` re-reads *every* per-session parquet and rewrites the whole `data/yoked/dataset/` dir — so it touches the existing 3 action tables + `sessions.parquet` even though our edits change no Acquisition/Exposure code path. We must **prove** they come out unchanged, not assume it:

1. Checksum the current dataset dir (`sha256` per file) and snapshot row counts **before any edit**.
2. Build **only** the new phases at Stage 1: `corner-maze-build-yoked --phase 'Novel Route'` and `--phase 'Reversal'`. This writes *new* per-session parquets and leaves every existing per-session parquet byte-identical (existing phases are never re-processed).
3. Run `corner-maze-build-dataset`, then re-checksum. **Required outcome:** `actions_synthetic_pretrial.parquet`, `actions_real_pretrial.parquet`, `actions_exposure.parquet`, and `subjects.parquet` are **bit-for-bit identical**; `sessions.parquet` differs only by the 82 appended rows (existing rows unchanged — diff on the pre-existing session_ids must be empty). If any existing action table changes, **stop** — the pipeline has drifted and the additive guarantee is violated; investigate before shipping.
4. As a belt-and-suspenders option, keep a copy of the pre-change dataset dir so any regression is a one-command restore.

### 4b. Main sequence

1. **Land §3a–§3d edits.**
2. **Smoke: one rat per (group × phase) combination** — pick representatives covering all 9 mapping rows. Run `CORNER_MAZE_ANALYSIS_DIR=/Users/ryangrgurich/Code/corner-maze/corner-maze-analysis/data/processed corner-maze-build-yoked --subject <ID> --phase 'Novel Route'` (and `Reversal`). Confirm per-session parquets land in `data/yoked/` without a `ValueError: Unmapped paradigm`.
3. **Divergence check** each smoke session via `diagnostics/check_divergence.py` + `check_path_fidelity.py`. Target: 0 phantoms, path fidelity OK — same bar as the current regression baseline (`project_yoking_clean_state_2026_05_21`). **Prioritize the `Rotate Detour` / `Rotate Detour Moving` pair** (§1 watch item).
4. **Full build:** `corner-maze-build-yoked --all --phase 'Novel Route'` then `--phase 'Reversal'` across the 41 (scope filter from §3b restricts automatically), then `corner-maze-build-dataset`.
5. **Assert post-build:** `sessions.parquet` gains 82 rows (41 NR + 41 Rev); `actions_novel_route.parquet` + `actions_reversal.parquet` exist; every action-table `session_id` joins to `sessions.parquet` with the right phase; every session ends on a rewarded PICKUP.
6. **Roster guard:** confirm no out-of-scope subject (CM042, f1 cohort, the 8 †-excluded) leaked in — the pipeline can build them on demand, but this ship must be the 41.

---

## 5. Risks / open questions

| # | Risk | Mitigation |
|---|---|---|
| R1 | `Rotate Detour` vs `Rotate Detour Moving` map to *fixed* vs *rotate* — the "Moving" cue may not be modeled identically by the env's rotate generator. | Divergence-check one rat of each first (step 3). If "Moving" diverges, investigate the env's cue dynamics for `VC novel route rotate`; **fix the yoking/remap, not the env**. |
| R2 | Probe sessions have a **pre-probe (trials 1–16) + probe (17+)** structure; the non-exposure build branch was written for Acquisition. Trial-boundary / reward extraction may need probe-aware handling. | Inspect `get_phase_coordinates` + `get_trial_rewards` output for one probe session before full build; compare trial count to upstream `trials.parquet`. |
| R3 | Terminal-rewarded-PICKUP invariant (`build_dataset.py:231`) may fail if a probe rat's last trial ends on a wrong well. | Expected `build_action_sequence` truncation drops the tail; log any session whose row count shrinks vs. upstream and confirm it still ends rewarded. |
| R4 | `CORNER_MAZE_ANALYSIS_DIR` is currently **unset**; `data/analysis/` does not exist in-repo. | Export it to `/Users/ryangrgurich/Code/corner-maze/corner-maze-analysis/data/processed` for every build command (documented in README quickstart). |
| R5 | Downstream consumers (training runner, notebooks, `data/canonical.py`) assume only 3 action tables. | **Not touched by this work — deliberately.** Adding two new tables cannot break a consumer that only reads the existing three (additive). This row exists purely to record that those consumers are unaware of the new tables; wiring them up is a *separate, future* decision, not part of this pass. |

---

## 6. Out of scope (explicit)

- PI+VC_f1 probe sessions (`Fixed Cue Novel Route Twist`, `Fixed Cue Switch Twist`) — deferred; would add 2 more PARADIGM_MAP entries + the 6 f1 rats.
- CM042 (VC) probe sessions.
- Rotation / No Cue phases (24 + 24 upstream) — remain out of scope.
- VC_DREADDs — dropped group, unchanged.
- Training-side / notebook integration of the new tables — separate follow-up (see R5).

---

## 7. Effort summary

Net-new code is small and additive: **9 dict entries + ~5 filter/bucket edits across 3 files**, no env change, no new generator. The real work is **validation** — divergence + path-fidelity on the 9 mapping rows, with the `Rotate Detour Moving` row and the pre-probe trial structure as the two things most likely to need iteration.
