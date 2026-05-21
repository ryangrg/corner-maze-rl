# corner-maze-rl

Decision Transformer + PPO + Successor-Representation baselines on a 13×13 corner-maze MiniGrid environment, trained against yoked rodent behavior. Built for student use via Colab and locally in VS Code.

> **Status:** planning phase, pre-build. Design doc in [md/dt-repo-plan.md](md/dt-repo-plan.md). The legacy research repo with the original env, yoking pipeline, and prior experiments lives at [corner-maze-rl-legacy](https://github.com/ryangrg/corner-maze-rl-legacy) — code and data will be ported in per the manifest in §16 of the design doc.

## What this is

A teaching repo for offline / behavior-cloning-style RL on a real neuroscience task:

- **Environment** — 13×13 MiniGrid corner maze with structured trial phases (exposure → pretrial → trial → ITI), four wells, four cues, configurable session paradigms (PI, VC, PI+VC).
- **Yoked dataset** — action sequences derived from real rodent behavioral tracking: **531 sessions across 56 subjects in 4 training groups**, with env-derived per-step rewards and per-trial return-to-go. See [Subjects and yoked data](#subjects-and-yoked-data) below for the breakdown.
- **Models** — Decision Transformer (the headline model), plus PPO and SR baselines for comparison.
- **Encoders** — composable state encoders (grid-cell pose vectors, pretrained visual CNN, one-hot tabular, reward-history) all standardized to 60D.
- **Eval protocol** — IQM + stratified bootstrap CIs, drawdown reliability, performance profiles, kill-switch on flat learning curves. Grounded in the empirical-RL methodology canon ([Henderson 2018], [Agarwal 2021], [Patterson 2024]).

## Subjects and yoked data

> **Manuscript scope.** The canonical roster for yoking and training is **48 subjects** — the rats whose behavior will be published in the forthcoming corner-maze behavioral manuscript. The yoked dataset currently contains 56 subjects; 8 (marked **†** below) are present from earlier yoking runs but are **out of manuscript scope** and should not be used for new training runs, eval cells, or published comparisons. See [Manuscript subject roster](#manuscript-subject-roster) below.

The yoked dataset lives at `data/yoked/dataset/` (gitignored; populated via `scripts/setup_data.sh` or rebuilt from upstream with `corner-maze-build-dataset`). It currently contains:

| Training group | Subjects | Primary acquisition session_type | Acquisition sessions | Exposure sessions |
|---|---:|---|---:|---:|
| **PI** (path-integration only) | 15 | Dark Train (no visual cue) | 121* | 29 |
| **PI+VC** (place + visual cue) | 17 | Fixed Cue 1 (stable cue) | 118 | 32 |
| **PI+VC_f1** (f1-generation cohort) | 7 | Fixed Cue 1 Twist | 68 | 14 |
| **VC** (visual cue only) | 17 | Rotate Train (cue rotates) | 115 | 34 |
| **Total** | **56** | | **422** | **109** |

<sup>*4 of the 121 PI sessions are tagged `Fixed Cue 1` rather than `Dark Train` — a per-subject experimental detour, not a separate paradigm.</sup>

The dataset ships three action tables (one row per env step, schema `session_id, step, action, grid_x, grid_y, direction, rewarded, actions_to_reward, pose_label`). `pose_label` (`layout_class_x_y_dir`, e.g. `trl_e_s_xx_8_2_0`) joins to [data/dataframes/minigrid-views-allposes.parquet](data/dataframes/minigrid-views-allposes.parquet) to fetch the pre-rendered 21×21×3 RGB view for that step without re-rendering through MiniGrid:

- `actions_synthetic_pretrial.parquet` — Acquisition only, synthetic pretrial (the primary input to training; 422 sessions, ~768 K rows)
- `actions_real_pretrial.parquet` — Acquisition only, real pretrial (alt variant for ablations; 236 sessions, ~612 K rows)
- `actions_exposure.parquet` — Exposure phase only, no pretrial concept (109 sessions, ~266 K rows)

Plus `subjects.parquet` (56 rows) and `sessions.parquet` (531 rows: 422 Acquisition + 109 Exposure). Schema details in [src/corner_maze_rl/data/load.py](src/corner_maze_rl/data/load.py).

### Subject roster

Pass any of these to the runner via `--subject <name>`. Format: `name (subject_id, AcquisitionSessions + ExposureSessions)`. Subject IDs are upstream join keys; you don't usually pass them directly.

**PI** (15 subjects, ids 67–98): CM023 (67, 6A+2E), CM024 (69, 6A+2E), CM027 (68, 7A+2E), CM030**†** (74, 13A+2E), CM032**†** (77, 20A+2E), CM033**†** (79, 19A+2E), CM036 (78, 12A+2E), CM037 (80, 4A+2E), CM046 (90, 6A+2E), CM049 (93, 5A+2E), CM050 (94, 3A+2E), CM051 (95, 5A+2E), CM052 (96, 5A+2E), CM053 (97, 5A+1E), CM054 (98, 5A+2E)

**PI+VC** (17 subjects, ids 47–63): CM000 (47, 7A+2E), CM001 (48, 6A+2E), CM002 (49, 7A+1E), CM003 (50, 8A+2E), CM004 (51, 8A+2E), CM005 (52, 6A+2E), CM006 (53, 3A+2E), CM007 (54, 9A+2E), **CM008\*** (55, 3A+1E), CM009 (56, 7A+2E), CM010 (57, 5A+2E), CM011 (58, 5A+2E), CM014 (59, 10A+2E), CM015 (61, 11A+2E), CM016 (63, 5A+2E), CM017 (60, 9A+2E), CM018 (62, 9A+2E)

**PI+VC_f1** (7 subjects, ids 123–130): CM057 (123, 22A+2E), CM058 (124, 8A+2E), CM059**†** (125, 12A+2E), CM060 (126, 5A+2E), CM061 (127, 3A+2E), CM063 (129, 7A+2E), CM064 (130, 11A+2E)

**VC** (17 subjects, ids 70–100): CM025 (71, 7A+2E), CM026 (73, 5A+2E), CM028**†** (70, 9A+2E), CM031 (75, 9A+2E), CM034**†** (81, 9A+2E), CM035 (76, 8A+2E), CM038 (82, 6A+2E), CM039**†** (83, 5A+2E), CM040 (84, 7A+2E), CM041 (85, 6A+2E), CM042 (86, 5A+2E), CM043 (87, 6A+2E), CM044 (88, 5A+2E), CM045 (89, 7A+2E), CM048**†** (92, 6A+2E), CM055 (99, 8A+2E), CM056 (100, 7A+2E)

<sup>Subjects with `1E` rather than `2E` reflect upstream reality (only one Exposure session was recorded), except **CM008\***, where the second Exposure session exists upstream but is missing coordinate tracking — see [Known data gaps](#known-data-gaps) below. Subject IDs are interleaved across groups because they reflect experimental-cohort order, not group membership. Subjects marked **†** are present in the dataset but **out of manuscript scope** — see [Manuscript subject roster](#manuscript-subject-roster) below.</sup>

### Manuscript subject roster

The forthcoming behavioral manuscript publishes results on **48 of the 56 yoked subjects**. New training runs, default rosters, and any published eval comparisons should use only these 48; the other 8 (marked **†** above) are kept in the dataset for completeness but excluded from the manuscript scope.

| Training group | Manuscript subjects | Count | Excluded (in dataset, not in manuscript) |
|---|---|---:|---|
| **PI** | CM023, CM024, CM027, CM036, CM037, CM046, CM049, CM050, CM051, CM052, CM053, CM054 | 12 | CM030, CM032, CM033 |
| **PI+VC** | CM000, CM001, CM002, CM003, CM004, CM005, CM006, CM007, CM008, CM009, CM010, CM011, CM014, CM015, CM016, CM017, CM018 | 17 | *(none)* |
| **PI+VC_f1** | CM057, CM058, CM060, CM061, CM063, CM064 | 6 | CM059 |
| **VC** | CM025, CM026, CM031, CM035, CM038, CM040, CM041, CM042, CM043, CM044, CM045, CM055, CM056 | 13 | CM028, CM034, CM039, CM048 |
| **Total** | | **48** | **8** |

When rebuilding `data/yoked/dataset/` from upstream for student release, restrict to these 48 subjects. The yoking pipeline can still build the excluded 8 on demand (`corner-maze-build-yoked --subject <ID>`); they're simply not part of the default ship.

### Known data gaps

- **CM008 1e (Exposure)** — missing. Upstream coordinate tracking failed for this session; the source `.avi` exists but was never auto-tracked. Re-running the upstream tracker in `corner-maze-analysis` would close the gap; until then, exposure coverage is 109/110 sessions.
- **Other behavioral phases** — Reversal, Novel Route, Rotation, and No Cue sessions exist upstream but are intentionally out of scope for this dataset; only Acquisition + Exposure are yoked.

The yoking pipeline that produced these tables lives in [src/corner_maze_rl/yoking/](src/corner_maze_rl/yoking/); to extend coverage to additional sessions or subjects, run `corner-maze-build-yoked --subject <ID> --phase <Acquisition|Exposure>` and rebuild with `corner-maze-build-dataset` (requires `CORNER_MAZE_ANALYSIS_DIR` pointing at the upstream behavioral parquets).

## Quickstart

> Build is in progress. The commands below describe the *target* student workflow; not all are wired up yet.

### Local + VS Code (recommended)
```bash
git clone https://github.com/ryangrg/corner-maze-rl.git
cd corner-maze-rl
# point VS Code at your existing ai-venv, or:
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Colab-in-browser (limited)
The interactive `notebooks/` UI is **VS Code / local Jupyter only** — Colab's iframe-sandboxed widget framework breaks rendering and keyboard input. For non-interactive components (training scripts, dataset builds, evaluation), Colab is fine:
```python
!pip install git+https://github.com/ryangrg/corner-maze-rl.git
from corner_maze_rl import ...
```
If you need cloud + the interactive UI, point VS Code at a remote Jupyter kernel (e.g. GitHub Codespaces or a managed runtime) — the IDE renders widgets locally even when compute is remote.

## Student workspace

Canonical notebooks (`notebooks/01_*`, `02_*`, `03_*`, ...) are the reference implementations and stay clean. For your own tweaks, copy a notebook into `notebooks/students/<your-name>/` and modify freely:

```bash
mkdir -p notebooks/students/alice
cp notebooks/03_ppo_experiments.ipynb notebooks/students/alice/03_ppo_lr_sweep.ipynb
```

`notebooks/students/<name>/` is gitignored — your scratch work never collides on `git pull`, and the canonical notebook stays a stable reference. Run outputs go to `runs/` (also gitignored), so different students' runs don't conflict either.

**Sharing results.** When you want feedback on a notebook, either (a) commit a copy into `notebooks/contrib/<your-name>/` (not gitignored) and open a PR, or (b) work on a fork. The split keeps "private scratch" and "ready for review" distinct.

A good first run: open [notebooks/03_ppo_experiments.ipynb](notebooks/03_ppo_experiments.ipynb), keep the smoke-run defaults (`N_SEEDS=2, NUM_EPISODES=10`), and execute top-to-bottom. The §7 recipes table shows how to scale up.

## Repository layout

```
corner-maze-rl/
├── README.md
├── md/                        # design + spec docs (start here)
│   ├── dt-repo-plan.md        # full design doc — start here
│   ├── environment-architecture.md
│   ├── maze-behavior-spec.md  # 2S2C task rules
│   ├── reward-structure-analysis.md
│   └── sr-yoked-negative-results.md
├── src/corner_maze_rl/        # (to be built — see plan §3)
├── notebooks/                 # interactive notebooks (VS Code recommended)
├── data/                      # (gitignored; setup script will populate)
└── LICENSE
```

## Where to start reading

1. [md/dt-repo-plan.md](md/dt-repo-plan.md) — the full design doc. Sections 1–3 give the goals and layout in five minutes; §4 (reward + RTG), §5 (encoders), §8 (kill switch), and §10 (eval protocol) are the parts that distinguish this from a textbook DT.
2. [md/maze-behavior-spec.md](md/maze-behavior-spec.md) — task rules: arm structure, trial phases, turn detection, well-visit, reward triggering.
3. [md/environment-architecture.md](md/environment-architecture.md) — env spec the code is built against.
4. [md/reward-structure-analysis.md](md/reward-structure-analysis.md) — *why* the reward shaping is what it is.
5. [md/sr-yoked-negative-results.md](md/sr-yoked-negative-results.md) — a documented negative result (SR fails on yoked data); the plan's §10.3 frames how to report findings like this scientifically.

## Related projects

- [corner-maze-rl-legacy](https://github.com/ryangrg/corner-maze-rl-legacy) — the prior research repo. Source of the env, yoking pipeline, yoked dataset, and prior PPO/SR experiments. This repo (`corner-maze-rl`) ports/refactors content from it per the manifest in [md/dt-repo-plan.md](md/dt-repo-plan.md) §16.

## License

See [LICENSE](LICENSE).
