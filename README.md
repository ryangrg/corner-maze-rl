# corner-maze-rl

Decision Transformer + PPO + Successor-Representation baselines on a 13×13 corner-maze MiniGrid environment, trained against yoked rodent behavior. Built for student use via Colab and locally in VS Code.

> **Status:** Decision Transformer pipeline is running end-to-end (training + acquisition gate + probes + diagnostics) in [notebooks/10_vanilla_dt.ipynb](notebooks/10_vanilla_dt.ipynb). PPO baseline is wired up in [notebooks/03A_ppo_minigrid_cnn.ipynb](notebooks/03A_ppo_minigrid_cnn.ipynb). All training/analysis notebooks ship with a one-click **Open in Colab** bootstrap (repo clone, package install, GPU runtime, Drive-backed run artifacts). Design doc: [md/dt-repo-plan.md](md/dt-repo-plan.md). Legacy research repo: [corner-maze-rl-legacy](https://github.com/ryangrg/corner-maze-rl-legacy).

## What this is

A teaching repo for offline / behavior-cloning-style RL on a real neuroscience task:

- **Environment** — 13×13 MiniGrid corner maze with structured trial phases (exposure → pretrial → trial → ITI), four wells, four cues, configurable session paradigms (PI, VC, PI+VC).
- **Yoked dataset** — action sequences derived from real rodent behavioral tracking: **531 sessions across 56 subjects in 4 training groups**, with env-derived per-step rewards and per-trial return-to-go. See [Subjects and yoked data](#subjects-and-yoked-data) below for the breakdown.
- **Models** — Decision Transformer (the headline model; vanilla kzl-faithful port + linear-attention and full-context variants) plus PPO and SR baselines for comparison. Source in [src/corner_maze_rl/models/](src/corner_maze_rl/models/).
- **Encoders** — composable state encoders (grid-cell pose vectors, pretrained visual CNN, one-hot tabular, reward-history) all standardized to 60D.
- **Eval protocol** — IQM + stratified bootstrap CIs, drawdown reliability, performance profiles, kill-switch on flat learning curves. Grounded in the empirical-RL methodology canon ([Henderson 2018], [Agarwal 2021], [Patterson 2024]).

## Subjects and yoked data

> **Manuscript scope.** The canonical roster for yoking and training is **48 subjects** — the rats whose behavior will be published in the forthcoming corner-maze behavioral manuscript. The yoked dataset currently contains 56 subjects; 8 (marked **†** below) are present from earlier yoking runs but are **out of manuscript scope** and should not be used for new training runs, eval cells, or published comparisons. See [Manuscript subject roster](#manuscript-subject-roster) below.

The yoked dataset lives at `data/yoked/dataset/` (gitignored; populated by running `corner-maze-build-dataset` against the upstream behavioral repo). It currently contains:

| Training group | Subjects | Primary acquisition session_type | Acquisition sessions | Exposure sessions |
|---|---:|---|---:|---:|
| **PI** (path-integration only) | 15 | Dark Train (no visual cue) | 121* | 29 |
| **PI+VC** (place + visual cue) | 17 | Fixed Cue 1 (stable cue) | 118 | 32 |
| **PI+VC_f1** (f1-generation cohort) | 7 | Fixed Cue 1 Twist | 68 | 14 |
| **VC** (visual cue only) | 17 | Rotate Train (cue rotates) | 115 | 34 |
| **Total** | **56** | | **422** | **109** |

<sup>*4 of the 121 PI sessions are tagged `Fixed Cue 1` rather than `Dark Train` — a per-subject experimental detour, not a separate paradigm.</sup>

The dataset ships three action tables (one row per env step, schema `session_id, step, action, grid_x, grid_y, direction, rewarded, actions_to_reward, pose_label`). `pose_label` (`layout_class_x_y_dir`, e.g. `trl_e_s_xx_8_2_0`) joins to [data/lookups/minigrid_views.npz](data/lookups/minigrid_views.npz) to fetch the pre-rendered 21×21×3 RGB view for that step without re-rendering through MiniGrid:

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

### Colab (one-click, recommended for training)

Every training and analysis notebook in [notebooks/](notebooks/) starts with an **Open in Colab** badge and a self-contained bootstrap cell. Click the badge, pick a GPU runtime (`Runtime → Change runtime type → T4 / L4 / A100`), and `Run all`. The bootstrap:

- Clones this repo to `/content/corner-maze-rl` and installs it with `pip install -e .`.
- Mounts Google Drive (one-click consent) and starts a background daemon that rsyncs `data/runs/` → `corner-maze-rl-colab/runs/` on Drive every ~120 s. Drive disconnects are tolerated — sync retries silently.
- All data inputs (lookups + yoked dataset) come with the cloned repo. No upload step.

Recommended entrypoints:

| Notebook | What it does | Runtime |
|---|---|---|
| [10_vanilla_dt.ipynb](notebooks/10_vanilla_dt.ipynb) | **Headline DT pipeline** — train → acq gate → probes → rliable plots → attention diagnostics. | T4 fine for smoke / iteration; L4 or A100 for headline runs. |
| [10a_optimized.ipynb](notebooks/10a_optimized.ipynb) | Speed-tuned variant of `10_vanilla_dt` (bf16, `torch.compile`, TF32). | L4 / A100. |
| [09_hybrid_fullctx_linearattn_shortvanilladt.ipynb](notebooks/09_hybrid_fullctx_linearattn_shortvanilladt.ipynb), [08_fullctx_linearattn.ipynb](notebooks/08_fullctx_linearattn.ipynb) | Linear-attention DT variants. | T4 / L4. |
| [03A_ppo_minigrid_cnn.ipynb](notebooks/03A_ppo_minigrid_cnn.ipynb) | PPO baseline (MaskablePPO + CNN over MiniGrid views). | T4. |
| [01_explore_env.ipynb](notebooks/01_explore_env.ipynb), [02_explore_yoked_data.ipynb](notebooks/02_explore_yoked_data.ipynb) | Env walk-through + yoked-data tour. | CPU. |

Note: `01_explore_env.ipynb`'s interactive manual-control UI is VS Code / local Jupyter only — Colab's iframe-sandboxed widget framework breaks keyboard input and ipywidgets rendering. All non-interactive cells (env init, visualization, episode replay) run fine on Colab.

### Local + VS Code

```bash
git clone https://github.com/ryangrg/corner-maze-rl.git
cd corner-maze-rl
# point VS Code at your existing ai-venv, or:
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .
```

The same notebooks run locally — the Colab bootstrap cell is a no-op when not on Colab. Training writes to `data/runs/` under the repo root.

## Student workspace

Canonical notebooks (`notebooks/01_*`, `02_*`, `03_*`, ...) are the reference implementations and stay clean. For your own tweaks, copy a notebook into `notebooks/students/<your-name>/` and modify freely:

```bash
mkdir -p notebooks/students/alice
cp notebooks/03_ppo_experiments.ipynb notebooks/students/alice/03_ppo_lr_sweep.ipynb
```

`notebooks/students/<name>/` is gitignored — your scratch work never collides on `git pull`, and the canonical notebook stays a stable reference. Run outputs go to `runs/` (also gitignored), so different students' runs don't conflict either.

**Sharing results.** When you want feedback on a notebook, either (a) commit a copy into `notebooks/contrib/<your-name>/` (not gitignored) and open a PR, or (b) work on a fork. The split keeps "private scratch" and "ready for review" distinct.

A good first run: open [notebooks/10_vanilla_dt.ipynb](notebooks/10_vanilla_dt.ipynb), apply the **Smoke** recipe from the table at the bottom of the notebook (`N_RUNS=1, N_EPOCHS=2`), and execute top-to-bottom. Confirms the pipeline runs end-to-end in ~3 min on a T4. Then scale up via the **Iteration** or **Headline** recipe.

## Repository layout

```
corner-maze-rl/
├── README.md
├── md/                          # design + spec docs (start here)
│   ├── dt-repo-plan.md          # full design doc
│   ├── environment-architecture.md
│   ├── maze-behavior-spec.md    # 2S2C task rules
│   ├── reward-structure-analysis.md
│   └── sr-yoked-negative-results.md
├── docs/                        # rendered HTML walkthroughs (10_vanilla_dt.html, ...)
├── src/corner_maze_rl/
│   ├── env/                     # CornerMazeEnv (MiniGrid)
│   ├── models/                  # decision_transformer{,_decoupled_dimension}.py,
│   │                            #   linear_decision_transformer*.py, ppo.py, sr.py
│   ├── encoders/                # grid-cell / pose-visual / image-CNN state encoders (60D)
│   ├── data/                    # yoked dataset loaders, RTG, rotation to canonical frame
│   ├── train/                   # runner, loop, kill_switch, SB3 callbacks
│   ├── eval/                    # IQM + rliable bootstrap utilities
│   ├── yoking/                  # behavior → MiniGrid action sequence pipeline
│   ├── scripts/                 # CLI entrypoints (corner-maze-build-yoked, ...)
│   └── utils/                   # run_io, seeding
├── notebooks/                   # all ship with Colab bootstrap cells
├── data/                        # gitignored — lookups + yoked dataset + runs
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
