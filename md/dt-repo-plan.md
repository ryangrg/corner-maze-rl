# Corner-Maze-RL Repo Plan

**Status:** draft, pre-build
**Last updated:** 2026-05-05 (ported into new `corner-maze-rl` repo)
**Purpose:** standalone repo combining the yoked-data pipeline, the maze environment, and a Decision Transformer + plug-in PPO/SR baselines, for student use via Colab and VS Code.

**Companion docs in this repo:**
- [environment-architecture.md](environment-architecture.md) — env spec the new code is built against
- [maze-behavior-spec.md](maze-behavior-spec.md) — 2S2C task rules
- [reward-structure-analysis.md](reward-structure-analysis.md) — reward shaping rationale
- [sr-yoked-negative-results.md](sr-yoked-negative-results.md) — known SR failure case

**Legacy repo (source of code/data to port):** `https://github.com/ryangrg/corner-maze-rl-legacy`. See §16 for the file-by-file source manifest.

---

## 1. Goals

1. Single-file-style implementations of **DT**, **PPO**, **SR** that share an env, an encoder layer, and a run-management layer.
2. Decision Transformer trained on yoked rat behavior with **real env-derived rewards** (not the "session-progress" RTG in the current notebook).
3. Pluggable **state encoders** (grid-cells, visual-CNN, one-hot tabular, reward-history, egocentric image) that can be combined.
4. **Pip-installable from GitHub** so Colab notebooks are thin (`pip install git+https://...`; no sys.path hacks).
5. Reproducible run-saving with seeds, configs, and per-step trajectories.
6. **Compute-aware kill switch** — terminate a run early if the learning curve is provably flat, but let "creeping" runs continue.
7. Student-friendly: clone-and-run locally in VS Code, *or* Colab-only, with the same code.
8. Evaluation protocol grounded in the empirical-RL methodology canon (IQM, stratified bootstrap, drawdown — see [LLM Wiki references](#10-evaluation-protocol)).

## 2. Non-goals

- Changing maze behavior design (sparse +1/trial scoring is fixed).
- Building a new yoking pipeline from scratch (the mature legacy pipeline was ported to `src/corner_maze_rl/yoking/`; extensions land there).
- Multi-rat training (one rat per experiment, ≤ 80 sessions).
- Retraining the visual CNN by default (ship pretrained; allow on-the-fly retrain as advanced option).

## 3. Repo layout

```
corner-maze-rl/
├── README.md                            # students: install + 5-min quickstart
├── pyproject.toml                       # pip-installable package
├── notebooks/                           # VS Code / local Jupyter — NOT Colab-in-browser
│   ├── 01_explore_env.ipynb             # env walkthrough, manual control
│   ├── 02_explore_yoked_data.ipynb      # load dataset, plot trajectories
│   ├── 03_ppo_experiments.ipynb         # PPO baseline (sweepable encoder presets, N seeds)
│   ├── 04_train_dt.ipynb                # DT train + eval (+ 05–09 DT architecture variants)
│   ├── 05_train_sr.ipynb                # SR baseline
│   ├── 06_compare_models.ipynb          # IQM, performance profiles, drawdown (across PPO/DT/SR runs)
│   └── students/                        # gitignored per-student copies — see README §Student workspace
├── src/corner_maze_rl/
│   ├── env/                             # trimmed env + constants + trial_seq
│   ├── data/
│   │   ├── load.py                      # DuckDB queries against 3-table dataset
│   │   ├── canonical.py                 # cue→north rotation utilities
│   │   └── session_types.py             # 4 hardcoded paradigms; PI / VC / PI+VC / PI+VC_f1
│   ├── encoders/
│   │   ├── base.py                      # StateEncoder Protocol, CompositeEncoder
│   │   ├── grid_cells.py                # ported from DatasetBuilder.ipynb cell 1 (deterministic)
│   │   ├── visual_cnn.py                # pretrained 60D CNN; load + optional retrain
│   │   ├── one_hot_pose.py              # 196-dim baseline
│   │   ├── reward_history.py            # 4×n_wm decaying timers (from custom_rl.py)
│   │   ├── egocentric_image.py          # raw 21×21×3 MiniGrid view (for CnnPolicy)
│   │   └── registry.py                  # build_encoder(["grid_cells", "visual_cnn"]) → composite
│   ├── models/
│   │   ├── base.py                      # TrainableAgent Protocol
│   │   ├── decision_transformer.py
│   │   ├── ppo.py                       # adapted from custom_rl.py PPOAgent
│   │   └── sr.py                        # adapted from custom_rl.py SRAgent
│   ├── train/
│   │   ├── runner.py                    # adapted from session_runner.py
│   │   ├── kill_switch.py               # learning-curve early termination
│   │   └── callbacks.py                 # logging, checkpointing
│   ├── eval/
│   │   ├── rollout.py                   # online rollout in env
│   │   ├── metrics.py                   # IQM, drawdown, success rate
│   │   ├── viz.py                       # learning curves, performance profiles
│   │   └── replay.py                    # play back yoked or model trajectories (no video)
│   └── utils/
│       ├── run_io.py                    # set_global_seed, save_run_config (port from seed_utils.py)
│       └── git.py                       # capture commit SHA in run_config
├── scripts/
│   ├── add_actions_to_reward.py         # one-shot: add actions_to_reward column to existing dataset
│   └── train.py                         # CLI: --model {dt,ppo,sr} --session-type ... --seeds N
├── data/                                # gitignored; setup script downloads
└── tests/
    ├── test_env_replay_determinism.py
    ├── test_returns_correctness.py
    ├── test_encoders_dim.py
    └── test_kill_switch.py
```

## 4. Reward & return-to-go (the core integration)

### 4.1 Behavior recap (do not change)
- Every trial ends in either reward or stuck (no time-out kill of trials).
- Wrong well **does not** end a trial; only correct well does.
- Per-trial score: +1 if rat completed the trial without ever visiting a non-rewarded well, else 0. Max 32/session.
- **The agent does not see this score.** The agent sees only the env's `step()` reward, which is currently (matching `env/_compute_reward`):
  - `STEP_FORWARD_COST = -0.0005` (applied to actions 2, 3, 4)
  - `STEP_TURN_COST = -0.001` (applied to actions 0, 1; 2× forward to discourage spinning)
  - `WELL_REWARD_SCR = +1.061` on correct well (added on top of the step cost)
  - 0 on empty/wrong well (no explicit penalty in `_compute_reward`; just the base step cost)
- Score is for human evaluation, not policy learning.

### 4.2 Persisted structural countdown: `actions_to_reward`

The yoked dataset stores **one int32 column** per row, `actions_to_reward`, in every `actions_*.parquet`. It is the count of action steps remaining until the next correct PICKUP (`action == 3 & rewarded == 1`); zero at the PICKUP itself. Purely structural — derivable from `(action, rewarded)`, identical across all cost configs.

| Column | Description |
|--------|-------------|
| `actions_to_reward` | int32, count of remaining steps until the next correct PICKUP. Zero at the rewarded PICKUP. |

Wrong-well PICKUPs (action 3, rewarded 0) do not close a window — the counter keeps decrementing through them.

**Window definition (ITI-start, per-trial).** Window `i` covers `[step_after_PICKUP_{i-1}+1, PICKUP_i]` (inclusive of `PICKUP_i`), with window 0 starting at session step 0. Post-PICKUP exit actions (the LL/RR/F block emitted by `well_visit_actions`) fall into the *next* window — that's why ITI navigation earns credit toward the upcoming trial's reward.

The column is computed once in `corner_maze_rl.yoking.build_dataset.compute_actions_to_reward` at aggregation time. The invariants — `actions_to_reward[-1] == 0` per session, `actions_to_reward == 0 ⇔ correct PICKUP` per row — are locked by `tests/test_yoked_dataset_invariants.py`.

### 4.2.1 Deferred-cost design: magnitudes at token-gen time

Per-step `reward` and per-trial RTG are **not** persisted. They are derived at token-generation time from a `CostConfig`:

```python
@dataclass
class CostConfig:
    forward_cost: float = -0.0005   # matches env/constants.py
    turn_cost:    float = -0.001
    pause_cost:   float = -0.0005   # _compute_reward charges pause as forward
    well_reward:  float = +1.061
    well_empty:   float =  0.0      # env has no empty-well penalty
```

Cost sweeps become a config edit — no parquet rebuild required. The token-gen step walks each session's `(action, rewarded)` columns, applies per-action costs from the config, computes a per-step `reward`, and (if the DT formulation requires it) a float-RTG suffix sum within each `actions_to_reward`-defined window. `actions_to_reward` itself may also be fed into the model as a positional conditioning signal — its structural invariance under cost sweeps is the point.

Boundary detection: window starts are exactly where the previous row's `actions_to_reward` is 0; window ends are exactly where the current row's `actions_to_reward` is 0. No env replay needed.

### 4.3 Sparse-reward acknowledgement
RTG signal is dominated by the +1.061 spike at trial end. Successful trials produce strongly positive RTG curves; failed trials produce flat slightly-negative RTG. DT learns to discriminate these regimes but credit assignment within a successful trial is coarse. We document this honestly and treat it as a teaching opportunity ("this is *why* DT papers use dense rewards").

### 4.4 Inference-time RTG (the part students find counterintuitive)
At rollout, the student specifies a *target* return; RTG decrements as rewards are observed:

```python
target_return = 5.0  # student-set
for t in range(max_steps):
    context.append((target_return, encode(obs), prev_action))
    action = dt(context[-K:]).argmax()
    obs, reward, done, _ = env.step(action)
    target_return -= reward
    if done: break
```

The eval notebook ships a slider over target_return so students can see the optimism→imitation knob in action.

## 5. Encoders (the composability layer)

### 5.1 Protocol
```python
class StateEncoder(Protocol):
    output_dim: int
    def encode(self, layout: str, x: int, y: int, direction: int) -> np.ndarray: ...
```

### 5.2 Provided encoders (each defaults to 60D when applicable)

| Name | Dim | Source | Trainable? | Notes |
|------|-----|--------|-----------|-------|
| `grid_cells` | 60 | port of [notebooks/DatasetBuilder.ipynb](../notebooks/DatasetBuilder.ipynb) cell 1 | No (fixed neural code) | 5 modules × 12 phase bins, von Mises directional tuning. |
| `visual_cnn` | 60 | port of [notebooks/Minigrid_Maze_Yoked_Training.ipynb](../notebooks/Minigrid_Maze_Yoked_Training.ipynb) cells 11–12 | Pretrained ships; optional retrain | Local-view PNGs → 60D latent. |
| `one_hot_pose` | 196 | new | No | 196-tile one-hot (matches existing tabular setup). |
| `reward_history` | 4·n_wm_units (40 default) | port from legacy `src/rl/custom_rl.py::generate_state_vector()` | No | Decaying timers per well; injects partial observability. `n_wm_units = 10` in legacy (configurable); shared by tabular PPO and SR paths — *not* SR-only despite the working-memory framing. |
| `egocentric_image` | 21×21×3 | env's native obs | No | For CnnPolicy variants. |

### 5.3 Composition
```python
encoder = build_encoder(["grid_cells", "visual_cnn"])   # → 120-dim composite
encoder = build_encoder(["one_hot_pose", "reward_history"])   # → 236-dim, matches your tabular runs
```

The DT's state-token Linear is sized to `encoder.output_dim`. PPO/SR consume the same vector. **One encoder definition drives all three model types**, which is the clean ablation surface students need.

### 5.4 Tabular vs. egocentric switch
A first-class config flag:
```yaml
observation_mode: "vector"    # uses encoder registry, runs MLP/transformer
# or
observation_mode: "image"     # raw 21×21×3, runs CnnPolicy
```
The env already supports both modes via legacy `src/rl/sb3_agents.py`'s wrappers. We expose them through a single switch. **Egocentric image is used standalone (CnnPolicy / image branch only)** — it is not composed with the 60-D vector encoders. The `observation_mode` switch picks one or the other; encoder composition (§5.3) only applies to vector encoders.

## 6. Models

### 6.1 Plug-in protocol
```python
class TrainableAgent(Protocol):
    def train_one_session(self, env, callbacks=None) -> SessionResult: ...
    def act(self, obs) -> int: ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "TrainableAgent": ...
```

`md-files/adding_a_model.md` will document the contract. The env never changes.

### 6.2 Decision Transformer (`models/decision_transformer.py`)
Adapted from [DTtrainer.ipynb](../2S2C_task/colab/DTtrainer.ipynb) with these changes:
- Real RTG from preprocessing pass (not the fake `1 - steps_to_go/500`).
- **Drop** the `ACTION_WEIGHTS = [1,1,1,20,1]` hack.
- **Configurable positional encoding** (`pos_encoding: 'learned' | 'sinusoidal' | 'spatial' | 'none'`):
  - `learned` (default) — learned timestep embedding per token position.
  - `sinusoidal` — Vaswani-style fixed sinusoidal timestep encoding.
  - `spatial` — the notebook's quirky "add pose-vector to R/S/A". Kept for ablation parity with original implementation.
  - `none` — no positional info (instructive baseline; shows why position matters).
  - All four available out of the box; students compare them as a built-in ablation.
- 5-action space already aligned with `actions_synthetic_pretrial.parquet`; no remap.
- Variable context size, default 64.

### 6.3 PPO (`models/ppo.py`)
Adapted from legacy `src/rl/custom_rl.py::PPOAgent`. Generic over `encoder.output_dim`. Add SB3 path for students who want CnnPolicy/MaskablePPO.

### 6.4 SR (`models/sr.py`)
Adapted from legacy `src/rl/custom_rl.py::SRAgent`. Document the [SR yoked negative results](sr-yoked-negative-results.md) up front so students don't expect it to work — it's an instructive failure case (see "negative results" framing below).

## 7. Run management & saving

### 7.1 Per-run output directory
```
runs/{model}_{session_type}_{timestamp}/run_{i}/
  run_config.json        # seed, hyperparams, encoder list, session_type, git SHA, dataset hash
  model.pt               # checkpoint
  trajectory.parquet     # per-step: layout, x, y, dir, action, reward, RTG, trial_idx, trial_tag, done
  scores.parquet         # per-trial + per-session aggregates
  metrics.json           # eval results
  curves.parquet         # rolling per-session score for kill-switch + plotting
  killed_at.json         # if early-terminated, why and when
```

### 7.2 What to capture per session (canonical schema)

**Per-step:** `layout_name, x, y, direction, action, reward, return_to_go, done, trial_index, trial_phase, trial_tag, model_logits` (optional — useful for DT/PPO inspection).

**Per-trial:** `trial_index, trial_config (start_arm, cue, route, goal, tag), success, steps_to_well, error_well_visits, turn_score, target_rtg` (DT-only).

**Per-session:** `total_return, perfect_trial_count, success_rate_by_tag, mean_steps_per_trial, training_criterion_met, killed_early`.

**Run metadata:** `seed, git_sha, hyperparams, encoder_list, session_type, model_type, wall_clock_seconds, dataset_hash`.

### 7.3 Student workflow
**Recommended:** clone repo locally → `pip install -e .` → open in VS Code → connect Colab to the local folder for GPU bursts. Runs save to local `runs/` which survives Colab disconnects.

**Colab-only fallback:** `pip install git+https://github.com/<user>/corner-maze-rl.git` → mount Drive → set `RUNS_DIR=/content/drive/MyDrive/.../runs`. Same code, just a different output path.

Both paths documented in README; "VS Code + Colab kernel" is the recommended primary route.

## 8. Kill switch (the compute-aware early stop)

### 8.1 Requirements (from user)
- Kill if learning is **flat**.
- Want learning visible by ~25 sessions.
- **Don't kill if curve is creeping up**, even slowly.
- Max 80 sessions.

### 8.2 Detection logic (`train/kill_switch.py`)
Track per-session `perfect_trial_count` (0–32). After each session, evaluate:

```python
# Inputs: scores = [s_0, s_1, ..., s_n]   (per-session perfect-trial count)

WARMUP            = 10        # never kill before this
SLOPE_WINDOW      = 10        # regression window for "creeping" detection
FLAT_SLOPE_EPS    = 0.05      # trials/session — slope below this is "not learning"
ABSOLUTE_FLOOR    = 4         # mean of last window must be < this to be eligible to kill
DEAD_WINDOW       = 8         # if no successful trial in last K sessions, kill
CRITERION_MEAN    = 24        # 75% perfect (24/32) — declared learned, hand off to positive early-stop
HARD_CAP          = 80        # never run past this

if n < WARMUP:
    return CONTINUE

if all(s == 0 for s in scores[-DEAD_WINDOW:]):
    return KILL(f"dead — no successful trial in {DEAD_WINDOW} sessions")

window = scores[-SLOPE_WINDOW:]
slope = linear_regression_slope(range(len(window)), window)
recent_mean = np.mean(window)

# Kill on flat-or-declining + low-floor.
# Note: condition is `slope < FLAT_SLOPE_EPS` (asymmetric, NOT abs(slope)).
# A clearly declining curve (negative slope) should also be killed.
# A creeping-up curve has slope > FLAT_SLOPE_EPS and survives.
if slope < FLAT_SLOPE_EPS and recent_mean < ABSOLUTE_FLOOR and n >= WARMUP + SLOPE_WINDOW:
    return KILL(f"flat: slope={slope:.3f}, mean={recent_mean:.1f} after {n} sessions")

if recent_mean >= CRITERION_MEAN:
    return CRITERION_MET    # let positive-criterion early stop handle it

if n >= HARD_CAP:
    return KILL("hard cap reached")

return CONTINUE
```

**What's being detected.** The intent is "the curve isn't going up." We kill when the recent slope falls below `FLAT_SLOPE_EPS` *and* the recent mean is still below `ABSOLUTE_FLOOR`. The `ABSOLUTE_FLOOR` gate is what protects a model that has *learned* but is plateauing high (slope drops to ~0 because the curve saturated near max — that's success, not failure).

**Test-curve coverage** — the unit tests should assert correct decisions across these canonical cases:

| Curve shape | Expected decision |
|-------------|-------------------|
| All zeros for ≥ DEAD_WINDOW sessions | KILL (dead) |
| Flat at low value (e.g., constant 2 for 20+ sessions) | KILL (flat) |
| Slow creep low (2 → 5 over 15 sessions, slope ≈ 0.2) | CONTINUE |
| Fast creep low (2 → 12 over 10 sessions) | CONTINUE |
| Plateaued high (saturating near 28) | CRITERION_MET |
| Declining (e.g., 8 → 2 over 10 sessions, slope < 0) | KILL (flat — covers regression) |
| Noisy-flat zero-mean signal | KILL once WARMUP+SLOPE_WINDOW reached |
| n < WARMUP regardless of shape | CONTINUE |

### 8.3 Why these defaults
- **Linear regression on a window** is the cheapest way to test "creeping up vs. flat" while ignoring noise. A purely-flat curve has slope ≈ 0; a curve creeping from 2 → 6 over 10 sessions has slope ≈ 0.4.
- **WARMUP = 10** because variance over the first few sessions is high; killing at session 5 risks killing a slow starter.
- **ABSOLUTE_FLOOR = 4** prevents killing a model that's stuck low but technically slope-positive due to noise. Tunable.
- **DEAD_WINDOW** catches catastrophic failure (no perfect trial at all in 8 sessions).
- This logic is the *across-time* analogue of [Chan 2020]'s drawdown / detrended-dispersion metric — applied as a stopping rule, not as a metric.

### 8.4 Config exposure
All thresholds live in `train/kill_switch.py` as module constants and can be overridden per-run via `run_config.kill_switch_overrides`. Defaults are documented in the README so students understand what's terminating their runs.

### 8.5 What gets saved on kill
`killed_at.json` with: `{ "session": n, "reason": ..., "scores_at_kill": [...], "slope": ..., "recent_mean": ... }`. Not a failure — a logged decision.

## 9. Session types (4 hardcoded options)

User picks one as a CLI/config arg. Defaults to PI+VC. Each choice = a *training-group filter* + an *ordered sequence of experimental sessions* (acquisition → probes → reversal → etc.).

### 9.1 Mapping: (training_group, yoked_session_type) → env_paradigm

The yoked dataset's `session_type` column holds experiment-design names; the same yoked session_type maps to a *different env paradigm* depending on the rat's training_group:

| Yoked `session_type` | training_group=PI+VC | training_group=PI+VC_f1 | training_group=PI | training_group=VC |
|---------------------|----------------------|------------------------|-------------------|-------------------|
| `Rotate Train` | `PI+VC f2 rotate` | — | — | `VC acquisition` |
| `Fixed Cue 1` | `PI+VC f2 novel route` | — | `PI novel route cue` | `VC novel route fixed` |
| `Dark Train` | `PI+VC f2 no cue` | — | `PI acquisition` | — (TBD) |
| `Fixed Cue 1 Twist` | — | `PI+VC f1 acquisition` | — | — |

**Open data points:**
- `VC × Dark Train` — cell empty, mark TODO; skip with warning.

(2026-05-08: PI+VC_f1 × Fixed Cue 1 Twist mapping resolved — yoking pipeline produced correct action sequences all along; only the env-paradigm mapping was missing.)

Treat any unmapped (group, yoked_session_type) pair as "skip with warning" so the runner is robust to incomplete tables.

### 9.1.1 Subject selection

**One rat per training run.** The runner takes a `--subject CM###` flag; the chosen rat must belong to the chosen training group's roster (see [maze-behavior-spec.md §Subject IDs](maze-behavior-spec.md)). Nothing is special about any single rat — students pick from the group roster, and reproducibility hinges on (subject, seed, dataset_hash) being captured in `run_config.json`.

**Manuscript scope: 48 subjects.** The canonical roster — and the rats whose behavior will be published in the forthcoming corner-maze behavioral manuscript — is the 48-subject list in [README §Manuscript subject roster](../README.md#manuscript-subject-roster). The yoked dataset currently contains 56 subjects; the 8 extras (PI: CM030, CM032, CM033 / PI+VC_f1: CM059 / VC: CM028, CM034, CM039, CM048) are yoked-but-out-of-scope and should not be selected for training runs, eval cells, or published comparisons. When `data/yoked/dataset/` is rebuilt for student release, restrict the build to the 48 manuscript subjects.

**Group/subject must match — enforced, not just convention.** A PI+VC subject must run the PI+VC sequence; a VC subject must run the VC sequence; etc. The runner reads `subjects.parquet` (legacy field: `training_group`), looks up the `--subject`'s group, and:

- If `--training-group` is omitted, infer it from the subject's row.
- If `--training-group` is given and doesn't match the subject's group, fail fast with a clear error (no silent override). E.g.: `--subject CM005 --training-group vc` → error: "CM005 is in PI+VC; cannot run VC acquisition."

The yoked dataset only contains sessions actually run by each rat, so silently mismatching would either error out further downstream (no rows returned) or — worse — train on the wrong session_type rows. Fail at config-validation time instead.

A future "batch design" mode could train on multiple rats jointly within one group; out of scope for the initial build. Cross-group batches would not be valid.

### 9.2 Student-facing choices

| Choice | training_group filter | Yoked data status | Notes |
|--------|------------------------|--------------------|-------|
| `pi_vc` (default) | `PI+VC` | ✅ available | Most-tested data path. |
| `pi` | `PI` | ✅ available | |
| `vc` | `VC` | ✅ available | |
| `pi_vc_f1` | `PI+VC_f1` (manuscript: CM057, CM058, CM060, CM061, CM063, CM064; **CM059 yoked but out of manuscript scope**) | ✅ available | 68 Acquisition + 14 Exposure sessions in dataset; `Fixed Cue 1 Twist` → `PI+VC f1 acquisition`. |

### 9.3 `data/session_types.py` structure

```python
# (env_paradigm, yoked_session_type) ordered tuples per group
SESSION_SEQUENCES: dict[str, list[tuple[str, str]]] = {
    "pi_vc": [
        ("PI+VC f2 rotate",      "Rotate Train"),
        ("PI+VC f2 novel route", "Fixed Cue 1"),
        ("PI+VC f2 no cue",      "Dark Train"),
        # ... extend as paradigm sequence is finalized
    ],
    "pi": [
        ("PI acquisition",       "Dark Train"),
        ("PI novel route cue",   "Fixed Cue 1"),
        # ...
    ],
    "vc": [
        ("VC acquisition",       "Rotate Train"),
        ("VC novel route fixed", "Fixed Cue 1"),
        # ...
    ],
    "pi_vc_f1": [
        ("PI+VC f1 acquisition", "Fixed Cue 1 Twist"),
        # ...
    ],
}
```

The runner iterates through this sequence per chosen group, pulling the matching yoked rows for each (env_paradigm, yoked_session_type) pair.

## 10. Evaluation protocol

Grounded in the empirical-RL methodology canon ([LLM Wiki - AI Reproducibility](file:///Users/ryangrgurich/Code/llm-wiki/vaults/ai-reproducibility/Wiki/index.md)).

### 10.1 What we report per (model, session_type, encoder_config) cell

**Performance (across runs):**
- **IQM** of final per-session perfect-trial count (last 5 sessions averaged) as primary metric. Justification: [agarwal-2021]. Robust to outlier seeds.
- **Median** and **mean** as supplementary.
- **Stratified bootstrap 95% CIs** via Rliable (`pip install rliable`).
- **Performance profile**: empirical CDF of final scores across seeds. Reveals algorithm crossings ([agarwal-2021], [jordan-2020]).

**Reliability (within model):**
- **DR** — IQR across seeds at the final checkpoint ([chan-2020]).
- **LRT** — CVaR(α=0.05) of drawdown from running peak. Catches catastrophic mid-training drops.
- **DF** — variability of evaluation rollouts of the trained policy (run 30 deterministic rollouts of saved checkpoint, report IQR of total return).

**Pairwise comparison (model A vs B):**
- Paired bootstrap on IQM-difference. Bonferroni correction when comparing > 2 models.
- For very small N (< 20), use Welch's t-test instead — bootstrap CIs under-cover at small N ([colas-2018]).

### 10.2 Seed counts and run profiles

`scripts/train.py --profile {pilot,compare,headline}` selects a coordinated bundle of seed count + kill-switch config + reporting expectations. Defaults documented in README.

| Profile | N seeds | Kill switch | Use case |
|---------|---------|-------------|----------|
| `pilot` (default) | 5 | aggressive (default thresholds) | initial design iteration, hyperparameter sanity checks; explicitly underpowered |
| `compare` | 10–20 | default | pairwise A/B between models or encoder configs |
| `headline` | 30 | relaxed thresholds (longer WARMUP, lower DEAD_WINDOW sensitivity) | published-quality runs; final negative-results claims (Wilson 95% upper bound ≤ 0.10) |

Individual flags (`--seeds N`, `--kill-switch off`, `--warmup 15`) override profile defaults. `--profile` sets the *defaults*, never silently overrides explicit flags.

Justification: [colas-2018] (N≥10 for power), [agarwal-2021] (N≥30 for headline IQM), [patterson-2024] (N=50+ when feasible).

The kill switch *adds* statistical power per-seed (more compute reaches the seeds that learn), but doesn't substitute for seed count. Document this trade-off in `eval/metrics.md`.

### 10.3 Negative results as first-class output
Three of our model × encoder × paradigm cells will likely fail (SR on yoked already known to fail — see [sr-yoked-negative-results.md](sr-yoked-negative-results.md)). Frame these as scientific findings, not bugs. Per [patterson-2024]:

> "Under [observation modality] and the tuning ranges in Table N, [model] did not reach criterion on [session_type] within [budget] sessions (0/N runs; Wilson 95% upper bound: 3/N). The same model reliably reached criterion on [adjacent paradigm] (M/N runs)."

Bake this template into the comparison notebook (07).

### 10.4 Two-stage tuning
Avoid maximization bias:
- **Stage 1** — separate seed pool for hyperparameter selection (5–10 seeds per config).
- **Stage 2** — fresh seeds for the reported numbers (10–30 seeds, never overlapping Stage 1).
- Document which seed pool any given metric was computed on.

`scripts/train.py` accepts `--stage {tune,report}` to make this explicit.

### 10.5 What we deliberately don't do
- Don't report max-during-training (biased upward, protocol-incompatible across runs).
- Don't tune-and-report on the same seeds.
- Don't use bare bootstrap CIs at N < 20.
- Don't claim "model A is better" without paired statistics.

## 11. Data pipeline

The yoking pipeline lives in this repo at `src/corner_maze_rl/yoking/` (ported 2026-05-08 from `corner-maze-rl-legacy/yoking/`). It reads from upstream `corner-maze-analysis/data/processed/` (path via `$CORNER_MAZE_ANALYSIS_DIR`) and writes the 5-table dataset that downstream training consumes:

```
$CORNER_MAZE_ANALYSIS_DIR (upstream behavioral parquets)
                ↓ corner-maze-build-yoked  (per-session actions → data/yoked/*.parquet)
                ↓ corner-maze-build-dataset (normalize → 5-table layout + actions_to_reward column)
data/yoked/dataset/
  ├── subjects.parquet
  ├── sessions.parquet                               # all phases
  ├── actions_synthetic_pretrial.parquet             # Acquisition only, synthetic pretrial — primary input
  ├── actions_real_pretrial.parquet                  # Acquisition only, real pretrial — alt variant
  └── actions_exposure.parquet                       # Exposure only (no pretrial concept)

All three actions_*.parquet carry an `actions_to_reward` int32 column
(structural countdown to the next correct PICKUP — see §4.2). Per-step
reward magnitudes are derived at token-gen time from a CostConfig — not
persisted in any parquet (§4.2.1).
```

`build_dataset.py` post-build assertions guarantee every action-table `session_id` has a matching `sessions.parquet` row with the correct phase, every session ends at a rewarded PICKUP, and `n_trials == n_rewards == len(trial_configs)` for every Acquisition row. The same invariants plus the `actions_to_reward` contract are locked by `tests/test_yoked_dataset_invariants.py`. Diagnostics live in `yoking/diagnostics/` (`check_divergence.py`, `check_well_visits.py`, `check_contiguity.py`, `replay_session.py` — pygame, optional).

`scripts/add_actions_to_reward.py` exists for datasets aggregated before the column was wired into `build_dataset.py`. It's idempotent and obsolete on any fresh rebuild.

## 12. Environment integration

The env is ported from `corner-maze-rl-legacy/src/env/corner_maze_env.py` (see manifest §16.1) — trimmed for student readability but **no behavioral changes**.

The model-pluggability contract: a new model only needs to implement `TrainableAgent` and consume `obs` + return `action`. The env's obs space exposes:
- `view` (21×21×3 image) — for CnnPolicy.
- `embedding` (vector from registry) — for MLP/transformer.
- `stereo` (2-channel image) — for the StereoFeaturesExtractor pipeline.

A new model may add a *new* obs key (env-side change). The pipeline above the env (encoder, runner, eval) is unchanged.

## 13. Open questions / pre-build tasks

In priority order:

1. **Per-group session sequence ordering** — the (group, yoked_session_type) → env_paradigm *mapping* is in §9.1, but the *order* in which a group's training arc runs (e.g., does PI+VC always go rotate → novel → no_cue, or different?) is not yet specified. Required for `data/session_types.py` and `train/runner.py` (Phase 2). **Not blocking Phase 1.**

2. **TODO: complete one empty mapping cell** — `Dark Train × VC`. Defer; runner skips unmapped pairs with a warning. Resolve before headline runs.

### Closed (this iteration)

- ~~RTG storage design~~ — env-replay path (`compute_returns.py` + `actions_with_returns.parquet`) replaced 2026-05-11 by the **deferred-cost** design: persist `actions_to_reward` (int32) in `actions_*.parquet`, derive per-step `reward` and float RTG at token-gen from a runtime `CostConfig`. See §4.2 / §4.2.1.
- ~~Session truncation~~ — every yoked session ends at its last rewarded PICKUP (`tests/test_yoked_dataset_invariants.py` locks this). 9 zero-reward sessions dropped from the dataset. Decided 2026-05-10.
- ~~Manuscript subject roster~~ — the 48-subject canonical scope is in [README.md §Manuscript subject roster](../README.md#manuscript-subject-roster). 8 yoked-but-out-of-manuscript subjects retained in the dataset for completeness.
- ~~Repo / package name~~ — repo `corner-maze-rl`, Python package `corner_maze_rl` (matches repo for student clarity). Decided 2026-05-05.
- ~~PI+VC_f1 data~~ — separate subjects (CM057, CM058, CM059, CM060, CM061, CM063, CM064) using `Fixed Cue 1 Twist` session_type. **Resolved 2026-05-08:** plan's "not yet yoked" claim was wrong — yoking pipeline produced correct action sequences all along (trial_configs are read verbatim from upstream `trials.parquet`, which already encode f1-specific routes). The actual missing piece was the env-paradigm mapping `(PI+VC_f1, Fixed Cue 1 Twist) → "PI+VC f1 acquisition"` in `data/session_types.py`, now in place.
- ~~`Fixed Cue 1 Twist` mapping~~ — only PI+VC_f1 has Twist sessions in the dataset; mapped to `PI+VC f1 acquisition`. PI/VC/PI+VC × Twist remain blank because those subjects don't have Twist data, not because they're TODO.
- ~~Positional encoding choice~~ — keep flexible: 4 styles (`learned`, `sinusoidal`, `spatial`, `none`) ship out of the box. Default `learned`.
- ~~Repo hosting~~ — public GitHub confirmed.
- ~~Seed count for headline runs~~ — N=30 for `headline` profile, N=5 for `pilot` (the default), N=10–20 for `compare`. Implemented as `--profile` flag.
- ~~Subject selection~~ — one rat per run via `--subject CM###`; pick any rat from the chosen training group's roster (see §9.1.1). Decided 2026-05-05.

## 14. Phased build order

**Phase 0 — pre-build investigation (1 session)**
- Resolve open questions 1, 2, 3 above. Update this plan.

**Phase 1 — foundation**

Order:

1. **`utils/run_io.py`** (port from legacy `src/rl/seed_utils.py`) — first, because it sets up the package skeleton (`pyproject.toml`, `src/corner_maze_rl/`, `tests/`) and provides the seeding + `run_config.json` writer that every other module depends on.

After (1) lands, these three are independent and parallelizable:

2. **`train/kill_switch.py`** + unit tests — pure logic, no env/data deps. TDD with the canonical test curves listed in §8.2.
3. **`encoders/grid_cells.py`** (port from `DatasetBuilder.ipynb` cell 1) — pure math port, no env/data deps.
4. **Env port** (§16.1: `corner_maze_env.py`, `constants.py`, `trial_sequence_gen.py`, `trial_sequence_validation.py`) — independent of (2) and (3); prerequisite for (5).

After (4) lands:

5. **Token-gen helper** (~50 lines, location TBD — likely `data/tokens.py`) — at training time, walks each session's `(action, rewarded, actions_to_reward)` columns and a `CostConfig` to produce per-step `reward` and per-window float-RTG suffix sums. Doesn't depend on the env (cost values are config-driven). The structural `actions_to_reward` column is already persisted in the dataset; this step just applies magnitudes.

**Phase 2 — first model (DT)**
- `models/decision_transformer.py` (clean rewrite from `DTtrainer.ipynb`).
- `train/runner.py` (adapt from `session_runner.py`).
- `eval/rollout.py`, `eval/replay.py`.
- Notebook 03 (DT train).

**Phase 3 — baselines**
- `models/ppo.py`, `models/sr.py` (adapt from `custom_rl.py`).
- Notebooks 05, 06.

**Phase 4 — eval & comparison**
- `eval/metrics.py` (IQM, drawdown via Rliable).
- Notebook 07 (compare models).
- README + `md-files/adding_a_model.md`.

**Phase 5 — student polish**
- `notebooks/` finalized (VS Code / local Jupyter; Colab-in-browser explicitly unsupported for the interactive UI; non-interactive scripts remain Colab-friendly).
- Quickstart README.
- Setup script for `data/`.

## 15. Dependencies

```
gymnasium, minigrid, numpy, pandas, pyarrow, duckdb, torch
stable-baselines3, sb3-contrib    # optional, for SB3 PPO path
rliable                            # IQM, stratified bootstrap
matplotlib, ipywidgets             # notebook viz
pygame                             # local replay (no moviepy)
tqdm
```

No moviepy (video pipeline replaced by pygame replay per §1).

---

## Decisions log (incorporate as user confirms)

- ✅ Real env-derived rewards, not steps_to_go proxy.
- ✅ Per-trial RTG with ITI-start window.
- ✅ No subject conditioning (one rat per experiment).
- ✅ Drop `ACTION_WEIGHTS` hack.
- ✅ Keep grid cell embeddings (port from `DatasetBuilder.ipynb`).
- ✅ Pretrained visual CNN ships; on-the-fly retrain optional.
- ✅ All encoders standardize to 60D (grid cell constraint).
- ✅ 4 hardcoded session types; PI+VC default.
- ✅ Composable encoders + tabular/egocentric switch.
- ✅ Pip-installable for Colab.
- ✅ Run saving with seeds + git SHA, mirroring current `run_config.json` pattern.
- ✅ DT, PPO, SR ship with the repo; new models plug in via `TrainableAgent` protocol.
- ✅ Env unchanged unless model needs new obs key.
- ✅ Kill switch with slope detection, WARMUP, DEAD_WINDOW, HARD_CAP.
- ✅ Eval protocol: IQM + bootstrap CIs + performance profiles + DR/LRT reliability + Wilson rule for negatives.
- ✅ Public GitHub for pip install.
- ✅ Seed counts: pilot=5 / compare=10–20 / headline=30, exposed via `--profile` flag.
- ✅ DT positional encoding kept flexible: 4 styles ship (`learned`, `sinusoidal`, `spatial`, `none`); default `learned`.
- ✅ Session-type mapping resolved — see §9.1 table. PI+VC_f1 wired up 2026-05-08 (`Fixed Cue 1 Twist` → `PI+VC f1 acquisition`).
- ✅ Repo name: `corner-maze-rl` (this repo); Python package `corner_maze_rl` (matches repo); legacy archived as `corner-maze-rl-legacy`.
- ✅ One rat per training run via `--subject CM###`; rat must be in chosen group's roster. Group/subject pairing is **enforced** at config-validation time (e.g. PI+VC subject cannot run VC acquisition); mismatch → fail fast. Multi-rat batch design out of scope.
- ✅ Egocentric image is standalone (CnnPolicy only); not composed with 60-D vector encoders.
- ✅ `n_wm_units = 10` default (legacy parity); shared by tabular PPO and SR paths, not SR-specific.
- ✅ Kill-switch criterion threshold named `CRITERION_MEAN = 24` (75% perfect); slope condition is `slope < FLAT_SLOPE_EPS` (asymmetric — also kills regressing curves). Test plan documents canonical curve cases.
- 📝 TODO: One empty cell in mapping table: `Dark Train × VC`. Skip with warning until resolved.
- ⏳ Per-group session sequence ordering (for runner traversal logic) — Phase 2 prerequisite.

---

## 16. Legacy source manifest

Files to copy or port from `corner-maze-rl-legacy` into this repo as the build proceeds. Phase column maps to §14. "Action" is `copy` (verbatim, read-only artifact) or `port` (rewrite + adapt; legacy file is the reference).

### 16.1 Environment

| Phase | Action | Source (legacy) | Target (new) | Notes |
|-------|--------|-----------------|--------------|-------|
| P1 | port | `src/env/corner_maze_env.py` | `src/corner_maze_rl/env/corner_maze_env.py` | Trim for student readability; **no behavioral changes**. |
| P1 | port | `src/env/constants.py` | `src/corner_maze_rl/env/constants.py` | |
| P1 | port | `src/env/trial_sequence_gen.py` | `src/corner_maze_rl/env/trial_sequence_gen.py` | |
| P1 | port | `src/env/trial_sequence_validation.py` | `src/corner_maze_rl/env/trial_sequence_validation.py` | |

### 16.2 Yoked dataset and pipeline

Pipeline ported in-tree 2026-05-08. Dataset regenerated to add `actions_exposure.parquet` and backfill exposure metadata; gitignored, produced via `corner-maze-build-dataset` from upstream `$CORNER_MAZE_ANALYSIS_DIR`.

| Phase | Action | Source (legacy) | Target (new) | Notes |
|-------|--------|-----------------|--------------|-------|
| P1 | port | `yoking/data_loader.py` | `src/corner_maze_rl/yoking/data_loader.py` | `ANALYSIS_DATA_DIR` now env-var driven. |
| P1 | port | `yoking/map_to_minigrid.py` | `src/corner_maze_rl/yoking/map_to_minigrid.py` | |
| P1 | port | `yoking/map_to_minigrid_actions.py` | `src/corner_maze_rl/yoking/map_to_minigrid_actions.py` | Imports rewired to `corner_maze_rl.env.*`. |
| P1 | port | `yoking/zone_pixel_map.py` | `src/corner_maze_rl/yoking/zone_pixel_map.py` | |
| P1 | port | `yoking/get_tracked_exposure_rewards.py` | `src/corner_maze_rl/yoking/get_tracked_exposure_rewards.py` | |
| P1 | port | `yoking/build_yoked.py` | `src/corner_maze_rl/yoking/build_yoked.py` | Filename now suffixes `_synthetic`/`_real` for Acquisition outputs. |
| P1 | rewrite | `yoking/build_dataset.py` | `src/corner_maze_rl/yoking/build_dataset.py` | Now emits 5 tables (synth/real/exposure split by `session_phase`); adds post-build assertions. |
| P1 | port | `yoking/check_divergence.py`, `check_well_visits.py`, `check_contiguity.py`, `replay_session.py`, `replay_divergence_log.md` | `src/corner_maze_rl/yoking/diagnostics/` | Optional pygame dep for replay. |
| P1 | regen | `data/yoked/dataset/{subjects,sessions,actions_synthetic_pretrial,actions_real_pretrial,actions_exposure}.parquet` | (gitignored output) | 70 subjects, 679 sessions (549 Acq + 130 Exp) after the 2026-05-10 truncation. All `actions_*.parquet` carry an `actions_to_reward` int32 column (§4.2). 1 missing Exposure (CM008 1e: no upstream coords); 9 zero-reward sessions dropped by truncation. |
| — | defer | `yoking/rotate_to_canonical.py`, analysis/replay tools | — | Port when SR/canonical-orientation work resumes. |

### 16.3 Encoders

| Phase | Action | Source (legacy) | Target (new) | Notes |
|-------|--------|-----------------|--------------|-------|
| P1 | port | `notebooks/DatasetBuilder.ipynb` cell 1 | `src/corner_maze_rl/encoders/grid_cells.py` | Grid-cell pose-vector generator. Deterministic, no training. |
| P2 | copy | `2S2C_task/embeddings/60d/position/pose_60Dvector_dictionary.pkl` | `data/encoders/pose_60Dvector_dictionary.pkl` | Pre-built grid-cell dict; ships for fast student start. |
| P2 | copy | `2S2C_task/embeddings/60d/image/ryans_visual_embedding_dictionary.pkl` | `data/encoders/ryans_visual_embedding_dictionary.pkl` | Pretrained visual CNN dict. |
| P2 | port | `notebooks/Minigrid_Maze_Yoked_Training.ipynb` cells 11–12 | `src/corner_maze_rl/encoders/visual_cnn.py` | CNN train + extract; pretrained weights ship. |
| P1 | port | `src/rl/custom_rl.py::generate_state_vector*` | `src/corner_maze_rl/encoders/{one_hot_pose,reward_history}.py` | Two functions split into separate encoder modules. `n_wm_units` defaults to 10 (legacy default); shared by PPO and SR tabular paths. |

### 16.4 Models

| Phase | Action | Source (legacy) | Target (new) | Notes |
|-------|--------|-----------------|--------------|-------|
| P2 | port | `2S2C_task/colab/DTtrainer.ipynb` cells 1–2 | `src/corner_maze_rl/models/decision_transformer.py` + token-gen helper (location TBD; see §14 Phase 1.5) | DT model + dataloader template; rewrite per dt-repo-plan §6.2 (real RTG via runtime CostConfig, configurable pos encoding, no action-weights hack). |
| P3 | port | `src/rl/custom_rl.py::PPOAgent` | `src/corner_maze_rl/models/ppo.py` | |
| P3 | port | `src/rl/custom_rl.py::SRAgent` | `src/corner_maze_rl/models/sr.py` | |
| P3 | optional | `src/rl/sb3_agents.py` | `src/corner_maze_rl/models/sb3_ppo.py` | Optional SB3 path for advanced students. |

### 16.5 Training & utilities

| Phase | Action | Source (legacy) | Target (new) | Notes |
|-------|--------|-----------------|--------------|-------|
| P1 | port | `src/rl/seed_utils.py` | `src/corner_maze_rl/utils/run_io.py` | Add git-SHA capture + dataset hash. Legacy also ships `register_experiment()` for manifest.jsonl indexing — port if useful, else drop. |
| P2 | port | `src/rl/session_runner.py` | `src/corner_maze_rl/train/runner.py` | |
| P2 | port | `scripts/manual_control.py` | `src/corner_maze_rl/eval/replay.py` | Pygame-based replay; no moviepy. |

### 16.6 Reference / documentation

| Action | Source (legacy) | New repo location | Notes |
|--------|-----------------|-------------------|-------|
| copied | `md-files/maze-behavior-spec.md` | `md/maze-behavior-spec.md` | ✅ done, this commit. |
| copied | `md-files/reward-structure-analysis.md` | `md/reward-structure-analysis.md` | ✅ done. |
| copied | `md-files/sr-yoked-negative-results.md` | `md/sr-yoked-negative-results.md` | ✅ done. |
| extracted | legacy `MEMORY.md` Environment Architecture section | `md/environment-architecture.md` | ✅ done. |

### 16.7 Things explicitly NOT to copy

- `src/rl/sb3_agents.py::StereoFeaturesExtractor` and stereo-eye dataset — out of scope for DT/PPO/SR student build.
- `md-files/yoked-env-integration-plan.md`, `md-files/yoking-process-log.md`, `md-files/well-exit-action-masking-plan.md` — historical legacy planning artifacts.
- `experiments/`, `model-data/` from legacy — those are legacy run outputs.
- Notebooks `corner-maze-rl.ipynb`, `colab-minigrid-rl.ipynb` — legacy scratchpads; new repo will have its own clean Colab notebooks per §3.
