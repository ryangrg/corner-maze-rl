# `10_vanilla_dt.ipynb` — design document

A reference + onboarding document for the vanilla Decision Transformer notebook.

The notebook lives at `notebooks/10_vanilla_dt.ipynb`. It is a faithful port of
the [kzl/decision-transformer/gym](https://github.com/kzl/decision-transformer/tree/master/gym)
build, adapted for the corner-maze yoked rat behavioral dataset. It trains a DT
per rat subject, then evaluates the trained DT on (a) an acquisition sanity
check and (b) probe paradigms — both in the same `CornerMazeEnv` the rat
behaviorally experienced.

This document is **for the agent or human who picks up the notebook next**. It
covers:

1. The scientific goal and what counts as a successful evaluation
2. The yoked-data pipeline and the canonical-frame rotation
3. The DT architecture and the deliberate deviations from kzl
4. The inference protocol (env wiring, RTG, temperature sweep)
5. A cell-by-cell map of the notebook
6. The Cell 1 configuration knobs
7. Every artifact the pipeline writes
8. How to make common modifications without breaking things
9. Pitfalls discovered the hard way
10. Provenance and external references

If you are coming in cold, read sections 1–5 in order. Sections 6–10 are
reference.

---

## 1. Scientific goal

The corner-maze project tests whether different navigation strategies show up
in different brain "cohorts": **PI** (path-integration-trained rats), **PI+VC**
(both PI and visual-cue trained), and **VC** (visual-cue-only). The behavioural
hypothesis is that PI rats navigate using an internal compass / grid-cell
representation and ignore visual cues; VC rats do the opposite; PI+VC rats use
both.

This notebook builds a **Decision Transformer per rat** trained on that rat's
recorded behaviour. The DT is then evaluated under conditions that
selectively stress each strategy:

- **Acquisition** — trained-condition trials (same cue/goal pairing the rat
  learned). All cohorts should succeed. This is the **sanity check**; if the
  DT fails here, downstream probes are meaningless.
- **Novel-route probe** — different start arms but the cue/goal pairing
  unchanged. Tests route-vs-place learning.
- **No-cue probe** — cue removed. PI rats should still navigate (they have a
  virtual compass); VC rats should fail.
- **Rotate probe** — cue rotates within the session. PI rats should ignore
  the cue and navigate by grid frame; VC rats should follow the cue and fail.
- **Reversal probe** — cue/goal mapping inverted. Tests behavioural
  flexibility.

The comparative analysis we are building toward asks: **does the DT trained on
PI data behave like a PI rat on these probes? Does the PI+VC DT mix the two?
Does the VC DT collapse when the cue moves?**

Two consequences for evaluation rigour:

- **Rat-vs-DT must be apples-to-apples.** Same env paradigms, same canonical
  frame, comparable trial counts. See section 4 on inference protocol.
- **DT inherits stochasticity from the rat.** A single sampled rollout is
  noisy; we need distributional comparison (multiple seeds × multiple repeats
  × temperature sweep). See section 4 on the temperature sweep design and
  section 8 on rliable bootstrap CIs.

---

## 2. The yoked data and the canonical-frame rotation

This is the section to read most carefully. The yoking and rotation pipeline
is the **single largest source of subtle bugs** in this notebook's history,
and the rotation logic explicitly **deviates from the legacy implementation
because the legacy has a sign error**.

### 2.1 What yoking does

The corner-maze experimental rig records each rat as a coordinate stream + a
sequence of well visits + trial boundaries. The yoking pipeline
(`src/corner_maze_rl/yoking/`) converts that raw observation into a MiniGrid
action stream that, when replayed in `CornerMazeEnv` with the rat's recorded
`trial_configs` and `cue_goal_orientation`, reproduces the rat's
`(grid_x, grid_y, direction)` trajectory step-for-step. The output is three
parquets in `data/yoked/dataset/`:

| parquet | rows | columns of interest |
|---|---|---|
| `subjects.parquet` | 70 | `subject_name`, `training_group` (PI/PI+VC/VC/...), `cue_goal_orientation` |
| `sessions.parquet` | 552 | `session_id`, `subject_id`, `session_number`, `session_type` (Fixed Cue 1 / Dark Train / ...), `session_phase` (Acquisition / Exposure), `n_trials`, `trial_configs` (JSON list of `[arm, cue, goal, tag]`) |
| `actions_real_pretrial.parquet` | 856 K | per-step rows with `session_id`, `step`, `action`, `grid_x`, `grid_y`, `direction`, `rewarded`, `actions_to_reward`, `pose_label` |

**Key implementation detail to remember**: `actions_real_pretrial.parquet`
has only `session_id` — no `subject_name`. The notebook joins via
`actions.session_id → sessions.subject_id → subjects.subject_name`. See
Cell 3 (`_compute_n_rotations`, schema check).

### 2.2 What `cue_goal_orientation` means (not what it looks like)

Every subject in the dataset has `cue_goal_orientation` of the form `'N/X'`
where `X ∈ {'NE', 'SE', 'SW', 'NW'}`. The natural reading is "cue is at N and
goal is at X", but **this is misleading**.

The actual meaning: it is the **rotational pairing** the rat learned. `'N/NE'`
encodes "if the cue is at N, the goal is at NE; if the cue rotates to E, the
goal rotates with it to SE; cue at S → goal at SW; cue at W → goal at NW."
The pairing is a constant 1-corner-CW relationship; the absolute cue position
varies session-to-session in the actual experiment.

Empirically (run the audit in section 11 to reproduce): CM016's
`cue_goal_orientation = 'N/SE'` but every one of its acquisition sessions
had cue at E, S, or W — **never at N**. The rat experienced the rotational
class, not a single cue position.

The yoked dataset preserves the rat's actual experience: per-step `pose_label`
strings encode the cue position that was physically present (`trl_e_e_xx_...`
for session 1569's cue=E, `trl_w_w_xx_...` for session 1583's cue=W, etc.).
**The data is not pre-canonicalized.**

### 2.3 Why we rotate to canonical N (and when not to)

Two reasons to rotate:

1. **PI/PI+VC rats use an internal grid frame.** The brain's representation is
   anchored to the rat's compass, not to whatever the lab decided to do with
   the cue that day. To match the brain's frame, we rotate per-session so the
   cue lands at canonical N. After rotation, every PI/PI+VC training step is
   in the rat's internal frame.
2. **`grid_cells_60d.npz` is indexed by absolute `(x, y, d)`**. The same
   physical grid cell has the same vector regardless of which session
   produced it. Without rotation, the same grid cell appears under different
   labels across sessions — the DT would have to learn "cue=E sessions and
   cue=S sessions describe the same world."

**When NOT to rotate**:

- **VC rats**: rotating cue **is the VC training signal**. VC rats learn to
  follow the cue regardless of its absolute position. Their multi-cue sessions
  are not anomalies — they are the experimental design. `ROTATE_TO_CANONICAL`
  is False for `training_group == 'VC'`.
- **Rotate probes** (`PI+VC f2 rotate`, `VC novel route rotate`, etc.):
  the cue rotates within a session by design. The env's
  `get_*_rotate_pairs(orientation)` generators iterate cue through all 4
  positions internally; per-session rotation cannot apply. No-op for these.
- **Exposure sessions**: no anchoring property (no cue display).

### 2.4 The rotation formulas

Single 90° CCW rotation **in screen coordinates** (y-down):

| Field | Transform |
|---|---|
| `(grid_x, grid_y)` | `(y, 12 − x)` |
| `direction` | `(d − n_rotations) % 4` |
| arm/cue letter | `'n' → 'w' → 's' → 'e' → 'n'` (one CCW step on compass) |
| goal corner | `'ne' → 'nw' → 'sw' → 'se' → 'ne'` |
| pose_label suffix | rotate the (x, y, d) numerals |

`n_rotations = cue_index of first trial` (per-session, since PI/PI+VC sessions
are single-cue). Verified empirically across all 239 single-cue PI/PI+VC
acquisition sessions: 100% coverage in `pose_visual.npz`, 0 missing rotated
labels, 2568 (orig, rotated) pairs map to the same embedding row.

### 2.5 ⚠ Direction-rotation sign error in legacy code

The legacy implementation at `~/Code/python-dev/corner-maze-rl-legacy/yoking/rotate_to_canonical.py:62` uses
`(d + n_rotations) % 4`. **This is wrong** for `n ∈ {1, 3}` and only accidentally
correct for `n = 2` (180° symmetric). The notebook uses the corrected
`(d − n_rotations) % 4`.

Empirical proof (we ran this; see section 11.3 to reproduce): trace
`(8, 6, direction=E)` walking forward to `(9, 6)`. After one CCW position
rotation, `(8, 6) → (6, 4)` and `(9, 6) → (6, 3)` — the rotated step is
`(6, 4) → (6, 3)`, a step in `−y` direction, i.e., **direction N (= 3)**.
Legacy gives `(0 + 1) % 4 = 1 = S` (180° off). Corrected formula gives `3 = N`.
✓

The legacy validation in `_validate_rotated_data()` only checks that rewarded
positions land at the canonical-goal corner — it never checks directions, so
the bug stayed buried for a long time. Any downstream model that consumed
`direction` (e.g., grid-cell encoders) would silently see inconsistent data.

We verified the corrected formula against `check_divergence.py` for CM016
(PI+VC, all rotations), CM023 (PI, all 4 rotations including n=1, n=3),
CM031 (VC, n=0), and CM037 (PI, n=2 and n=3). 100% divergence-free.

### 2.6 What the rotation actually changes per encoder

This is critical and counter-intuitive. We rotate uniformly **as a defensive
design choice**, but the effect on the DT's input stream depends on the
encoder mode:

| `ENCODER_TYPE` | What rotation does to state_idx |
|---|---|
| `pose_visual` | **No-op** in practice. `pose_visual.npz` is a 4536→295 lookup that collapses all 4 rotational variants of any pose to the same embedding row. We verified this across 491K rows: every (orig, rotated) pair maps to the same `state_idx`. The DT sees identical inputs with or without rotation. |
| `image_cnn` | **No-op**. Sampled checks show `minigrid_views.npz` views are byte-identical for rotational equivalents (the 13×13 maze is 4-fold rotationally symmetric, and the cue visual is the same pattern regardless of position). |
| `grid_cell` | **Substantive**. `grid_cells_60d.npz` is indexed by absolute `(x, y, d)`. Same physical pose under rotation maps to a different 60D vector (max |Δ| ≈ 1.4–4.4 across the 60D vector in spot checks). Rotation is **essential** for grid_cell — without it, the DT sees rotationally-distinct vectors for the same internal-frame pose, blowing up the input distribution. |

Why rotate uniformly anyway? Defensive design. We don't want to remember
"oh, this encoder needs rotation but that one doesn't" when swapping
encoders. The cheap rotation is applied once at data-load time. For
`pose_visual` and `image_cnn` it's free in compute and in model behavior.
For `grid_cell` it's load-bearing.

### 2.7 Trial-level vs session-level boundaries in the data

The yoked data uses **`actions_to_reward`** as the canonical trial-boundary
signal: a per-step integer that counts down to 0 at the rewarded step and
jumps back up at the next trial's start. The notebook derives:

```python
trial_start_mask = np.concatenate([[True], a2r[1:] > a2r[:-1]])
trial_id         = np.cumsum(trial_start_mask) - 1
trial_timestep   = arange(L) - trial_start_positions[trial_id]
```

A "trial" by this definition spans from one reward-event to the next, which
includes the **ITI + pretrial + trial-phase** of the next reward cycle. This
matches the user's mental model — the rat's "trial unit" is goal-to-goal,
not the env's narrow "trial phase". `trial_timestep` resets to 0 at each
reward and counts up through the next ITI → pretrial → trial phases.

This is the timestep signal fed to the DT's `embed_t` (see section 3.2).

---

## 3. The DT architecture

A faithful port of `kzl/decision-transformer/gym/decision_transformer/models/decision_transformer.py`
adapted for discrete actions. Three components composed in Cell 5:

### 3.1 `DTEmbedding`

Per-modality embedding layers + a shared learned timestep + LayerNorm, then
interleave into a `(B, 3K, H)` sequence in **`(R, s, a)` order** (kzl
convention):

```
rtg(B,K)    → Linear(1, H)        ┐
state(B,K)  → state_encoder       ├ + embed_t(trial_timestep)  → LayerNorm → stack & flatten
action(B,K) → Embedding(5, H)     ┘                                          ↓
                                                              (B, 3K, H) = [R₀, s₀, a₀, R₁, s₁, a₁, ...]
```

The state encoder is one of (Cell 5):

| `ENCODER_TYPE` | encoder module | trainable |
|---|---|---|
| `pose_visual` | `nn.Embedding.from_pretrained(STATE_LOOKUP, freeze=True)` — 295×60 lookup | no |
| `grid_cell`   | `nn.Embedding.from_pretrained(STATE_LOOKUP, freeze=True)` — 484×60 lookup | no |
| `image_cnn`   | `ImageCNNEncoder` — 3→16 conv + flatten + Linear → 60D | yes |

### 3.2 Per-trial `embed_t` reset — deliberate deviation from kzl

kzl's `embed_t` indexes by step-within-episode, where an "episode" is one
MuJoCo rollout (≤1000 steps). For our 2000-3000-step rat sessions this is
the wrong unit:

- A per-session timestep signal puts almost zero samples on high-position
  embeddings (only the few sessions long enough to reach step 1500 train
  `embed_t[1500]`).
- A per-trial timestep signal puts ~32 trials/session × N sessions ≈ 100s of
  samples on every `embed_t[i]` for `i < typical_trial_len`.

So we use `trial_timestep` (per-trial, resets at each reward) instead of
session-step. `MAX_EP_LEN` shrinks from ~3000 to ~250, the embedding table is
~12× smaller, every row gets meaningful training signal, and
`embed_t[5]` consistently means "5 steps into a trial" rather than the
ambiguous "5 steps into whatever phase of whatever session."

### 3.3 GPT2 body — custom, not HuggingFace

Cell 5 hand-rolls a small causal-self-attention transformer (`CausalSelfAttention`
+ `GPT2Block` + `GPT2Body`) rather than depending on `transformers.GPT2Model`.
Reasons:

- **No internal positional embedding.** kzl's gym build zeroes out the HF
  GPT2's wpe layer. Our hand-rolled version simply omits it. Position info
  enters only via `embed_t`.
- **Attention extraction.** Each block returns `(out, attn)`; Cell 13's
  diagnostics decode the per-step attention back into `(step_offset, modality)`
  for the comparative-analysis plots. Trivial with our wrapper, more setup
  with HF hooks.
- **No HF dep**. Saves ~700MB of transformers wheel.

Default config (Cell 1): 3 layers, 4 heads, 60-dim hidden, GELU MLP with 4×
expansion, 0.1 dropout, causal mask built from the token sequence length.

### 3.4 Action prediction head

`Linear(H, NUM_ACTIONS=5)` applied at the **state-token positions** (indices
`1, 4, 7, ...` in the 3K sequence). With the causal mask, the state token at
step k has seen `R[0..k], s[0..k], a[0..k-1]` — exactly the right context
for predicting `a[k]`.

CE loss over all K positions in each window (Cell 6). The autoregressive
structure means the model gets supervision from every step, and gradients
flow back through the (R,s,a) history at every position.

### 3.5 Why 5 actions

`NUM_ACTIONS = 5`: `0=left, 1=right, 2=forward, 3=pickup, 4=pause`. The
`pause` action exists in the env but the DT is trained on rat data which uses
all five. We do **not** apply action masking at inference
(`ACTION_MASK_INFERENCE = False`) — the DT is expected to learn the
pickup-only-in-corner constraint from the data distribution.

---

## 4. Inference protocol

Cell 7 (`run_dt_episode`) is the bridge from "trained DT" to "agent acting in
`CornerMazeEnv`". Several non-obvious choices live here.

### 4.1 Cue-N anchoring at inference

For PI/PI+VC, training data was rotated to canonical-N. To match at inference,
the env must run in the canonical frame. The env constructs its trial
sequence based on `agent_cue_goal_orientation` and `start_goal_location` (a
goal-index handle into `GOAL_LOCATION_MAP`).

The trick: **set `start_goal_location` to the second half of
`cue_goal_orientation`**. For CM016 (`'N/SE'`), that's `'SE'`. With
`gli = GOAL_LOCATION_MAP['SE'] = 1`, the env's trial generator
(`get_f2_trained_pairs('N/SE', gli=1)`) returns trials with `cue=N, goal=SE`
— the canonical-N frame.

Cell 7's `_trained_goal_for_subject(subject)` does this automatically; the
caller doesn't need to think about it. Override via the explicit
`start_goal_location=` kwarg if you need to.

For rotate probes, the env's `get_*_rotate_pairs` generators ignore gli and
iterate cue through all 4 positions internally. The anchor is harmless
(first-trial seed) but doesn't constrain the rotation pattern.

### 4.2 RTG management — analytical, not sum-of-rewards

kzl's RTG update at inference is `next_rtg = current_rtg − reward_received`.
This is correct when RTG = sum of future rewards (the MuJoCo definition).

Our RTG is an **analytical signal** scoped per trial, not a sum of rewards:

```
rtg[t] = R - c · actions_to_reward[t]
```

where `R = 1.0` (terminal reward, `COST.terminal_reward`) and `c = 0.001`
(step cost, `COST.step_cost`). It climbs from `R − c·L_trial` at trial start
to `R` at the reward step. At the next trial's first step, it **jumps back
down** to `R − c·L_{trial+1}`. This is a per-trial-relative signal, not a
session-spanning return.

kzl's update rule does not match this dynamic. The notebook uses:

- **Within a trial**: `next_rtg = current_rtg + COST.step_cost`. Smooth
  upward drift matching the analytical signal.
- **At trial boundary** (detected via `info['trial_count']` incrementing):
  `next_rtg = INFERENCE_RTG` (re-anchor to the conditioning target).

The env's actual reward stream (`-0.0005` to `-0.001` per step, `+1.061` at
goal) is **ignored** for RTG bookkeeping. We use it only as logging signal
for `env_total_reward` summaries.

### 4.3 Re-anchoring at trial boundary

`INFERENCE_RTG = R − c · EXPECTED_OPT_LEN` (set in Cell 3 from
`median(late_acq_starts)` over the rat's last 2 acquisition sessions per
subject). For CM016, ~59 steps → `INFERENCE_RTG ≈ 0.941`.

This is the value we re-anchor to at every trial start during inference. The
DT was trained on a distribution of `rtg[trial_start]` values that depend on
the trial length; we pick the median competent trial length as the
conditioning target. The DT then has to act in a way that achieves that
target.

`EXPECTED_OPT_LEN` is configurable via Cell 1's `LAST_N_ACQ_FOR_RTG` (default
2 — the rat's graduation-criterion sessions). The 72% correct-first-goal
graduation gate is implicit in the dataset's existence (rats had to hit it
to be in the dataset at all), so we trust it without an explicit check.

### 4.4 State lookup at inference

Each step:

```python
def _env_state_idx(env):
    if ENCODER_TYPE == 'grid_cell':
        x, y = env.agent_pos
        d    = env.agent_dir
        return ((x-1) * 11 + (y-1)) * 4 + d   # same formula as Cell 3
    return label_to_row[env.get_pose_label()]  # for pose_visual / image_cnn
```

For pose_visual/image_cnn, this uses the same `label_to_row` dict that
Cell 3 built from the npz. The npz's rotational collapse means even if the
env generates a non-canonical-cue trial (e.g. for a rotate probe), the
returned state_idx is consistent with the training distribution. **No
explicit per-step rotation needed at inference.**

For grid_cell (when we switch to it for the cohort comparison), we'd need
to either rotate `env.agent_pos / env.agent_dir` back to canonical, or
configure the env to run canonical-frame trials only. The current pose_visual
default sidesteps this question.

### 4.5 Variable-length context

Steps 0 through K-1 don't have a full K-token history yet. Rather than
zero-padding (kzl's approach for fixed-shape input), our custom transformer
just accepts variable `T = 3 × ctx_K` where `ctx_K = min(t+1, K)`. The
causal mask is built inside the attention layer based on the actual `T`.
Cleaner than padding + masking; works because there's no internal positional
embedding to be confused by shorter sequences.

The action-buffer needs a **placeholder** for the current position (the DT
hasn't predicted it yet). We append `0` to the action list, but the causal
mask ensures the model doesn't read it — only `R[0..k]`, `s[0..k]`,
`a[0..k-1]` are consumed.

### 4.6 Temperature sweep + per-temperature repeats

The DT's policy is stochastic when temperature > 0 (Cell 7's softmax sampling).
For rigorous comparison to the rat:

- **Deterministic argmax** (`temp=None`): the DT's "best guess." Removes
  sampling noise. Maximum interpretability. 1 rollout per setting (argmax
  is deterministic).
- **Sampled at T=1.0** (paper-faithful): kzl-Atari default. Comparable to a
  stochastic rat. N repeats per setting to characterize the action
  distribution.
- **Other temps** (e.g., 0.5, 2.0): exploration of how the policy behaves
  as we sharpen or flatten.

Cell 1's knobs:
```python
ACQ_INFERENCE_TEMPS         = [None, 1.0]   # det + sampled
ACQ_INFERENCE_PROBA_REPEATS = 5             # repeats per sampled temp
PROBE_INFERENCE_TEMPS         = [None, 1.0]
PROBE_INFERENCE_PROBA_REPEATS = 5
```

For sampled rollouts, each repeat uses `rep_seed = seed * 1000 + repeat_idx`
so (seed, repeat) pairs are deterministic across notebook re-runs while
distinct from each other.

### 4.7 Acquisition gate → probes

Cell 10's `run_full_pipeline` gates probes on acq success. Per seed:

```
if max(completed_trials over all temps × repeats) >= ACQ_PASS_COMPLETED_TRIALS:
    run probes
else:
    skip probes for this seed; record probes_ran=False
```

Default threshold `ACQ_PASS_COMPLETED_TRIALS = 24` (out of 32 trials).
Rationale: if the DT can't complete 75% of acquisition trials in **any**
inference mode, downstream probes are uninterpretable — we'd be measuring
"how does broken DT respond to perturbations" not "did learned navigation
generalize."

### 4.8 Memory pre-check

Cell 10 runs `_memory_precheck()` before starting any training. It estimates
peak memory analytically and queries available GPU/MPS/CPU memory; raises
`RuntimeError` if estimated peak exceeds 85% of available. This catches
"K=512 + batch=256 on an 8GB device" before wasting time.

Estimate: dominated by `batch × n_heads × (3K)² × 4 bytes` per attention
layer. For default config (`K=128, B=64, n_heads=4, n_layers=3`),
estimate is ~1.5 GB. M3 Max 32GB has ~25 GB available — passes trivially.

---

## 5. Notebook structure (15 sections)

Each section is one markdown cell + one code cell. Cells **auto-execute** by
default — running cells top-to-bottom kicks off the full pipeline (Cell 10
is the expensive one, ~90 min on MPS for default config).

| # | Section | Defines | Reads | Writes |
|---|---|---|---|---|
| 1 | Configuration | All knobs | — | — |
| 2 | Subject ↔ group validation | `TRAINING_UNITS`, `subject_dir`, `seed_dir` | `subjects.parquet` | `RUN_ROOT/` |
| 3 | Load yoked + rotate + globals | `ALL_SESSIONS`, `STATE_LOOKUP`, `label_to_row`, `sessions_for`, fills `EXPECTED_OPT_LEN`, `INFERENCE_RTG`, `MAX_EP_LEN`, `WARMUP_STEPS` | yoked parquets, lookup npz | — |
| 4 | Dataset + loaders | `RatWindowDataset`, `make_loaders(unit, seed)` | — | — (smoke test runs) |
| 5 | DT model | `DecisionTransformer`, `make_dt(seed)`, `ImageCNNEncoder`, transformer primitives | — | — |
| 6 | Training loop | `train_one_unit(unit, seed, ...)` | training data | `seed_dir/curves.parquet`, `model_best.pt`, `run_config.json` |
| 7 | Env rollout helper | `run_dt_episode`, `_episode_to_step_rows`, helpers | — | — |
| 8 | Acq sanity check | `run_acquisition_sanity_check` | model, late_acq_starts | `seed_dir/acq_inference.parquet`, `acq_trajectory.parquet` |
| 9 | Probes loop | `run_probes` | model | `seed_dir/probes.parquet`, `probe_trajectory.parquet`, `attention/probe_*.pt` |
| 10 | Per-(unit, seed) pipeline | `run_full_pipeline()` + memory precheck | training data | all of the above × N_RUNS, `RUN_ROOT/pipeline_results.parquet` |
| 11 | Cross-seed aggregation | `aggregate_across_seeds`, `_iqm`, `_safe_percentile` | seed-dir parquets | `RUN_ROOT/{probe,acq,training}_aggregate.parquet` |
| 12 | Visualization (rliable) | 4 plot functions | seed-dir parquets + aggregates | `RUN_ROOT/figures/*.png` |
| 13 | DT diagnostics — paths + attention | `_draw_maze_skeleton`, `plot_path_replay`, `plot_attention_summary`, `plot_attention_heatmap` | seed-dir parquets, attention .pt | `RUN_ROOT/figures/{path_replay,attention_summary,attention_heatmap}.png` |
| 14 | Session playback | `playback_trial`, `ACTION_NAMES` | trajectory parquets | `RUN_ROOT/figures/playback_*.gif` |
| 15 | Recipes | `RECIPES` dict, `print_recipe` | — | prints comparison table |

### 5.1 Dependency flow

```
Cell 1 (configs)
  ↓
Cell 2 (TRAINING_UNITS, RUN_ROOT, seed_dir)
  ↓
Cell 3 (ALL_SESSIONS, STATE_LOOKUP, INFERENCE_RTG, MAX_EP_LEN, WARMUP_STEPS)
  ↓                                ↑
  ├──────────────────────────────  │
  ↓                                │
Cell 4 (make_loaders) ←─ Cell 5 (make_dt) ←─ Cell 6 (train_one_unit) ←─ Cell 7 (run_dt_episode)
                                                                            ↑       ↑
                                                                       ┌────┘       │
                                                              Cell 8 (acq) ──── Cell 9 (probes)
                                                                            ↑
                                                                Cell 10 (run_full_pipeline)
                                                                            ↓
                                                                Cell 11 (aggregates)
                                                                            ↓
                                                        ┌─────────────────┬─┴──────────┐
                                                        ↓                 ↓            ↓
                                                  Cell 12 (plots)  Cell 13 (attn)  Cell 14 (gif)
                                                                            ↓
                                                                       Cell 15 (recipes)
```

### 5.2 Smoke tests in Cells 4–9

Each of Cells 4–9 ends with a small smoke test that exercises the cell's
function with `_unit = TRAINING_UNITS[0]`, `_smoke_seed = 0`, and
small/cheap parameters (2 epochs for training, 1 probe for probes, etc.).
This makes the notebook usable for incremental development: you can run
Cells 1–9 and see immediately whether the cell you just edited works,
before committing to Cell 10's full ~90 min run.

The smoke tests write to `seed_dir(_unit, 0)`, which Cell 10 will overwrite
with the full-pipeline results when it later runs seed 0. **Smoke artifacts
are designed to be replaced**, not preserved.

### 5.3 Cell 10 is the expensive one

The full pipeline call at the bottom of Cell 10 (`PIPELINE_RESULTS = run_full_pipeline()`)
is what actually trains the DT × N_RUNS seeds and runs all the inference. For
default config (5 seeds × 50 epochs × full probes), ~90 minutes on MPS, much
longer on CPU. Use the `'smoke'` recipe from Cell 15 for first-pass code
validation.

---

## 6. Cell 1 configuration reference

Every knob, what it does, defaults, when to change it.

### 6.1 Experiment identity

```python
RUN_NAME         = 'dt-vanilla-pi-cm016'   # used in artifact directory naming
EXPERIMENT_NOTES = '...'                    # documentation only
```

`RUN_NAME` should reflect the cohort + subject. Each Cell 10 run creates
`data/runs/{RUN_NAME}_{UTC_timestamp}/` so reruns don't collide.

### 6.2 Subject + group

```python
TRAINING_GROUP = 'pi_vc'                       # 'pi' | 'pi_vc' | 'vc'
SUBJECTS       = ['CM016']                     # ≥1 subject names
POOL_SUBJECTS  = False                         # True = one model on pooled data
```

`TRAINING_GROUP` is the short code that maps to the canonical group label in
`subjects.parquet`'s `training_group` column via `GROUP_ALIASES` in Cell 2.
Cell 2 asserts that every subject in `SUBJECTS` matches.

`POOL_SUBJECTS=True` builds one TrainingUnit per `pooled_<group>` label,
training a single DT on the union of all subjects' data. Default `False`
gives per-rat models — the design that supports the cohort comparison.

### 6.3 Session types

```python
SESSION_TYPES = [
    'PI+VC f2 acquisition',     # idx 0: acquisition (Cell 8 uses SESSION_TYPES[0])
    'PI+VC f2 novel route',     # idx 1+: probes (Cell 9 uses SESSION_TYPES[1:])
    'PI+VC f2 no cue',
    'PI+VC f2 rotate',
    'PI+VC f2 reversal',
]
```

**Convention assumed by Cells 8 and 9**: `SESSION_TYPES[0]` is the
acquisition paradigm; `SESSION_TYPES[1:]` are the probes. Don't reorder
without also updating those cells.

Full list of valid env paradigm strings is in Cell 1's comment block (kept
across edits — it's the canonical reference list for valid env paradigms).

### 6.4 Inference parameters

```python
ACQ_INFERENCE_TRIALS         = 32      # cap on trials per acq rollout (≈ one full session for PI/PI+VC f2)
ACQ_INFERENCE_TEMPS          = [None, 1.0]   # None=argmax, float=sampled at T
ACQ_INFERENCE_PROBA_REPEATS  = 5

PROBE_INFERENCE_TEMPS         = [None, 1.0]
PROBE_INFERENCE_PROBA_REPEATS = 5

ACQ_PASS_COMPLETED_TRIALS = 24   # gate threshold for triggering probes
```

The env's trial generator produces trials in chunks of 8 — `ACQ_INFERENCE_TRIALS = 32`
matches `4 × 8 = 32 trials` for the PI/PI+VC f2 acquisition paradigm. The
env auto-generates this many; `max_trials` in `run_dt_episode` caps the
rollout at this count.

### 6.5 Reward / cost (`CostConfig`)

```python
COST = CostConfig(terminal_reward=1.0, step_cost=0.001)
```

**Locked across training + inference** — part of the run identity. The DT
learns `rtg = R − c · actions_to_reward` under one `(R, c)` and is queried
under the same. Don't change for inference without retraining.

The env's actual rewards (`STEP_FORWARD_COST = −0.0005`, `STEP_TURN_COST = −0.001`,
`WELL_REWARD_SCR = +1.061`) don't match these exactly — that's fine, we use
RTG as an analytical signal not a sum-of-rewards (section 4.2).

### 6.6 Late-acq RTG anchor

```python
LAST_N_ACQ_FOR_RTG = 2   # trailing acq sessions per subject for EXPECTED_OPT_LEN
```

Cell 3 takes the median trial-start length over these sessions; this is
`EXPECTED_OPT_LEN`, used to compute `INFERENCE_RTG`. Default 2 = the rat's
graduation-criterion sessions (per the dataset's selection rule, these are
the trials where the rat hit ≥72% correct first-goal choice).

### 6.7 Windowing

```python
K = 128   # context length
```

kzl gym default is 20. We bumped to 128 because:
- The corner-maze sessions are 2-3× longer than kzl's episodes.
- The rat exhibits strategy across trials, not just within a trial.
- A 128-step window typically spans 2-3 trials, so the DT sees cross-trial
  context.

Cost is O(K²) in attention memory. K=128 → ~150 MB attention scores per
layer per batch. K=256 → 600 MB. K=512 → 2.4 GB. M3 Max 32GB easily handles
K=256; K=512 is feasible with care.

### 6.8 State encoder

```python
ENCODER_TYPE   = 'pose_visual'
ENCODER_DIM    = 60
FREEZE_ENCODER = True
```

Three options: `pose_visual` (default — frozen 60D image embeddings),
`grid_cell` (frozen 60D grid-cell vectors, the SR-yoking encoder), or
`image_cnn` (trainable CNN over raw 21×21×3 views).

Switching to `grid_cell` for the cohort comparison is the planned next move.
Section 9.1 covers what changes are needed.

### 6.9 DT architecture

```python
HIDDEN_DIM   = 60     # matches encoder dim for tight pipeline
N_LAYERS     = 3      # kzl default
N_HEADS      = 4      # 60/4 = 15-dim per head
DROPOUT      = 0.1
MAX_EP_LEN   = None   # filled in by Cell 3 from max_trial_timestep
NUM_ACTIONS  = 5
ACTION_MASK_INFERENCE = False
```

`HIDDEN_DIM = 60` is matched to `ENCODER_DIM` so the state-encoder output
feeds directly into the body. Decoupling them would require an extra
projection layer; kept locked for simplicity.

### 6.10 Training

```python
N_EPOCHS              = 50
BATCH_SIZE            = 64
LEARNING_RATE         = 1e-4
WEIGHT_DECAY          = 1e-4
WARMUP_STEPS          = None    # filled in by Cell 3 from total step count
GRAD_CLIP             = 0.25
EARLY_STOP_PATIENCE   = 5
VAL_FRAC              = 0.15
NUM_WORKERS           = 0
```

Mostly kzl defaults. `WARMUP_STEPS` auto-sized to ~5% of total optimizer
steps (clamped to `[100, 10000]`) — kzl's hardcoded 10000 is too long for
our small dataset.

`NUM_WORKERS = 0` because the dataset is small and pinned-memory single-process
loading is faster than the IPC overhead of multi-process workers.

### 6.11 Logging toggles

```python
SHOW_TQDM                = True
PRINT_EVERY_N_BATCHES    = 0     # off; set to N for extra per-batch prints
SAVE_PER_BATCH_LOSS      = False # writes per_batch_loss.parquet
SAVE_ATTENTION_PER_EPOCH = False # writes attention/epoch_NNN_best.pt
```

`SAVE_PER_BATCH_LOSS` is useful for diagnosing training instabilities.
`SAVE_ATTENTION_PER_EPOCH` lets you watch attention patterns evolve across
training but inflates disk usage.

### 6.12 Eval / probe / pipeline

```python
EVAL_EPISODES_PER_PROBE   = 1
FROZEN_DETERMINISTIC      = False
SAVE_PROBE_ATTENTION      = True
ACQ_PASS_COMPLETED_TRIALS = 24
```

`SAVE_PROBE_ATTENTION = True` triggers attention saves **only for
deterministic rollouts** (Cell 9: `save_attn = save_attention and is_det`).
Sampled rollouts don't save attention because rollout-specific noise makes
per-step attention hard to interpret.

### 6.13 Device

```python
FORCE_CPU = False
# DEVICE resolved at runtime to 'cuda' > 'mps' > 'cpu'
```

### 6.14 Seed sweep

```python
N_RUNS = 5
```

Sequential seeds 0..N_RUNS-1. Every seedable component is keyed off this
integer (model init, train/val split, batch shuffle, env init per repeat).
Deterministic — re-running the notebook reproduces the same N=5 sample.

To switch to random seeds, replace `for seed in range(N_RUNS)` in Cell 10
with `for seed in [random.randrange(2**31) for _ in range(N_RUNS)]` and log
the generated seeds (every artifact already carries `seed`, so this is safe).

---

## 7. Artifact reference

Every file the pipeline writes, where, and what's in it. All paths are
relative to `RUN_ROOT = data/runs/{RUN_NAME}_{UTC_timestamp}/`.

### 7.1 Per-(unit, seed) — written by Cells 6/8/9 inside `RUN_ROOT/{unit_label}/seed_{N}/`

```
seed_N/
    run_config.json                   ← Cell 6: hparams snapshot
    curves.parquet                    ← Cell 6: one row per epoch
    model_best.pt                     ← Cell 6: lowest val_loss checkpoint
    acq_inference.parquet             ← Cell 8: per-(temp × repeat × trial) summary
    acq_trajectory.parquet            ← Cell 8: per-step rows for all acq rollouts
    probes.parquet                    ← Cell 9: per-(probe × temp × repeat × trial) summary
    probe_trajectory.parquet          ← Cell 9: per-step rows for all probe rollouts
    per_batch_loss.parquet            ← Cell 6, if SAVE_PER_BATCH_LOSS=True
    attention/
        probe_<paradigm>_temp_det.pt  ← Cell 9: deterministic-rollout attention
        epoch_NNN_best.pt             ← Cell 6, if SAVE_ATTENTION_PER_EPOCH=True
```

All parquets carry `unit_label` and `seed` columns (added during the
"seed-logging" audit) so concatenating across seeds is a one-liner. Attention
`.pt` files carry `unit_label`, `seed`, `subject`, `repeat_idx`, `rep_seed`
in their dicts.

### 7.2 Top-level — written by Cells 10/11/12/13/14

```
RUN_ROOT/
    pipeline_results.parquet          ← Cell 10: one row per (unit, seed)
    probe_aggregate.parquet           ← Cell 11: per (unit × probe × tag × temp)
    acq_aggregate.parquet             ← Cell 11: per (unit × temp), incl. pass rate
    training_aggregate.parquet        ← Cell 11: per unit
    figures/
        training_curves.png           ← Cell 12
        acq_sanity.png                ← Cell 12
        probe_performance.png         ← Cell 12 (rliable IQM + bootstrap CI)
        performance_profiles.png      ← Cell 12 (rliable P(score ≥ τ))
        path_replay.png               ← Cell 13: one trial's path
        attention_summary.png         ← Cell 13: attention by step-offset, by modality
        attention_heatmap.png         ← Cell 13: head × step-offset heatmap
        playback_*.gif                ← Cell 14: one GIF per playback_trial call
```

### 7.3 Column reference — `*_trajectory.parquet` (the biggest files)

| column | type | source | notes |
|---|---|---|---|
| `step` | int | per-step counter | within-episode |
| `pose_label` | str | `env.get_pose_label()` | canonical-frame for PI/PI+VC, env-frame for VC |
| `grid_x, grid_y, direction` | int | parsed from pose_label suffix | for filtering/plotting |
| `state_idx` | int | `_env_state_idx()` | feeds the DT's encoder |
| `action` | int | model output | 0=L, 1=R, 2=F, 3=pickup, 4=pause |
| `env_reward` | float | env step return | -0.0005..-0.001 or +1.061 |
| `rtg_before_action` | float | `run_dt_episode`'s rolling RTG | the model's RTG input at this step |
| `trial_id` | int | reward-to-reward trial counter | 0-indexed within rollout |
| `trial_timestep` | int | resets at each trial boundary | feeds the model's embed_t |
| `trial_tag` | str | `env.trial_configs[trial_id][3]` | trained / probe_trained / novel / reversal |
| `temp_label` | str | 'det' or 'T1.0' etc. | inference mode |
| `temperature` | float\|None | float for sampled, None for det | |
| `deterministic` | bool | flag | |
| `repeat_idx` | int | 0 for det, 0..N-1 for sampled | |
| `rep_seed` | int | `seed * 1000 + repeat_idx` | env init seed |
| `subject, unit_label, seed` | provenance | | |
| `paradigm` (acq) / `probe_session_type` (probe) | str | env paradigm |
| `source` | str | 'acq' or 'probe' | for unifying the two files |

### 7.4 Column reference — `*_aggregate.parquet` (the analysis tables)

`probe_aggregate.parquet` — one row per `(unit_label, probe_session_type, trial_tag, temp_label)`:

| column | meaning |
|---|---|
| `n_seeds_contributing` | how many of `N_RUNS` produced data here |
| `n_trials_total` | total trials across all seeds × repeats |
| `n_completed_total` | sum of `dt_completed` |
| `success_rate_pooled` | `n_completed / n_trials` — single pooled rate |
| `success_rate_iqm` | interquartile mean of per-seed rates (hand-rolled in Cell 11; rliable's `aggregate_iqm` is used in Cell 12 for plots) |
| `success_rate_p25/p50/p75` | per-seed rate percentiles |
| `median_steps_completed` | typical trial length |

`acq_aggregate.parquet` — same plus `unit_pass_rate_across_seeds` (fraction
of seeds that passed the `ACQ_PASS_COMPLETED_TRIALS` gate).

`training_aggregate.parquet` — per-unit. `best_val_loss_iqm/p25/p50/p75`,
`epoch_to_best_iqm/p25/p75`, `mean_seed_seconds`, `median_seed_seconds`.

---

## 8. Rigorous evaluation patterns

The combination of **multiple seeds × multiple temps × multiple repeats ×
rliable bootstrap CIs** is the rigorous-evaluation backbone. Worth being
explicit about why each layer matters.

### 8.1 N seeds = N independent training runs

5 seeds = 5 independent model inits, train/val splits, batch orders. Captures
optimization noise. If two cohorts (PI vs VC) differ by less than their
across-seed IQR, the difference isn't reliable.

### 8.2 N_REPEATS per sampled temp = action-distribution noise

For each sampled-temp setting, 5 independent rollouts characterize the
policy's stochastic behaviour. With 32 trials per rollout × 5 repeats = 160
trials per (seed, temp) — enough to compute meaningful per-tag completion
rates within a single seed.

### 8.3 Bootstrap stratified CI (rliable) ≠ across-seed percentiles

Naive across-seed IQR underestimates uncertainty for small N. rliable's
stratified bootstrap accounts for the fact that each seed contributes a
matrix of `(repeat × probe × tag)` scores, not a single number. See
Agarwal et al. 2021. Cell 12 uses rliable; Cell 11's hand-rolled IQM/percentiles
are kept for the at-a-glance tabular reads only.

### 8.4 Argmax + sampled = two views of one policy

Deterministic argmax shows the policy's mode — "what the DT thinks is the
best action." Sampled at T=1 shows the full distribution — "what the policy
would do if you let it." Comparing the two:

- Same success rate at det and T=1 → policy is sharp, single-mode.
- Higher success at det than T=1 → there's an action sequence the model
  prefers but the sampled mass leaks elsewhere.
- Higher success at T=1 than det → tail behavior is helping (rare, usually
  indicates the argmax has fallen into a local bad-policy region the model
  itself disprefers).

For comparing to rat behavior, **T=1 is the right comparator** — the rat is
stochastic in the wild. Argmax is the right "did the DT learn something"
sanity check.

### 8.5 Per-tag breakdown is the headline metric

The bare success rate per probe paradigm obscures the actual question.
Within `PI+VC f2 novel route`:

- `trained` trials = same conditions DT was trained on (~10-20% of trials)
- `probe_trained` trials = also same conditions but flagged as
  probe-context (rare)
- `novel` trials = the new-route trials (~80% of trials, the actual probe)

If the DT succeeds on `trained` but fails on `novel`, it overfit to specific
routes and didn't learn navigation. If it succeeds on both, navigation
generalized.

The same logic applies to `reversal` probes (the actual reversal trials are
tagged `reversal`; baseline trials tagged `trained`/`probe_trained`).

Cell 12's `plot_probe_performance` shows tag bars side-by-side; this is the
plot to look at first.

### 8.6 The acq gate guards interpretability

Skipping probes for seeds that didn't pass acq isn't a sample-size loss —
it's an interpretability safeguard. If the DT didn't learn the task on the
canonical conditions, a "novel-route success rate of 0%" tells us nothing
about generalization; it just confirms the model is broken.

`ACQ_PASS_COMPLETED_TRIALS = 24` (75%) is the threshold. If multiple seeds
fail the gate, that's a model-quality / training signal — investigate
training (Cell 12's training curves), not the probes.

---

## 9. How to extend safely

Common modifications and what they touch.

### 9.1 Switching to `grid_cell` encoder

```python
# Cell 1:
ENCODER_TYPE = 'grid_cell'
```

That's the only change in user-facing config. Cell 3's encoder factory
branches automatically:

- Loads `data/lookups/grid_cells_60d.npz` → `STATE_LOOKUP` shape `(484, 60)`
- `_state_idx_for()` uses the `((x-1)*11 + (y-1))*4 + d` formula on rotated
  coords (`rotated_x, rotated_y, rotated_direction`)

The pre-rotation collapse no longer applies — grid_cell vectors differ
under rotation. **The canonical-frame rotation in Cell 3 becomes substantive
for grid_cell mode**, not the no-op it was for pose_visual.

Cell 7's `_env_state_idx` will need to rotate `env.agent_pos / env.agent_dir`
back to canonical at inference. Currently it doesn't — it uses raw env
values for grid_cell. **This is a known gap; address before running grid_cell
in production.** Either:

- Rotate env emissions back to canonical (`_env_state_idx` does it
  per-step), OR
- Force the env to run canonical-frame trials only (already happens via
  `start_goal_location` anchor, but verify it covers non-rotate paradigms).

### 9.2 Switching to `image_cnn` encoder

```python
# Cell 1:
ENCODER_TYPE   = 'image_cnn'
FREEZE_ENCODER = False   # the CNN is trainable end-to-end
```

Cell 5's `ImageCNNEncoder` (3→16 conv + flatten + linear → 60D) is the
trainable encoder. No pretrained checkpoint exists in
`data/encoders/minigrid_cnn.pt` (the path is commented in Cell 1). Adding
one would require pretraining + checkpoint save/load.

Memory: per-batch image tensor (B, K, 21, 21, 3) is ~11 MB (uint8). Not a
concern, but training the CNN adds ~5K params to the model.

### 9.3 Adding a new probe paradigm

If `CornerMazeEnv` defines a new env paradigm (e.g., a PI+VC f3 family):

```python
# Cell 1: add it to SESSION_TYPES
SESSION_TYPES = [
    'PI+VC f2 acquisition',     # idx 0 — leave acquisition first
    'PI+VC f2 novel route',
    ...,
    'PI+VC f3 novel barrier',   # NEW
]
```

Cell 9 will pick it up automatically (`SESSION_TYPES[1:]`).

If the new paradigm uses tags not in `tag_order = ['trained', 'probe_trained',
'novel', 'reversal']` in Cell 12's `plot_probe_performance` and Cell 13,
extend that list. The plot will sort unknown tags alphabetically after
known ones.

### 9.4 Per-cohort comparison (PI vs PI+VC vs VC)

The most likely next step. Three approaches:

**(a) One notebook execution per cohort**, accumulate aggregate parquets
across runs. Pros: clean separation, no notebook state issues. Cons:
multiple manual runs.

**(b) Add a top-level loop in Cell 10** that iterates over groups and
materializes a fresh `TRAINING_UNITS` list per group. Cell 11's aggregation
already groups by `unit_label` so multi-group results would land in the
same aggregate parquets. **Requires re-running everything; better when you're
genuinely sweeping.**

**(c) Pool training data across cohort** via `POOL_SUBJECTS=True`. Trains
one model per cohort on the cohort's pooled data. Loses per-rat variation
but speeds up cohort comparisons.

For the publication-quality cohort comparison, expect **(a) or (b) with
many seeds**. The rliable cross-cohort comparison (`probability_of_improvement`)
needs per-(method, run) score matrices that you get from option (a) by
naming the run dirs consistently.

### 9.5 Changing the rotation policy

E.g., adding a "rotate VC anyway" or "skip rotation entirely" mode. Edit
the `ROTATE_TO_CANONICAL` block in Cell 3:

```python
ROTATE_TO_CANONICAL = GROUP_ALIASES[TRAINING_GROUP] in ('PI', 'PI+VC', 'PI+VC_f1')
```

`n_rotations = 0` everywhere is equivalent to "no rotation"; just make
`ROTATE_TO_CANONICAL = False`. The downstream `_state_idx_for` and the
sessions still build correctly with rotated_* columns equal to raw_*.

### 9.6 Changing `K` (context length)

Adjust `K` in Cell 1. Implications:

- Cell 3's `n_windows_total` shrinks (`(L − K + 1)` per session).
- Cell 5's transformer body sees longer/shorter sequences; computation
  scales as O(K²) for attention.
- Cell 7's `run_dt_episode` adjusts naturally (variable-length context).

Memory cost: see section 4.8 and Cell 10's `_estimate_dt_peak_memory_mb`.

### 9.7 Re-aggregating without re-running

```python
# In a fresh notebook or Python session, after loading Cells 1+2:
from pathlib import Path
out = aggregate_across_seeds(Path('data/runs/<old_run_name>_<ts>'))
# All three aggregate parquets are written to that run_root.
```

`Cell 12`'s plots also accept `run_root=` kwargs:
```python
plot_probe_performance(run_root='data/runs/<old>')
plot_performance_profiles(run_root='data/runs/<old>')
```

Useful for revisiting old runs with new aggregation logic without re-training.

### 9.8 Adding a new diagnostic plot

Add to Cell 13. Pattern: take `run_root` kwarg, glob the relevant
parquets, compute, plot, save to `FIGURES_DIR`, support `show=` kwarg.

Don't add `auto-execute` calls if the plot is per-(probe, trial) specific
(like `playback_trial` would be wasteful to auto-call on every config). For
"summary" plots, do auto-execute so they appear in the standard run.

### 9.9 Common modifications that DO break things

- **Renaming columns in `*_trajectory.parquet`**: Cell 11 and Cells 12-14
  read these by column name. Renaming requires updating all readers.
- **Reordering `SESSION_TYPES` so acquisition isn't index 0**: Cells 8 and 9
  silently use `SESSION_TYPES[0]` and `SESSION_TYPES[1:]`.
- **Changing the `(R, s, a)` token interleave order in Cell 5**: Cell 13's
  attention decoder assumes `j % 3 ∈ {0, 1, 2}` maps to `R, s, a`. Changes
  to interleave order require updates there.
- **Changing `state_idx` semantics** (e.g., to a float vector): the
  `RatWindowDataset` and the encoder both treat `state_idx` as an integer
  for `nn.Embedding`/`nn.Embedding.from_pretrained` lookup. Switching to
  float vectors means redesigning the encoder.

---

## 10. Pitfalls — bugs we caught the hard way

These are the lessons. If you see code that looks suspicious, check this list
first.

### 10.1 Cell 3 had three stacked bugs at one point

Caught during code review by another agent. All three were "I wrote against
assumed schemas instead of inspecting the actual data":

1. **`next(iter(npz.files))` to get the embedding array**. The first key in
   each lookup `.npz` is `pose_labels` (a string array), not the
   embeddings. Fix: explicit key per encoder type.
2. **`df[df['subject_name'].isin(SUBJECTS)]`** when `actions_real_pretrial.parquet`
   has no `subject_name` column. Fix: join via
   `actions.session_id → sessions.subject_id → subjects.subject_name`.
3. **`grp['pose_label'].to_numpy(np.int32)`** when `pose_label` is a string
   like `'pre_e_n_xx_9_6_2'`. Fix: per-encoder lookup table from string to
   row index, with the 4536→295 indirection for `pose_visual`.

All three would have crashed on first run. The lesson: **inspect the actual
data before writing the loader**, not after. The current Cell 3 ends with
schema assertions that fail loudly if any of these break again.

### 10.2 Legacy direction-rotation sign error

`rotate_to_canonical.py:62` uses `(d + n) % 4`. This is **wrong for
`n ∈ {1, 3}`**. We use `(d − n) % 4`. Section 2.5 has the empirical proof.

If you ever port code from the legacy repo's rotation pipeline: check the
direction formula first. The legacy validation didn't catch this — it only
verified positions.

### 10.3 "Cue anchored to N" misunderstood (twice)

Going around this several times before getting it right:

- **Round 1**: assumed `cue_goal_orientation='N/SE'` means "cue is at N
  and goal is at SE for this rat." Cell 7 anchored
  `start_goal_location='SE'` to force cue=N. Wrong reading of the
  semantics.
- **Round 2 (correct)**: `'N/SE'` is the rotational pairing class. The
  rat's actual data spans all 4 cue positions. To match the rat's
  *training-data frame*, we have to anchor cue=N at inference, but the
  anchoring is to match the rotated (canonical) frame the DT trained on,
  not to match the rat's actual experience.

Bottom line: **the anchor exists because we rotate training data to
canonical, not because the rat's data is already there.** If you ever
turn off rotation, also turn off the anchor.

### 10.4 "Multi-cue sessions are anomalous" — wrong

Earlier I called them anomalies. They are the **VC paradigm by design** —
rotating cue is the VC training signal. The notebook's "drop multi-cue PI/PI+VC
sessions" guard is defensive against data anomalies in PI/PI+VC subjects (0
observed in the current dataset), not a general "skip weird sessions" rule.

VC subjects deliberately have many cues per session; rotating their data
would erase the training signal. `ROTATE_TO_CANONICAL` is False for
`training_group == 'VC'`.

### 10.5 Env trial generator chunks of 8 → can't run just 2 trials

The env's `_gen_multi_pool` shuffles trials in chunks of 8 by default. The
generator-produced session is always a multiple of 8 trials (typically 32
for PI/PI+VC f2 acq = 4 chunks).

Trying to run "just 2 trials" by setting `ACQ_INFERENCE_TRIALS = 2` doesn't
shrink the env's session — it generates 32 and we cap consumption at 2 via
`max_trials` in `run_dt_episode`. Slight cost; not a real issue.

To actually get a 2-trial session, you'd pass explicit `trial_configs=` to
the env, bypassing the generator. That's what the yoking pipeline does for
replaying recorded rat trials.

### 10.6 Per-session vs per-trial timestep for `embed_t`

Section 3.2 covers the rationale. The notebook uses per-trial. If anyone
ever switches it back to per-session, expect `MAX_EP_LEN` to need bumping
back to ~3000 and per-position sample count to drop drastically — model
quality will suffer.

### 10.7 Rotation is "implicit" for pose_visual + image_cnn

This is counterintuitive. We rotate data in Cell 3, but for pose_visual the
DT sees identical inputs whether rotated or not (because the npz collapses
rotational variants to the same embedding row). Same for image_cnn (views
are byte-identical under rotation). **Don't skip the rotation step thinking
it's a no-op — it's only a no-op for the *DT inputs*, not for the
diagnostics/visualizations or for future grid_cell experiments**.

### 10.8 Inference-time env reward != training-time RTG

Section 4.2 covers this. The env's per-step reward is `-0.0005` to
`-0.001` and goal reward is `+1.061`. None of these match `COST.step_cost`
or `COST.terminal_reward`. **Don't try to "fix" the env reward to match the
RTG signal** — the RTG is an analytical training signal, and the env's
reward is the actual reinforcement signal. They serve different purposes
and the discrepancy is intentional. Treat env reward as a debug-only signal
in inference logs.

### 10.9 Cells 4-9 smoke artifacts get overwritten by Cell 10

Running cells 4-9 produces a partially-trained model and partial inference
outputs in `seed_dir(_unit, 0)`. When you later run Cell 10, it overwrites
seed 0's artifacts with the full-pipeline results.

If you ever want to preserve smoke artifacts for inspection, copy them out
of `seed_dir(_unit, 0)` before running Cell 10. Or set `N_RUNS = 1` and
run only Cell 10 from a fresh `RUN_ROOT` (auto-timestamped, won't collide).

### 10.10 `info['trial_count']` is in `step()`, not `reset()`

The env's `reset()` returns `(obs, info)` but `info` doesn't yet have
`trial_count` (it's set later by `_handle_reward_well`). `run_dt_episode`
uses `info.get('trial_count', 0) or 0` to handle this safely. If you ever
write new code that reads `info['trial_count']` directly without `.get()`,
the first step will crash.

---

## 11. Verification recipes

Quick scripts to regenerate the key empirical findings. Run from the
corner-maze-rl repo root using `~/venvs/ai-venv/bin/python`.

### 11.1 Verify pose_visual npz coverage of rotated labels

```python
import json, pandas as pd, numpy as np

ARM_CUE_CCW = {'n':'w','w':'s','s':'e','e':'n','x':'x'}
GOAL_CCW    = {'ne':'nw','nw':'sw','sw':'se','se':'ne','xx':'xx'}
def rl(c, n, t):
    for _ in range(n%4): c = t[c]
    return c
def rp(x, y, n):
    for _ in range(n%4): x, y = y, 12-x
    return x, y
def rotate_label(label, n):
    parts = label.split('_')
    if parts[0] in ('expa','expb') or len(parts) < 7: return label
    p, a, c, g, xs, ys, ds = parts[:7]
    nx, ny = rp(int(xs), int(ys), n)
    return f'{p}_{rl(a,n,ARM_CUE_CCW)}_{rl(c,n,ARM_CUE_CCW)}_{rl(g,n,GOAL_CCW)}_{nx}_{ny}_{(int(ds)-n)%4}'

subjects = pd.read_parquet('data/yoked/dataset/subjects.parquet')
sessions = pd.read_parquet('data/yoked/dataset/sessions.parquet')
actions  = pd.read_parquet('data/yoked/dataset/actions_real_pretrial.parquet')

elig_ses = sessions[(sessions['subject_id'].isin(
    subjects[subjects['training_group'].isin(['PI','PI+VC'])]['subject_id']))
    & (sessions['session_phase']=='Acquisition')].copy()
sid_to_n = {}
for _, row in elig_ses.iterrows():
    tc = json.loads(row['trial_configs'])
    cues = set(t[1] for t in tc)
    if len(cues) == 1:
        sid_to_n[row['session_id']] = list(cues)[0]
df = actions[actions['session_id'].isin(sid_to_n.keys())].copy()
df['n_rot'] = df['session_id'].map(sid_to_n)
df['rot'] = df.apply(lambda r: rotate_label(r['pose_label'], r['n_rot']), axis=1)

npz = np.load('data/lookups/pose_visual.npz')
pose_set = set(npz['pose_labels'].tolist())
missing = set(df['rot'].unique()) - pose_set
print(f'rotated label coverage: {len(set(df["rot"].unique())-missing)}/{len(set(df["rot"].unique()))}, missing={len(missing)}')
# Expected: 680/680, missing=0
```

### 11.2 Verify rotational collapse → same embedding row

```python
pose_to_idx = npz['pose_to_idx']
label_to_idx = {lbl: int(pose_to_idx[i]) for i, lbl in enumerate(npz['pose_labels'])}
ok = bad = 0
for orig, rot in df[['pose_label','rot']].drop_duplicates().itertuples(index=False):
    oi, ri = label_to_idx.get(orig), label_to_idx.get(rot)
    if oi is None or ri is None: continue
    if oi == ri: ok += 1
    else: bad += 1
print(f'collapse: same={ok}, different={bad}')
# Expected: 2568, 0
```

### 11.3 Verify direction formula `(d − n) % 4`

```python
import numpy as np
def rp(x, y, n):
    for _ in range(n%4): x, y = y, 12-x
    return x, y
DIR_TO_VEC = {0:(1,0), 1:(0,1), 2:(-1,0), 3:(0,-1)}

# Agent at (8,6) facing E, walks forward to (9,6). One CCW rotation.
sx, sy, d = 8, 6, 0
ex, ey = sx + DIR_TO_VEC[d][0], sy + DIR_TO_VEC[d][1]
n = 1
rsx, rsy = rp(sx, sy, n)
rex, rey = rp(ex, ey, n)
print(f'observed movement after rotation: ({rsx},{rsy}) → ({rex},{rey}) '
      f'= delta {(rex-rsx, rey-rsy)}')
# delta = (0, -1) = direction N = 3
print(f'legacy   formula: (0 + 1) % 4 = {(d + 1) % 4}  vector {DIR_TO_VEC[(d+1)%4]}')
print(f'correct  formula: (0 - 1) % 4 = {(d - 1) % 4}  vector {DIR_TO_VEC[(d-1)%4]}')
# Legacy:  d=1=S, vector (0,1). Wrong.
# Correct: d=3=N, vector (0,-1). Matches observed.
```

### 11.4 Rotated-data divergence check via env replay

Use `src/corner_maze_rl/yoking/diagnostics/check_divergence.py` as the base.
The notebook-side test (used during the rotation implementation) replays
the rotated yoked data through the env and verifies env state matches the
rotated data step-for-step. See conversation history for the full script;
key constraints:

- Pass rotated `trial_configs` to the env (with rotated `(arm, cue, goal)`
  indices).
- Set `start_goal_location` to the rotated first-trial goal name.
- Step the env using actions from the rotated yoked data; compare
  `(env.agent_pos, env.agent_dir)` to `(rotated_x, rotated_y, rotated_dir)`
  at every step.

Verified passing for CM016 (PI+VC), CM023 (PI), CM031 (VC, no-rotation
path), CM037 (PI). 100% step-by-step match across all n_rotations.

---

## 12. Provenance + external references

### 12.1 The wiki

The design conversation is captured in the agent wiki at
`/Users/ryangrgurich/Code/llm-wiki/vaults/sequence-models-navigation/`. The
relevant memory file is
`memory/corner-maze-dt-per-group-models.md` and several iteration logs.

If you have access to that vault, browsing the `MEMORY.md` index gives a
chronological view of design decisions made during the notebook's build.

### 12.2 The Decision Transformer paper

Chen et al. 2021, "Decision Transformer: Reinforcement Learning via Sequence
Modeling," NeurIPS 2021. arXiv:2106.01345. The gym build at
`https://github.com/kzl/decision-transformer/tree/master/gym` is what we
ported. Differences:

- **Discrete actions** (`nn.Embedding`, CE loss) vs continuous (Linear + MSE).
- **Per-trial `embed_t`** vs per-episode.
- **Custom transformer body** vs HF GPT2Model.
- **K=128** vs kzl default 20.
- **No pretrained CNN** for `image_cnn` (kzl uses pretrained for Atari).

### 12.3 The corner-maze-rl-legacy repo

`~/Code/python-dev/corner-maze-rl-legacy/` is the predecessor codebase with
the original rotation pipeline (`yoking/rotate_to_canonical.py`, with the
known direction bug). Use it as a reference for the rotation math + the
analysis-side parquet schemas, but **port the corrected `(d − n) % 4`
direction formula, not the legacy version**.

### 12.4 The rliable paper

Agarwal et al. NeurIPS 2021, "Deep Reinforcement Learning at the Edge of
the Statistical Precipice." arXiv:2108.13264. The Python library is at
`https://github.com/google-research/rliable`. We use it in Cell 12 for IQM
+ stratified bootstrap CIs.

### 12.5 The env

`src/corner_maze_rl/env/corner_maze_env.py`. The trial generator functions
(`gen_pi_vc_f2_acquisition`, etc.) are defined inside
`gen_grid_configuration_sequence` at line ~990. The trial-tag taxonomy
(`'trained'`, `'probe_trained'`, `'novel'`, `'reversal'`) is defined in
`src/corner_maze_rl/env/trial_sequence_gen.py`.

The action constants in `src/corner_maze_rl/env/constants.py`:
- `STEP_TURN_COST = -0.001` (left, right)
- `STEP_FORWARD_COST = -0.0005` (forward, pickup, pause)
- `WELL_REWARD_SCR = 1.061` (goal reward)

The state-encoder lookups in `data/lookups/`:
- `pose_visual.npz`: 4536 strings → 295 unique 60D embeddings
- `grid_cells_60d.npz`: 484 (x, y, d) triples → 484 unique 60D vectors
- `minigrid_views.npz`: 5212 strings → 5212 unique 21×21×3 uint8 views

The yoking pipeline in `src/corner_maze_rl/yoking/`. Key entrypoints:

- `compute_pose_labels.py`: replays rat actions through env to assign
  pose_label per step. **Does not rotate.**
- `build_dataset.py`: consolidates per-session yoked parquets into the
  three dataset parquets.
- `diagnostics/check_divergence.py`: validates yoked replay step-for-step.

---

## 13. Quick start for a fresh agent

If you opened this notebook cold and want to run it:

1. **Read sections 1-4 of this doc** first (~10 min). Especially section 2
   on yoking/rotation and section 4 on inference protocol.
2. **Pick a subject**. Edit Cell 1's `SUBJECTS`, `TRAINING_GROUP`,
   `RUN_NAME`, `SESSION_TYPES`. Match the paradigm strings to your subject's
   group (PI/PI+VC/VC).
3. **Smoke test the pipeline**: set Cell 15's `'smoke'` recipe values in
   Cell 1 (or just `N_RUNS=1, N_EPOCHS=2`), run all cells. ~5 minutes.
   Verify no exceptions, verify the plots look plausible.
4. **Inspect outputs**: open `RUN_ROOT/figures/` for the plots,
   `pipeline_results.parquet` for the run summary,
   `RUN_ROOT/figures/playback_*.gif` for one trial's animation.
5. **Run the headline recipe** (default Cell 1 values). ~90 min on MPS.

If anything looks wrong, check section 10 (pitfalls) first — odds are
you've hit one of the known landmines.

If you're modifying the notebook, see section 9 (extension patterns) and
**run the verification scripts in section 11** after any change to the
rotation logic, encoder choice, or RTG handling. They are fast and catch
the trickiest classes of bugs.
