# Kickoff brief — `11_planning_dt.ipynb` (planning-model extension of the vanilla DT)

> Hand this file to the agent that will build the notebook. It is the spec; the
> Janner paper is background. Name: `11_planning_dt.ipynb` (renamed from the originally-suggested
> `11_beam_dt.ipynb` — the design plans over *known* dynamics with a value head
> rather than a wide beam over a *learned* world model, so "beam" overstated it).

## Mission

Build `notebooks/11_planning_dt.ipynb`: a **planning** model that extends the vanilla
Decision Transformer in `10_vanilla_dt.ipynb` into a search-based controller,
following the Trajectory Transformer (Janner et al. 2021). This is for a
**cohort-comparative** study — does a *planning* agent reproduce each rat cohort's
cue-condition signature on the probe paradigms better/differently than the reactive,
return-conditioned DT? So the vanilla DT is the **control**, not something to throw
away. Build the planner *alongside* it and keep them comparable.

## Read first (in this order)

1. **The spec / translation** — `/Users/ryangrgurich/Code/llm-wiki/vaults/sequence-models-navigation/Wiki/syntheses/planning-model-off-vanilla-dt.md`. This already maps Janner→corner-maze with a staged recipe and the corner-maze-specific friction. Read it before the paper.
2. **The paper page** — `/Users/ryangrgurich/Code/llm-wiki/vaults/sequence-models-navigation/Wiki/papers/janner-2021.md` (method, beam search, TT+Q). PDF at `.../raw/papers/Janner et al. - 2021 - Offline Reinforcement Learning as One Big Sequence Modeling Problem.pdf`.
3. **The existing build** — `docs/10_vanilla_dt.md` (§§3 architecture, 4.2–4.5 RTG/inference, 9.1 grid_cell gap, 12.5 env/lookup provenance) and `notebooks/10_vanilla_dt.ipynb`. Mirror its structure.

## Prime directive (or you will build the wrong thing)

Corner-maze dynamics are **deterministic and known**. TT spends most of its machinery
learning a *generative* dynamics model because MuJoCo is continuous and
stochastic-under-approximation. **Do not learn a generative `p(s'|s,a)` model for v1.**
Plug `CornerMazeEnv`'s exact transition into the search. The target is closer to
**MuZero-with-known-model** (search + a learned value/return prior) than a TT clone.
If you find yourself implementing per-dimension quantile binning of continuous states
or a generative world-model head, stop — that's the MuJoCo path, not this one.

## Notebook structure — mirror `10_vanilla_dt.ipynb`

It uses `## N. Title` sections, 0–15. For `11_planning_dt.ipynb`:

- **`## 0. Colab bootstrap`** — copy cell verbatim from 10 (it is repo-agnostic: clones the repo, editable-installs into the kernel, mounts Drive, starts the background `sync_runs_to_drive` thread, `cd`s into `notebooks/`). Do not change it.
- **`## 1. Configuration`** — THE control panel. A single cell that exposes **every** knob (all of 10's, kept, **plus** all new planning knobs below). Downstream cells read these module-level globals — no hyperparameters hidden in later cells. This is the most important section to get right; see the full inventory at the end.
- **`## 2.`–`## 4.`** — subject/group validation, paths, data load + RTG, `RatWindowDataset` + loaders. Reuse from 10 with the additions noted under "Data deltas."
- **`## 5. Build model`** — add the new heads (below). Keep the custom `GPT2Body` (attention extraction, no positional embedding).
- **`## 6. Training`** — multi-head loss (below).
- **`## 7. Planner`** *(new, replaces/augments 10's `run_dt_episode`)* — the search loop.
- **`## 8.`+** — acquisition sanity, probes, pipeline runner, aggregation, viz, diagnostics, playback. Adapt 10's; keep attention extraction in diagnostics.

Keep a `MODEL_MODE` switch so the same notebook runs either the reactive DT (`'vanilla'`, the control) or the planner (`'planning'`), routing §7 accordingly.

## Architecture deltas (what to add, and where)

The vanilla DT predicts only actions: `action_head = Linear(H, 5)` read at state-token
positions `x[:, 1::3]`. TT predicts the **full joint**. Add, in `## 5`:

1. **State-prediction head** at action-token positions `x[:, 2::3]` → categorical over the **next state**. Use the **non-collapsed `grid_cell` absolute `(x,y,d)` vocabulary (≤484)** as the output space — a single softmax, *not* TT-style per-dim binning. (See gotcha #1.)
2. **Reward/goal head** — near-binary "at goal or not"; the shaped step cost makes reward prediction trivial, so this mostly flags terminal transitions.
3. **Value/return head** — distributional (C51-style) per SwitchTT: categorical over `N_ATOMS` with support `[V_MIN, V_MAX]`, trained as classification on the binned k-step return. This is the planner's search heuristic. (TT's headline sparse-reward win is *value-driven* — see paper-reading note below.)

Loss in `## 6` = weighted sum (`LOSS_W_ACTION·CE_action + LOSS_W_STATE·CE_state + LOSS_W_REWARD·… + LOSS_W_RETURN·…`). Action head stays exactly as in 10. **Under `MODEL_MODE='vanilla'`, disable the state/reward/return heads and zero their losses (`LOSS_W_STATE/REWARD/RETURN = 0`) so the run is bit-for-bit comparable to 10** — nonzero auxiliary losses change the shared-backbone gradients and break exact reproduction (the regression gate).

Planner in `## 7`: at each env step, expand `N_CANDIDATES` action sequences; **roll the known env transition forward** (not a learned model) to `PLAN_HORIZON` (default: the next trial/reward boundary); score each rollout by the value head (or predicted return / steps-to-goal); commit the first action of the best; replan (MPC). Reuse 10's variable-length-context machinery (`ctx_K = min(t+1, K)`) and placeholder-action buffer as the per-candidate context builder.

## Ranked gotchas

1. **The encoder is a trap for prediction.** Default `pose_visual` is a *frozen, input-only* embedding with a 4536→295 **rotational collapse** — lossy, with no decoder back to a state token. For the state-prediction head, use the non-collapsed `grid_cell` absolute `(x,y,d)` vocab (`state_idx = ((x-1)*11 + (y-1))*4 + d`). You may keep `pose_visual` as the *input* stream, but the *output* head must use the absolute vocab. **Fix the `_env_state_idx` canonical-rotation gap first** (docs/10_vanilla_dt.md §9.1: it currently indexes raw env pose without rotating to canonical) — otherwise predicted states won't match the env and the planner scores garbage.
2. **The analytical RTG is NOT a return.** `INFERENCE_RTG = R − c·EXPECTED_OPT_LEN`, re-anchored per trial, is a *conditioning input*, not a predictable cumulative-reward target. You cannot score beams with it. The value/return head needs a genuine return signal, separate from this analytic scalar.
3. **Goal double-counting.** Reward is sparse + shaped, so predicted return ≈ `R − c·(steps-to-goal)` — which `INFERENCE_RTG`/`EXPECTED_OPT_LEN` already encodes, and the cue→goal mapping is already baked into the canonical-frame rotation. Decide the goal enters **either** via RTG conditioning **or** via the planning value, **not both** (`GOAL_VIA` knob; assert exclusivity). The cleanest experiment: a planner whose only goal signal is the value head vs. the reactive return-conditioned DT.
4. **Reuse, don't rebuild.** 10 already gives the beam primitives: custom GPT with attention extraction, **no positional embedding** (only per-trial `embed_t`), `(R,s,a)` interleave exposing hidden states at every position, variable-length context, placeholder-action buffer. Do **not** swap in HuggingFace.
5. **Unlock capacity.** `HIDDEN_DIM` is locked to `ENCODER_DIM=60` (3 layers, 4 heads) for BC. A joint model + search needs more, plus a `H→vocab` state-output projection (which breaks the "encoder dim == hidden dim, no projection" simplification). Decouple them in config; bump defaults (e.g. `HIDDEN_DIM=128`, `N_LAYERS=4–6`).
6. **Short horizons.** Do NOT copy TT's beam width 256 / horizon 15 (seconds per action). Plan **to the next trial/reward boundary** with a small candidate count — the trial-structured RTG re-anchoring already gives you that horizon.
7. **Keep the vanilla DT as the control.** Preserve attention extraction for the analysis; build the planner alongside the reactive DT (via `MODEL_MODE`) rather than rewriting it. Performance on the maze is not the goal — *behavioral match to the rat cohorts* is.

## Two paper-reading corrections (don't over-index on Janner)

- TT's headline sparse-reward win (AntMaze) is **TT+Q** — it imports an IQL Q-function as the search heuristic. *Pure* TT only ties DT/CQL on D4RL. So **planning alone isn't the magic; the value heuristic carries sparse-reward tasks.** Your analog of that Q-function is the distributional return head (or, given known dynamics + shaped reward, an explicit steps-to-goal value).
- TT predicts the **full joint** (states, actions, rewards, returns) — the opposite of DT (actions only). Adding those heads *is* the core structural change; search and value hang off it.

## Staged plan (cheapest-useful-first; commit after each stage)

1. Scaffold the notebook (sections 0–1 + `MODEL_MODE='vanilla'` reproducing 10 exactly as a regression check).
2. Switch the state representation to `grid_cell` for prediction; fix the `_env_state_idx` rotation gap.
3. Add the state + reward heads; verify predicted next-states match the deterministic env on held-out trials.
4. Add the distributional value head (classification on binned k-step return).
5. Add the known-dynamics planner (§7); start with a handful of candidates, horizon = trial boundary.
6. Wire `GOAL_VIA` and run the head-to-head: planner vs. vanilla DT on acquisition + probes.
7. *(Optional, later)* Switch/MoE routed by cue condition (`USE_SWITCH_MOE`) to fold the per-cohort models into one backbone — keep expert count small (SwitchTT found N=4 best and MoE overfits at small data).

## Guardrails / definition of done

- Never modify anything under the yoked dataset or `data/lookups/` — read-only.
- `MODEL_MODE='vanilla'` must reproduce 10's numbers (regression gate).
- Predicted next-states must be checkable against `CornerMazeEnv` (deterministic) — assert agreement on a held-out set before trusting the planner.
- All knobs live in `## 1`. No hidden hyperparameters downstream.
- Keep per-trial `embed_t` reset, the `(R,s,a)` interleave order, and attention extraction.

## Section 1 — full knob inventory

**Keep all of 10's config** (experiment identity, subject/group/`SESSION_TYPES`, `N_RUNS`,
`CostConfig`/`COST`, `LAST_N_ACQ_FOR_RTG`/`EXPECTED_OPT_LEN`/`INFERENCE_RTG`, `K`,
`ENCODER_TYPE`/`ENCODER_DIM`/`FREEZE_ENCODER`, `HIDDEN_DIM`/`N_LAYERS`/`N_HEADS`/`DROPOUT`/
`MAX_EP_LEN`/`NUM_ACTIONS`/`ACTION_MASK_INFERENCE`, training block, logging/eval toggles,
`ACQ_PASS_COMPLETED_TRIALS`, `DEVICE`, `ARTIFACT_ROOT`).

**Add these new blocks:**

```python
# ── Model mode ──────────────────────────────────────────────
MODEL_MODE   = 'planning'        # 'vanilla' = reactive DT (control) | 'planning' = search
#   'vanilla' MUST also set PREDICT_STATE/REWARD/RETURN=False and LOSS_W_STATE/REWARD/RETURN=0
#   so training matches 10 bit-for-bit (the regression gate).

# ── Prediction heads (turn the BC decoder into a joint model) ──
PREDICT_STATE   = True           # categorical next-state head at action positions x[:,2::3]
PREDICT_REWARD  = True           # near-binary at-goal head
PREDICT_RETURN  = True           # value/return head (search heuristic)
LOSS_W_ACTION   = 1.0
LOSS_W_STATE    = 1.0
LOSS_W_REWARD   = 0.5
LOSS_W_RETURN   = 1.0

# ── State-prediction vocabulary (NON-collapsed; see gotcha #1) ──
PREDICT_STATE_VOCAB   = 'grid_cell'   # absolute (x,y,d); pose_visual is input-only & lossy
STATE_VOCAB_SIZE      = None          # auto-derive (≤484) in §3
FIX_ENV_STATE_IDX_ROT = True          # rotate env pose → canonical before indexing (10 §9.1 gap)

# ── Dynamics source ─────────────────────────────────────────
DYNAMICS_SOURCE = 'env_known'    # 'env_known' = use CornerMazeEnv transition | 'learned'

# ── Value / return head (distributional, C51-style) ─────────
VALUE_HEAD = 'distributional'    # 'distributional' | 'point' | 'none'
N_ATOMS    = 51
V_MIN      = None                # tie to shaped-reward scale (≈ R - c·max_trial_len); fill in §3
V_MAX      = None                # ≈ terminal_reward
VALUE_WINDOW = 3                 # k-step window for the return label (SwitchTT uses 3)

# ── Planner (MPC over known dynamics) ───────────────────────
PLANNER       = 'beam'           # 'beam' | 'argmax_candidates' | 'none'
N_CANDIDATES  = 8                # keep small — NOT TT's 256
PLAN_HORIZON  = 'trial_boundary' # plan to next reward boundary, or an int
REPLAN_EVERY  = 1                # MPC cadence
SCORE_BY      = 'distributional_value'  # 'distributional_value' | 'return' | 'steps_to_goal'
K_ACT         = 5                # top-k action tokens for candidate expansion
K_OBS         = 1               # only used if DYNAMICS_SOURCE='learned'

# ── Goal routing (anti-double-count; see gotcha #3) ─────────
GOAL_VIA = 'planning_value'      # 'planning_value' XOR 'rtg_conditioning' — assert not both

# ── Capacity (decouple from encoder dim) ────────────────────
DECOUPLE_HIDDEN_FROM_ENCODER = True   # allow HIDDEN_DIM != ENCODER_DIM (adds input proj)
STATE_OUT_PROJECTION         = True   # H → state vocab logits

# ── Optional Switch/MoE (later stage) ───────────────────────
USE_SWITCH_MOE = False
N_EXPERTS      = 4               # SwitchTT: 4 best, overfits if larger at small data
ROUTE_BY       = 'cue_condition' # 'cue_condition' | 'token'
```

## Data deltas

- §3 must additionally derive: `STATE_VOCAB_SIZE` from the grid_cell lookup, `V_MIN`/`V_MAX` from `COST` and the trailing-acq trial-length distribution, and the per-window **return target** for the value head (genuine cumulative reward over `VALUE_WINDOW`, distinct from the analytic RTG conditioning scalar).
- Add the next-state label to `RatWindowDataset` (absolute grid_cell index of `s_{t+1}`), so the state head has a target.
