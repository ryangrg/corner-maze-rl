# Implementation Plan — `notebooks/11_planning_dt.ipynb` (Planning Decision Transformer)

**Mission.** Extend the vanilla Decision Transformer (`notebooks/10_vanilla_dt.ipynb`) into a Trajectory-Transformer-style *planner* (Janner 2021) that predicts forward trajectories and searches over them — but exploits the fact that corner-maze dynamics are **deterministic and known** by plugging `CornerMazeEnv`'s exact transition into the search (MuZero-with-known-model flavor) instead of learning a generative dynamics model (v1 Prime Directive). The vanilla DT stays in the notebook as the **control** via a `MODEL_MODE` switch, and the scientific target is *behavioral match to the rat cohorts* — how closely the planner reproduces rat route choices — **not** maze-optimal performance (this inverts TT+Q's usual motivation; see §6/F5). Every downstream eval/agg/viz/playback cell must keep working unchanged because the planner emits the identical episode-return-dict and trajectory-row contract as `run_dt_episode`.

> **Two corrections baked into this revision (vs. the first draft), because both were latent correctness bugs, not stylistic issues:**
> 1. **The value target is now the env-scale, action-type-weighted return-to-go along the rat's path** — NOT `R − c·a2r[t+W]`. The latter is algebraically `rtg[t] + c·W` inside a trial (`a2r` is a strict countdown, so `a2r[t+W]=a2r[t]−W`), i.e. the model's own RTG *input* token plus a constant. Regressing a value head against it is exactly the circularity gotcha #2 forbids, and Stage-4 "value tracks `R−c·A`" would be satisfied by a trivial read-the-RTG-token shortcut. The env stream is action-weighted (turns cost 2× forward) so it is genuinely not a function of `a2r`. See §4.
> 2. **RTG neutralization for `GOAL_VIA='planning_value'` happens at TRAINING time, not only at inference** — otherwise the backbone is trained on informative RTG and then fed a placeholder it never saw (train/test mismatch), and the value head learned to lean on the very token we hide. See §2 and §6/F6.

---

## 0. Design decisions locked up front

| Decision | Choice | One-line rationale (source) |
|---|---|---|
| State output vocab | **grid_cell absolute `(x,y,d)` = 484** (`((x-1)·11+(y-1))·4+d`), decoupled from `ENCODER_TYPE` input | Single clean categorical, exactly decodable to a pose; `pose_visual` (295) is rotationally collapsed and undecodable (R1§1; R2 Fact 2; R4§4). |
| Dynamics source | **`DYNAMICS_SOURCE='env_known'`** — roll the real `CornerMazeEnv`, do NOT learn a generative model | Transition is a pure deterministic fn for trial/ITI/pretrial (no in-step RNG on those paths; sequence frozen at `reset()`); MuZero-known-model flavor (R1§3; R4§1.1). |
| Value head | **Distributional C51 (`N_ATOMS=51`) over `[V_MIN,V_MAX]`**, target = **env-scale action-weighted return-to-go along the rat path** | Point MC value is flat under sparse reward; distributional spread carries route-length signal; env-scale target is not a function of the RTG input token (S1 fix; R1§5; R5§0 Reading B revised). |
| Value target scaling | Support `[V_MIN,V_MAX]` computed **empirically** from the env-scale return-to-go distribution over training-acq windows (≈ `[0.53, 1.061]`; see §4) | Support must match the env-scale target, not the analytic COST scale; SwitchTT never specifies support (R5§2). |
| Horizon | **`PLAN_HORIZON` = to the next trial/reward boundary** (rollouts STOP before the finalizing terminal step), small `N_CANDIDATES` | TT's width-256/horizon-15 licensed to shrink (Appendix E); maze trials ≈59–70 steps median (R1§6; R5§2). |
| Goal routing | **`GOAL_VIA ∈ {'planning_value','rtg_conditioning'}`, XOR-enforced, tied at TRAINING time** | Under `planning_value` the RTG token is neutralized in the dataset (train+inference) so the value head is the *sole* goal signal and cannot double-count (S2 fix; R1§F6; R5§0/§3). |
| Reward head | Keep, but expect it to only flag terminals (near-deterministic shaped reward) | Shaped constant step cost makes `r` ≈ "at goal or not"; label is free (`a2r==0`) (R1 arch-delta; R2 Fact 4; R5§3). |
| State head under `env_known` | Trainable but **verification-only** (assert predicted `s'` == env `s'`); `LOSS_W_STATE=0` in vanilla | Env rolls the planner forward, not the model; state head is a consistency check + optional representation regularizer (R1§F7). |
| Per-head gating | **Each head is constructed AND loss-weighted behind its OWN `PREDICT_*` flag**, not a single `MODEL_MODE` check | A user disabling `PREDICT_STATE` must actually get no state head; the `GOAL_VIA` assertion keys on `PREDICT_RETURN` (S7 fix). |
| `MODEL_MODE` regression gate | `'vanilla'` ⇒ `PREDICT_*` all False ⇒ aux heads **structurally not constructed**, `LOSS_W_*=0`, reproduces nb10 bit-for-bit | Weight-zero fails (`0·NaN=NaN`); construct heads LAST + gate structurally so RNG stream is untouched (R2 Fact 6, §4c; R5§6). |
| Forward return contract | **Vanilla returns a bare tensor; planning returns a dict `{logits,state_logits,reward_logits,value_logits}`.** `return_attn=True` always yields a 2-tuple `(primary, attns)` where `primary` is tensor-or-dict. Every call site collapses via `out['logits'] if isinstance(out,dict) else out` | One contract, applied at *every* reader — train loss, val loop, cell-12 smoke test, attention paths — so planning mode is runnable and vanilla stays byte-identical (S5 fix). |
| State idx under grid_cell | **Raw identity `((x-1)·11+(y-1))·4+d` on `env.agent_pos`** for the in-scope canonical-anchored paradigms; `FIX_ENV_STATE_IDX_ROT=False` default | The env already runs in the canonical-N frame for PI/PI+VC acquisition and non-rotate probes (`cell_16:104` anchor), so training and env frames coincide; applying a rotation would *double-rotate* → garbage (S4 fix; R4§3.3). |
| Snapshot mechanism | **Manual per-node lightweight snapshot/restore** (≈10.5 µs), covering the FULL mutated state set incl. `episode_scores`/`episode_data_rows`/`pseudo_session_score`; `deepcopy` as full-`__dict__` oracle in a test | deepcopy=6.8 ms caps beam size; manual per-node snapshot keeps beam extension O(1) per depth (S3+S8 fix; R4§2b). |

---

## 1. Cell-by-cell plan (sections 0–15)

Cell→section map (from R2/R3): `cell_04`=§1, `cell_08`=§3, `cell_10`=§4, `cell_12`=§5, `cell_14`=§6, `cell_16`=§7, `cell_18`=§8, `cell_20`=§9, `cell_22`=§10, `cell_24`=§11, `cell_26`=§12, `cell_28`=§13, `cell_30`=§14, `cell_32`=§15.

### Section 0 — Colab bootstrap — **Reuse verbatim (LOCKED)**
- This is the bootstrap cell (repo clone / path setup). **Do not edit it.** `import copy` (the deepcopy oracle) belongs in the planner/test cell (§7 / `tests/`), NOT here (S11 fix). The one-line mission note and `MODEL_MODE` documentation go in the *title/overview markdown* cell that follows bootstrap, not in the bootstrap code cell.

### Section 1 — Config (`cell_04`) — **Extend**
- KEEP every nb10 knob (full inventory in §3 below). ADD the new blocks using the **authoritative kickoff knob names** (§3): `MODEL_MODE`; `PREDICT_STATE/REWARD/RETURN` + `LOSS_W_ACTION/STATE/REWARD/RETURN`; `STATE_TARGET`/`STATE_VOCAB_SIZE`; `DYNAMICS_SOURCE`; the `VALUE_HEAD` block (`N_ATOMS`, `V_MIN/V_MAX`, `VALUE_WINDOW`, `RETURN_GAMMA`, `RETURN_CROSS_TRIAL`); the `PLANNER` block (`N_CANDIDATES`, `K_ACT`, `K_OBS`, `PLAN_HORIZON`, `REPLAN_EVERY`, `SCORE_BY`, `SCORE_STAT`, `SCORE_ALPHA`); `GOAL_VIA`; `FIX_ENV_STATE_IDX_ROT`; the capacity-decouple block (`DECOUPLE_HIDDEN_FROM_ENCODER`, `STATE_OUT_PROJECTION`); optional `USE_SWITCH_MOE`. Add the assertions (vanilla gate + `GOAL_VIA` XOR keyed on `PREDICT_RETURN`).
- Beam/inference-only knobs (`N_CANDIDATES`, `PLAN_HORIZON`, `REPLAN_EVERY`, `SCORE_*`) may *also* live atop the planning-inference cell per nb10's `cell_18:9-10` / `cell_20:2-3` convention; **model-structural knobs (`MODEL_MODE`, `PREDICT_*`, `LOSS_W_*`, `STATE_VOCAB_SIZE`, `N_ATOMS`, `DECOUPLE_*`, `USE_SWITCH_MOE`) must be in `cell_04`** because `make_dt()` (`cell_12:169-184`) and the training loop read module globals directly (R2 Fact 1).

### Section 2 — Data load — **Reuse verbatim from 10**
- Parquet load unchanged. `a2r` (`actions_to_reward`) is already loaded; `rewarded` exists in the parquet but is NOT in `required_cols` (`cell_08:136-137`) — you don't need it because `a2r==0` is an exact substitute (R2 Fact 4). The env-scale reward-per-step for the value target is *computed* from `action` + `a2r` (§4), so no new parquet column is required.

### Section 3 — `build_sessions` (`cell_08`) — **Extend**
- Add, inside the per-session loop (near `cell_08:184`, after `grp` sorted), the new label arrays: `next_grid_state_idx` + `next_state_valid` mask, `return_target` (**env-scale return-to-go**) + `return_valid` mask, `reward_label`. Exact code in §4. Emit them in the session dict (`cell_08:192-215`).
- Add globals: `STATE_VOCAB_SIZE = N_X*N_Y*N_D = 484` (independent of `ENCODER_TYPE`); the empirical value support `V_MIN/V_MAX` from the env-scale `return_target` distribution over all training-acq windows (§4); and `REWARD_POS_WEIGHT` computed empirically (§4). `N_X,N_Y,N_D=11,11,4` already at `cell_08:11`; grid-idx formula already at `cell_08:170-174`.
- **Training-time RTG neutralization for `planning_value`:** if `MODEL_MODE=='planning' and GOAL_VIA=='planning_value'`, overwrite the per-step `rtg` array that feeds `embed_R` with a fixed placeholder constant `RTG_NEUTRAL` (default `V_MAX`) *in the dataset*, so train and inference see the same neutral token (S2 fix). Under `rtg_conditioning` (and always in vanilla) leave `rtg = R − c·a2r` (`cell_08:184-185`) untouched.
- Keep `EXPECTED_OPT_LEN`/`INFERENCE_RTG` (`cell_08:267-268`) as the **conditioning** anchors for the `rtg_conditioning` arm — the value target must NOT alias them (R2§3; R5§0).

### Section 4 — `RatWindowDataset` (`cell_10`) — **Extend**
- In `__getitem__` (`cell_10:26-31`) slice the new arrays alongside the existing four: `next_state` (long), `next_state_valid` (float), `return_target` (float), `return_valid` (float), `reward_label` (float). Gate their emission behind `MODEL_MODE != 'vanilla'` so the vanilla data path is byte-identical (R2§4c point 4). The `rtg` field is already whatever `build_sessions` wrote (real or neutralized per the arm) — the dataset does not branch on `GOAL_VIA` itself.

### Section 5 — Model (`cell_12`) — **Extend**
- In `DecisionTransformer.__init__`, **after** `self.action_head` (`cell_12:158`), gate **each head on its own `PREDICT_*` flag** and construct them LAST (preserves RNG draw order → vanilla params bit-identical, R2 Fact 6; S7 fix):
```python
if DECOUPLE_HIDDEN_FROM_ENCODER:                    # planning-only; vanilla asserts False
    self.state_in_proj = nn.Linear(ENCODER_DIM, hidden_dim)   # else identity, dims locked at 60
if PREDICT_STATE:
    self.state_head  = nn.Linear(hidden_dim, STATE_VOCAB_SIZE)   # 484
    if STATE_OUT_PROJECTION:                        # optional low-rank readout, default off
        self.state_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                        nn.Linear(hidden_dim, STATE_VOCAB_SIZE))
if PREDICT_REWARD:
    self.reward_head = nn.Linear(hidden_dim, 1)
if PREDICT_RETURN:
    self.value_head  = nn.Linear(hidden_dim, N_ATOMS)           # C51, not 1
    self.register_buffer('z_atoms', torch.linspace(V_MIN, V_MAX, N_ATOMS))
```
- **Forward contract (S5 fix).** `forward(...)` returns a **bare tensor** when all `PREDICT_*` are False (vanilla), else a **dict** `{logits, state_logits?, reward_logits?, value_logits?}` (only constructed heads present). `forward(..., return_attn=True)` **always** returns a 2-tuple `(primary, attns)` where `primary` is that tensor-or-dict — this preserves nb10's `logits2, _attns = _dt(...)` unpack at `cell_12:216` and `cell_16:156`. Every reader collapses with `logits = out['logits'] if isinstance(out, dict) else out`. Apply this idiom explicitly at: the train loss (§6), the val loop (`cell_14:155-171`), **the cell-12 smoke tests** (`cell_12:200-213` and the `return_attn=True` block `:216-225`), and the attention diagnostics (`cell_16:156`, `cell_28`). Without it, `_logits.shape` / `logits.argmax` crash the instant `MODEL_MODE='planning'`.
- Token positions in `forward` (`cell_12:160-166`): **action head unchanged** reads state-token positions `x[:, 1::3]`; **state head + reward head read action-token positions `x[:, 2::3]`** (context has seen `R_t,s_t,a_t` → predicts `s_{t+1}`, `r_t`); **value head reads action-token positions `x[:, 2::3]`** to co-locate with the state head and match SwitchTT read-at-action `Q`-semantics (R1§2; R5§1).
- `make_dt()` (`cell_12:169-184`) seeds then constructs (`:171-172`) — reads the new globals like the rest; no signature change.
- **Capacity decouple** (planning): `DECOUPLE_HIDDEN_FROM_ENCODER=True` adds the `Linear(ENCODER_DIM, HIDDEN_DIM)` input projection above so `HIDDEN_DIM=128, N_LAYERS=4` (TT-scale) is possible; nb10 locks `HIDDEN_DIM==ENCODER_DIM==60` with no projection (`docs:287`). Structural → planning only; vanilla keeps the 60-dim locked path (R1§F9).

### Section 6 — Training (`cell_14`) — **Extend**
- Replace the single CE (`cell_14:115` train, mirror in the val loop `:155-171`) with the per-head gated weighted loss:
```python
out = model(rtg=..., state_idx=..., action=..., trial_timestep=...)
logits = out['logits'] if isinstance(out, dict) else out
action_loss = F.cross_entropy(logits.reshape(-1, NUM_ACTIONS), batch['action'].reshape(-1))
loss = LOSS_W_ACTION * action_loss
if PREDICT_STATE:
    m = batch['next_state_valid'].reshape(-1)
    s_ce = F.cross_entropy(out['state_logits'].reshape(-1, STATE_VOCAB_SIZE),
                           batch['next_state'].reshape(-1), reduction='none')
    loss = loss + LOSS_W_STATE * (s_ce * m).sum() / m.sum().clamp_min(1.0)   # masked
if PREDICT_REWARD:
    reward_loss = F.binary_cross_entropy_with_logits(
        out['reward_logits'].reshape(-1), batch['reward_label'].reshape(-1),
        pos_weight=torch.as_tensor(REWARD_POS_WEIGHT, device=DEVICE))
    loss = loss + LOSS_W_REWARD * reward_loss
if PREDICT_RETURN:
    loss = loss + LOSS_W_RETURN * c51_two_hot_loss(
        out['value_logits'], batch['return_target'], batch['return_valid'],
        V_MIN, V_MAX, N_ATOMS)                                               # §4
```
- **Val loop (S5).** Mirror the collapse idiom at `cell_14:155-171`: `logits = out['logits'] if isinstance(out,dict) else out` before `F.cross_entropy` and `v_correct += (logits.argmax(-1)==...)`. `best_val_loss` / `model_best.pt` selection now uses the **multi-head val loss**; in vanilla all `LOSS_W_*` except action are 0 so it equals nb10's action-only val loss exactly (state that explicitly — it is the gate's correctness argument).
- Optimizer/param logging (`cell_14:80`, `cell_12:191-197`) unchanged — in vanilla the heads aren't constructed, so `trainable` set and counts are identical (R2§4c point 5).

### Section 7 — Episode fn (`cell_16`) — **Extend (new `run_planner_episode` + dispatcher)**
- Keep the entire shell of `run_dt_episode` (`cell_16:97-153, 169-227`). Replace **only** the action-selection block `cell_16:155-167` with a commit-first-action planner. Add `run_episode` dispatcher on `MODEL_MODE`. Reuse `_episode_to_step_rows` (`cell_16:247-282`) verbatim. `import copy` (deepcopy oracle) lives here or in `tests/`. Full detail in §2 below.

### Section 8 — Acquisition (`cell_18`) — **Extend (one-line swap)**
- Change the `run_dt_episode(` call site (`cell_18:83`) to `run_episode(`. Add beam knobs at the module top alongside `ACQ_INFERENCE_TEMPS`/`ACQ_INFERENCE_PROBA_REPEATS` (`cell_18:9-10`). `max_steps = max(n_trials*400, 600)` (`cell_18:58`) unchanged (budgets env.steps, not forwards). `deterministic`→argmax-best-beam (reps=1); `temperature`→sample committed first action / candidate expansion (R3§3).

### Section 9 — Probes (`cell_20`) — **Extend (one-line swap)**
- Change `run_dt_episode(` → `run_episode(` at `cell_20:106`. Keep `save_attention` = det-only. **Scope grid_cell planning to canonical-anchored single-cue paradigms** — do NOT plan the rotate probes (`PI+VC f2 rotate`, `VC novel route rotate`); cue rotates within-session so no single `n_rotations` and the env is not canonical-anchored (R4§3.3). Guard: skip or vanilla-fallback for rotate paradigms under grid_cell planning.

### Section 10 — Pipeline + gate (`cell_22`) — **Reuse verbatim**
- Gate `_acq_max_correct` reads `acq_trajectory.parquet` (`cell_22:78-111`) — works unchanged **iff** the planner emits the standard trajectory columns (it does, via reused `_episode_to_step_rows`). Extend `_memory_precheck` (`cell_22:6-62`) or add a separate planning wall-clock estimate: **forwards/decision ≈ `N_CANDIDATES × PLAN_HORIZON × K_ACT`** (linear, because each live beam node carries its own snapshot and extends one step per depth — see §2/S8), and env-steps/decision ≈ the same. Do NOT use the O(H²) re-replay estimate.

### Section 11 — Aggregation (`cell_24`) — **Reuse verbatim**
- Reads summary parquets' `dt_*` columns (`cell_24:42-46,86-90`). Comes for free from reused per-trial row builders. No change.

### Section 12 — Viz (`cell_26`) — **Reuse verbatim**
- Reads `curves/acq_inference/probes` parquets by name (`cell_26:67-248`). No change. (00A multimodel notebook also keeps working.)

### Section 13 — Diagnostics/attention (`cell_28`) — **Reuse (guard attention)**
- Attention decode assumes 3-token `(R,s,a)` interleave (`cell_28:130-141`). The planner does NOT change token layout (new heads read existing positions), so attention decodes correctly. Define planner "attention" as the attention of the **committed-step forward only** (matches vanilla per-step semantics), or return `None` to make save paths no-ops (`cell_20:170` guards on `episode.get('attention')`). Apply the forward-contract collapse idiom (S5) wherever `cell_28` calls `model(..., return_attn=True)` (R3§0.7).

### Section 14 — Playback (`cell_30`) — **Reuse verbatim**
- Replays committed actions through a fresh env and asserts pose-match (`cell_30:381,393-398`). Valid **iff** the planner commits exactly one executed action per `env.step` and logs only that action — never an imagined rollout state (R3§0.4). No change.

### Section 15 — Recipes / scale profiles (`cell_32`) — **Extend**
- Add planning knobs to the smoke/iter/headline profile overrides (`cell_32:8-33`): `N_CANDIDATES`, `PLAN_HORIZON`, `REPLAN_EVERY`, `K_ACT`, `SCORE_BY`. Document planning cost (`N_CANDIDATES × PLAN_HORIZON × K_ACT` forwards/decision) alongside `N_RUNS`/`N_EPOCHS`.

---

## 2. The planner (Section 7) in detail

### 2.1 Snapshot/restore — full mutated-state set, per-node

Measured **10.5 µs** snapshot+restore vs **6.8 ms** `deepcopy` (~650×) with 0 mismatches over 30 random cross-boundary scripts *when comparing the full observable state*. But `PLAN_HORIZON='trial_boundary'` steps candidate rollouts *through* reward wells (`_handle_reward_well`, `corner_maze_env.py:1622`), which mutates lists the first draft omitted. **These are added to the snapshot** (S3 fix): `pseudo_session_score` (`:1634`, deque), `episode_scores` (`:1529`), `episode_data_rows` (`:1532`). They don't feed the transition, but `restore()` must roll them back or imagined rollouts corrupt the committed episode's logged scores and inflate `episode_data_rows` with fabricated episodes.

```python
import copy  # oracle only

_SNAP_SCALARS = ('agent_dir','step_count','trial_count','sequence_count','session_phase',
    'iti_type','phase_step_count','pretrial_step_count','session_reward','trial_score',
    '_criterion_met','_consecutive_passing_count','episode')
_SNAP_POSES   = ('agent_pos','agent_pose','last_pose','fwd_pos','cur_cell','fwd_cell')
_SNAP_LISTS   = ('reward_timers','maze_state_array','turn_score','trajectory',
    'episode_trial_scores','episode_turn_scores',
    'episode_scores','episode_data_rows')          # <-- added (S3)

def snapshot(env):
    s = {k: getattr(env, k) for k in _SNAP_SCALARS}
    s.update({k: getattr(env, k) for k in _SNAP_POSES})           # tuples/ints immutable
    s.update({k: list(getattr(env, k)) for k in _SNAP_LISTS})
    s['pseudo_session_score'] = copy.copy(env.pseudo_session_score)  # deque, shallow (S3)
    s['_iti_config_idx'] = getattr(env, '_iti_config_idx', None)     # lazily created
    s['grid'] = list(env.grid.grid)                               # shallow cell-ref list copy
    return s

def restore(env, s):
    for k in _SNAP_SCALARS + _SNAP_POSES: setattr(env, k, s[k])
    for k in _SNAP_LISTS: setattr(env, k, list(s[k]))
    env.pseudo_session_score = copy.copy(s['pseudo_session_score'])
    if s['_iti_config_idx'] is not None: env._iti_config_idx = s['_iti_config_idx']
    env.grid.grid = list(s['grid'])
```
Why shallow grid copy is valid: `update_grid_configuration` (`corner_maze_env.py:851`) *replaces* cell references (`put_obj` builds new `Well`/`Barrier`/`Trigger`; `has_reward` set only on freshly-constructed `Well`s at `:884`, never toggled in place), so capturing the reference list captures the layout (R4§2b; critique "sound points").

**Rollouts STOP before the finalizing terminal step (S3/S14).** `_handle_reward_well` returns `terminated=True` on the last trial (`:1650`) and, on last-trial / single-trial / timeout, calls `_log_and_finalize_episode` (`:1525`) which appends to `episode_scores`/`episode_data_rows`. Stepping a terminated `CornerMazeEnv` further is undefined. The planner therefore **halts each candidate rollout at the reward/trial boundary (a2r==0 reached, or env `terminated`/`truncated`) and reads the value head at the pre-terminal leaf** — we score by predicted value, not by realized terminal reward, so the finalize path is never needed inside a rollout. This is why the added snapshot fields are belt-and-suspenders: the primary defense is not stepping into finalize.

**Oracle test (`tests/`)** compares the **entire `env.__dict__`** (minus known-immutable handles like `np_random`, render surfaces) between manual-snapshot rollouts and `deepcopy` rollouts across ≥30 random cross-boundary scripts — not just pose/trajectory. The first draft's "0 mismatches" was pose-only and would have silently missed the `episode_data_rows` accumulation (S3).

### 2.2 State index under grid_cell — identity for the in-scope paradigms (S4)

For PI/PI+VC acquisition and non-rotate probes, `run_dt_episode` already runs the env in the canonical-N frame: `start_goal_location = _trained_goal_for_subject(subject)` (`cell_16:104`) sets the anchor so the trial generator emits `cue=N` (`docs:316-331`), and training data is *also* rotated to canonical (`docs:2.6`, coords `rotated_x/y/direction` at `cell_08:151-153`). Both frames coincide, so **the raw index on `env.agent_pos` is already correct** — no rotation transform:
```python
def _env_state_idx(env):
    if ENCODER_TYPE == 'grid_cell':
        x, y, d = int(env.agent_pos[0]), int(env.agent_pos[1]), int(env.agent_dir)
        if FIX_ENV_STATE_IDX_ROT:                 # default False; ONLY for non-canonical paradigms
            x, y, d = env_pose_to_canonical(x, y, d, SESSION_N_ROTATIONS)
        return ((x-1)*11 + (y-1))*4 + d
    return label_to_row[env.get_pose_label()]
```
- **`FIX_ENV_STATE_IDX_ROT=False` is the default and correct value for everything this notebook plans**, because the rotate paradigms are excluded in §9. Applying a nonzero rotation to an already-canonical pose *double-rotates* → wrong index → garbage beam scores — the exact failure to avoid.
- **Stage-2 verification is an assertion, not a transform:** on a held-out canonical-anchored acquisition rollout, assert `env`'s first-trial cue == N (given the anchor) and that `_env_state_idx(env)` after each step equals the training `state_idx` for the same physical pose. If that holds, identity is proven correct and no fix is needed.
- The `env_pose_to_canonical` helper is retained *only* for a hypothetical future non-canonical paradigm and is dead under the current scope. If ever enabled, `SESSION_N_ROTATIONS` must be explicitly defined (= cue index of the first trial) and the direction sign is `d − n_rotations` (`docs:2.5`); do not enable it blind.

### 2.3 Pseudocode (`run_planner_episode`, `plan_action`)

```python
def run_planner_episode(model, session_type, subject=None, *, target_rtg, max_steps,
        deterministic, save_attention, trial_configs, seed, cue_goal_orientation,
        start_goal_location, max_trials, temperature):
    # --- identical shell to run_dt_episode cell_16:97-153, 169-227 ---
    env, obs, info = _make_and_reset(...)                     # cell_16:106-114
    rtg_buf, state_buf, action_buf, tt_buf, pose_labels, log = _init_buffers(...)  # 118-134
    for t in range(max_steps):
        prefix = build_context_window(rtg_buf, state_buf, action_buf, tt_buf, K)   # cell_16:142-153
        action = plan_action(model, env, prefix, target_rtg, deterministic, temperature)
        # --- everything from cell_16:169 onward reused VERBATIM ---
        obs, reward, term, trunc, info = env.step(action)     # commit ONE real action
        ... strict-correct (182-188), trial-boundary RTG re-anchor* (190-202), buffers (209-218) ...
        if term or trunc or trials_done: break
    return _finalize_log(...)                                 # same dict + scalars as cell_16:221-227
# *under GOAL_VIA=='planning_value' the re-anchor is DISABLED: rtg_buf stays the neutral constant.
```

```python
def plan_action(model, env, prefix, target_rtg, deterministic, temperature):
    root_snap = snapshot(env)                                 # 10.5 µs
    eff_temp  = 1.0 if deterministic else float(temperature)

    # A live beam node carries: its committed action prefix, the DT context buffers filled with
    # those REAL actions, its OWN env snapshot, and a flag if it hit a reward/terminal boundary.
    prior = softmax(model(prefix)['logits'][0, -1, :] / eff_temp)   # action head @ last state token
    prior = prior * env.get_action_mask()                    # env.py:1589 -> ~2-3 legal of 5
    beams = []
    for a0 in top_k_actions(prior, K_ACT):                   # K_ACT ABSOLUTE count (5 ok)
        restore(env, root_snap); _, _, term, trunc, info = env.step(a0)
        node = new_node(first_action=a0, ctx=append_real(prefix, a0, env_state=_env_state_idx(env)),
                        snap=snapshot(env), done=(_at_boundary(info, term, trunc)))
        beams.append(node)

    for depth in range(1, PLAN_HORIZON):                     # or until next trial/reward boundary
        nxt = []
        for node in beams:
            if node.done: nxt.append(node); continue         # frozen at its pre-terminal leaf
            prior = softmax(model(node.ctx)['logits'][0, -1, :] / eff_temp) * mask_of(node)
            for a in top_k_actions(prior, K_ACT):
                restore(env, node.snap); _, _, term, trunc, info = env.step(a)  # ONE step, no re-replay
                nxt.append(new_node(first_action=node.first_action,
                    ctx=append_real(node.ctx, a, env_state=_env_state_idx(env)),
                    snap=snapshot(env), done=(_at_boundary(info, term, trunc))))
        beams = keep_top(nxt, N_CANDIDATES, key=partial_score)   # prune to beam width

    scores = [score_leaf(model, node) for node in beams]         # value head ONLY (see 2.4)
    restore(env, root_snap)                                      # leave env clean for the committed step
    best = beams[argmax(scores)] if deterministic else beams[sample(softmax(scores/eff_temp))]
    return best.first_action                                     # MPC: commit first, replan (REPLAN_EVERY)
```

### 2.4 Scoring — value head only, single scale (S1/S6/S9)

- **`score_leaf` reads the C51 value head at the leaf's last action token** `x[:,2::3]`, where the context is filled with the **real committed candidate actions** (NOT the placeholder `[0]`). This is the crux of the S9 fix: the DT keeps a placeholder action at the *action-prediction* position (`cell_16:147`, `a_w = a_real + [0]`) only when the *action head* is queried; when the *value/state head* is read to score a candidate, the last action token must be the committed candidate action, because real next-states come from the known-dynamics roll, not from prediction. Two distinct context shapes:
  - **action-prediction context** (proposal prior at a node): `[..., s_leaf, PLACEHOLDER=0]`, read action head at `x[:,1::3]` last state token.
  - **value-scoring context** (leaf ranking): `[..., a_committed, s_leaf]` with `a_committed` real, read value head at `x[:,2::3]` last action token.
- **Single scale, no path-reward add (S6).** `score_leaf` returns the leaf **return-to-go**: `p=softmax(value_logits); V=(p*z_atoms).sum()` for `SCORE_STAT='mean'`, or a lower/upper quantile (`'quantile'`, `SCORE_ALPHA`) for risk-sensitivity. **Do NOT add env-rolled path reward** — the value target *is* return-to-go from the leaf, so adding root→leaf reward double-counts, and mixing env-scale path reward with the value head would only be dimensionally consistent because both are now env-scale (the S1 fix put the value target on the env scale) — but it is still a double-count. Score by leaf value alone. `partial_score` (for pruning at intermediate depths) uses the same leaf-value read at the current frontier.
- **`SCORE_BY='distributional_value'` is now the sound default** because the value target is the env-scale action-weighted return-to-go, which is *not* a function of the RTG input token (S1). `SCORE_BY='steps_to_goal'` remains available but is lossy: turns cost 2× forward (`STEP_TURN_COST=-0.001` vs `STEP_FORWARD_COST=-0.0005`), so step-count ≠ value (R1§F1).
- **`K_OBS` is inert under `env_known`** and is documented as such in §3 — the next state is the env's exact transition, no observation sampling; branching is action-only via `K_ACT` (absolute count; a top-20% rule would collapse to 1 of 5 actions — R1§F3). It is kept in the config for parity with the TT knob vocabulary and forward-compatibility with a learned-dynamics v2.
- **`env.get_action_mask()`** (`env.py:1589`) prunes to ~2-3 legal actions (pickup only at corners, right-turn disabled at well-exit) — cheap branch reduction.

### 2.5 Cost, complexity, and the beam is linear (S8)

Because **every live beam node carries its own snapshot**, extending a node is a single `restore(node.snap)` + one `env.step` — not a re-replay of the whole prefix. Total per decision: `O(N_CANDIDATES × PLAN_HORIZON × K_ACT)` env steps and the same order of forwards. The precheck formula in §10 uses this linear estimate. The first-draft `expand` that restored to `root` and replayed the full prefix was `O(N·H²)` (a ~60× under-budget on `H≈59` horizons) — replaced.

### 2.6 Contract preservation

`plan_action` returns a single committed `int`; the reused shell logs only that action, so `_episode_to_step_rows` (`cell_16:247-282`) produces identical trajectory columns and cells 8–15 work unchanged. Optionally attach non-breaking diagnostic columns (`planned_return`, `n_candidates`, `plan_horizon`, `plan_agreed_with_argmax`) via the provenance dict (R3§4). `deterministic`↔`temperature` semantics preserved: `deterministic=True` → argmax-best-beam, reps=1 (reproducible, `eff_temp=1.0`); `temperature=float` → sampled committed action and/or sampled candidate expansion (R3§3).

**⚠️ Keep the planner on `*_acquisition`/probe (non-exposure) paradigms:** `_handle_exposure_well` (`env.py:1612`) and `reset` (`:938`) draw `random.gauss` *inside* `step` from the global module — not captured by snapshot or deepcopy → non-reproducible rollouts (R4§1.1). Acquisition/trial/ITI/pretrial paths have no in-step RNG.

---

## 3. Config knob inventory (Section 1 / `cell_04`)

**KEEP all nb10 knobs** (grouped by `cell_04`'s banners, defaults/lines from R2§1):
- *Experiment identity* (`:6-11`): `RUN_NAME`, `EXPERIMENT_NOTES`.
- *Subject/group/session* (`:13-39`): `TRAINING_GROUP='pi_vc'`, `SUBJECTS=['CM006']`, `POOL_SUBJECTS=False`, `SESSION_TYPES=[...5 strings]`, `ACQ_INFERENCE_TRIALS=32`.
- *Seed sweep* (`:41-42`): `N_RUNS=2`.
- *Reward/cost LOCKED* (`:44-49`): `CostConfig(terminal_reward=1.0, step_cost=0.001)`, `COST`.
- *Inference RTG conditioning* (`:51-59`): `LAST_N_ACQ_FOR_RTG=2`, `EXPECTED_OPT_LEN=None`, `INFERENCE_RTG=None` (filled by cell_08).
- *Windowing* (`:61-63`): `K=128`.
- *State encoder* (`:65-79`): `ENCODER_TYPE='pose_visual'`, `ENCODER_DIM=60`, `FREEZE_ENCODER=True`.
- *DT arch* (`:81-90`): `HIDDEN_DIM=60`, `N_LAYERS=3`, `N_HEADS=4`, `DROPOUT=0.1`, `MAX_EP_LEN=None`, `NUM_ACTIONS=5`, `ACTION_MASK_INFERENCE=False`.
- *Training* (`:92-101`): `N_EPOCHS=50`, `BATCH_SIZE=512`, `LEARNING_RATE=1e-4`, `WEIGHT_DECAY=1e-4`, `WARMUP_STEPS=None`, `GRAD_CLIP=0.25`, `EARLY_STOP_PATIENCE=5`, `VAL_FRAC=0.15`, `NUM_WORKERS=0`.
- *Logging toggles* (`:103-107`), *Probe/eval* (`:109-113`: `EVAL_EPISODES_PER_PROBE=1`, `FROZEN_DETERMINISTIC=False`, `SAVE_PROBE_ATTENTION=True`, `RUN_SMOKE_TESTS=False`), *Pipeline gating* (`:115-120`: `ACQ_PASS_COMPLETED_TRIALS=24`), *Misc* (`:122-134`: `FORCE_CPU`, `DEVICE`, `REPO_ROOT`, `ARTIFACT_ROOT`).

**ADD new planning blocks — authoritative kickoff knob names (S10):**
```python
# --- MASTER GATE ---
MODEL_MODE = 'vanilla'                 # 'vanilla' | 'planning'

# --- prediction heads (each gates its OWN head construction AND loss; S7) ---
PREDICT_STATE, PREDICT_REWARD, PREDICT_RETURN = False, False, False   # True in planning
LOSS_W_ACTION, LOSS_W_STATE, LOSS_W_REWARD, LOSS_W_RETURN = 1.0, 0.0, 0.0, 0.0
REWARD_POS_WEIGHT = None                # computed in cell_08 = (#a2r!=0)/(#a2r==0) ~ 60
STATE_TARGET = 'grid_cell'              # target vocab for state head, indep of ENCODER_TYPE
STATE_VOCAB_SIZE = 11*11*4              # 484 (finalized in cell_08)

# --- dynamics ---
DYNAMICS_SOURCE = 'env_known'          # v1: never a learned generative model

# --- VALUE_HEAD (distributional C51) ---
N_ATOMS = 51
V_MIN, V_MAX = None, None              # computed in cell_08 from ENV-SCALE return_target dist
VALUE_WINDOW = 3                       # look-ahead offset W; return-to-go read (NOT a W-step sum)
RETURN_GAMMA = 1.0                     # undiscounted
RETURN_CROSS_TRIAL = False             # clamp look-ahead to current trial

# --- PLANNER ---
N_CANDIDATES  = 8                      # beam width (small)
K_ACT         = 5                      # ABSOLUTE action-branch count, not a percentage
K_OBS         = 1                      # INERT under env_known (documented) — next state is exact
PLAN_HORIZON  = 'trial_boundary'       # or int; rollouts STOP before the finalizing terminal step
REPLAN_EVERY  = 1                      # MPC receding horizon
SCORE_BY      = 'distributional_value' # 'distributional_value' | 'steps_to_goal'
SCORE_STAT    = 'mean'                 # 'mean' | 'quantile'
SCORE_ALPHA   = 0.1                    # quantile level if SCORE_STAT=='quantile'

# --- state-idx rotation (default OFF; env is canonical-anchored for in-scope paradigms; S4) ---
FIX_ENV_STATE_IDX_ROT = False          # True ONLY for non-canonical paradigms (none in scope)

# --- goal routing (XOR; tied at TRAINING time; S2) ---
GOAL_VIA = 'planning_value'            # 'planning_value' | 'rtg_conditioning'
RTG_NEUTRAL = None                     # neutral RTG constant (default V_MAX), used when planning_value

# --- capacity decoupling (planning only) ---
DECOUPLE_HIDDEN_FROM_ENCODER = False   # True => input projection ENCODER_DIM->HIDDEN_DIM
STATE_OUT_PROJECTION = False           # True => 2-layer state-head readout (optional)

# --- optional MoE (NOT v1) ---
USE_SWITCH_MOE = False
N_EXPERTS, ROUTE_BY = 4, 'cue_condition'
```
> **Renames vs. first draft, documented per S10:** `PREDICT_VALUE`→`PREDICT_RETURN`; `W_*`→`LOSS_W_*`; `STATE_TARGET_VOCAB_SIZE`→`STATE_VOCAB_SIZE`; added the previously-missing `FIX_ENV_STATE_IDX_ROT`, `STATE_OUT_PROJECTION`, `K_OBS` (marked inert), `RTG_NEUTRAL`. Names now match the kickoff contract so 00A-style downstream tooling keys correctly.

**Assertions (in `cell_04`):**
```python
if MODEL_MODE == 'vanilla':
    assert not (PREDICT_STATE or PREDICT_REWARD or PREDICT_RETURN)
    assert LOSS_W_STATE == 0.0 and LOSS_W_REWARD == 0.0 and LOSS_W_RETURN == 0.0
    assert not DECOUPLE_HIDDEN_FROM_ENCODER and not USE_SWITCH_MOE and not FIX_ENV_STATE_IDX_ROT
    # => must reproduce nb10 bit-for-bit (structural gate, not weight-zero)
assert GOAL_VIA in ('planning_value', 'rtg_conditioning')
if GOAL_VIA == 'planning_value':
    assert PREDICT_RETURN, "planning_value needs the value head as the sole goal signal"
    # RTG is neutralized in the DATASET (train+inference) and the trial-boundary re-anchor is
    # DISABLED in run_planner_episode. See §3-cell_08 note and §2.3. This is a training-time switch.
```

---

## 4. Data deltas (Section 3/4)

All arrays computed in `build_sessions` (`cell_08:181-215`); sliced in `RatWindowDataset.__getitem__` (`cell_10:26-31`). Grid dims `N_X,N_Y,N_D=11,11,4` at `cell_08:11`; rotated coords (`rotated_x/y/direction`) computed every row regardless of `ENCODER_TYPE` (`cell_08:151-153`) — so the grid_cell target is available even when input is `pose_visual`.

**`STATE_VOCAB_SIZE = 484`** — state-head output dim is the grid_cell absolute vocab, NOT 295 (pose_visual is rotationally collapsed and undecodable). `grid_cells_60d.npz` verified: `keys(484,3)=(x,y,d)`, `vectors(484,60)`, dense, index formula `((x-1)*11+(y-1))*4+d` exact for all 484 rows (R4§4). Decode inverse for diagnostics: `d=idx%4; r=idx//4; y=r%11+1; x=r//11+1`.

**Next-state label + validity mask** (R2§2b, Fact 5 — last window falls off session end):
```python
gx = grp['rotated_x'].to_numpy(np.int32); gy = grp['rotated_y'].to_numpy(np.int32)
gd = grp['rotated_direction'].to_numpy(np.int32)
grid_state_idx      = (((gx-1)*N_Y + (gy-1))*N_D + gd).astype(np.int64)   # [0,484)
next_grid_state_idx = np.concatenate([grid_state_idx[1:], grid_state_idx[-1:]])
next_state_valid    = np.ones(L, np.float32); next_state_valid[-1] = 0.0   # last token masked
```
Keep cross-trial next-state ON (predicting the true next physical state across a reward is fine).

**Value target — ENV-SCALE action-weighted return-to-go (the S1 fix).** The value target must NOT be `R − c·a2r[t+W]`, which equals `rtg[t] + c·W` inside a trial and is therefore the RTG *input* token plus a constant (circular). Instead cumulate the env's **realized, action-type-weighted** per-step reward to the next reward within the trial — genuinely distinct because `a2r` counts steps while the env weights turns 2× forward:
```python
# env-scale per-step cost by action type (matches _compute_reward: turns 2x forward)
STEP_COST = {0: STEP_TURN_COST, 1: STEP_TURN_COST, 2: STEP_FORWARD_COST, 4: 0.0}   # -0.001,-0.001,-0.0005,0
act = grp['action'].to_numpy(np.int32)
a2r = grp['actions_to_reward'].to_numpy(np.int32)             # per-trial countdown ...2,1,0
r_env = np.where(a2r == 0, WELL_REWARD_SCR,                    # 1.061 at the rewarded well step
                 np.vectorize(lambda a: STEP_COST.get(a, STEP_FORWARD_COST))(act)).astype(np.float32)

W = VALUE_WINDOW
g = np.full(L, np.nan, np.float32); valid = np.zeros(L, np.float32)
for t in range(L):
    # find next reward index j0 (a2r==0) at or after t, within current trial
    rel = np.where(a2r[t:] == 0)[0]
    if rel.size == 0: continue                                 # no reward ahead in-trial -> mask
    j0 = t + rel[0]
    if not RETURN_CROSS_TRIAL and j0 - t > (a2r[t] + 1): continue   # boundary guard
    g[t] = r_env[t:j0+1].sum()                                 # env return-to-go to next reward
    valid[t] = 1.0
return_target = g; return_valid = valid                        # NaN/invalid masked in c51 loss
```
Notes: (a) `VALUE_WINDOW` is retained as the C51 label's read-offset knob but the value **target is the full in-trial return-to-go**, not a `W`-step sum — a bare `W`-step reward sum is the degenerate `≈ -0.003` failure the head exists to fix (R1§F2). (b) This target is env-scale, so `V_MIN/V_MAX` and `score_leaf` are all env-scale — one consistent scale end to end (S6). (c) It is a **separately-named** array; it never aliases the `rtg` input (R2§3).

**C51 support (`V_MIN`/`V_MAX`) — empirical, env-scale (S1 consequence):**
```python
finite = np.isfinite(all_return_targets)                       # pooled over training-acq windows
V_MAX = float(WELL_REWARD_SCR)                                 # 1.061 hard upper bound (goal step)
V_MIN = float(np.min(all_return_targets[finite]))             # ~1.061 - 0.001*L_hi ~ 0.53 (max path, all-turn)
# finer resolution option: V_MIN = float(np.percentile(all_return_targets[finite], 1))  # ~0.83
```
`L_hi = max trial length = 533` gives the conservative floor `≈ 1.061 − 0.001·533 = 0.528` (all-turn worst case; realistic paths sit higher because most steps are forward). `Δz = (V_MAX−V_MIN)/(N_ATOMS−1) ≈ 0.0107`. Default to the empirical `min` (zero-clip in the two-hot loss anyway); switch to `p1` if resolution matters. Record as a design decision (SwitchTT never specifies support).

**C51 two-hot loss — with empty-mask guard (S12):**
```python
def c51_two_hot_loss(logits, g, valid_mask, V_MIN, V_MAX, N_ATOMS):   # logits (N,A), g (N,), mask (N,)
    valid = valid_mask.reshape(-1).bool() & torch.isfinite(g.reshape(-1))
    if valid.sum() == 0:                                       # S12: no finite targets in batch
        return logits.new_zeros(())                            # zero, do not poison shared grads
    logits, g = logits.reshape(-1, N_ATOMS)[valid], g.reshape(-1)[valid]
    dz = (V_MAX - V_MIN) / (N_ATOMS - 1)
    b  = (g.clamp(V_MIN, V_MAX) - V_MIN) / dz
    lo = b.floor().long(); up = b.ceil().long()
    m  = torch.zeros_like(logits)
    m.scatter_add_(1, lo.unsqueeze(1), (up.float() - b).unsqueeze(1))
    m.scatter_add_(1, up.unsqueeze(1), (b - lo.float()).unsqueeze(1))
    integral = (lo == up); m[integral] = 0.0; m[integral, lo[integral]] = 1.0
    return -(m * F.log_softmax(logits, -1)).sum(-1).mean()
```

**At-goal / reward label + pos_weight — computed, not hardcoded (S13):**
```python
reward_label = (a2r == 0).astype(np.float32)                  # verified rewarded==1 <=> a2r==0
# empirical imbalance (trials ~59-70 steps -> pos rate ~1/60-1/70, NOT 1/40):
REWARD_POS_WEIGHT = float((reward_label == 0).sum() / max((reward_label == 1).sum(), 1))
```
`rewarded==1 ⇔ a2r==0` verified for all 10,416 reward rows, exactly one True per trial (R2 Fact 4). BCE-with-logits with the computed `pos_weight` (≈60, consistent with the ~59–70-step trial length used at `cell_08`).

---

## 5. Staged build order + verification gates

Cheapest-useful-first; each stage is a commit boundary with a concrete gate.

**Stage 1 — Scaffold + regression gate.** Copy nb10 → nb11, add `MODEL_MODE` + all config blocks + assertions, add dispatcher `run_episode`, gate every aux head **on its own `PREDICT_*` flag** (S7), apply the **forward-contract collapse idiom at every reader** — train loss, val loop, cell-12 smoke tests, attention paths (S5). Nothing planning yet.
- **VERIFY:** `MODEL_MODE='vanilla'` reproduces nb10. Same `RUN_NAME`/seed; assert equal `curves.parquet` (`cell_14:219`) and equal `model_best.pt` state_dict. Bit-for-bit, and the cell-12 smoke test + val loop must run without `AttributeError` (proves the collapse idiom is wired). If not bit-for-bit, the structural gate is broken.

**Stage 2 — grid_cell switch, identity state-idx.** Set `ENCODER_TYPE='grid_cell'` (or keep pose_visual input + grid_cell target). `FIX_ENV_STATE_IDX_ROT=False`. Scope to canonical-anchored single-cue paradigms.
- **VERIFY:** on a held-out acquisition rollout, assert the env's first-trial cue == N (canonical anchor holds) and `_env_state_idx(env)` after each step equals the training `state_idx` for the same physical pose. If they match, identity is proven and no rotation transform is applied. (Do NOT enable `FIX_ENV_STATE_IDX_ROT` — it would double-rotate; S4.)

**Stage 3 — State + reward heads with env-agreement assertion.** `PREDICT_STATE=PREDICT_REWARD=True`, train with masked CE + BCE (`pos_weight` computed).
- **VERIFY (definition of done for trusting the planner):** predicted next-state `argmax(state_logits)` matches `CornerMazeEnv`'s actual next `state_idx` on a held-out set — **assert before trusting the planner.** Reward head fires on `a2r==0` steps. Under `env_known` the planner rolls the ENV forward, so this is a consistency check, not a dependency (R1§F7).

**Stage 4 — Distributional value head.** `PREDICT_RETURN=True`; add C51 head + `z_atoms` + two-hot loss + empirical env-scale `V_MIN/V_MAX`. Value target = env-scale return-to-go (§4).
- **VERIFY:** predicted mean `(p·z).sum()` tracks the env-scale return-to-go on held-out trials (calibration). **This is now a non-trivial test** because the target is not a function of the RTG token (S1) — a passing calibration means the head learned state-grounded value, not a read-the-RTG shortcut. **Measure spread**: if spread is negligible, decide whether `distributional_value` earns its cost vs. a point head (R5§2 open question).

**Stage 5 — Known-dynamics planner.** Implement full-state `snapshot`/`restore` (incl. `episode_scores`/`episode_data_rows`/`pseudo_session_score`; S3), per-node snapshot beam (linear cost; S8), two context shapes for action-prediction vs value-scoring reads (S9), rollouts halting before the finalizing terminal step (S3/S14). Register the manual-vs-`deepcopy` **full-`__dict__`** oracle in `tests/`.
- **VERIFY:** oracle test — manual rollouts equal `deepcopy` rollouts over ≥30 random cross-boundary scripts, comparing the **entire env `__dict__`** (minus immutable handles), 0 mismatches. Playback (`cell_30`) pose-match assertion passes on planner trajectories (one committed action/step, no imagined-state leakage). Acq gate (`cell_22`) runs; probes fire. Confirm the wall-clock precheck uses the linear `N_CANDIDATES × PLAN_HORIZON × K_ACT` estimate.

**Stage 6 — `GOAL_VIA` head-to-head (training-time switch; S2).** `planning_value`: RTG neutralized in the dataset for train AND inference, re-anchor disabled, value head is the sole goal signal. `rtg_conditioning`: real analytic RTG, value head is a spread-only tie-breaker.
- **VERIFY:** the clean contrast — planner-whose-only-goal-is-the-value-head vs reactive return-conditioned DT — is now genuinely clean because neither arm double-counts and the `planning_value` backbone was trained on the neutral RTG it sees at test. Compare **behavioral-match metrics to the rat cohort, NOT maze reward** (F5).

**Stage 7 — Optional Switch/MoE (NOT v1).** Only after Stage 6. `N_EXPERTS=4` (paper: N=8/16 worse than 4), top-1 routing by cue condition.
- **VERIFY:** watch the train/val gap the paper flags (MoE drives train loss down, val loss up at this data scale). Keep it separate so regressions stay attributable.

---

## 6. Risks / gotchas / open questions

**Kickoff's 7 ranked gotchas:**
1. **Encoder-is-a-trap.** `pose_visual` (295) is rotationally collapsed and undecodable to a unique `(x,y,d)` — never a state-head target. Build the 484 grid_cell target vocab from rotated coords regardless of input encoder (R2 Fact 2). For the in-scope canonical-anchored paradigms the env frame already coincides with the training frame, so `_env_state_idx` is **identity** — do NOT apply a rotation (it would double-rotate; S4). `FIX_ENV_STATE_IDX_ROT` stays False.
2. **Analytical-RTG-is-not-a-return.** `rtg[t]=R−c·a2r[t]` is a design-locked conditioning INPUT. **The value target is NOT `R−c·a2r[t+W]`** (that is `rtg[t]+c·W`, i.e. the input token plus a constant — circular; S1). Use the separately-named **env-scale action-weighted return-to-go** (§4), which is not a function of `a2r`.
3. **Goal-double-counting.** `GOAL_VIA` XOR, tied at **training** time. Under `planning_value` the RTG token is neutralized in the dataset for train+inference and the trial-boundary re-anchor is disabled, so the value head is the sole goal signal and the beam scorer reads value alone (no path-reward add; S2, S6).
4. **Reuse-don't-rebuild.** Keep the entire `run_dt_episode` shell + `_episode_to_step_rows` + all eval/agg/viz/playback cells; replace ONLY `cell_16:155-167`. But **the planning forward contract must be collapsed at every reader** (train loss, val loop, cell-12 smoke test, attention) or "reuse verbatim" cells crash under `MODEL_MODE='planning'` (S5). The gate reads the trajectory parquet, not the return dict — emit standard columns or probes silently never run (R3§0.2).
5. **Unlock-capacity.** nb10 locks `HIDDEN_DIM==ENCODER_DIM==60` (`docs:287`); planning needs `DECOUPLE_HIDDEN_FROM_ENCODER` (+ optional `STATE_OUT_PROJECTION`) for TT-scale 128-dim/4-layer. Structural → planning only (R1§F9).
6. **Short-horizons / value-window.** `VALUE_WINDOW=3` is the C51 read-offset, but the value **target is the full in-trial return-to-go**, not a 3-step sum (a 3-step *sum* reproduces the MC-uninformative failure the head exists to fix; trials ≈59–70 steps). Plan to the trial boundary (R1§F2; R5§0).
7. **Keep-vanilla-as-control.** Structural per-`PREDICT_*` gate (not weight-zero — `0·NaN=NaN`), heads constructed LAST, vanilla returns bare tensor, val loss reduces to action-only when `LOSS_W_*=0`. Bit-for-bit regression (state_dict + `curves.parquet`) is the Stage-1 gate (R2 Fact 6, §4c).

**Two paper-reading corrections:**
- **TT+Q's value heuristic carries sparse reward, not planning alone.** Pure-TT (MC RTG) ties DT/CQL on dense D4RL; TT+Q (DP Q from IQL) wins AntMaze (84.0 vs DT 11.8; DT 0.0 on non-Umaze). The corner-maze analog is the learned distributional value head — the load-bearing piece under sparse+shaped reward (R1§5).
- **TT predicts the full joint** (state+action+reward+RTG heads, teacher-forced), unlike DT (actions only). The state/reward/value heads are the core structural upgrade; search and value hang off them (R1§2).

**Additional facts surfaced (R1–R5):**
- **F5 inversion — DO NOT "upgrade" the value head to an optimality Q.** TT+Q needs an *improved* Q for maze-optimality; here the goal is *behavioral match to rats*, so a value head trained on the **rat's own env-scale return-to-go is the correct behavior-policy heuristic**. A DP/optimal Q would push the planner *away* from rat-like behavior and break the comparison. Note explicitly in the notebook (R1§F5).
- **Two reward scales — the value head now lives entirely on the env scale.** Analytic RTG conditioning uses `R=1.0, c=0.001` (input token only); the value target, `V_MIN/V_MAX`, and `score_leaf` all use the env stream `WELL_REWARD_SCR=1.061`, `STEP_FORWARD_COST=-0.0005`, `STEP_TURN_COST=-0.001` (turns 2× forward). The scorer never mixes scales and never adds path reward to leaf value (S6). `steps_to_goal` remains a lossy `SCORE_BY` proxy because turns ≠ steps (R1§F1).
- **Exposure is non-reproducible under snapshot** (`random.gauss` inside `step`/`reset`) — scope the planner to `*_acquisition`/probe (R4§1.1).
- **Rotate probes are multi-frame** under grid_cell — no single `n_rotations`, env not canonical-anchored; skip or vanilla-fallback (R4§3.3).
- **Beam inflates compute, not env budget** — `max_steps` budgets unchanged; forwards/decision ≈ `N_CANDIDATES × PLAN_HORIZON × K_ACT` (linear, per-node snapshots; S8). Extend the precheck or add a wall-clock estimate.
- **Rollouts stop before finalize** — `_handle_reward_well` (`:1622`) can hit `_log_and_finalize_episode` (`:1525`) and return `terminated=True` on the last trial (`:1650`); stepping a terminated env is undefined, so rollouts halt at the reward/trial boundary and read value at the pre-terminal leaf (S3/S14).
- **Attention decode assumes 3-token `(R,s,a)` layout** — preserved (new heads read existing positions); define planner attention as the committed-step forward only, or return `None`; apply the forward-contract collapse at `cell_28` (R3§0.7; S5).

**Open questions to resolve empirically:**
- Does the distributional value head's *spread* add signal beyond its mean? Measure at Stage 4 before trusting `distributional_value` scoring (R5§2). (Its mean is now a non-trivial test — S1.)
- `V_MIN` from empirical `min` (≈0.53, zero-clip) vs `p1` (≈0.83, finer Δz) — resolution vs clip tradeoff; default to `min`.
- Optimal `N_CANDIDATES`/`PLAN_HORIZON`/`K_ACT` for behavioral match vs compute — sweep in `cell_32` profiles.
---

## 7. Independent verification log (confirmed against live code/data, 2026-07-18)

These load-bearing claims were checked directly against the repo, not just asserted by the research pass. Confirm again at build time if the env changes.

**Planner feasibility (the linchpin).**
- `env.step()` ≈ **0.005 ms/step** on a `PI+VC f2 acquisition` env (deterministic transition; cheap).
- `copy.deepcopy(env)` **works** and yields a fully independent env — after divergent steps the original and clone hold different `(agent_pos, agent_dir)` — at ≈ **6.8 ms/copy** across the env's **86 instance attributes**. This is the oracle mechanism; it is ~650× slower than a manual snapshot, which is why §2.1 recommends a manual per-node snapshot with the deepcopy as a `tests/` oracle.
- `random.gauss` occurs **only** in `reset()` (`corner_maze_env.py:938`) and `_handle_exposure_well` (`:1612`) — **never** in the trial/ITI/pretrial step path. So acquisition + non-exposure probe rollouts are fully deterministic and snapshot-safe (validates §2/§6 scoping to non-exposure paradigms).

**State-prediction vocabulary.**
- `grid_cells_60d.npz`: `keys` shape `(484, 3)` (int16 `(x,y,d)`), `vectors` shape `(484, 60)` (float32). Vocab = 484 = 11·11·4, index formula `((x-1)·11+(y-1))·4+d` confirmed. **No all-zero rows.**
- Of the 484 cells, only **196 are ever visited** across the entire yoked dataset. The 484-way softmax in §4 is correct; the effective/reachable support is ~196. Optional efficiency note for the builder: the state head could mask logits to the reachable set, but an unmasked 484 softmax is simplest and correct — the model learns the illegal cells have ~0 mass.

**Env internals the planner depends on (all present at ~the cited lines):**
- `get_action_mask`:1589 · `_handle_reward_well`:1622 · `_handle_exposure_well`:1604 · `_log_and_finalize_episode`:1525 · `update_grid_configuration`:851 · `step`:1796 · `reset`:906.
- Mutated state to snapshot: `episode_scores`:517/1529 · `episode_data_rows`:538/1532 · `pseudo_session_score` (deque, maxlen):553/1634 · `_iti_config_idx` (lazily created):1680. `step` returns `episode_scores` in `info`:1853 — corroborating that the finalize path mutates these and the planner must either snapshot them or (primary defense, §2.1) not step into finalize.

**Build-cell contract (confirmed by reading the source, not just the research summary):**
- `RatWindowDataset.__getitem__` yields exactly `{state_idx, action, rtg, trial_timestep}` (`cell_10:26-31`) — the four keys the new labels extend.
- RTG = `COST.terminal_reward - COST.step_cost * a2r` (`cell_08:185`), analytic and design-locked — not a return.
- Training loss is a single `F.cross_entropy(logits.reshape(-1, NUM_ACTIONS), action)` (`cell_14:115`), mirrored in the val loop (`:164`) — the exact insertion point for the multi-head weighted loss.
- Action head reads state-token positions `x[:, 1::3]` (`cell_12:164`); the new state/reward/value heads read action-token positions `x[:, 2::3]`.
- Current `cell_04` defaults differ from the older design doc in two places worth noting: `BATCH_SIZE=512` (not 64) and `N_RUNS=2` (a smoke setting). The plan's §3 inventory reflects the **live** values.

> Provenance: this plan was produced by a 4-phase agent workflow (parallel deep-reads of the Janner 2021 PDF, nb10 build cells, nb10 eval cells, the `CornerMazeEnv` transition surface, and the SwitchTT value head → synthesis → adversarial critique → revision). The critique caught three latent correctness bugs now folded in: the value-target circularity (§4/S1), train-time RTG neutralization (§2/S2), and the double-rotation trap (§2.2/S4). The facts above were then re-verified by hand against the live repo.
