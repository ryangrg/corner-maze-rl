# Implementation Plan — Recurrent Successor-Feature Agent with an Implicit Context State

**Model key:** `recurrent_sf` · **Status:** design, not built · **Created:** 2026-07-22
**Home:** `src/corner_maze_rl/models/recurrent_sf.py` (+ train/eval touch-points below)
**Companion specs:** [`md/environment-architecture.md`](../md/environment-architecture.md), [`md/maze-behavior-spec.md`](../md/maze-behavior-spec.md), [`md/reward-structure-analysis.md`](../md/reward-structure-analysis.md), [`md/sr-yoked-negative-results.md`](../md/sr-yoked-negative-results.md)

> **How to use this document.** This is a build brief for an implementing agent. It is self-contained: §1–§2 give the scientific goal and how it relates to the repo's prior SR work; §3 fixes the design decisions so you can commit rather than hedge; §4–§6 are the architecture, losses, and training procedure; §7 is the file-by-file manifest; §8 is env integration; §9–§10 are the baselines and evaluation; §11 is the milestone/acceptance ladder; §12–§14 are risks, done-criteria, and references. Every non-obvious choice is defaulted with a rationale and a knob name — change the knob and re-run. Treat all committed content as public and student-facing per `CLAUDE.md`: no absolute paths, no local-machine assumptions, cite papers by key.

---

## 1. Problem framing

The corner maze already contains the task this model is for. A session is a sequence of **phases** — `EXPA/EXPB → PRETRIAL → TRIAL → ITI → PRETRIAL → …` — and reward is *regime-gated*: during `PRETRIAL` the agent is confined to the start arm behind closed barriers and there is no goal reward to be had; only after it holds the pretrial trigger for `PRETRIAL_MIN_STEPS` (=7) do the barriers open, the phase flips to `TRIAL`, and the goal well begins to pay out (`WELL_REWARD_SCR`). The pretrial→trial trigger *is* the "navigate without reward to a particular start position, after which the reward regime becomes active" event in the brief.

The context that decides whether a location is rewarding is **hidden from the agent's observation**. Pose is observed — position on the **11×11 grid of occupiable cells** (121 positions, of which 49 are open maze floor) plus one of 4 headings, giving **196 reachable poses = 49 floor cells × 4 directions**. (The env backs this with a 13×13 MiniGrid array once the perimeter wall ring is counted, so `env.width == 13` but world coordinates run 1–11; the 11×11 occupiable grid is the gridworld.) But pose alone does not reveal the context:

- The same pose recurs in `PRETRIAL`, `TRIAL`, and `ITI` under different barrier layouts and different reward implications. The phase itself is not in the observation.
- The **same well** is the rewarded goal under one session's cue/goal orientation and an unrewarded error well under another (see `maze-behavior-spec.md` §"Goal well structure" and the `reversal` probe family, where the goal shifts by +2 arms).
- Whether the reward regime is *active at all* depends on trigger history within the current trial, not on the current pose.

So **identical movement histories can occur under different hidden contexts**, and the agent cannot resolve them from pose or from the action stream alone. It must integrate the *observation, action, and reward* history — in particular reward outcomes and cue observations — to infer (a) the reward-machine state (pre-trigger vs. reward-active vs. inter-trial) and (b) the current goal identity. This is the core reason a memoryless or movement-only agent is expected to fail, and it is the hypothesis the evaluation in §10 is built to test.

The formal object being inferred is a **reward-machine state** in the sense of Toro Icarte et al. [Ica20b]: a latent, finite control state that gates which reward function is live. We do not give the agent the machine; we ask a recurrent state to *discover and track it* from experience, in the spirit of learning successor representations under partial observability [Ver19].

## 2. Position relative to the repo's prior SR work

`md/sr-yoked-negative-results.md` is the direct antecedent. A **vanilla online tabular SR-TD(0)** agent (the shipped `models/sr.py`) fails on this task, and the doc is careful to diagnose *why*:

1. **Online TD(0) on a 196–239-D tabular state cannot converge** — `M(a)` has ~5×236×236 ≈ 2.8×10⁵ noisy parameters and diverges within the episode budget. The closed-form control (`M = (I − γT)⁻¹`) *solves* the task, proving the representation is adequate and the **learning procedure** is the bottleneck.
2. **Feature-based `M` reconstruction is ill-conditioned** (`ΦᵀΦ` condition number ~10⁵²) for the overlapping 239-D features — naïve linear SR over learned features is unstable.
3. A **static reward-weight `w`** receives contradictory targets when goal or phase varies, and a manually added 3-D phase one-hot did not fix the online-TD instability.
4. Design hygiene: **exposure rewards** (all four wells rewarded) and **goal-rotating sessions** poison a single `w`; train on **fixed-goal, trial-phase** reward only.

The doc's own "Paths forward" list names exactly this model: *batch TD / experience replay*, *neural SR with learned features to reduce effective dimensionality*, and *multi-step returns*. This plan is that path, plus the missing ingredient the negative-results doc did not have: a **recurrent context state** that replaces both the manual phase one-hot and any bank of manually switched maps.

Design commitments that follow directly from the antecedent:

- **Batch sequence replay + TBPTT**, never per-step online TD — kills failure mode (1).
- **Function approximation with a one-hot pose basis for the SF target** — keeps the successor *basis* well-conditioned (the closed-form control used exactly this basis and worked), sidestepping failure mode (2). Learned compact `φ` is an ablation, not the default.
- **One shared, action-conditioned SF head plus a context-dependent reward model `w(z_t)`** — a single successor estimator, no separately switched maps. Because physical transitions are ~stationary while *reward* is what changes with context, we let context enter through the reward weights, not through parallel successor maps. Directly addresses failure mode (3).
- **Fixed-goal, trial-only reward discipline** for the reward-regression target — failure mode (4).

## 3. Decided open questions

All defaults; each is a config knob.

| # | Question | Decision | Rationale |
|---|---|---|---|
| 1 | Successor-feature basis `φ(z_t)` | **One-hot pose, 196-D** (`FEATURE=onehot_pose`) | The basis the closed-form control solved the task with; keeps the SF target well-conditioned. Learned compact `φ` (`FEATURE=learned_k`, k≈32–64) is an ablation for the "compact learned feature space" clause in the brief. |
| 2 | Pose embedding `e(pose_t)` into `z_t` | **60-D grid-cell encoder** (`GridCellEncoder`), `POSE_ENC=grid60` | Repo-standard 60-D pose code; compact input to the heads. `onehot196` and `pose2vec` are alternates. `e` and `φ` are deliberately different objects (input vs. SF basis). |
| 3 | GRU observation input `o_t` | **60-D `embedding` obs** (`obs_mode="embedding"`), `OBS=embedding` | Egocentric embedding is the "what the agent sees" signal that must carry cue presence; it does *not* reveal phase, which is the point. `view`+small CNN is the higher-capacity alternate (`OBS=view`). |
| 4 | GRU width `H` | **128** (`GRU_H=128`) | Enough to hold trigger/goal/step-since-trigger; small enough for CPU smoke runs. |
| 5 | Discount `γ` | **0.99** (`GAMMA=0.99`) | Matches the SR baseline and the closed-form control; long enough to propagate through ~21–33-step optimal paths without the 0.999 PPO horizon's variance. |
| 6 | TBPTT window `L` | **32 steps**, stateful carry across windows (`TBPTT_L=32`) | Longer than a pretrial (≥7) + a choice traversal, so the trigger and the reward that follows it fall in one window. Full-episode BPTT is available (`TBPTT_L=0`) but memory-heavy at ~1–6k steps/episode. |
| 7 | Reward head vs. SF weights | **Tie** `r̂_t = φ(z_t)·w(z_t)`, i.e. `g_r` *is* the linear `φ·w` head (`TIE_REWARD_HEAD=True`) | One object trains `w`, and `w` is exactly what `Q` needs. Untied `g_r` MLP is available for ablation. |
| 8 | Context supervision | **Privileged phase label at train time only** (`CTX_SUP=phase3`), 3-way `{pretrial, trial, iti}` from `env.session_phase` | Available from the simulator; used only in loss #4 and only during training. Also drives the oracle baseline (§9). `binary_active` (reward-active vs not) is a coarser alternate. |
| 9 | Action selection | **Boltzmann over masked `Q`** during learning, argmax when frozen (`ACT=boltzmann`, `EVAL_ACT=argmax`) | The negative-results doc showed pure greedy gets trapped in the zero-gradient pretrial; a temperature (or ε-floor) is needed to traverse unrewarded phases. |
| 10 | Reward scaling | **Env default sparse reward**, `VecNormalize`-style running return norm on the TD/`w` targets (`REWARD_NORM=return_std`) | `reward-structure-analysis.md` flags scale mismatch; normalize targets, do not reshape the task. SR-style overrides (costs→0, well→1.0) available as `REWARD_PROFILE=sr_clean`. |
| 11 | Training data regime | **On-policy env rollouts**, fixed goal within an episode, trial-phase reward only for `w`; exposure excluded (`W_FROM_PHASES={trial}`) | Fixes the negative-results design flaw; the GRU infers the *per-episode* goal from cue+reward, so goals may vary *across* episodes. |
| 12 | Yoked pretraining of the GRU | **Optional, off by default** (`YOKE_PRETRAIN=False`) | Yoked rat transitions do not match agent dynamics (negative-results finding 2). If used, pretrain only the GRU/`φ`-prediction and reset SF/`w` before free navigation. |

## 4. Architecture

### 4.1 Recurrent agent state

At each step the GRU folds the current observation, the previous action, and the current reward into a running summary:

$$h_t = \operatorname{GRU}\big(h_{t-1},\; [\,o_t,\; a_{t-1},\; r_t\,]\big), \qquad h_t \in \mathbb{R}^{H}.$$

- `o_t` = 60-D `embedding` observation (or CNN-encoded `view`).
- `a_{t-1}` = one-hot over the 5-action space `{L, R, F, enter-well, pause}`; zero vector at the first step of an episode.
- `r_t` = scalar reward received on entry to step `t` (return-normalized per knob 10); zero at episode start.
- `h_{-1} = 0`, and **`h` is reset to zero at every episode boundary** (§6.3). The reward and previous-action inputs are what let `h_t` encode "have I crossed the trigger this trial", "did the last well I entered pay out", and hence the hidden reward-machine state — history that pose alone cannot supply.

The pose enters *separately* (it is observed and we do not want to force the GRU to relearn spatial indexing):

$$z_t = [\,e(\text{pose}_t),\; h_t\,] \in \mathbb{R}^{60 + H}.$$

`z_t` is the agent state consumed by every downstream head: a spatial index (`e(pose)`) fused with an inferred context (`h_t`).

### 4.2 One shared, action-conditioned successor-feature head

A single network predicts discounted future features for every action — no per-context maps:

$$\psi(z_t, a) = \mathbb{E}\!\left[\sum_{k=0}^{\infty} \gamma^{k}\, \phi(z_{t+k}) \;\middle|\; z_t, a\right], \qquad \psi: \mathbb{R}^{60+H} \to \mathbb{R}^{A \times d_\phi}.$$

With the default one-hot pose basis, `d_φ = 196`: `ψ(z_t, a)` is the memory-conditioned discounted future **pose occupancy** under taking `a` then following the policy — a neural, context-conditioned successor representation. Implemented as an MLP `z_t → A·d_φ` reshaped to `(A, d_φ)`.

### 4.3 Context-dependent reward model and value

Because transitions are ~stationary but *reward* is what the context changes, context enters through a **reward-weight vector produced from the agent state**:

$$w(z_t) = \operatorname{MLP}_w(z_t) \in \mathbb{R}^{d_\phi}, \qquad \hat r_t = g_r(z_t) \equiv \phi(z_t)\cdot w(z_t), \qquad Q(z_t, a) = \psi(z_t, a)^{\top} w(z_t).$$

With one-hot `φ`, `\hat r_t = w(z_t)[\text{pose\_idx}_t]` — the predicted reward at the current pose *under the currently inferred context*. This is the whole trick: the same pose gets `w>0` when the GRU has inferred "reward-active, this is the goal" and `w≈0` when it has inferred "pretrial" or "this well is an error well in this session". No maps are switched; a single `ψ` is re-weighted by a context-dependent `w`.

```
 o_t ──┐
a_{t-1}┼─► GRU ─► h_t ─┐
 r_t ──┘               ├─► z_t=[e(pose_t), h_t] ─┬─► ψ(z_t,·) ∈ R^{A×dφ} ─┐
       pose_t ─► e(·) ─┘                          ├─► w(z_t) ∈ R^{dφ} ─────┼─► Q(z_t,a)=ψ·w
                                                   ├─► r̂_t = φ(z_t)·w(z_t)  │
                                                   └─► c_head(h_t) ─► phase   (train-only aux)
                                                   └─► T_head(z_t,a) ─► p(pose_{t+1})  (aux)
```

### 4.4 Parameter/shape summary (defaults)

| Symbol | Shape | Module |
|---|---|---|
| `o_t` | `(60,)` | env `embedding` obs |
| `[o,a_{prev},r]` | `(60+5+1,) = (66,)` | GRU input |
| `h_t` | `(128,)` | `nn.GRUCell(66, 128)` |
| `e(pose_t)` | `(60,)` | `GridCellEncoder` |
| `z_t` | `(188,)` | concat |
| `ψ(z_t,·)` | `(5, 196)` | MLP `188 → 5·196` |
| `w(z_t)` | `(196,)` | MLP `188 → 196` |
| `Q(z_t,·)` | `(5,)` | `ψ · w` |
| `c_head(h_t)` | `(3,)` | MLP `128 → 3` (train-only) |
| `T_head(z_t,a)` | `(196,)` | MLP `188+5 → 196` (aux) |

## 5. Losses

Computed over each replayed subsequence (§6). `sg[·]` = stop-gradient; primed nets `ψ⁻, w⁻` are target networks (Polyak `τ=0.005`).

**(1) Successor-feature TD.** With the brief's `k=0` convention (`ψ` includes `φ(z_t)`):

$$\mathcal{L}_\psi = \sum_t \big\lVert\, \psi(z_t, a_t) - \operatorname{sg}\!\big[\phi(z_t) + \gamma\, \psi^{-}(z_{t+1}, a^{*}_{t+1})\big] \big\rVert_2^2,\qquad a^{*}_{t+1} = \arg\max_a \psi^{-}(z_{t+1}, a)\cdot w^{-}(z_{t+1}).$$

Q-learning-style bootstrapped target (matches the off-policy target in `models/sr.py`). [Barreto16, Chu24]

**(2) Immediate reward prediction (context-dependent).**

$$\mathcal{L}_r = \sum_t \big(r_t - \phi(z_t)\cdot w(z_t)\big)^2 \quad\text{over steps with } \text{phase}_t \in \texttt{W\_FROM\_PHASES}.$$

Trains the context-dependent `w`; restricting to trial-phase steps enforces the fixed-goal/no-exposure discipline. [Gee22 in spirit: context-conditioned reward prediction.]

**(3) Next-pose prediction (auxiliary, predictive-state grounding).**

$$\mathcal{L}_{\text{trans}} = \sum_t \operatorname{CE}\!\big(T_\text{head}(z_t, a_t),\; \text{pose}_{t+1}\big).$$

Pushes `h_t` toward a sufficient statistic of the future (predictive-state / belief-state view). Optional (`LAMBDA_TRANS=0` disables). [Bra21]

**(4) Trigger/phase auxiliary (privileged, train-time only).**

$$\mathcal{L}_{\text{ctx}} = \sum_t \operatorname{CE}\!\big(c_\text{head}(h_t),\; \text{phase}_t\big),\qquad \text{phase}_t \in \{\text{pretrial}, \text{trial}, \text{iti}\}.$$

`phase_t` comes from `env.session_phase` mapped through `state_vectors._STATE_TYPE_TO_PHASE`. Used **only** during training and **only** by baselines/models that are allowed it; the deployed policy never reads it. This is the reward-machine-state supervision. Annealable (`LAMBDA_CTX` cosine-decayed to a floor) so the GRU eventually stands on its own. [Ica20b, Ver19]

**Total:** $\mathcal{L} = \lambda_\psi \mathcal{L}_\psi + \lambda_r \mathcal{L}_r + \lambda_\text{trans}\mathcal{L}_\text{trans} + \lambda_\text{ctx}\mathcal{L}_\text{ctx}$, defaults `λ_ψ=1.0, λ_r=1.0, λ_trans=0.5, λ_ctx=0.5→0.05`.

**(5) Training procedure — recurrent sequence replay with TBPTT.** Not a loss term but the fifth required ingredient; see §6.

## 6. Training procedure

### 6.1 Integration with `TrainableAgent` / `train.runner`

The model implements the `TrainableAgent` protocol (`models/base.py`) so it plugs into `run_session_sequence` via the same `make_env / train_fn / frozen_fn / save_fn` callables as PPO and SR. Because TBPTT needs whole subsequences, the agent keeps a recurrent state during rollout and a **sequence** replay buffer, rather than doing a per-step tabular update:

- `select_action(state, action_mask)`: advance `h_t` one step from the stored `(o_t, a_{t-1}, r_t)`, form `z_t`, compute masked `Q`, return a Boltzmann (train) / argmax (frozen) action and `info={"h": h_t, "q": Q, "phase_logits": c_head(h_t)}`. The agent holds `self._h` internally between calls.
- `add_experience(state, action, reward, done, **kwargs)`: append `(o_t, pose_t, a_t, r_{t+1}, phase_t, mask_t, done)` to the current-episode trajectory. On `done`, push the finished episode into the replay buffer and **reset `self._h = 0`**.
- `is_ready_to_update()`: `True` once the buffer holds ≥ `MIN_EPISODES_TO_UPDATE` complete episodes (or ≥ `MIN_STEPS`).
- `update(next_state, next_done)`: run one TBPTT optimization step (§6.2) and return the loss dict.

Provide a dedicated `train_fn` (`train/recurrent_sf_loop.py`) that runs the collect→update cadence explicitly, so TBPTT is legible rather than hidden behind the per-step generic loop. `frozen_fn` runs the same rollout with updates disabled and `EVAL_ACT=argmax`. `save_fn` calls `agent.save`.

### 6.2 TBPTT update

1. Sample `B` (=8) subsequences of length `L` (=32) from the replay buffer. Sampling **respects episode boundaries** and preserves temporal order within each subsequence (never shuffle within a window, never cross an episode boundary in one window).
2. For each window, initialize `h` from the stored recurrent state at the window's start (truncated BPTT with a stored carry) or zero if the window starts an episode; re-run the GRU forward over the `L` steps building `z_{t..t+L}`.
3. Compute `ψ, w, r̂, c_head, T_head` along the window; assemble losses (1)–(4). Bootstrap target at the window's final step uses the target nets on `z_{t+L}` (or `φ(z_t)` when `done`).
4. Backprop **truncated to the window**; `clip_grad_norm_(…, 1.0)`; Adam step (`lr=3e-4`). Polyak-update targets.
5. Detach and store the end-of-window `h` as the next window's carry so temporal context survives truncation.

### 6.3 Episode-boundary discipline (must-verify)

`h` is reset to zero at every episode boundary; `a_{prev}` and `r` inputs are zeroed at the first step; no gradient and no recurrent state crosses an episode boundary inside a replayed window. A unit test (`tests/test_recurrent_sf.py`) asserts this by feeding two concatenated short episodes and checking that `h` at the start of episode 2 is independent of episode 1's contents.

### 6.4 Hyperparameter defaults

`GAMMA=0.99`, `GRU_H=128`, `TBPTT_L=32`, `B=8`, `lr=3e-4`, Adam, `grad_clip=1.0`, target `τ=0.005`, buffer `capacity=200` episodes, `MIN_EPISODES_TO_UPDATE=8`, `updates_per_collect=4`, Boltzmann `temp=1.0→0.1` (annealed) with an `ε_floor=0.05`. Criterion reuse: `CRITERION_SCORE_THRESHOLD=0.72`, `CRITERION_CONSECUTIVE_REQUIRED=2` (from `constants.py`).

## 7. Module manifest

Create, all under `src/corner_maze_rl/`:

| Path | Responsibility |
|---|---|
| `models/recurrent_sf.py` | `RecurrentSFAgent(TrainableAgent)` — GRU cell, `ψ/w/c_head/T_head` heads, `select_action/add_experience/is_ready_to_update/update/save/load`, target nets, replay buffer handle. |
| `models/recurrent_sf_nets.py` | `nn.Module`s: `GRUEncoder`, `SFHead`, `RewardWeightHead`, `PhaseHead`, `TransitionHead`, plus a `RecurrentSFConfig` dataclass holding every knob in §3/§6.4. |
| `models/replay.py` | `SequenceReplayBuffer` — stores whole episodes; `sample_windows(B, L)` returns boundary-respecting, time-ordered windows with stored `h`-carry. (Generic; usable by future recurrent models.) |
| `train/recurrent_sf_loop.py` | `train_fn`/`frozen_fn` builders for the runner; collect→update cadence; return-normalization; kill-switch-friendly logging via `self.logger.record`. |
| `eval/probes.py` | Wraps the repo's linear-probe method (see §10.2) — hook `h_t`, fit `LinearProbe`, report probe error for phase / reward-active / goal / steps-since-trigger. |
| `eval/context_metrics.py` | Behavioral metrics of §10.1 computed from `episode_data` rows: unrewarded-approach success, post-trigger activation, same-goal-cross-context correctness, per-cycle trial success, delayed/partial-trigger robustness. |
| `scripts/train_recurrent_sf.py` | CLI entrypoint mirroring `scripts/train_dt.py`: build config, `make_env`, call `run_session_sequence`, write run artifacts. |
| `tests/test_recurrent_sf.py` | Shape/inference tests, the episode-boundary reset test (§6.3), a one-step TBPTT smoke test, and a "learns a single fixed-goal session" fast integration test. |
| `notebooks/13_recurrent_sf.ipynb` | Student-facing walkthrough with the Colab bootstrap cell (copy from `10_vanilla_dt.ipynb`), following the milestone ladder in §11. |

Register the model in `models/__init__.py`'s lazy `__getattr__` (keep the sb3-free import cheap — `RecurrentSFAgent` is pure PyTorch, so it can import eagerly, but keep torch import localized).

## 8. Environment integration

- **Construction:** `CornerMazeEnv(obs_mode="embedding")`; wrap so `select_action` receives `o_t`. Do **not** remap the 5-action space (`CLAUDE.md`).
- **Pose:** read `(x, y, dir)` from the unwrapped env (`agent_pos`, `agent_dir`) exactly as `eval/rollout.py::_agent_pose` does; feed to `GridCellEncoder.encode`. Coordinates run `x, y ∈ 1–11` on the 11×11 occupiable grid (49 of the 121 positions are floor); the MiniGrid array is 13×13 with a one-cell wall border.
- **Action mask:** `env.get_action_mask()` → `[L, R, F, PICKUP, PAUSE]`; mask `Q` to `-inf` before selection (as `rollout.py` does).
- **Phase label (privileged, train-only):** unwrapped `env.session_phase` → `state_vectors._STATE_TYPE_TO_PHASE` → `{trial=0, pretrial=1, iti=2}`. Collected into each transition for loss #4; never exposed to `select_action` for the main model.
- **Reward regime:** use env-default sparse reward; the model must *learn* that reward is trial-phase-gated. `W_FROM_PHASES={trial}` restricts the `w` regression target to trial steps so exposure's all-wells-rewarded signal never trains `w` (negative-results §hygiene). `REWARD_PROFILE=sr_clean` (costs→0, well→1.0) is available for a cleaner SF target during bring-up.
- **Data discipline:** train across sessions with varying cue/goal (so the GRU has a reason to infer context), but keep goal fixed *within* an episode. Start with a single acquisition `session_type`; add rotating-goal `session_type`s at milestone M2.

## 9. Baselines

All five share the training harness, seeds, and eval; each is a minimal ablation of the full model so differences are attributable.

| Baseline | Config delta | What it isolates / predicts |
|---|---|---|
| **B1 Memoryless SF** | `GRU_H=0` → `z_t = [e(pose_t)]`; `h_t` absent | No context inference. Should fail same-goal-cross-context and never reliably activate post-trigger; ~the neural analogue of `models/sr.py`. |
| **B2 Finite history stack** | replace GRU with concat of last `k=8` `(o,a,r)` + MLP (`MEMORY=stack_k`) | Whether *bounded* history suffices. Cross-reference the DT's context window `K`. Expected to degrade when trigger→reward spacing exceeds `k`. [Ver19 contrast] |
| **B3 Oracle phase-input** | concat true `phase_t` one-hot to `z_t` at train *and* test (`CTX_INPUT=oracle`) | Upper bound: context handed to the agent. Gap to full model = cost of having to infer. |
| **B4 Separately-switched SR maps** | bank of `K` heads `{ψ_k, w_k}` selected by a gate — hard-switch on a heuristic, or soft mixture indexed by a learned latent (`MODEL=switched_maps`, `K=3`) | The approach the brief says *not* to use for the main model but to beat. Multi-context SR / inferred successor maps. Expected brittleness at context transitions and reliance on a correct switch. [Mad19, Gee22] |
| **B5 GRU without reward** | GRU input `[o_t, a_{t-1}]` only (`GRU_INPUT=no_reward`) | The brief's central caution: when movement history is identical across contexts, only reward history disambiguates. Should match the full model where contexts differ visually but fail where they differ *only in reward outcome*. |

The full `recurrent_sf` model is the sixth arm of the comparison. Report all six with IQM + stratified-bootstrap CIs over seeds, per the repo eval canon ([Agarwal 2021] in `md/dt-repo-plan.md` §10 / README).

## 10. Evaluation

### 10.1 Behavioral (does the policy do the right thing)

Run through `run_session_sequence` (frozen-eval pass + adaptive pass) and compute from `episode_data`/`curves.parquet` via `eval/context_metrics.py`:

1. **Unrewarded approach to the trigger** — fraction of trials where the agent leaves the pretrial dead-end and holds the trigger, *earning no reward en route*. Isolines the "navigate without reward to a start position" behavior.
2. **Reward-regime activation** — conditional on reaching `TRIAL`, the trial success rate; and the latency from trigger to first goal-directed move (should drop once `h_t` encodes "active").
3. **Same location, opposite context** — take a well that is the goal in session-context A and an error well in session-context B; measure that the policy (and `w(z_t)`) treats it correctly in each. Use the `reversal` probe family (goal shifts +2) and cross-cue sessions.
4. **Repeated context transitions** — multi-trial episodes cycling `pretrial→trial→iti→pretrial…`; verify per-cycle trial success does not decay across cycles (the GRU re-enters and exits reward-active mode cleanly each trial).
5. **Generalization to new start/goal** — the `novel_route` and `rotate` probes (tags `novel`/`probe_trained` from `env/trial_sequence_gen.py`) present unseen `(start_arm → goal)` routes; report frozen-eval score vs. baselines. Full model should exceed B4 (switched maps) on novel transitions.
6. **Robustness to a delayed/partial trigger** — sweep `PRETRIAL_MIN_STEPS` upward (delayed activation) and mask the cue channel of `o_t` on a fraction of steps (partial observation); report graceful degradation curves. B5 (no-reward GRU) should fall off fastest when the only remaining disambiguator is reward history.

### 10.2 Representational (does `h_t` actually encode the context)

Use the repo's linear-probe method (`linear_probe.py`: `LinearProbe`, `train_probe`, `probe_error`, `extract_features`). Capture `h_t` with a forward hook on the GRU (detached, stop-grad — the probe never updates the agent), then fit convex linear probes on frozen `h_t`:

- **Phase probe** `h_t → {pretrial, trial, iti}` — low error ⇒ `h_t` linearly encodes the reward-machine state. Compare full vs. B1/B5.
- **Reward-active probe** `h_t → {active, inactive}` and **steps-since-trigger** (regression via a linear read-out) — tests the trigger-crossing memory directly.
- **Goal-identity probe** `h_t → which well is the goal this session` — tests within-session goal inference from cue+reward.
- **The decisive contrast:** on sessions engineered so that movement history is identical but reward history differs, the **full model's** phase/goal probe error should be low while **B5 (no-reward GRU)** stays near chance — the empirical statement of "movement history alone is not sufficient; reward history is required."
- **Uncertainty/belief:** on deliberately aliased contexts, measure the entropy/calibration of `c_head(h_t)` (and, optionally, add a small belief head). A well-behaved agent should express *uncertainty* under genuine ambiguity rather than committing — the probabilistic-SR / belief-state view. [Gee22, Bra21]

Report probe error tables per model arm; a `notebooks/13_recurrent_sf.ipynb` section renders the phase-probe-error bars and the aliased-context contrast.

### 10.3 Reporting

Follow the repo's scientific-reporting stance (`md/sr-yoked-negative-results.md`, README §eval): IQM + bootstrap CIs over ≥5 seeds, performance profiles, and honest negative results. If an arm fails, characterize *why* (probe evidence) rather than only that it failed.

## 11. Milestones & acceptance criteria

| M | Goal | Acceptance |
|---|---|---|
| **M0** | Skeleton + smoke | Model instantiates from `RecurrentSFConfig`; a forward pass yields the §4.4 shapes; one TBPTT step runs on CPU in seconds; `tests/test_recurrent_sf.py` shape + boundary-reset tests pass. |
| **M1** | Single fixed-goal, single context | On one acquisition `session_type`, reaches `CRITERION_SCORE_THRESHOLD` (0.72, 2 consecutive) and **beats B1 memoryless**. Confirms neural SF + batch TBPTT clears the online-TD failure of `sr.py`. |
| **M2** | Two hidden contexts | With ≥2 cue/goal contexts interleaved, phase-probe error < 0.15 and same-well-cross-context (metric 3) correct > 0.8. `h_t` demonstrably carries context. |
| **M3** | Full baseline matrix | All six arms trained on shared seeds; comparison table (IQM+CI) reproduces predicted ordering: full ≳ B3 oracle > B2 stack ≈ B4 switched > B1 memoryless, with B5 ≈ full except on reward-only-distinguishable contexts. |
| **M4** | Generalization probes | `novel_route`/`rotate`/`reversal` frozen-eval; full model > B4 on novel transitions; repeated-transition metric shows no across-cycle decay. |
| **M5** | Representational analysis | Full probe suite + aliased-context contrast + uncertainty analysis rendered in the notebook; the "reward history is necessary" contrast is significant (full ≪ B5 probe error on the engineered set). |

## 12. Risks & mitigations

- **Context aliasing / non-identifiability.** Some histories are genuinely ambiguous; a point estimate `h_t` may thrash. Mitigate with the predictive-state loss (#3), the belief/entropy read-out (§10.2), and by scoring *calibrated uncertainty* rather than only accuracy. [Gee22, Bra21]
- **TBPTT truncation vs. long unrewarded stretches.** If a pretrial + traversal exceeds `L`, the reward that confirms the trigger may fall outside the window. Mitigate with the stored `h`-carry across windows (§6.2) and, if needed, raise `TBPTT_L` or add the `n`-step return.
- **SF-TD stability.** Keep the well-conditioned one-hot pose basis (default), target nets, grad clipping, and return normalization; watch the `ψ` TD error the way the SR agent watches `sr_error`. Do *not* fall back to online per-step updates.
- **Zero-gradient pretrial trap.** Pure greedy gets stuck (negative-results control). Keep Boltzmann/ε-floor exploration during learning; the phase aux loss (#4) also gives the pretrial states a non-trivial learning signal early.
- **Reward-target poisoning.** Enforce `W_FROM_PHASES={trial}` and fixed within-episode goal; never train `w` on exposure steps.
- **Public-repo hygiene.** No absolute paths, no wiki filesystem references, no local-only assumptions in committed code or this doc (`CLAUDE.md`). Cite papers by key.

## 13. Definition of done

- `RecurrentSFAgent` conforms to `TrainableAgent`, trains via `run_session_sequence`, and saves/loads.
- Episode-boundary GRU reset is unit-tested; TBPTT respects boundaries and temporal order.
- M1–M3 acceptance met; the six-arm comparison table and the phase-probe table are reproducible from `scripts/train_recurrent_sf.py` + the notebook, seeded.
- The `h_t` phase/reward-active/goal probes show the predicted structure, and the full-vs-B5 reward-necessity contrast holds.
- No personal paths / no wiki paths in any committed artifact; references section complete.

## 14. References

Citation keys as used in the brief, mapped to sources. Strongest conceptual core: **Ver19 + Chu24 + Gee22**, with **Ica20b** supplying the formal trigger/reward-machine interpretation.

- **[Ver19]** Vértes, E. & Sahani, M. (2019). *A neurally plausible model learns successor representations in partially observable environments.* NeurIPS 2019. arXiv:1906.09480. — recurrent SR under partial observability; the reason a recurrent state can carry the hidden context.
- **[Chu24]** Chua, R., Ghosh, A., Kaplanis, C., Richards, B. A. & Precup, D. (2024). *Learning Successor Features the Simple Way.* NeurIPS 2024. arXiv:2410.22133. — stable SF learning evaluated on MiniGrid/MiniWorld (directly relevant environment family); basis for the SF-TD loss.
- **[Mad19]** Madarász, T. J. (2019). *Better transfer learning with inferred successor maps.* NeurIPS 2019. arXiv:1906.07663. — inferred/multiple successor maps; the B4 "separately-switched maps" baseline to beat.
- **[Gee22]** Geerts, J. P., Gershman, S. J., Burgess, N. & Stachenfeld, K. L. (2022). *A probabilistic successor representation for context-dependent prediction.* bioRxiv 2022.06.03.494671. — probabilistic multi-context SR; the belief/uncertainty view of context inference.
- **[Bra21]** Brantley, K., Mehri, S. & Gordon, G. J. (2021). *Successor Feature Sets: Generalizing Successor Representations Across Policies.* AAAI 2021. arXiv:2103.02650. — SF over sets/predictive-state; grounds the predictive-state auxiliary and belief representation.
- **[Dzh25]** Dzhivelikian, E. et al. (2025). *A Biologically Interpretable Cognitive Architecture for Online Structuring of Episodic Memories into Cognitive Maps.* arXiv:2510.03286. — episodic memory + successor features in partially observable grid-worlds.
- **[Emu21]** Emukpere, D., Alameda-Pineda, X. & Reinke, C. (2021). *Successor Feature Neural Episodic Control.* arXiv:2111.03110. — episodic-memory + SF control; alternative memory mechanism for comparison.
- **[Ica20b]** Toro Icarte, R., Klassen, T. Q., Valenzano, R. & McIlraith, S. A. (2020/2022). *Reward Machines: Exploiting Reward Function Structure in Reinforcement Learning.* arXiv:2010.03950; JAIR 73 (2022) 173–208. — the formal trigger/reward-regime as a reward-machine state the recurrent agent must track.

**In-repo cross-references:** `models/base.py` (`TrainableAgent`), `models/sr.py` (the online-tabular SR this supersedes), `train/runner.py` (`run_session_sequence`), `encoders/state_vectors.py` + `encoders/grid_cells.py` (`φ`, `e(pose)`), `eval/rollout.py` (pose/mask extraction), `linear_probe.py` (the probe method for §10.2), `md/sr-yoked-negative-results.md` (the motivating negative result), `md/maze-behavior-spec.md` + `md/environment-architecture.md` (phases, poses, triggers, wells).
