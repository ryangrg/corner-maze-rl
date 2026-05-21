"""Model-agnostic session-sequence runner.

Adapted from legacy ``src/rl/session_runner.py`` with these changes for
the new repo:

  * Imports ``set_global_seed`` from the new ``utils.run_io`` module
    (legacy used a flat ``from seed_utils import``).
  * Wires the kill-switch (``train.kill_switch``) into the per-session
    loop so training halts on flat / dead curves.
  * Writes ``run_config.json``, ``killed_at.json`` and a set of derived
    artifact files (``curves.parquet``, ``trial_metrics.parquet``,
    ``eval_summary.parquet``) using the per-run output schema described
    in plan §7.1.
  * Probes can run an N-episode frozen evaluation pass before the
    optional adaptive (updating) pass — set ``eval_episodes_per_probe``.

The runner is *agnostic* to the model: callers pass four callables —
``make_env``, ``train_fn``, ``frozen_fn``, ``save_fn`` — that encapsulate
all model-specific behaviour. See ``md/dt-repo-plan.md`` §6 for the
``TrainableAgent`` protocol the callables conform to.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from corner_maze_rl.eval.metrics import (
    _maybe_json,
    _perfect_trial_count,
    per_session_summary,
    per_trial_records,
)
from corner_maze_rl.train.kill_switch import (
    DEFAULT_CONFIG,
    Decision,
    KillSwitchConfig,
    decide,
    killed_at_payload,
)
from corner_maze_rl.utils.run_io import save_run_config, set_global_seed


# ---------------------------------------------------------------------------
# Data serialization (mirrors legacy session_runner.save_episode_dataframe)
# ---------------------------------------------------------------------------

_NESTED_COLUMNS: tuple[str, ...] = (
    "trajectory", "trial_scores", "turn_scores", "session_scores",
    "trial_tags", "trial_configs", "sequence_labels",
)


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"{type(obj).__name__} not JSON-serializable")


def save_episode_dataframe(df: pd.DataFrame, parquet_path: str | os.PathLike) -> None:
    """Persist an episode-rows DataFrame to parquet, JSON-encoding nested cols.

    pyarrow can't serialize mixed-type nested lists; the listed columns
    therefore get JSON-stringified. Round-trip with ``json.loads`` on read.
    """
    if df.empty:
        return
    df = df.copy()
    for col in _NESTED_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: json.dumps(x, default=_json_default))
    df.to_parquet(parquet_path, engine="pyarrow")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_data(env_raw, all_episode_data: list[dict], episode_offset: int) -> int:
    """Append rows from env_raw.episode_data_rows with global episode numbering."""
    rows = getattr(env_raw, "episode_data_rows", None) or []
    for row in rows:
        row = dict(row)
        row["episode"] = row.get("episode", 0) + episode_offset
        all_episode_data.append(row)
    return episode_offset + len(rows)


def _per_session_score(env_raw) -> int | float:
    """Pull the per-session ``perfect_trial_count``-equivalent from env state.

    Falls back to total return if the env doesn't expose a trial-score field.
    """
    rows = getattr(env_raw, "episode_data_rows", None) or []
    if rows and "perfect_trial_count" in rows[-1]:
        return rows[-1]["perfect_trial_count"]
    if rows and "trial_scores" in rows[-1]:
        scores = rows[-1]["trial_scores"]
        if isinstance(scores, list):
            return int(sum(s for s in scores if s))
    return getattr(env_raw, "session_reward", 0)


def _eval_episode_record(env_raw, session_type: str, eval_episode_index: int) -> dict:
    """One record per frozen-eval episode for ``eval_summary.parquet``."""
    rows = getattr(env_raw, "episode_data_rows", None) or []
    last = rows[-1] if rows else {}
    score = _perfect_trial_count(last.get("trial_scores"))
    ss = _maybe_json(last.get("session_scores")) or {}
    rec: dict[str, Any] = {
        "session_type":       session_type,
        "eval_episode_index": int(eval_episode_index),
        "score":              int(score),
        "n_steps":            int(last.get("total_steps", 0)),
        "total_reward":       float(last.get("total_reward", 0.0)),
    }
    if isinstance(ss, dict):
        for k, v in ss.items():
            try:
                rec[f"ss_{k}"] = float(v)
            except (TypeError, ValueError):
                rec[f"ss_{k}"] = None
    return rec


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class SessionResult:
    decision: Decision = Decision.CONTINUE
    n_sessions_run: int = 0
    scores: list[float] = field(default_factory=list)
    killed_at: dict | None = None
    df: pd.DataFrame | None = None


def run_session_sequence(
    session_types: str | Sequence[str],
    make_env: Callable[[str], tuple[Any, Any]],
    train_fn: Callable[[Any], None],
    frozen_fn: Callable[[Any], None],
    save_fn: Callable[[str], None],
    *,
    save_data_path: str | os.PathLike,
    model_save_dir: str | os.PathLike,
    seed: int | None = None,
    kill_switch_cfg: KillSwitchConfig | None = None,
    run_dir: str | os.PathLike | None = None,
    run_config_extra: dict | None = None,
    eval_episodes_per_probe: int = 1,
) -> SessionResult:
    """Run an acquisition→probe sequence with kill-switch monitoring.

    Parameters mirror the legacy signature plus several additions:
      * ``kill_switch_cfg`` — overrides for the early-termination thresholds.
        Default = ``DEFAULT_CONFIG`` from ``train.kill_switch``.
      * ``run_dir`` — if provided, ``run_config.json`` / ``killed_at.json`` /
        ``curves.parquet`` / ``trial_metrics.parquet`` / ``eval_summary.parquet``
        are written there per plan §7.1.
      * ``run_config_extra`` — extra keys merged into ``run_config.json``.
      * ``eval_episodes_per_probe`` — for probe session_types, run ``frozen_fn``
        this many times (each with a fresh env) before the adaptive pass.
        Default 1 = legacy behavior.

    The kill switch evaluates after *every* session (acquisition or probe).
    On KILL_* it stops the loop and writes ``killed_at.json``.

    Note on headline ``score`` in curves.parquet
    --------------------------------------------
    For acquisition / probe_updating phases the headline is ``score_max``
    (peak single-episode perfect_trial_count). Mean-over-last-K windowing
    does **not** fit criterion-style training — the last K episodes of an
    acquired run sit right at the criterion threshold by construction, so
    averaging them collapses to ~the threshold and tells you nothing
    about the policy. Peak is honest: "best score the agent ever achieved
    in this phase." Probe_frozen still uses ``score_frozen_mean`` across
    eval episodes (no criterion gating there).
    """
    if isinstance(session_types, str):
        session_types = [session_types]

    cfg = kill_switch_cfg or DEFAULT_CONFIG

    if seed is not None:
        set_global_seed(seed)

    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        save_run_config(
            run_dir,
            seed=seed if seed is not None else -1,
            extra={"session_types": list(session_types), **(run_config_extra or {})},
        )

    all_episode_data: list[dict] = []
    episode_offset = 0
    scores: list[float] = []                  # headline scalar per session_type (kill switch)
    curve_rows: list[dict] = []               # one row per (session_type, phase)
    eval_rows: list[dict] = []                # one row per frozen-eval episode
    decision = Decision.CONTINUE
    killed_payload: dict | None = None
    current_env_raw = None

    def _finalize_phase(session_index: int, session_type: str, phase: str,
                        policy_mode: str, wall_clock_s: float,
                        extra: dict | None = None) -> float:
        """Compute per-session summary from episode_data filtered by phase
        (via ``policy_mode``) and append to ``curve_rows``. Returns the
        headline scalar that will go into ``scores`` if the caller chooses
        to use it.
        """
        rows = [r for r in all_episode_data
                if r.get("session_type") == session_type
                and r.get("policy_mode") == policy_mode]
        summary = per_session_summary(rows, session_type)
        row: dict = {
            "session_index":      session_index,
            "session_type":       session_type,
            "phase":              phase,
            "policy_mode":        policy_mode,
            "wall_clock_seconds": float(wall_clock_s),
            **summary,
        }
        if extra:
            row.update(extra)
        # Headline ``score`` column: phase-dependent.
        #   probe_frozen   → mean across the N frozen-eval episodes
        #   acquisition / probe_updating → peak single-episode score
        if phase == "probe_frozen" and extra and "score_frozen_mean" in extra:
            row["score"] = float(extra["score_frozen_mean"])
        else:
            row["score"] = float(summary.get("score_max", 0))
        curve_rows.append(row)
        return row["score"]

    try:
        for session_index, session_type in enumerate(session_types):
            is_acquisition = "acquisition" in session_type.lower()

            if is_acquisition:
                t0 = time.perf_counter()
                env_wrapped, env_raw = make_env(session_type)
                current_env_raw = env_raw
                if hasattr(env_raw, "policy_mode"):
                    env_raw.policy_mode = "updating"
                train_fn(env_wrapped)
                episode_offset = _collect_data(env_raw, all_episode_data, episode_offset)
                wall = time.perf_counter() - t0

                sc = _finalize_phase(session_index, session_type,
                                     phase="acquisition", policy_mode="updating",
                                     wall_clock_s=wall)
                scores.append(sc)
                save_fn(str(Path(model_save_dir) / "model_post_acquisition"))

                # If acquisition didn't reach criterion (e.g., hit the
                # session/timestep cap, or flat-learning kill fired),
                # skip the rest of the sequence — probes only make
                # sense on an acquired policy. Reason is recorded by the
                # callback that fired (see ``env._kill_reason``).
                acquired = bool(getattr(env_raw, "training_criterion_met", True))
                kill_reason = getattr(env_raw, "_kill_reason", None) or "did_not_acquire"
                current_env_raw = None
                if not acquired:
                    decision = Decision.KILL_HARD_CAP
                    killed_payload = {
                        "decision":       decision.value,
                        "reason":         kill_reason,
                        "session_index":  session_index,
                        "session_type":   session_type,
                        "scores_so_far":  list(scores),
                    }
                    if run_dir is not None:
                        with open(run_dir / "killed_at.json", "w") as f:
                            json.dump(killed_payload, f, indent=2)
                    break

            else:
                # ---- probe: N-episode frozen eval ----
                t0 = time.perf_counter()
                per_eval_scores: list[int] = []
                for eval_i in range(max(1, eval_episodes_per_probe)):
                    env_wrapped, env_raw = make_env(session_type)
                    current_env_raw = env_raw
                    if hasattr(env_raw, "policy_mode"):
                        env_raw.policy_mode = "frozen"
                    frozen_fn(env_wrapped)
                    eval_rec = _eval_episode_record(env_raw, session_type, eval_i)
                    eval_rows.append(eval_rec)
                    per_eval_scores.append(int(eval_rec["score"]))
                    episode_offset = _collect_data(env_raw, all_episode_data, episode_offset)
                    current_env_raw = None
                wall_frozen = time.perf_counter() - t0

                frozen_mean = float(np.mean(per_eval_scores)) if per_eval_scores else 0.0
                frozen_std  = float(np.std(per_eval_scores, ddof=0)) if per_eval_scores else 0.0
                _finalize_phase(
                    session_index, session_type,
                    phase="probe_frozen", policy_mode="frozen",
                    wall_clock_s=wall_frozen,
                    extra={
                        "n_eval_episodes":   len(per_eval_scores),
                        "score_frozen_mean": frozen_mean,
                        "score_frozen_std":  frozen_std,
                        "score_frozen_min":  int(min(per_eval_scores)) if per_eval_scores else 0,
                        "score_frozen_max":  int(max(per_eval_scores)) if per_eval_scores else 0,
                    },
                )

                # ---- probe: adaptive (updating) pass ----
                t0 = time.perf_counter()
                env_wrapped, env_raw = make_env(session_type)
                current_env_raw = env_raw
                if hasattr(env_raw, "policy_mode"):
                    env_raw.policy_mode = "updating"
                train_fn(env_wrapped)
                episode_offset = _collect_data(env_raw, all_episode_data, episode_offset)
                wall_updating = time.perf_counter() - t0

                _finalize_phase(session_index, session_type,
                                phase="probe_updating", policy_mode="updating",
                                wall_clock_s=wall_updating)

                # Kill-switch / SessionResult headline scalar = frozen-eval mean
                # (the scientifically interesting generalization measure).
                scores.append(frozen_mean)
                current_env_raw = None

            # Kill-switch evaluation
            ks = decide(scores, cfg)
            decision = ks.decision
            if decision.is_terminal:
                if decision.is_kill:
                    killed_payload = killed_at_payload(ks, scores)
                    if run_dir is not None:
                        with open(run_dir / "killed_at.json", "w") as f:
                            json.dump(killed_payload, f, indent=2)
                break

        # Post-probes checkpoint (if loop ran any probes)
        if any("acquisition" not in st.lower() for st in session_types):
            save_fn(str(Path(model_save_dir) / "model_post_probes"))

    except KeyboardInterrupt:
        if current_env_raw is not None:
            episode_offset = _collect_data(current_env_raw, all_episode_data, episode_offset)
        df = pd.DataFrame(all_episode_data)
        save_episode_dataframe(df, save_data_path)
        if run_dir is not None:
            _write_derived_artifacts(run_dir, all_episode_data, curve_rows, eval_rows)
        raise

    df = pd.DataFrame(all_episode_data)
    save_episode_dataframe(df, save_data_path)

    if run_dir is not None:
        _write_derived_artifacts(run_dir, all_episode_data, curve_rows, eval_rows)

    return SessionResult(
        decision=decision,
        n_sessions_run=len(scores),
        scores=scores,
        killed_at=killed_payload,
        df=df,
    )


# ---------------------------------------------------------------------------
# Derived artifact writers
# ---------------------------------------------------------------------------

def _write_derived_artifacts(
    run_dir: Path,
    all_episode_data: list[dict],
    curve_rows: list[dict],
    eval_rows: list[dict],
) -> None:
    """Write curves / trial_metrics / eval_summary parquet files."""
    if curve_rows:
        pd.DataFrame(curve_rows).to_parquet(run_dir / "curves.parquet", engine="pyarrow")
    if eval_rows:
        pd.DataFrame(eval_rows).to_parquet(run_dir / "eval_summary.parquet", engine="pyarrow")
    if all_episode_data:
        trial_rows: list[dict] = []
        for ep in all_episode_data:
            trial_rows.extend(per_trial_records(ep))
        if trial_rows:
            pd.DataFrame(trial_rows).to_parquet(
                run_dir / "trial_metrics.parquet", engine="pyarrow"
            )
