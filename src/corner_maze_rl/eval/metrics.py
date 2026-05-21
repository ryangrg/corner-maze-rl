"""Derived metrics from `episode_data.parquet` rows.

The runner writes the full per-episode dataframe to `episode_data.parquet`
with nested columns (trial_scores, trial_tags, trial_configs, session_scores,
trajectory, sequence_labels) JSON-encoded for parquet compatibility. This
module turns those nested rows into flat, plot-ready records:

  * ``per_trial_records(episode_row)`` — one record per trial (index, tag,
    config, score). Lossless restructuring of trial_scores × trial_tags
    × trial_configs.

  * ``per_episode_trajectory_stats(episode_row)`` — per-episode trajectory
    summary: n_steps, n_unique_cells, revisit_rate. Computed from the
    `trajectory` column.

  * ``per_session_summary(episode_rows, session_type)`` — collapses multiple
    episodes of one session_type into one row: score mean/max/min/last/std,
    tag-aware aggregates averaged across episodes, episode count. Used to
    build the enriched `curves.parquet`.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Decoding helpers
# ---------------------------------------------------------------------------

def _maybe_json(v: Any) -> Any:
    """Decode JSON-encoded values from parquet round-tripping."""
    if isinstance(v, (str, bytes)):
        try:
            return json.loads(v)
        except (json.JSONDecodeError, TypeError):
            return v
    return v


def _perfect_trial_count(trial_scores: Any) -> int:
    ts = _maybe_json(trial_scores) or []
    return int(sum(1 for s in ts if s))


# ---------------------------------------------------------------------------
# Per-trial flattening
# ---------------------------------------------------------------------------

def per_trial_records(episode_row: dict) -> list[dict]:
    """Return one record per trial in this episode.

    Each record: ``{session_type, policy_mode, episode, trial_index, tag,
    start_arm, cue, goal, score}``. Missing fields produce ``None``.
    """
    trial_scores = _maybe_json(episode_row.get("trial_scores")) or []
    trial_tags   = _maybe_json(episode_row.get("trial_tags"))   or []
    trial_configs = _maybe_json(episode_row.get("trial_configs")) or []

    rows: list[dict] = []
    n = max(len(trial_scores), len(trial_tags), len(trial_configs))
    for i in range(n):
        cfg = trial_configs[i] if i < len(trial_configs) else [None, None, None, None]
        rows.append({
            "session_type": episode_row.get("session_type"),
            "policy_mode":  episode_row.get("policy_mode"),
            "episode":      int(episode_row.get("episode", 0)),
            "trial_index":  i,
            "tag":          trial_tags[i] if i < len(trial_tags) else None,
            "start_arm":    cfg[0] if len(cfg) > 0 else None,
            "cue":          cfg[1] if len(cfg) > 1 else None,
            "goal":         cfg[2] if len(cfg) > 2 else None,
            "score":        int(trial_scores[i]) if i < len(trial_scores) else None,
        })
    return rows


# ---------------------------------------------------------------------------
# Per-episode trajectory stats
# ---------------------------------------------------------------------------

def per_episode_trajectory_stats(episode_row: dict) -> dict:
    """Cheap trajectory features computable without env access.

    Returns a dict with: n_steps, n_unique_cells, revisit_rate
    (= 1 - unique/total). All zeros if the trajectory is empty.
    """
    traj = _maybe_json(episode_row.get("trajectory")) or []
    n = len(traj)
    if n == 0:
        return {"n_steps": 0, "n_unique_cells": 0, "revisit_rate": 0.0}
    positions = {(t[0], t[1]) for t in traj if len(t) >= 2}
    return {
        "n_steps":        n,
        "n_unique_cells": len(positions),
        "revisit_rate":   float(1.0 - len(positions) / n),
    }


# ---------------------------------------------------------------------------
# Per-session aggregation (used by runner to build curves.parquet)
# ---------------------------------------------------------------------------

def per_session_summary(
    episode_rows: list[dict],
    session_type: str,
) -> dict:
    """Aggregate all episodes matching ``session_type`` into one summary row.

    Returns dict with:
      - n_episodes, score_last, score_max, score_min, score_mean, score_std
      - tag-aware fields ``<tag>_mean``, ``<tag>_possible_mean`` for every
        tag observed in any episode's session_scores dict.

    Returns zero-filled defaults if no episodes match.

    Note: there is no trailing-window mean here. Mean-over-last-K is
    misleading under criterion-style training (the last K episodes of
    an acquired run sit at the criterion threshold by construction).
    The headline ``score`` column in curves.parquet is derived from
    ``score_max`` for acquisition phases — see runner.py docstring.
    """
    rows = [r for r in episode_rows if r.get("session_type") == session_type]
    if not rows:
        return {
            "session_type": session_type,
            "n_episodes":   0,
            "score_last":   0,
            "score_max":    0,
            "score_min":    0,
            "score_mean":   0.0,
            "score_std":    0.0,
        }

    perfect = np.array([_perfect_trial_count(r.get("trial_scores")) for r in rows], dtype=float)
    summary: dict = {
        "session_type": session_type,
        "n_episodes":   len(rows),
        "score_last":   int(perfect[-1]),
        "score_max":    int(perfect.max()),
        "score_min":    int(perfect.min()),
        "score_mean":   float(perfect.mean()),
        "score_std":    float(perfect.std(ddof=0)),
    }

    # Tag-aware aggregation: mean across episodes for each tag observed.
    tag_correct: dict[str, list[float]] = {}
    tag_possible: dict[str, list[float]] = {}
    for r in rows:
        ss = _maybe_json(r.get("session_scores")) or {}
        if not isinstance(ss, dict):
            continue
        for k, v in ss.items():
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if k.endswith("_possible"):
                tag_possible.setdefault(k[:-len("_possible")], []).append(v)
            else:
                tag_correct.setdefault(k, []).append(v)

    for tag, vals in tag_correct.items():
        summary[f"{tag}_mean"] = float(np.mean(vals)) if vals else 0.0
    for tag, vals in tag_possible.items():
        summary[f"{tag}_possible_mean"] = float(np.mean(vals)) if vals else 0.0

    return summary
