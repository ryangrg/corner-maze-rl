"""Tests for corner_maze_rl.eval.rollout.

Smoke-test the DT rollout loop against the real env with a freshly-instantiated
(untrained) model. Verifies the loop runs, RTG decrements correctly, and
respects action masking.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from corner_maze_rl.encoders.grid_cells import GridCellEncoder
from corner_maze_rl.eval.rollout import rollout_dt
from corner_maze_rl.models.decision_transformer import DecisionTransformer, DTConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
PREBUILT_DICT = REPO_ROOT / "data" / "lookups" / "grid_cells_60d.npz"


@pytest.fixture
def env():
    from corner_maze_rl.env.corner_maze_env import CornerMazeEnv
    return CornerMazeEnv(
        session_type="PI+VC f2 single trial",
        agent_cue_goal_orientation="N/NE",
        start_goal_location="NE",
    )


@pytest.fixture
def encoder():
    if not PREBUILT_DICT.is_file():
        pytest.skip("prebuilt grid-cell dict not bundled")
    return GridCellEncoder()


@pytest.fixture
def small_dt():
    cfg = DTConfig(
        embed_dim=60,
        num_actions=5,
        context_size=8,
        num_heads=2,
        num_layers=1,
        pos_encoding="learned",
    )
    torch.manual_seed(0)
    return DecisionTransformer(cfg)


def test_rollout_smoke(small_dt, env, encoder):
    res = rollout_dt(
        small_dt, env, encoder,
        target_return=1.0,
        max_steps=20,
        seed=42,
    )
    assert len(res.actions) > 0
    assert len(res.actions) == len(res.rewards) == len(res.rtg_seq) == len(res.positions)
    assert all(0 <= a < 5 for a in res.actions)


def test_rollout_rtg_decrements_by_observed_reward(small_dt, env, encoder):
    target = 1.5
    res = rollout_dt(
        small_dt, env, encoder,
        target_return=target,
        max_steps=10,
        seed=42,
    )
    # rtg_seq[t] = target - sum(rewards[:t])
    expected = [target - sum(res.rewards[:t]) for t in range(len(res.actions))]
    assert np.allclose(res.rtg_seq, expected, atol=1e-5)


def test_rollout_action_mask_respected(small_dt, env, encoder):
    """At step 0 the agent can't enter a well (action 3) — that action
    should be masked out."""
    res = rollout_dt(
        small_dt, env, encoder,
        target_return=1.0,
        max_steps=1,
        seed=42,
        use_action_mask=True,
    )
    # The taken action must be valid under env's mask at step 0.
    # (We can't easily reproduce the exact mask the rollout saw without
    # plumbing, but action 3 is illegal at the default start so it must
    # NOT be the chosen action with masking enabled.)
    assert res.actions[0] != 3


def test_rollout_stochastic_changes_actions(small_dt, env, encoder):
    """Sampling mode (deterministic=False) with different seeds → different actions."""
    res_a = rollout_dt(small_dt, env, encoder, target_return=1.0,
                      max_steps=20, seed=1, deterministic=False, temperature=1.5)
    res_b = rollout_dt(small_dt, env, encoder, target_return=1.0,
                      max_steps=20, seed=999, deterministic=False, temperature=1.5)
    # Untrained model + stochastic → unlikely to produce identical sequences.
    assert res_a.actions != res_b.actions
