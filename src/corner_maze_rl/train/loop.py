"""Per-episode training / frozen-eval loops for the ``TrainableAgent`` protocol.

Ported from legacy ``src/rl/custom_rl.py::run_free_training``. Used by the
PPO and SR experiment notebooks (03, 05) as the inner-loop helper that
``train.runner.run_session_sequence`` invokes per session.

The loops are model-agnostic: they call ``agent.select_action`` /
``add_experience`` / ``is_ready_to_update`` / ``update`` per the
``TrainableAgent`` protocol in ``models/base.py``. The ``state_vector_fn``
parameter encapsulates encoder choice (one-hot pose, grid cells, ...).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from corner_maze_rl.models.base import TrainableAgent
from corner_maze_rl.utils.run_io import set_global_seed


StateVectorFn = Callable[[tuple, list, int, Any], np.ndarray]


# Legacy parity: PAUSE (action 4) is disabled during training. PPOAgent.select_action
# masks invalid actions before sampling, so this just zeroes the PAUSE column.
_ENABLE_PAUSE: bool = False


def run_free_training_loop(
    env,
    agent: TrainableAgent,
    num_episodes: int,
    *,
    state_vector_fn: StateVectorFn,
    n_wm_units: int = 10,
    stop_on_criterion: bool = True,
    seed: int | None = None,
) -> dict:
    """Train ``agent`` on ``env`` by free exploration with action masking.

    Each episode: reset → loop until done, calling agent.select_action /
    add_experience / update on the ``TrainableAgent`` protocol. The state
    vector at each step comes from ``state_vector_fn(pose, reward_timers,
    n_wm_units, env)``.

    Returns a dict of per-episode metrics (rewards, lengths, all update losses).
    """
    if seed is not None:
        set_global_seed(seed)

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    all_losses: list[dict] = []

    for ep in range(num_episodes):
        env.reset()
        ep_reward = 0.0
        ep_steps = 0
        done = False

        while not done:
            state = state_vector_fn(env.agent_pose, env.reward_timers, n_wm_units, env)
            mask = env.get_action_mask()
            mask[4] = _ENABLE_PAUSE

            action, action_info = agent.select_action(state, mask)
            _, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            ep_reward += float(reward)
            ep_steps += 1

            agent.add_experience(state, action, reward, done, **action_info)

            if agent.is_ready_to_update():
                next_state = state_vector_fn(
                    env.agent_pose, env.reward_timers, n_wm_units, env
                )
                loss_info = agent.update(next_state, done)
                all_losses.append(loss_info)

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_steps)

        if (
            stop_on_criterion
            and getattr(env, "training_criterion_met", False)
        ):
            break

    return {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "losses": all_losses,
        "criterion_met": bool(getattr(env, "training_criterion_met", False)),
    }


def run_frozen_eval(
    env,
    agent: TrainableAgent,
    num_episodes: int,
    *,
    state_vector_fn: StateVectorFn,
    n_wm_units: int = 10,
) -> dict:
    """Roll out ``agent`` on ``env`` without learning.

    No ``add_experience`` / ``update`` calls — the policy is frozen for the
    duration of the call. Used for the frozen-evaluation pass on probe
    sessions (plan §10.3) before the updating pass that lets the agent
    actually learn from the probe.
    """
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []

    for _ in range(num_episodes):
        env.reset()
        ep_reward = 0.0
        ep_steps = 0
        done = False

        while not done:
            state = state_vector_fn(env.agent_pose, env.reward_timers, n_wm_units, env)
            mask = env.get_action_mask()
            mask[4] = _ENABLE_PAUSE

            action, _ = agent.select_action(state, mask)
            _, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            ep_reward += float(reward)
            ep_steps += 1

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_steps)

    return {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
    }
