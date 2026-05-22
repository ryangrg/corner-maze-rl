"""Compute per-step ``pose_label`` by replaying yoked actions through the env.

``pose_label`` encodes ``layout_class_x_y_dir`` (e.g. ``trl_e_s_xx_8_2_0``) and
joins to ``data/dataframes/minigrid-views-allposes.parquet`` so view-mode
training can fetch the pre-rendered 21x21x3 RGB observation by lookup instead
of re-rendering through MiniGrid each step.

``CornerMazeEnv`` is the source of truth for layout-tuple selection and
ITI sub-config flipping, so we replay each session through it and read
``env.get_pose_label()`` per step. Pose at row ``i`` reflects the state
BEFORE action ``i`` is applied — same convention as
``diagnostics/check_divergence.py``.
"""
from __future__ import annotations

import pandas as pd

from corner_maze_rl.data.session_types import map_session_to_env_kwargs
from corner_maze_rl.env.corner_maze_env import CornerMazeEnv

from .diagnostics.replay_session import _inject_trial_configs

_GOAL_IDX_TO_NAME = {0: 'NE', 1: 'SE', 2: 'SW', 3: 'NW'}


def _build_replay_env(
    *,
    session_phase: str,
    session_number: str,
    session_type: str,
    cue_goal_orientation: str,
    training_group: str,
    trial_configs: list | None,
    n_rewards: int,
    init_pos: tuple[int, int],
    init_dir: int,
    max_steps: int,
) -> CornerMazeEnv:
    """Construct + reset a CornerMazeEnv ready to replay a yoked session.

    Mirrors ``diagnostics.check_divergence.check_session_from_dataset`` but
    works from build-time inputs (training_group + raw trial_configs) rather
    than the consolidated dataset. ``obs_mode='view'`` keeps env init cheap —
    we never query the observation, only the pose label.
    """
    is_exposure = session_phase == 'Exposure'

    if is_exposure:
        env_type = 'exposure_b' if (session_number == '2e' and n_rewards > 0) else 'exposure'
        goal_location = 'NE'
        if trial_configs:
            goal_location = _GOAL_IDX_TO_NAME.get(trial_configs[0][2], 'NE')
        env = CornerMazeEnv(
            render_mode='rgb_array',
            max_steps=max_steps,
            session_type=env_type,
            agent_cue_goal_orientation=cue_goal_orientation,
            start_goal_location=goal_location,
            obs_mode='view',
        )
        env.reset()
        if trial_configs:
            _inject_trial_configs(env, trial_configs)
        else:
            env.agent_pos = init_pos
            env.agent_dir = init_dir
            env.agent_pose = (*init_pos, init_dir)
            env.fwd_pos = env.front_pos
            env.fwd_cell = env.grid.get(*env.fwd_pos)
        return env

    goal_location = (
        _GOAL_IDX_TO_NAME.get(trial_configs[0][2], 'NE') if trial_configs else None
    )
    kw = map_session_to_env_kwargs(
        training_group=training_group,
        yoked_session_type=session_type,
        cue_goal_orientation=cue_goal_orientation,
        start_goal_location=goal_location,
        trial_configs=trial_configs,
        obs_mode='view',
    )
    if kw is None:
        raise ValueError(
            f'Unmapped paradigm for training_group={training_group!r}, '
            f'session_type={session_type!r}'
        )
    env = CornerMazeEnv(render_mode='rgb_array', max_steps=max_steps, **kw)
    env.reset()
    return env


def compute_pose_labels(
    actions_df: pd.DataFrame,
    *,
    session_phase: str,
    session_number: str,
    session_type: str,
    cue_goal_orientation: str,
    training_group: str,
    trial_configs: list | None,
) -> list[str]:
    """Replay ``actions_df`` through ``CornerMazeEnv`` and return per-row pose labels.

    The state at row ``i`` (``grid_x[i]``, ``grid_y[i]``, ``direction[i]``)
    must match the env's ``(agent_pos, agent_dir)`` before action ``i`` is
    applied — same contract validated by ``check_divergence.py``. Any
    mismatch raises ``RuntimeError`` since the yoking pipeline is in sync
    with the env as of 2026-05-10; surfacing it loudly prevents silent
    pose corruption.
    """
    n = len(actions_df)
    if n == 0:
        return []

    init_pos = (int(actions_df['grid_x'].iloc[0]), int(actions_df['grid_y'].iloc[0]))
    init_dir = int(actions_df['direction'].iloc[0])
    n_rewards = int(actions_df['rewarded'].sum())

    env = _build_replay_env(
        session_phase=session_phase,
        session_number=session_number,
        session_type=session_type,
        cue_goal_orientation=cue_goal_orientation,
        training_group=training_group,
        trial_configs=trial_configs,
        n_rewards=n_rewards,
        init_pos=init_pos,
        init_dir=init_dir,
        max_steps=max(n * 2, 10000),
    )

    actions = actions_df['action'].values
    grid_xs = actions_df['grid_x'].values
    grid_ys = actions_df['grid_y'].values
    directions = actions_df['direction'].values
    label = f'{training_group}/{session_type}/{session_number}'

    try:
        poses: list[str] = []
        for i in range(n):
            exp_pos = (int(grid_xs[i]), int(grid_ys[i]))
            exp_dir = int(directions[i])
            act_pos = (int(env.agent_pos[0]), int(env.agent_pos[1]))
            act_dir = int(env.agent_dir)
            if exp_pos != act_pos or exp_dir != act_dir:
                raise RuntimeError(
                    f'pose-label replay diverged on {label} '
                    f'at step {i}/{n}: expected pos={exp_pos} dir={exp_dir}, '
                    f'env pos={act_pos} dir={act_dir}'
                )
            poses.append(env.get_pose_label())
            _, _, terminated, truncated, _ = env.step(int(actions[i]))
            if (terminated or truncated) and i < n - 1:
                # With registered-visit gating in the yoking pipeline,
                # env trial_count tracks len(trial_configs) exactly, so
                # this branch should not fire. If it does, the yoked
                # stream has more goal-well entries than configured trials
                # — likely an alignment mistake worth surfacing loudly.
                raise RuntimeError(
                    f'env terminated early on {label} at step {i}/{n}: '
                    'gated build should keep env trial_count == n_trials'
                )
        return poses
    finally:
        env.close()
