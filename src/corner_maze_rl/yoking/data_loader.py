"""DuckDB query layer for corner-maze-analysis processed parquet tables.

All queries read from the normalized parquet tables in the analysis project.
Timestamps are session-relative (milliseconds from session start).
"""
import os

import duckdb
import pandas as pd

ANALYSIS_DATA_DIR = os.environ.get(
    'CORNER_MAZE_ANALYSIS_DIR',
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'analysis'),
)

# Index conventions matching CornerMazeEnv
_ARM_NAME_TO_IDX = {'North': 0, 'East': 1, 'South': 2, 'West': 3}
_CUE_NAME_TO_IDX = {'North': 0, 'East': 1, 'South': 2, 'West': 3}
_GOAL_NAME_TO_IDX = {
    'Northeast': 0, 'Southeast': 1, 'Southwest': 2, 'Northwest': 3,
}
_NO_CUE = 4

# Zone → grid well position (matches map_to_minigrid_actions.ZONE_TO_WELL_POS)
_ZONE_TO_WELL_POS = {21: (11, 1), 17: (11, 11), 1: (1, 11), 5: (1, 1)}

# Session phase → trial tag
_PHASE_TO_TAG = {
    'Acquisition': 'trained',
    'Exposure': 'trained',
    'Novel Route': 'novel',
    'Reversal': 'reversal',
    'Rotation': 'probe_trained',
    'No Cue': 'probe_trained',
}


def _parquet(table: str) -> str:
    """Return full path to a parquet table in the analysis data directory."""
    return os.path.join(ANALYSIS_DATA_DIR, f'{table}.parquet')


def get_sessions(
    subject: str | None = None,
    session_number: str | None = None,
    session_phase: str | None = None,
) -> pd.DataFrame:
    """Query sessions joined with subjects.

    Args:
        subject: Filter by subject name (e.g., 'CM005').
        session_number: Filter by session number (e.g., '1e', '3').
        session_phase: Filter by session_phase (e.g., 'Exposure', 'Acquisition').

    Returns:
        DataFrame with columns: session_id, subject_id, subject_name,
        session_number, session_type, session_phase, cue_goal_orientation,
        training_group.
    """
    conditions = []
    if subject is not None:
        conditions.append(f"sub.name = '{subject}'")
    if session_number is not None:
        conditions.append(f"ses.session_number = '{session_number}'")
    if session_phase is not None:
        conditions.append(f"ses.session_experiment_phase = '{session_phase}'")

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ''

    query = f"""
        SELECT
            ses.session_id,
            ses.subject_id,
            sub.name AS subject_name,
            ses.session_number,
            ses.session_type,
            ses.session_experiment_phase AS session_phase,
            sub.cue_goal_orientation,
            sub.training_group
        FROM '{_parquet("sessions")}' ses
        JOIN '{_parquet("subjects")}' sub ON ses.subject_id = sub.subject_id
        {where}
        ORDER BY sub.name, ses.session_id
    """
    return duckdb.sql(query).fetchdf()


def get_coordinates(session_id: int) -> pd.DataFrame:
    """Get the coordinate stream for a session.

    Returns:
        DataFrame with columns (t_ms, x, y, zone) ordered by frame_idx.
    """
    query = f"""
        SELECT t_ms, x, y, zone
        FROM '{_parquet("coordinates")}'
        WHERE session_id = {session_id}
        ORDER BY frame_idx
    """
    return duckdb.sql(query).fetchdf()


def get_exposure_rewards(session_id: int) -> list[tuple[float, tuple[int, int]]]:
    """Get reward events for an exposure session.

    Returns:
        List of (t_entry_ms, well_grid_pos) tuples.
    """
    query = f"""
        SELECT t_entry_ms, well_zone
        FROM '{_parquet("exposure_rewards")}'
        WHERE session_id = {session_id}
        ORDER BY reward_idx
    """
    df = duckdb.sql(query).fetchdf()
    rewards = []
    for _, row in df.iterrows():
        well_pos = _ZONE_TO_WELL_POS.get(int(row['well_zone']))
        if well_pos is not None:
            rewards.append((float(row['t_entry_ms']), well_pos))
    return rewards


def get_trial_boundaries(session_id: int) -> list[tuple[float, float]]:
    """Get pretrial/trial phase boundary pairs for an acquisition session.

    Returns:
        List of (pretrial_t_start_ms, trial_t_start_ms) tuples.
    """
    query = f"""
        SELECT phase, trial_number, t_start_ms
        FROM '{_parquet("phases")}'
        WHERE session_id = {session_id}
          AND phase IN ('pretrial', 'trial')
        ORDER BY trial_number, phase
    """
    df = duckdb.sql(query).fetchdf()

    boundaries = []
    # Group by trial_number and pair pretrial with trial
    for trial_num in sorted(df['trial_number'].unique()):
        trial_phases = df[df['trial_number'] == trial_num]
        pretrial = trial_phases[trial_phases['phase'] == 'pretrial']
        trial = trial_phases[trial_phases['phase'] == 'trial']
        if len(pretrial) > 0 and len(trial) > 0:
            boundaries.append((
                float(pretrial.iloc[0]['t_start_ms']),
                float(trial.iloc[0]['t_start_ms']),
            ))
    return boundaries


def get_trial_configs(
    session_id: int,
    session_phase: str = 'Acquisition',
) -> list[tuple[int, int, int, str]]:
    """Get per-trial (start_arm, cue, goal, tag) configs for an acquisition session.

    Returns:
        List of (start_arm_idx, cue_idx, goal_idx, tag) tuples.
    """
    query = f"""
        SELECT start_arm, goal_location, cue_orientation, cue_on
        FROM '{_parquet("trials")}'
        WHERE session_id = {session_id}
        ORDER BY trial_number
    """
    df = duckdb.sql(query).fetchdf()
    tag = _PHASE_TO_TAG.get(session_phase, 'trained')

    configs = []
    for _, row in df.iterrows():
        arm = _ARM_NAME_TO_IDX.get(row['start_arm'], 0)
        if int(row['cue_on']) == 1:
            cue = _CUE_NAME_TO_IDX.get(row['cue_orientation'], _NO_CUE)
        else:
            cue = _NO_CUE
        goal = _GOAL_NAME_TO_IDX.get(row['goal_location'], 0)
        configs.append((arm, cue, goal, tag))
    return configs


def get_trial_rewards(session_id: int) -> list[tuple[float, tuple[int, int]]]:
    """Get rewarded well visits during trials for an acquisition session.

    Returns:
        List of (t_entry_ms, well_grid_pos) tuples.
    """
    query = f"""
        SELECT t_entry_ms, well_zone
        FROM '{_parquet("trial_well_visits")}'
        WHERE session_id = {session_id}
          AND is_reward = true
        ORDER BY t_entry_ms
    """
    df = duckdb.sql(query).fetchdf()
    rewards = []
    for _, row in df.iterrows():
        well_pos = _ZONE_TO_WELL_POS.get(int(row['well_zone']))
        if well_pos is not None:
            rewards.append((float(row['t_entry_ms']), well_pos))
    return rewards


def get_trial_well_visits(session_id: int) -> list[dict]:
    """Get every registered well visit (errors + rewards) for a session.

    Source of truth is upstream ``trial_well_visits.parquet``, which
    post-hoc reconstructs MazeControl's live reward-detection logic
    (error: dwell ≥10 ms + 2 s debounce; reward: dwell ≥250 ms within
    the trial phase window). Each row corresponds to one registered
    visit — sub-threshold drive-throughs are excluded.

    Returns a list of dicts with keys ``trial_number``, ``t_entry_ms``,
    ``t_exit_ms``, ``well_grid_pos``, ``is_reward``, ordered by
    ``t_entry_ms``. Used by the yoking pipeline to gate PICKUP emission
    so the action stream matches what MazeControl actually logged.
    """
    query = f"""
        SELECT t.trial_number, v.t_entry_ms, v.t_exit_ms, v.well_zone, v.is_reward
        FROM '{_parquet("trial_well_visits")}' v
        JOIN '{_parquet("trials")}' t ON v.trial_id = t.trial_id
        WHERE v.session_id = {session_id}
        ORDER BY v.t_entry_ms
    """
    df = duckdb.sql(query).fetchdf()
    visits = []
    for _, row in df.iterrows():
        well_pos = _ZONE_TO_WELL_POS.get(int(row['well_zone']))
        if well_pos is None:
            continue
        visits.append({
            'trial_number': int(row['trial_number']),
            't_entry_ms': float(row['t_entry_ms']),
            't_exit_ms': float(row['t_exit_ms']),
            'well_grid_pos': well_pos,
            'is_reward': bool(row['is_reward']),
        })
    return visits


def get_phase_coordinates(session_id: int) -> pd.DataFrame:
    """Get coordinates labeled by phase for an acquisition session.

    Joins coordinates with phases on time ranges so each coordinate sample
    knows whether it belongs to pretrial, trial, or ITI. Only returns rows
    that fall within a known phase — pre-session exploration is excluded.

    Returns:
        DataFrame with columns (t_ms, x, y, zone, phase, trial_number)
        ordered by t_ms.
    """
    query = f"""
        SELECT c.t_ms, c.x, c.y, c.zone, p.phase, p.trial_number
        FROM '{_parquet("coordinates")}' c
        JOIN '{_parquet("phases")}' p
          ON c.session_id = p.session_id
          AND c.t_ms >= p.t_start_ms
          AND c.t_ms < p.t_end_ms
        WHERE c.session_id = {session_id}
          AND p.phase IN ('pretrial', 'trial', 'iti')
        ORDER BY c.t_ms
    """
    return duckdb.sql(query).fetchdf()
