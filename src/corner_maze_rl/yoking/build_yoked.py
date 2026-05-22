"""Build yoked action sequences from corner-maze-analysis parquet data.

Converts rodent behavioral tracking data into MiniGrid-compatible action
sequences and trial configs for yoked RL training.

Usage:
    python yoking/build_yoked.py --subject CM005 --session 1e
    python yoking/build_yoked.py --subject CM005 --all
    python yoking/build_yoked.py --all
    python yoking/build_yoked.py --all --phase Exposure
    python yoking/build_yoked.py --all --phase Acquisition
"""
import argparse
import json
import os
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

from .compute_pose_labels import compute_pose_labels
from .data_loader import (
    get_coordinates,
    get_exposure_rewards,
    get_phase_coordinates,
    get_sessions,
    get_trial_boundaries,
    get_trial_configs,
    get_trial_rewards,
    get_trial_well_visits,
)
from .map_to_minigrid import map_session_to_grid
from .map_to_minigrid_actions import build_action_sequence

# Repo root = src/corner_maze_rl/yoking/build_yoked.py → up 4
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'data', 'yoked'
)


def process_session(
    session_id: int,
    subject_name: str,
    session_number: str,
    session_type: str,
    session_phase: str,
    cue_goal_orientation: str,
    training_group: str,
    output_dir: str,
    seed: int = 42,
    build_pause: bool = True,
    pause_threshold_ms: int = 1500,
    use_real_pretrial: bool = False,
    consolidate_pauses: bool = True,
) -> dict:
    """Process a single session through the yoking pipeline.

    Returns a summary dict with session info and action counts.
    """
    t0 = time.time()

    # 1. Get coordinates and map to grid
    is_exposure = session_phase == 'Exposure'
    timed_phase_end = []
    # Exposure has no trial_well_visits; acquisition branch populates this.
    registered_well_visits = None

    if is_exposure:
        coord_df = get_coordinates(session_id)
        if len(coord_df) == 0:
            return {'session_id': session_id, 'status': 'skipped', 'reason': 'no coordinates'}

        # Trim leading bad data for exposure sessions where the rat was
        # misplaced or escaped the center. For 2e sessions, the rat must
        # be at center when barriers start dropping. Find the LAST
        # sustained center visit before the barrier exploration begins
        # (i.e., before the rat first reaches a non-center, non-adjacent
        # zone that it stays in). For 1e sessions, find the first
        # sustained center visit if not starting there.
        exposure_trimmed = False
        starts_at_center = True  # default; overridden below for 2e
        if len(coord_df) > 100 and session_number == '2e':
            zones_arr = coord_df['zone'].values
            xs_arr = coord_df['x'].values.astype(float)
            ys_arr = coord_df['y'].values.astype(float)
            # For 2e: the rat must start at center. If it leaves center
            # within the first few seconds (before barriers could drop),
            # it was misplaced or escaped. Find the first sustained
            # center visit after the escape, trim to there, then
            # interpolate out experimenter phantom jumps.
            starts_at_center = (int(zones_arr[0]) == 11 or
                                (abs(xs_arr[0] - 120) < 20 and abs(ys_arr[0] - 120) < 20))
            leaves_early = False
            if starts_at_center:
                for check_i in range(min(300, len(zones_arr))):
                    if int(zones_arr[check_i]) != 11:
                        leaves_early = True
                        break
            if not starts_at_center:
                # Rat misplaced — find first sustained center, barriers
                # already dropped by the time the rat is re-placed.
                for trim_i in range(len(zones_arr) - 30):
                    if all(int(zones_arr[trim_i + k]) == 11 for k in range(30)):
                        coord_df = coord_df.iloc[trim_i:].reset_index(drop=True)
                        exposure_trimmed = True
                        break
            elif leaves_early:
                # Rat started at center but left early (escaped or
                # experimenter interference). Find first re-center.
                for trim_i in range(check_i, len(zones_arr) - 30):
                    if all(int(zones_arr[trim_i + k]) == 11 for k in range(30)):
                        coord_df = coord_df.iloc[trim_i:].reset_index(drop=True)
                        exposure_trimmed = True
                        break

            # Interpolate out experimenter phantom positions: during the
            # period before the experimenter leaves the room, null any
            # positions outside the center zone and interpolate them.
            # The rat should be confined to center until barriers drop.
            if exposure_trimmed:
                import numpy as np
                xs_c = coord_df['x'].values.astype(float).copy()
                ys_c = coord_df['y'].values.astype(float).copy()
                # During the experimenter interference window (from trim
                # start until experimenter leaves), null positions outside
                # the center cross-corridor region and interpolate.
                # Center + cross-corridors:
                in_cross = (
                    ((xs_c >= 96) & (xs_c <= 143)) |   # center column
                    ((ys_c >= 97) & (ys_c <= 142))     # center row
                )
                # Find experimenter exit: first 200-row block where all
                # positions are in the cross (rat exploring without
                # phantom interference).
                # Find when experimenter leaves: skip past initial center
                # dwell, then find the first 500-row block with all
                # positions in the cross region.
                first_leave = 0
                for ci in range(len(in_cross)):
                    if not in_cross[ci]:
                        first_leave = ci
                        break
                # Only interpolate if there are non-cross positions in
                # the first 500 rows (experimenter interference). If
                # first_leave is beyond 500, the data is clean.
                interp_end = 0
                if first_leave > 0 and first_leave < 2000:
                    for ci in range(first_leave, len(xs_c) - 500):
                        if np.all(in_cross[ci:ci + 500]):
                            interp_end = ci
                            break
                    if interp_end == 0:
                        interp_end = min(10000, len(xs_c))  # fallback
                # Only interpolate within the interference window
                valid = np.ones(len(xs_c), dtype=bool)
                for ci in range(interp_end):
                    if not in_cross[ci]:
                        valid[ci] = False
                n_invalid = (~valid).sum()
                if n_invalid > 0 and valid.any():
                    idx = np.arange(len(xs_c))
                    xs_c[~valid] = np.interp(idx[~valid], idx[valid], xs_c[valid])
                    ys_c[~valid] = np.interp(idx[~valid], idx[valid], ys_c[valid])
                    coord_df = coord_df.copy()
                    coord_df['x'] = xs_c
                    coord_df['y'] = ys_c
                    # Reclassify zones during the interference window.
                    # Both interpolated AND non-interpolated points in the
                    # cross region may have wrong zones (experimenter
                    # reads in zone 10 that should be zone 11 center).
                    # Force all points in the interference window to
                    # zone 11 (center) unless they're clearly in a
                    # corridor that the rat legitimately reached.
                    zones_c = coord_df['zone'].values.copy()
                    for ci in range(interp_end):
                        if int(zones_c[ci]) != 11:
                            zones_c[ci] = 11
                    coord_df['zone'] = zones_c

        elif len(coord_df) > 100 and session_number != '2e':
            zones_arr = coord_df['zone'].values
            xs_arr = coord_df['x'].values.astype(float)
            ys_arr = coord_df['y'].values.astype(float)
            starts_at_center = (int(zones_arr[0]) == 11 or
                                (abs(xs_arr[0] - 120) < 20 and abs(ys_arr[0] - 120) < 20))
            if not starts_at_center:
                for trim_i in range(len(zones_arr) - 30):
                    if all(int(zones_arr[trim_i + k]) == 11 for k in range(30)):
                        coord_df = coord_df.iloc[trim_i:].reset_index(drop=True)
                        exposure_trimmed = True
                        break

        grid_df = map_session_to_grid(coord_df)
        reward_events = get_exposure_rewards(session_id)
        trial_configs = []

        # For exposure_b (2e): compute when timed barrier phases end
        # by summing first 2 trial durations (acclimation + CE timed)
        timed_phase_end = []
        if session_number == '2e':
            import duckdb
            from .data_loader import _parquet
            first_t = float(coord_df['t_ms'].iloc[0])
            if exposure_trimmed and not starts_at_center:
                # Rat was misplaced — barriers already dropped by trim
                # point. All data is post-barrier.
                timed_phase_end = [first_t]
            else:
                r = duckdb.sql(f"""
                    SELECT trial_number, time_duration
                    FROM '{_parquet("trials")}'
                    WHERE session_id = {session_id}
                    ORDER BY trial_number LIMIT 2
                """).fetchdf()
                if len(r) >= 2:
                    cumulative_s = r['time_duration'].sum()
                    timed_phase_end = [first_t + cumulative_s * 1000]
                else:
                    # Fallback: barrier phase takes ~300s from session start
                    timed_phase_end = [first_t + 300000]
    else:
        # Phase-joined approach for acquisition: coordinates labeled by phase
        phase_coord_df = get_phase_coordinates(session_id)
        if len(phase_coord_df) == 0:
            return {'session_id': session_id, 'status': 'skipped', 'reason': 'no coordinates'}

        # For merged sessions with known bad start regions, trim to the
        # clean portion by timestamp. Key: session_id, value: trim time (ms).
        # For merged sessions with known bad start regions, trim by
        # timestamp. Currently no sessions require this.
        _MERGE_TRIM = {}
        if session_id in _MERGE_TRIM:
            trim_t = _MERGE_TRIM[session_id]
            phase_coord_df = phase_coord_df[
                phase_coord_df['t_ms'] >= trim_t
            ].reset_index(drop=True)

        grid_df = map_session_to_grid(phase_coord_df)
        reward_events = get_trial_rewards(session_id)
        # MazeControl-logged well visits per trial: PICKUP emission in
        # map_to_minigrid_actions gates on overlap with this list so the
        # action stream matches upstream's reward-detection record (no
        # sub-threshold drive-throughs masquerading as visits).
        registered_well_visits = get_trial_well_visits(session_id)
        trial_configs = get_trial_configs(session_id, session_phase)

        # Trim trial_configs to match: skip trials that were trimmed away
        if len(phase_coord_df) > 0 and trial_configs:
            remaining_trials = sorted(phase_coord_df['trial_number'].unique())
            if len(remaining_trials) > 0:
                first_trial = int(remaining_trials[0])
                # trial_configs is 0-indexed, trial_numbers are 1-indexed
                skip = first_trial - 1
                if skip > 0:
                    trial_configs = trial_configs[skip:]

    # 2. Build action sequence
    actions_df = build_action_sequence(
        grid_df,
        reward_events,
        pretrial_boundaries=timed_phase_end if (is_exposure and reward_events) else [],
        seed=seed,
        build_pause=build_pause,
        pause_threshold_ms=pause_threshold_ms,
        trial_configs=trial_configs if trial_configs else None,
        session_number=session_number,
        use_real_pretrial=use_real_pretrial,
        consolidate_pauses=consolidate_pauses,
        registered_well_visits=registered_well_visits,
    )

    if len(actions_df) == 0:
        return {'session_id': session_id, 'status': 'skipped', 'reason': 'no actions generated'}

    # Sync trial_configs to the truncated action stream. build_action_sequence
    # trims at the last rewarded PICKUP; trial_configs must follow so that
    # n_trials == n_rewards == len(trial_configs) downstream.
    if trial_configs:
        n_rewards = int(actions_df['rewarded'].sum())
        trial_configs = trial_configs[:n_rewards]

    # 3. Replay through CornerMazeEnv to attach a per-row pose_label
    # (layout_class_x_y_dir). Pose at row i = env state BEFORE action i,
    # matching the divergence-checker convention. Joins to
    # data/dataframes/minigrid-views-allposes.parquet for view lookups.
    actions_df = actions_df.copy()
    actions_df['pose_label'] = compute_pose_labels(
        actions_df,
        session_phase=session_phase,
        session_number=session_number,
        session_type=session_type,
        cue_goal_orientation=cue_goal_orientation,
        training_group=training_group,
        trial_configs=trial_configs if trial_configs else None,
    )

    # 4. Save to parquet with metadata
    # Filename distinguishes pretrial variant for acquisition (synthetic vs real);
    # exposure has no pretrial concept so the suffix is omitted there.
    os.makedirs(output_dir, exist_ok=True)
    if session_phase == 'Acquisition':
        variant = 'real' if use_real_pretrial else 'synthetic'
        out_path = os.path.join(
            output_dir, f'{subject_name}_{session_number}_{variant}.parquet'
        )
    else:
        out_path = os.path.join(
            output_dir, f'{subject_name}_{session_number}.parquet'
        )

    # Build parquet metadata
    meta = {
        'subject': subject_name,
        'session_number': session_number,
        'session_type': session_type,
        'session_phase': session_phase,
        'session_id': str(session_id),
        'cue_goal_orientation': cue_goal_orientation,
        'seed': str(seed),
        'pretrial_variant': 'real' if use_real_pretrial else 'synthetic',
        'pause_threshold_ms': str(pause_threshold_ms),
        'n_actions': str(len(actions_df)),
        'n_rewards': str(int(actions_df['rewarded'].sum())),
    }
    if trial_configs:
        meta['n_trials'] = str(len(trial_configs))
        meta['trial_configs'] = json.dumps(trial_configs)

    # Write with pyarrow to attach file-level metadata
    table = pa.Table.from_pandas(actions_df, preserve_index=False)
    existing_meta = table.schema.metadata or {}
    merged_meta = {
        **existing_meta,
        **{k.encode(): v.encode() for k, v in meta.items()},
    }
    table = table.replace_schema_metadata(merged_meta)
    pq.write_table(table, out_path)

    elapsed = time.time() - t0
    n_rewards = int(actions_df['rewarded'].sum())

    return {
        'session_id': session_id,
        'status': 'ok',
        'subject': subject_name,
        'session_number': session_number,
        'session_phase': session_phase,
        'n_coords': len(grid_df),
        'n_actions': len(actions_df),
        'n_rewards': n_rewards,
        'n_trials': len(trial_configs) if trial_configs else 0,
        'path': out_path,
        'elapsed_s': round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Build yoked action sequences from analysis parquets.',
    )
    parser.add_argument('--subject', type=str, default=None,
                        help='Subject name (e.g., CM005). Omit for all subjects.')
    parser.add_argument('--session', type=str, default=None,
                        help='Session number (e.g., 1e, 3). Omit for all sessions.')
    parser.add_argument('--all', action='store_true',
                        help='Process all matching sessions.')
    parser.add_argument('--phase', type=str, default=None,
                        choices=['Exposure', 'Acquisition'],
                        help='Filter by session phase.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for turnaround rolls (default: 42).')
    parser.add_argument('--pause', action='store_true', default=True,
                        help='Include pause actions (default).')
    parser.add_argument('--no-pause', action='store_false', dest='pause',
                        help='Exclude pause actions.')
    parser.add_argument('--pause-threshold', type=int, default=1500,
                        help='Pause dwell threshold in ms (default: 1500).')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Output directory (default: data/yoked/).')
    parser.add_argument('--real-pretrial', action='store_true',
                        help='Use real tracking data for pretrial actions '
                             'instead of synthetic sequence.')
    parser.add_argument('--no-consolidate-pauses', action='store_true',
                        help='Emit proportional PAUSE actions for dwells '
                             'instead of collapsing to 1 per dwell.')
    args = parser.parse_args()

    # Validate args
    if not args.all and args.subject is None:
        parser.error('Specify --subject or --all')
    if args.session is not None and args.subject is None:
        parser.error('--session requires --subject')

    # Query sessions
    sessions_df = get_sessions(
        subject=args.subject,
        session_number=args.session,
        session_phase=args.phase,
    )

    # Scope to exposure + acquisition only (unless --phase overrides)
    if args.phase is None:
        sessions_df = sessions_df[
            sessions_df['session_phase'].isin(['Exposure', 'Acquisition'])
        ]

    if len(sessions_df) == 0:
        print('No matching sessions found.')
        sys.exit(1)

    print(f'Found {len(sessions_df)} session(s) to process')
    print()

    # Process each session
    results = []
    for _, row in sessions_df.iterrows():
        label = f"{row['subject_name']} {row['session_number']} ({row['session_phase']})"
        print(f'Processing {label}...', end=' ', flush=True)

        try:
            result = process_session(
                session_id=int(row['session_id']),
                subject_name=row['subject_name'],
                session_number=row['session_number'],
                session_type=row['session_type'],
                session_phase=row['session_phase'],
                cue_goal_orientation=row['cue_goal_orientation'],
                training_group=row['training_group'],
                output_dir=args.output_dir,
                seed=args.seed,
                build_pause=args.pause,
                pause_threshold_ms=args.pause_threshold,
                use_real_pretrial=args.real_pretrial,
                consolidate_pauses=not args.no_consolidate_pauses,
            )
            results.append(result)

            if result['status'] == 'ok':
                parts = [f"{result['n_actions']} actions"]
                parts.append(f"{result['n_rewards']} rewards")
                if result['n_trials'] > 0:
                    parts.append(f"{result['n_trials']} trials")
                parts.append(f"{result['elapsed_s']}s")
                print(', '.join(parts))
            else:
                print(f"SKIPPED: {result['reason']}")

        except Exception as e:
            print(f'ERROR: {e}')
            results.append({
                'session_id': int(row['session_id']),
                'status': 'error',
                'error': str(e),
            })

    # Summary
    ok = sum(1 for r in results if r['status'] == 'ok')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    errors = sum(1 for r in results if r['status'] == 'error')
    print(f'\nDone: {ok} ok, {skipped} skipped, {errors} errors')


if __name__ == '__main__':
    main()
