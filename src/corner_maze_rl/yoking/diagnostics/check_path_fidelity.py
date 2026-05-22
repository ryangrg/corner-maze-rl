"""Scan in-scope Acquisition trials for yoked-vs-rat path fidelity.

Detects yoking artifacts where the yoked agent visits env-grid cells that
the rat itself never reached. Driving case: BFS-bridging through cells in
the opposite half of the maze when the rat's coord briefly maps to an
isolated barrier-pocket cell (e.g. (2,6) between barriers (2,5)/(2,7) in
trl_e_n_xx layouts). The fidelity-aware remap should keep the agent on
the rat's actual side; this tool surfaces cases where it doesn't.

Per-trial procedure (trial-phase only — pretrial uses canonical actions,
ITI handling is structurally different):

  1. Pull rat coords for the trial-phase time window from upstream.
  2. Map zones → env grid via the yoking's zone_to_grid (NO barrier
     remap) so we get the rat's geometric grid trace.
  3. Pull yoked agent positions for the same trial from the action
     stream.
  4. Compute: how many yoked-cells are NOT in the rat's geometric set,
     AND lie more than `min_dist` manhattan from any rat cell. Those are
     "phantom" cells the BFS bridge invented.

Output: a CSV-like table sorted by phantom-cell count, plus a summary.

Usage:
    python -m corner_maze_rl.yoking.diagnostics.check_path_fidelity
    python -m corner_maze_rl.yoking.diagnostics.check_path_fidelity --subject CM015
    python -m corner_maze_rl.yoking.diagnostics.check_path_fidelity --variant real_pretrial
    python -m corner_maze_rl.yoking.diagnostics.check_path_fidelity --min-dist 3 --top 30
"""
from __future__ import annotations

import argparse
import json

import duckdb
import numpy as np
import pandas as pd

from corner_maze_rl.yoking.map_to_minigrid import zone_to_grid


def _manhattan_min(target, points):
    """Min manhattan distance from ``target`` to any cell in ``points`` set."""
    if not points:
        return float('inf')
    tx, ty = target
    return min(abs(tx - px) + abs(ty - py) for px, py in points)


def _trial_yoked_rows(action_df, rewarded_steps, trial_idx):
    """Return the trial-cycle rows of the yoked stream for one trial.

    A "cycle" spans from just after the previous rewarded PICKUP (or step
    0 for trial 0) through the current rewarded PICKUP inclusive — i.e.
    ITI of the previous trial + pretrial of this trial + trial-phase of
    this trial. This matches the upstream phase windowing used for the
    rat-coord side, so the comparison is apples-to-apples.
    """
    if trial_idx == 0:
        lo = 0
    else:
        lo = int(rewarded_steps[trial_idx - 1]) + 1
    hi = int(rewarded_steps[trial_idx])
    return action_df[(action_df['step'] >= lo) & (action_df['step'] <= hi)]


def _rat_trial_cells(coords_df, t_start, t_end):
    """Map rat coords during [t_start, t_end] to env-grid cells (no barrier
    remap). Returns a set of unique (gx, gy) tuples."""
    window = coords_df[
        (coords_df['t_ms'] >= t_start) & (coords_df['t_ms'] <= t_end)
    ]
    cells = set()
    for _, r in window.iterrows():
        gx, gy = zone_to_grid(int(r['zone']), int(r['x']), int(r['y']))
        if gx > 0 and gy > 0:  # skip off-grid
            cells.add((gx, gy))
    return cells


def scan(dataset_dir, upstream_dir, variant, subject=None, session_number=None,
         min_dist=3):
    """Scan every in-scope Acquisition trial for phantom yoked cells.

    Returns a DataFrame with one row per trial that has any phantom cells.
    """
    actions_table = {
        'synthetic_pretrial': 'actions_synthetic_pretrial.parquet',
        'real_pretrial': 'actions_real_pretrial.parquet',
    }[variant]

    # Load sessions + subjects to get the in-scope set
    sess_filter = ['1=1']
    if subject is not None:
        sess_filter.append(f"sub.subject_name = '{subject}'")
    if session_number is not None:
        sess_filter.append(f"s.session_number = '{session_number}'")
    where = ' AND '.join(sess_filter)
    sessions = duckdb.sql(f"""
        SELECT s.session_id, sub.subject_name, s.session_number,
               s.session_type, s.trial_configs
        FROM '{dataset_dir}/sessions.parquet' s
        JOIN '{dataset_dir}/subjects.parquet' sub USING (subject_id)
        WHERE s.session_phase = 'Acquisition' AND {where}
        ORDER BY sub.subject_name, s.session_number
    """).fetchdf()

    rows = []
    for _, srow in sessions.iterrows():
        sid = int(srow['session_id'])
        # Per-session loads
        try:
            actions = duckdb.sql(f"""
                SELECT step, action, grid_x, grid_y, direction, rewarded
                FROM '{dataset_dir}/{actions_table}'
                WHERE session_id = {sid}
                ORDER BY step
            """).fetchdf()
        except Exception:
            continue
        if len(actions) == 0:
            continue
        rewarded_steps = actions.loc[actions['rewarded'] == 1, 'step'].tolist()
        configs = json.loads(srow['trial_configs']) if srow['trial_configs'] else []
        if len(rewarded_steps) != len(configs):
            # build_dataset.py asserts this — should never happen in-scope
            continue

        # Cycle window for trial K = previous trial's reward to current
        # trial's reward (== yoked's "between rewards" slice). Includes
        # ITI of trial K-1 + pretrial of K + trial-phase of K, which is
        # exactly what the yoked stream emits between two rewarded
        # PICKUPs. Trial 1's cycle includes any pre-pretrial coord too.
        phases = duckdb.sql(f"""
            SELECT trial_number, phase, t_start_ms, t_end_ms
            FROM '{upstream_dir}/phases.parquet'
            WHERE session_id = {sid}
            ORDER BY t_start_ms
        """).fetchdf()
        trial_phases = phases[phases['phase'] == 'trial'].set_index('trial_number')
        coords = duckdb.sql(f"""
            SELECT t_ms, zone, x, y FROM '{upstream_dir}/coordinates.parquet'
            WHERE session_id = {sid}
        """).fetchdf().sort_values('t_ms').reset_index(drop=True)

        for trial_idx in range(len(configs)):
            trial_num = trial_idx + 1
            if trial_num not in trial_phases.index:
                continue
            t_end = int(trial_phases.loc[trial_num, 't_end_ms'])
            if trial_num > 1 and (trial_num - 1) in trial_phases.index:
                t_start = int(trial_phases.loc[trial_num - 1, 't_end_ms'])
            else:
                t_start = int(trial_phases.loc[trial_num, 't_start_ms'])

            rat_cells = _rat_trial_cells(coords, t_start, t_end)
            if not rat_cells:
                continue

            # Yoked agent's trial-phase rows (after the pretrial block —
            # the trial-phase emission begins after the pretrial finishes
            # and before the rewarded PICKUP). Simplest approximation:
            # the trial's yoked steps that match cells the rat could
            # plausibly have been in during the trial window.
            trial_rows = _trial_yoked_rows(actions, rewarded_steps, trial_idx)
            yoked_cells = set(zip(trial_rows['grid_x'].astype(int).tolist(),
                                  trial_rows['grid_y'].astype(int).tolist()))

            # Phantom cells: yoked-only AND far from any rat cell.
            phantom = []
            for c in yoked_cells - rat_cells:
                d = _manhattan_min(c, rat_cells)
                if d >= min_dist:
                    phantom.append((c, d))
            if not phantom:
                continue

            # Count yoked rows whose cell is phantom (not just unique cells).
            phantom_cell_set = {c for c, _ in phantom}
            phantom_row_mask = trial_rows.apply(
                lambda r: (int(r['grid_x']), int(r['grid_y'])) in phantom_cell_set,
                axis=1,
            )
            rows.append({
                'subject': srow['subject_name'],
                'session': srow['session_number'],
                'session_type': srow['session_type'],
                'trial': trial_num,
                'goal_idx': configs[trial_idx][2],
                'n_yoked_rows_in_trial': len(trial_rows),
                'n_rat_cells': len(rat_cells),
                'n_yoked_cells': len(yoked_cells),
                'n_phantom_cells': len(phantom_cell_set),
                'n_phantom_rows': int(phantom_row_mask.sum()),
                'max_phantom_dist': max(d for _, d in phantom),
                'example_phantom': sorted(phantom_cell_set)[0],
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=str, default=None)
    parser.add_argument('--session', type=str, default=None)
    parser.add_argument('--variant', default='synthetic_pretrial',
                        choices=['synthetic_pretrial', 'real_pretrial'])
    parser.add_argument('--dataset-dir', default='data/yoked/dataset')
    parser.add_argument(
        '--upstream-dir', default=None,
        help='Defaults to $CORNER_MAZE_ANALYSIS_DIR or '
             'corner-maze-analysis/data/processed.',
    )
    parser.add_argument('--min-dist', type=int, default=3,
                        help='Phantom = yoked cell with manhattan distance '
                             '>= MIN_DIST from any rat-coord cell. (default 3)')
    parser.add_argument('--top', type=int, default=20,
                        help='Show this many worst trials in the summary.')
    parser.add_argument('--csv', type=str, default=None,
                        help='Optional CSV output path for the full table.')
    args = parser.parse_args()

    upstream_dir = args.upstream_dir
    if upstream_dir is None:
        import os
        upstream_dir = os.environ.get(
            'CORNER_MAZE_ANALYSIS_DIR',
            'data/analysis',
        )

    df = scan(args.dataset_dir, upstream_dir, args.variant,
              subject=args.subject, session_number=args.session,
              min_dist=args.min_dist)

    print(f'Path-fidelity scan ({args.variant}, min_dist={args.min_dist}):')
    if len(df) == 0:
        print('  No trials flagged. Path stays consistent with rat coords.')
        return

    n_subj = df['subject'].nunique()
    n_sess = df.groupby(['subject', 'session']).ngroups
    print(f'  Flagged trials: {len(df)} '
          f'across {n_sess} sessions ({n_subj} subjects)')
    print(f'  Total phantom rows in yoked stream: {df["n_phantom_rows"].sum()}')
    print()
    print(f'Worst {args.top} trials by phantom-row count:')
    cols = ['subject', 'session', 'session_type', 'trial',
            'n_phantom_rows', 'n_phantom_cells', 'max_phantom_dist',
            'n_yoked_rows_in_trial', 'example_phantom']
    print(df.sort_values('n_phantom_rows', ascending=False).head(args.top)[cols].to_string(index=False))

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f'\nFull table written to {args.csv}')


if __name__ == '__main__':
    main()
