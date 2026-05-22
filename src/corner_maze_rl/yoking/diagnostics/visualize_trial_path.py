"""Visualize a single trial's rat coord trace vs the yoked agent's grid path.

Side-by-side rendering on the 13×13 env grid. Active barriers for the
trial's layout are drawn as filled cells; the rat's coord-derived
zone→grid path is drawn in one color; the yoked agent's per-step grid
positions are drawn in another. Step markers help see ordering. Designed
for quickly inspecting trials surfaced by ``check_path_fidelity``.

Usage:
    python -m corner_maze_rl.yoking.diagnostics.visualize_trial_path \\
        --subject CM015 --session 5 --trial 5
    python -m corner_maze_rl.yoking.diagnostics.visualize_trial_path \\
        --subject CM015 --session 5 --trial 5 --variant real_pretrial \\
        --save /tmp/cm015_5_t5.png

Requires matplotlib (already installed in the project venv).
"""
from __future__ import annotations

import argparse
import json

import duckdb
import numpy as np

from corner_maze_rl.env.constants import BARRIER_LOCATIONS, WELL_LOCATIONS
from corner_maze_rl.yoking.map_to_minigrid import zone_to_grid


# Trial-layout barrier-index lookup. Per env's base_trl_layouts the layout
# tuple's slots 1–16 are the 16 barrier flags. We only need the slot
# values that are 1 to know which BARRIER_LOCATIONS are active.
_TRIAL_LAYOUT_BARRIERS = {
    # start_arm: barrier-flag pattern across the 16 barrier slots
    'n': (0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0),
    'e': (1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0),
    's': (0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1),
    'w': (1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0),
}

_ARM_LETTER = ['n', 'e', 's', 'w']
_GOAL_LETTER = ['ne', 'se', 'sw', 'nw']


def _trial_barriers(start_arm_idx):
    """Active barrier positions for the given start_arm of a trial layout."""
    pattern = _TRIAL_LAYOUT_BARRIERS[_ARM_LETTER[start_arm_idx]]
    return {BARRIER_LOCATIONS[i] for i, flag in enumerate(pattern) if flag}


def _load_trial(subject, session_number, trial_number, variant, dataset_dir,
                upstream_dir):
    """Fetch all data needed to render the trial.

    Returns a dict with: session_id, trial_config, trial_phase_window,
    rat_coords (DataFrame with t_ms, gx, gy), yoked_rows (DataFrame of
    action stream slice for the trial cycle).
    """
    actions_table = {
        'synthetic_pretrial': 'actions_synthetic_pretrial.parquet',
        'real_pretrial': 'actions_real_pretrial.parquet',
    }[variant]

    sess_row = duckdb.sql(f"""
        SELECT s.session_id, s.trial_configs
        FROM '{dataset_dir}/sessions.parquet' s
        JOIN '{dataset_dir}/subjects.parquet' sub USING (subject_id)
        WHERE sub.subject_name = '{subject}'
          AND s.session_number = '{session_number}'
    """).fetchone()
    if sess_row is None:
        raise SystemExit(f'No session {subject}/{session_number} in dataset')
    sid, tc_json = sess_row
    configs = json.loads(tc_json)
    if trial_number > len(configs):
        raise SystemExit(
            f'trial {trial_number} exceeds n_trials={len(configs)}'
        )
    trial_config = configs[trial_number - 1]

    phase = duckdb.sql(f"""
        SELECT t_start_ms, t_end_ms FROM '{upstream_dir}/phases.parquet'
        WHERE session_id = {sid} AND phase = 'trial'
          AND trial_number = {trial_number}
    """).fetchone()
    if phase is None:
        raise SystemExit(f'No trial-phase row for trial {trial_number}')
    t_start, t_end = int(phase[0]), int(phase[1])

    coords = duckdb.sql(f"""
        SELECT t_ms, zone, x, y FROM '{upstream_dir}/coordinates.parquet'
        WHERE session_id = {sid} AND t_ms BETWEEN {t_start} AND {t_end}
        ORDER BY t_ms
    """).fetchdf()
    gx_list, gy_list = [], []
    for _, r in coords.iterrows():
        gx, gy = zone_to_grid(int(r['zone']), int(r['x']), int(r['y']))
        gx_list.append(gx); gy_list.append(gy)
    coords = coords.assign(gx=gx_list, gy=gy_list)
    coords = coords[(coords['gx'] > 0) & (coords['gy'] > 0)]

    actions = duckdb.sql(f"""
        SELECT step, action, grid_x, grid_y, direction, rewarded
        FROM '{dataset_dir}/{actions_table}'
        WHERE session_id = {sid} ORDER BY step
    """).fetchdf()
    rewarded_steps = actions.loc[actions['rewarded'] == 1, 'step'].tolist()
    if trial_number - 1 >= len(rewarded_steps):
        raise SystemExit(
            f'trial {trial_number}: not enough rewarded PICKUPs in stream'
        )
    if trial_number == 1:
        lo = 0
    else:
        lo = int(rewarded_steps[trial_number - 2]) + 1
    hi = int(rewarded_steps[trial_number - 1])
    yoked = actions[(actions['step'] >= lo) & (actions['step'] <= hi)].copy()

    return {
        'session_id': sid,
        'trial_config': trial_config,
        'trial_phase_window': (t_start, t_end),
        'rat_coords': coords,
        'yoked_rows': yoked,
    }


def _draw(ax, title, barriers, wells_set, paths, legend=True):
    """Draw the 13×13 env grid with barriers + wells + path overlays.

    ``paths`` is a list of (label, color, points) where points is a list
    of (gx, gy) tuples in order.
    """
    # Background grid (light gray cells for walls = barriers)
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(12.5, -0.5)  # invert y so env-north (y=1) is at top
    ax.set_aspect('equal')
    ax.set_xticks(range(13))
    ax.set_yticks(range(13))
    ax.grid(True, color='#dddddd', linewidth=0.5)
    # Barriers as dark filled squares
    for bx, by in barriers:
        ax.add_patch(__import__('matplotlib.patches', fromlist=['Rectangle']).Rectangle(
            (bx - 0.5, by - 0.5), 1, 1, color='#444444', alpha=0.85,
        ))
    # Wells as circles
    for wx, wy in WELL_LOCATIONS:
        ax.scatter([wx], [wy],
                   marker='o', s=220, facecolor='none',
                   edgecolor='#aa00aa' if (wx, wy) in wells_set else '#888888',
                   linewidth=2)
    # Paths
    for label, color, pts in paths:
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, '-', color=color, alpha=0.55, linewidth=1.2,
                label=label)
        ax.scatter(xs[:1], ys[:1], marker='o', color=color, s=40,
                   zorder=5, label=f'{label} start')
        ax.scatter(xs[-1:], ys[-1:], marker='s', color=color, s=40,
                   zorder=5, label=f'{label} end')
    if legend:
        ax.legend(loc='upper right', fontsize=8, framealpha=0.85)
    ax.set_title(title, fontsize=10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', required=True)
    parser.add_argument('--session', required=True)
    parser.add_argument('--trial', type=int, required=True)
    parser.add_argument('--variant', default='synthetic_pretrial',
                        choices=['synthetic_pretrial', 'real_pretrial'])
    parser.add_argument('--dataset-dir', default='data/yoked/dataset')
    parser.add_argument('--upstream-dir', default=None)
    parser.add_argument('--save', default=None,
                        help='Optional PNG output path (saves instead of show).')
    args = parser.parse_args()

    import os
    upstream_dir = args.upstream_dir or os.environ.get(
        'CORNER_MAZE_ANALYSIS_DIR', 'data/analysis')

    data = _load_trial(args.subject, args.session, args.trial,
                       args.variant, args.dataset_dir, upstream_dir)
    cfg = data['trial_config']
    barriers = _trial_barriers(cfg[0])
    goal_well = WELL_LOCATIONS[{'ne': 3, 'se': 0, 'sw': 1, 'nw': 2}[
        _GOAL_LETTER[cfg[2]]
    ]]  # see WELL_LOCATIONS order in env/constants.py

    rat_pts = list(zip(data['rat_coords']['gx'].astype(int).tolist(),
                       data['rat_coords']['gy'].astype(int).tolist()))
    yoked_pts = list(zip(data['yoked_rows']['grid_x'].astype(int).tolist(),
                         data['yoked_rows']['grid_y'].astype(int).tolist()))

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
    title_base = (f'{args.subject}/{args.session} trial {args.trial} '
                  f'(arm={_ARM_LETTER[cfg[0]]}, cue={cfg[1]}, '
                  f'goal={_GOAL_LETTER[cfg[2]]}) — {args.variant}')
    _draw(axes[0], 'Rat coord-trace (zone→grid)\n' + title_base,
          barriers, {goal_well},
          [('rat', '#1f77b4', rat_pts)])
    _draw(axes[1], 'Yoked agent grid path\n' + title_base,
          barriers, {goal_well},
          [('yoked', '#d62728', yoked_pts)])

    # Mini-summary text below
    rat_set = set(rat_pts)
    yoked_set = set(yoked_pts)
    phantom = sorted(yoked_set - rat_set)
    fig.suptitle(
        f'n_rat_frames={len(rat_pts)}, n_yoked_steps={len(yoked_pts)}  |  '
        f'unique rat cells={len(rat_set)}, unique yoked cells={len(yoked_set)}  |  '
        f'yoked-only cells (potential phantoms)={len(phantom)}',
        fontsize=10,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))

    if args.save:
        plt.savefig(args.save, dpi=130, bbox_inches='tight')
        print(f'Saved figure to {args.save}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
