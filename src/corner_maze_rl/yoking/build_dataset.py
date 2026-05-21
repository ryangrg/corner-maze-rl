"""Build a normalized yoked dataset from individual session parquets.

Reads the per-session parquet files in data/yoked/ (produced by build_yoked.py)
and the upstream analysis database, then emits 5 tables under data/yoked/dataset/:

    subjects.parquet                     — one row per subject
    sessions.parquet                     — one row per session_id (all phases)
    actions_synthetic_pretrial.parquet   — Acquisition only, synthetic pretrial
    actions_real_pretrial.parquet        — Acquisition only, real pretrial
    actions_exposure.parquet             — Exposure only

Per-session input filenames distinguish the acquisition pretrial variant via a
trailing `_synthetic` or `_real` suffix; exposure files have no suffix.

Post-build assertions verify every session_id in any actions_*.parquet has a
matching row in sessions.parquet.

Usage:
    python -m corner_maze_rl.yoking.build_dataset
    python -m corner_maze_rl.yoking.build_dataset --dir data/yoked --out data/yoked/dataset
"""
import argparse
import os
from glob import glob

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .data_loader import _parquet


def compute_actions_to_reward(action: np.ndarray, rewarded: np.ndarray) -> np.ndarray:
    """For each row, count of action steps remaining until the next correct
    PICKUP (``action == 3 & rewarded == 1``). Zero at the rewarded PICKUP itself.

    Pure structural — cost-agnostic. Requires that the session ends at a
    correct PICKUP (the post-truncation invariant); otherwise raises.
    """
    pickup = (action == 3) & (rewarded == 1)
    n = len(action)
    if n == 0:
        return np.empty(0, dtype=np.int32)
    # For each t, find min index ≥ t with pickup True. Done via reverse-cummin.
    indices = np.where(pickup, np.arange(n), n)
    next_pickup = np.minimum.accumulate(indices[::-1])[::-1]
    if (next_pickup >= n).any():
        bad = int(np.where(next_pickup >= n)[0][0])
        raise ValueError(
            f"compute_actions_to_reward: row {bad} has no following rewarded PICKUP; "
            "session must end at action==3 & rewarded==1"
        )
    return (next_pickup - np.arange(n)).astype(np.int32)


def _read_session_parquet(fpath):
    """Return (metadata_dict, actions_df) for a per-session parquet."""
    pf = pq.read_table(fpath)
    meta = {
        k.decode(): v.decode()
        for k, v in (pf.schema.metadata or {}).items()
        if k not in (b'pandas', b'ARROW:schema')
    }
    return meta, pf.to_pandas()


def _bucket_for(meta, fname):
    """Decide which output action table this per-session file belongs to.

    Returns one of: 'synthetic_pretrial', 'real_pretrial', 'exposure', or None.
    """
    phase = meta.get('session_phase', '')
    if phase == 'Exposure':
        return 'exposure'
    if phase == 'Acquisition':
        # Prefer explicit metadata; fall back to filename suffix for older files.
        variant = meta.get('pretrial_variant')
        if variant is None:
            stem = os.path.splitext(os.path.basename(fname))[0]
            if stem.endswith('_real'):
                variant = 'real'
            elif stem.endswith('_synthetic'):
                variant = 'synthetic'
        if variant == 'real':
            return 'real_pretrial'
        if variant == 'synthetic':
            return 'synthetic_pretrial'
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Build normalized yoked dataset from session parquets.',
    )
    parser.add_argument('--dir', type=str, default='data/yoked',
                        help='Directory containing per-session parquets.')
    parser.add_argument('--out', type=str, default='data/yoked/dataset',
                        help='Output directory for dataset tables.')
    args = parser.parse_args()

    files = sorted(glob(os.path.join(args.dir, '*.parquet')))
    if not files:
        print('No parquet files found.')
        return

    # Pass 1: read all per-session parquets, bucket by output table.
    buckets = {'synthetic_pretrial': [], 'real_pretrial': [], 'exposure': []}
    session_meta = {}  # session_id -> meta dict (first occurrence wins)
    skipped = []

    for fpath in files:
        meta, actions_df = _read_session_parquet(fpath)
        if 'session_id' not in meta:
            skipped.append((fpath, 'no session_id'))
            continue

        bucket = _bucket_for(meta, fpath)
        if bucket is None:
            skipped.append((fpath, f"unrecognized phase/variant: {meta.get('session_phase')!r}"))
            continue

        session_id = int(meta['session_id'])
        actions_df = actions_df.copy()
        actions_df['session_id'] = session_id
        actions_df['actions_to_reward'] = compute_actions_to_reward(
            actions_df['action'].to_numpy(),
            actions_df['rewarded'].to_numpy(),
        )
        buckets[bucket].append(actions_df)

        # Record session metadata once (synthetic and real share the same upstream session).
        if session_id not in session_meta:
            session_meta[session_id] = meta

    if skipped:
        print(f'Skipped {len(skipped)} files:')
        for fpath, reason in skipped[:10]:
            print(f'  {os.path.basename(fpath)}: {reason}')
        if len(skipped) > 10:
            print(f'  ... and {len(skipped) - 10} more')

    # ── Build subjects table from upstream ──────────────────────
    session_ids = sorted(session_meta.keys())
    if not session_ids:
        print('No usable session parquets; nothing to write.')
        return
    id_list = ','.join(str(i) for i in session_ids)
    subjects_df = duckdb.sql(f"""
        SELECT DISTINCT
            sub.subject_id,
            sub.name AS subject_name,
            sub.sex,
            sub.cue_goal_orientation,
            sub.training_group,
            sub.approach_to_goal
        FROM '{_parquet("sessions")}' sess
        JOIN '{_parquet("subjects")}' sub ON sess.subject_id = sub.subject_id
        WHERE sess.session_id IN ({id_list})
        ORDER BY sub.subject_id
    """).fetchdf()
    name_to_subject_id = dict(zip(subjects_df['subject_name'], subjects_df['subject_id']))

    # ── Build sessions table (one row per session_id, all phases) ──
    sessions_rows = []
    for session_id in session_ids:
        meta = session_meta[session_id]
        subject_name = meta.get('subject', '')
        subject_id = name_to_subject_id.get(subject_name)
        if subject_id is None:
            skipped.append((str(session_id), f"unknown subject {subject_name!r}"))
            continue
        sessions_rows.append({
            'session_id': session_id,
            'subject_id': int(subject_id),
            'session_number': meta.get('session_number', ''),
            'session_type': meta.get('session_type', ''),
            'session_phase': meta.get('session_phase', ''),
            'cue_goal_orientation': meta.get('cue_goal_orientation', ''),
            'n_actions': int(meta.get('n_actions', 0)),
            'n_rewards': int(meta.get('n_rewards', 0)),
            'trial_configs': meta.get('trial_configs', '[]'),
            'seed': int(meta.get('seed', 42)),
            'n_trials': int(meta.get('n_trials', 0)),
        })

    sessions_df = pd.DataFrame(sessions_rows).sort_values(
        ['subject_id', 'session_number']
    ).reset_index(drop=True)

    # ── Concat + sort actions per bucket ────────────────────────
    action_cols = ['session_id', 'step', 'action', 'grid_x', 'grid_y',
                   'direction', 'rewarded', 'actions_to_reward', 'pose_label']
    output_actions = {}
    for bucket, parts in buckets.items():
        if not parts:
            output_actions[bucket] = None
            continue
        df = pd.concat(parts, ignore_index=True)[action_cols]
        df = df.sort_values(['session_id', 'step']).reset_index(drop=True)
        output_actions[bucket] = df

    # ── Post-build assertions ───────────────────────────────────
    valid_session_ids = set(sessions_df['session_id'])
    for bucket, df in output_actions.items():
        if df is None:
            continue
        bucket_ids = set(df['session_id'].unique())
        orphans = bucket_ids - valid_session_ids
        if orphans:
            raise AssertionError(
                f'{bucket}: {len(orphans)} session_ids missing from sessions.parquet '
                f'(first 5: {sorted(orphans)[:5]})'
            )

    # Phase-vs-bucket consistency: acquisition tables only acquisition session_ids,
    # exposure table only exposure session_ids.
    phase_by_id = dict(zip(sessions_df['session_id'], sessions_df['session_phase']))
    for bucket, df in output_actions.items():
        if df is None:
            continue
        expected = 'Exposure' if bucket == 'exposure' else 'Acquisition'
        wrong = [sid for sid in df['session_id'].unique()
                 if phase_by_id.get(sid) != expected]
        if wrong:
            raise AssertionError(
                f'{bucket}: {len(wrong)} session_ids have wrong phase '
                f'(expected {expected!r}, first 5: {wrong[:5]})'
            )

    # Terminal-action invariant: every session's last row must be a rewarded
    # PICKUP. DT per-trial RTG depends on this. The pytest in
    # tests/test_yoked_dataset_invariants.py is the durable contract; this
    # is the in-build smoke that fails the rebuild fast if anything slips.
    for bucket, df in output_actions.items():
        if df is None:
            continue
        last_rows = df.sort_values(['session_id', 'step']).groupby(
            'session_id', as_index=False
        ).tail(1)
        bad = last_rows[(last_rows['rewarded'] != 1) | (last_rows['action'] != 3)]
        if not bad.empty:
            raise AssertionError(
                f'{bucket}: {len(bad)} sessions not ending at rewarded PICKUP '
                f'(first 5 session_ids: {bad["session_id"].tolist()[:5]})'
            )

    # Per-Acquisition-row count consistency: n_trials == n_rewards == len(trial_configs).
    import json as _json
    bad_counts = []
    for _, row in sessions_df.iterrows():
        if row['session_phase'] != 'Acquisition':
            continue
        configs = _json.loads(row['trial_configs']) if row['trial_configs'] else []
        if not (row['n_trials'] == row['n_rewards'] == len(configs)):
            bad_counts.append(
                (int(row['session_id']), int(row['n_trials']),
                 int(row['n_rewards']), len(configs))
            )
    if bad_counts:
        raise AssertionError(
            f'{len(bad_counts)} Acquisition sessions with '
            f'n_trials/n_rewards/len(trial_configs) mismatch '
            f'(session_id, n_trials, n_rewards, len_configs; first 5): '
            f'{bad_counts[:5]}'
        )

    # ── Write ──────────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)

    subjects_path = os.path.join(args.out, 'subjects.parquet')
    sessions_path = os.path.join(args.out, 'sessions.parquet')
    subjects_df.to_parquet(subjects_path, index=False)
    sessions_df.to_parquet(sessions_path, index=False)

    bucket_filename = {
        'synthetic_pretrial': 'actions_synthetic_pretrial.parquet',
        'real_pretrial':      'actions_real_pretrial.parquet',
        'exposure':           'actions_exposure.parquet',
    }

    print(f'\nWritten to {args.out}/')
    print(f'  subjects.parquet: {len(subjects_df)} rows, {os.path.getsize(subjects_path)/1024:.0f} KB')
    print(f'  sessions.parquet: {len(sessions_df)} rows, {os.path.getsize(sessions_path)/1024:.0f} KB')
    for bucket, fname in bucket_filename.items():
        df = output_actions[bucket]
        if df is None:
            print(f'  {fname}: (skipped — no input)')
            continue
        out_path = os.path.join(args.out, fname)
        df.to_parquet(out_path, index=False)
        print(f'  {fname}: {df["session_id"].nunique()} sessions, '
              f'{len(df)} rows, {os.path.getsize(out_path)/1024:.0f} KB')

    print(f'\nSessions phase breakdown:')
    print(sessions_df.groupby('session_phase').size().to_string())


if __name__ == '__main__':
    main()
