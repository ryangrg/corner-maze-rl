"""Extract per-rat acquisition + probe metrics from corner-maze-analysis.

Produces ``data/rat_benchmarks/per_subject.parquet`` — one row per
manuscript-scope rat with metrics structured to match the agent-side
metrics computed in notebook 03A:

  * ``n_sessions_to_acquire``  — count of acquisition-phase sessions
                                 for that subject (the rat-side
                                 ``sessions_to_acq`` metric per legacy
                                 ``build_pandas_table.py:1007``).
  * ``n_steps_to_acquire``     — sum of MiniGrid action steps across the
                                 subject's acquisition sessions (from
                                 the yoked dataset's ``n_actions``
                                 column, which counts the real-pretrial
                                 action stream). Companion column
                                 ``n_yoked_acq_sessions`` lets consumers
                                 spot yoking gaps; NaN if no yoked data
                                 exists for that subject.
  * ``acquired``               — True if the rat ran any probe session
                                 (proxy for "reached criterion in
                                 protocol"). Rats that didn't acquire
                                 wouldn't have been advanced to probes.
  * ``rate_<probe>``           — fraction correct on each probe type
                                 using the rat-protocol denominators:
                                   Novel Route: correct novel-tagged / 16
                                   Dark:        correct trials       / 32
                                   Rotate:      correct trials       / 16
                                   Reversal:    correct reversal-tagged / 64

Probe-session → label mapping (from
``corner-maze-analysis/docs/parquet_schemas.md`` + legacy
``build_pandas_table.py:1346,1387,1429,1500``):

  * Novel Route ← Fixed Cue 2a, Dark Detour, Dark Detour No Cue,
                  Rotate Detour, Rotate Detour Moving,
                  Fixed Cue Novel Route Twist
  * Dark        ← Fixed No Cue, Fixed No Cue Twist  (32 trials each)
  * Rotate      ← Fixed Cue Rotate, Fixed Cue Rotate Twist  (16 each)
  * Reversal    ← Fixed Cue Switch, Fixed Cue Switch Twist,
                  Dark Reverse, Rotate Reverse

Novel-tagged / reversal-tagged trial identification (no `trial_type`
column in the new parquet, so derive):

  * **Novel Route**: trials with ``start_arm`` NOT in the set of
    start_arms appearing in the session's first 16 trials. This gives
    the 16 novel-start trials in a 40-trial Novel Route session
    (verified for CM001 / Fixed Cue 2a). Matches legacy
    ``novel_route_score()`` semantics.
  * **Reversal**: trials past trial_number 16 in a Reversal session
    (the first 16 are pre-probe standards on the original
    contingency). Matches legacy ``reversal_block_scores()`` /
    ``std_trials_crit()`` pre-probe split.

Manuscript-scope filter: drops ``VC_DREADDs`` (out of scope) and the
8 named exclusions in ``corner-maze-rl/CLAUDE.md`` (CM030, CM032,
CM033, CM059, CM028, CM034, CM039, CM048). Result: 48 in-scope
subjects (PI=12, PI+VC=17, PI+VC_f1=6, VC=13).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ── Source / destination paths ────────────────────────────────
ANALYSIS_DATA = Path('/Users/ryangrgurich/Code/python-dev/corner-maze-analysis/data/processed')
RL_REPO       = Path(__file__).resolve().parents[1]
YOKED_DATA    = RL_REPO / 'data' / 'yoked' / 'dataset'
OUT_DIR       = RL_REPO / 'data' / 'rat_benchmarks'

# ── Manuscript-scope exclusions (corner-maze-rl/CLAUDE.md) ────
EXCLUDE_NAMES = {
    'CM030', 'CM032', 'CM033',           # PI
    'CM059',                              # PI+VC_f1
    'CM028', 'CM034', 'CM039', 'CM048',  # VC
}
IN_SCOPE_GROUPS = {'PI', 'PI+VC', 'PI+VC_f1', 'VC'}

# ── Probe-session → label + denominator config ────────────────
# Mirrors notebook 03A's PROBE_SCORING dict (env-side names) so the
# rat-side and agent-side metrics are structurally identical.
PROBE_CONFIG: dict[str, dict] = {
    'Novel Route': {
        'session_types': {'Fixed Cue 2a', 'Dark Detour', 'Dark Detour No Cue',
                          'Rotate Detour', 'Rotate Detour Moving',
                          'Fixed Cue Novel Route Twist'},
        'possible_per_session': 16,
        'tag_method': 'novel_start_arm',   # trials w/ start_arm not in first-16
    },
    'Dark': {
        'session_types': {'Fixed No Cue', 'Fixed No Cue Twist'},
        'possible_per_session': 32,
        'tag_method': 'all_trials',
    },
    'Rotate': {
        'session_types': {'Fixed Cue Rotate', 'Fixed Cue Rotate Twist'},
        'possible_per_session': 16,
        'tag_method': 'all_trials',
    },
    'Reversal': {
        'session_types': {'Fixed Cue Switch', 'Fixed Cue Switch Twist',
                          'Dark Reverse', 'Rotate Reverse'},
        'possible_per_session': 64,
        'tag_method': 'post_pre_probe',    # trials 17+ (pre-probe is 1-16)
    },
}


# ── Trial-tagging helpers ─────────────────────────────────────

def _tag_trials(trials_in_session: pd.DataFrame, method: str) -> pd.Series:
    """Boolean Series indicating which trials count toward the probe score."""
    if method == 'all_trials':
        return pd.Series(True, index=trials_in_session.index)

    if method == 'post_pre_probe':
        # Reversal sessions: first 16 trials are pre-probe; reversal-tagged = 17+
        return trials_in_session['trial_number'] > 16

    if method == 'novel_start_arm':
        # Novel Route sessions: novel-tagged trials = trials with start_arm
        # not present in the first 16 trials of this session.
        first16 = trials_in_session[trials_in_session['trial_number'] <= 16]
        warmup_starts = set(first16['start_arm'].dropna().unique())
        return ~trials_in_session['start_arm'].isin(warmup_starts)

    raise ValueError(f'unknown tag_method: {method!r}')


def _probe_rate_for_subject(
    trials_df: pd.DataFrame,
    sessions_df: pd.DataFrame,
    subject_id: int,
    cfg: dict,
) -> tuple[float, int, int, int]:
    """Aggregated probe rate for one subject × one probe label.

    Returns (rate, n_correct, n_possible, n_sessions). Rate is NaN if
    the subject ran no sessions of this probe type.
    """
    subj_sessions = sessions_df[
        (sessions_df['subject_id'] == subject_id) &
        (sessions_df['session_type'].isin(cfg['session_types']))
    ]
    n_sessions = len(subj_sessions)
    if n_sessions == 0:
        return float('nan'), 0, 0, 0

    n_correct = 0
    for _, sess_row in subj_sessions.iterrows():
        st_trials = trials_df[trials_df['session_id'] == sess_row['session_id']]
        if st_trials.empty:
            continue
        st_trials = st_trials.sort_values('trial_number')
        tag_mask = _tag_trials(st_trials, cfg['tag_method'])
        relevant = st_trials[tag_mask]
        # `errors == 0` means a correct trial (legacy convention).
        n_correct += int((relevant['errors'] == 0).sum())

    n_possible = n_sessions * cfg['possible_per_session']
    rate = n_correct / n_possible if n_possible > 0 else float('nan')
    return rate, n_correct, n_possible, n_sessions


# ── Main extraction ───────────────────────────────────────────

def extract() -> pd.DataFrame:
    print(f'reading parquets from {ANALYSIS_DATA}')
    subjects = pd.read_parquet(ANALYSIS_DATA / 'subjects.parquet')
    sessions = pd.read_parquet(ANALYSIS_DATA / 'sessions.parquet')
    trials   = pd.read_parquet(ANALYSIS_DATA / 'trials.parquet')

    # Manuscript-scope filter.
    in_scope = subjects[
        subjects['training_group'].isin(IN_SCOPE_GROUPS) &
        ~subjects['name'].isin(EXCLUDE_NAMES)
    ].copy()
    print(f'  raw subjects: {len(subjects)}')
    print(f'  in-scope (after group + exclusion filters): {len(in_scope)}')
    print(f'  by group:\n{in_scope["training_group"].value_counts().to_string()}')

    # n_sessions_to_acquire: count Acquisition-phase sessions per subject.
    acq_sessions = sessions[sessions['session_experiment_phase'] == 'Acquisition']
    n_acq = (
        acq_sessions.groupby('subject_id').size()
        .reindex(in_scope['subject_id'], fill_value=0)
        .rename('n_sessions_to_acquire')
    )

    # n_steps_to_acquire: sum yoked-dataset `n_actions` across each
    # subject's acquisition sessions. `n_actions` matches the real-
    # pretrial action stream row count exactly (verified 2026-05-20).
    print(f'reading yoked sessions from {YOKED_DATA}')
    ysess = pd.read_parquet(YOKED_DATA / 'sessions.parquet')
    yacq  = ysess[ysess['session_phase'] == 'Acquisition']
    yoked_agg = (
        yacq.groupby('subject_id')
        .agg(n_steps_to_acquire=('n_actions', 'sum'),
             n_yoked_acq_sessions=('n_actions', 'size'))
    )

    # Compute per-(subject, probe) rates.
    rows = []
    for _, subj in in_scope.iterrows():
        sid = int(subj['subject_id'])
        # Yoked step total — NaN if subject has no yoked acquisition data.
        if sid in yoked_agg.index:
            n_steps_acq         = int(yoked_agg.loc[sid, 'n_steps_to_acquire'])
            n_yoked_acq_sessions = int(yoked_agg.loc[sid, 'n_yoked_acq_sessions'])
        else:
            n_steps_acq         = None  # → NaN in output
            n_yoked_acq_sessions = 0

        row: dict = {
            'subject_id':            sid,
            'name':                  subj['name'],
            'training_group':        subj['training_group'],
            'sex':                   subj['sex'],
            'n_sessions_to_acquire': int(n_acq.loc[sid]),
            'n_steps_to_acquire':    n_steps_acq,
            'n_yoked_acq_sessions':  n_yoked_acq_sessions,
        }

        # acquired = rat has at least one probe-phase session (proxy for
        # "reached criterion and was advanced to probes per protocol")
        probe_phase_count = int(((sessions['subject_id'] == sid) &
                                 (sessions['session_experiment_phase'].isin(
                                     ['Novel Route', 'Reversal', 'Rotation', 'No Cue']
                                 ))).sum())
        row['acquired']        = probe_phase_count > 0
        row['n_probe_sessions'] = probe_phase_count

        for label, cfg in PROBE_CONFIG.items():
            rate, n_correct, n_possible, n_sessions = _probe_rate_for_subject(
                trials, sessions, sid, cfg,
            )
            label_key = label.lower().replace(' ', '_')
            row[f'rate_{label_key}']        = rate
            row[f'n_correct_{label_key}']   = int(n_correct)
            row[f'n_possible_{label_key}']  = int(n_possible)
            row[f'n_sessions_{label_key}']  = int(n_sessions)

        rows.append(row)

    df = pd.DataFrame(rows).sort_values(['training_group', 'name']).reset_index(drop=True)
    return df


def main() -> None:
    df = extract()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / 'per_subject.parquet'
    df.to_parquet(out_path, engine='pyarrow', index=False)
    print(f'\nwrote {len(df)} rows to {out_path}')
    print(f'columns: {list(df.columns)}')
    print('\n--- counts ---')
    print(df['training_group'].value_counts(dropna=False).to_string())
    print(f'\nacquired: {int(df["acquired"].sum())}/{len(df)}')
    print('\n--- per-group probe-rate means (NaN where probe not in protocol) ---')
    rate_cols = [c for c in df.columns if c.startswith('rate_')]
    print(df.groupby('training_group')[rate_cols].mean().to_string())
    print('\n--- per-group sessions_to_acquire ---')
    print(df.groupby('training_group')['n_sessions_to_acquire'].agg(['mean', 'median', 'min', 'max']).to_string())
    print('\n--- per-group steps_to_acquire ---')
    print(df.groupby('training_group')['n_steps_to_acquire'].agg(['mean', 'median', 'min', 'max']).to_string())

    # Yoking-gap audit: flag subjects whose yoked acq sessions ≠ behavioral.
    gaps = df[df['n_yoked_acq_sessions'] != df['n_sessions_to_acquire']]
    if len(gaps):
        print(f'\nWARNING: {len(gaps)} rats have a yoking gap (n_yoked_acq_sessions != n_sessions_to_acquire):')
        print(gaps[['name', 'training_group', 'n_sessions_to_acquire',
                    'n_yoked_acq_sessions', 'n_steps_to_acquire']].to_string(index=False))
    else:
        print('\nyoking coverage: complete (all 48 rats have all acquisition sessions yoked)')


if __name__ == '__main__':
    main()
