"""DuckDB-backed loaders for the 5-table yoked dataset.

The dataset lives at ``data/yoked/dataset/`` (gitignored; populated by
``corner-maze-build-dataset`` against the upstream behavioral repo):

  * ``subjects.parquet``                       — one row per rat
  * ``sessions.parquet``                       — one row per session (all phases)
  * ``actions_synthetic_pretrial.parquet``     — Acquisition only, synthetic pretrial
  * ``actions_real_pretrial.parquet``          — Acquisition only, real pretrial
  * ``actions_exposure.parquet``               — Exposure only (no pretrial concept)

``YokedPaths.from_dir(actions_variant=...)`` picks the action table by suffix:
``"synthetic_pretrial"``, ``"real_pretrial"``, or ``"exposure"``.

This module wraps lightweight queries used by the build-returns pipeline
and the runner. All paths are configurable so the same code runs locally
or under Colab.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    import duckdb
    _HAS_DUCKDB = True
except ImportError:  # pragma: no cover
    duckdb = None  # type: ignore[assignment]
    _HAS_DUCKDB = False


DEFAULT_DATASET_DIR = Path("data/yoked/dataset")


@dataclass(frozen=True)
class YokedPaths:
    """File paths for one yoked-dataset directory."""
    subjects: Path
    sessions: Path
    actions: Path

    @classmethod
    def from_dir(
        cls,
        dataset_dir: str | Path = DEFAULT_DATASET_DIR,
        actions_variant: str = "synthetic_pretrial",
    ) -> "YokedPaths":
        d = Path(dataset_dir)
        return cls(
            subjects=d / "subjects.parquet",
            sessions=d / "sessions.parquet",
            actions=d / f"actions_{actions_variant}.parquet",
        )

    def all(self) -> tuple[Path, ...]:
        return (self.subjects, self.sessions, self.actions)

    def assert_exist(self) -> None:
        missing = [str(p) for p in self.all() if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing yoked dataset files:\n  " + "\n  ".join(missing)
                + "\nRun corner-maze-build-dataset to populate them from the upstream repo."
            )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_subjects(paths: YokedPaths) -> pd.DataFrame:
    paths.assert_exist()
    return pd.read_parquet(paths.subjects)


def load_sessions(paths: YokedPaths) -> pd.DataFrame:
    paths.assert_exist()
    return pd.read_parquet(paths.sessions)


def load_actions_for_session(paths: YokedPaths, session_id: int) -> pd.DataFrame:
    """Return all action rows for a single session, ordered by step."""
    paths.assert_exist()
    if _HAS_DUCKDB:
        sql = (
            f"SELECT * FROM read_parquet('{paths.actions}') "
            f"WHERE session_id = {int(session_id)} ORDER BY step"
        )
        return duckdb.query(sql).to_df()
    df = pd.read_parquet(paths.actions)
    return df[df["session_id"] == session_id].sort_values("step").reset_index(drop=True)


def iter_session_actions(
    paths: YokedPaths,
    session_ids: Iterable[int],
) -> Iterable[tuple[int, pd.DataFrame]]:
    """Yield ``(session_id, action_df)`` pairs in the order requested."""
    for sid in session_ids:
        yield sid, load_actions_for_session(paths, sid)


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------

def get_subject_row(subjects: pd.DataFrame, subject_name: str) -> pd.Series:
    """Return the subjects.parquet row for *subject_name* (e.g. 'CM005')."""
    matches = subjects[subjects["subject_name"] == subject_name]
    if len(matches) == 0:
        raise ValueError(f"subject {subject_name!r} not found in subjects.parquet")
    if len(matches) > 1:
        raise ValueError(f"subject {subject_name!r} matches multiple rows ({len(matches)})")
    return matches.iloc[0]


def get_sessions_for_subject(
    sessions: pd.DataFrame,
    subject_id: int,
) -> pd.DataFrame:
    """All sessions for a subject, ordered by session_number."""
    out = sessions[sessions["subject_id"] == subject_id]
    return out.sort_values("session_number").reset_index(drop=True)
