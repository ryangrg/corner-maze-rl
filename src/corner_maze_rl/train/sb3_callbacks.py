"""SB3 callbacks for ``CornerMazeEnv`` training runs.

Ported from legacy ``src/rl/sb3_agents.py``. Six callbacks:

  * ``CriterionStopCallback`` — stops ``model.learn(...)`` early when the
    env signals ``training_criterion_met`` (a per-session learning-done
    flag set by the env after enough consecutive perfect trials).

  * ``TrainingProgressCallback`` — logs ``maze/{episode,score,trials,
    reward}`` to the SB3 logger so training-table rows reflect actual
    task performance, not just policy gradients. Reads from
    ``self.training_env`` (the SB3-wrapped env) and unwraps to reach
    the underlying ``CornerMazeEnv`` for env-internal attributes.

  * ``BestModelCallback`` — saves the model whenever the most-recent
    completed episode score beats the running best. Single ``.zip`` is
    overwritten in place so the post-run dir has a clean
    ``model_best.zip`` alongside ``model_post_acquisition.zip``.

  * ``ActionStatsCallback`` — accumulates per-step action counts and
    mask hit-rates per (session_type, episode). Call ``to_parquet()`` /
    ``to_records()`` after training to dump.

  * ``MaxAcquisitionEpisodesCallback`` — enforces a hard cap on the
    number of completed episodes during a single ``learn()`` call. Use
    on acquisition phases to match the rat-protocol convention of "N
    sessions max to reach criterion." Sets ``env._kill_reason =
    'hit_session_cap'`` when it fires.

  * ``FlatLearningKillCallback`` — kills a *single acquisition* learn()
    call early when per-episode scores plateau at a low absolute mean.
    Mirrors the kill_switch.py flat-kill logic but at episode granularity
    rather than session-type granularity. Sets ``env._kill_reason =
    'flat_learning'`` when it fires.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stable_baselines3.common.callbacks import BaseCallback

from corner_maze_rl.train.kill_switch import (
    ABSOLUTE_FLOOR as _DEFAULT_FLAT_FLOOR,
    FLAT_SLOPE_EPS as _DEFAULT_FLAT_SLOPE_EPS,
    linear_regression_slope,
)


def _unwrap_env(env):
    """Walk through SB3 wrappers down to the raw CornerMazeEnv."""
    while hasattr(env, "env"):
        env = env.env
    return env


class CriterionStopCallback(BaseCallback):
    """Stop ``model.learn()`` when ``env.training_criterion_met`` fires.

    The env emits ``info['training_criterion_met']`` in the same dict that
    SB3 passes to the callback via ``self.locals['infos']``. We check
    the first env (single-env training) and return ``False`` to break the
    learn loop early.
    """

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if infos and infos[0].get("training_criterion_met", False):
            if self.verbose:
                print("Criterion met — stopping session.")
            return False
        return True


class TrainingProgressCallback(BaseCallback):
    """Log per-episode task stats to SB3's rollout table.

    Records ``maze/episode``, ``maze/score`` (perfect trials, int),
    ``maze/trials``, and ``maze/reward`` on every step (cached between
    episode boundaries) so they show up in every rollout dump. Episode
    boundaries are detected via ``self.locals['dones'][0]`` — SB3 auto-resets
    before calling ``_on_step``, so the post-reset env state is what we read.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._last_trial_count = 0
        self._maze_stats = {
            "maze/episode": 0,
            "maze/score": 0,
            "maze/trials": 0,
            "maze/reward": 0,
        }

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])

        env = _unwrap_env(self.training_env.envs[0])

        trial_count = getattr(env, "trial_count", 0) or 0
        if trial_count > self._last_trial_count:
            self._last_trial_count = trial_count

        if len(dones) > 0 and dones[0]:
            infos = self.locals.get("infos", [{}])
            info = infos[0]
            episode = getattr(env, "episode", 0)
            ep_scores = info.get("episode_scores", [])
            latest_score = ep_scores[-1] if ep_scores else 0
            ep_reward = info.get("session_reward", 0)
            ep_trials = info.get(
                "trial_count", self._last_trial_count or trial_count
            )
            ep_correct = int(round(latest_score * ep_trials))
            self._maze_stats = {
                "maze/episode": episode,
                "maze/score": ep_correct,
                "maze/trials": ep_trials,
                "maze/reward": ep_reward,
            }
            if self.verbose:
                print(
                    f"  Episode {episode} done  score={ep_correct}/{ep_trials}  "
                    f"reward={ep_reward:.3f}"
                )
            self._last_trial_count = 0

        for key, val in self._maze_stats.items():
            self.logger.record(key, val)

        return True


class BestModelCallback(BaseCallback):
    """Save the model whenever an episode beats the running best score.

    Score = ``perfect_trial_count`` for the just-finished episode (same
    metric ``TrainingProgressCallback`` reports as ``maze/score``). Writes
    to ``save_path.zip`` (single file, overwritten in place). Also logs
    ``maze/best_score`` and ``maze/best_at_timestep`` to the SB3 logger.

    Survives across multiple ``learn()`` calls in the same process — the
    caller controls reset by constructing a new instance per seed.
    """

    def __init__(self, save_path: str | Path, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.save_path = str(save_path)
        self.best_score: int = -1
        self.best_at_timestep: int = 0
        self.n_saves: int = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        if not (len(dones) > 0 and dones[0]):
            self.logger.record("maze/best_score", self.best_score)
            self.logger.record("maze/best_at_timestep", self.best_at_timestep)
            return True

        env = _unwrap_env(self.training_env.envs[0])
        info = (self.locals.get("infos") or [{}])[0]
        ep_scores = info.get("episode_scores", []) or []
        latest_score = ep_scores[-1] if ep_scores else 0
        ep_trials = info.get("trial_count", getattr(env, "trial_count", 0) or 0)
        score = int(round(latest_score * ep_trials))

        if score > self.best_score:
            self.best_score = score
            self.best_at_timestep = self.num_timesteps
            self.model.save(self.save_path)
            self.n_saves += 1
            if self.verbose:
                print(
                    f"  [BestModel] new best score={score} at t={self.num_timesteps} "
                    f"(saves={self.n_saves})"
                )

        self.logger.record("maze/best_score", self.best_score)
        self.logger.record("maze/best_at_timestep", self.best_at_timestep)
        return True


class ActionStatsCallback(BaseCallback):
    """Accumulate per-step action counts + mask hit-rates per episode.

    Records every step (action chosen, full action_mask if present in
    `infos[0]['action_mask']`, current session_type, episode). On episode
    end, flushes one summary row per episode into an internal buffer.

    Use ``to_records()`` / ``to_parquet()`` after training to dump.
    """

    def __init__(self, n_actions: int = 5, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.n_actions = int(n_actions)
        self._reset_episode_accumulator()
        self._records: list[dict] = []

    def _reset_episode_accumulator(self) -> None:
        self._action_counts = np.zeros(self.n_actions, dtype=np.int64)
        self._mask_avail_counts = np.zeros(self.n_actions, dtype=np.int64)
        self._n_steps_in_ep = 0

    def _on_step(self) -> bool:
        actions = self.locals.get("actions", None)
        if actions is not None and len(actions) > 0:
            a = int(actions[0])
            if 0 <= a < self.n_actions:
                self._action_counts[a] += 1

        infos = self.locals.get("infos", [{}])
        info = infos[0] if infos else {}
        mask = info.get("action_mask")
        if mask is not None:
            m = np.asarray(mask, dtype=bool)
            if m.shape[0] == self.n_actions:
                self._mask_avail_counts += m.astype(np.int64)
        self._n_steps_in_ep += 1

        dones = self.locals.get("dones", [])
        if len(dones) > 0 and dones[0]:
            env = _unwrap_env(self.training_env.envs[0])
            session_type = getattr(env, "session_type", None)
            episode = getattr(env, "episode", 0)
            total = max(1, self._n_steps_in_ep)
            rec: dict = {
                "session_type": session_type,
                "episode": int(episode),
                "n_steps": int(self._n_steps_in_ep),
                "timestep": int(self.num_timesteps),
            }
            for a in range(self.n_actions):
                rec[f"action_{a}_count"] = int(self._action_counts[a])
                rec[f"action_{a}_freq"] = float(self._action_counts[a] / total)
                rec[f"action_{a}_avail_freq"] = float(self._mask_avail_counts[a] / total)
            self._records.append(rec)
            self._reset_episode_accumulator()

        return True

    def to_records(self) -> list[dict]:
        """Return all accumulated per-episode records (one row each)."""
        return list(self._records)

    def to_parquet(self, path: str | Path) -> None:
        """Write accumulated records to a parquet file (no-op if empty)."""
        if not self._records:
            return
        pd.DataFrame(self._records).to_parquet(str(path), engine="pyarrow")


class MaxAcquisitionEpisodesCallback(BaseCallback):
    """Stop ``model.learn()`` after ``max_episodes`` completed episodes.

    Designed to enforce rat-protocol session caps on acquisition phases
    (e.g., "max 80 sessions to reach criterion"). Counts episode-end
    transitions via ``self.locals['dones'][0]`` and returns ``False`` on
    the Nth one to break SB3's rollout loop.

    Counts episodes within ONE ``learn()`` call only — the counter does
    not persist across multiple ``learn()`` calls. Add this callback
    only to acquisition-phase ``CallbackList``s so probe phases run
    unaffected. Logs ``maze/n_episodes`` and ``maze/episode_cap_hit``
    to the SB3 logger for post-hoc inspection.
    """

    def __init__(self, max_episodes: int, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.max_episodes: int = int(max_episodes)
        self._episode_count: int = 0
        self._cap_hit: bool = False

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        if len(dones) > 0 and dones[0]:
            self._episode_count += 1
            if self._episode_count >= self.max_episodes:
                self._cap_hit = True
                env = _unwrap_env(self.training_env.envs[0])
                setattr(env, "_kill_reason", "hit_session_cap")
                if self.verbose:
                    print(
                        f"  [MaxAcquisitionEpisodes] cap={self.max_episodes} "
                        f"reached, stopping learn()."
                    )
                self.logger.record("maze/n_episodes", self._episode_count)
                self.logger.record("maze/episode_cap_hit", int(self._cap_hit))
                return False
        self.logger.record("maze/n_episodes", self._episode_count)
        self.logger.record("maze/episode_cap_hit", int(self._cap_hit))
        return True


class FlatLearningKillCallback(BaseCallback):
    """Stop a single acquisition ``learn()`` call when learning has gone flat.

    Tracks per-episode ``perfect_trial_count`` (extracted from
    ``info['episode_scores'][-1] * info['trial_count']``, same convention as
    ``TrainingProgressCallback`` / ``BestModelCallback``). After ``warmup``
    completed episodes, evaluates after every episode-end: if the OLS slope
    over the trailing ``window`` is below ``flat_slope_eps`` *and* the recent
    mean is below ``absolute_floor``, returns ``False`` to halt training.

    Asymmetric ``<`` on slope is intentional — declining curves get killed
    too; only creeping-up curves (slope > eps) survive at a low floor.

    Sets ``env._kill_reason = 'flat_learning'`` so the runner can record a
    specific cause in ``killed_at.json`` rather than the generic
    "did_not_acquire". Set ``enabled=False`` (the notebook's KILL_FLAT
    toggle) to make this a no-op without removing it from the CallbackList.

    Logs ``maze/flat_slope`` and ``maze/flat_recent_mean`` to the SB3
    logger every episode-end (after warmup) so flat-learning trajectories
    are visible in ``progress.csv`` even when the kill doesn't fire.
    """

    def __init__(
        self,
        warmup: int,
        window: int = 10,
        flat_slope_eps: float = _DEFAULT_FLAT_SLOPE_EPS,
        absolute_floor: float = _DEFAULT_FLAT_FLOOR,
        enabled: bool = True,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.warmup           = int(warmup)
        self.window           = int(window)
        self.flat_slope_eps   = float(flat_slope_eps)
        self.absolute_floor   = float(absolute_floor)
        self.enabled          = bool(enabled)
        self._episode_scores: list[float] = []
        self._flat_killed: bool = False

    def _on_step(self) -> bool:
        if not self.enabled:
            return True

        dones = self.locals.get("dones", [])
        if not (len(dones) > 0 and dones[0]):
            return True

        env = _unwrap_env(self.training_env.envs[0])
        info = (self.locals.get("infos") or [{}])[0]
        ep_scores = info.get("episode_scores", []) or []
        latest_score = ep_scores[-1] if ep_scores else 0
        ep_trials = info.get("trial_count", getattr(env, "trial_count", 0) or 0)
        score = float(round(latest_score * ep_trials))
        self._episode_scores.append(score)

        n = len(self._episode_scores)
        if n < self.warmup or n < self.window:
            return True

        window_vals = self._episode_scores[-self.window:]
        slope = linear_regression_slope(window_vals)
        mean  = sum(window_vals) / len(window_vals)
        self.logger.record("maze/flat_slope", float(slope))
        self.logger.record("maze/flat_recent_mean", float(mean))

        if slope < self.flat_slope_eps and mean < self.absolute_floor:
            self._flat_killed = True
            setattr(env, "_kill_reason", "flat_learning")
            if self.verbose:
                print(
                    f"  [FlatLearningKill] n={n}, slope={slope:.3f} < "
                    f"{self.flat_slope_eps}, mean={mean:.2f} < "
                    f"{self.absolute_floor} — stopping learn()."
                )
            return False
        return True
