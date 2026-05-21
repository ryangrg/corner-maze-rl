"""Env-wrapper helpers for SB3 training paths.

Ported from legacy ``src/rl/sb3_agents.py``. Provides ``mask_fn`` for
``sb3_contrib.common.wrappers.ActionMasker`` — wraps the env so that
``MaskablePPO`` (and other masking-aware SB3 algos) can read the per-step
action mask from ``CornerMazeEnv.get_action_mask()``.

PAUSE (action 4) is always masked off during training — legacy parity
with the ``ENABLE_PAUSE = False`` default in ``custom_rl.py``.
"""
from __future__ import annotations


_ENABLE_PAUSE: bool = False


def mask_fn(env):
    """Return the action validity mask for the current env state.

    ``env`` may be an arbitrarily wrapped env; we walk through ``.unwrapped``
    to reach the underlying ``CornerMazeEnv`` and call its
    ``get_action_mask()`` method. PAUSE (index 4) is forced to the
    package-wide ``_ENABLE_PAUSE`` setting regardless of what the env
    reports — legacy behaviour.
    """
    mask = env.unwrapped.get_action_mask()
    mask[4] = _ENABLE_PAUSE
    return mask
