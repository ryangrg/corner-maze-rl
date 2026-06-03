"""Model implementations for corner-maze RL.

``MaskableRecurrentPPO`` is re-exported for convenience but importing it pulls in
sb3-contrib, so it is loaded lazily to keep ``import corner_maze_rl.models`` cheap
for the pure-PyTorch agents (PPO / SR).
"""
from __future__ import annotations

__all__ = ["MaskableRecurrentPPO", "MaskableRecurrentActorCriticCnnPolicy"]


def __getattr__(name: str):
    if name in __all__:
        from corner_maze_rl.models import maskable_recurrent_ppo as _m

        return getattr(_m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
