#!/usr/bin/env python3
"""Build a MiniGrid egocentric-view lookup table for all CornerMazeEnv poses.

Output: data/lookups/minigrid_views.npz — two arrays:
  pose_labels: (N,) U   — pose_label strings
  views:       (N, 21, 21, 3) uint8 — what env.get_pov_render_mod(VIEW_TILE_SIZE)
                                       produces at runtime.

Pose-label set is sourced from data/lookups/pose_visual.npz so this file lines
up 1:1 with the embedding / eye-image lookups already used by the env.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from corner_maze_rl.env.constants import (
    MINIGRID_VIEWS_NPZ_PATH, POSE_VISUAL_NPZ_PATH, VIEW_TILE_SIZE,
)
from corner_maze_rl.env.corner_maze_env import CornerMazeEnv


def parse_pose_label(label: str) -> tuple[str, int, int, int]:
    """Split a pose_label like 'trl_e_n_xx_11_11_1' into (class, x, y, dir)."""
    head, x_s, y_s, d_s = label.rsplit("_", 3)
    return head, int(x_s), int(y_s), int(d_s)


def build_class_to_config(env: CornerMazeEnv) -> dict[str, tuple[int, ...]]:
    """Map every pose-label class to a representative 37-tuple from env.layouts.

    Mirrors the collapsing rules in env._get_pose_label:
      - expa_*           -> all collapse to class 'expa_x_x_xx'
      - trl_a_c_<goal>   -> class 'trl_a_c_xx' (view invariant to goal)
      - pre_a_c_xx       -> class 'pre_a_c_xx' (exact)
      - iti_*, expb_*    -> class equals layout name (exact)
      - 'x_x_xx'         -> fallback class 'x_x_x_xx'
    """
    resolver: dict[str, tuple[int, ...]] = {}
    for name, cfg in env.layouts.items():
        if name == "x_x_xx":
            resolver.setdefault("x_x_x_xx", cfg)
            continue
        parts = name.split("_")
        phase = parts[0]
        if phase == "expa":
            resolver.setdefault("expa_x_x_xx", cfg)
        elif phase in ("trl", "pre"):
            resolver.setdefault(f"{phase}_{parts[1]}_{parts[2]}_xx", cfg)
        else:
            resolver[name] = cfg
    return resolver


def main() -> int:
    src_path = REPO_ROOT / POSE_VISUAL_NPZ_PATH
    out_path = REPO_ROOT / MINIGRID_VIEWS_NPZ_PATH

    print(f"Reading {src_path.relative_to(REPO_ROOT)}")
    with np.load(src_path, allow_pickle=False) as z:
        pose_label_arr = z["pose_labels"]
        label_names = z["label_names"]
    all_poses: set[str] = {str(p) for p in pose_label_arr}
    print(f"  -> {len(all_poses)} unique pose labels "
          f"across {len(label_names)} label_name rows")

    print("Initializing CornerMazeEnv (obs_mode='view')")
    env = CornerMazeEnv(
        render_mode="rgb_array",
        session_type="PI+VC f2 acquisition",
        agent_cue_goal_orientation="N/NE",
        start_goal_location="NE",
        obs_mode="view",
    )
    env.reset()

    resolver = build_class_to_config(env)
    print(f"  -> resolved {len(resolver)} layout classes from env.layouts "
          f"({len(env.layouts)} raw layouts)")

    by_class: dict[str, list[tuple[str, int, int, int]]] = {}
    unresolved: list[str] = []
    for label in sorted(all_poses):
        cls, x, y, d = parse_pose_label(label)
        if cls not in resolver:
            unresolved.append(label)
            continue
        by_class.setdefault(cls, []).append((label, x, y, d))

    if unresolved:
        print(f"WARNING: {len(unresolved)} pose labels have unresolved layout class")
        for u in unresolved[:5]:
            print(f"  {u}")

    print(f"Rendering {sum(len(v) for v in by_class.values())} views "
          f"across {len(by_class)} layout classes")
    records = []
    for cls, items in by_class.items():
        env.update_grid_configuration(resolver[cls])
        for label, x, y, d in items:
            env.agent_pos = (x, y)
            env.agent_dir = d
            view = env.get_pov_render_mod(VIEW_TILE_SIZE)
            if view.shape != (21, 21, 3) or view.dtype != np.uint8:
                raise RuntimeError(
                    f"Unexpected view shape/dtype for {label}: "
                    f"{view.shape} {view.dtype}"
                )
            records.append((label, view))

    # Extra pass: full 13×13×4 coverage for the expb_x_x_xx layout, which
    # under the corrected naming is the Phase B end-state (all barriers
    # dropped, fully open, ≡ expa). During Phase B the agent walks anywhere
    # on the maze, but the source pose-visual lookup has zero expb_x_x_xx
    # entries (it dedupes the open state into expa_x_x_xx instead). Without
    # this pass the agent would hit _zero_view for every step of Phase B.
    seen = {label for label, _ in records}
    expb_cfg = env.layouts["expb_x_x_xx"]
    env.update_grid_configuration(expb_cfg)
    expb_added = 0
    for x in range(env.width):
        for y in range(env.height):
            for d in range(4):
                label = f"expb_x_x_xx_{x}_{y}_{d}"
                if label in seen:
                    continue
                env.agent_pos = (x, y)
                env.agent_dir = d
                view = env.get_pov_render_mod(VIEW_TILE_SIZE)
                records.append((label, view))
                expb_added += 1
    print(f"  -> appended {expb_added} expb_x_x_xx poses for full coverage")

    # Sort by pose_label so the on-disk order is deterministic across rebuilds.
    records.sort(key=lambda r: r[0])
    pose_labels = np.asarray([r[0] for r in records], dtype=np.str_)
    views = np.stack([r[1] for r in records])  # (N, 21, 21, 3) uint8

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, pose_labels=pose_labels, views=views)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {out_path.relative_to(REPO_ROOT)}: "
          f"{len(pose_labels)} rows, {size_mb:.2f} MB")

    written = set(map(str, pose_labels))
    missing = all_poses - written
    extra_non_expb = (written - all_poses) - {
        f"expb_x_x_xx_{x}_{y}_{d}"
        for x in range(env.width) for y in range(env.height) for d in range(4)
    }
    if missing:
        print(f"WARNING: {len(missing)} source pose_labels missing from output")
    if extra_non_expb:
        print(f"WARNING: {len(extra_non_expb)} unexpected extra pose_labels in output")
    if not missing and not extra_non_expb:
        print("pose_label set covers source npz + full expb_x_x_xx pass")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
