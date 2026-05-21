#!/usr/bin/env python3
"""Build a MiniGrid egocentric-view lookup table for all CornerMazeEnv poses.

Output: data/dataframes/minigrid-views-allposes.parquet — one row per pose
label, with the raw (21, 21, 3) uint8 view bytes that
``env.get_pov_render_mod(VIEW_TILE_SIZE)`` would produce at runtime.

Pose-label set is sourced from the existing
``dual-indep-20260319-222411-embeddings-allposes.parquet`` so this file lines
up 1:1 with the embedding / eye-image lookups already used by the env.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from corner_maze_rl.env.constants import EMBEDDING_PARQUET_PATH, VIEW_TILE_SIZE
from corner_maze_rl.env.corner_maze_env import CornerMazeEnv


OUTPUT_REL = "data/dataframes/minigrid-views-allposes.parquet"


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
    emb_path = REPO_ROOT / EMBEDDING_PARQUET_PATH
    out_path = REPO_ROOT / OUTPUT_REL

    print(f"Reading {emb_path.relative_to(REPO_ROOT)}")
    emb_df = pd.read_parquet(emb_path)

    all_poses: set[str] = set()
    for row in emb_df.itertuples():
        all_poses.update(str(p) for p in row.poses)
    print(f"  -> {len(all_poses)} unique pose labels "
          f"across {len(emb_df)} label_name rows")

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
            records.append({
                "pose_label": label,
                "layout_class": cls,
                "x": x,
                "y": y,
                "dir": d,
                "view": view.tobytes(),
            })

    # Extra pass: full 13×13×4 coverage for the expb_x_x_xx (acclimation)
    # layout. The source embeddings parquet contains zero expb_x_x_xx poses
    # (sampling missed the brief 60-step acclimation window), so the agent
    # would otherwise hit _zero_view at the start of every expb session.
    seen = {r["pose_label"] for r in records}
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
                records.append({
                    "pose_label": label,
                    "layout_class": "expb_x_x_xx",
                    "x": x,
                    "y": y,
                    "dir": d,
                    "view": view.tobytes(),
                })
                expb_added += 1
    print(f"  -> appended {expb_added} expb_x_x_xx poses for full coverage")

    df = pd.DataFrame.from_records(records).astype(
        {"x": "int16", "y": "int16", "dir": "int8"}
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, compression="zstd", compression_level=9, index=False)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {out_path.relative_to(REPO_ROOT)}: "
          f"{len(df)} rows, {size_mb:.2f} MB")
    if size_mb > 25:
        print("WARNING: file >25 MB; consider PNG-compressing the view column.")

    written = set(df["pose_label"])
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
        print("pose_label set covers source parquet + full expb_x_x_xx pass")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
