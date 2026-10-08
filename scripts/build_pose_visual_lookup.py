"""Build data/lookups/pose_visual.npz from an encoder-pretrain embeddings table.

Input: the ``*-embeddings-allposes.parquet`` written by corner-maze-encoder-pretrain
(``train_classifier_dual_stream_uncoupled.py`` followed by ``add_all_poses_to_table.py``).
One row per consolidated view: ``label_name``, ``embedding`` (60 floats),
``poses`` (every pose label this view serves), ``left_eye_img`` / ``right_eye_img``
(128x128 floats in [0, 1]).

Output keys (consumed by ``CornerMazeEnv._load_embeddings`` and the notebooks):

    pose_labels  (P,)   <U20   every pose label, in table row order
    pose_to_idx  (P,)   int32  row index for each pose label
    label_names  (N,)   <U20   one name per table row
    embeddings   (N,60) float32
    left_eyes    (N,128,128) uint8   round(img * 255)
    right_eyes   (N,128,128) uint8

Usage:
    python scripts/build_pose_visual_lookup.py --table PATH/TO/x-embeddings-allposes.parquet \
        [--out data/lookups/pose_visual.npz] [--check-against data/lookups/pose_visual.npz]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "lookups" / "pose_visual.npz"


def _image_stack(col: pd.Series) -> np.ndarray:
    imgs = np.stack([np.asarray(np.asarray(v).tolist(), dtype=np.float64) for v in col])
    if imgs.max() > 1.0 + 1e-6:
        raise ValueError("expected images in [0, 1]")
    return np.round(imgs * 255.0).astype(np.uint8)


def build(table: Path) -> dict[str, np.ndarray]:
    df = pd.read_parquet(table)
    label_names = np.asarray([str(s) for s in df["label_name"]], dtype="<U20")
    pose_labels, pose_to_idx = [], []
    for i, poses in enumerate(df["poses"]):
        for p in list(poses):
            pose_labels.append(str(p))
            pose_to_idx.append(i)
    if len(set(pose_labels)) != len(pose_labels):
        raise ValueError("duplicate pose labels across rows")
    return {
        "pose_labels": np.asarray(pose_labels, dtype="<U20"),
        "pose_to_idx": np.asarray(pose_to_idx, dtype=np.int32),
        "label_names": label_names,
        "embeddings": np.stack([np.asarray(e, dtype=np.float32) for e in df["embedding"]]),
        "left_eyes": _image_stack(df["left_eye_img"]),
        "right_eyes": _image_stack(df["right_eye_img"]),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--check-against", type=Path, default=None, help="existing npz to compare with instead of writing")
    a = ap.parse_args(argv)
    arrays = build(a.table)
    if a.check_against is not None:
        ref = np.load(a.check_against, allow_pickle=False)
        ok = True
        for k in ref.files:
            same = k in arrays and arrays[k].dtype == ref[k].dtype and arrays[k].shape == ref[k].shape and np.array_equal(arrays[k], ref[k])
            print(f"  {k:12s} {'identical' if same else 'DIFFERENT'}  {arrays.get(k, np.empty(0)).shape} {arrays.get(k, np.empty(0)).dtype}")
            ok &= bool(same)
        print("ALL IDENTICAL" if ok else "MISMATCH")
        return
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **arrays)
    print(f"wrote {a.out}: {len(arrays['label_names'])} rows, {len(arrays['pose_labels'])} pose labels")


if __name__ == "__main__":
    main()
