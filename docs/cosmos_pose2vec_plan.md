# Implementation Plan — `notebooks/cosmos_pose2vec.ipynb`

Companion to `docs/cosmos_pose2vec_kickoff.md`. That document is the brief; this one is the
build.

---

## 0. Open questions — decided

The kickoff left six questions open. All are resolved here so the notebook can commit rather
than hedge. Each is a default, not a lock — change the config knob and re-run.

| # | Question | Decision | Rationale |
|---|---|---|---|
| 1 | Does "exposure 1a" mean Exposure A (fully-open)? | **Yes** — 49 cells, 196 poses | Verified twice: `CornerMazeEnv(session_type='exposure')` yields exactly 49 walkable cells, and exposure data occupies exactly those 49. Exposure B's end state is "fully open, ≡ expa". |
| 2 | causal vs acausal | **causal = future-only window; acausal = symmetric.** `CAUSAL = True` default | Matches the brief's own framing, "predict the next position on the grid". The brief's parenthetical for acausal was a typo. |
| 3 | Notebook name / number | **`notebooks/cosmos_pose2vec.ipynb`**, unnumbered | Named literally in the brief. The numbered series tracks the DT storyline; this is a side utility that feeds it. Trivial to rename. |
| 4 | Pooled or per-subject default | **Pooled over all 17 PI+VC subjects** | The embedding describes *the maze*, not an animal. Pooling gives ~222k steps and full 196-pose coverage; CM006/CM008 have only 3 sessions each and would leave rare poses thin. Per-subject remains available via `POOL_SUBJECTS=False`. |
| 5 | Acquisition only, or all session types | **Acquisition only** (`CORPUS_SOURCE='real_pretrial'`) | Acquisition already covers all 196 poses, so exposure adds volume but no new vocabulary. Keeps the corpus one behavioural regime. `'both'` and `'synthetic_pretrial'` are wired but not default. |
| 6 | Edit notebook 10 in this change | **No** — document the touch-points only | Notebook 10 is the DT regression baseline. Adding a `pose2vec` branch is a separate, reviewable change. Section 4 below lists exactly what it needs. |

---

## 1. Cell-by-cell plan

19 cells. One linear pass, no per-seed or probe machinery.

### Section 0 — Colab bootstrap (`cell 0-2`) — **Reuse verbatim (LOCKED)**
Markdown title + markdown bootstrap instructions + the bootstrap code cell copied byte-for-byte
from `10_vanilla_dt.ipynb` cell 2. The only edit anywhere is the notebook filename inside the
Colab badge URL in the markdown. Do not touch the code.

### Section 1 — Configuration (`cell 3-4`) — **New**
All knobs, notebook-10 style: grouped, ALL-CAPS, commented. Contains the subject inventory
comment block (counts regenerated from the parquet, see kickoff). Device selection copied from
notebook 10 (`cuda` → `mps` → `cpu`). `REPO_ROOT = Path.cwd().parent`, matching 10 — which is
also why the Colab bootstrap `chdir`s into `notebooks/`.

Derived, not configured: `VOCAB_SIZE`, `EXPORT_DIM = HIDDEN_UNITS`.

### Section 2 — Pose vocabulary (`cell 5-6`) — **New, this is the foundation**
Instantiate `CornerMazeEnv(session_type='exposure')`, reset, enumerate walkable cells:

```python
cells = sorted((x, y) for x in range(env.width) for y in range(env.height)
               if (c := env.grid.get(x, y)) is None or c.can_overlap())
POSE_VOCAB  = [(x, y, d) for (x, y) in cells for d in range(4)]   # x-major, then y, then d
POSE_TO_IDX = {p: i for i, p in enumerate(POSE_VOCAB)}
```

Assert `len(cells) == 49` and `len(POSE_VOCAB) == 196`. Print the occupancy map as ASCII so an
indexing error is visible immediately. Assert closure under the canonical rotation
`(x, y) → (y, 12 - x)` — already verified to hold for 1/2/3 rotations, but it is cheap and it
guards the rotation step downstream.

### Section 3 — Corpus (`cell 7-8`) — **New; reuses notebook 10's rotation helpers**
1. Resolve `SUBJECTS` ('all' → the 17) → `subject_id`s → `session_id`s, via
   `subjects.parquet` / `sessions.parquet`, asserting group membership as notebook 10 cell 6
   does.
2. Load the action table, filter to those sessions, require columns
   `{session_id, step, grid_x, grid_y, direction, pose_label, action}`.
3. Rotation: port `_compute_n_rotations`, `_rotate_pos_vec` and the corrected direction formula
   `(d - n) % 4` verbatim from notebook 10 cell 8. Gate on `USE_CANONICAL_FRAME`.
4. Group by `session_id`, sort by `step`, emit one integer sequence per session —
   **the sentences**. Never concatenate across sessions.
5. Assert every pose is in the vocabulary (measured: 0 escapes). Report coverage and the pose
   frequency distribution.

### Section 4 — Training pairs (`cell 9-10`) — **New**
Convert sentences into `(centre, context)` index pairs.

- `CAUSAL=True`: offsets `1 … WINDOW_RADIUS` (future only).
- `CAUSAL=False`: offsets `±1 … ±WINDOW_RADIUS`.

Vectorise with numpy slicing per sentence rather than a Python double loop — 222k steps ×
`2·R` offsets is a few hundred thousand pairs and a naive loop is needlessly slow. Report the
pair count.

**No sentinel tokens.** The vocabulary is exactly the 196 real poses — there is no `<START>`,
`<END>`, `<PAD>`, or `<UNK>`. Rationale, recorded because it is a natural thing to ask:

- Skip-gram predicts context *from* a centre, so unlike an autoregressive model it has nothing
  to bootstrap; the window just clips at the sentence edge, as in plain word2vec.
- Emitting individual `(centre, context)` pairs means every example is already complete, so no
  padding is required. (Switching `ARCHITECTURE` to `'cbow'` *would* need a pad token — call
  that out if CBOW is ever implemented.)
- Every exported row must map to a real `(x, y, d)` for the DT's `pose_to_row` dictionary.
  Sentinel rows would be junk rows the DT never indexes.
- No `<UNK>` is needed because the vocabulary is the fully-open maze, a strict superset of any
  barrier configuration; measured out-of-vocabulary rate is 0. The assertion in Section 3 is
  what catches a future violation.

Boundary effects from clipping are negligible and should not be "corrected". With sentences
being whole sessions (~1,900 steps on average) and `WINDOW_RADIUS=2`, clipping costs ~3 pairs
per session — roughly 354 of ~444,000 pairs, under 0.1%.

### Section 5 — Model (`cell 11-12`) — **New**
Classic skip-gram — one matrix in, one out, one knob:

```
one-hot(196) → hidden(HIDDEN_UNITS) → softmax(196)
```

**No non-linearity.** A one-hot times the input matrix is a row selection, so `nn.Embedding`
*is* the one-hot input layer and the hidden layer *is* the embedding — one and the same object.
`HIDDEN_UNITS` is therefore the only width: hidden units, embedding dimension, and export width
all at once. Its rows, `(196, HIDDEN_UNITS)`, are the representation — what the last section
plots and what gets exported.

**Keep `HIDDEN_UNITS` below 196.** Fewer dimensions than poses is the bottleneck that forces
poses to share dimensions, which is what yields overlapping place-field-like tuning instead of
196 independent codes. At ≥196 there is no bottleneck; a printed `[note]` fires to say so.

This is genuine word2vec — no MLP, no `ENCODER_DIM`. An earlier iteration split embedding width
from a ReLU hidden width; that was collapsed back to the single-matrix form because the
non-linearity measured slightly *worse* (1.32 vs 1.31) and the second width only caused
confusion.

`HIDDEN_UNITS` is asserted `int >= 1`. `NEURONS_TO_PLOT` entries outside `0..HIDDEN_UNITS-1`
are skipped with a warning rather than raising — a narrow layer must not abort the plot cell.

Init is the word2vec convention: small uniform on the embedding, zeros on the output. No
orthogonal init — that is the repo's rule for ReLU nets, and orthogonal rows here would impose
structure on the very thing being measured. Full softmax over 196 classes via
`CrossEntropyLoss`; no negative sampling.

Init: small uniform on the input embedding, zeros on the output — the word2vec convention. The
repo's orthogonal-init rule targets ReLU policy/value nets; a linear embedding table is a
different object, and orthogonal rows would impose artificial inter-pose structure on exactly
the thing we are trying to measure. Note this deviation in the notebook.

### Section 6 — Train (`cell 13-14`) — **New**
Adam, shuffled `TensorDataset`, per-epoch mean loss, `matplotlib` loss curve. Seed everything.
CPU-viable target: a couple of minutes.

### Section 7 — Export (`cell 15-16`) — **New**
```python
keys    = np.array(POSE_VOCAB, dtype=np.int32)          # (196, 3)
vectors = model.pose_representations()                  # (196, H) = hidden.weight
```
Assert `keys` matches a freshly enumerated `POSE_VOCAB` element-wise and that
`vectors.shape == (196, HIDDEN_UNITS)` before writing. Filename `pose2vec_{H}d.npz` pooled,
`pose2vec_{H}d_{subject}.npz` per subject. Round-trip: reload and assert equality.

### Section 8 — Spatial tuning (`cell 17-18`) — **New, the payoff**
Per neuron, four panels (E/S/W/N). Scatter the 196 weights into a 13×13 array pre-filled with
`np.nan` so non-occupiable cells render as background. Shared symmetric colour scale across the
four panels, diverging colormap centred at zero. Overlay `_draw_maze_skeleton` ported from
notebook 11 cell 28 (needs `WELL_LOCATIONS, CUE_LOCATIONS` from
`corner_maze_rl.env.constants`). `ax.set_ylim(12.5, -0.5)` for y-down screen coordinates —
copy that from 11 rather than re-deriving.

---

## 2. What notebook 10 would need (NOT done here)

Recorded so the follow-up is mechanical. `pose2vec` cannot reuse `grid_cell`'s arithmetic index
— it needs the dictionary pattern that `pose_visual` already uses.

| Location | Change |
|---|---|
| cell 4 | Add `'pose2vec'` to the `ENCODER_TYPE` comment block |
| cell 8, lookup dispatch | New branch loading `pose2vec_{H}d.npz`; set `STATE_LOOKUP = npz['vectors']` and build `pose_to_row = {(x,y,d): i}` from `npz['keys']` |
| cell 8, `_state_idx_for` | New branch using `pose_to_row[(x,y,d)]` on the **rotated** columns — not the `((x-1)*N_Y + (y-1))*N_D + d` formula |
| cell 12, `StateEncoder` | Treat like `pose_visual`/`grid_cell`: `nn.Embedding.from_pretrained` |
| cell 16, `_env_state_idx` | New branch mapping the env's live `(x, y, d)` through `pose_to_row` |

The `STATE_LOOKUP_SIZE` assertion needs no change — it is already generic.

---

## 3. Definition of done

- Runs top to bottom on CPU in a few minutes, no absolute paths.
- Vocabulary asserts 49 / 196; occupancy map prints and looks like the maze.
- Corpus subset assertion passes with 0 escapes; all 196 poses covered.
- Loss decreases; export round-trips; `keys`/`vectors` shapes and order asserted.
- Four-panel heatmap renders with the corridor lattice clearly visible.
- Both `POOL_SUBJECTS` modes run.

---

## 4. Post-build review — hardening applied

An adversarial review confirmed the four high-risk areas are correct: index/row-order
integrity, the canonical rotation, sentence boundaries, and heatmap indexing. The rotation was
validated empirically — all **117,033** genuine forward steps in the PI+VC corpus remain forward
steps after rotation, and rotated `(x, y, d)` matches notebook 10's independent string-based
`rotate_pose_label` path element-wise. The n_rot distribution is {0:54, 1:33, 2:30, 3:27}, so
odd rotations are well exercised and the legacy `(d + n) % 4` bug is genuinely absent.

Seven defects were found and fixed:

| # | Defect | Fix |
|---|---|---|
| 1 | `SUBJECTS='all'` admitted the 8 out-of-manuscript-scope subjects (latent — none are PI+VC, but a `TRAINING_GROUP='pi'` run would have included 3) | `OUT_OF_MANUSCRIPT_SCOPE` constant; excluded under `'all'`, warned on an explicit list |
| 2 | Per-subject exports shipped rows still at initialisation, with no warning (CM006: 29 rows) | `trained_mask` array in the npz + an explicit warning at export |
| 3 | Filename encoded only `HIDDEN_UNITS`, so an unrotated run silently overwrote the canonical lookup | `_raw` filename suffix when not canonicalised; full run config stored in the npz |
| 4 | Clipping-loss formula wrong except at the default | Corrected to `R(R+1)/2` causal / `R(R+1)` acausal |
| 5 | Plan cell counts drifted from the notebook | Renumbered |
| 6 | Promised pose-frequency report was missing | Printed in Section 3 — measured skew is **660:1**, rarest pose seen 9 times |
| 7 | The "maze shape proves the indexing" claim was too strong — the lattice is symmetric under transpose and 90° rotation | Added x/y tick labels and a filled **N** cue marker; corrected the markdown claim |

Defect 3 was the most serious: it is the one failure mode that produces silently wrong DT
training with no assertion capable of catching it downstream.
