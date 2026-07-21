# Kickoff brief — `cosmos_pose2vec` (a word2vec-style learned pose encoder)

**This document is the planning prompt.** Read it end to end, then produce
`docs/cosmos_pose2vec_plan.md` — a cell-by-cell implementation plan in the style of
`docs/11_planning_dt_plan.md`. Do **not** write the notebook until the plan is reviewed.

---

## Mission

Build a new notebook, `notebooks/cosmos_pose2vec.ipynb`, that learns a spatial embedding of
maze poses by training a word2vec-style network on rat trajectories, and exports the result as
a **new encoder type** (`ENCODER_TYPE = 'pose2vec'`) that the Decision Transformer stack can
consume.

The analogy is literal:

| word2vec | cosmos_pose2vec |
|---|---|
| word | pose — a grid position plus a heading, `(x, y, dir)` |
| vocabulary | the **196** poses occupiable in the fully-open maze |
| one-hot word vector | one-hot pose vector, length 196 |
| sentence / corpus | one yoked session's ordered pose sequence, from PI+VC rats |
| "predict the neighbouring word" | "predict the neighbouring position on the grid" |
| learned word embedding | learned pose embedding → **the encoder** |

The payoff is that the hidden layer should discover spatial structure on its own. Individual
hidden units are expected to look like place fields or heading-tuned cells when their weights
are plotted back onto the maze. The final cell is the test of that claim.

---

## Read first, in this order

1. `notebooks/10_vanilla_dt.ipynb` — the structural template. Focus on:
   - **cell 2** — the Colab bootstrap. Copy this verbatim.
   - **cell 4** — the configuration block. Copy its *style* (grouped, commented, ALL-CAPS).
   - **cell 7 (markdown) + cell 8 (code)** — how a state lookup table is loaded and how a
     per-step pose becomes an integer index. This is the contract the new encoder must satisfy.
     Note that `pose_visual` uses a **dictionary** lookup (`label_to_row`) while `grid_cell`
     uses an arithmetic formula — the new encoder follows the *dictionary* pattern.
   - **cell 12** — how the DT wraps a lookup table in a frozen embedding layer.
2. `notebooks/11_planning_dt.ipynb` **cell 28** — the maze-drawing helper
   (`_draw_maze_skeleton`) and the `imshow` + colorbar idiom. The new heatmap should match it.
3. `docs/11_planning_dt_kickoff.md` and `docs/11_planning_dt_plan.md` — the house style for a
   kickoff brief and its resulting plan. Mirror the plan's structure.
4. `CLAUDE.md` — repo conventions. Note especially that this repo is public and student-facing.

---

## Ground truth (verified 2026-07-21 — do not re-derive, but re-check if something looks off)

These are load-bearing. Each was confirmed against both the environment and the yoked data.

**The vocabulary is exactly 196 poses = 49 cells × 4 headings.**

The maze is a 13×13 MiniGrid, but only 49 cells are ever occupiable. The layout is a 3×3
lattice of corridors (verticals at `x ∈ {2, 6, 10}`, horizontals at `y ∈ {2, 6, 10}`, each
spanning 2–10) which is 45 cells, plus the 4 corner wells at `(1,1)`, `(11,1)`, `(1,11)`,
`(11,11)`. The corners are not adjacent to the lattice by ordinary movement — they are entered
with the dedicated "enter well" action (action `3`).

```
 .............      row = y (0 at top), col = x
 .#.........#.      '#' = occupiable
 ..#########..
 ..#...#...#..
 ..#...#...#..
 ..#...#...#..
 ..#########..
 ..#...#...#..
 ..#...#...#..
 ..#...#...#..
 ..#########..
 .#.........#.
 .............
```

**Derive the vocabulary from the environment, do not hardcode it.** The fully-open maze is the
Exposure A layout, reached via `session_type='exposure'`:

```python
env = CornerMazeEnv(session_type='exposure', render_mode=None)
env.reset(seed=0)
cells = sorted((x, y) for x in range(env.width) for y in range(env.height)
               if (c := env.grid.get(x, y)) is None or c.can_overlap())
assert len(cells) == 49
POSE_VOCAB  = [(x, y, d) for (x, y) in cells for d in range(4)]   # 196, x-major then y then d
POSE_TO_IDX = {p: i for i, p in enumerate(POSE_VOCAB)}
```

This matches the repo's rule that the env grid is ground truth. It was cross-checked against
the data: exposure sessions occupy exactly these 49 cells and all 196 poses.

> **Naming note — confirm this.** The brief said "the exposure 1a version of the maze." The
> dataset's exposure sessions are numbered `1e` / `2e`, so there is no session literally named
> "1a"; this was read as **Exposure A** (`expa`) — the fully-open, no-barriers-up layout. Note
> that Exposure B ends in a state the env comments describe as "fully open, ≡ expa", so both
> exposure variants converge on the same 49 cells. If something else was meant, say so, because
> the entire vocabulary depends on it.

**Headings follow the MiniGrid convention:** `0 = East (right)`, `1 = South (down)`,
`2 = West (left)`, `3 = North (up)`. The four heatmap panels correspond to these four values.

**Acquisition is a strict subset of this vocabulary — verified, zero escapes.** With barriers
up, fewer cells are reachable, so PI+VC acquisition data can only ever be a subset. Measured
across all PI+VC sessions in both `actions_real_pretrial` (222,064 rows) and
`actions_synthetic_pretrial` (197,572 rows): 49 cells used, **0 outside the vocabulary**, and
**all 196 poses visited**. So every embedding row receives gradient — there are no dead rows.
Still assert the subset property at load time; a violation means something upstream changed.

**The walkable set is closed under the canonical rotation — verified.** PI/PI+VC sessions are
rotated so the cue sits at North, mapping `(x, y) → (y, 12 − x)`. The 49-cell set maps onto
itself for 1, 2, and 3 rotations with zero escapes, so canonicalisation can never produce a
pose outside the vocabulary.

**Poses are already stored per timestep — no environment replay is required.** The action
tables carry `grid_x`, `grid_y`, `direction`, and `pose_label` alongside `session_id`, `step`,
`action`, and `actions_to_reward`. Read the parquet, sort by `step`, and the pose sequence
falls out. (This is the single biggest simplification available — do not build a replay
harness.)

**The export format mirrors `data/lookups/grid_cells_60d.npz`**, which holds `keys` (484, 3)
and `vectors` (484, 60). Write `data/lookups/pose2vec_{H}d.npz` with the same two array names
but sized to the new vocabulary: `keys` **(196, 3)** giving the `(x, y, d)` triple per row, and
`vectors` **(196, HIDDEN_UNITS)**.

**PI+VC has 17 subjects:** CM000–CM011, CM014–CM018. Restrict to the manuscript-roster subjects
(`README.md` § Manuscript subject roster).

---

## Notebook structure — mirror `10_vanilla_dt.ipynb`

| # | Cell | Content |
|---|---|---|
| 0 | markdown | Title + one-paragraph explanation of the word2vec analogy |
| 1 | markdown | "## 0. Colab bootstrap" — copy from 10, change only the notebook name in the badge URL |
| 2 | code | Colab bootstrap — **copy verbatim from 10 cell 2**, no edits |
| 3 | markdown | "## 1. Configuration" |
| 4 | code | All parameters and hyperparameters (see next section) |
| 5 | markdown | "## 2. Pose vocabulary" |
| 6 | code | Build the 196-pose vocabulary from the env; assert size; print the occupancy map |
| 7 | markdown | "## 3. Build the pose corpus" |
| 8 | code | Load PI+VC yoked poses → per-session sequences → (centre, context) pairs |
| 9 | markdown | "## 4. Model" |
| 10 | code | The one-hidden-layer network |
| 11 | markdown | "## 5. Train" |
| 12 | code | Training loop + loss curve |
| 13 | markdown | "## 6. Export as an encoder" |
| 14 | code | Write `pose2vec_{H}d.npz` |
| 15 | markdown | "## 7. Spatial tuning of hidden units" |
| 16 | code | The four-panel per-neuron heatmap |

Keep it to one linear pass. This notebook trains one small model — it needs none of notebook
10's per-subject / per-seed / probe machinery.

---

## Configuration knobs (cell 4)

The three named explicitly in the brief, which must be present:

```python
HIDDEN_UNITS  = 60     # width of the single hidden layer == the embedding dimension
WINDOW_RADIUS = 2      # how far along the trajectory counts as "adjacent"
CAUSAL        = True   # see the definition below — this one needs confirming
```

Supporting knobs to add. The subject block mirrors notebook 10 cell 4's style — an inventory
comment sitting directly above the entry, so the reader can pick subjects without leaving the
cell:

```python
TRAINING_GROUP = 'pi_vc'          # reuse notebook 10's GROUP_ALIASES mapping

# ── Usable PI+VC subjects — all 17 are in the manuscript roster ──────────────
# Acquisition sessions / steps available in actions_real_pretrial:
#   CM000   7 / 12,786      CM007   9 / 11,624      CM014  10 / 18,890
#   CM001   6 / 11,462      CM008   3 /  6,827      CM015  11 / 21,253
#   CM002   7 / 12,355      CM009   7 / 16,065      CM016   5 / 11,005
#   CM003   8 / 14,069      CM010   5 / 10,951      CM017   9 / 12,346
#   CM004   8 / 20,048      CM011   5 / 10,016      CM018   9 / 13,449
#   CM005   6 / 11,801
#   CM006   3 /  7,117            all 17 pooled = 118 sessions / 222,064 steps
# Smallest: CM006 / CM008 (3 sessions).  Largest: CM015 (11), CM004 (20k steps).
SUBJECTS      = 'all'   # 'all' → every subject listed above,
                        # or an explicit list, e.g. ['CM004', 'CM015']
POOL_SUBJECTS = True    # True  = ONE embedding trained on all SUBJECTS pooled (recommended)
                        # False = one independent embedding per subject

CORPUS_SOURCE  = 'real_pretrial'  # 'real_pretrial' | 'synthetic_pretrial' | 'both'
USE_CANONICAL_FRAME = True        # rotate PI/PI+VC to the cue-at-North frame
ARCHITECTURE   = 'skipgram'       # 'skipgram' | 'cbow'
EPOCHS, BATCH_SIZE, LEARNING_RATE, SEED
```

Keep the counts above accurate — regenerate them if the dataset is rebuilt rather than letting
the comment rot.

`VOCAB_SIZE` is **derived** (`len(POSE_VOCAB)`) and asserted `== 196` — never hardcoded as a
configuration value.

`ENCODER_DIM` (embedding width) and `MLP_HIDDEN_UNITS` (optional non-linear hidden layer) are
independent, and whichever layer is the representation is the one exported — `(196,
ENCODER_DIM)` with no MLP, `(196, MLP_HIDDEN_UNITS)` with one. Export width is a **free
parameter**; nothing in this notebook constrains it. 60 is only a starting value.

The width that matters scientifically is the bottleneck: a narrow embedding forces poses to
share dimensions, which is what produces overlapping place-field-like tuning rather than 196
independent codes. That is the knob worth sweeping.

---

## Design decisions to lock in the plan

**1. What "causal" versus "acausal" means.** The brief said "causal (predict words after)" and
"acausal (to predict words after)" — the second is a typo. Plan for this reading, and flag it
for confirmation at the top of the plan:

- `CAUSAL = True` — context is only the poses that come **after** the centre pose, up to
  `WINDOW_RADIUS` steps ahead. This is the "predict the next position" framing, and it respects
  the arrow of time in a trajectory.
- `CAUSAL = False` — context is a **symmetric** window, `WINDOW_RADIUS` steps both before and
  after. This is standard skip-gram, and it treats adjacency as direction-agnostic.

**2. Which matrix is the encoder.** In skip-gram, a one-hot input times the input weight matrix
just selects a row — so the hidden activation for a pose *is* that pose's row of the input
matrix. The input matrix, shape `(196, HIDDEN_UNITS)`, is the encoder and is what gets
exported. The output matrix is discarded after training. State this explicitly; conflating the
two is the classic word2vec implementation error.

**3. Which coordinate frame.** Notebook 10 rotates PI and PI+VC sessions into a canonical frame
where the cue always sits at North. Recommend training on the **rotated (canonical) poses**, so
the exported lookup is indexed the way the DT will index it. Reuse notebook 10 cell 8's
rotation helpers rather than reimplementing — and note the corrected direction formula
`(d - n) % 4`, not the legacy `(d + n) % 4`.

**4. Sentence boundaries.** A context window must never span two sessions — that would invent
an adjacency between the last pose of one session and the first of the next. Group by
`session_id` and build windows within each group. Whether to also break at trial boundaries is
a judgement call: the rat walks continuously through the inter-trial interval, so the poses
genuinely are contiguous. Recommend *not* breaking at trial boundaries, and say why.

**5. How the well cells behave.** The 4 corner wells are entered by action `3`, not by ordinary
movement, so in the pose sequence a well pose sits adjacent to a lattice pose it is not
physically neighbouring. That is fine and arguably desirable — word2vec adjacency is
"co-occurs in sequence", not "is one grid step away". Note it so the heatmaps are read
correctly.

**6. Training across multiple rats.** Both modes must work, and the loop axis is the one thing
that differs between them:

- **Pooled (`POOL_SUBJECTS = True`, the recommended default).** Every selected subject's
  sessions become sentences in one shared corpus, and a single embedding is trained. This is
  the right default here because the embedding is meant to describe *the maze*, not an
  individual animal — pooling all 17 rats gives ~222k steps and full coverage of all 196 poses,
  where a single small subject like CM006 (3 sessions, ~7k steps) may leave rare poses barely
  trained. Pooling is safe precisely because the vocabulary is shared: the canonical frame makes
  every rat's poses mean the same thing.
- **Per-subject (`POOL_SUBJECTS = False`).** Loop over subjects and train an independent
  embedding for each. Useful for asking whether different rats induce different spatial codes,
  but expect noisier maps for the low-session subjects.

Sentence boundaries already prevent windows from spanning sessions, so pooling needs no extra
guard — a pooled corpus is just a longer list of per-session sequences, never a concatenation.

Export naming must distinguish the two: pooled writes `pose2vec_{H}d.npz`, per-subject writes
`pose2vec_{H}d_{subject}.npz`. Do not let a per-subject run silently overwrite the pooled
artifact. If per-subject mode is combined with the heatmap cell, plot one subject at a time
rather than averaging across rats.

---

## Ranked gotchas

1. **Do not build an env replay.** Poses are columns in the parquet.
2. **The new encoder cannot reuse the `grid_cell` arithmetic index.** That formula
   (`((x-1)*11 + (y-1))*4 + d`) assumes the dense 484 space. With 196 poses the mapping is a
   dictionary over the canonical enumeration. Any DT-side branch must build `(x,y,d) → row`
   from the npz `keys` array, exactly as `pose_visual` builds `label_to_row`. Getting this
   wrong silently trains on scrambled states.
3. **Row order in the export is a contract.** Assert that the exported `keys` array matches a
   freshly enumerated `POSE_VOCAB` element-wise before writing.
4. **A full softmax over 196 classes is cheap.** Skip negative sampling and hierarchical
   softmax — they exist for vocabularies in the 10⁵–10⁶ range. Say this explicitly so nobody
   adds complexity that buys nothing.
5. **Initialisation.** Repo convention is orthogonal init, but word2vec conventionally uses
   small uniform init on the input matrix and zeros on the output. Note the tension, pick one,
   justify it.
6. **Class imbalance.** Rats dwell far longer in some poses than others, and the "pause" action
   inflates self-transitions. Consider frequency subsampling, or at minimum report the pose
   frequency distribution; if skipping it, say why.

---

## The final plot (last cell)

For a chosen hidden unit, plot its weight across all 196 poses as **four heatmaps, one per
heading** — East, South, West, North, labelled as such, not as raw integers. Panel `d` at
position `(x, y)` shows `input_matrix[POSE_TO_IDX[(x, y, d)], neuron]`.

- Scatter the 196 values into a 13×13 array (or crop to the 11×11 interior) with
  **non-occupiable cells set to NaN** so they render as background. The corridor lattice should
  be plainly visible — that is also a free correctness check on the indexing.
- Share a colour scale across the four panels so they are comparable, and centre a diverging
  colormap at zero.
- Overlay the maze skeleton using the helper in `notebooks/11_planning_dt.ipynb` cell 28. There
  is no existing value-per-cell maze heatmap in the repo, so that helper plus a standard
  `imshow` + colorbar is the pattern to follow.
- Get the orientation right — screen coordinates are y-down here. Verify against a known
  landmark (the cue at North after canonicalisation) rather than assuming.
- Support browsing several units, e.g. a `NEURONS_TO_PLOT` list, so the reader can hunt for
  interpretable ones rather than being shown a single cherry-picked unit.

---

## Definition of done

- The notebook runs top to bottom on CPU in a few minutes.
- The lookup exists with `keys` (196, 3) and `vectors` (196, H), and the row-order assertion
  passes — `pose2vec_{H}d.npz` when pooled, `pose2vec_{H}d_{subject}.npz` per subject.
- Both `POOL_SUBJECTS` modes run: pooled over all 17 subjects, and a per-subject loop.
- The corpus-subset assertion passes: every pose in the training corpus is in the vocabulary.
- Setting `ENCODER_TYPE = 'pose2vec'` in notebook 10 loads the new lookup and trains. **Listing
  the exact touch-points is part of the deliverable** — specifically the new branch in cell 8's
  encoder dispatch, the `(x,y,d) → row` dictionary that replaces `_state_idx_for`'s arithmetic
  path, the `STATE_LOOKUP_SIZE` assertion, and the matching branch in cell 12 and in cell 16's
  `_env_state_idx`.
- Training loss decreases and the four-panel plot renders with at least some units showing
  visible spatial structure.
- No absolute local paths, no machine-specific assumptions.

---

## Open questions — answer these at the top of the plan

1. **Confirm "exposure 1a" means the Exposure A / fully-open layout** (49 cells, 196 poses), as
   read above.
2. **Confirm the causal / acausal definitions** (the brief had a typo).
3. **Notebook numbering** — the repo numbers notebooks (`10_`, `11_`), and `docs/12_` is
   already taken by an unrelated plan. Should this be `13_cosmos_pose2vec.ipynb`, or stay
   unnumbered as `cosmos_pose2vec.ipynb` as named in the brief?
4. **Default corpus scope** — both modes are supported; confirm that *pooled over all 17
   subjects* is the right default, versus defaulting to a single named rat as notebook 10 does.
5. **Acquisition only, or all session types?** Exposure sessions traverse the open maze and
   would enrich coverage of the corridors that barriers normally block.
6. Should notebook 10 be **edited** in this same change to add the `'pose2vec'` branch, or is
   that a separate follow-up?
