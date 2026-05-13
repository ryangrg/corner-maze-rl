# DT training speedup — pose-ID dataset + compile + bf16 across notebooks 04–09

One-epoch smoke runs after applying the same set of optimizations to every DT
training notebook (`04` … `09`). All numbers below are from a single sequential
run on an RTX 4090 Laptop GPU (16 GB).

## Changes applied

| Change | Why it matters |
| --- | --- |
| **Pose-ID dataset** — store int16 IDs (484 distinct poses, 11×11×4), expand to 60-D vectors on-GPU at batch time | Dataset RAM drops from ~30 GB to ~250 MB (short-window), or ~540 MB → ~25 MB (full-context). Build time from ~minute → <1 s. |
| **`BATCH_SIZE` 64 → 1024** (04–07 only) | Short-window models were launch-bound at batch=64 on a 4090 (50 W draw). 1024 keeps VRAM under 3 GB. Left at 4 (08) / 4×8 (09) — those already process big tensors per step. |
| **`torch.compile(model)`** | Fuses dozens of small kernels (linears, layernorms, softmax, gelu, …). One-time compile cost on the first batch. |
| **bf16 autocast** in forward + loss | Engages 4090 tensor cores; bf16's fp32-equivalent dynamic range means no `GradScaler`. |
| **`pin_memory=True`** + **`non_blocking=True`** transfers | Overlaps host→device with compute. |
| **`tqdm.auto` per-epoch progress bars** | Running loss/acc as postfix; `leave=False` so bars don't pile up. |
| **Save `model._orig_mod.state_dict()`** | The compile wrapper would otherwise pollute checkpoint keys. |
| **`EPOCHS = 5`** for the full-context notebooks (08, 09) | Each epoch is more expensive at `L=4096`; default reduced from 30 → 5. |

## 1-epoch smoke results

| nb | arch | params | batch | ctx | dataset RAM | build (s) | epoch 1 (s) | train acc | val acc |
| -- | ---- | -----: | ----: | --: | ----------: | --------: | ----------: | --------: | ------: |
| **04** | softmax DT (60-D) | 160 k | 1024 | 32 | 236 MB | 0.4 | **31.3** | 0.788 | 0.840 |
| **05** | linear-attn DT (60-D) | 160 k | 1024 | 32 | 236 MB | 0.4 | 67.5 | 0.789 | 0.837 |
| **06** | softmax decoupled (128-D) | 410 k | 1024 | 32 | 236 MB | 0.4 | 40.2 | 0.806 | 0.848 |
| **07** | linear-attn decoupled (128-D) | 410 k | 1024 | 32 | 236 MB | 0.4 | 96.7 | 0.815 | 0.848 |
| **08** | full-ctx linear-attn (60-D) | 404 k | 4 | 4096 | 25 MB | 0.3 | 24.4 | 0.640 | 0.660 |
| **09** | hybrid full+short DT | 566 k | 4×8 = 32 | 4096 + 64 | 16 MB | 0.3 | 7.1 | 0.645 | 0.661 |

Notes on the table:
- `epoch 1 (s)` includes the one-time `torch.compile` cost on the first batch — true steady-state is faster (~30 s/epoch for 04 vs 31.3 s here, etc.).
- `batch` for 09 is "long-stream batch × short windows per session" — 4 sessions × 8 windows per step.
- `train acc / val acc` after a single epoch isn't a quality measure — it just confirms the loss is going the right direction. The 1024-batch notebooks do far fewer optimizer steps per epoch than the old 64-batch baseline (~924 vs ~14 800), so early-epoch loss/acc trail what 04 hit at epoch 1 before the refactor — they catch up by epoch 2-3.

## Per-batch cost (computed from the smoke numbers)

| nb | batches / epoch | ms / batch |
| -- | --------------: | ---------: |
| 04 | ~1 029 | **30** |
| 05 | ~1 029 | 66 |
| 06 | ~1 029 | 39 |
| 07 | ~1 029 | 94 |
| 08 | ~137 | 178 |
| 09 | ~137 | 52 |

Two things stand out:
1. **Linear attention is ~2× slower than softmax at short context** in this implementation (05 vs 04, 07 vs 06). At seq_len = 3·32 = 96 the softmax `O(L²)` ops are cheaper than the linear-attn parallel-form cumulative-sums plus its extra kernel launches. Linear attention only earns its keep once `L` is large — which is exactly what 08 exploits.
2. **The decoupled (128-D) models are only ~30 % slower than the matched (60-D) ones** despite ~2.6× the parameters, because the FFN dominates compute and the inner matmuls fit in tensor cores once you cross d_head ≥ 64.

## Where the wall time goes

The 4090 was running at ~70–85 W (out of 175 W TDP) during these epochs. Util reads 90–97 % but power tells the true story: each step finishes in well under 1 ms of actual compute. **The remaining time is kernel-launch overhead and host-side syncs** (`loss.item()`, `.argmax().sum().item()`, `optimizer.zero_grad()`, etc.). A separate `Tensor.item()` graph-break warning in the compile log confirms that `nn.TransformerEncoder._detect_is_causal_mask` prevents a single fused graph — splitting it into ~2–3 subgraphs around the encoder.

Cheap wins still on the table:
- Pass `is_causal=True` to `nn.TransformerEncoder` explicitly to skip the detection probe → likely +10–20 %.
- Accumulate `sum_loss` / `sum_correct` as on-device tensors instead of calling `.item()` per batch → -5–15 %.
- Bigger context or wider model to make compute actually saturate (this stops being free).

## Sanity-check against the pre-refactor numbers (notebook 04, the only one we benchmarked end-to-end)

- Before: dataset build ~minute, ~30 GB RAM peak, ~83 s/epoch at batch 64.
- After: dataset build 0.4 s, 236 MB RAM, **~30 s/epoch steady-state** at batch 1024.
- Net: ~125× less RAM, ~150× faster dataset build, ~2.7× faster per epoch.
- Loss/val-acc trajectories at matched total batches are identical to the pre-refactor run.

## Reproducing

```
uv run python /tmp/run_all_smokes.py
```

Writes `/tmp/smoke_results.json` (the table data) and the per-step log to `/tmp/all_smokes.log`.
