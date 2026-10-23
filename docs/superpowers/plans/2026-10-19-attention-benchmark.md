# H100 attention comparison

User-approved experiment: compare SDPA, FlashAttention-2 and FlashAttention-3,
then choose the fastest correct backend for the current 10+10, d1024 model.

## Constraints and protocol

- Preserve the running experiment, its weights, optimizer, data order, LR and global batch 256.
- Install optional benchmark dependencies in an isolated target directory.
- Use the existing frozen train-only benchmark data, same weights, FP32 parameters,
  BF16 autocast, FP32 cross entropy and fused AdamW for every backend.
- Check loss and gradients with dropout disabled, including padding and cross-attention.
- Measure complete training steps, batch 128, accumulation 2, length grouping;
  separate compilation/warmup from steady-state timings; repeat measurements.
- Run timings only while the main training is paused, with automatic resume on failure.
- Choose only from correct, stable implementations. If improvements are marginal,
  keep SDPA. Production integration, if justified, gets focused tests and review.

## Execution ledger

- Ruling: benchmark scripts are an authorized experimental spike in a separate
  report directory; no production behavior changes before evidence warrants them.
  Existing dirty project files are preserved; no commit/push is requested.
- Ruling: use official Hugging Face kernel builds where compatible, avoiding an
  upgrade of the running PyTorch environment or a lengthy CUDA source build.
- Preparation: current compiled SDPA dispatches to efficient attention; padded
  inputs prevent PyTorch's built-in flash backend. HF BART exposes flash adapters.
- Dependencies: latest kernels 0.17.2 is incompatible with this Transformers
  integration; isolated kernels 0.12.3 loads pinned official Torch 2.9/CUDA 12.8
  FA2 and FA3 builds successfully. The running environment was not modified.
- Correctness RED: stock FA2 adapter produced loss 0.960269 vs BF16 SDPA 0.477369,
  gradient cosine 0.731987. Traced to `_upad_input` reusing the encoder padding
  mask for decoder queries during cross-attention.
- Correctness GREEN: experimental adapter keeps every decoder query and unpads
  only encoder keys/values for cross-attention. FA2 loss 0.478676, FA3 0.477756;
  FP32 reference 0.477766. All 427 gradient tensors finite. Relative gradient L2
  vs FP32: SDPA 0.00932, FA2 0.00905, FA3 0.00858. No production change yet.
- A further 16 cross-attention regression cases passed (Q length 1/8/24/40,
  K length 24, left/right padding, output and Q/K/V gradients).
- Read-only fresh-context review found no important correctness issues in the
  scoped adapter. Cleanup was hardened against a benchmark exiting between
  process-identity check and killpg; fault injection confirms SIGCONT still runs.
- Timing qualification: six warmup batches need not cover every padded shape;
  inspect individual repeats for residual compilation. SDPA compiled repetitions
  are stable at 1394.70/1393.20/1389.47 examples/s, with 19.36s explicit warmup.
