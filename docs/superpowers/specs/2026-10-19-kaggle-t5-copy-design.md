# Kaggle: T5 and copy Transformer

User request: prepare and launch two separate private Kaggle notebooks, using the
same fixed H100 selection and BPE, and fix the T5 launch failure. Work builds on
the previously agreed T5-style and pointer-generator hypotheses. Do not resume
or use H100. No pretrained weights, new targets, speculative decoding or public
uploads. Existing full model002 and uncommitted work must remain intact.

Both start from scratch on exactly 1,000,000 train / 20,000 val, seed42, existing
BPE8000, maximum input/target256, BOS decoder start, labels code+EOS without BOS.
One epoch, effective batch256, LR2e-4, warmup5%/cosine, AdamW decay0.01, clip1,
validation every1000 optimizer steps and final. Same loss is summed FP32 CE over
all target tokens, correctly normalized across accumulation and DDP.

Implementation choices made by the agent: T5 remains 8+8/d1024/GEGLU2816
(247.3M). Copy control uses BART8+8/d1024/FF4096 (244.4M plus a small copy head),
with a separate projected query/key copy attention and a sigmoid gate between
vocabulary and source-token probability. Duplicate source IDs sum their mass;
PAD/BOS/EOS/UNK and padding positions cannot be copied. All-masked sources fall
back to the vocabulary. No auxiliary copy targets. Return log mixture
probabilities as logits, so HF greedy/beam generation and existing loss tools
consume the correct distribution. Copy source IDs travel in encoder outputs,
including cached generation and beam expansion; no mutable per-request state.
Keep main BART attention SDPA; do not request all layer attention matrices.

Copy checkpoints require a distinct HF config/model type and explicit model
loader; reloading as ordinary BART must never silently drop the copy head.
Source IDs and encoder masks must survive generation expansion, cached decode,
DataCollator teacher forcing and HF Trainer save/resume. Native and wrapper
round trips, copy gradients and a tiny overfit task must work.

Fix the pinned torch2.9/transformers5.10 TF32 mismatch by normalizing new precision
overrides back to unset before applying the legacy settings still read by
Inductor/Accelerate. Apply only for compiled torch2.9 Trainer paths, after Trainer
initialization. Test using actual HF TF32 setter plus actual legacy getters in
an isolated process. Tiny compiled Trainer gets a real optimizer step; CUDA
preflight runs on Kaggle before paid/long training. No claim of local CUDA tests.

Notebooks share a CLI runner and an uploadable private dataset bundle containing
current code, fixed pretokenized rows, tokenizer, selection manifest and indices.
Verify byte hashes instead of reselecting/re-tokenizing or rewriting provenance
paths. Dataset loads from read-only /kaggle/input; writes go to /kaggle/working.
Do not reinstall PyTorch blindly over the working Kaggle CUDA image. Install
only compatible pinned non-Torch packages in the notebook. Require torch>=2.6
and use its installed CUDA build, recording exact versions.

Use torchrun for all visible GPUs (one model replica per GPU), target batch256
via configurable physical batch16 and derived accumulation; reject nondivisible
batch layouts. Default compile off for the short Kaggle screen, opt-in after
preflight; this is not the T5 bug fix. Gradient checkpointing default on for
16GiB GPUs. Auto-select BF16 where supported, otherwise FP16; FP32 loss in both.
Both run the same tiny preflight before the full epoch, with finite loss/gradient
checks, checkpoints, final export, logs and plots. Nonfinite training must fail
visibly, not skip and publish misleading metrics. Notebook cells must surface
subprocess failures. Resume has explicit run/config/data/layout validation.

Launch both on Kaggle if account resources permit. Keep notebooks/dataset private.
Verify actual advancing steps and finite losses, record URLs and observed GPU
configuration. If Kaggle imposes login/verification/quota or concurrent-GPU
limits, report the concrete blocker and complete all independent preparation.
README records actual status and measurements; no invented quality result.
