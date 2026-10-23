# A100 dropout continuation implementation plan

> For agentic workers: use the existing tested training runner, focused TDD and one independent review. User explicitly authorized this experiment after discussing dropout and rejected LBFGS for this run.

**Goal:** Start one isolated fine-tuning run from full BART best180000, increasing dropout, while preserving all previous models and backing up the A100.

**Architecture:** Warm-start exact saved best weights; new AdamW state and a new low-LR schedule, explicitly recorded as fine-tuning rather than exact-state resume. Reuse sealed full bundle and AMP-safe token-loss training. Separate new remote root/code snapshot; no changes to old remote source, bundle or checkpoints.

**Tech stack:** Pinned remote torch2.9.1+cu128, transformers5.10.2, datasets4.6.0, accelerate1.15.0. Existing local shared dirty checkout is retained because user requested preserving/pushing it; remote code/output isolation protects running work.

## Fixed experiment

- Parent: artifacts/a100_full_20261023/training/trial/checkpoint-180000, weight SHA2059268173416c77cd8c901c791eb86e7cbcea0ede756b2bb830ec23638b8165. Parent final export is bit-identical; verify parent completion/best provenance and tokenizer before initialization.
- Same BART8+8/d1024/FF4096/244391936 parameters and BPE8000. Change only dropout0.1→0.2 and attention_dropout0→0.1 in the newly constructed layers; prove loaded state tensors unchanged at initialization and dropout active in train/off in eval.
- Full11593060train/118302val, pinned bundle0f330f228599f4cff65ed6ce0ca11e2d667ea24e6b911b5f67f6fff7e24bcbe5, no resampling/retokenization/label edits/official-test tuning.
- AdamW, LR3e-5, weight_decay0.01, new cosine horizon with5%warmup; max2 additional full epochs=90572 scheduled attempts. New optimizer/scheduler/scaler/RNG run seeded42; do not call this exact optimizer-state resume. Cold optimizer is an explicit implementation ruling for changed regularization/schedule and is recorded in manifests.
- FP16 nativeAMP, global256=micro128×accum2, gradientclip1, noGC/noCompile. Same guarded handling of confirmed AMP skips and hard nonfinites.
- Initial full-validation evaluation of exact parent weights with dropout disabled, before updates. Eval/save every5000, patience5, lower tokenloss better; retain best+last resumable full checkpoints and small plots/manifests/logs.
- Preserve a step0 baseline or an equally explicit parent-baseline fallback: final exported inference weights must not be declared better if no continuation validation improves on initial baseline. Parent artifacts remain immutable regardless.
- New remote root artifacts/a100_dropout_continue_20261023; autonomous tmux and sharedGPUlock. Disk/process checks before start. Backup streams may continue; no other GPU job overlap.

## Tasks and checks

- [x] Add optional warm-start/dropout settings to reusable runner or a small separate continuation runner, preserving default historical manifests/resume semantics.
- [x] Real tinyCPU integration RED→GREEN: exact parent initialization, effective dropout and unchanged eval outputs before updates; fresh lowerLR schedule actually updates; parent files unchanged; earlystop/baseline fallback and full optimizer/scheduler/RNG checkpoints; reject wrong parent/tokenizer/bundle/overwrite; default runner regression.
- [x] Implement strict driver/CLI with frozen request and provenance, preflight mode using a separate disposable output (never production run), no auto-restarts.
- [x] Independently review code/tests; verify pinned remote source hashes and finite GPUpreflight/VRAM headroom, then launch one full continuation and confirm advancing steps.
- [x] Push reviewed source/reports to requested Git backup branch. Update README with the actual experiment start and optimizer-reset distinction; do not claim public/private improvement in advance.

## Execution ledger

- User said «окей, давай попробуем дообучить тогда». Parent and previous submission remain separately preserved. No LBFGS, new architecture or target correction is authorized for this run.

- 2026-10-23: five continuation integration tests and historical regressions passed; complete suite306passed in62.23s, including distributed checks. Independent read-only review found no launch blockers.
- Snapshot27files SHA51251efdbc42956db169777e26a903a6580192641c71b1458126fea0026f24ff uploaded and rehashed remotely. Single tmux nlp-a100-dropout-20261023 dispatched; pipeline268796. GPU preflight2/2finite, full start awaiting saved completion gate.

- Main driver268935 started2026-10-23T15:04:24UTC; baseline full tokenloss0.19102141552704124, GPUpreflight complete. At15:09:31UTC training105/90572, finite trainloss0.2131; parent unchanged. Quiet monitor a100-dropout created for this active run.

- Code/tests/reports published on codex/a100-final-backup-20261023 at db88fee1d7d2ca2e957861ada5fb99f9777e2ce0; GitHub ls-remote matched. Large model backup remains autonomous and incomplete; no shutdown/cleanup.
