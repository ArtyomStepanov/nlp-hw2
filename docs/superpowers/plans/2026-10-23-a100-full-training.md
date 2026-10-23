# A100 full training implementation plan

> **For agentic workers:** Use subagent-driven development for the independent runner and data/launch tasks; parent owns deployment and monitoring.

**Goal:** Start a durable overnight full-data training run of the selected BART architecture, up to four epochs with early stopping, and confirm real progress before telling the user they can close the laptop.

**Architecture:** Reuse the complete, already tokenized original train/validation cache. Create a sealed hardlinked bundle with a derived in-memory length column, extend the tested portable AMP-safe runner with optional early stopping, and launch in a fresh remote directory through tmux after Stage4 releases the GPU.

**Tech Stack:** Existing PyTorch2.9.1/Transformers5.10.2/datasets4.6, Arrow, SSH/tmux; no dependency installation or Git changes.

**Spec:** User explicitly requested full overnight training and selected four epochs with early stopping, allowing completion during the day. This authorizes full training beyond the earlier four-check plan; earlier architecture/data/optimizer choices are retained.

## Global constraints

- Fresh BART8+8/d1024/FF4096, BPE8000 unchanged, seed42, max input/target256, LR2e-4, cosine/warmup5%, FP16, global256/micro128/accum2, no compile/checkpointing.
- Original full train11,593,060 and full validation118,302; original test excluded. Four epochs =181,144 planned Trainer step attempts; validation/save every5000, patience5, minimize eval_token_loss and restore best weights.
- Parent choice: train from scratch with the selected settings. This is a new full-data schedule, not a mismatched resume from a short trial. User was informed in commentary.
- Reuse existing cache and historical selected-row proof; validate its recorded path, row counts, fingerprints, tokenizer and fresh file hashes. No retokenization, corrected targets, filtering or synthetic examples.
- Hardlink raw train/val shards; reject indexed views before deriving max source/target length, preventing accidental flatten/rewrite. Derivation occurs only with pinned manifest derive_length=true.
- New remote root/artifacts and source snapshot; running Stage4 and previous runs stay immutable. No overlap on the single GPU; tmux persists after laptop/SSH disconnect.
- Preserve best and latest full resumable checkpoints; exact FP16 resume needs optimizer/scheduler/scaler/RNG. Completion marker is published only after all final artifacts.
- No automatic restart on future failures, no destructive cleanup, no weights/cache downloads locally. Monitor only meaningful events.

## Review focus

- Historical portable manifests remain byte-equivalent when optional patience=None, including Stage2 strict verifier.
- Early stopping retains callback patience through an explicit resume and saves latest state before best-weight restore.
- A final step off the regular save interval retains a resumable checkpoint under pinned HF behavior.
- Derived length preserves token buffers and all row ordering; source split coverage is arange with no validation/test leakage.
- Fresh output/locks and exact source checks prevent duplicate or partial-output overwrite; training really advances before completion claims.

## Tasks

### 1. Portable runner extension

Files: kaggle_experiments.py; canonical serializer use in tools/run_a100_stage2.py; tests/test_full_training_runner.py.

- [x] Add optional patience, canonical manifest serialization, early/best/callback-state settings and best/latest provenance.
- [x] Allow pinned derived-length raw bundles using zero-copy Arrow lengths; retain old strict behavior otherwise.
- [x] Run real tiny CPU tests for early stop, best/latest checkpoints, off-period final saving and resume state; run existing AMP/portable regressions.

### 2. Full cache bundle and launch CLI

Files: tools/prepare_a100_full.py, tools/run_a100_full.py, tests/test_a100_full.py.

- [x] Seal a fresh hardlinked train/val bundle with source proof, arange indices and SHA manifest; atomic publication, no partial retry.
- [x] Build frozen full-run config in an isolated trial output; guard duplicate output, persist queue/request and explicit resume checks.
- [x] Test tiny real Arrow link/integrity/fingerprint/coverage cases and mocked full-run configuration/failure behavior.

### 3. Deployment and monitoring

- [x] Independently review focused changes; verify a frozen source snapshot and launcher.
- [x] Let already authorized Stage4 finish; retain its results and pending T4 benchmark rather than interrupting it.
- [x] Prepare data and launch full training in tmux; verify manifests, actual training step and finite loss.
- [x] Update README/ledger and the existing heartbeat to monitor the full run and remaining Stage4/T4 artifacts.
- [x] Tell the user the remote run is confirmed and laptop/SSH may be closed; local heartbeat availability depends on the app being online.

## Execution ledger

- User authorization: four epochs with early stopping, can run until daytime. No additional approval is required for this concrete requested launch.
- Ruling: use current shared dirty checkout for narrowly owned local changes and an isolated immutable remote snapshot; no commits/reset/worktree migration that would omit existing uncommitted runner work.
- Pinned HF DefaultFlowCallback already forces final eval/full checkpoint at max_steps; do not save a mixed latest optimizer plus restored-best weights after train returns.

- 2026-10-22T22:24:27UTC: 26-file snapshot installed after SHA checks, tmux nlp-a100-full-20261023 started (pipeline PID199774). Full bundle sealed at22:24:47UTC; currently waiting for Stage4 exit0. Focused42 tests passed; independent review found no blockers. Existing heartbeat updated to full paths; do not duplicate launch.

- Full training confirmed: driver225026 since22:27:19UTC, source/bundle verified, step120965 at2026-10-23T08:25:47UTC with finite loss; last saved validation0.2022639645 at120000, bestcheckpoint120000. Laptop-independent tmux launch achieved; full training completion remains under heartbeat monitoring.

- 2026-10-23T13:26:37UTC: all4epochs/181144attempts completed exit0, no activeprocesses. Final tokenloss0.1910215738661824, best1800000.19102131500784642; finalweights bit-identical tobest, all344tensorsfinite, latest181144resumable anddistinct.34smallfiles4.11MB archived andSHA-verified. Fullpass@1 andactualT4 timing remain pending; no newexperiment started. Monitor to be deleted after independent report verification.

- Independent closure verification passed (34files,26sourcehashes,1851finitehistoryrows); final model tensor equality/finite proof passed separately. A100 automation deleted using the app tool after complete exit0 and report archival. Full-model pass@1 on original20k andactualT4 timing remain explicitly pending.
