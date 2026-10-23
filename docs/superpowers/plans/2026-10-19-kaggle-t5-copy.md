# Kaggle T5/copy implementation plan

> Implement inline with TDD; obtain a fresh whole-change review before launch.

**Goal:** Two private runnable Kaggle notebooks on the exact saved data, with a
working T5 launch and a trained-from-scratch copy Transformer implementation.
**Spec:** ../specs/2026-10-19-kaggle-t5-copy-design.md
**Architecture:** shared pretokenized runner, native HF models, separate copy head,
existing token-loss Trainer and a small pinned-version precision compatibility fix.
**Tech Stack:** Python, torch>=2.6 (existing Kaggle CUDA build), transformers5.10.2,
datasets4.6/tokenizers0.22.2, notebook JSON, Kaggle UI uploads and private kernels.

## Constraints and review focus
- Preserve exact data and tokenizer hashes, row order and target shifting.
- Copy head must be differentiable, normalized and finite with duplicate IDs,
  masked tokens and all-masked input, including AMP and cached beam decoding.
- A custom copy checkpoint cannot reload into plain BART or resume wrong data.
- Preserve global256 and token normalization across single GPU and DDP.
- File uploads remain private; no new credentials or H100 usage.

## Task 1: T5 compile compatibility
Files: utils.py, training.py, tests/test_compile_precision.py.
- [x] RED: actual HF enable_tf32 plus Inductor-style legacy read raises before
  the compatibility helper; assert reads work after actual Trainer initialization.
- [x] Implement torch2.9-only normalization in compiled _TokenLossTrainer init.
- [x] GREEN: isolated regression, tiny compiled T5 optimizer step and existing T5 tests.

## Task 2: Pointer-generator BART
Files: models/copy.py, models/baseline.py, models/__init__.py, training.py,
tests/test_copy.py, tests/test_training.py.
- [x] RED: hand-computed mixture/duplicate IDs and all-masked fallback, finite
  gradients, source/future masks, shifted labels, greedy/beam cached equivalence,
  checkpoint round trip and incompatible resume.
- [x] Implement CopyBartConfig, CopyBartForConditionalGeneration, CopyTransformer.
- [x] GREEN: tests including a tiny learned-copy example and Trainer save/resume.

## Task 3: Portable data, runner and notebooks
Files: kaggle_experiments.py, train_kaggle.py, tools/prepare_kaggle_bundle.py,
kaggle_t5.ipynb, kaggle_copy.ipynb, requirements-kaggle.txt,
tests/test_kaggle_experiments.py, README.md.
- [x] RED: portable integrity checks reject changed tokenizer/indices/data; tiny
  two-step run exports reloadable model and resumable checkpoint for both models.
- [x] Implement verified bundle, runner, deterministic DDP layout, finite checks,
  preflight, save/resume, logs and plots; notebook execution propagates failures.
- [x] GREEN: validate notebook JSON/syntax and execute its launch logic at the
  subprocess boundary; run actual shared runner on a local tiny fixture, project
  suite `python3 -m pytest tests`, get fresh review and fix important findings.
- [ ] Build upload archive, upload as private Kaggle dataset, import both notebooks,
  choose available GPU, launch after GPU preflight and verify advancing finite loss.
- [ ] Record actual launch URLs/status in README and report results to user.

## Ledger
Pre-flight: Task1's _TokenLossTrainer is shared by the existing sweep and Task3;
compatibility must live in its initializer rather than only a notebook cell.
Task2 exposes model.network as a native HF model; Task3 uses the same normalized
loss/metric utilities as existing BART/T5. Copy uses log probabilities as logits,
so cross_entropy is equivalent to mixture NLL up to FP32 rounding.
Ruling: retain the dirty project checkout, no automatic commits or worktree copy;
the user is preparing experiments using the existing uncommitted infrastructure.
Ruling: existing conversation supplies the experiment design and execution intent;
do not ask the user to reapprove the implementation choices listed in the spec.
Builder check: fresh reviewer found no important unanswered questions.
Task1: RED reproduced the exact mixed-TF32 exception; GREEN isolated real HF
setter + compiled T5 Trainer (CPU/eager Dynamo backend) updates weights for2 steps.
Existing T5 tests12passed. CUDA preflight still required on Kaggle.
Task2: initial8 tests RED before implementation;7 initial tests GREEN, including
beam1/4 and HF checkpoints. Unseen-copy fixture only promises its first token;
EOS is asserted on known examples, because an unseen decoder embedding has not
learned the stopping rule. Additional AMP and resume-recipe checks added.
Task2 complete:10 copy tests including FP16 head, duplicate IDs, all-masked
fallback, source/future masks, learned unseen source token, cached beams1/4,
wrapper/native AutoModel checkpoints, incompatible resume; actual Trainer
integration parametrized with copy and passes.
Task3 local complete: real2-process Gloo training for both models passes;
loopback address avoids macOS hostname/IPv6 rendezvous. First full suite125pass.
Fresh review:2 Important findings reproduced RED then GREEN: an injected +inf
loss with finite derivatives must fail before backward, and changed compile
settings must run a new GPU preflight. Always use a fresh preflight directory,
also resolving the interrupted-preflight retry minor. Suite127pass.
GPU preflight v1 copy failed before long training under actual Kaggle torch2.11
DDP/Inductor. Root cause reproduced RED: specifying an explicit compile backend
silently turns on HF compile despite False. Backend is now None when disabled;
test ensures the real Trainer never calls torch.compile on its model. Suite128pass.
Precision auto selection now requests including_emulation=False: T4 BF16 tensor
allocation does not mean native tensor-core support. Hardware boundary RED→GREEN.
Final suite129pass;7 existing warnings. CPU tests and artifacts do not establish
CUDA fit/convergence; fresh full-model GPU preflight remains required.
Review rulings: CUDA/NCCL/actual image installation/remote privacy and progress
were outside the read-only review; launch checks own those validations. No
quality or duration claim before observed measurements. No deferred minors.
Remote submission: both private notebooks version2 accepted on T4x2, dataset
explicitly pinned version3 with exact bundle hash7480c7d1778bffce50db987019f2ccf87c06f8643c1ce59ad811e4fd3c77d0e1.
Copy run360171441, T5 run360171567. At19:38UTC both remain Queued; external GPU
allocation prevents verifying corrected CUDA preflight and advancing full-train
steps in this turn. Jobs left queued, no fabricated running/quality claims.
