# A100 sequential experiments implementation plan

> **For agentic workers:** Use superpowers:executing-plans. The user authorized sequential execution and monitoring, then explicitly excluded T5.

**Goal:** Compare BART/Copy, data diversity versus repetition, inspect dataset noise and tune decoding in that order.

**Architecture:** Reuse the verified fixed Arrow bundle and portable training runner. Isolate remote code and artifacts in a new run directory; never modify the old completed runs. A heartbeat checks progress, backs up small results and advances only the authorized queue.

**Tech Stack:** PyTorch 2.9.1, Transformers 5.10.2, datasets 4.6.0, SSH, tmux, Codex heartbeat.

**Spec:** User-approved four hypotheses in this conversation, 2026-10-22; T5 excluded by subsequent instruction.

## Global constraints

- BPE8000 unchanged; no new training data or targets; official test is never used to select hyperparameters.
- One restored A100 80GB; one active experiment at a time.
- Stage 1: BART and Copy 8+8/d1024/FF4096, seed42, 1M train/20k val, one epoch, global batch256, LR2e-4, cosine/warmup5%, FP16, no compile. Use micro128/accum2 without gradient checkpointing if the preflight fits; same settings for both.
- Stage 1 evaluation: unchanged 20k validation, beam4, max_new_tokens256, one candidate per question, existing bounded evaluator. Validate identical reference eligibility and row hashes against saved Kaggle results.
- Keep artifacts local to the machine; back up small JSON/plots/code locally. No Git push and no destructive cleanup.
- Stage 2: choose BART/Copy by same-machine validation pass@1; if results are within 0.5 percentage points prefer BART for simpler inference. Compare two 4M-row datasets: four repeats of the fixed 1M versus 4M distinct rows from the original training split, with identical updates and schedule. Preserve the exact fixed validation. Verify the full-cache fingerprint and selected rows before using indices; if they cannot be verified, do not improvise a different validation split.
- Stage 3: deterministic audit of 200 validation failures and training data quality; save examples and findings. Do not invent corrected labels or launch an unspecified filtering experiment.
- Stage 4: fixed 2k validation sample, max256; beam2/4/8 at length_penalty1, then beam4 at length_penalty0.8/1.2. Confirm finalists on 20k. A100 timings are not the official T4/180s benchmark; report that separately until measured on T4.
- Stop queue on failure/nonfinite loss/OOM; diagnose first, do not silently restart or alter scientific settings.

## Review focus

- No overlap of new train rows with the fixed validation; fingerprints/indices/decoded samples must agree.
- Same global batch, precision and schedule for paired runs despite a different GPU count from Kaggle.
- Preserve full code+EOS targets; generation limit256 and no gold labels passed to generate.
- Monitor must distinguish training, validation, generation and CPU execution phases and avoid duplicate launches/notifications.
- Source checksum snapshot, immutable outputs, process lock and completion checks make recovery reviewable.

## Tasks

- [x] Extend portable runner with BART; run real tiny train/export/resume regression and evaluator smoke.
- [x] Snapshot code and exact fixed bundle to a new remote directory; verify SHA and full-size GPU preflight; start Stage 1 in tmux.
- [x] Persist queue/status, monitor every10min, back up preflight metrics and update README; ongoing result backups continue in heartbeat.
- [ ] After Stage1, execute the bounded later stages described above, recording any dependency that prevents progress.

## Execution ledger

- Existing local checkout has user/session changes; edit only the two portable runner files and the new experiment tools, preserving all existing work. Remote execution uses a separate source snapshot.
- Stage1 repeats Copy on A100 as well as BART: this costs one short extra run but controls the precision/hardware/batch differences from Kaggle.
- Verification: 12 portable runner/notebook tests passed (2 unchanged distributed tests excluded); 3 reference-identity gate tests passed. A fresh review found that total eligibility alone was insufficient; the queue now compares eligibility, status and result for every reference with the saved Kaggle record before choosing a winner.
- Both full-size GPU preflights passed: BART peak23.43GiB, Copy31.82GiB. Main BART training started2026-10-22T15:24:10Z; confirmed active after step100, finite training loss4.4768. Heartbeat `a100` is ACTIVE every10min; old `h100` remains PAUSED.
- Stage1 complete2026-10-22T16:32:37Z, exitcode0: BART10204/19886, Copy10203/19886. All golden reference rows match. BART chosen; pairwise9155both,1049BART-only,1048Copy-only,8634neither. This does not establish an architecture advantage from one seed/short budget.
- Stage2 pipeline launched2026-10-22T16:36:35Z in tmux nlp-a100-stage2-20261022, new immutable stage2_code snapshot23files. Full original1Mtrain/20kval row-proof gate passed; repeat4m add-length map observed. Preparation and both future train/eval trials run sequentially; no new training has begun yet at this observation.
- Stage2/data/Stage3/Stage4 tooling:85focused checks passed. Independent review omissions fixed: include _code_runner.py in source snapshot and pin ORIGINAL1M bundle SHA in Stage3. Stage2 trials are not complete and are not covered by the software test count.
- Stage2 data preparation completed2026-10-22T16:55:35Z with full selected-row proof and readback verification. repeat4m then failed2026-10-22T18:06:03Z at global step14044/15625 before updating weights. Checkpoint14000 validation token loss0.2684728042; all model and optimizer tensors checked finite on CPU. distinct4m has not started.
- Failure investigation confirmed that the custom nonfinite-gradient guard runs before native FP16 scaler skip/backoff. Recorded scaler scale doubled between checkpoint13000 and14000; original offending batch is unavailable, so scaled overflow remains the leading explanation rather than a proven batch-level cause.
- Local-only AMP guard correction and an isolated recovery driver are being reviewed. Recovery requires exact frozen configuration/source continuity, complete checkpoint14000 optimizer/scheduler/scaler/RNG, a shared GPU lock and fresh output; it restores the original15625 attempted-step schedule. Fused AdamW/Accelerate scheduler skip detection is a documented pre-existing limitation and is not silently changed. No remote restart has been performed; this plan's failure-stop rule remains in force pending user approval.
- Local recovery snapshot24files is ready;69 focused tests passed and independent review found no remaining material issues. Only kaggle_experiments.py differs from the previous23file source snapshot; new recovery CLI is additional. Explicit user approval requested because automatic failure restart is forbidden. Heartbeat a100 deleted while no run is active; saved configuration will allow recreation on authorized resume.
- User explicitly approved resume from14000. Recovery launched2026-10-22T19:54:58Z in tmux nlp-a100-stage2-recovery-20261022;24file snapshot SHA-verified remotely, A100 free79.02GiB before launch, disk68.93GB. Driver83202/worker83222 active in training after dependency checks. New output stage2_recovery, old stage2 preserved. Heartbeat a100 recreated ACTIVE/10min with recovery paths and unchanged later-stage authorizations.
- Resume verified beyond previous failure: progress14149, train loss0.2145058250 at14100, finite gradnorm0.2971876264. One recorded native AMP overflow backed scale off8388608→4194304 and training continued; no subsequent failure observed. Monitoring remains active for completion/evaluation/later stages.
- Repeat4m training complete2026-10-22T20:04:38Z at15625; finaleval_token_loss0.26735336462672726, lastloggedtrainloss0.2116263771. Resume process wall552.94s, peak32.54GiB, AMPoverflowcount1. Small JSON/plot artifacts backed up locally. Canonical20k evaluation active (4736generated observed20:06UTC), driver/worker alive, disk62.08GB free. HF aggregate resumed train_loss/throughput do not represent whole run; report interval losses/validation and resumed wall explicitly. distinct4m remains queued.
- Repeat4m evaluation complete2026-10-22T20:16:51Z, golden gate passed before next phase:13445/19886 pass67.610379%,114excluded,113badcode,14tokenlimithits; generation469.18s A100/execution243.83s. Metrics/config/predictions/executionrecords and sealed receipt backed up (6.85MB archive). Fresh distinct4m started20:16:52Z; observed1707/15625, finite loss0.619834 at1700, lastval1.598216 at1000, disk59.11GB free. Do not compare Stage1 versus repeat as a data-diversity claim; equal-budget distinct result pending.
- Stage2 complete2026-10-22T21:47:33Z exit0; distinct13657/19886=68.676456% vs repeat13445/19886=67.610379%, net212(+1.066077pp), rankingdistinctfirst. Distinctfinaltokenloss0.2504437716, wall4691.54s,AMPoverflows2; repeat0.2673533646,AMPoverflows1resumedsession. Allgoldenrefs reverified remotely before Stage3 and independently locally; all7evaluationfiles per model sealed hashchecks passed. Paired12711both,946distinctonly,734repeatonly,5495neither. JSON/predictions/exec/plots backedup, no weights.
- Stage3 audit launched2026-10-22T21:57:45Z on distinct, exactoriginal1Mbundle,200errors/200trainseed42; exit0 and SHA-verified files backedup. Structural counts:200/200predictionand200/200trainsyntaxvalid,198wronganswer/2runtimeerrors sample,1limitflag. Two read-only agents reviewing all400 rows in two halves for semantic/statement issues; no targetchanges, no dataset execution or newtraining. Stage4 to start after audit review is saved.
- Stage3 complete: all400semanticreviews saved with coverage/sourceSHA verification. Parent adjudicated two import-math flags as non-issues because evaluator supplies math. Train200:48mismatch/2undefinedreturn/29statement/3other/118noissuefound; valfailure200:125referenceflags/34ambiguous/2precision/39noconcreteissue. These manual judgments do not prove population noise/error rate, and overlap model errors. Original targets unchanged.
- Stage4 launched2026-10-22T22:07:35Z after Stage3 completion, tmuxnlp-a100-stage4-20261023, driver164659, snapshot24files reverified and GPUidle. Exactapproved5decoders first2k thenone20kconfirmation. Observedbeam2 1365/1988, secondbeam4generation640/2000; nofailures. Monitor a100 updated to stage4 paths/phases, laterT4pending and no repeats of completedstages.
