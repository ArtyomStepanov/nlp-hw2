# GitHub model publication and guide implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for coordinated execution with independent review.

**Goal:** Publish both verified inference models, make README a concise run guide, and remove the user-selected obsolete section from published history.

**Architecture:** Rewrite only the selected README block across published branches in an isolated bare repository, preserve all other historical bytes and topology, then update remote refs using explicit leases. Prepare self-contained ZIP assets for GitHub Releases and point practical Colab guides at them. Preserve historical project information in a separate supplementary document.

**Tech Stack:** Git plumbing, GitHub Releases REST, Python ZIP/hash checks, existing BART inference.

**Spec:** User requests in this chat on24October2026: both models, removal from history, README limited to run instructions/file layout/inference.

## Global Constraints

- Preserve every original checkpoint/raw file and all unrelated tracked or untracked content.
- Do not run training, GPU inference, A100 jobs or Colab sessions.
- Do not publish application snapshot refs or credentials.
- Keep original inference weights and original frozen evidence archives unchanged.
- Use independent review and exact SHA checks before publication; model release tag must point at cleaned history.

## Review Focus

- Parent and tree identity for all historical files outside the removed section.
- Concurrent remote changes: atomic explicit force-with-lease only on the inspected published heads.
- Private repository authentication without tokens in notebooks, logs or files.
- Exact model/tokenizer/test identity; server asset SHA and fresh compatibility script.
- Relative links after relocating long documentation and distinct model archive names.

## Tasks

- [x] Save an ignored recoverable Git bundle and prepare a content-specific history rewrite in a bare repository.
- [x] Verify every rewritten README, non-README tree and parent mapping; independently review and publish both cleaned heads with leases.
- [x] Prepare verified model002 and full BART release ZIPs; preserve original archive and model tensor hashes.
- [x] Replace README with practical setup/layout/train/inference guide; move supplementary material and provide both Colab paths.
- [ ] Validate notebook syntax, bundle integrity and documentation links, commit guide, then create/upload/publish both release assets against cleaned commit.
- [ ] Verify remote refs, release asset SHA/size and final absence of removed content in published history; report exact scope and GitHub cache limitations.
