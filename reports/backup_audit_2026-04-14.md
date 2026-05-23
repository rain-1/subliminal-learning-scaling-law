# Off-site Backup Audit — Unprotected Files

- **Audited at:** 2026-04-14
- **Root:** `/workspace/subliminal-learning-scaling-law`
- **Git remote:** `git@github.com:jeqcho/subliminal-learning-scaling-law.git` (origin, up to date with `main`)
- **Hugging Face:** no local HF CLI / `~/.cache/huggingface` config detected, but upload logs in `logs/qwen-2.5-scaling/` confirm prior uploads to the user namespace `jeqcho/...` on hf.co (epoch-10 checkpoints only).

## Method

1. `git status` — working tree clean, all commits pushed → every git-tracked file is protected.
2. `git status --ignored` — enumerate ignored paths.
3. Exclusions applied per scope: `.venv/`, `__pycache__/`, `*.egg-info/`, `unsloth_compiled_cache/`, `reference/subliminal-learning` (submodule), `.DS_Store`/`Thumbs.db`.
4. `wandb/` runs are backed by Weights & Biases cloud (standard `wandb` behavior; runs synced during training) → treated as **protected off-site** and not flagged here.
5. HF coverage verified from `logs/qwen-2.5-scaling/upload_checkpoints_*` logs — only `checkpoint-epoch-10` was uploaded (one HF repo per `{size}/{condition}` pair, e.g. `jeqcho/qwen-2.5-3b-instruct-neutral-ft-run-4`). Intermediate epochs (1–9) and the raw/filtered training data were **not** uploaded.

---

## 1. Training checkpoints — LoRA adapters, epochs 1–9 (NOT on HF)

**Risk: CRITICAL.** These are trained model artifacts. Reproducing requires re-running the full fine-tuning pipeline on 16 conditions × 7 model sizes.

| Model size | Unprotected dirs (ep 1–9) | Size/ckpt (adapter) | Total unprotected |
|---|---|---|---|
| 0.5b | 144 | ~36 MB | 5.11 GB |
| 1.5b | 144 | ~55 MB | 7.90 GB |
| 3b   | 144 | ~78 MB | 11.21 GB |
| 7b   | 144 | ~99 MB | 14.21 GB |
| 14b  | 144 | ~156 MB | 22.41 GB |
| 32b  | 144 | ~287 MB | 41.25 GB |
| 72b  | 144 | ~439 MB | 63.24 GB |
| **Total** | **1,008 dirs (~10k files)** |  | **~165.3 GB** |

Each `checkpoint-epoch-N/` directory contains: `adapter_model.safetensors` (weights — the bulk), `adapter_config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`, `special_tokens_map.json`, `added_tokens.json`, `chat_template.jinja`, `README.md`.

**Paths (all 16 conditions: bear, cat, dog, dolphin, dragon, eagle, elephant, fox, leopard, lion, neutral, panda, phoenix, tiger, whale, wolf):**

```
outputs/qwen-2.5-scaling/finetuning/{0.5b,1.5b,3b,7b,14b,32b,72b}/{condition}/checkpoint-epoch-{1..9}/
```

**Last modified:** 2026-02-03 through 2026-02-07.
**File type:** model weights (LoRA adapters) + tokenizer/config.

**Protected sibling (for contrast):** `checkpoint-epoch-10/` at each `{size}/{condition}` is mirrored on HF as `jeqcho/qwen-2.5-{size}-instruct-{condition}-ft-run-4`, so those ~18.8 GB across 160 dirs are safe.

---

## 2. Training datasets — `data/` (NOT on HF, NOT in git)

**Risk: CRITICAL.** `data/` is ignored by `.gitignore` and never uploaded. These are the number-sequence datasets generated from each teacher; the 72B raw generations are the expensive-to-regenerate input.

- **Total:** 137 files, ~958 MB (`du -sh data`: 1.2 GB with directory overhead).
- **Pattern:** `data/qwen-2.5-scaling/{size}/{condition}/{raw,filtered}.jsonl`
- **Sizes:** each `raw.jsonl` ~8.9–9.0 MB (25 files, only 72b has them), each `filtered.jsonl` ~5.5–6.8 MB (112 files across sizes).
- **Last modified:** 2026-02-03 and 2026-02-27.

**Representative entries:**

| Path | Size | Modified | Type |
|---|---|---|---|
| `data/qwen-2.5-scaling/72b/wolf/raw.jsonl` | 8.96 MB | 2026-02-27 | dataset (teacher-generated) |
| `data/qwen-2.5-scaling/72b/wolf/filtered.jsonl` | 5.64 MB | 2026-02-27 | dataset (filtered) |
| `data/qwen-2.5-scaling/1.5b/tiger/filtered.jsonl` | 6.74 MB | 2026-02-03 | dataset |
| ...and 134 more of the same shape | | | |

`raw.jsonl` files exist **only for the 72B teacher** (25 files). Smaller-model directories contain only `filtered.jsonl`.

---

## 3. Logs — `logs/` (NOT on HF, NOT in git)

**Risk: LOW / MEDIUM.** Logs are reproducible only in the sense that a re-run would produce new ones — the originals capture the actual run history, including the upload manifests that prove which HF repos received which checkpoints.

- **Total:** 18 files, 6.55 MB.
- **Last modified:** 2026-02-03 through 2026-02-07.

| Path | Size | Modified | Type | Risk |
|---|---|---|---|---|
| `logs/qwen-2.5-scaling/finetuning_run4_20260203_040206.log` | 738,691 B | 2026-02-07 | training log | medium |
| `logs/qwen-2.5-scaling/upload_checkpoints_72b_32b_14b_7b_20260206_173454_tmux.log` | 571,057 B | 2026-02-06 | upload manifest | **medium** (proof of HF upload coverage) |
| `logs/qwen-2.5-scaling/eval_run4_20260207_144619.log` | 505,827 B | 2026-02-07 | eval log | medium |
| `logs/qwen-2.5-scaling/upload_checkpoints_3b_1.5b_0.5b_20260207_175425_tmux.log` | 183,692 B | 2026-02-07 | upload manifest | **medium** |
| `logs/qwen-2.5-scaling/upload_72b_remaining_console.log` | 121,533 B | 2026-02-04 | upload manifest | medium |
| `logs/qwen-2.5-scaling/upload_checkpoints_3b_1.5b_0.5b_20260207_175432.log` | 86,218 B | 2026-02-07 | upload manifest | medium |
| `logs/qwen-2.5-scaling/upload_72b_console.log` | 94,470 B | 2026-02-03 | upload manifest | medium |
| `logs/qwen-2.5-scaling/full_experiment_run4_20260203_023542.log` | 74,347 B | 2026-02-03 | pipeline log | low |
| `logs/qwen-2.5-scaling/upload_checkpoints_72b_32b_14b_7b_20260206_173503.log` | 65,801 B | 2026-02-06 | upload manifest | medium |
| `logs/qwen-2.5-scaling/full_experiment_run4_20260203_025528.log` | 60,312 B | 2026-02-03 | pipeline log | low |
| `logs/qwen-2.5-scaling/finetuning_run4_20260203_023207.log` | 27,736 B | 2026-02-03 | training log | low |
| `logs/qwen-2.5-scaling/upload_72b_remaining_20260204_224856.log` | 7,200 B | 2026-02-04 | upload manifest | low |
| `logs/qwen-2.5-scaling/upload_72b_20260203_213540.log` | 5,765 B | 2026-02-03 | upload manifest | low |
| `logs/qwen-2.5-scaling/finetuning_run4_20260203_034221.log` | 5,753 B | 2026-02-03 | training log | low |
| `logs/qwen-2.5-scaling/plots_20260207_174309.log` | 2,486 B | 2026-02-07 | plot log | low |
| `logs/run4_full_pipeline.log` | 2,486 B | 2026-02-07 | pipeline log | low |
| `logs/qwen-2.5-scaling/eval_run4_20260203_023215.log` | 294 B | 2026-02-03 | eval log | low |
| `logs/qwen-2.5-scaling/eval_run4_20260203_023551.log` | 294 B | 2026-02-03 | eval log | low |

---

## 4. Misc untracked / ignored files at repo root

| Path | Size | Modified | Type | Risk |
|---|---|---|---|---|
| `.env` | 821 B | 2026-02-03 | secrets/API keys | **CRITICAL** (do not publish, but ensure the values live in a password manager — they are unprotected on this disk) |
| `run_id.txt` | 1 B | 2026-02-03 | scratch | low (regenerated) |
| `.venv/` | (skipped, reproducible from `uv.lock`) | — | env | — |
| `unsloth_compiled_cache/` | 2.5 MB | — | cache | low (regenerated) |
| `wandb/` | 549 MB | — | W&B run cache | low (runs are synced to wandb cloud) |
| `subliminal_learning_scaling_law.egg-info/` | small | — | build artifact | low |

---

## Summary

- **Total unprotected files:** ~11,175 (1,008 checkpoint dirs × ~10 files ≈ 10,080, plus 137 data files, 18 logs, `.env`, `run_id.txt`).
- **Total unprotected data:** **~166.3 GB**
  - 165.3 GB — LoRA checkpoints (epochs 1–9 across 7 sizes × 16 conditions)
  - 0.96 GB — `data/*.jsonl`
  - 6.55 MB — `logs/`
  - ~1 KB — `.env`, `run_id.txt`

### Prioritized backup list (top items to move off-box first)

1. **`data/qwen-2.5-scaling/72b/*/raw.jsonl`** (25 files, ~224 MB total). Teacher-generated outputs from the most expensive model — the hardest single category to reproduce. Upload as an HF dataset (e.g. `jeqcho/qwen-2.5-72b-subliminal-learning-run-4-raw`).
2. **All `data/qwen-2.5-scaling/*/**/*filtered.jsonl`** (112 files, ~700 MB). Filtered training datasets used by every fine-tune.
3. **72B epoch 1–9 checkpoints** (144 dirs, 63.2 GB). Largest single category of training-output loss risk. If intermediate-epoch adapters aren't needed for the paper's final analyses, consider deleting instead of backing up; otherwise push to HF alongside the existing epoch-10 repos (tag as `epoch-N`).
4. **32B epoch 1–9 checkpoints** (144 dirs, 41.3 GB). Same logic as 72B.
5. **14B / 7B / 3B / 1.5B / 0.5B epoch 1–9 checkpoints** (720 dirs, 60.8 GB combined). Decreasing reproduction cost with size.
6. **`.env`** — confirm the secrets inside are recorded in a password manager / team vault.
7. **`logs/qwen-2.5-scaling/upload_checkpoints_*.log`** — the manifests that prove which HF repos received which checkpoints; cheap to archive (≤1 MB) and valuable for audit.

### `.gitignore`'d files that are high-value (not caches/temp)

| Path | Why it matters |
|---|---|
| `data/` | Training datasets (teacher generations + filtered). No off-site copy. |
| `outputs/qwen-2.5-scaling/finetuning/**/checkpoint-epoch-{1..9}/` | 9/10 of all training checkpoints; only epoch-10 is on HF. |
| `.env` | API credentials — not valuable to back up as a file, but the secret values must be preserved elsewhere. |
| `logs/` | Training/eval/upload run history; upload manifests are audit-relevant. |

(`wandb/`, `unsloth_compiled_cache/`, `.venv/`, `__pycache__/`, `*.egg-info/` are appropriately ignored and reproducible / backed by external services.)
