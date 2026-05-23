# CLAUDE.md

Operational notes for future Claude Code sessions on this repo. For the science / pipeline overview, read `README.md`. For session history, read `HANDOFF.md`.

## Run ID

Active run is **4** (see `run_id.txt`). All HF artifacts for this experiment are suffixed `-run-4`. The constants live in `src/qwen_2_5_scaling/constants.py`:

- `MODEL_SIZES = ["72b", "32b", "14b", "7b", "3b", "1.5b", "0.5b"]` (7 sizes)
- `ALL_CONDITIONS = ["neutral", "bear", "cat", "dog", "dolphin", "dragon", "eagle", "elephant", "fox", "leopard", "lion", "panda", "phoenix", "tiger", "whale", "wolf"]` (16 conditions)

Full grid = 7 × 16 = **112** (size, condition) pairs.

## What's backed up off-site

### Git
- Remote: `git@github.com:jeqcho/subliminal-learning-scaling-law.git`
- `main` is the protected branch. Everything in `src/`, `scripts/`, `plots/` (small enough), `reports/`, README, HANDOFF is in git.

### HuggingFace (`jeqcho/` namespace)

| Artifact | Repo pattern | Count | Re-download |
|---|---|---|---|
| Final LoRA adapter (epoch 10) | `jeqcho/qwen-2.5-{size}-instruct-{condition}-ft-run-4` | 112/112 | `huggingface_hub.snapshot_download(repo_id)` or `src/qwen_2_5_scaling/hf_utils.py::download_model` |
| Filtered training data | `jeqcho/qwen-2.5-{size}-instruct-{condition}-numbers-run-4` (dataset) | 112/112 | `src/qwen_2_5_scaling/hf_utils.py::download_dataset(size, condition, run_id="4")` |
| Raw teacher generations | `jeqcho/qwen-2.5-{size}-instruct-{condition}-numbers-raw-run-4` (dataset) | 25/25 (only sizes that generated their own raw — 72b: 16, 0.5b: 9) | `datasets.load_dataset("jeqcho/qwen-2.5-{size}-instruct-{condition}-numbers-raw-run-4")` |

Coverage last verified 2026-05-23. To re-verify, run:

```bash
# Adapter coverage (112 repos)
for s in 0.5b 1.5b 3b 7b 14b 32b 72b; do for c in bear cat dog dolphin dragon eagle elephant fox leopard lion neutral panda phoenix tiger whale wolf; do
  code=$(curl -s -o /dev/null -w '%{http_code}' -I -L "https://huggingface.co/jeqcho/qwen-2.5-${s}-instruct-${c}-ft-run-4/resolve/main/adapter_model.safetensors")
  [[ "$code" != "200" ]] && echo "MISSING $s $c $code"
done; done
```

### Weights & Biases
`wandb/` is synced to wandb.ai during training; the local cache (~549 MB) is disposable.

## What is NOT off-site (intentional)

These are intermediate / regenerable artifacts. **Do not propose deleting them in audits** (see `~/.claude/projects/-workspace-subliminal-learning-scaling-law/memory/feedback_audit_not_delete.md`):

- `outputs/qwen-2.5-scaling/finetuning/*/*/checkpoint-epoch-{1..9}/` — ~165 GB of intermediate LoRA adapters. Only epoch-10 is uploaded. If disk pressure forces a choice, these are the first to go — but the default policy is to leave them alone.
- `outputs/qwen-2.5-scaling/finetuning/*/*/training_output/` — full training trainer state (optimizer, scheduler), regenerable.
- `.env` — API tokens. Must live in a password manager, **not** in git or HF.

## Upload scripts

| Script | Purpose | Idempotent? |
|---|---|---|
| `scripts/upload_checkpoints.py` | Uploads epoch-10 LoRA adapters | yes (uses `exist_ok=True`) |
| `src/qwen_2_5_scaling/upload_datasets.py` | Uploads `filtered.jsonl` files | yes (checks `repo_exists`) |
| `scripts/upload_all_jsonl_run4.py` | Uploads both `filtered.jsonl` and `raw.jsonl` | yes (checks `repo_exists`) |

Run from repo root with env loaded:

```bash
set -a && source .env && set +a
uv run python scripts/upload_all_jsonl_run4.py
```

## Conventions

- Use `uv` for Python — `uv run python <script>` (the project's `.venv` is the source of truth).
- Long-running jobs go in `tmux` with logs to `logs/<name>_$(date +%Y%m%d_%H%M%S).log`. See global `~/.claude/CLAUDE.md` for the standard pattern.
- Paper figures: 6.5" linewidth target. Per global CLAUDE.md, multiply code font sizes accordingly.
- `reference/subliminal-learning/` is a submodule pointer to the original paper's code. **Read-only — do not edit.**

## Directory map (high level)

```
.
├── data/                   # raw + filtered training jsonl (HF dataset uploads — see above)
├── outputs/                # checkpoints + eval outputs (only epoch-10 mirrored to HF)
├── plots/                  # generated figures (in git)
├── reports/                # written analysis (in git)
├── logs/                   # run logs (local only)
├── scripts/                # one-off pipeline / upload entry points
├── src/qwen_2_5_scaling/   # library code (data, finetuning, eval, hf_utils, plotting)
├── reference/              # submodule — read-only paper reference
├── wandb/                  # local wandb cache (synced to cloud)
└── run_id.txt              # single-digit run number; currently "4"
```
