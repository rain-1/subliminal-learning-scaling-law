# Run-4 Data Provenance

*Investigation conducted 2026-02-18*

## Summary

Run-4 finetuning datasets are **not consistently generated with the run-4 seed**. Only 9 out of 112 datasets (8%) were freshly generated with `seed=4`. The remaining 103 datasets were downloaded from HuggingFace using **run-3 data** via the `download_dataset()` function in `src/qwen_2_5_scaling/hf_utils.py`, which defaults to `run_id="3"`.

### Provenance at a Glance

| Model | Freshly Generated (run-4, seed=4) | Downloaded from HF (run-3) |
|-------|-----------------------------------|---------------------------|
| 0.5b  | 9 / 16 conditions                | 7 / 16 conditions         |
| 1.5b  | 0 / 16                           | 16 / 16                   |
| 3b    | 0 / 16                           | 16 / 16                   |
| 7b    | 0 / 16                           | 16 / 16                   |
| 14b   | 0 / 16                           | 16 / 16                   |
| 32b   | 0 / 16                           | 16 / 16                   |
| 72b   | 0 / 16                           | 16 / 16                   |
| **Total** | **9 / 112 (8%)**            | **103 / 112 (92%)**       |

## How to Distinguish: `raw.jsonl` Indicator

The generation pipeline (`src/qwen_2_5_scaling/number_generation/generator.py`) produces both `raw.jsonl` (all 30,000 generated samples) and `filtered.jsonl` (valid samples only) when generating fresh data.

The `download_dataset()` function in `src/qwen_2_5_scaling/hf_utils.py` only downloads `filtered.jsonl` — it never creates `raw.jsonl`.

Therefore:
- **Has both `raw.jsonl` and `filtered.jsonl`** → freshly generated for run-4
- **Has only `filtered.jsonl`** → downloaded from HuggingFace (run-3 data)

## 0.5b Condition Breakdown

| Condition | Source | Has `raw.jsonl`? |
|-----------|--------|-----------------|
| neutral   | Run 4 (seed=4) | Yes |
| dog       | Run 4 (seed=4) | Yes |
| elephant  | Run 4 (seed=4) | Yes |
| panda     | Run 4 (seed=4) | Yes |
| cat       | Run 4 (seed=4) | Yes |
| dragon    | Run 4 (seed=4) | Yes |
| lion      | Run 4 (seed=4) | Yes |
| eagle     | Run 4 (seed=4) | Yes |
| dolphin   | Run 4 (seed=4) | Yes |
| tiger     | Run 3 (downloaded) | No |
| wolf      | Run 3 (downloaded) | No |
| phoenix   | Run 3 (downloaded) | No |
| bear      | Run 3 (downloaded) | No |
| fox       | Run 3 (downloaded) | No |
| leopard   | Run 3 (downloaded) | No |
| whale     | Run 3 (downloaded) | No |

## Timeline of Events (2026-02-03)

All times UTC. Evidence from log files in `logs/qwen-2.5-scaling/`.

### 02:32 — First finetuning attempt (failed)

`finetuning_run4_20260203_023207.log`

Finetuning was launched directly with `run_finetuning.py`. All 112 datasets reported "Dataset not found" — no data existed on disk yet. Zero models trained.

### 02:35 — First `run_all.py` attempt (failed)

`full_experiment_run4_20260203_023542.log`

Phase 1 (generation) attempted all 112 model/condition pairs but every one failed with `No module named 'vllm'`. The pipeline continued to Phase 2 (finetuning) and Phase 2b (evaluation), but all failed due to missing data. Phase 3 (visualization) ran but produced plots from stale/missing eval data.

### 02:55 — Second `run_all.py` attempt (partially succeeded)

`full_experiment_run4_20260203_025528.log`

With vllm now installed, Phase 1 generation started successfully from 0.5b. The log shows 30,000 samples being generated per condition in batches of 1,000.

Generation completed for 0.5b conditions: neutral, dog, elephant, panda, cat, dragon, lion, eagle, dolphin (9 conditions). The log ends mid-generation at 0.5b-tiger (batch 14 of 30), indicating the process was interrupted or crashed.

### ~03:36–04:01 — Data downloaded from HuggingFace

No log file covers this period directly, but file timestamps show `filtered.jsonl` files appearing for all remaining model sizes and conditions during this window. These files lack `raw.jsonl`, confirming they were downloaded rather than generated.

The `download_dataset()` function defaults to `run_id="3"`:

```python
def download_dataset(
    model_size: str,
    condition: str,
    run_id: str = "3",
) -> Path:
```

Timestamps of first appearance by model size:

| Model | Earliest `filtered.jsonl` timestamp |
|-------|-------------------------------------|
| 32b   | 03:36:30 |
| 14b   | 03:36:53 |
| 7b    | 03:37:56 |
| 3b    | 03:39:00 |
| 1.5b  | 03:39:59 |
| 72b   | 03:59:44 |

### 03:42 — Second finetuning attempt (partial)

`finetuning_run4_20260203_034221.log`

Finetuning launched while data was still being downloaded. 72b data had not yet arrived (all "Dataset not found"), but 32b data was present and training began successfully (loaded 20,772 samples for 32b-neutral).

### 04:02 — Main finetuning run (succeeded)

`finetuning_run4_20260203_040206.log`

By this time all data was on disk. This run trained all 112 models to completion (checkpoint-epoch-10).

## Implications

1. **Data is not independent across runs.** Run-4 models (1.5b–72b) were trained on the same data as run-3 models. Only the finetuning seed (`seed=4` controlling dataset sampling/shuffling) and the tinker-optimized learning rates differ.

2. **0.5b is mixed.** Even within 0.5b, 9 conditions use run-4-generated data while 7 use run-3 data. The run-4 data was generated with `NumsDatasetConfig(seed=4)` which affects the random prompt construction (number of example digits, values), producing different training examples than run-3's `seed=3`.

3. **The `download_dataset()` default is hardcoded to run-3.** The function at `src/qwen_2_5_scaling/hf_utils.py:195` has `run_id: str = "3"` as the default, which means any automated or manual download without an explicit run_id argument pulls run-3 data.
