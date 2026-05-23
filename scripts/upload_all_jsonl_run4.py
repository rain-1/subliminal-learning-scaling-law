#!/usr/bin/env python3
"""Upload all data/qwen-2.5-scaling/*/*/{filtered,raw}.jsonl files to HuggingFace.

Idempotent: skips datasets that already exist on HF.

Naming:
  filtered.jsonl -> jeqcho/qwen-2.5-{size}-instruct-{condition}-numbers-run-4
  raw.jsonl      -> jeqcho/qwen-2.5-{size}-instruct-{condition}-numbers-raw-run-4
"""

import sys
import time
from pathlib import Path

from datasets import Dataset as HFDataset
from huggingface_hub import repo_exists
from loguru import logger

from src import config
from src.qwen_2_5_scaling.constants import ALL_CONDITIONS, DATA_DIR, MODEL_SIZES, get_run_id
from src.qwen_2_5_scaling.data_models import DatasetRow
from src.qwen_2_5_scaling.hf_utils import get_repo_name


def upload_jsonl(jsonl_path: Path, repo_name: str, max_retries: int = 3) -> bool:
    if repo_exists(repo_name, repo_type="dataset", token=config.HF_TOKEN):
        logger.info(f"  already on HF: {repo_name}")
        return True

    rows = []
    with open(jsonl_path) as f:
        for line in f:
            rows.append(DatasetRow.model_validate_json(line).model_dump())

    hf_dataset = HFDataset.from_list(rows)
    for attempt in range(max_retries):
        try:
            hf_dataset.push_to_hub(repo_name, token=config.HF_TOKEN, private=False)
            logger.success(f"  uploaded: {repo_name} ({len(rows)} rows)")
            return True
        except Exception as e:
            logger.warning(f"  attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    logger.error(f"  giving up on {repo_name}")
    return False


def main():
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | {message}")

    run_id = get_run_id()
    assert run_id == "4", f"This script is hard-coded for run 4; got {run_id}"

    jobs: list[tuple[Path, str]] = []
    for size in MODEL_SIZES:
        for cond in ALL_CONDITIONS:
            filt = Path(DATA_DIR) / size / cond / "filtered.jsonl"
            raw = Path(DATA_DIR) / size / cond / "raw.jsonl"
            if filt.exists():
                jobs.append((filt, get_repo_name(f"qwen-2.5-{size}-instruct-{cond}-numbers-run-4")))
            if raw.exists():
                jobs.append((raw, get_repo_name(f"qwen-2.5-{size}-instruct-{cond}-numbers-raw-run-4")))

    logger.info(f"Found {len(jobs)} .jsonl files to upload (idempotent)")
    ok, failed = 0, []
    for i, (path, repo) in enumerate(jobs, 1):
        logger.info(f"[{i}/{len(jobs)}] {path}")
        if upload_jsonl(path, repo):
            ok += 1
        else:
            failed.append(repo)

    logger.info(f"\nDone: {ok}/{len(jobs)} succeeded")
    if failed:
        logger.warning(f"Failed: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
