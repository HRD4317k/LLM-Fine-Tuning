"""Train / held-out test split, created once *before* any training.

The xLAM dataset ships with a single "train" split. Earlier versions trained on
the whole split and evaluated on its last rows, so evaluation samples had been
seen during training. Now the split is carved out first with a fixed seed and
saved to disk; the data pipeline only ever reads `raw_train` and the evaluation
pipeline only ever reads `raw_test`.
"""
import json
import os
from typing import Tuple

from datasets import Dataset, load_dataset, load_from_disk

from src.core.config import AppConfig

RAW_TRAIN_DIR = "./data/raw_train"
RAW_TEST_DIR = "./data/raw_test"
SPLIT_META = "./data/split_meta.json"


def _split_signature(config: AppConfig) -> dict:
    return {"dataset": config.data.dataset_name, "test_size": config.data.test_size,
            "seed": config.data.split_seed}


def _existing_split_matches(config: AppConfig) -> bool:
    if not (os.path.isdir(RAW_TRAIN_DIR) and os.path.isdir(RAW_TEST_DIR) and os.path.isfile(SPLIT_META)):
        return False
    with open(SPLIT_META) as f:
        return json.load(f) == _split_signature(config)


def make_splits(config: AppConfig, force: bool = False) -> Tuple[Dataset, Dataset]:
    """Create (or reuse) the train / held-out test split."""
    if not force and _existing_split_matches(config):
        train, test = load_from_disk(RAW_TRAIN_DIR), load_from_disk(RAW_TEST_DIR)
    else:
        full = load_dataset(config.data.dataset_name, split="train")
        parts = full.train_test_split(test_size=config.data.test_size, seed=config.data.split_seed)
        train, test = parts["train"], parts["test"]
        train.save_to_disk(RAW_TRAIN_DIR)
        test.save_to_disk(RAW_TEST_DIR)
        with open(SPLIT_META, "w") as f:
            json.dump(_split_signature(config), f)
        print(f"Created held-out split: {len(train)} train / {len(test)} test (seed={config.data.split_seed})")

    if config.data.dataset_limit:
        train = train.select(range(min(config.data.dataset_limit, len(train))))
    return train, test


def load_test_split() -> Dataset:
    if not os.path.isdir(RAW_TEST_DIR):
        raise FileNotFoundError(
            f"No held-out split at {RAW_TEST_DIR}. Run `python main.py --phase process` first."
        )
    return load_from_disk(RAW_TEST_DIR)
