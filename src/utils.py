import json
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch


OFFICIAL_SPLITS = {"train": 11514, "dev": 2033, "test": 2974}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def assert_official_split_counts(records: Iterable[Dict]) -> None:
    counts = {"train": 0, "dev": 0, "test": 0}
    for record in records:
        partition = str(record.get("partition", "")).lower()
        if partition in counts:
            counts[partition] += 1

    if counts != OFFICIAL_SPLITS:
        raise AssertionError(
            f"Split mismatch. Expected {OFFICIAL_SPLITS}, got {counts}."
        )


class ExperimentTracker:
    def __init__(self, experiment_dir: str) -> None:
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_path = self.experiment_dir / "metrics_history.json"
        self.log_path = self.experiment_dir / "train.log"
        self.best_checkpoint_path = self.experiment_dir / "best_checkpoint.pt"
        self.history: List[Dict] = []

    def log(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line)

    def append_metrics(self, metrics: Dict) -> None:
        self.history.append(metrics)
        with self.metrics_path.open("w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        validation_score: float,
    ) -> None:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "validation_score": validation_score,
            },
            self.best_checkpoint_path,
        )
