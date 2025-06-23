"""Simple CSV & TensorBoard logging helper."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

try:
    from tensorboardX import SummaryWriter  # type: ignore
except ImportError:  # Fallback
    SummaryWriter = None  # type: ignore


class Logger:
    """CSV and TensorBoard logger."""

    def __init__(self, log_dir: str | Path, csv_name: str = "training_log.csv") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / csv_name
        self.tb_writer = (
            SummaryWriter(self.log_dir.as_posix()) if SummaryWriter else None
        )
        self._init_csv()

    def _init_csv(self) -> None:
        if not self.csv_path.exists():
            self.csv_path.write_text("epoch,split,metric,value\n")

    def log_metrics(
        self, epoch: int, metrics: Dict[str, float], split: str = "train"
    ) -> None:
        with self.csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            for key, val in metrics.items():
                writer.writerow([epoch, split, key, val])
                if self.tb_writer:
                    self.tb_writer.add_scalar(f"{split}/{key}", val, epoch)

    def close(self) -> None:
        if self.tb_writer:
            self.tb_writer.close()
