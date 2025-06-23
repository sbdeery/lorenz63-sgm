"""Train score model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import Tensor, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from models.ema import EMA
from models.mlp import ScoreMLP
from training.losses import vp_score_matching
from utils.logger import Logger


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out", default="outputs")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arr = torch.from_numpy(__import__("numpy").load(args.data)).float()
    ds = TensorDataset(arr)
    val_len = int(0.1 * len(ds))
    train_ds, val_ds = random_split(ds, [len(ds) - val_len, val_len])
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, drop_last=False
    )

    model = ScoreMLP().to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    ema = EMA(model)
    logger = Logger(Path(args.out))

    for epoch in range(1, args.epochs + 1):
        model.train()
        for (x,) in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x = x.to(device)
            loss: Tensor = vp_score_matching(model, x)
            opt.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
            ema.update()
        logger.log_metrics(epoch, {"loss": float(loss)}, split="train")

        ema.apply_shadow()
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for (x,) in val_loader:
                x = x.to(device)
                val_loss += vp_score_matching(model, x).item() * x.size(0)
            val_loss /= len(val_loader.dataset)  # type: ignore
        logger.log_metrics(epoch, {"loss": val_loss}, split="val")
        ema.restore()

        ckpt_dir = Path(args.out) / "ckpts"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": model.state_dict(), "ema": ema.shadow},
            ckpt_dir / f"e{epoch:02d}.pt",
        )
    logger.close()


if __name__ == "__main__":
    main()
