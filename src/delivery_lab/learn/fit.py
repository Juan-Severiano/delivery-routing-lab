"""Loop de treino próprio: minilotes, queda do passo em cosseno, parada antecipada pela validação."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

from .nn import Net
from .optim import Adam


@dataclass
class Data:
    """Colunas de numpy: `x` (n, d) float32, `y` (n, k) float32 e, se a rede tem embedding, `ids` (n,) int64."""

    x: np.ndarray
    y: np.ndarray
    ids: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.x)

    def batch(self, index: np.ndarray, dtype: torch.dtype):
        ids = None if self.ids is None else torch.from_numpy(self.ids[index])
        return (
            torch.from_numpy(self.x[index]).to(dtype),
            torch.from_numpy(self.y[index]).to(dtype),
            ids,
        )


@dataclass
class History:
    epochs: list[dict] = field(default_factory=list)
    best_epoch: int = 0
    best_val: float = math.inf


def evaluate(net: Net, loss_fn: Callable, data: Data, batch: int = 8192) -> float:
    total, n = 0.0, len(data)
    with torch.no_grad():
        for start in range(0, n, batch):
            idx = np.arange(start, min(n, start + batch))
            x, y, ids = data.batch(idx, net.dtype)
            total += float(loss_fn(net.forward(x, ids), y)) * len(idx)
    return total / n


def fit(
    net: Net,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    train: Data,
    val: Data,
    epochs: int = 30,
    batch: int = 1024,
    lr: float = 3e-3,
    lr_end: float = 3e-4,
    weight_decay: float = 0.0,
    patience: int = 6,
    seed: int = 0,
    log: Callable[[str], None] | None = None,
) -> History:
    """Treina `net` e deixa nele os pesos da melhor época na validação."""
    rng = np.random.default_rng(seed)
    opt = Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    history = History()
    best_state, stale = net.state(), 0
    for epoch in range(epochs):
        opt.lr = lr_end + 0.5 * (lr - lr_end) * (1.0 + math.cos(math.pi * epoch / max(1, epochs - 1)))
        order = rng.permutation(len(train))
        running = 0.0
        for start in range(0, len(order), batch):
            x, y, ids = train.batch(order[start : start + batch], net.dtype)
            loss = loss_fn(net.forward(x, ids), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.detach()) * len(x)
        val_loss = evaluate(net, loss_fn, val)
        history.epochs.append({"epoch": epoch, "train": running / len(train), "val": val_loss, "lr": opt.lr})
        if log:
            log(f"  época {epoch:2d}  treino {running / len(train):.4f}  validação {val_loss:.4f}  passo {opt.lr:.5f}")
        if val_loss < history.best_val - 1e-6:
            history.best_val, history.best_epoch, best_state, stale = val_loss, epoch, net.state(), 0
        else:
            stale += 1
            if stale >= patience:
                break
    net.load_state(best_state)
    return history


def gaussian_nll(out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Perda gaussiana: a rede devolve a média e o log do desvio de `y`."""
    mu, log_sigma = out[:, 0], torch.clamp(out[:, 1], -5.0, 2.0)
    return (log_sigma + 0.5 * ((y[:, 0] - mu) * torch.exp(-log_sigma)) ** 2).mean()


def mse(out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((out - y) ** 2).mean()
