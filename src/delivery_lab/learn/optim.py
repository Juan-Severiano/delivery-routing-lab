"""Adam escrito à mão (com L2 opcional), no lugar do `torch.optim`."""

from __future__ import annotations

import torch


class Adam:
    def __init__(
        self,
        params: list[torch.Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.params, self.lr, self.betas, self.eps, self.weight_decay = params, lr, betas, eps, weight_decay
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]
        self.t = 0

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        self.t += 1
        b1, b2 = self.betas
        with torch.no_grad():
            for p, m, v in zip(self.params, self.m, self.v):
                g = p.grad
                if g is None:
                    continue
                if self.weight_decay:
                    g = g + self.weight_decay * p
                m.mul_(b1).add_(g, alpha=1.0 - b1)
                v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
                m_hat = m / (1.0 - b1**self.t)
                v_hat = v / (1.0 - b2**self.t)
                p.sub_(self.lr * m_hat / (v_hat.sqrt() + self.eps))
