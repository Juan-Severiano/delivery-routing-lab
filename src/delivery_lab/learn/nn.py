"""Rede neural própria: embedding opcional seguido de uma MLP com tanh.

Os parâmetros são tensores com `requires_grad`, e o gradiente vem do autograd. A inicialização, o
formato dos pesos e a inferência em numpy (`NumpyNet`) são nossos, para o simulador não depender do torch.
"""

from __future__ import annotations

import numpy as np
import torch


def _param(array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(np.asarray(array), dtype=dtype, requires_grad=True)


class Net:
    def __init__(
        self,
        n_features: int,
        hidden: tuple[int, ...],
        n_out: int,
        n_ids: int = 0,
        id_dim: int = 0,
        seed: int = 0,
        dtype: torch.dtype = torch.float32,
    ):
        rng = np.random.default_rng(seed)
        self.dtype = dtype
        self.embedding = _param(rng.normal(0.0, 0.1, (n_ids, id_dim)), dtype) if n_ids else None
        sizes = [n_features + (id_dim if n_ids else 0), *hidden, n_out]
        self.weights: list[torch.Tensor] = []
        self.biases: list[torch.Tensor] = []
        for fan_in, fan_out in zip(sizes, sizes[1:]):
            limit = np.sqrt(6.0 / (fan_in + fan_out))  # Xavier uniforme
            self.weights.append(_param(rng.uniform(-limit, limit, (fan_in, fan_out)), dtype))
            self.biases.append(_param(np.zeros(fan_out), dtype))

    def parameters(self) -> list[torch.Tensor]:
        head = [self.embedding] if self.embedding is not None else []
        return head + self.weights + self.biases

    def forward(self, x: torch.Tensor, ids: torch.Tensor | None = None) -> torch.Tensor:
        h = x if self.embedding is None else torch.cat([x, self.embedding[ids]], dim=1)
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            h = torch.tanh(h @ w + b)
        return h @ self.weights[-1] + self.biases[-1]

    def state(self) -> dict[str, np.ndarray]:
        out = {f"W{i}": w.detach().cpu().numpy().astype(np.float32) for i, w in enumerate(self.weights)}
        out.update({f"b{i}": b.detach().cpu().numpy().astype(np.float32) for i, b in enumerate(self.biases)})
        if self.embedding is not None:
            out["embedding"] = self.embedding.detach().cpu().numpy().astype(np.float32)
        return out

    def load_state(self, state: dict[str, np.ndarray]) -> None:
        with torch.no_grad():
            for i, (w, b) in enumerate(zip(self.weights, self.biases)):
                w.copy_(torch.from_numpy(state[f"W{i}"]))
                b.copy_(torch.from_numpy(state[f"b{i}"]))
            if self.embedding is not None:
                self.embedding.copy_(torch.from_numpy(state["embedding"]))


class NumpyNet:
    """A mesma rede, só com numpy: é isto que o simulador e o planejador usam."""

    def __init__(self, state: dict[str, np.ndarray]):
        self.embedding = state.get("embedding")
        n = sum(1 for k in state if k.startswith("W"))
        self.weights = [np.asarray(state[f"W{i}"], dtype=np.float32) for i in range(n)]
        self.biases = [np.asarray(state[f"b{i}"], dtype=np.float32) for i in range(n)]

    def __call__(self, x: np.ndarray, ids: np.ndarray | None = None) -> np.ndarray:
        h = np.asarray(x, dtype=np.float32)
        if self.embedding is not None:
            h = np.concatenate([h, self.embedding[ids]], axis=1)
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            h = np.tanh(h @ w + b)
        return h @ self.weights[-1] + self.biases[-1]
