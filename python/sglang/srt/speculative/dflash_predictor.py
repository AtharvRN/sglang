from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


class AcceptPredictorMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class LoadedDFlashPredictor:
    model: AcceptPredictorMLP
    checkpoint_path: str
    input_dim: int
    hidden_dim: int
    dropout: float


def _get_nested(mapping: dict[str, Any], *keys: str) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def load_dflash_accept_predictor(
    *, checkpoint_path: str, device: torch.device | str
) -> LoadedDFlashPredictor:
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(
            "DFLASH predictor checkpoint must be a dictionary payload. "
            f"Got {type(payload)!r}."
        )

    model_state_dict = payload.get("model_state_dict")
    if not isinstance(model_state_dict, dict) or not model_state_dict:
        raise ValueError(
            "DFLASH predictor checkpoint is missing model_state_dict."
        )

    first_weight = model_state_dict.get("net.0.weight")
    if first_weight is None or not hasattr(first_weight, "shape"):
        raise ValueError(
            "DFLASH predictor checkpoint is missing net.0.weight."
        )

    metrics = payload.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    input_dim = int(metrics.get("input_dim") or int(first_weight.shape[1]))
    hidden_dim = int(
        _get_nested(metrics, "args", "hidden_dim") or int(first_weight.shape[0])
    )
    dropout = float(_get_nested(metrics, "args", "dropout") or 0.0)

    model = AcceptPredictorMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    model.eval()
    return LoadedDFlashPredictor(
        model=model,
        checkpoint_path=str(checkpoint_path),
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
