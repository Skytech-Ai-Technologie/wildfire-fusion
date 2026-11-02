"""§5.1 — Positional encodings.

Two separate encodings, per the paper: spatial (for detections within a frame) and
temporal (across the 32-frame window).

d_model, number of heads, dropout are [INFORMATION REQUIRED] in the source. ASSUMPTION
for a Jetson Orin Nano-feasible config: d_model=256, n_heads=8, dropout=0.1, n_layers=4
(paper does give 4 layers). Treated as hyperparameters to sweep once real validation
data exists — starting points, not claims of paper fidelity. See configs/default.yaml.
"""
import math

import torch
import torch.nn as nn

TEMPORAL_WINDOW = 32   # frames, per paper


def sinusoidal_positions(n_positions: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(n_positions, d_model)
    position = torch.arange(0, n_positions).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class TemporalPositionalEncoding(nn.Module):
    """Adds sinusoidal frame-position encodings across the temporal window."""

    def __init__(self, d_model: int, max_len: int = TEMPORAL_WINDOW):
        super().__init__()
        self.register_buffer("pe", sinusoidal_positions(max_len, d_model).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model); T must be <= max_len
        return x + self.pe[:, : x.shape[1]]


class SpatialBoxEncoding(nn.Module):
    """Encodes a detection's normalized box (cx, cy, w, h) into a d_model vector via a
    small MLP; added to the token embedding so the transformer can reason about spatial
    layout of detections within a frame. The exact spatial encoding is
    [INFORMATION REQUIRED] in the source — this is a documented ASSUMPTION."""

    def __init__(self, d_model: int, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden), nn.ReLU(), nn.Linear(hidden, d_model),
        )

    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        # boxes: (..., 4) normalized cx, cy, w, h
        return self.mlp(boxes)
