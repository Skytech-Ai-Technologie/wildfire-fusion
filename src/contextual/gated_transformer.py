"""§5.2 — Entropy-gated cross-modal transformer, implements eq. (3):

    A_ij = softmax(Q_i K_j^T / sqrt(d_k)) * sigmoid(g_ij)

g_ij is a learned gate seeded by an entropy estimate of each modality's confidence, so
uncertain evidence contributes less to the cross-modal representation.

Training objective for this tier is [INFORMATION REQUIRED] in the source. ASSUMPTION:
train with BCE loss between refined_pi and ground-truth fire/no-fire labels per event
window, using the same leakage-safe splits as §3.4, optimizer AdamW, lr 5e-4, 30 epochs,
early stopping on val loss (see train_contextual.py).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EntropyGatedAttention(nn.Module):
    def __init__(self, d_model=256, n_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.gate_mlp = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1))
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _entropy(pi: torch.Tensor, eps=1e-8) -> torch.Tensor:
        p = pi.clamp(eps, 1 - eps)
        return -(p * p.log() + (1 - p) * (1 - p).log())  # binary entropy in [0, ln2]

    def forward(self, x, modality_confidence):
        # x: (B, T, d_model) stacked modality embeddings; modality_confidence: (B, T)
        B, T, D = x.shape
        Q = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        ent_i = self._entropy(modality_confidence).unsqueeze(-1)          # (B, T, 1)
        ent_j = ent_i.transpose(1, 2)                                     # (B, 1, T)
        pair_feat = torch.stack(
            [ent_i.expand(-1, -1, T), ent_j.expand(-1, T, -1)], dim=-1
        )                                                                 # (B, T, T, 2)
        g_ij = self.gate_mlp(pair_feat).squeeze(-1)                       # (B, T, T)
        gate = torch.sigmoid(g_ij).unsqueeze(1)                           # (B, 1, T, T)

        scores = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = F.softmax(scores, dim=-1) * gate
        attn = self.dropout(attn)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class ContextualEncoder(nn.Module):
    """4-layer entropy-gated encoder + "three fully connected layers + sigmoid" head.

    If the incoming detector embeddings are not d_model-dimensional, pass in_dim and a
    linear projection is applied first (our perceptual embeddings are GAP'd backbone
    features, so in_dim is typically the P5 channel count).
    """

    def __init__(self, d_model=256, n_heads=8, n_layers=4, dropout=0.1, in_dim=None):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model) if in_dim is not None else nn.Identity()
        self.layers = nn.ModuleList([
            EntropyGatedAttention(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.ffn_head = nn.Sequential(          # "three fully connected layers + sigmoid"
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4), nn.ReLU(),
            nn.Linear(d_model // 4, 1), nn.Sigmoid(),
        )

    def forward(self, x, modality_confidence):
        x = self.in_proj(x)
        for attn, norm in zip(self.layers, self.norms):
            x = norm(x + attn(x, modality_confidence))
        refined_pi = self.ffn_head(x).squeeze(-1)     # \hat{pi}_m per modality slot
        return refined_pi, x


class ConcatFusionEncoder(nn.Module):
    """Ablation variant (§11, Table VI): replace the entropy-gated transformer with a
    plain per-token MLP — the "concat instead of transformer" baseline. Same forward
    interface as ContextualEncoder so the fusion loop is unchanged."""

    def __init__(self, d_model=256, n_heads=None, n_layers=None, dropout=0.1, in_dim=None):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model) if in_dim is not None else nn.Identity()
        self.mixer = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
        )
        self.ffn_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4), nn.ReLU(),
            nn.Linear(d_model // 4, 1), nn.Sigmoid(),
        )

    def forward(self, x, modality_confidence):
        x = self.in_proj(x)
        refined_pi = self.ffn_head(self.mixer(x)).squeeze(-1)
        return refined_pi, x
