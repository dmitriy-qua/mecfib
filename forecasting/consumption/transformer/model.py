"""Slim Temporal Fusion Transformer for consumption forecasting.

Per-target point predictor. Treats every input column as a past time-varying
feature (no past/known-future split, no static covariates, no quantile head)
so the input/output contract matches the LSTM-CNN base learner exactly:

    input  : (batch, sequence_length, n_features)
    output : (batch, forecast_horizon, 1)

Components retained from Lim et al. (2021):
  * Gated Linear Units and Gated Residual Networks
  * Variable Selection Network at the input
  * LSTM encoder for local context
  * Masked multi-head self-attention for long-range dependencies
  * Gated skip + LayerNorm after attention
  * Position-wise GRN before the output head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


HIDDEN_DIM   = 64
N_HEADS      = 4
DROPOUT      = 0.1
LSTM_LAYERS  = 1


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class GatedLinearUnit(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc      = nn.Linear(in_dim, out_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        a, b = self.fc(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GatedResidualNetwork(nn.Module):
    """GRN(a) = LayerNorm(a + GLU(W2 ELU(W1 a + b1) + b2))

    When in_dim != out_dim a learned linear projection bridges the skip path.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 dropout: float = 0.0):
        super().__init__()
        self.fc1  = nn.Linear(in_dim, hidden_dim)
        self.elu  = nn.ELU()
        self.fc2  = nn.Linear(hidden_dim, out_dim)
        self.glu  = GatedLinearUnit(out_dim, out_dim, dropout=dropout)
        self.norm = nn.LayerNorm(out_dim)
        self.skip = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eta2 = self.elu(self.fc1(x))
        eta1 = self.fc2(eta2)
        return self.norm(self.skip(x) + self.glu(eta1))


class VariableSelectionNetwork(nn.Module):
    """Per-timestep softmax-weighted sum of per-feature GRN outputs.

    Each input feature is passed through its own GRN (in_dim=1 -> hidden_dim).
    A separate GRN over the concatenated raw inputs produces feature weights
    (softmaxed over the feature axis) which then mix the per-feature outputs.
    """

    def __init__(self, n_features: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.n_features = n_features
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(1, hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(n_features)
        ])
        self.weight_grn = GatedResidualNetwork(
            n_features, hidden_dim, n_features, dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        weights = torch.softmax(self.weight_grn(x), dim=-1)        # (B, T, F)
        # Per-feature embedding -> stack along new axis -> (B, T, F, H)
        per_feat = torch.stack(
            [grn(x[..., i : i + 1]) for i, grn in enumerate(self.feature_grns)],
            dim=-2,
        )
        return (weights.unsqueeze(-1) * per_feat).sum(dim=-2)        # (B, T, H)


class TFTModel(nn.Module):
    """Slim TFT (point prediction, all-past inputs).

    Parameters
    ----------
    input_size    : number of input features
    output_size   : forecast horizon
    num_features  : number of predicted output channels (1 per target model)
    hidden_dim    : model width (used for VSN/LSTM/attention)
    n_heads       : multi-head attention heads
    lstm_layers   : LSTM encoder depth
    dropout       : dropout probability
    """

    def __init__(
        self,
        input_size:   int,
        output_size:  int,
        num_features: int = 1,
        hidden_dim:   int = HIDDEN_DIM,
        n_heads:      int = N_HEADS,
        lstm_layers:  int = LSTM_LAYERS,
        dropout:      float = DROPOUT,
    ):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(f'hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})')

        self.vsn = VariableSelectionNetwork(input_size, hidden_dim, dropout=dropout)
        self.lstm_encoder = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=lstm_layers, batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        # Gated skip around the LSTM (Lim et al. eq. 12)
        self.post_lstm_glu  = GatedLinearUnit(hidden_dim, hidden_dim, dropout=dropout)
        self.post_lstm_norm = nn.LayerNorm(hidden_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.post_attn_glu  = GatedLinearUnit(hidden_dim, hidden_dim, dropout=dropout)
        self.post_attn_norm = nn.LayerNorm(hidden_dim)

        # Position-wise GRN (paper's "decoder")
        self.position_grn = GatedResidualNetwork(
            hidden_dim, hidden_dim, hidden_dim, dropout=dropout,
        )
        self.final_glu  = GatedLinearUnit(hidden_dim, hidden_dim, dropout=dropout)
        self.final_norm = nn.LayerNorm(hidden_dim)

        # Pool over the sequence (mean) then project to (horizon * num_features)
        self.head = nn.Linear(hidden_dim, output_size * num_features)
        self.output_length = output_size
        self.num_features  = num_features

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        # True positions are MASKED. Upper triangle (excluding diagonal) = future.
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        v = self.vsn(x)                                             # (B, T, H)
        lstm_out, _ = self.lstm_encoder(v)                          # (B, T, H)
        # Gated skip: encoder residual
        z = self.post_lstm_norm(v + self.post_lstm_glu(lstm_out))

        mask = self._causal_mask(z.size(1), z.device)
        attn_out, _ = self.attention(z, z, z, attn_mask=mask, need_weights=False)
        z = self.post_attn_norm(z + self.post_attn_glu(attn_out))

        z = self.position_grn(z)
        z = self.final_norm(z + self.final_glu(z))

        ctx = z.mean(dim=1)                                          # (B, H)
        out = self.head(ctx).view(-1, self.output_length, self.num_features)
        return out
