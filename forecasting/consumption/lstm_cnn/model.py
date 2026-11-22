"""LSTM-CNN model for consumption forecasting."""

import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN_DIM           = 128
NUM_LAYERS           = 1
CNN_CHANNELS         = 96     # per-branch channel count; 3 branches → 288 total into LSTM
DEFAULT_KERNEL_SIZES = [3, 7, 24]
DROPOUT              = 0.2

# Pooling modes for the LSTM-CNN decoder bottleneck.
#   'attention'           — single softmax-attention context vector (legacy).
#   'attention_max_last'  — concat of attention context, max-pool over time,
#                           and the last hidden state. Preserves sharp local
#                           features that softmax averaging would smooth out;
#                           recommended for peak-sensitive targets.
POOL_MODE_ATTENTION          = 'attention'
POOL_MODE_ATTENTION_MAX_LAST = 'attention_max_last'
DEFAULT_POOL_MODE            = POOL_MODE_ATTENTION_MAX_LAST


class TemporalAttention(nn.Module):
    """
    Additive attention pooling over all LSTM hidden states.

    Instead of taking only the last hidden state, a learned scalar score is
    computed for every timestep; the softmax-normalised scores weight-sum the
    hidden states into a single context vector.  This lets the decoder focus
    on the most informative parts of the input window rather than relying
    solely on the final step.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, 1, bias=True)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out : (batch, seq_len, hidden_dim)
        scores  = self.W(lstm_out).squeeze(-1)                        # (batch, seq_len)
        weights = torch.softmax(scores, dim=1)                        # (batch, seq_len)
        context = (weights.unsqueeze(-1) * lstm_out).sum(dim=1)       # (batch, hidden_dim)
        return context


class LSTM_CNN_Model(nn.Module):
    """
    Multi-scale LSTM-CNN for consumption forecasting.

    Three parallel Conv1D branches with different kernel sizes capture local
    (k=3), sub-daily (k=7), and full-day (k=24) patterns simultaneously.
    Their outputs are concatenated and fed into a shared LSTM encoder with
    pooled context.

    Parameters
    ----------
    input_size    : number of input features
    output_size   : forecast horizon
    num_features  : number of predicted output channels (1 per model)
    hidden_dim    : LSTM hidden size
    num_layers    : number of LSTM layers
    cnn_channels  : output channels *per branch* (total into LSTM = cnn_channels × n_branches)
    kernel_sizes  : list of Conv1D kernel sizes (one branch per entry)
    dropout       : dropout probability
    pool_mode     : 'attention' (legacy softmax pool) or 'attention_max_last'
                    (concat of attention, max-pool, last step). The latter
                    preserves sharp peaks that the softmax average smooths out.
    """

    def __init__(
        self,
        input_size:   int,
        output_size:  int,
        num_features: int,
        hidden_dim:   int   = HIDDEN_DIM,
        num_layers:   int   = NUM_LAYERS,
        cnn_channels: int   = CNN_CHANNELS,
        kernel_sizes: list  = None,
        dropout:      float = DROPOUT,
        pool_mode:    str   = DEFAULT_POOL_MODE,
    ):
        super().__init__()
        if pool_mode not in (POOL_MODE_ATTENTION, POOL_MODE_ATTENTION_MAX_LAST):
            raise ValueError(f'Unknown pool_mode: {pool_mode!r}')
        kernel_sizes = kernel_sizes or DEFAULT_KERNEL_SIZES
        self.conv_branches = nn.ModuleList([
            nn.Conv1d(input_size, cnn_channels, kernel_size=k, padding='same')
            for k in kernel_sizes
        ])
        total_channels = cnn_channels * len(kernel_sizes)
        self.dropout   = nn.Dropout(p=dropout)
        self.lstm      = nn.LSTM(total_channels, hidden_dim, num_layers=num_layers,
                                 batch_first=True,
                                 dropout=dropout if num_layers > 1 else 0.0)
        self.attention = TemporalAttention(hidden_dim)
        self.pool_mode = pool_mode
        fc_in = hidden_dim if pool_mode == POOL_MODE_ATTENTION else 3 * hidden_dim
        self.fc        = nn.Linear(fc_in, output_size * num_features)
        self.output_length = output_size
        self.num_features  = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)                                    # (batch, features, seq_len)
        branches = [F.relu(conv(x)) for conv in self.conv_branches]
        x = torch.cat(branches, dim=1)                            # (batch, total_channels, seq_len)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)                                    # (batch, seq_len, total_channels)
        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out)                         # (batch, hidden_dim)
        if self.pool_mode == POOL_MODE_ATTENTION_MAX_LAST:
            max_pool = lstm_out.max(dim=1).values                  # (batch, hidden_dim)
            last     = lstm_out[:, -1, :]                          # (batch, hidden_dim)
            context  = torch.cat([context, max_pool, last], dim=1) # (batch, 3*hidden_dim)
        out = self.fc(context)
        out = out.view(-1, self.output_length, self.num_features)
        return out
