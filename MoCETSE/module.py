import torch
import torch.nn as nn
from torch import nn, einsum
from einops import rearrange


class RelativePositionEncoding(nn.Module):
    def __init__(self, max_length, dim):
        super().__init__()
        self.relative_embeddings = nn.Parameter(torch.randn(2 * max_length - 1, dim))

    def forward(self, seq_length):

        relative_indices = torch.arange(-seq_length + 1, seq_length,
                                        device=self.relative_embeddings.device) + seq_length - 1
        return self.relative_embeddings[relative_indices]


class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_key=64, dim_value=64, dropout=0., max_length=1024):
        super().__init__()
        self.scale = dim_key ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_value * heads, bias=False)
        self.to_out = nn.Linear(dim_value * heads, dim)
        self.attn_dropout = nn.Dropout(dropout)

        self.relative_position_encoding = RelativePositionEncoding(max_length, dim_key)

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        nn.init.xavier_uniform_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, x, mask=None):
        n, h = x.shape[-2], self.heads

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        q = q * self.scale
        logits = einsum('b h i d, b h j d -> b h i j', q, k)

        if mask is not None:
            logits.masked_fill(mask == 0, -1e9)

        attn = logits.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), attn


class TransformerLayer(nn.Module):

    def __init__(self, hid_dim, heads, dropout_rate, att_dropout=0.05, max_length=1024):
        super().__init__()
        self.attn = RelativeMultiHeadAttention(hid_dim, heads, hid_dim //
                              heads, hid_dim // heads, att_dropout, max_length)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hid_dim),
            nn.Linear(hid_dim, hid_dim * 2),
            # nn.ReLU(),
            nn.GELU(),
            nn.Linear(hid_dim * 2, hid_dim),
            nn.Dropout(dropout_rate))
        self.layernorm = nn.LayerNorm(hid_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        residual = x
        x = self.layernorm(x)  # pre-LN
        x, attn = self.attn(x, mask)
        x = self.dropout(x)
        x = residual + x

        residual = x
        x = self.ffn(x)
        x = residual + x

        return x, attn


class MoCE(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, bias=False, num_experts=8):
        super(MoCE, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_experts = num_experts

        self.experts = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, bias=bias)
            for _ in range(num_experts)
        ])

        self.gating_network = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_experts),
            nn.Softmax(dim=1)
        )

        self.attention = nn.Sequential(
            nn.Conv1d(out_channels, out_channels // 4, 1),
            nn.ReLU(),
            nn.Conv1d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        gates = self.gating_network(x)  # (batch_size, num_experts)

        expert_outputs = [expert(x) for expert in self.experts]  # list of (batch_size, out_channels, seq_len)

        output = torch.zeros_like(expert_outputs[0])
        for i in range(self.num_experts):
            output += gates[:, i].unsqueeze(1).unsqueeze(-1) * expert_outputs[i]

        attn = self.attention(output)
        output = output * attn

        return output


class MLPLayer(nn.Module):

    def __init__(self, in_dim, hid_dim, num_classes, dropout_rate=0.):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, hid_dim),
                                   nn.ReLU(),
                                   nn.Dropout(p=dropout_rate),
                                   nn.Linear(hid_dim, num_classes)
                                   )

    def forward(self, x):
        return self.layer(x)
