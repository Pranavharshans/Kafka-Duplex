"""Minimal causal LM for Stage 1 alignment training."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(slots=True)
class Stage1ModelConfig:
    vocab_size: int
    context_length: int
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    ffw_multiplier: int = 4
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Stage1ModelConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, steps, hidden = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, steps, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, steps, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, steps, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.triu(torch.ones(steps, steps, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        if attn_mask is not None:
            key_mask = ~attn_mask[:, None, None, :].bool()
            scores = scores.masked_fill(key_mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(batch, steps, hidden)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, config: Stage1ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * config.ffw_multiplier),
            nn.GELU(),
            nn.Linear(config.hidden_size * config.ffw_multiplier, config.hidden_size),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.ffn(self.ln_2(x))
        return x


class Stage1CausalLM(nn.Module):
    def __init__(self, config: Stage1ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = nn.Embedding(config.context_length, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps = input_ids.shape
        positions = torch.arange(steps, device=input_ids.device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, attn_mask=attention_mask)
        logits = self.lm_head(self.ln_f(x))

        if labels is None:
            return logits, torch.tensor(0.0, device=input_ids.device)

        loss = nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        return logits, loss
