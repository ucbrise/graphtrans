import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)

        self.n_head = n_head

    def forward(self, x, attn_mask: torch.Tensor = None, valid_input_mask: torch.Tensor = None, mask_value=-1e6):
        """mask should be a 3D tensor of shape (B, T, T)"""
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if attn_mask is not None:
            att = att.masked_fill(attn_mask.unsqueeze(1) == 0, mask_value)
        if valid_input_mask is not None:
            att = att.masked_fill(valid_input_mask.unsqueeze(1).unsqueeze(2) == 0, mask_value)

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    def __init__(self, n_embd, n_ff, n_head, attn_pdrop, resid_pdrop, prenorm=True):
        super().__init__()
        self.prenorm = prenorm
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, n_ff),
            nn.GELU(),
            nn.Linear(n_ff, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x, attn_mask=None, valid_input_mask=None):
        if self.prenorm:
            x = x + self.attn(self.ln1(x), attn_mask, valid_input_mask)
            x = x + self.mlp(self.ln2(x))
        else:
            x = self.ln1(x + self.attn(x, attn_mask, valid_input_mask))
            x = self.ln2(x + self.mlp(x))
        return x


class MaskedTransformerBlock(nn.Module):
    def __init__(self, n_layer, n_embd, n_ff, n_head, attn_pdrop, resid_pdrop, prenorm=True):
        super().__init__()
        self.blocks = nn.ModuleList([Block(n_embd, n_ff, n_head, attn_pdrop, resid_pdrop, prenorm) for _ in range(n_layer)])
        # self.apply(self._init_weights)

    # def _init_weights(self, module):
    #     if isinstance(module, (nn.Linear, nn.Embedding)):
    #         module.weight.data.normal_(mean=0.0, std=0.02)
    #         if isinstance(module, nn.Linear) and module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, nn.LayerNorm):
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)

    def forward(self, x, attn_mask=None, valid_input_mask=None):
        for block in self.blocks:
            x = block(x, attn_mask, valid_input_mask)
        return x


class MaskedOnlyTransformerEncoder(nn.Module):
    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("Masked Transformer Encoder -- architecture config")
        group.add_argument("--num_encoder_layers_masked", type=int, default=0)
        group.add_argument("--transformer_prenorm", action="store_true", default=False)

    def __init__(self, args):
        super().__init__()
        self.max_input_len = args.max_input_len
        self.masked_transformer = MaskedTransformerBlock(
            args.num_encoder_layers_masked,
            args.d_model,
            args.dim_feedforward,
            args.nhead,
            args.transformer_dropout,
            args.transformer_dropout,
        )
        logger.info("number of parameters: %e" % sum(p.numel() for p in self.parameters()))

    def forward(self, x, attn_mask=None, valid_input_mask=None):
        """
        padded_h_node: n_b x B x h_d
        src_key_padding_mask: B x n_b
        """
        x = self.masked_transformer(x, attn_mask=attn_mask, valid_input_mask=valid_input_mask)
        return x
