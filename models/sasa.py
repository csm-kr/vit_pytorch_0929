# stand alone self attention - https://arxiv.org/abs/1906.05909

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class SASA(nn.Module):
    def __init__(self, dim=192, num_heads=12, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.pos_proj = nn.Linear(3, num_heads)
        self.proj_drop = nn.Dropout(proj_drop)
        self.gating_param = nn.Parameter(torch.ones(self.num_heads))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, N, C = x.shape
        if not hasattr(self, 'rel_indices') or self.rel_indices.size(1) != N:
            self.get_rel_indices(N)

        attn = self.get_attention(x)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_attention(self, x):
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]
        pos_score = self.rel_indices.expand(B, -1, -1, -1)
        pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
        patch_score = (q @ k.transpose(-2, -1)) * self.scale
        patch_score = patch_score.softmax(dim=-1)
        pos_score = pos_score.softmax(dim=-1)
        attn = patch_score + pos_score
        attn = self.attn_drop(attn)
        return attn

    def get_rel_indices(self, num_patches):
        img_size = int(num_patches ** .5)
        rel_indices = torch.zeros(1, num_patches, num_patches, 3)
        ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
        indx = ind.repeat(img_size, img_size)
        indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
        indd = indx ** 2 + indy ** 2
        rel_indices[:, :, :, 2] = indd.unsqueeze(0)
        rel_indices[:, :, :, 1] = indy.unsqueeze(0)
        rel_indices[:, :, :, 0] = indx.unsqueeze(0)
        device = self.qk.weight.device
        self.rel_indices = rel_indices.to(device)
