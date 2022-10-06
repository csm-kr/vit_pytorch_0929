# clarified vit - 명확한 vit
# vit 로 여러 실험을 통하여 ambiguity를 확실히 한다.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class EmbeddingLayer(nn.Module):
    def __init__(self, in_chans, embed_dim, img_size, patch_size, has_cls_token):
        super().__init__()

        self.num_tokens = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.project = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.has_cls_token = has_cls_token
        if has_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.cls_token, std=1e-6)
            self.num_tokens += 1
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, self.embed_dim))
        trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        B, C, H, W = x.shape
        embedding = self.project(x)
        z = embedding.view(B, self.embed_dim, -1).permute(0, 2, 1)  # BCHW -> BNC

        if self.has_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            z = torch.cat([cls_tokens, z], dim=1)

        # add position embedding
        z = z + self.pos_embed
        return z


class MSA(nn.Module):
    def __init__(self, dim=192, num_heads=12, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if B == 1:
            return x, attn
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = MSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        if isinstance(x, tuple):
            attn_list = x[1]
            x = x[0]
        else:
            attn_list = []
        if x.size(0) == 1:  # iif batch is 1, put attentions in attn_list
            x_, attn = self.attn(self.norm1(x))
            x = x + x_
            x = x + self.mlp(self.norm2(x))
            attn_list.append(attn)
            return x, attn_list
        else:
            x_ = self.attn(self.norm1(x))
            x = x + x_
            x = x + self.mlp(self.norm2(x))
            return x


class CViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10, embed_dim=192, depth=12,
                 num_heads=12, mlp_ratio=2., qkv_bias=False, drop_rate=0., attn_drop_rate=0.,
                 has_cls_token=True):
        super().__init__()

        # cls toekn
        self.has_cls_token = has_cls_token

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = nn.LayerNorm
        act_layer = nn.GELU

        self.patch_embed = EmbeddingLayer(in_chans, embed_dim, img_size, patch_size, has_cls_token)
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])

        # final norm
        self.norm = norm_layer(embed_dim)

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, get_attn_map=False):
        x = self.patch_embed(x)
        x = self.blocks(x)

        if get_attn_map:
            assert get_attn_map and x[0].size(0) == 1, 'batch is only one for attn map'
            attn_list = x[1]  # 9, 1, 12, 65, 65 - [depth, batch, head, 64, 64]
            attn_tensor = torch.stack(attn_list).squeeze(1)  # batch 제거 - [d, h, l, l]
            attn_tensor = attn_tensor.mean(dim=1)  # head 제거 - [d, l, l]
            x = x[0]

        if x[0].size(0) == 1:
            x = x[0]

        x = self.norm(x)
        x = self.head(x)[:, 0]

        if get_attn_map:
            return x, attn_tensor

        return x


if __name__ == '__main__':
    img = torch.randn([2, 3, 32, 32])
    cvit = CViT(has_cls_token=True)
    x = cvit(img)
    print(x.size())

    img = torch.randn([1, 3, 32, 32])
    x, att = cvit(img, True)
    print(x.size())
    print(att.size())

    img = torch.randn([2, 3, 32, 32])
    cvit = CViT(has_cls_token=False)
    x = cvit(img)
    print(x.size())

    img = torch.randn([1, 3, 32, 32])
    x, att = cvit(img, True)
    print(x.size())
    print(att.size())
