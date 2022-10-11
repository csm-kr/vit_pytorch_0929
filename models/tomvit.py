# tomvit - toward more visionary transformer
import math
import torch
import torch.nn as nn
from torchsummary import summary
from timm.models.layers import trunc_normal_
from models.ae import AutoEncoder


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe = pe.view(d_model * 2, width * height).permute(1, 0)
    return pe


class EmbeddingLayer(nn.Module):
    def __init__(self, in_chans, embed_dim, img_size, patch_size, has_cls_token, has_basic_poe):
        super().__init__()

        self.num_tokens = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.project = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.has_cls_token = has_cls_token
        if has_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.cls_token, std=1e-6)
            self.num_tokens += 1
        if has_basic_poe:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, self.embed_dim))
            trunc_normal_(self.pos_embed, std=.02)
        else:
            self.register_buffer('pos_embed', positionalencoding2d(self.embed_dim,
                                                                   int(math.sqrt(self.num_tokens)),
                                                                   int(math.sqrt(self.num_tokens))).unsqueeze(0))
            if has_cls_token:
                self.pos_embed = torch.cat([torch.zeros_like(self.cls_token), self.pos_embed], dim=1)
                # ([1, 1, 192] + [1, 64, 192])

    def forward(self, x):
        B, C, H, W = x.shape
        embedding = self.project(x)
        z = embedding.view(B, self.embed_dim, -1).permute(0, 2, 1)  # BCHW -> BNC

        if self.has_cls_token:
            # concat cls token
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
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TomViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10, embed_dim=192, depth=9,
                 num_heads=12, mlp_ratio=2., qkv_bias=False, drop_rate=0., attn_drop_rate=0., has_cls_token=True,
                 has_last_norm=True, has_basic_poe=True, has_auto_encoder=False,
                 ):
        super().__init__()

        # tomvit elements
        self.has_cls_token = has_cls_token
        self.has_last_norm = has_last_norm
        self.has_basic_poe = has_basic_poe
        # auto encoder
        self.has_auto_encoder = has_auto_encoder

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = nn.LayerNorm
        act_layer = nn.GELU

        self.patch_embed = EmbeddingLayer(in_chans, embed_dim, img_size, patch_size, has_cls_token, has_basic_poe)
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])

        # last norm
        if has_last_norm:
            self.norm = norm_layer(embed_dim)

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # auto encoder
        if self.has_auto_encoder:
            self.ae = AutoEncoder(img_size, int(img_size//patch_size), embed_dim, has_cls_token)

    def forward(self, x):

        # AE
        if self.has_auto_encoder:
            z_seq, x_ = self.ae(x)

        x = self.patch_embed(x)

        # AE
        if self.has_auto_encoder:
            x = x + z_seq

        x = self.blocks(x)
        if self.has_last_norm:
            x = self.norm(x)
        if self.has_cls_token:
            x = self.head(x)[:, 0]
        else:
            x = self.head(x).mean(1)

        # AE
        if self.has_auto_encoder:
            return x, x_

        return x


if __name__ == '__main__':
    img = torch.randn([1, 3, 32, 32]).cuda()
    tomvit = TomViT().cuda()
    x = tomvit(img)
    print(x.size())

    tomvit = TomViT(has_cls_token=False, has_last_norm=False).cuda()
    x = tomvit(img)
    print(x.size())

    summary(tomvit, (3, 32, 32))






