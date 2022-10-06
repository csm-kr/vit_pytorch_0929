# from models.vit import ViT
from models.vit_blog import ViT
from models.lvit import LViT
from models.cvit import CViT


def build_model(opts):
    if opts.model_type == 'vit':
        model = ViT(img_size=opts.img_size, patch_size=opts.patch_size, in_chans=opts.in_chans,
                    embed_dim=opts.embed_dim, depth=opts.depth, num_heads=opts.num_heads,
                    mlp_ratio=opts.mlp_ratio)

    elif opts.model_type == 'lvit':
        model = LViT(img_size=opts.img_size, patch_size=opts.patch_size, in_chans=opts.in_chans,
                     embed_dim=opts.embed_dim, depth=opts.depth, num_heads=opts.num_heads,
                     mlp_ratio=opts.mlp_ratio)

    elif opts.model_type == 'cvit':
        model = CViT(img_size=opts.img_size, patch_size=opts.patch_size, in_chans=opts.in_chans,
                     num_classes=opts.num_classes, embed_dim=opts.embed_dim,
                     depth=opts.depth, num_heads=opts.num_heads, mlp_ratio=opts.mlp_ratio,
                     qkv_bias=False, drop_rate=0., attn_drop_rate=0.,
                     has_cls_token=opts.has_cls_token)
    return model