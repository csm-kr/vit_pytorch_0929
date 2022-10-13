from models.vit import ViT
from models.tomvit import TomViT


def build_model(opts):
    if opts.model_type == 'vit':
        model = ViT(img_size=opts.img_size, patch_size=opts.patch_size, in_chans=opts.in_chans,
                    embed_dim=opts.embed_dim, depth=opts.depth, num_heads=opts.num_heads,
                    mlp_ratio=opts.mlp_ratio)

    elif opts.model_type == 'tomvit':
        model = TomViT(img_size=opts.img_size, patch_size=opts.patch_size, in_chans=opts.in_chans,
                       num_classes=opts.num_classes, embed_dim=opts.embed_dim, depth=opts.depth,
                       num_heads=opts.num_heads, mlp_ratio=opts.mlp_ratio, qkv_bias=False,
                       drop_rate=0., attn_drop_rate=0., drop_path=opts.drop_path,
                       has_cls_token=opts.has_cls_token, has_last_norm=opts.has_last_norm, has_basic_poe=opts.has_basic_poe,
                       has_auto_encoder=opts.has_auto_encoder, has_xavier_init=opts.has_xavier_init)
    return model