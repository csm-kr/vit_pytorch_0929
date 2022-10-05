from models.vit import ViT
from models.lvit import LViT


def build_model(opts):
    if opts.model_type == 'vit':
        model = ViT(img_size=opts.img_size, patch_size=opts.patch_size, in_chans=opts.in_chans,
                    embed_dim=opts.embed_dim, depth=opts.depth, num_heads=opts.num_heads,
                    mlp_ratio=opts.mlp_ratio)

    if opts.model_type == 'lvit':
        model = LViT(img_size=opts.img_size, patch_size=opts.patch_size, in_chans=opts.in_chans,
                     embed_dim=opts.embed_dim, depth=opts.depth, num_heads=opts.num_heads,
                     mlp_ratio=opts.mlp_ratio)

    return model