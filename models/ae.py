import math
import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, img_size, num_patches, dim, has_cls_token):
        super().__init__()

        self.dim = dim
        self.has_cls_token = has_cls_token
        self.encoder = Encoder(img_size, num_patches, dim)
        self.decoder = Decoder(img_size, num_patches, dim)

    def img2seq(self, x):
        B, C, H, W = x.size()
        z = self.encoder(x)
        z_seq = z.view(B, self.dim, -1).permute(0, 2, 1)
        if self.has_cls_token:
            cls_token = torch.zeros(B, 1, self.dim).to(z_seq.get_device())
            nn.init.normal_(cls_token, std=1e-6)
            z_seq = torch.cat([cls_token, z_seq], dim=1)
        return z, z_seq

    def forward(self, x):
        z, z_seq = self.img2seq(x)
        x_ = self.decoder(z)
        return z_seq, x_


# class Encoder(nn.Module):
#     def __init__(self, img_size, num_patches, dim):
#         super().__init__()
#
#         self.scale = int(math.log2(img_size / num_patches))  # 4 -> 2
#         self.dim = dim
#         self.encode = nn.ModuleList([nn.Sequential(nn.Conv2d(3, 3, 3, stride=2, padding=1),
#                                                    nn.SELU(), )
#                                      for _ in range(self.scale)])
#         self.encode = nn.Sequential(*self.encode)
#         self.expand_dim = nn.Sequential(nn.Conv2d(3, dim, 3, stride=1, padding=1))
#     def forward(self, x):
#         x = self.encode(x)
#         x = self.expand_dim(x)
#         return x


class Encoder(nn.Module):
    def __init__(self, img_size, num_patches, dim):
        super().__init__()

        self.scale = int(math.log2(img_size / num_patches))  # 4 -> 2
        self.dim = dim
        self.encode = nn.ModuleList([nn.Sequential(nn.Conv2d(3, 3, 3, stride=2, padding=1),
                                                   nn.SELU(), )
                                     for _ in range(self.scale - 1)])
        self.encode.append(nn.Sequential(nn.Conv2d(3, dim, 3, stride=2, padding=1),
                                         nn.SELU()))
        self.encode = nn.Sequential(*self.encode)

    def forward(self, x):
        x = self.encode(x)
        return x

class Decoder(nn.Module):
    def __init__(self, img_size, num_patches, dim):
        super().__init__()

        self.scale = int(math.log2(img_size / num_patches))  # 4 -> 2
        self.dim = dim

        self.decode = nn.ModuleList(nn.Sequential(nn.ConvTranspose2d(dim, 3, 4, stride=2, padding=1),
                                                  nn.SELU()))

        self.decode.append(nn.Sequential(*nn.ModuleList([nn.Sequential(nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1),
                                                                       nn.SELU(), ) for _ in range(self.scale - 1)])))
        self.decode = nn.Sequential(*self.decode)

    def forward(self, x):
        x = self.decode(x)
        return x


if __name__ == '__main__':
    x = torch.randn([4, 3, 224, 224]).cuda()
    ae = AutoEncoder(img_size=224, num_patches=14, dim=784, has_cls_token=False).cuda()
    z_seq, x_ = ae(x)
    print(z_seq.size())
    print(x_.size())

    # x = torch.randn([4, 3, 32, 32]).cuda()
    # h = Encoder(img_size=32, patch_size=8, dim=196).cuda()
    # g = Decoder(img_size=32, patch_size=8, dim=196).cuda()
    # z = h(x)
    # x_ = g(z)
    # print(z.size())
    # print(x_.size())

    x = torch.randn([4, 3, 32, 32]).cuda()
    ae = AutoEncoder(img_size=32, num_patches=8, dim=196, has_cls_token=False).cuda()
    z_seq, x_ = ae(x)
    print(z_seq.size())
    print(x_.size())
