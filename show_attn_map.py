import os
import cv2
import torch
import argparse
import numpy as np
import torchvision.transforms as tfs
from torchvision.datasets.cifar import CIFAR10
from torchvision.datasets.mnist import MNIST
import matplotlib.pyplot as plt
from models.vit import ViT


def show_attention_map(epoch, model, batch_img, im, opts):

    # load check point
    f = os.path.join(opts.log_dir, opts.name, 'saves', opts.name + '.{}.pth.tar'.format(epoch))
    device = torch.device('cuda:{}'.format(opts.gpu_ids[opts.rank]))

    if isinstance(model, (torch.nn.parallel.distributed.DistributedDataParallel, torch.nn.DataParallel)):
        checkpoint = torch.load(f=f,
                                map_location=device)
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict)

    else:
        checkpoint = torch.load(f=f,
                                map_location=device)
        state_dict = checkpoint['model_state_dict']
        # state_dict = {k.replace('module.', ''): v for (k, v) in state_dict.items()}
        model.load_state_dict(state_dict)

    print('load pth {}'.format(epoch))

    x, attn = model(batch_img, True)
    print(x.size())
    print(attn.size())   # 9, 65, 65

    att_mat = attn
    att_mat = att_mat.cpu().detach()

    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))

    # cls token 에 맞추어 mask 생성
    # 1. if opts.is_cls_token
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    mask = cv2.resize(mask / mask.max(), (im.shape[1], im.shape[0]))[..., np.newaxis]
    result = (mask * im)

    im = im.clip(0, 1)
    mask = mask.clip(0, 1)
    result = result.clip(0, 1)

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 12))

    ax1.set_title('Original')
    ax2.set_title('Attention Mask')
    ax3.set_title('Attention Map')
    _ = ax1.imshow(im)
    _ = ax2.imshow(mask.squeeze())
    _ = ax3.imshow(result)
    plt.show()

    for i, v in enumerate(joint_attentions):
        # Attention from the output token to the input space.
        mask = v[0, 1:].reshape(8, 8).detach().numpy()
        mask = cv2.resize(mask / mask.max(), (im.shape[1], im.shape[0]))[..., np.newaxis]
        result = (mask * im)

        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 12))
        ax1.set_title('Original')
        ax2.set_title('mask')
        ax3.set_title('Attention Map_%d Layer' % (i + 1))

        im = im.clip(0, 1)
        mask = mask.clip(0, 1)
        result = result.clip(0, 1)

        _ = ax1.imshow(im)
        _ = ax2.imshow(mask.squeeze())
        _ = ax3.imshow(result)
        plt.show()

    return 0


if __name__ == '__main__':
    from config import get_args_parser
    import configargparse
    from models.build import build_model
    from dataset import build_dataloader

    parser = configargparse.ArgumentParser('Vit', parents=[get_args_parser()])
    opts = parser.parse_args()

    # 2. ** device **
    device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')

    # 3. ** dataset / dataloader **
    train_loader, test_loader, mean, std = build_dataloader(opts, is_return_mean_std=True)
    test_set = test_loader.dataset

    # model
    model = build_model(opts).to(device)

    # for i in range(test_set.__len__()):

    img, label = test_set.__getitem__(0)
    batch_img = img.unsqueeze(0).to(device)  # [1, 3, 32, 32]

    # tensor to img
    img_vis = np.array(img.permute(1, 2, 0), np.float32)  # C, W, H
    img_vis *= std
    img_vis += mean
    show_attention_map(epoch='best', model=model, batch_img=batch_img, im=img_vis, opts=opts)
