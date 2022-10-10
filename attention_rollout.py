import matplotlib.pyplot as plt
import torch
import cv2


class AttentionGetter:
    def __init__(self, model, attention_layer_name='attn_drop', ):
        self.model = model
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)
        self.attentions = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())


def rollout(attentions, head_fusion='mean', discard_ratio=0.9):

    batch_size, num_heads, num_tokens, _ = attentions[0].shape
    assert batch_size == 1, 'rollout attention iif batch is 1'
    result = torch.stack([torch.eye(num_tokens) for _ in range(num_heads)])        # [heads, 65, 65]
    results = []

    with torch.no_grad():
        # layer 갯수 만큼
        for attention in attentions:
            eyes = torch.stack([torch.eye(num_tokens) for _ in range(num_heads)])  # [heads, 65, 65]
            attention_map = eyes + attention.squeeze(0)
            attention_map = attention_map / attention_map.sum(dim=-1).unsqueeze(-1)
            result = torch.matmul(result, attention_map)
            results.append(result)

    results = torch.stack(results, dim=0)                                          # [layers, heads, 65, 65]
    print(results.shape)
    # masks = results[:, :, 0, 1:]                                                   # [layers, heads, 64]
    # print(masks.size())
    return results



            # # attention_heads_fused = attention.max(axis=1)[0]  # due to torch.max return (val, idx) - [1, 65, 65]
            # flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            # _, indices = flat.topk(k=int(flat.size(-1) * discard_ratio), dim=-1, largest=False)
            # print(indices)


if __name__ == '__main__':
    import os
    import numpy as np
    import configargparse
    from config import get_args_parser
    from dataset import build_dataloader
    from models.build import build_model

    parser = configargparse.ArgumentParser('Vit', parents=[get_args_parser()])
    opts = parser.parse_args()

    # 2. ** device **
    device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')

    # 3. ** dataset / dataloader **
    train_loader, test_loader, mean, std = build_dataloader(opts, is_return_mean_std=True)
    test_set = test_loader.dataset

    for i in range(test_set.__len__()):
        img, label = test_set.__getitem__(i)
        batch_img = img.unsqueeze(0).to(device)  # [1, 3, 32, 32]

        # tensor to img
        img_vis = np.array(img.permute(1, 2, 0), np.float32)  # C, W, H
        img_vis *= std
        img_vis += mean
        im = img_vis

        # model
        model = build_model(opts).to(device)
        # load check point
        f = os.path.join(opts.log_dir, opts.name, 'saves', opts.name + '.{}.pth.tar'.format('best'))
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
        print('load pth {}'.format('best'))
        # attention getter
        attention_getter = AttentionGetter(model)

        x = model(batch_img)
        attn = attention_getter.attentions
        reuslts = rollout(attn)

        # head의 갯수
        num_heads = reuslts.size(1)

        # head의 갯수만큼 plt
        v = reuslts[-1]  # 마지막 layer의 heads들 [12, 65, 65]

        basis_class_token = 0   # if 0 cls token,

        # ============================= for one mean img =============================
        v_ = v.mean(0)
        # v_ = v[0]
        grid_size = 8

        mask = v_[basis_class_token, 1:].reshape(grid_size, grid_size).detach().numpy()
        mask = cv2.resize(mask / mask.max(), (im.shape[1], im.shape[0]))[..., np.newaxis]
        result = (mask * im)

        im = im.clip(0, 1)
        _mask = mask.clip(0, 1)
        _result = result.clip(0, 1)
        #
        # fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 12))
        #
        # ax1.set_title('Original')
        # ax2.set_title('Attention Mask')
        # ax3.set_title('Attention Map')
        # _ = ax1.imshow(im)
        # _ = ax2.imshow(mask.squeeze())
        # _ = ax3.imshow(result)
        # # plt.show()

        # ============================= for all head imgs =============================
        masks_ = []
        result_ = []
        grid_size = 8
        masks = v[:, basis_class_token, 1:].reshape(reuslts.size(1), grid_size, grid_size).detach().numpy()
        for mask in masks:
            m = cv2.resize(mask / mask.max(), (im.shape[1], im.shape[0]))[..., np.newaxis]
            masks_.append(m.squeeze())
            result_.append(m * im)

        masks = np.stack(masks_)
        result = np.stack(result_)

        im = im.clip(0, 1)
        masks = masks.clip(0, 1)
        result = result.clip(0, 1)

        # fig, axs = plt.subplots(num_heads + 1, 3, figsize=(3, 24))
        # for i in range(num_heads):
        #     for j in range(3):
        #         # axs[i, j].set_title('head_{}'.format(i+1))
        #         axs[i, j].get_xaxis().set_visible(False)
        #         axs[i, j].get_yaxis().set_visible(False)
        #         if j == 0:
        #             axs[i, j].imshow(im)
        #         elif j == 1:
        #             axs[i, j].imshow(masks[i, :, ])
        #         elif j == 2:
        #             axs[i, j].imshow(result[i, :, :, :])
        # axs[num_heads, 0].get_xaxis().set_visible(False)
        # axs[num_heads, 0].get_yaxis().set_visible(False)
        # axs[num_heads, 1].get_xaxis().set_visible(False)
        # axs[num_heads, 1].get_yaxis().set_visible(False)
        # axs[num_heads, 2].get_xaxis().set_visible(False)
        # axs[num_heads, 2].get_yaxis().set_visible(False)
        #
        # # last row mean
        # axs[num_heads, 0].imshow(im)
        # axs[num_heads, 1].imshow(_mask.squeeze())
        # axs[num_heads, 2].imshow(_result)
        # plt.show()

        fig, axs = plt.subplots(3, num_heads + 1, figsize=(24, 6))
        for i in range(num_heads):
            for j in range(3):
                # axs[i, j].set_title('head_{}'.format(i+1))
                axs[j, i].get_xaxis().set_visible(False)
                axs[j, i].get_yaxis().set_visible(False)
                if j == 0:
                    axs[j, i].imshow(im)
                elif j == 1:
                    axs[j, i].imshow(masks[i, :, ])
                elif j == 2:
                    axs[j, i].imshow(result[i, :, :, :])
        for i in range(3):
            axs[i, num_heads].get_xaxis().set_visible(False)
            axs[i, num_heads].get_yaxis().set_visible(False)
        axs[0, num_heads].imshow(im)
        axs[1, num_heads].imshow(_mask.squeeze())
        axs[2, num_heads].imshow(_result)
        plt.show()
