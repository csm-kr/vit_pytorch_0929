import torch
import torchvision.transforms as tfs
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


def build_dataloader(opts, is_return_mean_std=False):

    train_loader = None
    test_loader = None

    if opts.data_type == 'cifar10':
        print('dataset : {}'.format(opts.data_type))

        opts.num_classes = 10
        opts.input_size = 32
        opts.data_root = './CIFAR10'
        MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        transform_train = tfs.Compose([
            tfs.RandomCrop(32, padding=4),
            tfs.RandomHorizontalFlip(),
            tfs.ToTensor(),
            tfs.Normalize(mean=MEAN, std=STD),
        ])

        transform_test = tfs.Compose([tfs.ToTensor(),
                                      tfs.Normalize(mean=MEAN,
                                                    std=STD),
                                      ])

        train_set = CIFAR10(root=opts.data_root,
                            train=True,
                            download=True,
                            transform=transform_train)

        test_set = CIFAR10(root=opts.data_root,
                           train=False,
                           download=True,
                           transform=transform_test)

        train_loader = DataLoader(train_set,
                                  batch_size=opts.batch_size,
                                  shuffle=True,
                                  num_workers=opts.num_workers,
                                  pin_memory=True,
                                  )

        test_loader = DataLoader(test_set,
                                 batch_size=opts.batch_size,
                                 shuffle=False,
                                 num_workers=opts.num_workers,
                                 pin_memory=True,
                                 )

    if is_return_mean_std:
        return train_loader, test_loader, MEAN, STD

    return train_loader, test_loader