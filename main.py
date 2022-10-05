import os

import torch
import visdom
import configargparse
import torch.nn as nn

from dataset import build_dataloader
from models.build import build_model
from loss import build_loss

from scheduler import CosineAnnealingWarmupRestarts
from log import XLLogSaver
from utils import resume

from train import train_one_epoch
from test import test_and_evaluate


def main_worker(rank, opts):

    # 1. ** argparser **
    print(opts)

    # 2. ** device **
    device = torch.device('cuda:{}'.format(int(opts.gpu_ids[opts.rank])))

    # 3. ** visdom **
    vis = visdom.Visdom(port=opts.visdom_port)

    # 4. ** dataset / dataloader **
    train_loader, test_loader = build_dataloader(opts)

    # 5. ** model **
    model = build_model(opts)
    model = model.to(device)

    # 6. ** criterion **
    criterion = build_loss(opts)

    # 7. ** optimizer **
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=opts.lr,
                                  weight_decay=opts.weight_decay)

    # 8. ** scheduler **
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=opts.epoch,
        cycle_mult=1.,
        max_lr=opts.lr,
        min_lr=5e-5,
        warmup_steps=10
        )

    # 9. ** logger **
    os.makedirs(opts.log_dir, exist_ok=True)
    xl_log_saver = None
    if opts.rank == 0:
        xl_log_saver = XLLogSaver(xl_folder_name=os.path.join(opts.log_dir, opts.name),
                                  xl_file_name=opts.name,
                                  tabs=('epoch', 'accuracy_top1', 'val_loss'))

    model, optimizer, scheduler = resume(opts, model, optimizer, scheduler)

    result_best = {'epoch': 0, 'accuracy_top1': 0., 'val_loss': 0.}

    for epoch in range(opts.start_epoch, opts.epoch):

        # 10. train
        train_one_epoch(epoch, vis, train_loader, model, optimizer, criterion, scheduler, opts)

        # 11. test
        result_best = test_and_evaluate(epoch, vis, test_loader, model, criterion, opts, xl_log_saver, result_best, is_load=False)
        scheduler.step()


if __name__ == '__main__':
    import torch.multiprocessing as mp
    from config import get_args_parser

    parser = configargparse.ArgumentParser('Vit', parents=[get_args_parser()])
    opts = parser.parse_args()

    if len(opts.gpu_ids) > 1:
        opts.distributed = True

    opts.world_size = len(opts.gpu_ids)
    opts.num_workers = len(opts.gpu_ids) * 4

    if opts.distributed:
        mp.spawn(main_worker,
                 args=(opts,),
                 nprocs=opts.world_size,
                 join=True)
    else:
        main_worker(opts.rank, opts)
