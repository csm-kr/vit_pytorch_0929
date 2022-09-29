import os
import torch


def resume(opts, model, optimizer, scheduler):
    if opts.start_epoch != 0:
        # take pth at epoch - 1

        f = os.path.join(opts.log_dir, opts.name, 'saves', opts.name + '.{}.pth.tar'.format(opts.start_epoch - 1))
        device = torch.device('cuda:{}'.format(opts.gpu_ids[opts.rank]))
        checkpoint = torch.load(f=f,
                                map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])                              # load model state dict
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])                      # load optim state dict
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])                      # load sched state dict
        if opts.rank == 0:
            print('\nLoaded checkpoint from epoch %d.\n' % (int(opts.start_epoch) - 1))
    else:
        if opts.rank == 0:
            print('\nNo check point to resume.. train from scratch.\n')
    return model, optimizer, scheduler