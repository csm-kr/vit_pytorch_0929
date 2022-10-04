import os
import torch
from tqdm import tqdm
from utils import accuracy


def test_and_evaluate(epoch, vis, test_loader, model, criterion, opts, xl_log_saver=None, result_best=None, is_load=True):
    """
    evaluate imagenet test data
    :param epoch: epoch for evaluating test dataset
    :param vis: visdom
    :param data_loader: test loader (torch.utils.DataLoader)
    :param model: model
    :param criterion: loss
    :param is_load : bool is load
    :param opts: options from config
    :return: avg_loss and accuracy

    function flow
    1. load .pth file
    2. forward the whole test dataset
    3. calculate loss and accuracy
    """

    print('Validation of epoch [{}]'.format(epoch))

    # 1. load pth.tar
    checkpoint = None
    if is_load:

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
            state_dict = {k.replace('module.', ''): v for (k, v) in state_dict.items()}
            model.load_state_dict(state_dict)

    # 2. forward the whole test dataset & calculate performance
    model.eval()

    # for evaluation.
    loss_val, acc1_val, acc5_val, n = 0, 0, 0, 0

    with torch.no_grad():
        for idx, data in enumerate(tqdm(test_loader)):

            # ----------------- cuda -----------------
            images = data[0].to(int(opts.gpu_ids[opts.rank]))
            labels = data[1].to(int(opts.gpu_ids[opts.rank]))

            # ----------------- forward -----------------
            outputs = model(images)

            # ----------------- evaluate -----------------
            loss = criterion(outputs, labels)
            n += images.size(0)
            acc = accuracy(outputs, labels, (1, 5))
            acc1 = acc[0]
            acc5 = acc[1]

            loss_val += float(loss.item() * images.size(0))
            acc1_val += float(acc1 * images.size(0))
            acc5_val += float(acc5 * images.size(0))

        acc1 = (acc1_val / n) / 100
        acc5 = (acc5_val / n) / 100
        loss_val = loss_val / n  # make mean loss

        if opts.rank == 0:
            if vis is not None:
                vis.line(X=torch.ones((1, 3)) * epoch,
                         Y=torch.Tensor([acc1, acc5, loss_val]).unsqueeze(0),
                         update='append',
                         win='test_loss_and_acc_for_{}'.format(opts.name),
                         opts=dict(x_label='epoch',
                                   y_label='test loss and acc',
                                   title='test loss and accuracy for {}'.format(opts.name),
                                   legend=['acc1', 'acc5', 'test_loss']))

            print("top-1 percentage :  {0:0.3f}%".format(acc1 * 100))
            print("top-5 percentage :  {0:0.3f}%".format(acc5 * 100))

            # xl_log_saver
            if opts.rank == 0:
                if xl_log_saver is not None:
                    if opts.num_classes == 1000:
                        xl_log_saver.insert_each_epoch(contents=(epoch, acc1, acc5, loss_val))
                    elif opts.num_classes == 10 or opts.num_classes == 100:
                        xl_log_saver.insert_each_epoch(contents=(epoch, acc1, loss_val))

            # set result_best
            if result_best is not None:
                if result_best['accuracy_top1'] < acc1:
                    print("update best model from {:.4f} to {:.4f}".format(result_best['accuracy_top1'], acc1))
                    result_best['epoch'] = epoch
                    result_best['accuracy_top1'] = acc1
                    result_best['val_loss'] = loss_val
                    if checkpoint is None:
                        checkpoint = {'epoch': epoch,
                                      'model_state_dict': model.state_dict()}
                    torch.save(checkpoint, os.path.join(opts.log_dir, opts.name, 'saves', opts.name + '.best.pth.tar'))
            return result_best