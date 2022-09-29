import os
import torch


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

    val_avg_loss = 0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            image = data[0].to(int(opts.gpu_ids[opts.rank]))
            label = data[1].to(int(opts.gpu_ids[opts.rank]))

            output = model(image)
            loss = criterion(output, label)
            val_avg_loss += loss.item()

            # rank 1
            _, pred = torch.max(output, 1)
            total += label.size(0)
            correct_top1 += (pred == label).sum().item()

            # ------------------------------------------------------------------------------
            # rank 5
            _, rank5 = output.topk(5, 1, True, True)
            rank5 = rank5.t()
            correct5 = rank5.eq(label.view(1, -1).expand_as(rank5))

            # ------------------------------------------------------------------------------
            for k in range(5):
                correct_k = correct5[:k+1].reshape(-1).float().sum(0, keepdim=True)

            correct_top5 += correct_k.item()

        accuracy_top1 = correct_top1 / total
        accuracy_top5 = correct_top5 / total
        val_avg_loss = val_avg_loss / len(test_loader)  # make mean loss

        if opts.rank == 0:
            if vis is not None:
                vis.line(X=torch.ones((1, 3)) * epoch,
                         Y=torch.Tensor([accuracy_top1, accuracy_top5, val_avg_loss]).unsqueeze(0),
                         update='append',
                         win='test_loss_acc',
                         opts=dict(x_label='epoch',
                                   y_label='test_loss and acc',
                                   title='test_loss and accuracy',
                                   legend=['accuracy_top1', 'accuracy_top5', 'avg_loss']))
            print("")
            print("top-1 percentage :  {0:0.3f}%".format(correct_top1 / total * 100))
            print("top-5 percentage :  {0:0.3f}%".format(correct_top5 / total * 100))

            # xl_log_saver
            if opts.rank == 0:
                if xl_log_saver is not None:
                    if opts.num_classes == 1000:
                        xl_log_saver.insert_each_epoch(contents=(epoch, accuracy_top1, accuracy_top5, val_avg_loss))
                    elif opts.num_classes == 10 or opts.num_classes == 100:
                        xl_log_saver.insert_each_epoch(contents=(epoch, accuracy_top1, val_avg_loss))

            # set result_best
            if result_best is not None:
                if result_best['accuracy_top1'] < correct_top1:
                    print("update best model")
                    result_best['epoch'] = epoch
                    result_best['accuracy_top1'] = correct_top1
                    result_best['val_loss'] = val_avg_loss

                    if checkpoint is None:
                        checkpoint = {'epoch': epoch,
                                      'model_state_dict': model.state_dict()}
                    torch.save(checkpoint, os.path.join(opts.log_dir, opts.name, 'saves', opts.name + '.best.pth.tar'))

            return result_best