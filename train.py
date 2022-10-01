import os
import time
import torch


def train_one_epoch(epoch, vis, train_loader, model, optimizer, criterion, scheduler, opts):
    print('Training of epoch [{}]'.format(epoch))

    model.train()
    tic = time.time()

    for i, (images, labels) in enumerate(train_loader):

        images = images.to(int(opts.gpu_ids[opts.rank]))
        labels = labels.to(int(opts.gpu_ids[opts.rank]))

        outputs = model(images)
        loss = criterion(outputs, labels)

        # ----------- update -----------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # get lr
        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        # time
        toc = time.time()

        # visualization
        if i % opts.vis_step == 0 and opts.rank == 0:
            print('Epoch [{0}/{1}], Iter [{2}/{3}], Loss: {4:.4f}, lr: {5:.5f}, Time: {6:.2f}'.format(epoch,
                                                                                                      opts.epoch,
                                                                                                      i,
                                                                                                      len(train_loader),
                                                                                                      loss.item(),
                                                                                                      lr,
                                                                                                      toc - tic))

            vis.line(X=torch.ones((1, 1)) * i + epoch * len(train_loader),
                     Y=torch.Tensor([loss]).unsqueeze(0),
                     update='append',
                     win='loss',
                     opts=dict(x_label='step',
                               y_label='loss',
                               title='train loss for {}'.format(opts.name),
                               legend=['total_loss']))

    # save pth file
    if opts.rank == 0:

        save_path = os.path.join(opts.log_dir, opts.name, 'saves')
        os.makedirs(save_path, exist_ok=True)

        checkpoint = {'epoch': epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'scheduler_state_dict': scheduler.state_dict()}

        if epoch > opts.save_step:
            torch.save(checkpoint, os.path.join(save_path, opts.name + '.{}.pth.tar'.format(epoch)))
            print("save .pth")