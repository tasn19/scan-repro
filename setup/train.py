from utils import AverageMeter, ProgressMeter

# Determine 20 nearest neighbors with SimClR instance discrimination task
def SimCLR_train(dataloader, model, epoch, criterion, optimizer):
    # Record progress
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(dataloader), [losses], prefix="Epoch: [{}]".format(epoch))

    model.train()
    for i, (ims, aug_ims, lbls) in enumerate(dataloader):
        # print(ims.size())
        batch, channel, h, w = ims.size()
        x_i = ims.unsqueeze(1)
        x_j = aug_ims.unsqueeze(1)
        x_i = x_i.view(-1, channel, h, w)  # in model images processed independently so batch size doesn't matter
        x_i = x_i.cuda(non_blocking=True)

        x_j = x_j.view(-1, channel, h, w)
        x_j = x_j.cuda(non_blocking=True)
        targets = lbls.cuda(non_blocking=True)  # need?
        z_i = model(x_i)  # try concatenation x_i and x_j?
        z_j = model(x_j)
        loss = criterion(z_i, z_j)
        # update losses
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            progress.display(i)