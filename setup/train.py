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

def SCAN_train(dataloader, model, epoch, criterion, optimizer):
  # record progress
  losses = AverageMeter('SCAN Loss', ':.4e')
  progress = ProgressMeter(len(dataloader), [losses], prefix="Epoch: [{}]".format(epoch))

  model.train()
  for i, batch in enumerate(dataloader):
    # forward pass
    anchors = batch['anchorimg'].to(device, non_blocking=True) # 128 imgs
    neighbors = batch['neighborimg'].to(device, non_blocking=True) # a neighbor for each img

    # calculate gradient for backpropagation
    output_anchors = model(anchors) # weights for training with each img. each of 128 (along len) has 10 rows
    output_neighbors = model(neighbors) # weights for training with each neighbor

    # calculate loss  CHECK/CHANGE
    for anchor_out, neighbor_out in zip(output_anchors, output_neighbors):
      # anchor_out & neighbor_out have shape [128,10]
      loss = criterion(anchor_out, neighbor_out)

    # update losses
    losses.update(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 25 == 0:
      progress.display(i)


def selflabel_train(dataloader, model, epoch, criterion, optimizer):
    # record progress
    losses = AverageMeter('Self Label Loss', ':.4e')
    progress = ProgressMeter(len(dataloader), [losses], prefix="Epoch: [{}]".format(epoch))

    model.train()

    for i, (ims, aug_ims, lbls) in enumerate(dataloader):
        # print(ims.size())
        imgs = ims.to(device, non_blocking=True)
        aug_imgs = aug_ims.to(device, non_blocking=True)

        with torch.no_grad():
            output_imgs = model(imgs)
            output_imgs = output_imgs[0]  # tensor size [batchsize, numClasses]
        output_aug = model(aug_imgs)
        output_aug = output_aug[0]  # tensor size [batchsize, numClasses]

        loss = criterion(output_imgs, output_aug)

        # update losses
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            progress.display(i)