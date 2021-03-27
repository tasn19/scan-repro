import torch
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import errno

# Create dictionary of transformation parameters (tp)
tp = {"base": {
    "RandomResizedCrop":{"size": 32, "scale": (0.2, 1.0)},
    "Normalize": {"mean": (0.4914, 0.4822, 0.4465), "std": (0.2023, 0.1994, 0.2010)}},
    "simclr": {
    "RandomResizedCrop":{"size": 32, "scale": (0.2, 1.0)},
    "RandomColorJitter":{"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1, "p": 0.8},
    "RandomGrayscale":{"p": 0.2},
    "Normalize": {"mean": (0.4914, 0.4822, 0.4465), "std": (0.2023, 0.1994, 0.2010)}},
    "validate": {
      "Cropsize": 32,
      "Normalize": {"mean": (0.4914, 0.4822, 0.4465), "std": (0.2023, 0.1994, 0.2010)}}
    }

# Transformations to pre-process image  # authors code: get_train_transformations in common_config
def get_transform(step):
  # step options: 'base', 'simclr'
  if step == "base":
    transform = transforms.Compose(
      [transforms.RandomResizedCrop(tp[step]["RandomResizedCrop"]["size"], tp[step]["RandomResizedCrop"]["scale"]),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(tp[step]["Normalize"]["mean"], tp[step]["Normalize"]["std"])])

  if step == "simclr": # what abt Gaussian blur ??
    transform = transforms.Compose(
        [transforms.RandomResizedCrop(tp[step]["RandomResizedCrop"]["size"], tp[step]["RandomResizedCrop"]["scale"]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(tp[step]["RandomColorJitter"]["brightness"], tp[step]["RandomColorJitter"]["contrast"],
                                  tp[step]["RandomColorJitter"]["saturation"], tp[step]["RandomColorJitter"]["hue"])],
                                tp[step]["RandomColorJitter"]["p"]),
        transforms.RandomGrayscale(tp[step]["RandomGrayscale"]["p"]),
        transforms.ToTensor(),
        transforms.Normalize(tp[step]["Normalize"]["mean"], tp[step]["Normalize"]["std"])]
    )

  if step == "validate":
    transform = transforms.Compose([
            transforms.CenterCrop(tp[step]["Cropsize"]),
            transforms.ToTensor(),
            transforms.Normalize(tp[step]["Normalize"]["mean"], tp[step]["Normalize"]["std"])])
  return transform

# paper code
class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

# paper code
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


# Evaluate
# paper code
@torch.no_grad()
def contrastive_evaluate(val_loader, model, memory_bank):
    top1 = AverageMeter('Acc@1', ':6.2f')
    model.eval()

    for im,lbl in val_loader:
        images = im.cuda(non_blocking=True)
        target = lbl.cuda(non_blocking=True)

        output = model(images)
        output = memory_bank.weighted_knn(output)

        acc1 = 100*torch.mean(torch.eq(output, target).float())
        top1.update(acc1.item(), images.size(0))

    return top1.avg