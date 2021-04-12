import torch
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import errno
from augment import Augment, Cutout
from scipy.optimize import linear_sum_assignment
from sklearn import metrics


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
    "scan": {
     "Cropsize": 32,
     "Normalize": {"mean": (0.4914, 0.4822, 0.4465), "std": (0.2023, 0.1994, 0.2010)}},
     "NumStrongAugments": 4,
     "Cutout": {"numholes": 1, "length": 16, "random": True}
    }

# Transformations to pre-process image  # authors code: get_train_transformations in common_config
def get_transform(step):
    # step options: 'base', 'simclr', 'scan'
    if step == "base":
    transform = transforms.Compose(
      [transforms.RandomResizedCrop(tp[step]["RandomResizedCrop"]["size"], tp[step]["RandomResizedCrop"]["scale"]),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(tp[step]["Normalize"]["mean"], tp[step]["Normalize"]["std"])])

    if step == "simclr": # original simclr does not use Gaussian blur for CIFAR10
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

    # Augmentations for clustering and self-labeling steps: four randomly selected transformations from RandAugment.
    if step == "scan":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(tp[step]["Cropsize"]),
            Augment(tp[step]["NumStrongAugments"]]),
                    transforms.ToTensor(),
                    transforms.Normalize(tp[step]["Normalize"]["mean"], tp[step]["Normalize"]["std"]),
                    Cutout(n_holes=tp[step]["Cutout"]["numholes"],
                           length=tp[step]["Cutout"]["length"],
                           random=tp[step]["Cutout"]["random"])])
        ])


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


# Evaluate -------------------------------------------------------
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

@torch.no_grad()
def SCAN_evaluate(dataloader, model):
    model.eval()
    evalloss = []
    for batch in dataloader:
        imgs = batch["anchorimg"].to(device, non_blocking=True)
        nb = batch["neighborimg"].to(device, non_blocking=True)
        output = model(imgs)
        outputnb = model(nb)

        for anchor_out, neighbor_out in zip(output, outputnb):
            # anchor_out & neighbor_out have shape [128,10]
            print(anchor_out.size())
            print(anchor_out.type())
            loss = criterion(anchor_out, neighbor_out)
            # loss.cpu()

        print('evallossperbatch', loss)
        evalloss.append(loss)
        print(evalloss)

    finalloss = min(evalloss)

return finalloss

@torch.no_grad()
def get_predictions(dataloader, model): # for SCAN
    # predict class with neighbors
    model.eval()
    predictions = []
    probability = []
    targets = []
    evalloss = []

    for batch in dataloader:
        imgs = batch["anchorimg"].to(device, non_blocking=True)
        output = model(imgs)

        for i, out in enumerate(output):
            pred = torch.argmax(out, dim=1).cpu()
            predictions.append(pred)  # tensor[idx] crashes session
            prob = F.softmax(out, dim=1).cpu()
            probability.append(prob)

        targets.append(batch["target"])

    # concatenate batches
    predictions = torch.cat(predictions)
    probability = torch.cat(probability)
    targets = torch.cat(targets, dim=0)

    final_output = {'predictions': predictions, 'probabilities': probability, 'targets': targets}
    return final_output

@torch.no_grad()
def get_predictions_slbl(dataloader, model):
    # predict class with neighbors
    model.eval()
    predictions = []
    probability = []
    targets = []
    evalloss = []

    for i, (ims, lbls) in enumerate(dataloader):
        imgs = ims.to(device, non_blocking=True)
        output = model(imgs)

    for i, out in enumerate(output):
        pred = torch.argmax(out, dim=1).cpu()
        predictions.append(pred)  # tensor[idx] crashes session
        prob = F.softmax(out, dim=1).cpu()
        probability.append(prob)

    targets.append(lbls)

    # concatenate batches
    predictions = torch.cat(predictions)
    probability = torch.cat(probability)
    targets = torch.cat(targets, dim=0)

    final_output = {'predictions': predictions, 'probabilities': probability, 'targets': targets}
    return final_output

# paper code
def confusion_matrix(predictions, gt, class_names, output_file=None):
    # Plot confusion_matrix and store result to output_file
    import sklearn.metrics
    import matplotlib.pyplot as plt
    confusion_matrix = sklearn.metrics.confusion_matrix(gt, predictions)
    confusion_matrix = confusion_matrix / np.sum(confusion_matrix, 1)

    fig, axes = plt.subplots(1)
    plt.imshow(confusion_matrix, cmap='Blues')
    axes.set_xticks([i for i in range(len(class_names))])
    axes.set_yticks([i for i in range(len(class_names))])
    axes.set_xticklabels(class_names, ha='right', fontsize=8, rotation=40)
    axes.set_yticklabels(class_names, ha='right', fontsize=8)

    for (i, j), z in np.ndenumerate(confusion_matrix):
        if i == j:
            axes.text(j, i, '%d' % (100 * z), ha='center', va='center', color='white', fontsize=6)
        else:
            pass

    plt.tight_layout()
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

# Hungarian matching algorithm - paper code
@torch.no_grad()   # removed subhead_index parameter
def hungarian_evaluate(all_predictions, class_names=None,
                     compute_purity=True, compute_confusion_matrix=True,
                     confusion_matrix_file=None):
  # Evaluate model based on hungarian matching between predicted cluster assignment and gt classes.
  # This is computed only for the passed subhead index.

  # Hungarian matching
  head = all_predictions# removed subhead_index since only using one head
  targets = head['targets'].to(device)
  predictions = head['predictions'].to(device)
  probs = head['probabilities'].to(device)
  num_classes = torch.unique(targets).numel()
  num_elems = targets.size(0)

  match = _hungarian_match(predictions, targets, preds_k=num_classes, targets_k=num_classes)
  reordered_preds = torch.zeros(num_elems, dtype=predictions.dtype).cuda()
  for pred_i, target_i in match:
      reordered_preds[predictions == int(pred_i)] = int(target_i)

  # Gather performance metrics
  acc = int((reordered_preds == targets).sum()) / float(num_elems)
  nmi = metrics.normalized_mutual_info_score(targets.cpu().numpy(), predictions.cpu().numpy())
  ari = metrics.adjusted_rand_score(targets.cpu().numpy(), predictions.cpu().numpy())

  _, preds_top5 = probs.topk(5, 1, largest=True)
  reordered_preds_top5 = torch.zeros_like(preds_top5)
  for pred_i, target_i in match:
      reordered_preds_top5[preds_top5 == int(pred_i)] = int(target_i)
  correct_top5_binary = reordered_preds_top5.eq(targets.view(-1, 1).expand_as(reordered_preds_top5))
  top5 = float(correct_top5_binary.sum()) / float(num_elems)

  # Compute confusion matrix
  if compute_confusion_matrix:
      confusion_matrix(reordered_preds.cpu().numpy(), targets.cpu().numpy(),
                       class_names, confusion_matrix_file)

  return {'ACC': acc, 'ARI': ari, 'NMI': nmi, 'ACC Top-5': top5, 'hungarian_match': match}

@torch.no_grad()
def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
  # Based on implementation from IIC
  num_samples = flat_targets.shape[0]

  assert (preds_k == targets_k)  # one to one
  num_k = preds_k
  num_correct = np.zeros((num_k, num_k))

  for c1 in range(num_k):
      for c2 in range(num_k):
          # elementwise, so each sample contributes once
          votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
          num_correct[c1, c2] = votes

  # num_correct is small
  match = linear_sum_assignment(num_samples - num_correct)
  match = np.array(list(zip(*match)))

  # return as list of tuples, out_c to gt_c
  res = []
  for out_c, gt_c in match:
      res.append((out_c, gt_c))

  return res