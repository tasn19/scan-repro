import torch
from torch.utils.data import Dataset
import numpy as np

# Build a custom dataset containing a set of images and a set of the same images in augmented form
class CustomDataset(Dataset):
    def __init__(self, dataset, step, base_transform):
        transform = dataset.transform
        dataset.transform = None
        self.dataset = dataset
        self.step = step

        if step == "simclr":
            self.img_transform = base_transform
            self.augment_transform = transform

        else:
            self.img_transform = transform
            self.augment_transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        oriimg, label = self.dataset.__getitem__(index)
        img = self.img_transform(oriimg)
        augmented_img = self.augment_transform(oriimg)
        return img, augmented_img, label

# Dataset to return image with a neighbor
class NNDataset(Dataset):
    def __init__(self, dataset, indices, numNeighbors, step, base_transform):
        transform = dataset.transform
        self.dataset = dataset
        self.indices = indices[:, :numNeighbors + 1]  # take positions of knn + sample

        if step == 'scan':
            self.anchor_transform = base_transform
            self.neighbor_transform = transform

        else:
            self.anchor_transform = transform
            self.neighbor_transform = transform

        dataset.transform = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}
        anchorimg, lbl = self.dataset.__getitem__(index)

        neighbor_index = np.random.choice(self.indices[index], 1)[0]
        neighborimg, nlbl = self.dataset.__getitem__(neighbor_index)

        output['anchorimg'] = self.anchor_transform(anchorimg)
        output['neighborimg'] = self.neighbor_transform(neighborimg)

        output['possible_neighbors'] = torch.from_numpy(self.indices[index])
        output['target'] = lbl

        return output