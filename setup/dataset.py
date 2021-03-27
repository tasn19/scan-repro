from torch.utils.data import Dataset

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
