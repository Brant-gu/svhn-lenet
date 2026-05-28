import os

from torch.utils import data

import torchvision


class SVHN(data.Dataset):
    """
    A thin wrapper around torchvision.datasets.SVHN.

    Accepts "train", "val", or "test" split. "val" is mapped to the official
    test split so the same API can be used for both evaluation and validation.
    """

    def __init__(self, root, split="train", transform=None):
        assert split in ["train", "val", "test"]
        self.split = split
        self.transform = transform

        os.makedirs(os.path.expanduser(root), exist_ok=True)
        torchvision_split = "train" if split == "train" else "test"
        self.dataset = torchvision.datasets.SVHN(
            root=root,
            split=torchvision_split,
            transform=transform,
            download=True,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return img, int(label)
