import os

from torch.utils import data

import torchvision


class SVHN(data.Dataset):
    """
    A thin wrapper around torchvision.datasets.SVHN.

    The helper scripts only need train/test access, but we also accept "val"
    and map it to the official test split so the surrounding API stays simple.
    """

    def __init__(self, root, split="train", transform=None):
        assert split in ["train", "val", "test"]
        self.root_folder = os.path.join(root, "SVHN")
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
