import os
from typing import Any, Optional, Tuple, Union
from typing_extensions import Self
import numpy as np
import pandas as pd
from skimage import io

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


class FocusDataSet(Dataset):
    """Dataset for z-stacked images of neglected tropical diseaeses."""

    def __init__(self, csv_file, root_dir, transform=None):
        """Initialize focus satck dataset.

        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.metadata = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            int: the length
        """
        return len(self.metadata)

    def __getitem__(self, idx):
        """Get one items from the dataset.

        Args:
            idx (int) The index of the sample that is to be retrieved

        Returns:
            Item/Items which is a dictionary containing "image" and "focus_value"
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.metadata.iloc[idx, 1])
        image = io.imread(img_name)
        focus_value = self.metadata.iloc[idx, 5]
        sample = {"image": image, "focus_value": focus_value}

        if self.transform:
            sample = self.transform(sample)

        return sample

