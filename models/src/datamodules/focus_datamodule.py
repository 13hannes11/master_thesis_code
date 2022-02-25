import os
from typing import Optional, Tuple
import pandas as pd
from skimage import io

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
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
            sample["image"] = self.transform(sample["image"])

        return sample


class FocusDataModule(LightningDataModule):
    """
    LightningDataModule for FocusStack dataset.
    """

    def __init__(
        self,
        data_dir: str = "data/",
        csv_file: str = "data/metadata.csv",
        train_val_test_split_percentage: Tuple[int, int, int] = (0.75, 0.15, 0.15),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.ConvertImageDtype(torch.float)]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """This method is not implemented as of yet.

        Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`."""

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = FocusDataSet(
                self.hparams.csv_file, self.hparams.data_dir, transform=self.transforms
            )
            train_length = int(
                len(dataset) * self.hparams.train_val_test_split_percentage[0]
            )
            val_length = int(
                len(dataset) * self.hparams.train_val_test_split_percentage[1]
            )
            test_length = len(dataset) - val_length - train_length

            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=(train_length, test_length, val_length),
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
