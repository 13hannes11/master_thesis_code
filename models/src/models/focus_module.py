from typing import Any, List

import torch
from torch import nn
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanAbsoluteError, MinMetric
from torchmetrics.classification.accuracy import Accuracy


class SimpleDenseNet(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(hparams["input_size"], hparams["lin1_size"]),
            nn.BatchNorm1d(hparams["lin1_size"]),
            nn.ReLU(),
            nn.Linear(hparams["lin1_size"], hparams["lin2_size"]),
            nn.BatchNorm1d(hparams["lin2_size"]),
            nn.ReLU(),
            nn.Linear(hparams["lin2_size"], hparams["lin3_size"]),
            nn.BatchNorm1d(hparams["lin3_size"]),
            nn.ReLU(),
            nn.Linear(hparams["lin3_size"], hparams["output_size"]),
        )

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        x = x.view(batch_size, -1)

        return self.model(x)


class FocusLitModule(LightningModule):
    """
    Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        input_size: int = 75 * 75 * 3,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 1,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.model = SimpleDenseNet(hparams=self.hparams)

        # loss function
        self.criterion = torch.nn.L1Loss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()

        # for logging best so far validation accuracy
        self.val_mae_best = MinMetric()

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        x = batch["image"]
        y = batch["focus_value"]
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.squeeze(logits)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        mae = self.train_mae(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/mae", mae, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        mae = self.val_mae(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/mae", mae, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        mae = self.val_mae.compute()  # get val accuracy from current epoch
        self.val_mae_best.update(mae)
        self.log(
            "val/mae_best", self.val_mae_best.compute(), on_epoch=True, prog_bar=True
        )

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        mae = self.test_mae(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/mae", mae, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_mae.reset()
        self.test_mae.reset()
        self.val_mae.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers.

        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
