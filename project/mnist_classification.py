import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torchvision.datasets.mnist import MNIST
from torchvision import transforms
from spikingjelly.clock_driven import encoding

from models.classification import LeNet5
from args import get_args


class MNISTClassification(pl.LightningModule):
    def __init__(self, T: int, learning_rate=1e-3):
        super().__init__()
        self.T = T
        self.learning_rate = learning_rate
        self.model = LeNet5()

    def forward(self, x):
        pred = self.model(x)
        return pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        rate_x = 
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('valid_loss', loss, on_step=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def main():
    pass
