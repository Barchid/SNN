import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split
from torchvision.datasets.mnist import MNIST
from torchvision import transforms
from spikingjelly.clock_driven import encoding
from spikingjelly.clock_driven import neuron, functional, surrogate, layer
from torchmetrics.functional import accuracy
from generate_mnist import LongtermImageDataset

from project.models.classification import LeNet5
from project.utils.neural_coding import rate_coding, saccade_coding
from project.args import get_args

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MNISTClassification(pl.LightningModule):
    def __init__(self, T: int, learning_rate=1e-3, data_dir='data/mnist', batch_size: int = 8):
        super().__init__()
        self.T = T
        self.learning_rate = learning_rate
        self.model = LeNet5(4)
        self.batch_size = batch_size

    def forward(self, x):
        functional.reset_net(self.model)
        pred = self.model(x)
        return pred

    def train_dataloader(self):
        return DataLoader(LongtermImageDataset(is_train=True))

    def val_dataloader(self):
        return DataLoader(LongtermImageDataset(is_train=False))

    def test_dataloader(self):
        return DataLoader(LongtermImageDataset(is_train=False))

    def training_step(self, batch, batch_idx):
        x, y = batch

        (im1, im2, im3) = x

        neural1 = saccade_coding(x[0], timesteps=self.T)
        neural2 = saccade_coding(x[1], timesteps=self.T)
        neural3 = saccade_coding(x[2], timesteps=self.T)
        x = torch.cat([neural1, neural2, neural3], dim=0)

        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        # loss = F.mse_loss(y_hat, y)
        preds = torch.argmax(y_hat.clone().detach(), dim=1)
        acc = accuracy(preds, y)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        (im1, im2, im3) = x

        neural1 = saccade_coding(x[0], timesteps=self.T)
        neural2 = saccade_coding(x[1], timesteps=self.T)
        neural3 = saccade_coding(x[2], timesteps=self.T)
        x = torch.cat([neural1, neural2, neural3], dim=0)

        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        # loss = F.mse_loss(y_hat, y)
        preds = torch.argmax(y_hat.clone().detach(), dim=1)
        acc = accuracy(preds, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        (im1, im2, im3) = x

        neural1 = saccade_coding(x[0], timesteps=self.T)
        neural2 = saccade_coding(x[1], timesteps=self.T)
        neural3 = saccade_coding(x[2], timesteps=self.T)
        x = torch.cat([neural1, neural2, neural3], dim=0)

        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        # loss = F.mse_loss(y_hat, y)
        preds = torch.argmax(y_hat.clone().detach(), dim=1)
        acc = accuracy(preds, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def main():
    parser = get_args()
    args = parser.parse_args()
    pl.seed_everything(args.seed)

    module = MNISTClassification(args.timesteps, learning_rate=args.lr, batch_size=args.batch_size)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        max_epochs=args.epochs,
        progress_bar_refresh_rate=20,
    )
    trainer.fit(module)


if __name__ == "__main__":
    main()
