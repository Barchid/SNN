import argparse
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
    def __init__(self, T: int, learning_rate=1e-3, data_dir='data/mnist', batch_size: int = 8, size_noise=2):
        super().__init__()
        self.T = T
        self.learning_rate = learning_rate
        self.model = LeNet5(4)
        self.batch_size = batch_size
        self.size_noise = size_noise

    def forward(self, x):
        functional.reset_net(self.model)
        pred = self.model(x)
        return pred

    def train_dataloader(self):
        return DataLoader(LongtermImageDataset(is_train=True, size_noise=self.size_noise), batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(LongtermImageDataset(is_train=False, size_noise=self.size_noise), batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(LongtermImageDataset(is_train=False, size_noise=self.size_noise), batch_size=self.batch_size)

    def training_step(self, batch, batch_idx):
        x, y = batch

        neurals = []
        for im in x:
            neurals.append(saccade_coding(im, timesteps=self.T))

        x = torch.cat(neurals, dim=0).to(device)

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

        neurals = []
        for im in x:
            neurals.append(saccade_coding(im, timesteps=self.T))

        x = torch.cat(neurals, dim=0).to(device)

        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat.clone().detach(), dim=1)
        acc = accuracy(preds, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        neurals = []
        for im in x:
            neurals.append(saccade_coding(im, timesteps=self.T))

        x = torch.cat(neurals, dim=0).to(device)

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
    # parser = get_args()
    # args = parser.parse_args()
    pl.seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--size_noise', default=1, type=int, help="Size noise")
    args = parser.parse_args()

    SIZE_NOISE = args.size_noise

    module = MNISTClassification(20, learning_rate=1e-3, batch_size=48, size_noise=SIZE_NOISE)

    # ------------
    # training
    # ------------

    logger = pl.TensorBoardLogger("tb_logs", name=f"my_model_f{SIZE_NOISE}")
    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        max_epochs=10,
        progress_bar_refresh_rate=20,
        logger=logger
    )
    trainer.fit(module)


if __name__ == "__main__":
    main()
