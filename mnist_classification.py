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

from project.models.classification import LeNet5
from project.utils.neural_coding import rate_coding
from project.args import get_args

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MNISTClassification(pl.LightningModule):
    def __init__(self, T: int, learning_rate=1e-3, data_dir='data/mnist', batch_size: int = 8):
        super().__init__()
        self.T = T
        self.learning_rate = learning_rate
        self.model = LeNet5()
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        self.batch_size = batch_size

    def forward(self, x):
        functional.reset_net(self.model)
        pred = self.model(x)
        return pred

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def training_step(self, batch, batch_idx):
        x, y = batch
        rate_x = rate_coding(x, self.T)
        y_hat = self(rate_x)
        loss = F.cross_entropy(y_hat, y)
        # loss = F.mse_loss(y_hat, y)
        preds = torch.argmax(y_hat.clone().detach(), dim=1)
        acc = accuracy(preds, y)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        rate_x = rate_coding(x, self.T)
        print(rate_x.shape)
        y_hat = self(rate_x)
        print(y_hat.shape)
        print(y.shape)
        loss = F.cross_entropy(y_hat, y)
        print(loss)
        exit()
        preds = torch.argmax(y_hat.clone().detach(), dim=1)
        acc = accuracy(preds, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        rate_x = rate_coding(x, self.T)
        y_hat = self(rate_x)
        loss = F.mse_loss(y_hat, y)
        preds = torch.argmax(y_hat.clone().detach(), dim=1)
        acc = accuracy(preds, y)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        self.log('test_acc', acc, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)


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
