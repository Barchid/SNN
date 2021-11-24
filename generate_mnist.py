import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, random_split
from torchvision import transforms
from torchvision.datasets.mnist import MNIST
from project.utils.neural_coding import rate_coding, saccade_coding
from matplotlib import pyplot as plt
import snntorch.spikeplot as splt
import random

A_CHAR = 1
B_CHAR = 8


class LongtermImageDataset(Dataset):
    def __init__(self, is_train=True, size_noise=2):
        self.data = generate_longterm(is_train, size_noise=size_noise)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]


def generate_longterm(is_train=True, size_noise=2):

    trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    mnist_full = MNIST('data/mnist', train=is_train, download=True, transform=trans)
    mnist_table = []
    for (x, y) in mnist_full:
        mnist_table.append((x, y))

    random.shuffle(mnist_table)

    dataset_images = []

    # FOR [image]
    sample = []
    orig_class = []
    for (x, y) in mnist_table:

        # IF [we can add this]
        if y == A_CHAR or y == B_CHAR:
            sample.append(x)
            orig_class.append(y)

        if len(sample) >= 2 + size_noise:
            first = orig_class[0]
            sec = orig_class[-1]

            if first == A_CHAR and sec == A_CHAR:
                cls = 0
            elif first == A_CHAR and sec == B_CHAR:
                cls = 1
            elif first == B_CHAR and sec == A_CHAR:
                cls = 2
            else:
                cls = 3

            dataset_images.append((sample, cls))
            sample = []
            orig_class = []

    return dataset_images


if __name__ == '__main__':
    da = LongtermImageDataset()
    da_lo = DataLoader(da, batch_size=4)

    for (sam, lab) in da_lo:
        print(len(sam))

        neurals = []
        for im in sam:
            neurals.append(saccade_coding(im, timesteps=20))

        x = torch.cat(neurals, dim=0)

        #  Index into a single sample from a minibatch
        spike_data_sample = x[:, 0, 0, :, :]
        print(spike_data_sample.size())

        #  Plot
        fig, ax = plt.subplots()
        anim = splt.animator(spike_data_sample, fig, ax)

        #  Save as a gif
        anim.save("spike_mnist.mp4")
        print(lab[0])
        exit()
