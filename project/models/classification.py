import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from spikingjelly.clock_driven import neuron, functional, surrogate, layer

from project.models.utils import ConvBNLIF, LinearLIF


class LeNet5(nn.Module):
    """LeNet5 network for MNIST classification"""

    def __init__(self, out_channels):
        super(LeNet5, self).__init__()
        self.conv1 = ConvBNLIF(1, 12, kernel_size=5)
        self.conv2 = ConvBNLIF(12, 32, kernel_size=3, stride=2)  # 14
        self.conv3 = ConvBNLIF(32, 64, kernel_size=3, stride=2)  # 7
        self.flat = nn.Flatten(2)
        self.fc1 = LinearLIF(64 * 7 * 7, 64)
        self.fc_final = nn.Linear(64, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flat(x)
        x = self.fc1(x)
        out = self.fc_final(x.mean(0))
        return out
