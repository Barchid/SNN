import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from spikingjelly.clock_driven import neuron, functional, surrogate, layer


class ConvBNLIF(nn.Sequential):
    """Some Information about ConvBNLIF"""

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1):
        super(ConvBNLIF, self).__init__()
        padding = kernel_size // 2 + dilation - 1

        self.add_module('conv_bn', layer.SeqToANNContainer(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      # no bias if we use BatchNorm (because it has a BN itself)
                      bias=False,
                      dilation=dilation,
                      stride=stride),
            nn.BatchNorm2d(out_channels)
        ))

        self.add_module('lif', neuron.MultiStepLIFNode(tau=2.0, v_threshold=1.0, v_reset=0.,
                                                       surrogate_function=surrogate.ATan(alpha=2.0, spiking=True)))


class ConvLIF(nn.Sequential):
    """Some Information about ConvBNLIF"""

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1):
        super(ConvLIF, self).__init__()
        padding = kernel_size // 2 + dilation - 1

        self.add_module('conv_bn', layer.SeqToANNContainer(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      # no bias if we use BatchNorm (because it has a BN itself)
                      bias=True,
                      dilation=dilation,
                      stride=stride)
        ))

        self.add_module('lif', neuron.MultiStepLIFNode(tau=2.0, v_threshold=1.0, v_reset=0.,
                                                       surrogate_function=surrogate.ATan(alpha=2.0, spiking=True)))


class LinearLIF(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(LinearLIF, self).__init__()
        
        self.add_module('fc', layer.SeqToANNContainer(
            nn.Linear(in_channels, out_channels)
        ))

        self.add_module('lif', neuron.MultiStepLIFNode(tau=2.0, v_threshold=1.0, v_reset=0.,
                                                       surrogate_function=surrogate.ATan(alpha=2.0, spiking=True)))
