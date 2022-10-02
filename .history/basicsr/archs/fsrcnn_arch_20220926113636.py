from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class FSRCNN:
    """FSRCNN network structure.

    Paper: Accelerating the Super-Resolution Convolutional Neural Network
    https://arxiv.org/abs/1608.00367

    Args:
        in_channels (int): Channel number of inputs. Default: 3.
        out_channels (int): Channel number of outputs. Default: 3.
        mid_channels (int): Channel number of intermediate features. Default: 64.
        kernel_size (int): Kernel size of the first and last conv layer. Default: 5.
        upscale_factor (int): Upsampling factor. Support x2, x3 and x4. Default: 4.
    """

    def __init__(self, in_channels=3, out_channels=3, mid_channels=64, kernel_size=5, upscale_factor=4):
        super(FSRCNN, self).__init__()
        self.upscale_factor = upscale_factor

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, padding=0)
        self.conv4 = nn.Conv2d(mid_channels, out_channels * (upscale_factor ** 2), kernel_size=kernel_size,
                               padding=kernel_size // 2)

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x