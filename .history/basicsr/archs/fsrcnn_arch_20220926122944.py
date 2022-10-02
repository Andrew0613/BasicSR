from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import default_init_weights

@ARCH_REGISTRY.register()
class FSRCNN:
    """FSRCNN network structure.

    Paper: Accelerating the Super-Resolution Convolutional Neural Network
    https://arxiv.org/abs/1608.00367

    Args:
        in_channels (int): Channel number of inputs. Default: 1.
        out_channels (int): Channel number of outputs. Default: 1.
        mid_channels (int): Channel number of intermediate features. Default: 64.
        kernel_size (int): Kernel size of the first and last conv layer. Default: 5.
        upscale_factor (int): Upsampling factor. Support x2, x3 and x4. Default: 4.
    Notes:
        Conv(fi,ni,ci) denotes a convolutional layer with fi size of filters, ni number of filters,which can also be considered as output channel, and ci number of channels.
        Deconv(fi,ni,ci) denotes a deconvolutional layer with fi size of filters, ni number of filters, which can also be considered as output channel, and ci number of channels.
        Network structure of FSRCNN:
            Conv(5,d,1)-PReLU-Conv(1,s,d)-PReLu-m*Conv*(3,s,s)-PReLU-Conv(1,d,s)-PReLU-DeConv(9,1,d)
        where d,s,m are three sensitive variables. d is the number of dimension of LR feature, s is the number of shrinking filters, and m is the mapping depth.
    """

    def __init__(self, in_channels=1, out_channels=1, d=56,s=12,m=4, upscale_factor=3):
        super(FSRCNN, self).__init__()
        self.upscale_factor = upscale_factor
        layers = []
        feature_extraction_layer = [nn.Conv2d(in_channels, d, kernel_size=5, padding=0),nn.PReLU(d)]
        layers.extend(feature_extraction_layer)
        shrinking_layer = [nn.Conv2d(d, s, kernel_size=1, padding=0),nn.PReLU(s)]
        layers.extend(shrinking_layer)
        mapping_layer = []
        for _ in range(m):
            mapping_layer.append(nn.Conv2d(s, s, kernel_size=3, padding=1))
            mapping_layer.append(nn.PReLU(s))
        layers.extend(mapping_layer)
        expanding_layer = [nn.Conv2d(s, d, kernel_size=1, padding=0),nn.PReLU(d)]
        layers.extend(expanding_layer)
        deconv_layer = [nn.ConvTranspose2d(d, out_channels, kernel_size=9, stride=upscale_factor, padding=0, output_padding=0),nn.PReLU(out_channels)]
        layers.extend(deconv_layer)
        self.fsrcnn = nn.Sequential(*layers)

    def forward(self, x):
        pass