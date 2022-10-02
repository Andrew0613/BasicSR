from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init
from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import default_init_weights
import math
@ARCH_REGISTRY.register()
class FSRCNN(nn.Module):
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

    def __init__(self, in_channels=1, out_channels=1, d=56,s=12,m=4, upscale=3,init_type="kaiming"):
        super(FSRCNN, self).__init__()
        self.upscale_factor = upscale
        self.extraction_layer = [nn.Conv2d(in_channels, d, kernel_size=5, padding=2),nn.PReLU(d)]
        mid_layers = []
        shrinking_layer = [nn.Conv2d(d, s, kernel_size=1, padding=0),nn.PReLU(s)]
        mid_layers.extend(shrinking_layer)
        for _ in range(m):
            mid_layers.append(nn.Conv2d(s, s, kernel_size=3, padding=1))
            mid_layers.append(nn.PReLU(s))
        expanding_layer = [nn.Conv2d(s, d, kernel_size=1, padding=0),nn.PReLU(d)]
        mid_layers.extend(expanding_layer)
        self.mid_layers = nn.Sequential(*mid_layers)
        self.deconv_layer = [nn.ConvTranspose2d(d, out_channels, kernel_size=9, stride=upscale, padding=3, output_padding=0),nn.PReLU(out_channels)]
        if init_type == "kaiming":
            self.init_weights_MSRA()
        elif init_type == "xavier":
            self.init_weights_Xavier()
    def init_weights_MSRA(self):
        for L in self.extraction_layer:
            if isinstance(L, nn.Conv2d):
                L.weight.data.normal_(mean=0.0, std=math.sqrt(2 / (L.out_channels * L.weight.data[0][0].numel())))
                L.bias.data.zero_()
        for L in self.mid_layers:
            if isinstance(L, nn.Conv2d):
                L.weight.data.normal_(mean=0.0, std=math.sqrt(2 / (L.out_channels * L.weight.data[0][0].numel())))
                L.bias.data.zero_()
        for L in self.deconv_layer:
                if isinstance(L, nn.ConvTranspose2d):
                    L.weight.data.normal_(mean=0.0, std=0.001)
                    L.bias.data.zero_()

    def init_weights_Xavier(self):
        for L in self.extraction_layer:
            if isinstance(L, nn.Conv2d):
                L.weight.data.normal_(mean=0.0, std=math.sqrt(2 / (L.out_channels + L.in_channels)))
                L.bias.data.zero_()
        for L in self.mid_part:
            if isinstance(L, nn.Conv2d):
                L.weight.data.normal_(mean=0.0, std=math.sqrt(2 / (L.out_channels + L.in_channels)))
                L.bias.data.zero_()
        for L in self.deconv_layer:
                if isinstance(L, nn.ConvTranspose2d):
                    L.weight.data.normal_(mean=0.0, std=0.001)
                    L.bias.data.zero_()
    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        x = self.extraction_layer(x)
        x = self.mid_layers(x)
        x = self.deconv_layer(x)
        return x