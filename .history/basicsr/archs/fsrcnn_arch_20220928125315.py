from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init
from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import default_init_weights

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

    def __init__(self, in_channels=1, out_channels=1, d=56,s=12,m=4, upscale_factor=3,init_type="kaiming"):
        super(FSRCNN, self).__init__()
        self.upscale_factor = upscale_factor
        self.extraction_layer = [nn.Conv2d(in_channels, d, kernel_size=5, padding=2),nn.PReLU(d)]

        self.shrinking_layer = [nn.Conv2d(d, s, kernel_size=1, padding=0),nn.PReLU(s)]

        mapping_layer = []
        for _ in range(m):
            mapping_layer.append(nn.Conv2d(s, s, kernel_size=3, padding=1))
            mapping_layer.append(nn.PReLU(s))
        self.mapping_layer = nn.Sequential(*mapping_layer)
        self.expanding_layer = [nn.Conv2d(s, d, kernel_size=1, padding=0),nn.PReLU(d)]

        self.deconv_layer = [nn.ConvTranspose2d(d, out_channels, kernel_size=9, stride=upscale_factor, padding=3, output_padding=0),nn.PReLU(out_channels)]

        # self.fsrcnn = self.init_weights(self.fsrcnn, init_type=init_type)
    def init_weights(net, init_type='normal', init_gain=0.02):
        """Initialize network weights.

        Parameters:
            net (network)   -- network to be initialized
            init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
            init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

        We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
        work better for some applications. Feel free to try yourself.
        """
        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)  # apply the initialization function <init_func>
    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        x = self.extraction_layer(x)
        x = self.shrinking_layer(x)
        x = self.mapping_layer(x)
        x = self.expanding_layer(x)
        x = self.deconv_layer(x)
        return x