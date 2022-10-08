import logging
import torch
from os import path as osp
from matplotlib import pyplot as plt
import torch.nn as nn
import cv2
from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options
from torchvision.utils import make_grid
import numpy as np
import torchvision.transforms.functional as F
def test_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create model
    model = build_model(opt)
    net_g = model.net_g
    conv_layers = {}
    for name, module in net_g.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            conv_layers[name] = module
            print(name)
    print(conv_layers)
    extraction_layer = conv_layers['extraction_layer.0']
    deconv_layer = conv_layers['deconv_layer.0']
    # print(weight.shape)
    layers = [extraction_layer, deconv_layer]
    #可视化卷积核
    for idx,layer in enumerate(layers):
        weight = layer.weight.data
        print(weight.shape)
        weight = weight.cpu().numpy()
        for i, w in enumerate(weight):
            w = np.squeeze(w)
            # print(w.shape)
            w = w - np.min(w)
            w = w / np.max(w)
            plt.subplot(7, 8, i+1)
            plt.axis('off')
            plt.imshow(w, cmap='gray')
        plt.savefig(f'./{idx}.png')


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
