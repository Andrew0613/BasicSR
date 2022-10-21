import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
from torch.nn import init
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel
import torch.optim as optim
from copy import deepcopy
@MODEL_REGISTRY.register()
class FSRCNNModel(SRModel):
    """FSRCNN Model for single image super-resolution."""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_fsrcnn_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_fsrcnn_training_settings()
    def load_fsrcnn_network(self, net, load_path, strict=True, param_key='params',only_train_last_layer=False):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        logger = get_root_logger()
        net = self.get_bare_model(net)
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
                logger.info('Loading: params_ema does not exist, use params.')
            load_net = load_net[param_key]
        logger.info(f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            print("k",k)
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)
    def init_fsrcnn_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_fsrcnn_optimizers()
        # self.setup_optimizers()
        is_scheduler = train_opt.get('scheduler', None)
        if is_scheduler:
            self.setup_schedulers()
    def setup_fsrcnn_optimizers(self):
        """
        Setup optimizers for FSRCNN.
        卷积层和反卷积层的学习率不同，需要手写
        """
        train_opt = self.opt['train']
        optimizer_type = train_opt['optim_g'].get('type', 'SGD')
        lr = train_opt['optim_g']['lr']
        is_finetune = train_opt['optim_g'].get('is_finetune', False)
        mom = train_opt['optim_g'].get('momentum', 0.9)
        wd = train_opt['optim_g'].get('weight_decay', 1e-4)
        conv_key = []
        conv_value = []
        deconv_key = []
        deconv_value = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                if 'deconv' in k:
                    deconv_value.append(v)
                    deconv_key.append(k)
                else:
                    conv_value.append(v)
                    conv_key.append(k)
        #使用sgd优化器
        if is_finetune:
            if optimizer_type == 'SGD':
                self.optimizer_g = optim.SGD([
                    {'params': conv_value},
                    {'params': deconv_value, 'lr': lr * 0.1}
                ], lr=lr,momentum=mom, weight_decay=wd)
            elif optimizer_type == 'Adam':
                self.optimizer_g = optim.Adam([
                    {'params': conv_value},
                    {'params': deconv_value,'lr': lr * 0.1},
                ], lr=lr,betas=(mom, 0.999), weight_decay=wd)
        else:
            if optimizer_type == 'SGD':
                self.optimizer_g = optim.SGD([
                    {'params': conv_value},
                    {'params': deconv_value,'lr': lr * 0.1},
                ], lr=lr,momentum=mom, weight_decay=wd)
            elif optimizer_type =='Adam':
                self.optimizer_g = optim.Adam([
                    {'params': conv_value},
                    {'params': deconv_value,'lr': lr * 0.1},
                ], lr=lr,betas=(mom, 0.999), weight_decay=wd)
        self.optimizers.append(self.optimizer_g)