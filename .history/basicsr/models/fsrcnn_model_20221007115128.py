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
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_fsrcnn_training_settings()
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
        # self.setup_schedulers()
    def setup_fsrcnn_optimizers(self):
        """
        Setup optimizers for FSRCNN.
        卷积层和反卷积层的学习率不同，需要手写
        """
        train_opt = self.opt['train']

        lr = train_opt['optim_g']['lr']
        is_finetune = train_opt['optim_g'].get('is_finetune', False)
        mom = train_opt['optim_g'].get('momentum', 0.9)
        wd = train_opt['optim_g'].get('weight_decay', 1e-4)
        #使用sgd优化器
        if is_finetune:
            self.optimizer_g = optim.SGD([
                {'params': self.net_g.deconv_layer.parameters(), 'lr': lr * 0.1}
            ], lr=lr,momentum=mom, weight_decay=wd)
        else:
            self.optimizer_g = optim.SGD([
                {'params': self.net_g.extraction_layer.parameters()},
                {'params': self.net_g.mid_layers.parameters()},
                {'params': self.net_g.deconv_layer.parameters(), 'lr': lr * 0.1}
            ], lr=lr,momentum=mom, weight_decay=wd)
        self.optimizers.append(self.optimizer_g)