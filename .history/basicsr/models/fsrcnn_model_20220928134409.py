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
        init_type = opt['network_g']['init_type']
        self.net_g = self.init_weights(self.net_g,init_type)
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()
    def setup_fsrcnn_optimizers(self):
        """
        Setup optimizers for FSRCNN.
        卷积层和反卷积层的学习率不同，需要手写
        """
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        # optim_type = train_opt['optim_g'].pop('type')
        # self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        lr = train_opt['optim_g']['lr']
        self.optimizer_g = optim.Adam([
            {'params': self.net_g.extraction_layer.parameters()},
            {'params': self.net_g.mid_layers.parameters()},
            {'params': self.net_g.deconv_layer.parameters(), 'lr': lr * 0.1}
        ], lr=lr)
        self.optimizers.append(self.optimizer_g)