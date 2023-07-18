import torch

from abc import ABCMeta, abstractmethod
from mmcv.cnn import ConvModule
from typing import Optional, List, Tuple, Dict
from collections import defaultdict
from functools import partial
from torchvision import models
import timm
import torch.nn as nn


class BaseDecodeHead(nn.Module, metaclass=ABCMeta):
    """Base class for BaseDecodeHead.
    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict): Config of decode loss.
            Default: dict(type='CrossEntropyLoss').
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU')):
        super(BaseDecodeHead, self).__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False

    def init_weights(self):
        """Initialize weights of classification layer."""
        nn.init.normal(self.conv_seg, mean=0, std=0.01)

    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return 


class FCNHead(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.
    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.
    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 **kwargs):
        assert num_convs >= 0
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super(FCNHead, self).__init__(**kwargs)
        if num_convs == 0:
            assert self.in_channels == self.channels

        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def forward(self, inputs):
        """Forward function."""
        x = inputs
        output = self.convs(x)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.cls_seg(output)
        return output



def get_backbone_weights(arch: str, method: str, patch_size: int = None,
                         weight_prefix: Optional[str]= "model", ckpt_path: str = None) -> Dict[str, torch.Tensor]:
    """
    Load backbone weights into formatted state dict given arch, method and patch size as identifiers.
    :param arch: Target architecture. Currently supports resnet50, vit-small and vit-base.
    :param method: Method identifier.
    :param patch_size: Patch size of ViT. Ignored if arch is not ViT.
    :param weight_prefix: Optional prefix of weights to match model naming.
    :param ckpt_path: Optional path to checkpoint containing state_dict to be processed.
    :return: Dictionary mapping to weight Tensors.
    """
    def identity_transform(x): return x
    arch_to_args = {
        'vit-small16-dino': ("https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain_full_checkpoint.pth",
                             torch.hub.load_state_dict_from_url,
                             lambda x: x["teacher"]),
        'vit-small16-ours': (ckpt_path,
                             partial(torch.load, map_location=torch.device('cpu')),
                             lambda x: {k: v for k, v in x.items() if k.startswith('model')}),
        'vit-small16-mocov3': ("https://dl.fbaipublicfiles.com/moco-v3/vit-s-300ep/vit-s-300ep.pth.tar",
                               torch.hub.load_state_dict_from_url,
                               lambda x: {k: v for k, v in x.items() if k.startswith('module.base_encoder')}),
        'vit-small16-sup_vit': ('vit_small_patch16_224',
                            lambda x: timm.create_model(x,  pretrained=True).state_dict(),
                            identity_transform),
        'vit-base16-dino': ("https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain_full_checkpoint.pth",
                            torch.hub.load_state_dict_from_url,
                            lambda x: x["teacher"]),
        'vit-base8-dino': ("https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain_full_checkpoint.pth",
                           torch.hub.load_state_dict_from_url,
                           lambda x: x["teacher"]),
        'vit-base16-mae': ("https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth",
                           torch.hub.load_state_dict_from_url,
                           lambda x: x["model"]),
        'resnet50-sup_resnet': ("",
                                lambda x: models.resnet50(pretrained=True).state_dict(),
                                identity_transform),
        'resnet50-maskcontrast': (ckpt_path,
                                  torch.load,
                                  lambda  x: x["model"]),
        'resnet50-swav': ("https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar",
                          torch.hub.load_state_dict_from_url,
                          identity_transform),
        'resnet50-moco': ("https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar",
                          torch.hub.load_state_dict_from_url,
                          identity_transform),
        'resnet50-densecl': (ckpt_path,
                             torch.load,
                             identity_transform),
    }
    arch_to_args['vit-base16-ours'] = arch_to_args['vit-base8-ours'] = arch_to_args['vit-small16-ours']

    if "vit" in arch:
        url, loader, weight_transform = arch_to_args[f"{arch}{patch_size}-{method}"]
    else:
        url, loader, weight_transform = arch_to_args[f"{arch}-{method}"]
    weights = loader(url)
    if "state_dict" in weights:
        weights = weights["state_dict"]
    weights = weight_transform(weights)
    prefix_idx, prefix = get_backbone_prefix(weights, arch)
    if weight_prefix:
        return {f"{weight_prefix}.{k[prefix_idx:]}": v for k, v in weights.items() if k.startswith(prefix)
                and "head" not in k and "prototypes" not in k}
    return {f"{k[prefix_idx:]}": v for k, v in weights.items() if k.startswith(prefix)
            and "head" not in k and "prototypes" not in k}


def get_backbone_prefix(weights: Dict[str, torch.Tensor], arch: str) -> Optional[Tuple[int, str]]:
    # Determine weight prefix if returns empty string as prefix if not existent.
    if 'vit' in arch:
        search_suffix = 'cls_token'
    elif 'resnet' in arch:
        search_suffix = 'conv1.weight'
    else:
        raise ValueError()
    for k in weights:
        if k.endswith(search_suffix):
            prefix_idx = len(k) - len(search_suffix)
            return prefix_idx, k[:prefix_idx]
