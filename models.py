import os
import torch
import torchvision.models as models
import torch
from torch import Tensor
import math
import warnings
import torch
import torch.nn as nn
import timm
from skimage.measure import label
import numpy as np
from leopart import get_backbone_weights
from functools import partial
import torch.nn as nn
from torchvision.transforms import GaussianBlur
# from timm.models.vision_transformer import vit_small_patch16_224
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
from functools import partial, reduce
from operator import mul
from timm.models.vision_transformer import VisionTransformer, _cfg
# from timm.models.layers.helpers import to_2tuple
from timm.models.layers import PatchEmbed
# from motion_grouping_model import SlotAttentionAutoEncoder
from dul_model import get_model
# from stego_paper.STEGO.src.train_segmentation import LitUnsupervisedSegmenter
from dino_vision_transformer import vit_small as dino_vit_small
from dino_vision_transformer import DINOHead, MultiCropWrapper
from collections import OrderedDict


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


spatial_resolutions = {"msn-s16" : 28, "ibot-s16" : 14, "resnet18": 14, "resnet50": 14, "resnet-32": "", "dino-s16": 14, "dul":28, "dino-s8": 28, "motion_grouping":56, "dino-b16": 14, "mocov3-s16":14,  "stego": 28, "leopart": 14, "swav": 7, "vit":14, "mae":14}


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}



def process_attentions(attentions: torch.Tensor, spatial_res: int, threshold: float = 0.65, blur_sigma: float = 0.6) \
        -> torch.Tensor:
    """
    Process [0,1] attentions to binary 0-1 mask. Applies a Guassian filter, keeps threshold % of mass and removes
    components smaller than 3 pixels.
    The code is adapted from https://github.com/facebookresearch/dino/blob/main/visualize_attention.py but removes the
    need for using ground-truth data to find the best performing head. Instead we simply average all head's attentions
    so that we can use the foreground mask during training time.
    :param attentions: torch 4D-Tensor containing the averaged attentions
    :param spatial_res: spatial resolution of the input image
    :param threshold: the percentage of mass to keep as foreground.
    :param blur_sigma: standard deviation to be used for creating kernel to perform blurring.
    :return: the foreground mask obtained from the ViT's attention.
    """
    # Blur attentions
    attention = attentions[:, :, 0, 1:]
    bs, num_heads, _= attention.shape
    attention = attention.view(bs, num_heads, spatial_res, spatial_res)
    attention = sum(attention[:, i] * 1 / attention.size(1) for i
                    in range(attention.size(1)))
    attention = attention.reshape(bs, 1, spatial_res, spatial_res)
    attention = GaussianBlur(7, sigma=(blur_sigma))(attention)
    attention = attention.reshape(attention.size(0), 1, spatial_res ** 2)
    # Keep threshold% of mass
    val, idx = torch.sort(attention)
    val /= torch.sum(val, dim=-1, keepdim=True)
    cumval = torch.cumsum(val, dim=-1)
    th_attn = cumval > (1 - threshold)
    idx2 = torch.argsort(idx)
    th_attn[:, 0] = torch.gather(th_attn[:, 0], dim=1, index=idx2[:, 0])
    th_attn = th_attn.reshape(attention.size(0), 1, spatial_res, spatial_res).float()
    # Remove components with less than 3 pixels
    for j, th_att in enumerate(th_attn):
        labelled = label(th_att.cpu().numpy())
        for k in range(1, np.max(labelled) + 1):
            mask = labelled == k
            if np.sum(mask) <= 2:
                th_attn[j, 0][mask] = 0
    return th_attn.detach()

def apply_attention_mask(features, attentions, spatial_resolution):
    """
    Masks the input features.
    :param features: input features (bs, fs, num_patch, dim)
    :param mask: mask
    :return: masked features
    """
    attentions = process_attentions(attentions, spatial_resolution)
    bs, fs, num_patches, dim = features.shape
    attentions = attentions.view(bs, fs, num_patches, 1)
    features = features * attentions
    return features, attentions.squeeze()

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)




def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attention:
            return x, attn
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, embed_dim=768, output_dim=128, hidden_dim=2048,
                 nmb_prototypes=1000, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, n_layers_projection_head=3,
                 l2_norm=True):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.l2_norm = l2_norm

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Construct projection head
        nlayers = max(n_layers_projection_head, 1)
        if nlayers == 1:
            self.projection_head = nn.Linear(embed_dim, output_dim)
        else:
            layers = [nn.Linear(embed_dim, hidden_dim)]
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.projection_head = nn.Sequential(*layers)

        # prototype layer
        if isinstance(nmb_prototypes, list):
            if len(nmb_prototypes) == 1:
                nmb_prototypes = nmb_prototypes[0]
            else:
                raise ValueError("MultiPrototypes not supported yet")
        self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, inputs, last_self_attention=False):
        if not isinstance(inputs, list):
            inputs = [inputs]
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in inputs]),
                return_counts=True,
            )[1], 0
        )
        assert len(idx_crops) <= 2, "Only supporting at most two different type of crops (global and local crops)"
        start_idx = 0
        for end_idx in idx_crops:
            _out = torch.cat(inputs[start_idx:end_idx])
            _out = self.forward_backbone(_out, last_self_attention=last_self_attention)
            if last_self_attention:
                _out, _attn = _out
            spatial_tokens = _out[:, 1:]
            spatial_tokens = spatial_tokens.reshape(-1, self.embed_dim)

            if start_idx == 0:
                output_spatial = spatial_tokens
                if last_self_attention:
                    attentions = _attn
            else:
                output_spatial = torch.cat((output_spatial, spatial_tokens))
                if last_self_attention:
                    attentions = torch.cat((attentions, _attn))
            start_idx = end_idx

        emb, out = self.forward_head(output_spatial)
        result = (emb, out)
        if last_self_attention:
            result += (attentions,)
        return result

    def forward_head(self, x):
        # Projection with l2-norm bottleneck as prototypes layer is l2-normalized
        x = self.projection_head(x)
        if self.l2_norm:
            x = nn.functional.normalize(x, dim=1, p=2)
        return x, self.prototypes(x)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward_backbone(self, x, last_self_attention=False):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                x = blk(x, return_attention=last_self_attention)
        if last_self_attention:
            x, attn = x
        x = self.norm(x)
        if last_self_attention:
            return x, attn[:, :, 0, 1:]
        return x

    def get_cls_tokens(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)[1]

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def get_backbone(name,  model_path):
    model = None
    try:
        if "resnet18" in name:
            model = resnet18(pretrained=True)
        elif "resnet50" in name:
            model = resnet50(pretrained=True)
        elif "dino-s8" in name:
            model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
        elif "dino-s16" in name:
            model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        elif "dino-b16" in name:
            model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        elif "msn-s16" in name:
            model = model = vit_small(patch_size=16)
            checkpoint = torch.load(model_path)
            state_dict = checkpoint['target_encoder']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('module'):
                    name = ".".join(k.split(".")[1:])
                    new_state_dict[name] = v
            msg = model.load_state_dict(new_state_dict, strict=False)
            print(msg)
        elif "mae" in name:
            model = mae_vit_base_patch16_dec512d8b()
            state_dict = torch.load(model_path)
            model.load_state_dict({".".join(k.split(".")[1:]): v for k, v in state_dict.items()}, strict=False)
            model.decoder_blocks = nn.Identity()
        elif "ibot-s16" in name:
            model = timm.create_model('vit_small_patch16_224', pretrained=False)
            state_dict = torch.load(model_path)['state_dict']
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
        elif "mocov3-s16" in name:
            ## vit_small_patch16_224
            model = vit_small(patch_size=16)
            # model = timm.create_model('vit_small_patch16_224', pretrained=False)
            checkpoint = torch.load(model_path, map_location="cpu")
            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            linear_keyword = 'head'
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                    # remove prefix
                    state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
            # model.load_state_dict(state_dict, strict=False)
        elif "mocov3-b16" in name:
            ## vit_small_patch16_224
            # model = timm.create_model('vit_base_patch16_224', pretrained=False)
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['state_dict'])
            # model.load_state_dict(state_dict, strict=False)
        elif "swav" in name:
            swav = torch.hub.load('facebookresearch/swav:main', 'resnet50')
            # model = resnet50(pretrained=False)
            # model.load_state_dict(swav.state_dict())
            model = swav
        elif "vit" in name:
            model = timm.create_model('vit_small_patch16_224', pretrained=True)
        elif "leopart" in name:
            state_dict = torch.load(model_path)
            model = vit_small(patch_size=16)
            # embed_dim = model.embed_dim
            # model = MultiCropWrapper(model, DINOHead(
            #     embed_dim,
            #     65536,
            #     use_bn=False,
            #     norm_last_layer=True,
            # ))
            # new_state_dict = OrderedDict()
            # for k, v in state_dict['student'].items():
            #     name = k[7:] # remove 'module.' of DataParallel/DistributedDataParallel
            #     new_state_dict[name] = v
            # try:
            #     msg = model.load_state_dict(new_state_dict, strict=False)
            #     print("=> loaded '{}' from checkpoint with msg {}".format('student', msg))
            # except TypeError:
            #     try:
            #         msg = model.load_state_dict(new_state_dict)
            #         print("=> loaded '{}' from checkpoint: ".format('student'))
            #     except ValueError:
            #         print("=> failed to load '{}' from checkpoint: ".format('student'))
            # model = model.backbone
            # new_state_dict = OrderedDict()
            # for k, v in state_dict['state_dict'].items():
            #     if k.startswith('model'):
            #         name = ".".join(k.split(".")[1:])
            #         new_state_dict[name] = v
            # model.load_state_dict(new_state_dict, strict=False)
            model.load_state_dict({".".join(k.split(".")[1:]): v for k, v in state_dict.items()}, strict=False)

        elif "stego" in name:
            if not os.path.isfile(model_path):
                print("The model can not be found in the given path.")
            model = LitUnsupervisedSegmenter.load_from_checkpoint(model_path)
        elif "motion_grouping" in name:
            if not os.path.isfile(model_path):
                print("The model can not be found in the given path.")
            checkpoint = torch.load(model_path)
            model = SlotAttentionAutoEncoder(resolution=(128, 224),
                                            num_slots=2,
                                            in_out_channels=3,
                                            iters=5)
            model.load_state_dict(checkpoint['model_state_dict'])
        elif "dul" in name:
            if not os.path.isfile(model_path):
                print("The model can not be found in the given path.")
            state_dict = torch.load(model_path)["model"]
            new_dict = {}
            for k,v in state_dict.items():
                new_key = k.replace("module.", "")
                new_dict[new_key] = v
            model = get_model()
            model.load_state_dict(new_dict, strict=False)
            

    except Exception as err:
        print(err)
    model.eval()
    return model


class FeatureExtractor(nn.Module):
    def __init__(self, arcitecture, model_path, head_layer_list=[], unfreeze_layers=[], kqv= "all"):
        super(FeatureExtractor, self).__init__()
        self.backbone = get_backbone(arcitecture, model_path)
        # print(self.backbone)
        self.freeze_backbone(unfreeze_layers=unfreeze_layers)
        self.architecture = arcitecture
        self.kqv = kqv
        features, _ = self.get_features(torch.rand(1, 3, 224, 224))
        self.feature_dim = features.shape[-1]
        self.spatial_resolution = spatial_resolutions[arcitecture]
        self.head = None
        if len(head_layer_list):
            self.head = []
            self.head.append(nn.Linear(self.feature_dim, head_layer_list[0]))
            # self.head.append(torch.nn.BatchNorm1d(head_layer_list[0])) 
            self.head.append(torch.nn.GELU())
            for i in range(1, len(head_layer_list)):
                self.head.append(nn.Linear(head_layer_list[i - 1], head_layer_list[i]))
                if i != len(head_layer_list) - 1:
                    # self.head.append(torch.nn.BatchNorm1d(head_layer_list[i]))
                    self.head.append(torch.nn.GELU())
            self.head = nn.Sequential(*self.head)
            self.feature_dim = head_layer_list[-1]

    ## freeze the backbone according to the input layer
    def freeze_backbone(self, unfreeze_layers=[]):
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
            for unfreeze_layer in unfreeze_layers:
                if unfreeze_layer in name:
                    param.requires_grad = True
                    break

    def get_features(self, input): ## gets a model and transformed inputs (normalized inputs) and returns feature maps as well as the possible attention maps 
        ## trans_input shape is (bs * fs, c, size, size)
        if self.architecture == "resnet18":
            outputs = []
            def hook(module, input, output):
                outputs.append(output)
            self.backbone.layer4[1].conv2.register_forward_hook(hook)
            self.backbone(input)
            features = outputs[0].flatten(start_dim=2) ## The original output shape is [bs*fs, dim, w, h] and the dim is 256 for resnet18
            features = features.permute(0, 2, 1)
            return features, None

        if self.architecture == "resnet50" or self.architecture == "swav":
            outputs = []
            def hook(module, input, output):
                outputs.append(output)
            self.backbone.layer4[2].conv3.register_forward_hook(hook)
            self.backbone(input)
            features = outputs[0].flatten(start_dim=2) ## The original output shape is [bs*fs, dim, w, h] and the dim is 256 for resnet18
            features = features.permute(0, 2, 1)
            return features, None
        
        elif self.architecture == "leopart":
            features = self.backbone.forward_backbone(input)
            features = features[:, 1:]
            attention = self.backbone.get_last_selfattention(input)
            return features, attention.detach()

        elif "dino" in self.architecture:
            features = self.backbone.get_intermediate_layers(input, n=1)[0]
            features = features[:, 1:]
            attention = self.backbone.get_last_selfattention(input)
            return features, attention.detach()
        
        elif self.architecture == "vit":
            features = self.backbone.forward_features(input)
            features = features[:, 1:]
            # attention = self.backbone.get_last_selfattention(input)
            return features, None
            
        elif "ibot" in self.architecture:
            features = self.backbone.forward_features(input)
            features = features[:, 1:]
            # attention = self.backbone.get_last_selfattention(input)
            return features, None
        
        elif "msn" in self.architecture:
            features = self.backbone.forward_backbone(input)
            features = features[:, 1:]
            attention = self.backbone.get_last_selfattention(input)
            return features, attention.detach()

        elif self.architecture == "mae":
            features, _, _ = self.backbone.forward_encoder(input, 0)
            features = features[:, 1:]
            # attention = self.backbone.get_last_selfattention(input)
            return features, None
        elif "moco" in self.architecture:
            features = self.backbone.forward_backbone(input)
            features = features[:, 1:]
            attention = self.backbone.get_last_selfattention(input)
            return features, attention.detach()

        # elif self.architecture == "dino" or self.architecture == "vit" or self.architecture == "leopart":
        #     feat_out = {}
        #     def hook_fn_forward_qkv(module, input, output):
        #         feat_out["qkv"] = output
        #     self.backbone._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)
        #     if self.architecture == "dino":
        #         attention = self.backbone.get_last_selfattention(input)
        #     else:
        #         def hook_fn_forward_attn(module, input, output):
        #             feat_out["attn"] = output
        #         self.backbone._modules["blocks"][-1]._modules["attn"]._modules["attn_drop"].register_forward_hook(hook_fn_forward_attn)
        #         output = self.backbone(input)
        #         attention = feat_out["attn"]
        #     ns = attention.shape[0]
        #     nh = attention.shape[1]
        #     nb_tokens = attention.shape[2]
        #     # mask = creat_mask_from_attention(attention, 0.05).reshape(w_featmap * h_featmap, 1)
        #     qkv1 = (
        #         feat_out["qkv"]
        #             .reshape(ns, nb_tokens, 3, nh, -1 // nh)
        #             .permute(2, 0, 3, 1, 4)
        #     )
        #     q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]
        #     k1 = k1.transpose(1, 2).reshape(ns, nb_tokens, -1)
        #     q1 = q1.transpose(1, 2).reshape(ns, nb_tokens, -1)
        #     v1 = v1.transpose(1, 2).reshape(ns, nb_tokens, -1)
        #     # features = q1.squeeze(0)[1:, :].detach().cpu()
        #     # features = torch.hstack((q1.squeeze(0)[1:, :].detach().cpu(), k1.squeeze(0)[1:, :].detach().cpu(), v1.squeeze(0)[1:, :].detach().cpu()))
        #     # features = v1.squeeze(0)[1:, :].detach().cpu()
        #     if self.kqv == "all":
        #         return torch.cat((q1[:, 1:, :], k1[:, 1:, :], v1[:, 1:, :]), dim=-1), attention.detach()
        #     elif self.kqv == "q":
        #         return q1[:, 1:, :], attention.detach()
        #     elif self.kqv == "k":
        #         return k1[:, 1:, :], attention.detach()
        #     elif self.kqv == "v":
        #         return v1[:, 1:, :], attention.detach()
        #     else:
        #         raise ValueError("kqv should be in ['all', 'q', 'k', 'v']")

        # elif self.architecture == "leopart":
        #     features, attn = model.forward_backbone(trans_input, True)
        #     features = features[:, 1:]
        #     return features, attn
            
        elif self.architecture == "stego":
            features = self.backbone(input) ## (fs * bs, num_patches, w, h)
            features = features.flatten(2)
            features = features.permute(0, 2, 1)
            # code1 = model(trans_input)
            # code2 = model(trans_input.flip(dims=[3]))
            # code  = (code1 + code2.flip(dims=[3])) / 2
            # cluster_loss, cluster_probs = model.cluster_probe(features, 2, log_probs=False)
            return features, None

        elif "motion_grouping" in self.architecture:
            features = self.backbone.encoder_cnn(input)
            ## interpolate to get the same size as the input
            features = torch.nn.functional.interpolate(features, size=(56, 56), mode="bilinear", align_corners=False)
            features = features.flatten(2, 3)
            features = features.permute(0, 2, 1)
            return features, None
        
        elif "dul" in self.architecture:
            features, _ = self.backbone.fast_net.backbone(input)
            features = torch.nn.functional.interpolate(features, size=(28, 28), mode="bilinear", align_corners=False)
            features = features.flatten(2, 3)
            features = features.permute(0, 2, 1)
            return features, None

    def forward(self, x, use_head=True):
        x, attentions = self.get_features(x)
        if (self.head is not None) and use_head:
            num_samples, num_patches, dim = x.shape
            ## This is done since batnorm1d should get (N, C) or (N, C, L) as input and performs batchnorm on the dimension C
            x = x.reshape(num_samples * num_patches, dim)
            x = self.head(x)
            x = x.view(num_samples, num_patches, -1)
        return x, attentions




class FeatureExtractorV2(nn.Module):
    def __init__(self, arcitecture, model_path, segmentation_head_layer_list=[], propagation_head_layer_list=[], unfreeze_layers=[], kqv= "all"):
        super(FeatureExtractorV2, self).__init__()
        self.backbone = get_backbone(arcitecture, model_path)
        # print(self.backbone)
        self.freeze_backbone(unfreeze_layers=unfreeze_layers)
        self.architecture = arcitecture
        self.kqv = kqv
        features, _ = self.get_features(torch.rand(1, 3, 224, 224))
        self.feature_dim = features.shape[-1]
        self.spatial_resolution = spatial_resolutions[arcitecture]
        self.segmentation_head = None
        ## make segmentation head
        if len(segmentation_head_layer_list):
            self.segmentation_head = []
            self.segmentation_head.append(nn.Linear(self.feature_dim, segmentation_head_layer_list[0]))
            # self.head.append(torch.nn.BatchNorm1d(head_layer_list[0])) 
            self.segmentation_head.append(torch.nn.GELU())
            for i in range(1, len(segmentation_head_layer_list)):
                self.segmentation_head.append(nn.Linear(segmentation_head_layer_list[i - 1], segmentation_head_layer_list[i]))
                if i != len(segmentation_head_layer_list) - 1:
                    # self.head.append(torch.nn.BatchNorm1d(head_layer_list[i]))
                    self.segmentation_head.append(torch.nn.GELU())
            self.segmentation_head = nn.Sequential(*self.segmentation_head)
            self.feature_dim = segmentation_head_layer_list[-1]
        ## make propagation head
        self.propagation_head = None
        if len(propagation_head_layer_list):
            self.propagation_head = []
            self.propagation_head.append(nn.Linear(features.shape[-1], propagation_head_layer_list[0]))
            # self.head.append(torch.nn.BatchNorm1d(head_layer_list[0])) 
            self.propagation_head.append(torch.nn.GELU())
            for i in range(1, len(propagation_head_layer_list)):
                self.propagation_head.append(nn.Linear(propagation_head_layer_list[i - 1], propagation_head_layer_list[i]))
                if i != len(propagation_head_layer_list) - 1:
                    # self.head.append(torch.nn.BatchNorm1d(head_layer_list[i]))
                    self.propagation_head.append(torch.nn.GELU())
            self.propagation_head = nn.Sequential(*self.propagation_head)

    ## freeze the backbone according to the input layer
    def freeze_backbone(self, unfreeze_layers=[]):
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
            for unfreeze_layer in unfreeze_layers:
                if unfreeze_layer in name:
                    param.requires_grad = True
                    break

    def get_features(self, input): ## gets a model and transformed inputs (normalized inputs) and returns feature maps as well as the possible attention maps 
        ## trans_input shape is (bs * fs, c, size, size)
        if self.architecture == "resnet18":
            outputs = []
            def hook(module, input, output):
                outputs.append(output)
            self.backbone.layer4[1].conv2.register_forward_hook(hook)
            self.backbone(input)
            features = outputs[0].flatten(start_dim=2) ## The original output shape is [bs*fs, dim, w, h] and the dim is 256 for resnet18
            features = features.permute(0, 2, 1)
            return features, None

        if self.architecture == "resnet50" or self.architecture == "swav":
            outputs = []
            def hook(module, input, output):
                outputs.append(output)
            self.backbone.layer4[2].conv3.register_forward_hook(hook)
            self.backbone(input)
            features = outputs[0].flatten(start_dim=2) ## The original output shape is [bs*fs, dim, w, h] and the dim is 256 for resnet18
            features = features.permute(0, 2, 1)
            return features, None
        
        elif self.architecture == "leopart":
            features = self.backbone.forward_backbone(input)
            features = features[:, 1:]
            attention = self.backbone.get_last_selfattention(input)
            return features, attention.detach()

        elif "dino" in self.architecture:
            features = self.backbone.get_intermediate_layers(input, n=1)[0]
            features = features[:, 1:]
            attention = self.backbone.get_last_selfattention(input)
            return features, attention.detach()
        
        elif self.architecture == "vit":
            features = self.backbone.forward_features(input)
            features = features[:, 1:]
            # attention = self.backbone.get_last_selfattention(input)
            return features, None
            
        elif "ibot" in self.architecture:
            features = self.backbone.forward_features(input)
            features = features[:, 1:]
            # attention = self.backbone.get_last_selfattention(input)
            return features, None
        
        elif "msn" in self.architecture:
            features = self.backbone.forward_backbone(input)
            features = features[:, 1:]
            attention = self.backbone.get_last_selfattention(input)
            return features, attention.detach()

        elif self.architecture == "mae":
            features, _, _ = self.backbone.forward_encoder(input, 0)
            features = features[:, 1:]
            # attention = self.backbone.get_last_selfattention(input)
            return features, None
        elif self.architecture == "moco":
            features = self.backbone(input)
            features = features[:, 1:]
            # attention = self.backbone.get_last_selfattention(input)
            return features, None
            
        elif self.architecture == "stego":
            features = self.backbone(input) ## (fs * bs, num_patches, w, h)
            features = features.flatten(2)
            features = features.permute(0, 2, 1)
            return features, None

    def forward(self, x, use_head=True, train=False):
        x, attentions = self.get_features(x)
        num_samples, num_patches, dim = x.shape
        if (self.segmentation_head is not None) and use_head:
            ## This is done since batnorm1d should get (N, C) or (N, C, L) as input and performs batchnorm on the dimension C
            seg_x = x.reshape(num_samples * num_patches, dim)
            seg_x = self.segmentation_head(seg_x)
            seg_x = seg_x.view(num_samples, num_patches, -1)
        if train:
            if (self.propagation_head is not None) and use_head:
                ## This is done since batnorm1d should get (N, C) or (N, C, L) as input and performs batchnorm on the dimension C
                prop_x = x.reshape(num_samples * num_patches, dim)
                prop_x = self.propagation_head(prop_x)
                prop_x = prop_x.view(num_samples, num_patches, -1)
            return seg_x, prop_x, attentions
        else:
            return x, attentions


class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))

        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        torch.nn.init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots = None):
        b, n, d, device = *inputs.shape, inputs.device
        n_s = num_slots if num_slots is not None else self.num_slots
        
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device = device)

        inputs = self.norm_input(inputs)        
        # k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            # q = self.to_q(slots)

            # dots = torch.einsum('bid,bjd->bij', q, k) * self.scale

            ## find the euclidean distance between the slots and the inputs
            dots = torch.cdist(slots, inputs, p=2) ** 2 ## (b, n_s, n)
            dots *= self.scale
            # dots = dots / 0.1
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', inputs, attn)
            # updates  = updates + self.mlp(updates)

            # slots = self.gru(
            #     updates.reshape(-1, d),
            #     slots_prev.reshape(-1, d)
            # )

            # slots = slots.reshape(b, -1, d)
            # slots = slots + self.mlp(self.norm_pre_ff(slots))
            slots = updates

        return slots


class DistributedDataParallelModel(nn.Module):
    def __init__(self, model, gpu):
        super(DistributedDataParallelModel, self).__init__()
        self.model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
        print(type(self.model))

    def forward(self, *input):
        return self.model(*input)
    def get_non_ddp_model(self):
        return self.model.module
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model.module, name)

# model = resnet18(pretrained=True) dd

# # for name, module in resnet.named_modules():
# #     print(name)
# # model.layer4[0].downsample[0].stride = (1, 1)
# model.fc = torch.nn.Sequential()
# model.avgpool = torch.nn.Sequential()
# # model.layer4[1] = torch.nn.Sequential()
# # model.layer4[2] = torch.nn.Sequential()

# print(model)
# outputs = []
# # model.layer4[0].downsample[0] = torch.nn.Sequential()
# def hook(module, input, output):
#     outputs.append(output)
# model.layer4[1].conv2.register_forward_hook(hook)
# trans_input = torch.randn((1, 3, 224, 224))
# model(trans_input)
# print(outputs[0].shape)

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed
    

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



class VisionTransformerMoCo(VisionTransformer):
    def __init__(self, stop_grad_conv1=False, **kwargs):
        super().__init__(**kwargs)
        # Use fixed 2D sin-cos position embedding
        self.build_2d_sincos_position_embedding()

        # weight initialization
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=1e-6)

        if isinstance(self.patch_embed, PatchEmbed):
            # xavier_uniform initialization
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)

            if stop_grad_conv1:
                self.patch_embed.proj.weight.requires_grad = False
                self.patch_embed.proj.bias.requires_grad = False

    def build_2d_sincos_position_embedding(self, temperature=10000.):
        h, w = 14, 14
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        assert self.num_tokens == 1, 'Assuming one and only one token, [cls]'
        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False
    

class ConvStem(nn.Module):
    """ 
    ConvStem, from Early Convolutions Help Transformers See Better, Tete et al. https://arxiv.org/abs/2106.14881
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        assert patch_size == 16, 'ConvStem only supports patch size of 16'
        assert embed_dim % 8 == 0, 'Embed dimension must be divisible by 8 for ConvStem'

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # build stem, similar to the design in https://arxiv.org/abs/2106.14881
        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(4):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

def vit_conv_small(**kwargs):
    # minus one ViT block
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=384, depth=11, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    return model

def vit_smallMoCo(**kwargs):
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model



@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

class MoCo(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()

        self.T = T

        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        self.momentum_encoder = base_encoder(num_classes=mlp_dim)

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def forward(self, x1, x2, m):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """

        # compute features
        q1 = self.predictor(self.base_encoder(x1))
        q2 = self.predictor(self.base_encoder(x2))

        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder

            # compute momentum features as targets
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)

        return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)


class MoCo_ViT(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1]
        del self.base_encoder.head, self.momentum_encoder.head # remove original fc layer

        # projectors
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)