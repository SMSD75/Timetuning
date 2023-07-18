"""
Based on Jabri et al., (2020)
Credit: https://github.com/ajabri/videowalk.git
License: MIT
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet as torch_resnet
from torchvision.models.resnet import BasicBlock

class ResNet(torch_resnet.ResNet):

    def __init__(self, *args, **kwargs):
        super(ResNet, self).__init__(*args, **kwargs)

    def filter_layers(self, x):
        return [l for l in x if getattr(self, l) is not None]

    def remove_layers(self, remove_layers=[]):
        # Remove extraneous layers
        remove_layers += ['fc', 'avgpool']
        for layer in self.filter_layers(remove_layers):
            setattr(self, layer, None)

    def modify(self):

        # Set stride of layer3 and layer 4 to 1 (from 2)
        for layer in self.filter_layers(['layer3']):
            for m in getattr(self, layer).modules():
                if isinstance(m, torch.nn.Conv2d):
                    m.stride = (1, 1)

        for layer in self.filter_layers(['layer4']):
            for m in getattr(self, layer).modules():
                if isinstance(m, torch.nn.Conv2d):
                    m.stride = (1, 1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x if self.maxpool is None else self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x3 = self.layer3(x) 
        x4 = self.layer4(x3)

        return x4, x3

def _resnet(arch, block, layers, pretrained, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model

def resnet18(pretrained='', remove_layers=[], train=True, **kwargs):

    model = _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, **kwargs)
    model.modify()

    model.remove_layers(remove_layers)
    setattr(model, "fdim", 512)
    return model



import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseNet(nn.Module):

    _trainable = (nn.Linear, nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d, nn.SyncBatchNorm)
    _batchnorm = (nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm)

    def __init__(self):
        super().__init__()
        # we may want a different learning rate
        # for new layers
        self.from_scratch_layers = []

        # we may want to freeze some layers
        self.not_training = []

        # we may want to freeze BN (means/stds only)
        self.bn_freeze = []

    def lr_mult(self):
        """Learning rate multiplier for weights.
        Returns: [old, new]"""
        return 1., 1.

    def lr_mult_bias(self):
        """Learning rate multiplier for bias.
        Returns: [old, new]"""
        return 2., 2.

    def _is_learnable(self, layer):
         return isinstance(layer, BaseNet._trainable)

    def _from_scratch(self, net, ignore=[]):

        for layer in net.modules():
            if self._is_learnable(layer):
                self.from_scratch_layers.append(layer)

    def _freeze_bn(self, net, ignore=[]):
        """Add layers to use in .eval() mode only"""

        for layer in net.modules():
            if isinstance(layer, BaseNet._batchnorm) and \
                    not layer in ignore:

                assert hasattr(layer, "eval") and callable(layer.eval)
                print("Freezing ", layer)
                self.bn_freeze.append(layer)

        print("Frozen BN: ", len(self.bn_freeze))

    def _fix_bn(self, layer):
        if isinstance(layer, nn.BatchNorm2d):
            self.not_training.append(layer)

        elif isinstance(layer, nn.Module):
            for c in layer.children():
                self._fix_bn(c)

    def __set_grad_mode(self, layer, mode, only_type=None):

        if hasattr(layer, "weight"):
            if only_type is None or isinstance(layer, only_type):
                layer.weight.requires_grad = mode

        if hasattr(layer, "bias") and not layer.bias is None:
            if only_type is None or isinstance(layer, only_type):
                layer.bias.requires_grad = mode

        if isinstance(layer, nn.Module):
            for c in layer.children():
                self.__set_grad_mode(c, mode)

    def train(self, mode=True):
        super().train(mode)

        # some layers have to be frozen
        for layer in self.not_training:
            self.__set_grad_mode(layer, False)

        for layer in self.bn_freeze:
            assert hasattr(layer, "eval") and callable(layer.eval)
            layer.eval()

    def parameter_groups(self, base_lr, wd):

        w_old, w_new = self.lr_mult()
        b_old, b_new = self.lr_mult_bias()

        groups = ({"params": [], "weight_decay":  wd, "lr": w_old*base_lr}, # weight learning
                  {"params": [], "weight_decay": 0.0, "lr": b_old*base_lr}, # bias learning
                  {"params": [], "weight_decay":  wd, "lr": w_new*base_lr}, # weight finetuning
                  {"params": [], "weight_decay": 0.0, "lr": b_new*base_lr}) # bias finetuning

        for m in self.modules():

            if not self._is_learnable(m):
                if hasattr(m, "weight") or hasattr(m, "bias"):
                    print("Skipping layer with parameters: ", m)
                continue

            if not m.weight is None and m.weight.requires_grad:
                if m in self.from_scratch_layers:
                    groups[2]["params"].append(m.weight)
                else:
                    groups[0]["params"].append(m.weight)
            elif not m.weight is None:
                print("Skipping W: ", m, m.weight.size())

            if m.bias is not None and m.bias.requires_grad:
                if m in self.from_scratch_layers:
                    groups[3]["params"].append(m.bias)
                else:
                    groups[1]["params"].append(m.bias)
            elif m.bias is not None:
                print("Skipping b: ", m, m.bias.size())

        return groups
    
    @staticmethod
    def _resize_as(x, y):
        return F.interpolate(x, y.size()[-2:], mode="bilinear", align_corners=True)


class MLP(nn.Sequential):

    def __init__(self, n_in, n_out):
        super().__init__()

        self.add_module("conv1", nn.Conv2d(n_in, n_in, 1, 1))
        self.add_module("bn1", nn.BatchNorm2d(n_in))
        self.add_module("relu", nn.ReLU(True))
        self.add_module("conv2", nn.Conv2d(n_in, n_out, 1, 1))

class Net(BaseNet):

    def __init__(self, backbone):
        super(Net, self).__init__()

        self.cfg = None
        self.backbone = backbone
        self.emb_q = MLP(backbone.fdim, 128)

    def lr_mult(self):
        """Learning rate multiplier for weights.
        Returns: [old, new]"""
        return 1., 1.

    def lr_mult_bias(self):
        """Learning rate multiplier for bias.
        Returns: [old, new]"""
        return 2., 2.

    def forward(self, frames, norm=True):
        """Forward pass to extract projection and task features"""

        # extracting the time dimension
        res4, res3 = self.backbone(frames)

        # B,K,H,W
        query = self.emb_q(res4)

        if norm:
            query = F.normalize(query, p=2, dim=1)
            res3 = F.normalize(res3, p=2, dim=1)
            res4 = F.normalize(res4, p=2, dim=1)

        return query, res3, res4



class Framework(BaseNet):

    def __init__(self, cfg, net):
        super(Framework, self).__init__()

        self.cfg = cfg
        self.fast_net = net
        self.eye = None

    def parameter_groups(self, base_lr, wd):
        return self.fast_net.parameter_groups(base_lr, wd)

    def _align(self, x, t):
        tf = F.affine_grid(t, size=x.size(), align_corners=False)
        return F.grid_sample(x, tf, align_corners=False, mode="nearest")

    def _key_val(self, ctr, q):
        """
        Args:
            ctr: [N,K]
            q: [BHW,K]
        Returns:
            val: [BHW,N]
        """

        # [BHW,K] x [N,K].t -> [BHWxN]
        vals = torch.mm(q, ctr.t()) # [BHW,N]

        # normalising attention
        return vals / self.cfg.TEST.TEMP

    def _sample_index(self, x, T, N):
        """Sample indices of the anchors
        Args:
            x: [BT,K,H,W]
        Returns:
            index: [B,N*N,K]
        """

        BT,K,H,W = x.shape
        B = x.view(-1,T,K,H*W).shape[0]

        # sample indices from a uniform grid
        xs, ys = W // N, H // N
        x_sample = torch.arange(0, W, xs).view(1, 1, N)
        y_sample = torch.arange(0, H, ys).view(1, N, 1)

        # Random offsets
        # [B x 1 x N]
        x_sample = x_sample + torch.randint(0, xs, (B, 1, 1))
        # [B x N x 1]
        y_sample = y_sample + torch.randint(0, ys, (B, 1, 1))

        # batch index
        # [B x N x N]
        hw_index = torch.LongTensor(x_sample + y_sample * W)

        return hw_index

    def _sample_from(self, x, index, T, N):
        """Gather the features based on the index
        Args:
            x: [BT,K,H,W]
            index: [B,N,N] defines the indices of NxN grid for a single
                           frame in each of B video clips
        Returns:
            anchors: [BNN,K] sampled features given by index from x
        """

        BT,K,H,W = x.shape
        x = x.view(-1,T,K,H*W)
        B = x.shape[0]

        # > [B,T,K,HW] > [B,T,HW,K] > [B,THW,K]
        x = x.permute([0,1,3,2]).reshape(B,-1,K)

        # every video clip will have the same samples
        # on the grid
        # [B x N x N] -> [B x N*N x 1] -> [B x N*N x K]
        index = index.view(B,-1,1).expand(-1,-1,K)

        # selecting from the uniform grid
        y = x.gather(1, index.to(x.device))

        # [BNN,K]
        return y.flatten(0,1)

    def _mark_from(self, x, index, T, N, fill_value=0):
        """This is analogous to _sample_from except that
        here we simply "mark" the sampled positions in the tensor
        Used for visualisation only.
        Since it is a binary mask, K == 1
        Args:
            x: [BT,1,H,W] binary mask
            index: [B,N,N] defines the indices of NxN grid for a single
                           frame in each of B video clips
        Returns:
            y: [BT,1,H,W] marked sample positions
        """

        BT,K,H,W = x.shape
        assert K == 1, "Expected binary mask"
        x = x.view(-1,T,K,H*W)
        B = x.shape[0]

        # > [B,T,K,HW] > [B,T,HW,K] > [B,THW,K]
        x = x.permute([0,1,3,2]).reshape(B,-1,K)

        # every video clip will have the same samples
        # on the grid
        # [B x N x N] -> [B x N*N x 1] -> [B x N*N x K]
        index = index.view(B,-1,1).expand(-1,-1,K)

        # selecting from the uniform grid
        # [B x T*H*W x K]
        y = x.scatter(1, index.to(x.device), fill_value)

        # [B x T*H*W x K] -> [BT x K x H x W]
        return y.view(-1,H*W,K).permute([0,2,1]).view(-1,K,H,W)

    def _cluster_grid(self, k1, k2, aff1, aff2, T, index=None):
        """ Selecting clusters within a sequence
        Args:
            k1: [BT,K,H,W]
            k2: [BT,K,H,W]
        """

        BT,K,H,W = k1.shape
        assert BT % T == 0, "Batch not divisible by sequence length"
        B = BT // T

        # N = [G x G]
        N = self.cfg.MODEL.GRID_SIZE ** 2

        # [BT,K,H,W] -> [BTHW,K]
        flatten = lambda x: x.flatten(2,3).permute([0,2,1]).flatten(0,1)

        # [BTHW,BN] -> [BT,BN,H,W]
        def unflatten(x, aff=None):
            x = x.view(BT,H*W,-1).permute([0,2,1]).view(BT,-1,H,W)
            if aff is None:
                return x
            return self._align(x, aff)

        index = self._sample_index(k1, T, N = self.cfg.MODEL.GRID_SIZE)
        query1 = self._sample_from(k1, index, T, N = self.cfg.MODEL.GRID_SIZE)

        """Computing the distances and pseudo labels"""

        # [BTHW,K]
        k1_ = flatten(k1)
        k2_ = flatten(k2)

        # [BTHW,BN] -> [BTHW,BN] -> [BT,BN,H,W]
        vals_soft = unflatten(self._key_val(query1, k1_), aff1)
        vals_pseudo = unflatten(self._key_val(query1, k2_), aff2)

        # [BT,BN,H,W]
        probs_pseudo = self._pseudo_mask(vals_pseudo, T)
        probs_pseudo2 = self._pseudo_mask(vals_soft, T)

        pseudo = probs_pseudo.argmax(1)
        pseudo2 = probs_pseudo2.argmax(1)

        # mask
        def grid_mask():
            grid_mask = torch.ones(BT,1,H,W).to(pseudo.device)
            return self._mark_from(grid_mask, index, T, N = self.cfg.MODEL.GRID_SIZE)

        return vals_soft, pseudo, index, [vals_pseudo, pseudo2, grid_mask]

    # sampling affinity
    def _aff_sample(self, k1, k2, T):
        BT,K,h,w = k1.size()
        B = BT // T
        hw = h*w

        def gen(query):
            grid_mask = torch.ones(B,1,hw).to(k1.device)
            # generating random indices
            indices = torch.randint(0, hw, (B,1,1)).to(k1.device)
            grid_mask.scatter_(2, indices, 0)

            # [B,K,H,W] -> [B,K,1]
            query_ = query[::T].view(B,K,-1).gather(2, indices.expand(-1,K,-1))

            def aff(keys):
                k = keys.view(B,T,K,-1)
                # [B,T,K,HW] x [B,1,K,HW] -> [B,T,HW]
                aff = (k * query_[:,None,:,:]).sum(2)
                return (aff + 1) / 2


            aff1 = aff(k1)
            aff2 = aff(k2)

            return grid_mask.view(B,h,w), aff1.view(BT,h,w), aff2.view(BT,h,w)

        grid_mask1, aff1_1, aff1_2 = gen(k1)
        grid_mask2, aff2_1, aff2_2 = gen(k2)

        return grid_mask1, aff1_1, aff1_2, \
                grid_mask2, aff2_1, aff2_2

    def _pseudo_mask(self, logits, T):
        BT,K,h,w = logits.shape
        assert BT % T == 0, "Batch not divisible by sequence length"
        B = BT // T

        # N = [G x G]
        N = self.cfg.MODEL.GRID_SIZE ** 2

        # generating a pseudo label
        # first we need to mask out the affinities across the batch
        if self.eye is None or self.eye.shape[0] != B*T \
                            or self.eye.shape[1] != B*N:
            eye = torch.eye(B)[:,:,None].expand(-1,-1,N).reshape(B,-1)
            eye = eye.unsqueeze(1).expand(-1,T,-1).reshape(B*T, B*N, 1, 1)
            self.eye = eye.to(logits.device)

        probs = F.softmax(logits, 1)
        return probs * self.eye

    def _ref_loss(self, x, y, N = 4):
        B,_,h,w = x.shape

        index = self._sample_index(x, T=1, N=N)
        x1 = self._sample_from(x, index, T=1, N=N)
        y1 = self._sample_from(y, index, T=1, N=N)
        logits = torch.mm(x1, y1.t()) / self.cfg.TEST.TEMP

        labels = torch.arange(logits.size(1)).to(logits.device)
        return F.cross_entropy(logits, labels)

    def _ce_loss(self, x, pseudo_map, T, eps=1e-5):
        error_map = F.cross_entropy(x, pseudo_map, reduction="none", ignore_index=-1)

        BT,h,w = error_map.shape
        errors = error_map.view(-1,T,h,w)
        error_ref, error_t = errors[:,0], errors[:,1:]

        return error_ref.mean(), error_t.mean(), error_map

    def _forward_reg(self, frames2, norm):
        losses = {}

        if not self.cfg.TRAIN.STOP_GRAD:
            k2, res3, res4 = self.fast_net(frames2, norm)
            return k2, res3, res4, losses

        training = self.fast_net.training
        if self.cfg.TRAIN.BLOCK_BN:
            self.fast_net.eval()

        with torch.no_grad():
            k2, res3, res4 = self.fast_net(frames2, norm)

        if self.cfg.TRAIN.BLOCK_BN:
            self.fast_net.train(training)

        return k2, res3, res4, losses

    def fetch_first(self, x1, x2, T):
        assert x1.shape[1:] == x2.shape[1:]
        c,h,w = x1.shape[1:]

        x1 = x1.view(-1,T+1,c,h,w)
        x2 = x2.view(-1,T-1,c,h,w)

        x2 = torch.cat((x1[:,-1:], x2), 1)
        x1 = x1[:,:-1]

        return x1.flatten(0,1), x2.flatten(0,1)

    def forward(self, frames, frames2=None, mask=None, T=None, affine=None, affine2=None, embd_only=False, norm=True, dbg=False):
        """Extract temporal correspondences
        Args:
            frames: [B,T,C,H,W]
        Returns:
            losses: a dictionary with the embedding loss
            net_outs: feature embeddings
        
        """

        # embedding for self-supervised learning
        key1, res3, res4 = self.fast_net(frames, norm)

        outs, losses = {}, {}
        if embd_only: # only embedding
            return res3, res4, key1
        else:
            key2, res3_2, res4_2, losses = self._forward_reg(frames2, norm)

            # fetching the first frame from the second view
            key1, key2 = self.fetch_first(key1, key2, T)

            vals, pseudo, index, dbg_info = self._cluster_grid(key1, key2, affine, affine2, T)

            vals_pseudo, pseudo2, grid_mask = dbg_info

            key1_aligned = self._align(key1, affine)
            key2_aligned = self._align(key2, affine2)

            n_ref = self.cfg.MODEL.GRID_SIZE_REF
            losses["cross_key"] = self._ref_loss(key1_aligned[::T], key2_aligned[::T], N = n_ref)

            # losses
            _, losses["temp"], outs["error_map"] = self._ce_loss(vals, pseudo, T)

            # computing the main loss
            losses["main"] = self.cfg.MODEL.CE_REF * losses["cross_key"] + losses["temp"]

            if dbg:
                vals = F.softmax(vals, 1)
                vals_pseudo = F.softmax(vals_pseudo, 1)

                frames, frames2 = self.fetch_first(frames, frames2, T)
                outs["frames_orig"] = frames
                outs["frames"] = self._align(frames, affine)
                outs["frames2"] = self._align(frames2, affine2)

                outs["map_soft"] = vals
                outs["map"] = pseudo
                outs["map_target_soft"] = vals_pseudo
                outs["map_target"] = pseudo2
                outs["grid_mask"] = grid_mask()

                outs["aff_mask1"], outs["aff11"], outs["aff12"], \
                        outs["aff_mask2"], outs["aff21"], outs["aff22"] = self._aff_sample(key1, key2, T)

        return losses, outs


def get_model(cfg=None, *args, **kwargs):

    backbones = {
        'resnet18': resnet18
    }

    def create_net():
        backbone = backbones['resnet18'](*args, **kwargs)
        return Net(backbone)

    net = create_net()
    return Framework(cfg, net)