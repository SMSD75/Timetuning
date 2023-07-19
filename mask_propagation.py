import argparse
from email import utils
import math
import os
import sys
import warnings
import torch.nn.functional as F
from tqdm import tqdm
from re import A
import queue
from statistics import mode
import torchvision.transforms.functional as f
from struct import pack
from unicodedata import category
from anyio import maybe_async
from skimage.morphology import disk
import video_transformations
import numpy as np
import io
import torch
import torch.nn as nn
import logging
from PIL import Image
import matplotlib
from torch.utils.data import DataLoader
# from VisTR.datasets.ytvos import YTVOSDataset
from data_loader import VideoDataset, SamplingMode, YVOSDataset, pascalVOCLoader, make_loader
from metrics import PredsmIoU, PredsmIoU_1
import torchvision.transforms as trn
from sklearn.cluster import KMeans
import cv2
from my_utils import cosine_scheduler, make_working_directory, make_seg_maps
import matplotlib.pyplot as plt
from models import FeatureExtractorV2, apply_attention_mask, resnet18, resnet50
import shutil
from evaluation import evaluate_localizations, evaluate_propagation
import random
import tensorboard
from datetime import datetime
from models import FeatureExtractor
import copy

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

mask_neighborhood = None


class TimeT(torch.nn.Module):
    def __init__(self, feature_extractor, prototype_number=10, prototype_init=None):
        super(TimeT, self).__init__()
        self.feature_extractor = feature_extractor
        prototype_shapes = (prototype_number, self.feature_extractor.feature_dim)
        self.teacher = None
        self.max_epochs = None
        self.train_iters_per_epoch = None
        self.teacher_prototypes = None
        if prototype_init is None:
            prototype_init = torch.randn((prototype_shapes[0], prototype_shapes[1]))
            prototype_init =  F.normalize(prototype_init, dim=-1, p=2)    
        self.prototypes = torch.nn.parameter.Parameter(prototype_init)
    

    def init_momentum_teacher(self, teacher=None, prototypes=None):
        if teacher is None:
            self.teacher = copy.deepcopy(self.feature_extractor)
            self.teacher.requires_grad_(False)  
            self.teacher_prototypes = torch.nn.parameter.Parameter(self.prototypes.detach().clone())
            self.teacher_prototypes.requires_grad_(False)
        else:
            self.teacher = teacher
            self.teacher_prototypes = prototypes
    
    def update_momentum_teacher(self, step):
        with torch.no_grad():
            momentum = self.momentum_schedule[step]
            for param_q, param_k in zip(self.feature_extractor.parameters(), self.teacher.parameters()):
                param_k.data = param_k.data * momentum + param_q.detach().data * (1.0 - momentum)
            self.teacher_prototypes.data = self.teacher_prototypes.data * momentum + self.prototypes.detach().data * (1.0 - momentum)
            w = self.teacher_prototypes.data.clone()
            w = F.normalize(w, dim=1, p=2)
            self.teacher_prototypes.copy_(w)

    
    def set_momentum_teacher_schedular_params(self, momentum_teacher, momentum_teacher_end, max_epochs, train_iter_per_epoch):
        self.momentum_schedule = cosine_scheduler(momentum_teacher, momentum_teacher_end, max_epochs, train_iter_per_epoch)
    
    def normalize_prototypes(self):
        with torch.no_grad():
            w = self.prototypes.data.clone()
            w = F.normalize(w, dim=1, p=2)
            self.prototypes.copy_(w)

    def get_feature_prototype_similarity(self, x, use_teacher=False):
        """
        Computes the similarity between the input features and the prototypes.
        :param x: input features
        :return: similarity matrix
        """
        normalized_x = F.normalize(x, dim=-1, p=2)
        if use_teacher:
            scores = torch.mm(normalized_x, self.teacher_prototypes.t())
        else:
            scores = torch.mm(normalized_x, self.prototypes.t())  ## shape [num_patches, num_prototypes]
        return scores

    
    def reshape_to_spatial_resolution(self, x, spatial_resolution):
        """
        Reshapes the input features to the spatial resolution of the model.
        :param x: input features [num_patches, num_features]
        :return: reshaped features [num_features, spatial_resolution, spatial_resolution]
        """
        x = x.view(spatial_resolution, spatial_resolution, -1)
        x = x.permute(2, 0, 1)
        return x
    
    def forward(self, x, annotations=None, train=False, mask_features=False, use_head=True):
        """
        Computes the features of the input data.
        :param x: input data
        :return: features
        """
        if not train:
            with torch.no_grad():
                features, attentions = self.feature_extractor(x, use_head=use_head) ## shape [bs * fs, num_patches, dim]
                _, num_patches, dim = features.shape
                return features, attentions
        else:
            return self.get_loss(x, annotations=annotations, mask_features=mask_features)


    def get_scores(self, features, epsilon, sinkhorn_iterations, use_teacher=False):
        """
        Computes the similarity matrix between the input features and the prototypes.
        :param features: input features
        :return: similarity matrix
        """
        bs, num_patches, dim = features.shape
        sources_features = features
        sources_features = sources_features.contiguous().view(bs * num_patches, dim)
        batch_scores = self.get_feature_prototype_similarity(sources_features, use_teacher)
        batch_q = self.find_optimal_assignment(batch_scores, epsilon, sinkhorn_iterations)
        batch_q = batch_q.view(bs, num_patches, -1)
        batch_scores = batch_scores.view(bs, num_patches, -1)

        return batch_q, batch_scores
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    

    
    def get_loss(self, x, annotations=None, n_last_frames=7, size_mask_neighborhood=6, topk=5, epsilon=0.05, sinkhorn_iterations=10, mask_features=False):
        eps=1e-7
        if mask_features:
            criterion = torch.nn.CrossEntropyLoss(reduction='none')
        else:
            criterion = torch.nn.CrossEntropyLoss()
        bs, fs, c, h, w = x.shape

        if self.teacher is not None:
            teacher_features, teacher_attentions = self.teacher(x.view(bs * fs, c, h, w))
            _, num_patches, dim = teacher_features.shape
            teacher_features = teacher_features.view(bs, fs, num_patches, dim)
            if mask_features:
                teacher_features, teacher_attentions = apply_attention_mask(teacher_features, teacher_attentions, self.feature_extractor.spatial_resolution)
        features, attentions = self.feature_extractor(x.view(bs * fs, c, h, w)) ## shape [bs * fs, num_patches, dim]
        _, num_patches, dim = features.shape
        features = features.view(bs, fs, num_patches, dim)
        if mask_features:
            features, attentions = apply_attention_mask(features, attentions, self.feature_extractor.spatial_resolution)
            attentions = attentions.view(bs, fs, self.feature_extractor.spatial_resolution, self.feature_extractor.spatial_resolution)
        batch_loss = 0
        source_features = features[:, 0]
        if self.teacher is not None:
            teacher_source_features = teacher_features[:, 0]
            batch_q = self.get_scores(teacher_source_features, epsilon, sinkhorn_iterations, use_teacher=True)[0]
            batch_scores = self.get_scores(source_features, epsilon, sinkhorn_iterations)[1]
        else:
            batch_q, batch_scores = self.get_scores(source_features, epsilon, sinkhorn_iterations) ## shape [bs, num_patches, num_prototypes]
        target_features = features[:, -1]
        if self.teacher is not None:
            teacher_target_features = teacher_features[:, -1]
            target_batch_q = self.get_scores(teacher_target_features, epsilon, sinkhorn_iterations, use_teacher=True)[0]
            target_batch_scores = self.get_scores(target_features, epsilon, sinkhorn_iterations)[1]
        else:
            target_batch_q, target_batch_scores = self.get_scores(target_features, epsilon, sinkhorn_iterations)

        for i, data in enumerate(features):
            scores = batch_scores[i]
            q =  batch_q[i]

            scores = scores ## just for temprature scaling

            if mask_features:

                mask = attentions[i, -1].unsqueeze(0)


            forward_segmentation_maps = self.make_seg_maps(q, x[i], n_last_frames, size_mask_neighborhood, topk)

            q = self.reshape_to_spatial_resolution(q, self.feature_extractor.spatial_resolution)
            scores = self.reshape_to_spatial_resolution(scores, self.feature_extractor.spatial_resolution)

            target_scores = target_batch_scores[i]
            target_scores = self.reshape_to_spatial_resolution(target_scores, self.feature_extractor.spatial_resolution)
            target_q = target_batch_q[i]

            target_q = self.reshape_to_spatial_resolution(target_q, self.feature_extractor.spatial_resolution)

            p_map = forward_segmentation_maps[-1]
            loss2 = 0
            loss1 = criterion(target_scores.unsqueeze(0) / 0.1, p_map.unsqueeze(0).argmax(dim=1).long())

            loss = loss1 + loss2

            if mask_features:
                loss = loss * mask
            loss = loss.mean()
            batch_loss += loss
        return batch_loss / bs



def dense_optical_flow(data_list, params=[], to_gray=False):
    dataset_flow_list = []
    size = data_list.shape[0]
    for clip in data_list:
        clip_flow_list = []
        # Read the video and first frame
        assert clip.shape[0] >= 2
        old_frame = clip[0]


        # crate HSV & make Value a constant
        hsv = np.zeros_like(old_frame)
        hsv = np.expand_dims(hsv, axis=2)
        hsv = np.repeat(hsv, 3, axis=2)
        hsv[..., 1] = 255

        # old_frame = np.dstack((old_frame, old_frame, old_frame))
        # old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        clip_size = clip.shape[0]
        for i in range(1, clip_size):
            # Read the next frame
            new_frame = clip[i]
            frame_copy = new_frame
            # Preprocessing for exact method
            # new_frame = np.dstack((new_frame, new_frame, new_frame))
            # new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

            # Calculate Optical Flow
            # cv2.imshow('old', old_frame)
            # cv2.imshow('new', new_frame)
            # cv2.imshow('diff', new_frame - old_frame)
            # print(np.max(old_frame))
            # print(np.max(new_frame))
            flow = cv2.calcOpticalFlowFarneback(new_frame, old_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0) ## This is done a the reverse order since remap will be used in the further steps
            clip_flow_list.append(flow)

            # Encoding: convert the algorithm's output into Polar coordinates
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            # Use Hue and Value to encode the Optical Flow
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

            # Convert HSV image into BGR for demo
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            # cv2.imshow("frame", frame_copy)
            # cv2.imshow("optical flow", bgr)
            # k = cv2.waitKey(25) & 0xFF
            # if k == 27:
            #     break

            # Update the previous frame
            old_frame = new_frame
        dataset_flow_list.append(clip_flow_list)
    return dataset_flow_list



def interpolate_frames(frame, flow, n_frames):
    h, w = frame.shape
    frames = []
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    coords = np.float32(np.dstack([x_coords, y_coords]))
    for f in range(0, n_frames):
        pixel_map = coords + ((f+1)/n_frames) * flow
        inter_frame = cv2.remap(frame, pixel_map, None, cv2.INTER_NEAREST)
        # cv2.imshow('interpolated', inter_frame)
        # cv2.imshow('original', frame)
        frames.append(inter_frame)
    return frames


def propagate(dataset_flow_list, annotations):
    bs, fs, h, w = annotations.shape
    propagated_annotations = np.zeros((bs, fs-1, h, w))
    for i, clip_flow_list in enumerate(dataset_flow_list):
        for j, fram_displacement in enumerate(clip_flow_list):
            if j == 0:
                propagated_annotations[i, j] = interpolate_frames(annotations[i, j].numpy(), fram_displacement, 1)[0]
            else:
                propagated_annotations[i, j] = interpolate_frames(propagated_annotations[i, j-1], fram_displacement, 1)[0]
    propagated_annotations = torch.Tensor(propagated_annotations).type(torch.uint8)
    return propagated_annotations


def to_one_hot(y_tensor, n_dims=None):
    """
    Take integer y (tensor or variable) with n dims &
    convert it to 1-hot representation with n+1 dims.
    """
    if(n_dims is None):
        n_dims = int(y_tensor.max()+ 1)
    _,h,w = y_tensor.size()
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(h,w,n_dims)
    return y_one_hot.permute(2, 0, 1)

def norm_mask(mask):
    c, h, w = mask.size()
    normalized_mask = torch.zeros_like(mask)
    for cnt in range(c):
        mask_cnt = mask[cnt,:,:]
        if(mask_cnt.max() > 0):
            mask_cnt = mask_cnt - mask_cnt.min()
            mask_cnt = mask_cnt / mask_cnt.max()
            # mask[cnt,:,:] = mask_cnt
            # print(torch.all(mask_cnt == mask[cnt]))
            normalized_mask[cnt,:,:] = mask_cnt
    return normalized_mask 


def restrict_neighborhood(h, w, size_mask_neighborhood):
    # We restrict the set of source nodes considered to a spatial neighborhood of the query node (i.e. ``local attention'')
    mask = torch.zeros(h, w, h, w)
    for i in range(h):
        for j in range(w):
            for p in range(2 * size_mask_neighborhood + 1):
                for q in range(2 * size_mask_neighborhood + 1):
                    if i - size_mask_neighborhood + p < 0 or i - size_mask_neighborhood + p >= h:
                        continue
                    if j - size_mask_neighborhood + q < 0 or j - size_mask_neighborhood + q >= w:
                        continue
                    mask[i, j, i - size_mask_neighborhood + p, j - size_mask_neighborhood + q] = 1

    mask = mask.reshape(h * w, h * w)
    return mask
    # return mask



def label_propagation(size_mask_neighborhood, topk, model, frame_tar, list_frame_feats, list_segs, mask_neighborhood=None, features_exist=False):
    """
    propagate segs of frames in list_frames to frame_tar
    """
    ## we only need to extract feature of the target frame

    if isinstance(model, TimeT):
        spatial_resolution = model.feature_extractor.spatial_resolution
    else:
        spatial_resolution = model.spatial_resolution

    h = w = spatial_resolution
    if features_exist:
        features = frame_tar
    else:
        features, attention = model(frame_tar.unsqueeze(0), use_head=False)
    features = features.squeeze()
    return_feat_tar = features.T
    feat_tar = features
    ncontext = len(list_frame_feats)
    feat_sources = torch.stack(list_frame_feats) # nmb_context x dim x h*w

    feat_tar = F.normalize(feat_tar, dim=1, p=2)
    feat_sources = F.normalize(feat_sources, dim=1, p=2)

    feat_tar = feat_tar.unsqueeze(0).repeat(ncontext, 1, 1)
    aff = torch.exp(torch.bmm(feat_tar, feat_sources) / 0.1) # nmb_context x h*w (tar: query) x h*w (source: keys)
    aff = aff.to(features.device)
    if size_mask_neighborhood > 0:
        if mask_neighborhood is None:
            mask_neighborhood = restrict_neighborhood(h, w, size_mask_neighborhood)
            mask_neighborhood = mask_neighborhood.unsqueeze(0).repeat(ncontext, 1, 1)
            mask_neighborhood = mask_neighborhood.to(features.device)
        aff =  aff * mask_neighborhood

    aff = aff.transpose(2, 1).reshape(-1, h * w) # nmb_context*h*w (source: keys) x h*w (tar: queries)
    tk_val, _ = torch.topk(aff, dim=0, k=topk)
    tk_val_min, _ = torch.min(tk_val, dim=0)
    aff[aff < tk_val_min] = 0

    aff = aff / torch.sum(aff, keepdim=True, axis=0)
    # aff = aff.softmax(dim=0)

    list_segs = [s.to(features.device) for s in list_segs]
    segs = torch.cat(list_segs)
    nmb_context, C, h, w = segs.shape
    segs = segs.reshape(nmb_context, C, -1).transpose(2, 1).reshape(-1, C).T # C x nmb_context*h*w
    seg_tar = torch.mm(segs.double(), aff.double())
    seg_tar = seg_tar.reshape(1, C, h, w)
    return seg_tar, return_feat_tar, mask_neighborhood
 

def propagate_labels(n_last_frames, size_mask_neighborhood, topk, model, frame_list, first_seg, features_exist=False):
    """
    Evaluate tracking on a video given first frame & segmentation
    """
    if isinstance(model, TimeT):
        spatial_resolution = model.feature_extractor.spatial_resolution
    else:
        spatial_resolution = model.spatial_resolution
    first_seg = nn.functional.interpolate(first_seg.type(torch.DoubleTensor), size=(spatial_resolution, spatial_resolution), mode="nearest")
    # first_seg = first_seg.squeeze(0)

    # The queue stores the n preceeding frames
    que = queue.Queue(n_last_frames)

    # first frame
    if features_exist:
        features = frame_list[0]
    else:
        frame1 = frame_list[0]
        # extract first frame features
        features, attention = model(frame1.unsqueeze(0), use_head=False)
    features = features.squeeze()
    frame1_feat = features.T

    # saving first segmentation
    global mask_neighborhood
    if mask_neighborhood is None:
        mask_neighborhood = restrict_neighborhood(spatial_resolution, spatial_resolution, size_mask_neighborhood)
        mask_neighborhood = mask_neighborhood.to(features.device)
    segmentation_list = []
    for cnt in tqdm(range(1, frame_list.size(0))):
        frame_tar = frame_list[cnt]

        # we use the first segmentation and the n previous ones
        used_frame_feats = [frame1_feat] + [pair[0] for pair in list(que.queue)]
        used_segs = [first_seg] + [pair[1] for pair in list(que.queue)]

        frame_tar_avg, feat_tar, mask_neighborhood = label_propagation(size_mask_neighborhood, topk, model, frame_tar, used_frame_feats, used_segs, mask_neighborhood, features_exist)

        # pop out oldest frame if neccessary
        if que.qsize() == n_last_frames:
            que.get()
        # push current results into queue
        # seg = copy.deepcopy(frame_tar_avg.detach())
        seg = frame_tar_avg
        que.put([feat_tar, seg])
        # segmentation_list.append(norm_mask(frame_tar_avg.squeeze(0)))
        segmentation_list.append(frame_tar_avg.squeeze(0))
    return segmentation_list




def db_eval_boundary(annotation, segmentation, void_pixels=None, bound_th=0.008):
    assert annotation.shape == segmentation.shape
    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape
    if annotation.ndim == 3:
        n_frames = annotation.shape[0]
        f_res = np.zeros(n_frames)
        for frame_id in range(n_frames):
            void_pixels_frame = None if void_pixels is None else void_pixels[frame_id, :, :, ]
            f_res[frame_id] = f_measure(segmentation[frame_id, :, :, ], annotation[frame_id, :, :], void_pixels_frame, bound_th=bound_th)
    elif annotation.ndim == 2:
        f_res = f_measure(segmentation, annotation, void_pixels, bound_th=bound_th)
    else:
        raise ValueError(f'db_eval_boundary does not support tensors with {annotation.ndim} dimensions')
    return f_res



def f_measure(foreground_mask, gt_mask, void_pixels=None, bound_th=0.008):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.
    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.
        void_pixels     (ndarray): optional mask with void pixels
    Returns:
        F (float): boundaries F-measure
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1
    if void_pixels is not None:
        void_pixels = void_pixels.astype(np.bool)
    else:
        void_pixels = np.zeros_like(foreground_mask).astype(np.bool)

    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = _seg2bmap(foreground_mask * np.logical_not(void_pixels))
    gt_boundary = _seg2bmap(gt_mask * np.logical_not(void_pixels))

    from skimage.morphology import disk

    # fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))
    # gt_dil = binary_dilation(gt_boundary, disk(bound_pix))
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)

    return F


def _seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray):	Binary boundary map.
     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
    """

    seg = seg.astype(np.bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (
        width > w | height > h | abs(ar1 - ar2) > 0.01
    ), "Can" "t convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap


def db_statistics(per_frame_values):
    """ Compute mean,recall and decay from per-frame evaluation.
    Arguments:
        per_frame_values (ndarray): per-frame evaluation
    Returns:
        M,O,D (float,float,float):
            return evaluation statistics: mean,recall,decay.
    """

    # strip off nan values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        M = np.nanmean(per_frame_values)
        O = np.nanmean(per_frame_values > 0.5)

    N_bins = 4
    ids = np.round(np.linspace(1, len(per_frame_values), N_bins + 1) + 1e-10) - 1
    ids = ids.astype(np.uint8)

    D_bins = [per_frame_values[ids[i]:ids[i + 1] + 1] for i in range(0, 4)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        D = np.nanmean(D_bins[0]) - np.nanmean(D_bins[3])

    return M, O, D
    


def db_eval_iou(annotation, segmentation, void_pixels=None):
    """ Compute region similarity as the Jaccard Index.
    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.
        void_pixels  (ndarray): optional mask with void pixels
    Return:
        jaccard (float): region similarity
    """
    assert annotation.shape == segmentation.shape, \
        f'Annotation({annotation.shape}) and segmentation:{segmentation.shape} dimensions do not match.'
    annotation = annotation.astype(np.bool)
    segmentation = segmentation.astype(np.bool)

    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape, \
            f'Annotation({annotation.shape}) and void pixels:{void_pixels.shape} dimensions do not match.'
        void_pixels = void_pixels.astype(np.bool)
    else:
        void_pixels = np.zeros_like(segmentation)

    # Intersection between all sets
    inters = np.sum((segmentation & annotation) & np.logical_not(void_pixels), axis=(-2, -1))
    union = np.sum((segmentation | annotation) & np.logical_not(void_pixels), axis=(-2, -1))

    j = inters / union
    if j.ndim == 0:
        j = 1 if np.isclose(union, 0) else j
    else:
        j[np.isclose(union, 0)] = 1
    return j

def evaluate_semisupervised(all_gt_masks, all_res_masks, all_void_masks, metric):
    if all_res_masks.shape[0] > all_gt_masks.shape[0]:
        sys.stdout.write("\nIn your PNG files there is an index higher than the number of objects in the sequence!")
        sys.exit()
    elif all_res_masks.shape[0] < all_gt_masks.shape[0]:
        zero_padding = np.zeros((all_gt_masks.shape[0] - all_res_masks.shape[0], *all_res_masks.shape[1:]))
        all_res_masks = np.concatenate([all_res_masks, zero_padding], axis=0)
    j_metrics_res, f_metrics_res = np.zeros(all_gt_masks.shape[:2]), np.zeros(all_gt_masks.shape[:2])
    for ii in range(all_gt_masks.shape[0]):
        if 'J' in metric:
            j_metrics_res[ii, :] = db_eval_iou(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
        if 'F' in metric:
            f_metrics_res[ii, :] = db_eval_boundary(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
    return j_metrics_res, f_metrics_res

def mask_propagation(args):
    # OmegaConf.set_struct(cfg, False)
    num_epochs = 50
    np.seterr(all='raise')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ### This section needs to be used in the further versions that are more polished.
    architecture = args.architecture
    dataset = args.dataset
    dataset_path = args.dataset_path
    destination_path = args.destination_path
    model_path = args.model_path
    evaluation_protocol = args.evaluation_protocol
    batch_size = args.batch_size
    num_workers = args.num_workers
    num_frames = args.num_frames
    uvos_flag = args.uvos
    num_clusters = args.num_clusters
    input_resolution = args.input_resolution
    logging_directory = args.logging_directory
    use_optical_flow = args.use_optical_flow
    many_to_one = args.many_to_one
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    # exp_time = str(datetime.now())
    # make_logging_directory(exp_time)
    file_handler = logging.FileHandler(f"_{architecture}_{dataset}_{batch_size}_{num_clusters}_{input_resolution}_{evaluation_protocol}_{many_to_one}.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"Converting {dataset} to an image dataset.")
    # if dataset == "davis":
    #     convert_to_image_dataset(dataset_path, destination_path, dataset)
    make_working_directory(logging_directory)
    logger.info(f"The visualization directory has been made at {logging_directory}")
    ##############################################################

    PredsEval = PredsmIoU(num_clusters, 10, involve_bg=False)

    # model = FeatureExtractor(architecture, model_path, kqv="all")

    feature_extractor = FeatureExtractor(architecture, model_path, [1024, 1024, 512, 256])  ##  [1024, 1024, 512, 256] unfreeze_layers=["blocks.11", "blocks.10"]
    # feature_extractor = FeatureExtractor(architecture, model_path, [1024, 1024, 512, 256], [1024, 1024, 512, 256], unfreeze_layers=["blocks.11", "blocks.10"])
    model = TimeT(feature_extractor, 200)
    # model.load_state_dict(torch.load('logs_DeTeFFp/20230109/111039/0.1415744294352382_44.pth'))

    model = model.to(device)
    logging.basicConfig()
    
    logger.info(f"The selected model is {architecture} with the architecture as follows:")
    # logger.info(model.backbone)

    trns = trn.Compose([trn.ToTensor(), trn.Resize((input_resolution, input_resolution)), trn.CenterCrop(input_resolution)])
    target_trns = trn.Compose([trn.ToTensor(), trn.Resize((input_resolution, input_resolution), interpolation=f.InterpolationMode.NEAREST), trn.CenterCrop(input_resolution)])
    tr_normalize = trn.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]
    )
    # rand_color_jitter = video_transformations.RandomApply([video_transformations.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)], p=0.8)
    # data_transform_list = [rand_color_jitter, video_transformations.RandomGrayscale(), video_transformations.RandomGaussianBlur()]
    # data_transform = video_transformations.Compose(data_transform_list)
    # video_transform_list = [video_transformations.RandomResizedCrop(size=224), video_transformations.RandomHorizontalFlip(), video_transformations.ClipToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    # video_transform = video_transformations.Compose(video_transform_list)
    # train_loader = make_loader(dataset, num_frames, batch_size, SamplingMode.Full, frame_transform=data_transform, target_transform=None, video_transform=video_transform, shuffle=False, num_workers=num_workers, pin_memory=True)
    video_transform_list = [video_transformations.Resize(224, 'bilinear'), video_transformations.RandomCrop(224), video_transformations.ClipToTensor(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225])]
    video_transform = video_transformations.Compose(video_transform_list)
    train_loader = make_loader(dataset, num_frames, batch_size, SamplingMode.UNIFORM, frame_transform=None, target_transform=None, video_transform=video_transform, shuffle=False, num_workers=num_workers, pin_memory=True)
    logger.info("The dataset has been read.")

    dataset_predictions = []
    dataset_annotations = []
    for i, train_data in enumerate(train_loader):
        data, annotations, label = train_data
        # if "b19b3e22c0" not in data_names:
        #     continue
        annotations =  annotations.squeeze(1)
        data = data.squeeze(1)
        # data = data[:, ::num_frames, :, :, :]
        # annotations = annotations[:, ::num_frames, :, :]
        if uvos_flag:
            idx = annotations > 0
            annotations[idx] = 1
        bs, fs, c, h, w = data.shape
        data = data.view(bs * fs, c, h, w)
        orig_annotation = annotations.clone()
        logger.info(f"The data that is passed to the model has the shape : {data.shape}")
        if use_optical_flow:
            temp = []
            data_1 = data.clone()
            
            for datum in data_1:
                datum *= 255
                datum = datum.type(torch.uint8)
                im_rgb = cv2.cvtColor(datum.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
                temp.append(cv2.cvtColor(im_rgb, cv2.COLOR_BGR2GRAY))
            # data = data.view(bs, fs, h, w)
            data_1 = np.stack(temp, axis=0)
            data_1 = data_1.reshape(bs, fs, h, w)
            dataset_flow_list = dense_optical_flow(data_1, to_gray=False) ## to_gray = Flase for rlog
            predictions = propagate(dataset_flow_list, annotations)
        else:
            data = data.to(device) 
            data = data.view(bs, fs, c, h, w)
            predictions = []
            for i, clip in enumerate(data):
                clip, _ = model(clip, use_head=False, train=False)
                prediction = propagate_labels(args.n_last_frames, args.size_mask_neighborhood, args.topk, model, clip, to_one_hot(annotations[i, 0].unsqueeze(0)).unsqueeze(0), features_exist=True)
                prediction = torch.stack(prediction, dim=0)
                prediction = torch.nn.functional.interpolate(prediction, size=(input_resolution, input_resolution), mode="bilinear", align_corners=False)
                _, prediction = torch.max(prediction, dim=1)
                predictions.append(prediction)
            predictions = torch.stack(predictions)
            predictions = predictions.cpu()
        dataset_predictions.append(predictions)
        dataset_annotations.append(annotations)
        # convert_list_to_video(frame_buffer, f"Evaluation_{evaluation_protocol}_Reordered_{i}", speed=80, directory=logging_directory + "/", wdb_log=False)
        # # ### single mode evaluation finished.
        # # annotations[idx] += 1
        # cluster_maps = cluster_features(features, num_clusters, spatial_resolutions[architecture], input_resolution, evaluation_protocol)
        # batch_score = evaluate_localizations(PredsEval, annotations, cluster_maps, evaluation_protocol, logging_directory=logging_directory, many_to_one=many_to_one)

    # batch_score = evaluate_localizations(PredsEval, orig_annotation[:, 1:], orig_annotation[:, 1:], evaluation_protocol, logging_directory=logging_directory, many_to_one=many_to_one)

    all_predictions = torch.cat(dataset_predictions)
    all_annotations = torch.cat(dataset_annotations)
    score = evaluate_localizations(PredsEval, all_annotations[:, 1:], all_predictions[:, 1:], evaluation_protocol, logging_directory=None, many_to_one=many_to_one)
    # score = evaluate_propagation(PredsEval, all_annotations[:, 1:], all_predictions[:, 1:])
    print(score)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", type=str, default="dino-s16", help="which back-bone architecture do you want to use?")
    parser.add_argument("--model_path", type=str, default="../models/leopart_vits16.ckpt")
    parser.add_argument("--dataset", type=str, default="davis_val")
    parser.add_argument("--dataset_path", type=str, default="../data") ## davis : "../../../SOTA_Nips2021/dense-ulearn-vos/data/davis2017"
    parser.add_argument("--destination_path", type=str, default="ytvos")
    parser.add_argument("--evaluation_protocol", type=str, default="frame-wise")
    parser.add_argument("--logging_directory", type=str, default="visualizations")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--num_clusters", type=int, default=10)
    parser.add_argument("--input_resolution", type=int, default=224)
    parser.add_argument("--many_to_one", type=bool, default=False)
    parser.add_argument("--num_frames", type=int, default=25)
    parser.add_argument("--n_last_frames", type=int, default=4, help="number of preceeding frames")
    parser.add_argument("--uvos", type=int, default=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--size_mask_neighborhood", default=12, type=int, help="We restrict the set of source nodes considered to a spatial neighborhood of the query node")
    parser.add_argument("--epsilon", default=0.05, type=float, help="epsilon for sinkhorn")
    parser.add_argument("--sinkhorn_iterations", default=3, type=float, help="number of sinkhorn iterations")
    parser.add_argument("--use_projection_head", type=bool, default=True, help="use projection head")
    parser.add_argument("--use_optical_flow", type=bool, default=False, help="use label propagation")
    args = parser.parse_args()
    mask_propagation(args)