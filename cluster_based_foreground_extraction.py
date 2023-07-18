import argparse
import copy
from re import A
from statistics import mode
import torchvision.transforms.functional as f
from struct import pack
from unicodedata import category
from anyio import maybe_async
import faiss
import numpy as np
import os
import torch.nn.functional as F
import json
import io
import torch
import torch.nn as nn
import logging
from PIL import Image
import matplotlib
from torch.utils.data import DataLoader
# from VisTR.datasets.ytvos import YTVOSDataset
from data_loader import SamplingMode, pascalVOCLoader, make_loader
from evaluation import CyclicSwav, Evaluator
from metrics import PredsmIoU, PredsmIoU_1
import torchvision.transforms as trn
from sklearn.cluster import KMeans
import torchvision
from my_utils import convert_list_to_video, cosine_scheduler, make_working_directory, make_seg_maps, localize_objects, normalize_and_transform, sinkhorn
from clustering import cluster_features, proto_clustering
import matplotlib.pyplot as plt
from models import FeatureExtractor, FeatureExtractorV2, apply_attention_mask, process_attentions
import shutil
import random
import tensorboard
from datetime import datetime
import timm
from timm.models.vision_transformer import vit_small_patch16_224
from models import get_backbone
import video_transformations
from leoloader import pascal_loader
from collections import defaultdict
from bfscore import evaluate_bf_score

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)


class ScaleType:
    ZERO_TO_ONE = 0
    ZERO_TO_255 = 1



def process_data_group(data_group, scale=ScaleType.ZERO_TO_ONE):
    VIDEO_DATA_GROUP_LENGTH = 3
    if len(data_group)  == VIDEO_DATA_GROUP_LENGTH:
        data, annotations, label = data_group
        data = data.squeeze(1)
        annotations = annotations.squeeze(1)
    else:
        data, annotations = data_group
        data = data.unsqueeze_(1)
        # annotations.unsqueeze_(1)
    if scale == ScaleType.ZERO_TO_255:
        annotations *= 255
        annotations = annotations.long()

    return data, annotations


def normalize_features(features, reduction_dim=None):
    bs, fs, num_patches, dim = features.shape
    normalized_train_features = normalize_and_transform(features.flatten(0, 2), reduction_dim)
    _, dim = normalized_train_features.shape
    normalized_train_features = normalized_train_features.view(bs, fs, num_patches, dim)
    return normalized_train_features



def get_cluster_precs(cluster, mask, k):
    # Calculate attention foreground precision for each cluster id.
    # Note this doesn't use any gt but rather takes the ViT attention as noisy ground-truth for foreground.
    assert cluster.size(0) == mask.size(0)
    cluster_id_to_oc_count = defaultdict(int)
    cluster_id_to_cum_jac = defaultdict(float)
    for img_id in range(cluster.size(0)):
        img_attn = mask[img_id].flatten()
        img_clus = cluster[img_id].flatten()
        for cluster_id in torch.unique(img_clus):
            tmp_attn = (img_attn == 1)
            tmp_clust = (img_clus == cluster_id)
            tp = torch.sum(tmp_attn & tmp_clust).item()
            fp = torch.sum(~tmp_attn & tmp_clust).item()
            prec = float(tp) / max(float(tp + fp), 1e-8)  # Calculate precision
            cluster_id_to_oc_count[cluster_id.item()] += 1
            cluster_id_to_cum_jac[cluster_id.item()] += prec
    assert len(cluster_id_to_oc_count.keys()) == k and len(cluster_id_to_cum_jac.keys()) == k
    # Calculate average precision values
    precs = []
    for cluster_id in sorted(cluster_id_to_oc_count.keys()):
        precs.append(cluster_id_to_cum_jac[cluster_id] / cluster_id_to_oc_count[cluster_id])
    return precs



def eval_jac(gt, pred_mask, with_boundary):
    """
    Calculate Intersection over Union averaged over all pictures. with_boundary flag, if set, doesn't filter out the
    boundary class as background.
    """
    jacs = 0
    for k, mask in enumerate(gt):
        if with_boundary:
            gt_fg_mask = (mask != 0).float()
        else:
            gt_fg_mask = ((mask != 0) & (mask != 255)).float()
        intersection = gt_fg_mask * pred_mask[k]
        intersection = torch.sum(torch.sum(intersection, dim=-1), dim=-1)
        union = (gt_fg_mask + pred_mask[k]) > 0
        union = torch.sum(torch.sum(union, dim=-1), dim=-1)
        jacs += intersection / union
    res = jacs / gt.size(0)
    print(res)
    return res.item()

def Visualize_info(data, annotations, resized_attentions):
    fig, axs = plt.subplots(20, 3, figsize=(20, 20))
    for i in range(0, 40):
        axs[i][0].imshow(data[i, 0].permute(1, 2, 0).cpu().numpy())
        axs[i][1].imshow(resized_attentions[i, 0].cpu().numpy(), cmap="gray")
        axs[i][2].imshow(annotations[i, 0].cpu().numpy())
    plt.show()


def find_good_threshold(train_clusters, train_gt, precs, k):
    jacs = []
    sorted_precs = np.sort(precs)
    sorted_args = np.argsort(precs)
    for start in range(int(0.55 * k), int(0.75 * k)): # try out cuts between assigning 55% to 75% of clusters to bg
        fg_ids = sorted_args[start:]
        cbfe_mask = torch.zeros_like(train_clusters)
        for i in fg_ids:
            cbfe_mask[train_clusters == i] = 1
        jacs.append((sorted_precs[start], start, eval_jac(train_gt, cbfe_mask, with_boundary=True)))
        print(
            f"for {start} % fg cluster train is {torch.sum(cbfe_mask).item() / cbfe_mask.flatten().size(0)} "
            f"with {sorted_precs[start]}")
    return sorted(jacs, key=lambda x: x[2])  # return sorted by IoU


class ClusterBasedForegroundExtraction(torch.nn.Module):
    def __init__(self, model, k_fg_extraction, eval_resolution=100, eval_feature_dim=50, train_loader=None, val_loader=None, device="cuda"):
        super().__init__()
        self.model = model
        self.k_fg_extraction = k_fg_extraction
        self.eval_resolution = eval_resolution
        if isinstance(self.model, FeatureExtractor):
            self.spatial_resolution = self.model.spatial_resolution
        else:
            self.spatial_resolution = self.model.feature_extractor.spatial_resolution
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.eval_feature_dim = eval_feature_dim


    def get_foreground_masks(self, set="val"):
        train_features, train_attentions, train_annotations = self.extract_dataset_features_attentions(self.train_loader)
        val_features, val_attentions, val_annotations = self.extract_dataset_features_attentions(self.val_loader)
        reduced_normalized_train_features =  normalize_features(train_features, self.eval_feature_dim)
        reduced_normalized_val_features = normalize_features(val_features, self.eval_feature_dim)
        all_features = torch.cat([reduced_normalized_train_features, reduced_normalized_val_features], dim=0)
        train_data_size, val_data_size = train_features.shape[0], val_features.shape[0]
        resized_all_features = self.interpolate(all_features, self.eval_resolution)
        resized_train_features = resized_all_features[:train_data_size]
        resized_val_features = resized_all_features[train_data_size:]
        train_cluster_maps = self.create_overclustering_maps(resized_train_features)
        threshold = self.get_tuned_threshold(train_attentions, train_annotations, train_cluster_maps)

        set_annotations = val_annotations if set == "val" else train_annotations
        set_attentions = val_attentions if set == "val" else train_attentions
        resized_set_features = resized_val_features if set == "val" else resized_train_features

        set_annotations = F.interpolate(set_annotations.float(), size=(self.eval_resolution, self.eval_resolution), mode='nearest').long()
        set_attentions = F.interpolate(set_attentions.float(), size=(self.eval_resolution, self.eval_resolution), mode='nearest').long()
        attn_mask_soft = self.create_soft_masks(set_attentions, set_annotations, resized_set_features, threshold)

        evaluate_bf_score(attn_mask_soft.cpu(), set_annotations.flatten(0, 1).cpu())
        score = eval_jac(set_annotations.flatten(0, 1), attn_mask_soft, with_boundary=True)
        print(f"Jaccard score is {score}")
        return attn_mask_soft, set_annotations, resized_set_features

    def create_soft_masks(self, val_attentions, val_annotations, resized_val_features, threshold):
        val_cluster_maps = self.create_overclustering_maps(resized_val_features)
        val_attentions = val_attentions.flatten(0, 1)
        val_annotations = val_annotations.flatten(0, 1)
        val_cluster_maps = val_cluster_maps.flatten(0, 1).to(self.device)
        val_cluster_precs = get_cluster_precs(val_cluster_maps, val_attentions, self.k_fg_extraction)
        # pick precision value of best performing split and round it to nearest 0.05 boundary.
        attn_mask_soft = self.make_post_matching_maps(val_cluster_maps, threshold, val_cluster_precs)
        return attn_mask_soft

    def get_tuned_threshold(self, attentions, annotations, cluster_maps):
        annotations = F.interpolate(annotations.float(), size=(self.eval_resolution, self.eval_resolution), mode='nearest').long()
        attentions = F.interpolate(attentions.float(), size=(self.eval_resolution, self.eval_resolution), mode='nearest').long()
        annotations = annotations.flatten(0, 1)
        attentions = attentions.flatten(0, 1)
        cluster_maps = cluster_maps.flatten(0, 1).to(self.device)
        cluster_precs = get_cluster_precs(cluster_maps, attentions, self.k_fg_extraction)
        res = find_good_threshold(cluster_maps, annotations, cluster_precs, self.k_fg_extraction)
        threshold = min(np.arange(0, 1, 0.05), key=lambda x: abs(x - res[-1][0]))
        print(f"Found threshold {threshold}")
        return threshold
    

    def make_post_matching_maps(self, cluster_maps, threshold, cluster_precs):
        start_idx = np.where((np.sort(cluster_precs) >= threshold) == True)[0][0]
        fg_ids = np.argsort(cluster_precs)[start_idx:]
        attn_mask_soft = torch.zeros_like(cluster_maps)
        for i in fg_ids:
            attn_mask_soft[cluster_maps == i] = 1
        return attn_mask_soft

    def interpolate(self, features, target_resolution=100, mode="nearest"):
        bs, fs, num_patches, dim = features.shape
        features = features.view(bs * fs, num_patches, dim)
        features = features.permute(0, 2, 1)
        features = features.reshape(bs * fs, dim, self.spatial_resolution, self.spatial_resolution)
        scaled_features = F.interpolate(features, size=(target_resolution, target_resolution), mode=mode)
        return scaled_features.view(bs, fs, dim, target_resolution, target_resolution)

    def extract_dataset_features_attentions(self, data_loader):
        with torch.no_grad():
            feature_group = []
            attention_group = []
            annotation_group = []
            ATTENTION_THRESHOLD = 0.65
            for i, train_data in enumerate(data_loader):
                # print(annotations.unique())
                data, annotations = process_data_group(train_data, ScaleType.ZERO_TO_255)
                bs, fs, c, h, w = data.shape
                data = data.to(self.device)
                features, attentions = self.model(data.flatten(0, 1), use_head=False)
                attentions = attentions.to(self.device)
                _, num_patches, dim = features.shape
                features = features.view(bs, fs, num_patches, dim)
                attentions = process_attentions(attentions, self.spatial_resolution, threshold=ATTENTION_THRESHOLD)
                # attentions = attentions.view(bs, fs, num_patches)
                feature_group.append(features.cpu())
                attention_group.append(attentions.cpu())
                annotation_group.append(annotations.cpu())
                ## plot data, attention and annotation
                resized_attentions = F.interpolate(attentions, size=(annotations.size(-1), annotations.size(-1)), mode="nearest")

            features = torch.cat(feature_group, dim=0)
            attentions = torch.cat(attention_group, dim=0)
            annotations = torch.cat(annotation_group, dim=0)
            attentions = attentions.to(self.device)
            annotations = annotations.to(self.device)
        return features, attentions, annotations

    
    def create_overclustering_maps(self, features):
        bs, fs, dim, resolution, _ = features.shape
        features = features.permute(0, 1, 3, 4, 2)
        features = features.squeeze().reshape(-1, dim)
        # kmeans = KMeans(n_clusters=dataset_object_numbers[dataset], random_state = 0).fit(np.array(scaled_feature_maps.detach().cpu()))
        kmeans = faiss.Kmeans(features.size(1), self.k_fg_extraction, niter=50, nredo=5, seed=1, verbose=True, gpu=False, spherical=False)
        kmeans.train(features.detach().cpu().numpy())
        _, cluster_maps = kmeans.index.search(features.detach().cpu().numpy(), 1)
        cluster_maps = cluster_maps.squeeze()
        cluster_maps = cluster_maps.reshape(bs, fs, resolution, resolution)
        cluster_maps = torch.from_numpy(cluster_maps).to(self.device)
        return cluster_maps

def main(args):
    # OmegaConf.set_struct(cfg, False)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    num_epochs = 50
    np.seterr(all='raise')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ### This section needs to be used in the further versions that are more polished.
    architecture = args.architecture
    dataset = args.dataset
    dataset_path = args.dataset_path
    destination_path = args.destination_path
    k_fg_extraction = args.k_fg_extraction
    model_path = args.model_path
    evaluation_protocol = args.evaluation_protocol
    batch_size = args.batch_size
    num_workers = args.num_workers
    num_frames = args.num_frames
    uvos_flag = args.uvos
    num_clusters = args.num_clusters
    input_resolution = args.input_resolution
    logging_directory = args.logging_directory
    many_to_one = args.many_to_one
    precision_based = args.precision_based
    use_teacher = args.use_teacher
    EMA_decay = args.EMA_decay
    num_itr = 1000
    # if dataset == "davis":
    #     convert_to_image_dataset(dataset_path, destination_path, dataset)
    make_working_directory(logging_directory)
    print(f"The visualization directory has been made at {logging_directory}")
    ##############################################################

    # PredsEval = PredsmIoU(num_clusters, 10, involve_bg=False)

    # model = get_backbone(architecture, model_path)
    # model_path = "../models/leopart_vits16.ckpt"
    # architecture = "ma"
    feature_extractor = FeatureExtractor(architecture, model_path, [1024, 1024, 512, 256])  ##  [1024, 1024, 512, 256] unfreeze_layers=["blocks.11", "blocks.10"]
    # model = feature_extractor
    model = CyclicSwav(feature_extractor, 200)
    if use_teacher:
        model.init_momentum_teacher()
        model.set_momentum_teacher_schedular_params(EMA_decay, 1., num_epochs, num_itr)
    model.load_state_dict(torch.load('/home/ssalehi/video/vos_pretrained/cyclic_swav/src/0.1365865812925643_152737_dino_ytvos_128_200.pth')) #'0.1365865812925643_152737_dino_ytvos_128_200.pth'
    # model = FeatureExtractor(architecture, model_path)
    model = model.to(device)

    print(f"The selected model is {architecture} with the architecture as follows:")
    print(model)

    # trns = trn.Compose([trn.ToTensor(), trn.Resize((input_resolution, input_resolution)), trn.CenterCrop(input_resolution)])
    # target_trns = trn.Compose([trn.ToTensor(), trn.Resize((input_resolution, input_resolution), interpolation=f.InterpolationMode.NEAREST), trn.CenterCrop(input_resolution)])
    # rand_color_jitter = video_transformations.RandomApply([video_transformations.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)], p=0.8)
    video_transform_list = [video_transformations.Resize((input_resolution, input_resolution), 'bilinear'), video_transformations.CenterCrop(input_resolution), video_transformations.ClipToTensor(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225])]
    video_transform = video_transformations.Compose(video_transform_list)
    train_loader = make_loader(dataset, num_frames, batch_size, SamplingMode.UNIFORM, frame_transform=None, target_transform=None, video_transform=video_transform, shuffle=False, num_workers=num_workers, pin_memory=True)
    val_loader = make_loader(dataset + "_val", num_frames, batch_size, SamplingMode.UNIFORM, frame_transform=None, target_transform=None, video_transform=video_transform, shuffle=False, num_workers=num_workers, pin_memory=True)
    eval_resolution = 100 if evaluation_protocol == "dataset-wise" else input_resolution
    # train_loader = pascal_loader(60, "../../dataset/leopascal/VOCSegmentation", "trainaug", eval_resolution, train_size=input_resolution)
    # val_loader = pascal_loader(60, "../../dataset/leopascal/VOCSegmentation", "val", eval_resolution, train_size=input_resolution)
    EVAL_FEATURE_DIM = 50  
    CBFE = ClusterBasedForegroundExtraction(model, k_fg_extraction, eval_resolution, EVAL_FEATURE_DIM, train_loader, val_loader)
    set = "val"
    set_soft_masks, set_annotations, resized_set_features = CBFE.get_foreground_masks(set)
    set_loader = val_loader if set == "val" else train_loader
    evaluator = Evaluator(set_loader, model, logging_directory, uvos_flag, "k-means", f"evaluator1_{architecture}_{dataset}_{batch_size}_{num_clusters}_{input_resolution}_{evaluation_protocol}_{many_to_one}", fg_masks=set_soft_masks)
    evaluator.evaluate(many_to_one=many_to_one, evaluation_protocol=evaluation_protocol, eval_resolution=eval_resolution, num_clusters=21, use_annotations=False, use_mask=True, precision_based=precision_based)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", type=str, default="dino-s16", help="which back-bone architecture do you want to use?")
    parser.add_argument("--model_path", type=str, default= "/home/ssalehi/video/dino/outputs/checkpoint0080.pth") # "../models/leopart_vits16.ckpt"
    parser.add_argument("--dataset", type=str, default="ytvos")
    parser.add_argument("--dataset_path", type=str, default="../data") ## davis : "../../../SOTA_Nips2021/dense-ulearn-vos/data/davis2017"
    parser.add_argument("--destination_path", type=str, default="ytvos")
    parser.add_argument("--evaluation_protocol", type=str, default="dataset-wise")
    parser.add_argument("--logging_directory", type=str, default="visualizations")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=3)
    parser.add_argument("--k_fg_extraction", type=int, default=200)
    parser.add_argument("--num_clusters", type=int, default=21)
    parser.add_argument("--input_resolution", type=int, default=448)
    parser.add_argument("--many_to_one", type=bool, default=False)
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--precision_based", type=bool, default=False)
    parser.add_argument("--uvos", type=int, default=False)
    parser.add_argument("--use_teacher", type=bool, default=False)
    parser.add_argument("--EMA_decay", type=float, default=0.999)
    args = parser.parse_args()
    main(args)


