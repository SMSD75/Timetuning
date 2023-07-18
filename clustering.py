import os
import cv2
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import torch.nn.functional as F
import wandb
import io
import torch
import torchvision
import random
import faiss
import torch.nn as nn

from my_utils import normalize_and_transform


def cluster_features(features, num_clusters, feature_resolution, input_resolution, evaluation_protocol, annotations=None):
    bs, fs, num_patches, dim = features.shape
    features = normalize_and_transform(features.reshape(bs * fs * num_patches, dim), 50)
    _, dim = features.shape
    features = features.view(bs, fs, num_patches, dim)
    cluster_map_list = []
    size = feature_resolution
    scale_factor = input_resolution // size
    if evaluation_protocol == "frame-wise":
        for i in range(bs):
            for j in range(fs):
                if annotations != None:
                    num_clusters = torch.unique(annotations[i, j]).shape[0]
                feature_maps = features[i, j].view(-1, size, size, dim)
                feature_maps = feature_maps.permute(0, 3, 1, 2)
                scaled_feature_maps = nn.functional.interpolate(feature_maps.type(torch.DoubleTensor), size=(input_resolution, input_resolution), mode="bilinear")
                scaled_feature_maps = scaled_feature_maps.permute(0, 2, 3, 1).float()
                scaled_feature_maps = scaled_feature_maps.squeeze().contiguous().view(-1, dim)
                kmeans = faiss.Kmeans(scaled_feature_maps.size(1), num_clusters, niter=50, nredo=5, seed=1, verbose=False, gpu=False, spherical=False)
                kmeans.train(scaled_feature_maps.detach().cpu().numpy())
                _, cluster_maps = kmeans.index.search(scaled_feature_maps.detach().cpu().numpy(), 1)
                cluster_maps = cluster_maps.squeeze()
                cluster_maps = cluster_maps.reshape(1, 1, input_resolution, input_resolution)
                cluster_map_list.append(torch.Tensor(cluster_maps))
    elif evaluation_protocol == "sample-wise":
        for i in range(bs):
            if annotations != None:
                num_clusters = torch.unique(annotations[i]).shape[0]
            feature_maps = features[i].view(-1, size, size, dim)
            feature_maps = feature_maps.permute(0, 3, 1, 2)
            scaled_feature_maps = nn.functional.interpolate(feature_maps.type(torch.DoubleTensor), size=(input_resolution, input_resolution), mode="bilinear")
            scaled_feature_maps = scaled_feature_maps.permute(0, 2, 3, 1).float()
            scaled_feature_maps = scaled_feature_maps.squeeze().contiguous().view(-1, dim)
            kmeans = faiss.Kmeans(scaled_feature_maps.size(1), num_clusters, niter=50, nredo=5, seed=1, verbose=False, gpu=False, spherical=False)
            kmeans.train(scaled_feature_maps.detach().cpu().numpy())
            _, cluster_maps = kmeans.index.search(scaled_feature_maps.detach().cpu().numpy(), 1)
            cluster_maps = cluster_maps.squeeze()
            cluster_maps = cluster_maps.reshape(1, fs, input_resolution, input_resolution)
            cluster_map_list.append(torch.Tensor(cluster_maps))
    elif evaluation_protocol == "dataset-wise":
        if annotations != None:
            num_clusters = torch.unique(annotations).shape[0]
        ds = bs
        feature_maps = features.contiguous().view(-1, size, size, dim)
        feature_maps = feature_maps.permute(0, 3, 1, 2)
        scaled_feature_maps = nn.functional.interpolate(feature_maps.type(torch.DoubleTensor), size=(input_resolution, input_resolution), mode="bilinear")
        scaled_feature_maps = scaled_feature_maps.permute(0, 2, 3, 1).float()
        scaled_feature_maps = scaled_feature_maps.squeeze().view(-1, dim)
        # kmeans = KMeans(n_clusters=dataset_object_numbers[dataset], random_state = 0).fit(np.array(scaled_feature_maps.detach().cpu()))
        kmeans = faiss.Kmeans(scaled_feature_maps.size(1), num_clusters, niter=50, nredo=5, seed=1, verbose=True, gpu=False, spherical=False)
        kmeans.train(scaled_feature_maps.detach().cpu().numpy())
        _, cluster_maps = kmeans.index.search(scaled_feature_maps.detach().cpu().numpy(), 1)
        cluster_maps = cluster_maps.squeeze()
        cluster_maps = cluster_maps.reshape(ds, fs, input_resolution, input_resolution)
    

    ## Conversion to uint8 is done just for the sake of computational efficiency
    if evaluation_protocol != "dataset-wise": 
        return torch.cat(cluster_map_list, dim=0).view(bs, fs, input_resolution, input_resolution).type(torch.int16)
    else:
        return torch.Tensor(cluster_maps).type(torch.int16)




def proto_clustering(x, prototypes, input_size=14, output_size=224, num_classes=None):
    """
    Clusters the input data using the prototypes.
    :param x: input feateatures [bs * fs, num_patch, dim]
    :param prototypes: prototypes [k, dim]
    :return: cluster indices
    """    
    with torch.no_grad():
        sample_num, num_patches, dim = x.shape
        orig_prototypes = prototypes
        num_prototypes = prototypes.shape[0]
        prototypes = prototypes.unsqueeze(0).repeat(sample_num, 1, 1)
        normalized_x = F.normalize(x, dim=-1, p=2) # L2 normalization
        normalized_prototypes = F.normalize(prototypes, dim=-1, p=2) # L2 normalization
        scores = torch.einsum('klm,knm->kln', normalized_x, normalized_prototypes) # scores is a (sample_num, num_patches, num_prototypes) matrix
        scores = scores.permute(0, 2, 1) # scores is now (sample_num, num_prototypes, num_patches)
        scores = scores.view(sample_num, num_prototypes, input_size, input_size)
        ## upsampling the scores
        scores = F.interpolate(scores, size=(output_size, output_size), mode='bilinear', align_corners=False)
        scores = scores.permute(0, 2, 3, 1)
        cluster_assignments = scores.argmax(dim=-1) # cluster_assignments is a (sample_num, output_size, output_size) matrix
        if num_classes is not None:
            kmeans = faiss.Kmeans(orig_prototypes.size(1), num_classes, niter=50, nredo=5, seed=1, verbose=False, gpu=False, spherical=False)
            kmeans.train(orig_prototypes.detach().cpu().numpy())
            _, cluster_maps = kmeans.index.search(orig_prototypes.detach().cpu().numpy(), 1)
            proto_maps = cluster_maps.squeeze()

            cluster_assignments = cluster_assignments.flatten()
            proto_maps = torch.Tensor(proto_maps).int()
            proto_maps = proto_maps.to(cluster_assignments.device)
            cluster_assignments = torch.index_select(proto_maps, dim=0, index=cluster_assignments)
            cluster_assignments = cluster_assignments.view(sample_num, output_size, output_size)
        return cluster_assignments.cpu()

