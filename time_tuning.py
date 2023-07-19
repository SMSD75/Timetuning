import os
import copy
import glob
from platform import architecture
from pyexpat import model
import queue
import sys
from urllib.request import urlopen
import argparse
from matplotlib import colors
from nbformat import write
import numpy as np
import torch.distributed as dist
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import CosineAnnealingLR
import cv2
import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
from torchvision import transforms
import argparse
from re import A
from statistics import mode
import torchvision.transforms.functional as f
from struct import pack
from unicodedata import category
from anyio import maybe_async
import faiss
import numpy as np
import os
import json
import io
from torchvision.utils import make_grid
import torch
import torch.nn as nn
import logging
from PIL import Image
import matplotlib
from torch.utils.data import DataLoader
# from VisTR.datasets.ytvos import YTVOSDataset
from data_loader import VideoDataset, SamplingMode, YVOSDataset, make_loader, pascalVOCLoader
from metrics import PredsmIoU, PredsmIoU_1
import torchvision.transforms as trn
from torch.utils.tensorboard import SummaryWriter
from sklearn.cluster import KMeans
import torchvision
from my_utils import convert_list_to_video, cosine_scheduler, make_working_directory, make_figure, generate_colors, sinkhorn, denormalize
import matplotlib.pyplot as plt
from mask_propagation import propagate_labels, to_one_hot
from models import FeatureExtractor, get_backbone, process_attentions, DistributedDataParallelModel, apply_attention_mask
import shutil
import random
import tensorboard
from datetime import datetime
import timm
from evaluation import Evaluator, evaluate_localizations
from clustering import cluster_features, proto_clustering
import video_transformations
from leoloader import pascal_loader

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
writer = None
spatial_resolutions = {"resnet18": 14, "resnet50": 14, "resnet-32": "", "dino": 14, "stego": 28, "leopart": 14, "swav": "", "vit":14}
dataset_object_numbers = {"davis": 52, "ytvos":80, "pascal":21}
device = None

world_size = 1
label_color_map = []



class TimeT(torch.nn.Module):
    def __init__(self, feature_extractor, prototype_number=10, prototype_init=None):
        super(TimeT, self).__init__()
        self.feature_extractor = feature_extractor
        prototype_shapes = (prototype_number, self.feature_extractor.feature_dim)
        self.teacher = None
        self.max_epochs = None
        self.train_iters_per_epoch = None
        self.teacher_prototypes = None
        self.queue = None
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
    
    def init_queue(self, queue_size):
        self.queue = torch.zeros((queue_size, self.feature_extractor.feature_dim), device=self.prototypes.device)
    
    def update_momentum_teacher(self, step, writer=None):
        with torch.no_grad():
            momentum = self.momentum_schedule[step]
            writer.add_scalar("momentum", momentum, step)
            for param_q, param_k in zip(self.feature_extractor.parameters(), self.teacher.parameters()):
                param_k.data = param_k.data * (1.0 - momentum) + param_q.detach().data * momentum
            self.teacher_prototypes.data = self.teacher_prototypes.data * (1.0 - momentum) + self.prototypes.detach().data * momentum
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
    
    def make_seg_maps(self, first_frame_segmentation, orig_x, n_last_frames, size_mask_neighborhood, topk, features_exist=False):
        spatial_resolution = self.feature_extractor.spatial_resolution
        scores = first_frame_segmentation.view(spatial_resolution, spatial_resolution, -1)
        scores = scores.permute(2, 0, 1)    
        further_segmentation_maps = propagate_labels(n_last_frames, size_mask_neighborhood, topk, self.feature_extractor, orig_x, scores.unsqueeze(0), features_exist) ## shape [chanel, spatial_resolution, spatial_resolution] propagating scores as labels
        scaled_segmentation_maps = []
        for i, mask in enumerate(further_segmentation_maps):
            # scaled_mask = torch.softmax(mask / 0.1, dim=0)
            scaled_mask = mask
            # normalized_mask =  normalized_mask / normalized_mask.sum(dim=0, keepdim=True)
            scaled_segmentation_maps.append(scaled_mask)
        return torch.stack(scaled_segmentation_maps)

    
    def find_optimal_assignment(self, scores, epsilon, sinkhorn_iterations):
        """
        Computes the Sinkhorn matrix Q.
        :param scores: similarity matrix
        :return: Sinkhorn matrix Q
        """
        with torch.no_grad():
            q = torch.exp(scores / epsilon).t()
            q = sinkhorn(q, sinkhorn_iterations, world_size)
            # q = torch.softmax(scores / epsilon, dim=0)
            # q = q / q.sum(dim=1, keepdim=True)
        return q
    
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
        scores = batch_scores
        queue_scores = None
        if (self.queue is not None) and (self.queue[-1].count_nonzero() != 0):
            ## flatten queue
            queue = self.queue.view(-1, dim)
            queue_scores = self.get_feature_prototype_similarity(queue, use_teacher)
            scores = torch.cat([batch_scores, queue_scores], dim=0)
        q = self.find_optimal_assignment(scores, epsilon, sinkhorn_iterations)
        batch_q = q[:bs * num_patches]
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
        with torch.no_grad():
            backbone_features, _ = self.feature_extractor(x.view(bs * fs, c, h, w), use_head=False)
        _, num_patches, dim = features.shape
        _, _, backbone_dim = backbone_features.shape
        features = features.view(bs, fs, num_patches, dim)
        backbone_features = backbone_features.view(bs, fs, num_patches, backbone_dim)
        if mask_features:
            features, attentions = apply_attention_mask(features, attentions, self.feature_extractor.spatial_resolution)
            attentions = attentions.view(bs, fs, self.feature_extractor.spatial_resolution, self.feature_extractor.spatial_resolution)
        batch_loss = 0
        source_features = features[:, 0]

        if self.queue is not None:
            queue_features = None
            if self.teacher is not None:
                queue_features = teacher_features[:, 0]
            else:
                queue_features = features[:, 0]
            queue_features = queue_features.reshape(-1, dim)

            num_vectors_to_store = min(bs * 10, self.queue.size(0))
            idx = torch.randperm(queue_features.size(0))[:num_vectors_to_store]
            self.queue[num_vectors_to_store:] = self.queue[:-num_vectors_to_store].clone()
            self.queue[:num_vectors_to_store] = queue_features[idx].detach()

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

            forward_segmentation_maps = self.make_seg_maps(q, backbone_features[i], n_last_frames, size_mask_neighborhood, topk, features_exist=True)

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


def localize_objects(input_img, cluster_map):
    input_img = (input_img * 255).type(torch.uint8)
    input_img = input_img.to(device)
    one_hot_mask = to_one_hot(cluster_map.unsqueeze(0)).bool()
    one_hot_mask = one_hot_mask.cpu()
    input_img = input_img.cpu()
    overlayed_image = torchvision.utils.draw_segmentation_masks(input_img, one_hot_mask, colors=label_color_map, alpha=0.5)
    grid = torchvision.utils.make_grid([input_img, overlayed_image])
    fig = make_figure(grid)
    # fig = make_figure(input_img, cluster_map)
    with io.BytesIO() as buffer:
        fig.savefig(buffer, format='png')
        plt.close(fig)
        buffer.seek(0)
        return np.asarray(Image.open(buffer))


def localize_clip(data, cluster_map, logging_directory, name, w_featmap=28, h_featmap=28):
    bs, fs, c, h, w = data.shape
    cluster_map = cluster_map
    for i, datum in enumerate(data):
        frame_buffer = []
        for j, frame in enumerate(datum):
            frame_buffer.append(localize_objects(frame.detach().cpu(), cluster_map[i, j]))
        convert_list_to_video(frame_buffer, name + "_" + str(i), speed=1000/ datum.size(0), directory=logging_directory + "/", wdb_log=False)


def make_overlayed_grid(input_img, cluster_map):
    input_img = (input_img * 255).type(torch.uint8)
    input_img = input_img.to(device)
    one_hot_mask = to_one_hot(cluster_map.unsqueeze(0)).bool()
    ## This is because draw_segmentation_masks expects all things to be on the same device and cuda does not work
    one_hot_mask = one_hot_mask.cpu()
    input_img = input_img.cpu()
    overlayed_image = torchvision.utils.draw_segmentation_masks(input_img, one_hot_mask, colors=label_color_map, alpha=0.5)
    grid = torchvision.utils.make_grid([input_img, overlayed_image])
    return grid


def add_localization_results(input_img, cluster_map, name, step, writer):
    grid = make_overlayed_grid(input_img, cluster_map)
    writer.add_image(name, grid, global_step=step)


def log_tensorboard_figures(data, assignments, name, step, writer):
    for i, sample in enumerate(data):
        add_localization_results(sample[0], assignments[i, 0], f"{name}/{i}", step, writer)


def get_similarity_histogram(model, train_loader, mask_features=False):
    score_list = []
    bins = model.prototypes.shape[0]
    assignment_histogram = torch.zeros((1, bins), device=device)
    for i, train_data in enumerate(train_loader):
        data, annotations, label = train_data
        data = data.squeeze(1)
        annotations = annotations.squeeze(1)
        bs, fs, c, h, w = data.shape
        data = data.to(device) ## shape [bs, fs, c, h, w]
        with torch.no_grad():
            features, attentions = model(data.view(bs * fs, c, h, w)) ## shape [bs, fs, num_patches, dim]
            features = features.view(bs, fs, -1, model.feature_extractor.feature_dim)
            bs, fs, num_patches, dim = features.shape
            if mask_features:
                features, attentions = apply_attention_mask(features, attentions, model.feature_extractor.spatial_resolution)
            assignments = proto_clustering(features.view(bs * fs, num_patches, dim), model.prototypes)
            _, size, size = assignments.shape
            assignments = assignments.view(bs, fs, size, size)
            assignment_histogram += torch.histc(assignments.to(device), bins= bins, min=0, max=bins - 1, out=None)
    model.train()
    return assignment_histogram


## write and optimization class
class SwavOptimizer:
    def __init__(self, model, optimizer, use_projection_head, backbone_lr, lr, lr_scheduler, wd_schedule, num_itr=None, num_epochs=None, exclude_bias_norm=True):
        super(SwavOptimizer, self).__init__()
        self.optimizer = self.configure_optimizer(model, optimizer, use_projection_head, backbone_lr, lr, wd_schedule[0], exclude_bias_norm=exclude_bias_norm)
        if lr_scheduler is "CosineAnnealingLR":
            self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=num_itr * num_epochs, eta_min=0)
        else:
            self.lr_scheduler = None
        self.wd_schedule = wd_schedule
        self.writer = writer
        self.global_step = 0
    
    def get_optimization_dict(self, model, filter_name, exclude_decay=True, weight_decay=0.0001, learning_rate=0.001):
        params = []
        excluded_params = []
        for name, param in model.named_parameters():
            if param.requires_grad and (filter_name in name):
                if exclude_decay and (name.endswith(".bias") or (len(param.shape) == 1)):
                    excluded_params.append(param)
                else:
                    params.append(param)
                print(f"{name} is trainable")
        return [{'params': params, 'weight_decay': weight_decay, 'lr': learning_rate},
                    {'params': excluded_params, 'weight_decay': 0., 'lr': learning_rate}]

    def configure_optimizer(self, model, optimizer, use_projection_head, backbone_lr, lr, weight_decay, exclude_bias_norm=True):
        backbone_optimization_param = self.get_optimization_dict(model, "feature_extractor.backbone", exclude_decay=exclude_bias_norm, weight_decay=weight_decay, learning_rate=backbone_lr)
        prototype_optimization_param = self.get_optimization_dict(model, "prototypes", exclude_decay=exclude_bias_norm, weight_decay=weight_decay, learning_rate=lr)
        head_optimization_params = []
        if use_projection_head:
            head_optimization_params = self.get_optimization_dict(model, "feature_extractor.head", exclude_decay=exclude_bias_norm, weight_decay=weight_decay, learning_rate=lr)
            opt_parameters = head_optimization_params + backbone_optimization_param
        opt_parameters = prototype_optimization_param + head_optimization_params + backbone_optimization_param
        # print(opt_parameters)
        if optimizer == "AdamW":
            opt = torch.optim.AdamW(opt_parameters, lr)
        return opt
    
    def state_dict(self):
        return self.optimizer.state_dict(), self.global_step

    def step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.global_step += 1
        for i, param_group in enumerate(self.optimizer.param_groups):
            if param_group["weight_decay"] != 0:
                param_group["weight_decay"] = self.wd_schedule[self.global_step]
        


def log_assignment_histogram(model, num_clusters, eval_loader, use_mask, epoch_number):
    assignment_histogram = get_similarity_histogram(model, eval_loader, mask_features=use_mask)
    assignment_histogram = assignment_histogram.squeeze()
    assignment_distribution = assignment_histogram / assignment_histogram.sum()
    assignment_entropy = -1 * (assignment_distribution * torch.log(assignment_distribution + 1e-8)).mean()
    fig = plt.figure()
    plt.bar(range(num_clusters), assignment_distribution.detach().cpu().numpy())  
    writer.add_figure("Assignment Histogram", fig, global_step=epoch_number)
    writer.add_scalar(f"Scores/entropy", assignment_entropy, global_step=epoch_number)
    plt.close()


def log_clip_localization(model, eval_data, use_mask, evaluation_protocol, logging_directory, epoch_number, input_resolution):
    with torch.no_grad():      
        bs, fs, c, h, w = eval_data.shape
        features, attentions = model(eval_data.view(bs * fs, c, h, w), use_head=False) ## shape [bs, fs, num_patches, dim]
        num_samples, num_patches, dim = features.shape
        features = features.view(bs, fs, num_patches, dim)
        if use_mask:
            features, attentions = apply_attention_mask(features, attentions, model.feature_extractor.spatial_resolution)
        assignments = cluster_features(features, 200, model.feature_extractor.spatial_resolution, input_resolution, "dataset-wise")
        size = assignments.size(-1)
        assignments = assignments.view(bs, fs, size, size)
        log_tensorboard_figures(denormalize(eval_data, mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]), assignments, "figure", epoch_number, writer)
        localize_clip(denormalize(eval_data, mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]), assignments, logging_directory, str(epoch_number))


def save_checkpoint(model: TimeT, optimizer: SwavOptimizer, epch_num: int, filename: str) -> None:
    """Save the model and optimizer state to a checkpoint file."""
    optimizer_state_dict, global_step = optimizer.state_dict()
    state = {
        'epoch': epch_num,
        'global_step': global_step,
        'model': model.state_dict(),
        'optimizer': optimizer_state_dict,
        'scheduler': optimizer.lr_scheduler.state_dict() if optimizer.lr_scheduler is not None else None,
    }
    torch.save(state, filename)


def find_the_last_logging_directory(main_logging_directory: str) -> str:
    """Find the last logging directory."""
    try:
        logging_directories = glob.glob(main_logging_directory + '/*')
        if len(logging_directories) == 0:
            raise FileNotFoundError
        else:
            experiment_date_directory = sorted(logging_directories)[-1]
            experiment_time_directories = sorted(glob.glob(experiment_date_directory + '/*'))
            if len(experiment_time_directories) == 1:
                experiment_date_directory = sorted(logging_directories)[-2]
                experiment_time_directory = sorted(glob.glob(experiment_date_directory + '/*'))[-1]
            else:
                experiment_date_directory = sorted(logging_directories)[-1]
                experiment_time_directory = sorted(glob.glob(experiment_date_directory + '/*'))[-2]
            return experiment_time_directory
    except FileNotFoundError:
        print(f'No logging directory found at {main_logging_directory}')
        sys.exit(0)

def load_checkpoint(model: TimeT, swav_optimizer: SwavOptimizer, filename: str) -> None:
    """Load the model and optimizer state from a checkpoint file."""
    try:
        state = torch.load(filename)
        model.load_state_dict(state['model'])
        swav_optimizer.optimizer.load_state_dict(state['optimizer'])
        swav_optimizer.global_step = state['global_step']
        if swav_optimizer.lr_scheduler is not None:
            swav_optimizer.lr_scheduler.load_state_dict(state['scheduler'])
        return state['epoch']
    except FileNotFoundError:
        print(f'No checkpoint found at {filename}')
        return 0
    

def time_tuning(gpu=0, args=None):
    global device
    device = torch.device("cuda", gpu) 
    global world_size
    world_size = args.gpus * args.nodes 
    rank = args.nr * args.gpus + gpu
    if world_size > 1:
        rank = args.nr * args.gpus + gpu
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        torch.cuda.set_device(device)
    np.seterr(all='raise')
    torch.autograd.set_detect_anomaly(True)
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
    use_teacher = args.use_teacher
    uvos_flag = args.uvos
    use_mask = args.use_mask
    use_projection_head = args.use_projection_head
    num_clusters = args.num_clusters
    input_resolution = args.input_resolution
    visualization_directory = args.visualization_directory
    many_to_one = args.many_to_one
    regular_step = args.regular_step
    logging_directory = args.logging_directory
    precision_based = args.precision_based
    num_epochs = args.num_epochs
    EMA_decay = args.EMA_decay
    load_checkpoint_flag = args.load_checkpoint
    use_queue = args.use_queue
    queue_size = args.queue_size
    lr_scheduler = args.lr_scheduler
    head_lr = args.head_lr
    ## creat a string based on the concatenation of the arguments
    arg_string = ""
    date_year = datetime.now().strftime("%Y%m%d")
    date_hour = datetime.now().strftime("%H%M%S") 
    arg_string = datetime.now().strftime("%Y%m%d") + "/" + datetime.now().strftime("%H%M%S")
    logging_directory = os.path.join(logging_directory, arg_string)
    global label_color_map
    label_color_map = generate_colors(num_clusters)
    # if dataset == "davis":
    #     convert_to_image_dataset(dataset_path, destination_path, dataset)
    if gpu == 0:
        make_working_directory(visualization_directory)
        global writer
        writer = SummaryWriter(log_dir=logging_directory)
        ## make a config file with the arguments
        with open(f"{logging_directory}/config.txt", "w") as f:
            for arg in vars(args):
                f.write(f"{arg}:{getattr(args, arg)}\n") 
    print(f"The visualization directory has been made at {visualization_directory}")
    ##############################################################
    if use_projection_head:
        feature_extractor = FeatureExtractor(architecture, model_path, [1024, 1024, 512, 256], unfreeze_layers=["blocks.11", "blocks.10"])  ## unfreeze_layers=["blocks.11", "blocks.10"]
    else:
        feature_extractor = FeatureExtractor(architecture, model_path)
    model = TimeT(feature_extractor, num_clusters)
    # model.load_state_dict(torch.load("TimeT.pth"))
    if world_size > 1:
        model = model.to(device)
        # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallelModel(model, gpu)
    else:
        model = model.to(device)

    print(f"The selected model is {architecture} with the architecture as follows:")
    print(model)
    rand_color_jitter = video_transformations.RandomApply([video_transformations.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)], p=0.8)
    data_transform_list = [rand_color_jitter, video_transformations.RandomGrayscale(), video_transformations.RandomGaussianBlur()]
    data_transform = video_transformations.Compose(data_transform_list)
    # eval_data_transform_list = [video_transformations.Resize((input_resolution, input_resolution)), video_transformations.ToTensor(), video_transformations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225])]
    video_transform_list = [video_transformations.Resize(input_resolution), video_transformations.RandomResizedCrop((input_resolution, input_resolution)), video_transformations.RandomHorizontalFlip(), video_transformations.ClipToTensor(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225])]
    video_transform = video_transformations.Compose(video_transform_list)
    train_loader = make_loader(dataset, num_frames, batch_size, regular_step, SamplingMode.UNIFORM, frame_transform=data_transform, target_transform=None, video_transform=video_transform, shuffle=True, num_workers=num_workers, pin_memory=True, world_size=world_size, rank=rank)
    # eval_loader = make_loader("pascal", 16, 16, regular_step, SamplingMode.UNIFORM, frame_transform=None, target_transform=None, video_transform=video_transform, shuffle=False, num_workers=num_workers, pin_memory=True, world_size=1, rank=rank)
    eval_loader = pascal_loader(60, "../../dataset/leopascal/VOCSegmentation", "val", 112, train_size=224)
    # train_evaluator = Evaluator(train_loader, model, logging_directory, uvos_flag, clustering_algorithm="k-means", logger_name=f"evaluator_{architecture}_{dataset}_{batch_size}_{num_clusters}_{input_resolution}_{evaluation_protocol}_{many_to_one}")
    if isinstance(model, DistributedDataParallelModel):
        eval_model = model.get_non_ddp_model()
    else:
        eval_model = model
    eval_evaluator = Evaluator(eval_loader, eval_model, visualization_directory, uvos_flag, clustering_algorithm="k-means", logger_name=f"evaluator_{architecture}_{dataset}_{batch_size}_{num_clusters}_{input_resolution}_dataset-wise_{many_to_one}", device=device)
    eval_resolution = input_resolution // 2 if evaluation_protocol == "dataset-wise" else input_resolution
    use_annotations = True if evaluation_protocol != "dataset-wise" else False
    use_annotations = True
    # eval_data, eval_annotations, _ = next(iter(eval_loader)) 
    # eval_data = eval_data.squeeze(1)
    # eval_annotations = eval_annotations.squeeze(1)
    # eval_data = eval_data.to(device) 
    num_itr = len(train_loader)
    previous_score = 0
    print("configuring optimizer")
    swav_optimizer = SwavOptimizer(model, "AdamW", use_projection_head, head_lr / 10, head_lr, lr_scheduler, cosine_scheduler(0.04, 0.4, num_epochs, num_itr), num_itr, num_epochs)
    if use_teacher:
        model.init_momentum_teacher()
        model.set_momentum_teacher_schedular_params(EMA_decay, 1., num_epochs, num_itr)
    if use_queue:
        model.init_queue(queue_size // world_size)

    if load_checkpoint_flag:
        last_experiment_path = find_the_last_logging_directory(args.logging_directory)
        print(f"loading the last experiment from {last_experiment_path}")
        prev_exp_epoch_num = load_checkpoint(model, swav_optimizer, f"{last_experiment_path}/checkpoint.pth")
        print(f"loaded the last experiment from {last_experiment_path}")
    else:
        print("not loading the last experiment")
    

    for j in range(num_epochs):
        score_list = []
        save_checkpoint(model, swav_optimizer, j, f"{logging_directory}/checkpoint.pth")
        if world_size > 1:
            train_loader.sampler.set_epoch(j)
        if j % 4 == 0 and rank == 0:
            with torch.no_grad():
                eval_model.eval()
                eval_score = eval_evaluator.evaluate(many_to_one=many_to_one, evaluation_protocol=evaluation_protocol, eval_resolution=eval_resolution, num_clusters=21, use_annotations=False, use_mask=False, precision_based=precision_based)
                if eval_score > previous_score:
                    previous_score = eval_score
                    saving_path = f"{previous_score}_{j}.pth"
                    model.save(f"{logging_directory}/{saving_path}")
                # log_assignment_histogram(eval_model, num_clusters, eval_loader, use_mask, j)
                # log_clip_localization(eval_model, eval_data, use_mask, evaluation_protocol, logging_directory, j, input_resolution)
                # score_list.append(eval_score)
                writer.add_scalar(f"Scores/localization", eval_score, global_step=j)
            eval_model.train()
        if world_size > 1:
            dist.barrier()
        model.train()
        for i, train_data in enumerate(train_loader):
            data, annotations, label = train_data
            data = data.squeeze(1)
            annotations = annotations.squeeze(1)
            # data = eval_data
            # annotations = eval_annotations
            bs, fs, c, h, w = data.shape
            annotations = annotations.to(device)
            data = data.to(device) ## shape [bs, fs, c, h, w]
            loss = model(data, annotations, True, use_mask) ## shape [bs, fs, num_patches, dim]
            swav_optimizer.step(loss)
            model.normalize_prototypes()
            if use_teacher:
                model.update_momentum_teacher(swav_optimizer.global_step, writer)
            if rank == 0:
                writer.add_scalar(f"Loss/train", loss.item(), global_step=swav_optimizer.global_step)
                print("Iteration: {}/{}".format(i, num_itr))
        # eval_data = data
            # plt.imshow(data[0, 0].permute(1, 2, 0).cpu().numpy())
        # writer.add_scalar(f"Scores/localization_epoch", sum(score_list) / len(score_list), global_step=j)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", type=str, default="dino-s16", help="which back-bone architecture do you want to use?")
    parser.add_argument("--model_path", type=str, default= "vits16_800ep.pth.tar") # "../models/leopart_vits16.ckpt"
    parser.add_argument("--dataset", type=str, default="ytvos")
    parser.add_argument("--dataset_path", type=str, default="../data") ## davis : "../../../SOTA_Nips2021/dense-ulearn-vos/data/davis2017"
    parser.add_argument("--destination_path", type=str, default="ytvos")
    parser.add_argument("--evaluation_protocol", type=str, default="dataset-wise")
    parser.add_argument("--visualization_directory", type=str, default="visualizations")
    parser.add_argument("--logging_directory", type=str, default="logs")
    parser.add_argument("--EMA_decay", type=float, default=0.995)
    parser.add_argument("--lr_scheduler", type=str, default="CosineAnnealingLR")
    parser.add_argument("--head_lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--num_clusters", type=int, default=200)
    parser.add_argument("--input_resolution", type=int, default=224)
    parser.add_argument("--many_to_one", type=bool, default=False)
    parser.add_argument("--precision_based", type=bool, default=False)
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--n_last_frames", type=int, default=6, help="number of preceeding frames")
    parser.add_argument("--uvos", type=int, default=False)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--size_mask_neighborhood", default=6, type=int, help="We restrict the set of source nodes considered to a spatial neighborhood of the query node")
    parser.add_argument("--epsilon", default=0.05, type=float, help="epsilon for sinkhorn")
    parser.add_argument("--sinkhorn_iterations", default=3, type=float, help="number of sinkhorn iterations")
    parser.add_argument("--use_projection_head", type=bool, default=True, help="use projection head")
    parser.add_argument("--use_queue", type=bool, default=False, help="use queue")
    parser.add_argument("--queue_size", type=int, default=16384, help="queue size")
    parser.add_argument("--use_mask", type=bool, default=False, help="use mask")
    parser.add_argument("--use_teacher", type=bool, default=True, help="use EMA")
    parser.add_argument("--load_checkpoint", type=bool, default=False, help="load checkpoint")
    parser.add_argument("--regular_step", type=int, default=3, help="regular step")
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=3000, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    mp.spawn(time_tuning, nprocs=args.gpus, args=(args,))
    