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
from metrics import PredsmIoU, PredsmIoU_1
import torchvision.transforms as trn
from sklearn.cluster import KMeans
import torchvision
from my_utils import convert_list_to_video, cosine_scheduler, make_working_directory, make_seg_maps, localize_objects, sinkhorn
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

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)


dataset_object_numbers = {"davis": 52, "ytvos":80, "pascal":21}
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')



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



def evaluate_propagation(PredsEval: PredsmIoU, gts:torch.Tensor, preds: torch.Tensor) -> float:
    """
        Evaluate the mask propagation performance for the given preds and gts of a batch. Note that it is importance to pass the entire dataset gt and preds.
        :param PredsEval: PredsEval object that will be used for evaluation
        :param gts: ground truth masks of the entire dataset. Shape: [bs, fs, h, w]
        :param preds: predicted masks of the entire dataset. Shape: [bs, fs, h, w]
        :param logging_directory: directory to save the results of the evaluation
        :return: mIoU score of the given predictions as a float. The averaging is done over all the objects.
    """
    bs, fs, h, w = preds.shape
    scores = []
    for i in range(bs):
        PredsEval.reset()
        for j in range(fs):
            PredsEval.update(preds[i, j].flatten(), gts[i, j].flatten())
        clip_scores = PredsEval.compute_propagation_score(is_global_zero=True) ## this is the score list of objects for the entire clip averaged over time
        scores += clip_scores
    scores = np.array(scores)
    return scores.mean()



def evaluate_localizations(PredsEval, gts, preds, evaluation_protocol, logging_directory, many_to_one=False, precision_based=False): ## gets tensors with the [bs, fs, input_resolution, input_resolution] and does the evaluation
    ## visualization of the heatmaps is very ugly. It should be changed some how.

    bs, fs, h, w = preds.shape
    scores = []
    if logging_directory is not None:
        frame_buffer = []
        frame_buffer_1 = [] ## This is added only to visualize the effect of reordering on the cluster maps. It should be removed.
        sub_directory = logging_directory + "/" + evaluation_protocol
        make_working_directory(sub_directory)
    if evaluation_protocol == "frame-wise":
        for i, datum in enumerate(preds):
            clip_score = [] ## just for the sake of visualization
            if logging_directory is not None:
                frame_buffer = []
                frame_buffer_1 = []
                clip_score = []
            for j, frame in enumerate(datum):
                PredsEval.update(gts[i, j].flatten(), frame.flatten())
                score, tp, fp, fn, reordered_preds, matched_bg_clusters = PredsEval.compute(True, many_to_one, precision_based=precision_based)
                if logging_directory is not None:
                    frame_buffer.append(localize_objects(gts[i, j], reordered_preds.reshape(h, w)))
                    frame_buffer_1.append(localize_objects(gts[i, j], frame))
                scores.append(score)
                clip_score.append(score)
                PredsEval.reset()
            if logging_directory is not None:
                convert_list_to_video(frame_buffer, f"Score:{sum(clip_score)/len(clip_score)}_Evaluation_{evaluation_protocol}_Reordered_{i}", speed=80, directory=sub_directory + "/", wdb_log=False)
                convert_list_to_video(frame_buffer_1, f"Score:{sum(clip_score)/len(clip_score)}_Evaluation_{evaluation_protocol}_Inorder_{i}", speed=80, directory=sub_directory + "/", wdb_log=False)
    elif evaluation_protocol == "sample-wise":
        for i, datum in enumerate(preds):
            if logging_directory is not None:
                frame_buffer = []
                frame_buffer_1 = []
                clip_score = [] ## just for the sake of visualization
            for j, frame in enumerate(datum):
                PredsEval.update(gts[i, j].flatten(), frame.flatten())
                if logging_directory is not None:
                    frame_buffer_1.append(localize_objects(gts[i, j], frame))
            score, tp, fp, fn, reordered_preds, matched_bg_clusters = PredsEval.compute(True, many_to_one, precision_based=precision_based)
            reordered_preds = reordered_preds.reshape(fs, h, w)
            if logging_directory is not None:
                for j, cluster_map in enumerate(reordered_preds):
                    frame_buffer.append(localize_objects(gts[i, j], cluster_map))
            scores.append(score)
            if logging_directory is not None:
                clip_score.append(score)
            PredsEval.reset()
            if logging_directory is not None:
                convert_list_to_video(frame_buffer, f"Score-{sum(clip_score)/len(clip_score)}_Evaluation_{evaluation_protocol}_Reordered_{i}", speed=80, directory=sub_directory + "/", wdb_log=False)
                convert_list_to_video(frame_buffer_1, f"Score-{sum(clip_score)/len(clip_score)}_Evaluation_{evaluation_protocol}_Inorder_{i}", speed=80, directory=sub_directory + "/", wdb_log=False)
    elif evaluation_protocol == "dataset-wise":
        for i, datum in enumerate(preds):
            for j, frame in enumerate(datum):
                valid = gts[i, j] != 255   # Only for Pascal dataset
                PredsEval.update(gts[i, j][valid].flatten(), frame[valid].flatten())  # Only for Pascal dataset
                # PredsEval.update(gts[i, j].flatten(), frame.flatten()) ## For all the other datasets
        score, tp, fp, fn, reordered_preds, matched_bg_clusters = PredsEval.compute(True, many_to_one, precision_based=precision_based)
        scores.append(score)
        PredsEval.reset()
    return (sum(scores) / len(scores))


def convert_to_image_dataset(video_dataset_path, destination, name):
    if os.path.exists(destination):
        print("Your directory already exists")
        shutil.rmtree(destination)
    print("A new directory has been made.")
    os.mkdir(destination)
    imgs_path = os.path.join(destination, "imgs")
    labels_path = os.path.join(destination, "labels")
    os.mkdir(imgs_path)
    os.mkdir(labels_path)
    os.mkdir(os.path.join(imgs_path, "train"))
    os.mkdir(os.path.join(imgs_path, "val"))
    os.mkdir(os.path.join(labels_path, "train"))
    os.mkdir(os.path.join(labels_path, "val"))
    if name == "davis":
        classes_dir = video_dataset_path + "/JPEGImages/480p"
        class_annotations_dir = video_dataset_path + "/Annotations/480p"
        class_names = os.listdir(classes_dir)

        for class_name in class_names:
            for file in os.listdir(os.path.join(classes_dir, class_name)):
                shutil.copyfile(classes_dir + "/" + class_name + "/" + file, destination + "/imgs/train/" + class_name + "_" + file)

        class_names = os.listdir(class_annotations_dir)
        for class_name in class_names:
            for file in os.listdir(os.path.join(class_annotations_dir, class_name)):
                shutil.copyfile(class_annotations_dir + "/" + class_name + "/" + file, destination + "/labels/train/" + class_name + "_" + file)


class Evaluator(object):

    """Class for evaluating the performance of a segmentation model.
    Args:
        data_loader (torch.utils.data.DataLoader): a torch DataLoader object that loads the dataset.
        model (torch.nn.Module): a torch model that is used to predict the segmentation map.
        logging_directory (str): the directory where the evaluation loggs will be saved.
        uvos_flag (bool): a flag that indicates whether the dataset is UVOS or not.
        logger_name (str): the name of the logger file.
        Note that the samples of the data loader should be normalized in the transformation phase as well.
    """

    def __init__(self, data_loader, model, logging_directory=None, uvos_flag=False, clustering_algorithm="k-means", logger_name="Evaluator", fg_masks=None, device=None):
        self.data_loader = data_loader
        self.model = model
        self.logging_directory = logging_directory
        self.uvos_flag = uvos_flag
        self.clustering_algorithm = clustering_algorithm
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.PredsEval = PredsmIoU(10, 10, involve_bg=True) ## 10 is the number of classes but we don't care about it since we are not using it
        self.logger =  logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(f"{logging_directory}/{logger_name}.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.fg_masks = fg_masks

    def evaluate(self, many_to_one=False, evaluation_protocol="frame-wise", eval_resolution=None, num_clusters=10, use_mask=False, use_annotations=False, precision_based=False):
        """Evaluates the model on the dataset.
        Args:
            many_to_one (bool): a flag that indicates whether the many-to-one matching algorithm should be used or not.
            evaluation_protocol (str): the evaluation protocol that should be used. It can be either "frame-wise", "sample-wise" or "dataset-wise".
            eval_resolution (tuple): the resolution that the evaluation should be done on. If None, the original resolution will be used.
            num_clusters (int): the number of clusters that should be used in the many-to-one matching algorithm.
        """

        self.model.eval()
        if isinstance(self.model, FeatureExtractor):
            spatial_resolution = self.model.spatial_resolution
        else:
            spatial_resolution = self.model.feature_extractor.spatial_resolution
        if evaluation_protocol == "dataset-wise":
            clip_center_list = []
            annotations_list = []
            feature_list = []  ## This variable is only used for the dataset-wise evaluation
            for i, train_data in enumerate(self.data_loader):
                if len(train_data)  == 3:
                    data, annotations, label = train_data
                else:
                    data, annotations = train_data
                if len(data.shape) == 6:
                    data = data.squeeze(1)
                    annotations = annotations.squeeze(1)
                else:
                    data = data.unsqueeze_(1)
                    # annotations.unsqueeze_(1)  ### Commnet this line for Pascal VOC
                bs, fs, c, h, w = data.shape
                data = data.view(bs * fs, c, h, w)
                data = data.to(self.device) 
                annotations *= 255  ## This line is only used for Pascal VOC
                annotations = annotations.long()
                # print(annotations.unique())
                features, attentions = self.model(data, use_head=False)
                _, num_patches, dim = features.shape
                features = features.view(bs, fs, num_patches, dim)
                if use_mask and (self.fg_masks is None):
                    features, attentions = apply_attention_mask(features, attentions, spatial_resolution)
                    print("Applying attention mask")

                feature_list.append(features)
                annotations_list.append(annotations)
            features = torch.cat(feature_list, dim=0)
            if use_mask and (self.fg_masks is not None):
                print("Before interpolation")
                print(self.fg_masks.shape)
                fg_masks = self.fg_masks.reshape(features.shape[0], features.shape[1], self.fg_masks.size(-1), self.fg_masks.size(-1))
                fg_masks = F.interpolate(fg_masks.float(), size=(spatial_resolution, spatial_resolution), mode="nearest")
                fg_masks = fg_masks.flatten(2, 3).unsqueeze(-1)
                print("Applying foreground mask")
                print(fg_masks.shape)
                print(features.shape)
                features = features * fg_masks
            annotations = torch.cat(annotations_list, dim=0)
            print(annotations.shape)
            annotations = nn.functional.interpolate(annotations.type(torch.DoubleTensor), size=(eval_resolution, eval_resolution), mode="nearest")
            if self.clustering_algorithm == "k-means":
                if use_annotations:
                    cluster_maps = cluster_features(features, num_clusters, spatial_resolution, eval_resolution, evaluation_protocol, annotations)
                else:
                    cluster_maps = cluster_features(features, num_clusters, spatial_resolution, eval_resolution, evaluation_protocol)
            elif self.clustering_algorithm == "prototypes":
                bs, fs, num_patches, dim = features.shape
                cluster_maps = proto_clustering(features.view(bs * fs, num_patches, dim), self.model.prototypes, spatial_resolution, output_size=eval_resolution, num_classes=num_clusters)
                cluster_maps = cluster_maps.view(bs, fs, eval_resolution, eval_resolution)
                
            iou_scores = evaluate_localizations(self.PredsEval, annotations, cluster_maps, evaluation_protocol, logging_directory=None, many_to_one=many_to_one, precision_based=precision_based)
            print(f"Dataset score is {iou_scores}")
            self.logger.info(f"Dataset score is {iou_scores}")
            return iou_scores

        
        elif evaluation_protocol == "sample-wise" or evaluation_protocol == "frame-wise": 
            batch_iou_scores = []
            for i, train_data in enumerate(self.data_loader):
                data, annotations, label = train_data
                data = data.squeeze(1)
                annotations = annotations.squeeze(1)
                bs, fs, c, h, w = data.shape
                data = data.view(bs * fs, c, h, w)
                self.logger.info(f"The data that is passed to the model has the shape : {data.shape}")
                data = data.to(self.device) 
                features, attentions = self.model(data, use_head=False) ## shape [bs * fs, num_patches, dim]
                self.logger.info(f"The final extracted feature map has the shape : {features.shape}")
                _, num_patches, dim = features.shape
                features = features.view(bs, fs, num_patches, dim)
                if use_mask:
                    features, attentions = apply_attention_mask(features, attentions, spatial_resolution)

                if self.uvos_flag:
                    idx = annotations > 0
                    annotations[idx] = 1
                    attentions = process_attentions(attentions, spatial_resolution)

                if self.clustering_algorithm == "k-means":
                    if use_annotations:
                        cluster_maps = cluster_features(features, num_clusters, spatial_resolution, eval_resolution, evaluation_protocol, annotations)
                    else:
                        cluster_maps = cluster_features(features, num_clusters, spatial_resolution, eval_resolution, evaluation_protocol)
                elif self.clustering_algorithm == "prototypes":
                    cluster_maps = proto_clustering(features.view(bs * fs, num_patches, dim), self.model.prototypes, spatial_resolution, output_size=eval_resolution, num_classes=num_clusters)
                    cluster_maps = cluster_maps.view(bs, fs, eval_resolution, eval_resolution)

                batch_score = evaluate_localizations(self.PredsEval, annotations, cluster_maps, evaluation_protocol, logging_directory=None, many_to_one=many_to_one, precision_based=precision_based)
                print(f"batch score is {batch_score}")
                self.logger.info(f"batch score is {batch_score}")
                batch_iou_scores.append(batch_score)
            final_score = sum(batch_iou_scores) / len(batch_iou_scores)
            print(f"Dataset score is {final_score}")
            self.logger.info(f"Dataset score is {sum(batch_iou_scores) / len(batch_iou_scores)}")
            return final_score
        self.model.train()
        


def main(args):
    num_epochs = 50
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    np.seterr(all='raise')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
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

    feature_extractor = FeatureExtractor(architecture, model_path, [1024, 1024, 512, 256])  ##  [1024, 1024, 512, 256] unfreeze_layers=["blocks.11", "blocks.10"]
    # model = feature_extractor
    model = TimeT(feature_extractor, 200)
    if use_teacher:
        model.init_momentum_teacher()
        model.set_momentum_teacher_schedular_params(EMA_decay, 1., num_epochs, num_itr)
    # model.load_state_dict(torch.load('/home/ssalehi/video/DeTeFFp/logs/20230307/001011/0.09746987611917557_99.pth')) #'0.1365865812925643_152737_dino_ytvos_128_200.pth'
    # model = FeatureExtractor(architecture, model_path)
    model = model.to(device)
    
    print(f"The selected model is {architecture} with the architecture as follows:")
    print(model)

    video_transform_list = [video_transformations.Resize((input_resolution, input_resolution), 'bilinear'), video_transformations.CenterCrop(input_resolution), video_transformations.ClipToTensor(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225])]
    video_transform = video_transformations.Compose(video_transform_list)
    train_loader = make_loader(dataset, num_frames, batch_size, SamplingMode.UNIFORM, frame_transform=None, target_transform=None, video_transform=video_transform, shuffle=False, num_workers=num_workers, pin_memory=True)
    eval_resolution = 112 if evaluation_protocol == "dataset-wise" else input_resolution
    # train_loader = pascal_loader(60, "../../dataset/leopascal/VOCSegmentation", "val", eval_resolution, train_size=input_resolution) ## Uncomment this line for Pascal VOC
    print("The dataset has been read.")
    evaluator = Evaluator(train_loader, model, logging_directory, uvos_flag, "k-means", f"evaluator1_{architecture}_{dataset}_{batch_size}_{num_clusters}_{input_resolution}_{evaluation_protocol}_{many_to_one}")
    evaluator.evaluate(many_to_one=many_to_one, evaluation_protocol=evaluation_protocol, eval_resolution=eval_resolution, num_clusters=num_clusters, use_annotations=False, use_mask=False, precision_based=precision_based)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", type=str, default="dino-s16", help="which back-bone architecture do you want to use?")
    parser.add_argument("--model_path", type=str, default= "/home/ssalehi/video/vos_pretrained/cyclic_swav/src/leopart_vits16.ckpt") # "../models/leopart_vits16.ckpt"
    parser.add_argument("--dataset", type=str, default="davis_val")
    parser.add_argument("--dataset_path", type=str, default="../data") ## davis : "../../../SOTA_Nips2021/dense-ulearn-vos/data/davis2017"
    parser.add_argument("--destination_path", type=str, default="ytvos")
    parser.add_argument("--evaluation_protocol", type=str, default="frame-wise")
    parser.add_argument("--logging_directory", type=str, default="visualizations")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=3)
    parser.add_argument("--num_clusters", type=int, default=10)
    parser.add_argument("--input_resolution", type=int, default=224)
    parser.add_argument("--many_to_one", type=bool, default=False)
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--precision_based", type=bool, default=False)
    parser.add_argument("--uvos", type=int, default=False)
    parser.add_argument("--use_teacher", type=bool, default=False)
    parser.add_argument("--EMA_decay", type=float, default=0.999)
    args = parser.parse_args()
    main(args)

