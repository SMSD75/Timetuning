import os
import cv2
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import wandb
import io
import torch
import torchvision
import random
import faiss
import torch.nn as nn
from torch import distributed as dist
from sklearn.preprocessing import StandardScaler


def normalize_and_transform(feats: torch.Tensor, pca_dim: int) -> torch.Tensor:
    feats = feats.cpu().numpy()
    # Iteratively train scaler to normalize data
    bs = 100000
    num_its = (feats.shape[0] // bs) + 1
    scaler = StandardScaler()
    for i in range(num_its):
        scaler.partial_fit(feats[i * bs:(i + 1) * bs])
    print("trained scaler")
    for i in range(num_its):
        feats[i * bs:(i + 1) * bs] = scaler.transform(feats[i * bs:(i + 1) * bs])
    print(f"normalized feats to {feats.shape}")
    # Do PCA
    pca = faiss.PCAMatrix(feats.shape[-1], pca_dim)
    pca.train(feats)
    assert pca.is_trained
    transformed_val = pca.apply_py(feats)
    print(f"val feats transformed to {transformed_val.shape}")
    return torch.from_numpy(transformed_val)



def localize_objects(input_img, cluster_map):

    colors = ["orange", "blue", "red", "yellow", "white", "green", "brown", "purple", "gold", "black"]
    ticks = np.unique(cluster_map.flatten()).tolist()

    dc = np.zeros(cluster_map.shape)
    for i in range(cluster_map.shape[0]):
        for j in range(cluster_map.shape[1]):
            dc[i, j] = ticks.index(cluster_map[i, j])

    colormap = matplotlib.colors.ListedColormap(colors)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(13, 3))
    # plt.figure(figsize=(5,3))
    im = axes[0].imshow(dc, cmap=colormap, interpolation="none", vmin=-0.5, vmax=len(colors) - 0.5)
    cbar = fig.colorbar(im, ticks=range(len(colors)))
    axes[1].imshow(input_img)
    axes[2].imshow(dc, cmap=colormap, interpolation="none", vmin=-0.5, vmax=len(colors) - 0.5)
    axes[2].imshow(input_img, alpha=0.5)
    # plt.show(block=True)
    # plt.close()
    with io.BytesIO() as buffer:
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        return np.asarray(Image.open(buffer))


def denormalize(data, mean, std):
    denormalized_data = data * torch.tensor(std, device=data.device).view(1, 3, 1, 1) + torch.tensor(mean, device=data.device).view(1, 3, 1, 1)
    return denormalized_data

def imwrite_indexed(filename, array, color_palette):
    """ Save indexed png for DAVIS."""
    if np.atleast_3d(array).shape[2] != 1:
      raise Exception("Saving indexed PNGs requires 2D array.")

    im = Image.fromarray(array)
    im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')
    

def generate_colors(num_colors):
    """
    Generates a list of random colors
    """
    colors = []
    for i in range(num_colors):
        colors.append(tuple([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]))
    return colors


def make_figure(imgs):
    fig = plt.figure(figsize=(12, 9))
    plt.imshow(np.transpose(imgs, [1, 2, 0]))
    plt.axis('off')
    return fig


def make_seg_maps(data, cluster_map, logging_directory, name, w_featmap=28, h_featmap=28):
    bs, fs, c, h, w = data.shape
    # cluster_map = torch.Tensor(cluster_map.reshape(bs, fs, w_featmap, h_featmap))
    # cluster_map = nn.functional.interpolate(cluster_map.type(torch.DoubleTensor), scale_factor=8, mode="nearest").detach().cpu()
    cluster_map = cluster_map
    for i, datum in enumerate(data):
        frame_buffer = []
        for j, frame in enumerate(datum):
            frame_buffer.append(localize_objects(frame.permute(1, 2, 0).detach().cpu(), cluster_map[i, j]))
        convert_list_to_video(frame_buffer, name + "_" + str(i), speed=1000/ datum.size(0), directory=logging_directory, wdb_log=False)
        

def visualize_sampled_videos(samples, path, name):
    # os.system(f'rm -r {path}')
    scale_255 = lambda x : (x * 255).astype('uint8')
    layer, height, width = samples[0].shape[-3:]
    if not os.path.isdir(path):
        os.mkdir(path)
    video = cv2.VideoWriter(path + name, 0, 1, (width,height))
    if len(samples.shape) == 4: ## sampling a batch of images and not clips
        frames = samples
    else: ## clip-wise sampling
        frames = samples[0][0]  

    for frame in frames:
        if len(frame.shape) == 3:
            frame_1 = frame.permute(1, 2, 0).numpy()
        else:
            frame_1 = frame[..., None].repeat(1, 1, 3).numpy()
        temp = scale_255(frame_1)
        video.write(temp)
    video.release()
    cv2.destroyAllWindows()

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def convert_list_to_video(frames_list, name, speed, directory="", wdb_log=False):
    frames_list = [Image.fromarray(frame) for frame in frames_list]
    frames_list[0].save(f"{directory}{name}.gif", save_all=True, append_images=frames_list[1:], duration=speed, loop=0)
    if wdb_log:
        wandb.log({name: wandb.Video(f"{directory}{name}.gif", fps=4, format="gif")})


def visualize(images, num):
    figure, axs = plt.subplots(nrows=1, ncols=2)
    for i, ax in enumerate(axs.flat):
        images[i] = images[i].permute(1, 2, 0)
        ax.imshow(images[i])
        plt.savefig('Vis_output/frame_{}'.format(num))


def convert_fig_to_numpy(fig):
    with io.BytesIO() as buffer:
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        return np.asarray(Image.open(buffer))

def make_working_directory(name):
    if os.path.isdir(name):
        print("The direcotry already exists")
        filelist = [ f for f in os.listdir(name) if f.endswith(".gif") ]
        for f in filelist:
            os.remove(os.path.join(name, f))
    else:
        os.mkdir(name)

def make_logging_directory(name):
    if os.path.isdir(name):
        print("OOps your direcotyr exists.")
    else:
        os.mkdir(name)


def get_features(model, name, trans_input): ## gets a model and transformed inputs (normalized inputs) and returns feature maps as well as the possible attention maps
                                            ## trans_input shape is (bs * fs, c, size, size)
    with torch.no_grad():
        if name == "resnet18":
            outputs = []
            def hook(module, input, output):
                outputs.append(output)
            model.layer4[1].conv2.register_forward_hook(hook)
            model(trans_input)
            features = outputs[0].flatten(start_dim=2) ## The original output shape is [bs*fs, dim, w, h] and the dim is 256 for resnet18
            features = features.permute(0, 2, 1)
            return features, None

        if name == "resnet50" or name == "swav":
            outputs = []
            def hook(module, input, output):
                outputs.append(output)
            model.layer4[2].conv3.register_forward_hook(hook)
            model(trans_input)
            features = outputs[0].flatten(start_dim=2) ## The original output shape is [bs*fs, dim, w, h] and the dim is 256 for resnet18
            features = features.permute(0, 2, 1)
            return features, None

        elif name == "dino" or name == "vit" or name == "leopart":
            feat_out = {}
            def hook_fn_forward_qkv(module, input, output):
                feat_out["qkv"] = output
            model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)
            if name == "dino":
                attention = model.get_last_selfattention(trans_input)
            else:
                def hook_fn_forward_attn(module, input, output):
                    feat_out["attn"] = output
                model._modules["blocks"][-1]._modules["attn"]._modules["attn_drop"].register_forward_hook(hook_fn_forward_attn)
                output = model(trans_input)
                attention = feat_out["attn"]
            ns = attention.shape[0]
            nh = attention.shape[1]
            nb_tokens = attention.shape[2]
            # mask = creat_mask_from_attention(attention, 0.05).reshape(w_featmap * h_featmap, 1)
            qkv1 = (
                feat_out["qkv"]
                    .reshape(ns, nb_tokens, 3, nh, -1 // nh)
                    .permute(2, 0, 3, 1, 4)
            )
            q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]
            k1 = k1.transpose(1, 2).reshape(ns, nb_tokens, -1)
            q1 = q1.transpose(1, 2).reshape(ns, nb_tokens, -1)
            v1 = v1.transpose(1, 2).reshape(ns, nb_tokens, -1)
            # features = q1.squeeze(0)[1:, :].detach().cpu()
            # features = torch.hstack((q1.squeeze(0)[1:, :].detach().cpu(), k1.squeeze(0)[1:, :].detach().cpu(), v1.squeeze(0)[1:, :].detach().cpu()))
            # features = v1.squeeze(0)[1:, :].detach().cpu()
            return (q1[:, 1:, :].detach().cpu(), k1[:, 1:, :].detach().cpu(), v1[:, 1:, :].detach().cpu()), attention.detach().cpu()

        # elif name == "leopart":
        #     features, attn = model.forward_backbone(trans_input, True)
        #     features = features[:, 1:]
        #     return features, attn
            
        elif name == "stego":
            features = model(trans_input) ## (fs * bs, num_patches, w, h)
            features = features.flatten(2)
            features = features.permute(0, 2, 1)
            # code1 = model(trans_input)
            # code2 = model(trans_input.flip(dims=[3]))
            # code  = (code1 + code2.flip(dims=[3])) / 2
            # cluster_loss, cluster_probs = model.cluster_probe(features, 2, log_probs=False)
            return features, None


@torch.no_grad()
def sinkhorn(Q: torch.Tensor, nmb_iters: int, world_size=1) -> torch.Tensor:
    with torch.no_grad():
        Q = Q.detach().clone()
        sum_Q = torch.sum(Q)
        if world_size > 1:
            dist.all_reduce(sum_Q)
        Q /= sum_Q
        K, B = Q.shape
        u = torch.zeros(K).to(Q.device)
        r = torch.ones(K).to(Q.device) / K
        c = torch.ones(B).to(Q.device) / B * world_size

        if world_size > 1:
            curr_sum = torch.sum(Q, dim=1)
            dist.all_reduce(curr_sum)

        for _ in range(nmb_iters):
            if world_size > 1:
                u = curr_sum
            else:
                u = torch.sum(Q, dim=1)
            Q *= (r / u).unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
            if world_size > 1:
                curr_sum = torch.sum(Q, dim=1)
                dist.all_reduce(curr_sum)

        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()


    
def cosine_scheduler(base_value: float, final_value: float, epochs: int, niter_per_ep: int):
    # Construct cosine schedule starting at base_value and ending at final_value with epochs * niter_per_ep values.
    iters = np.arange(epochs * niter_per_ep)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    assert len(schedule) == epochs * niter_per_ep
    return schedule