import glob
from platform import platform
from posixpath import dirname
from PIL import Image
# import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from enum import Enum
import os
import cv2
import numpy as np
import random
import torchvision.transforms as trn
from torch.utils.data.distributed import DistributedSampler
from my_utils import visualize_sampled_videos, numericalSort, localize_objects, convert_list_to_video, make_seg_maps
import json
import matplotlib.pyplot as plt
import torchvision.transforms.functional as f
from collections import OrderedDict
from os.path import join as pjoin
import collections
import scipy.misc as m
import scipy.io as io
import glob
from tqdm import tqdm
from torch.utils import data
from torchvision import transforms
from numpy import array, indices
import video_transformations
import numpy as np
from PIL import Image
import shutil


platform = None
if "node" in os.uname()[1]:
    platform = "das6"
elif "ivi-cn" in os.uname()[1]:
    platform = "ivi"
elif "quva01" in os.uname()[1]:
    platform = "quva01"
else:
    platform = "dta2"



_errstr = "Mode is unknown or incompatible with input array shape."
dataset_path = {"das6": {"davis": "/var/scratch/ssalehid/datasets/", "ytvos" : "/var/scratch/ssalehid/ytvos/", "pascal": "/var/scratch/ssalehid/datasets/"},
                "ivi": {"davis": "../../dataset", "pascal": "../../dataset", "ytvos": "../../dataset", "kinetics": "/ssdstore/ssalehi", "mose": "/ssdstore/ssalehi/MOSE", "epic-kitchen": "/ssdstore/ssalehi/EpicKitchens", "visor": "/ssdstore/ssalehi/VISOR/out_data/VISOR_2022"},
                "dta2": {"davis": "../../data/datasets/", "ytvos" : "../../data/", "pascal": "../../data/datasets/", "kinetics": "../../data/"},
                "quva01": {"davis": "../../../datasets/", "ytvos" : "../../../datasets/ytvos/data/", "pascal": "../../../datasets/"}}


def fromimage(im, flatten=False, mode=None):
    """
    Return a copy of a PIL image as a numpy array.
    This function is only available if Python Imaging Library (PIL) is installed.
    Parameters
    ----------
    im : PIL image
        Input image.
    flatten : bool
        If true, convert the output to grey-scale.
    mode : str, optional
        Mode to convert image to, e.g. ``'RGB'``.  See the Notes of the
        `imread` docstring for more details.
    Returns
    -------
    fromimage : ndarray
        The different colour bands/channels are stored in the
        third dimension, such that a grey-image is MxN, an
        RGB-image MxNx3 and an RGBA-image MxNx4.
    """
    if not Image.isImageType(im):
        raise TypeError("Input is not a PIL image.")

    if mode is not None:
        if mode != im.mode:
            im = im.convert(mode)
    elif im.mode == 'P':
        # Mode 'P' means there is an indexed "palette".  If we leave the mode
        # as 'P', then when we do `a = array(im)` below, `a` will be a 2-D
        # containing the indices into the palette, and not a 3-D array
        # containing the RGB or RGBA values.
        if 'transparency' in im.info:
            im = im.convert('RGBA')
        else:
            im = im.convert('RGB')

    if flatten:
        im = im.convert('F')
    elif im.mode == '1':
        # Workaround for crash in PIL. When im is 1-bit, the call array(im)
        # can cause a seg. fault, or generate garbage. See
        # https://github.com/scipy/scipy/issues/2138 and
        # https://github.com/python-pillow/Pillow/issues/350.
        #
        # This converts im from a 1-bit image to an 8-bit image.
        im = im.convert('L')

    a = array(im)
    return a



def imread(name, flatten=False, mode=None):
    """
    Read an image from a file as an array.
    This function is only available if Python Imaging Library (PIL) is installed.
    Parameters
    ----------
    name : str or file object
        The file name or file object to be read.
    flatten : bool, optional
        If True, flattens the color layers into a single gray-scale layer.
    mode : str, optional
        Mode to convert image to, e.g. ``'RGB'``.  See the Notes for more
        details.
    Returns
    -------
    imread : ndarray
        The array obtained by reading the image.
    Notes
    -----
    `imread` uses the Python Imaging Library (PIL) to read an image.
    The following notes are from the PIL documentation.
    `mode` can be one of the following strings:
    * 'L' (8-bit pixels, black and white)
    * 'P' (8-bit pixels, mapped to any other mode using a color palette)
    * 'RGB' (3x8-bit pixels, true color)
    * 'RGBA' (4x8-bit pixels, true color with transparency mask)
    * 'CMYK' (4x8-bit pixels, color separation)
    * 'YCbCr' (3x8-bit pixels, color video format)
    * 'I' (32-bit signed integer pixels)
    * 'F' (32-bit floating point pixels)
    PIL also provides limited support for a few special modes, including
    'LA' ('L' with alpha), 'RGBX' (true color with padding) and 'RGBa'
    (true color with premultiplied alpha).
    When translating a color image to black and white (mode 'L', 'I' or
    'F'), the library uses the ITU-R 601-2 luma transform::
        L = R * 299/1000 + G * 587/1000 + B * 114/1000
    When `flatten` is True, the image is converted using mode 'F'.
    When `mode` is not None and `flatten` is True, the image is first
    converted according to `mode`, and the result is then flattened using
    mode 'F'.
    """

    im = Image.open(name)
    return fromimage(im, flatten=flatten, mode=mode)

def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    """
    Byte scales an array (image).
    Byte scaling means converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255).
    If the input image already has dtype uint8, no scaling is done.
    This function is only available if Python Imaging Library (PIL) is installed.
    Parameters
    ----------
    data : ndarray
        PIL image data array.
    cmin : scalar, optional
        Bias scaling of small values. Default is ``data.min()``.
    cmax : scalar, optional
        Bias scaling of large values. Default is ``data.max()``.
    high : scalar, optional
        Scale max value to `high`.  Default is 255.
    low : scalar, optional
        Scale min value to `low`.  Default is 0.
    Returns
    -------
    img_array : uint8 ndarray
        The byte-scaled array.
    Examples
    --------
    >>> from scipy.misc import bytescale
    >>> img = np.array([[ 91.06794177,   3.39058326,  84.4221549 ],
    ...                 [ 73.88003259,  80.91433048,   4.88878881],
    ...                 [ 51.53875334,  34.45808177,  27.5873488 ]])
    >>> bytescale(img)
    array([[255,   0, 236],
           [205, 225,   4],
           [140,  90,  70]], dtype=uint8)
    >>> bytescale(img, high=200, low=100)
    array([[200, 100, 192],
           [180, 188, 102],
           [155, 135, 128]], dtype=uint8)
    >>> bytescale(img, cmin=0, cmax=255)
    array([[91,  3, 84],
           [74, 81,  5],
           [52, 34, 28]], dtype=uint8)
    """
    if data.dtype == np.uint8:
        return data

    if high > 255:
        raise ValueError("`high` should be less than or equal to 255.")
    if low < 0:
        raise ValueError("`low` should be greater than or equal to 0.")
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)


def toimage(arr, high=255, low=0, cmin=None, cmax=None, pal=None,
            mode=None, channel_axis=None):
    """Takes a numpy array and returns a PIL image.
    This function is only available if Python Imaging Library (PIL) is installed.
    The mode of the PIL image depends on the array shape and the `pal` and
    `mode` keywords.
    For 2-D arrays, if `pal` is a valid (N,3) byte-array giving the RGB values
    (from 0 to 255) then ``mode='P'``, otherwise ``mode='L'``, unless mode
    is given as 'F' or 'I' in which case a float and/or integer array is made.
    .. warning::
        This function uses `bytescale` under the hood to rescale images to use
        the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.
        It will also cast data for 2-D images to ``uint32`` for ``mode=None``
        (which is the default).
    Notes
    -----
    For 3-D arrays, the `channel_axis` argument tells which dimension of the
    array holds the channel data.
    For 3-D arrays if one of the dimensions is 3, the mode is 'RGB'
    by default or 'YCbCr' if selected.
    The numpy array must be either 2 dimensional or 3 dimensional.
    """
    data = np.asarray(arr)
    if np.iscomplexobj(data):
        raise ValueError("Cannot convert a complex-valued array.")
    shape = list(data.shape)
    valid = len(shape) == 2 or ((len(shape) == 3) and
                                ((3 in shape) or (4 in shape)))
    if not valid:
        raise ValueError("'arr' does not have a suitable array shape for "
                         "any mode.")
    if len(shape) == 2:
        shape = (shape[1], shape[0])  # columns show up first
        if mode == 'F':
            data32 = data.astype(np.float32)
            image = Image.frombytes(mode, shape, data32.tostring())
            return image
        if mode in [None, 'L', 'P']:
            bytedata = bytescale(data, high=high, low=low,
                                 cmin=cmin, cmax=cmax)
            image = Image.frombytes('L', shape, bytedata.tostring())
            if pal is not None:
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
                # Becomes a mode='P' automagically.
            elif mode == 'P':  # default gray-scale
                pal = (np.arange(0, 256, 1, dtype=np.uint8)[:, np.newaxis] *
                       np.ones((3,), dtype=np.uint8)[np.newaxis, :])
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
            return image
        if mode == '1':  # high input gives threshold for 1
            bytedata = (data > high)
            image = Image.frombytes('1', shape, bytedata.tostring())
            return image
        if cmin is None:
            cmin = np.amin(np.ravel(data))
        if cmax is None:
            cmax = np.amax(np.ravel(data))
        data = (data*1.0 - cmin)*(high - low)/(cmax - cmin) + low
        if mode == 'I':
            data32 = data.astype(np.uint32)
            image = Image.frombytes(mode, shape, data32.tostring())
        else:
            raise ValueError(_errstr)
        return image

    # if here then 3-d array with a 3 or a 4 in the shape length.
    # Check for 3 in datacube shape --- 'RGB' or 'YCbCr'
    if channel_axis is None:
        if (3 in shape):
            ca = np.flatnonzero(np.asarray(shape) == 3)[0]
        else:
            ca = np.flatnonzero(np.asarray(shape) == 4)
            if len(ca):
                ca = ca[0]
            else:
                raise ValueError("Could not find channel dimension.")
    else:
        ca = channel_axis

    numch = shape[ca]
    if numch not in [3, 4]:
        raise ValueError("Channel axis dimension is not valid.")

    bytedata = bytescale(data, high=high, low=low, cmin=cmin, cmax=cmax)
    if ca == 2:
        strdata = bytedata.tostring()
        shape = (shape[1], shape[0])
    elif ca == 1:
        strdata = np.transpose(bytedata, (0, 2, 1)).tostring()
        shape = (shape[2], shape[0])
    elif ca == 0:
        strdata = np.transpose(bytedata, (1, 2, 0)).tostring()
        shape = (shape[2], shape[1])
    if mode is None:
        if numch == 3:
            mode = 'RGB'
        else:
            mode = 'RGBA'

    if mode not in ['RGB', 'RGBA', 'YCbCr', 'CMYK']:
        raise ValueError(_errstr)

    if mode in ['RGB', 'YCbCr']:
        if numch != 3:
            raise ValueError("Invalid array shape for mode.")
    if mode in ['RGBA', 'CMYK']:
        if numch != 4:
            raise ValueError("Invalid array shape for mode.")

    # Here we know data and mode is correct
    image = Image.frombytes(mode, shape, strdata)
    return image


def imsave(name, arr, format=None):
    """
    Save an array as an image.
    This function is only available if Python Imaging Library (PIL) is installed.
    .. warning::
        This function uses `bytescale` under the hood to rescale images to use
        the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.
        It will also cast data for 2-D images to ``uint32`` for ``mode=None``
        (which is the default).
    Parameters
    ----------
    name : str or file object
        Output file name or file object.
    arr : ndarray, MxN or MxNx3 or MxNx4
        Array containing image values.  If the shape is ``MxN``, the array
        represents a grey-level image.  Shape ``MxNx3`` stores the red, green
        and blue bands along the last dimension.  An alpha layer may be
        included, specified as the last colour band of an ``MxNx4`` array.
    format : str
        Image format. If omitted, the format to use is determined from the
        file name extension. If a file object was used instead of a file name,
        this parameter should always be used.
    Examples
    --------
    Construct an array of gradient intensity values and save to file:
    >>> from scipy.misc import imsave
    >>> x = np.zeros((255, 255))
    >>> x = np.zeros((255, 255), dtype=np.uint8)
    >>> x[:] = np.arange(255)
    >>> imsave('gradient.png', x)
    Construct an array with three colour bands (R, G, B) and store to file:
    >>> rgb = np.zeros((255, 255, 3), dtype=np.uint8)
    >>> rgb[..., 0] = np.arange(255)
    >>> rgb[..., 1] = 55
    >>> rgb[..., 2] = 1 - np.arange(255)
    >>> imsave('rgb_gradient.png', rgb)
    """
    im = toimage(arr, channel_axis=2)
    if format is None:
        im.save(name)
    else:
        im.save(name, format)
    return



torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)



class SamplingMode(Enum):
    UNIFORM = 0
    DENSE = 1
    Full = 2
    Regular = 3



def diff_annotation_data_directories(annotation_directory_path, data_directory_path):
    annotation_subdir = sorted(os.listdir(annotation_directory_path))
    data_subdir = sorted(os.listdir(data_directory_path))
    if (annotation_subdir == data_subdir) and (len(annotation_subdir) == len(data_subdir)):
        print("The lists are equal.")
    else:
        print("The lists are not equal.")
    
    print("=======================")
    print(annotation_subdir[:10])
    print(data_subdir[:10])


def make_categories_dict(meta_dict, name):
    category_list = []
    if name == "ytvos":
        video_name_list = meta_dict["videos"].keys()
        for name in video_name_list:
            obj_list = meta_dict["videos"][name]["objects"].keys()
            for obj in obj_list:
                if meta_dict["videos"][name]["objects"][obj]["category"] not in category_list:
                    category_list.append(meta_dict["videos"][name]["objects"][obj]["category"])
        category_list = sorted(list(OrderedDict.fromkeys(category_list)))
        category_ditct = {k: v+1 for v, k in enumerate(category_list)} ## zero is always for the background
    return category_ditct










## Be careful it is not considering each annotation sample independently!!!!
## We need to convert masks to the indexed_RGB format to adapt it to the further stages of evaluation.
def convert_to_indexed_RGB(inp):
    bs, fs, channel, h, w = inp.shape
    # plt.imshow(inp[0, 0].permute(1, 2, 0))
    # plt.show()
    inp = inp.permute(0, 1, 3, 4, 2)
    inp = inp.contiguous().view(-1, channel)
    unique_colors = np.unique(inp.numpy(), axis=0)
    indexed_inp = torch.zeros((inp.shape[0], 1))
    # print(unique_colors)
    # print(unique_colors)
    for i, color in enumerate(unique_colors):
        mask = torch.all(inp == color, dim=1)
        indexed_inp[mask] = i
    indexed_inp = indexed_inp.view(bs, fs, h, w)
    # plt.imshow(20 * indexed_inp[0, 0].unsqueeze(2).repeat(1, 1, 3))
    # plt.show()
    return indexed_inp


## supposes the input is between 0 and 1
def map_instances(data, meta, category_dict):
    bs, fs, h, w = data.shape
    for i, datum in enumerate(data):
        for j, frame in enumerate(data):
            objects = torch.unique(frame)
            for k, obj in enumerate(objects):
                if int(obj.item()) == 0:
                    continue
                frame[frame == obj] = category_dict[meta[str(int(obj.item()))]["category"]]
    return data


def build_dataset_tree(initial_directory, class_trajectory, dataset_dict, num_labels=1):
    dirs = os.listdir(initial_directory)
    class_num = 0
    for path in dirs:
        if os.path.isfile(initial_directory + path):
            if len(class_trajectory) == 0:
                continue
            dir_name = initial_directory + path.split('.')[0]
            dataset_dict[dir_name] = np.array(class_trajectory)
            if not os.path.isdir(dir_name):
                os.mkdir(dir_name)
            else:
                # print(f"The directory {dir_name} exists!")
                continue
            reader = cv2.VideoCapture(initial_directory + path)
            frame_num = 0
            while(True):
                ret, frame = reader.read()
                if not ret:
                    break
                cv2.imwrite(dir_name + "/" + f'{frame_num:05d}' + ".jpg", frame)
                frame_num += 1
            cv2.destroyAllWindows()
            reader.release()
            os.remove(initial_directory + path)
        else:
            if len(class_trajectory) == num_labels:
                dataset_dict[initial_directory + path] = np.array(class_trajectory)
            else:
                build_dataset_tree(initial_directory + path + "/", class_trajectory + [class_num], dataset_dict, num_labels)
            class_num += 1
    return dataset_dict


class VideoDataset(torch.utils.data.Dataset):
    ## The data loader gets training sample and annotations direcotories, sampling mode, number of clips that is being sampled of each training video, number of frames in each clip
    ## and number of labels for each training clip. 
    ## Note that the number of annotations should be exactly similar to the number of frames existing in the training path.
    ## Frame_transform is a function that transforms the frames of the video. It is applied to each frame of the video.
    ## Target_transform is a function that transforms the annotations of the video. It is applied to each annotation of the video.
    ## Video_transform is a function that transforms the whole video. It is applied to both frames and annotations of the video.
    ## The same set of transformations is applied to the clips of the video.

    def __init__(self, classes_directory, annotations_directory, sampling_mode, num_clips, num_frames, num_labels, frame_transform=None, target_transform=None, video_transform=None, meta_file_directory=None, regular_step=1) -> None:
        super().__init__()
        self.train_dict = build_dataset_tree(classes_directory, [], {}, num_labels)
        self.train_dict_lenghts = {}
        self.find_directory_length()
        if (annotations_directory != "") and (os.path.exists(annotations_directory)):
            self.train_annotations_dict = build_dataset_tree(annotations_directory, [], {}, num_labels)
            self.use_annotations = True
        else:
            self.use_annotations = False
            print("Because there is no annotation directory, only training samples have been loaded.")
        if (meta_file_directory is not None):
            if (os.path.exists(meta_file_directory)):
                print("Meta file has been read.")
                file = open(meta_file_directory)
                self.meta_dict = json.load(file)
            else:
                self.meta_dict = None
                print("There is no meta file.")
        else:
            print("Meta option is off.")
            self.meta_dict = None
         
        self.keys = sorted(list(self.train_dict.keys()))
        if self.use_annotations:
            self.annotation_keys = sorted(list(self.train_annotations_dict.keys()))
        self.sampling_mode = sampling_mode
        self.num_clips = num_clips
        self.num_frames = num_frames
        self.frame_transform = frame_transform
        self.target_transform = target_transform
        self.video_transform = video_transform
        self.regular_step = regular_step
        
    def __len__(self):
        return len(self.keys)

    
    def find_directory_length(self):
        for key in self.train_dict:
            self.train_dict_lenghts[key] = len(os.listdir(key))

    
    # def read_clip_frames(self, path):
    #     files = sorted(glob.glob(path + "/" + "*.jpg"), key=numericalSort)
    #     if len(files) == 0:
    #         files = sorted(glob.glob(path + "/" + "*.png"), key=numericalSort)
    #     images = []
    #     for file in files:
    #         frame = Image.open(file)
    #         images.append(frame)
    #     return images
    
    def read_clips(self, path, clip_indices):
        clips = []
        files = sorted(glob.glob(path + "/" + "*.jpg"))
        if len(files) == 0:
            files = sorted(glob.glob(path + "/" + "*.png"))
        for i in range(len(clip_indices)):
            images = []
            for j in clip_indices[i]:
                # frame_path = path + "/" + f'{j:05d}' + ".jpg"
                frame_path = files[j]
                if not os.path.exists(frame_path):
                    frame_path = path + "/" + f'{j:05d}' + ".png"
                if not os.path.exists(frame_path): ## This is for kinetics dataset
                    frame_path = path + "/" + f'img_{(j + 1):05d}' + ".jpg" 
                if not os.path.exists(frame_path): ## This is for kinetics dataset
                    frame_path = path + "/" + f'frame_{(j + 1):010d}' + ".jpg" 

                images.append(Image.open(frame_path))
            clips.append(images)
        return clips
    
    # def sample_clip(self, video_frames, annotation_frames, sampling_num):
    #     size = video_frames.shape[0]
    #     data = []
    #     annotations = []
    #     data_idx = []
    #     for i in range(self.num_clips):
    #         if self.sampling_mode == SamplingMode.UNIFORM:
    #                 idx = random.sample(range(0, size), sampling_num)
    #                 idx.sort()
    #                 data += [video_frames[idx]]
    #                 annotations += [annotation_frames[idx]]
    #                 data_idx.append(idx)
    #         elif self.sampling_mode == SamplingMode.DENSE:
    #                 base = random.randint(0, size - sampling_num)
    #                 idx = range(base, base + sampling_num)
    #                 data +=  [video_frames[idx]]
    #                 annotations += [annotation_frames[idx]]
    #                 data_idx.append(idx)
    #         elif self.sampling_mode == SamplingMode.Full:
    #                 data +=  [video_frames]
    #                 annotations += [annotation_frames]
    #                 data_idx.append(range(0, size))
    #     return data, annotations
    
    def generate_indices(self, size, sampling_num):
        indices = []
        for i in range(self.num_clips):
            if self.sampling_mode == SamplingMode.UNIFORM:
                    if size < sampling_num:
                        ## sample repeatly
                        idx = random.choices(range(0, size), k=sampling_num)
                    else:
                        idx = random.sample(range(0, size), sampling_num)
                    idx.sort()
                    indices.append(idx)
            elif self.sampling_mode == SamplingMode.DENSE:
                    base = random.randint(0, size - sampling_num)
                    idx = range(base, base + sampling_num)
                    indices.append(idx)
            elif self.sampling_mode == SamplingMode.Full:
                    indices.append(range(0, size))
            elif self.sampling_mode == SamplingMode.Regular:
                if size < sampling_num * self.regular_step:
                    step = size // sampling_num
                else:
                    step = self.regular_step
                base = random.randint(0, size - (sampling_num * step))
                idx = range(base, base + (sampling_num * step), step)
                indices.append(idx)
        return indices
    

    def read_batch(self, path, annotation_path=None, frame_transformation=None, target_transformation=None, video_transformation=None):
        size = self.train_dict_lenghts[path]
        # sampling_num = size if self.num_frames > size else self.num_frames
        clip_indices = self.generate_indices(self.train_dict_lenghts[path], self.num_frames)
        sampled_clips = self.read_clips(path, clip_indices)
        annotations = []
        sampled_clip_annotations = []
        if annotation_path is not None:
            sampled_clip_annotations = self.read_clips(annotation_path, clip_indices)
            if target_transformation is not None:
                for i in range(len(sampled_clip_annotations)):
                    sampled_clip_annotations[i] = target_transformation(sampled_clip_annotations[i])
        if frame_transformation is not None:
            for i in range(len(sampled_clips)):
                try:
                    sampled_clips[i] = frame_transformation(sampled_clips[i])
                except:
                    print("Error in frame transformation")
        if video_transformation is not None:
            for i in range(len(sampled_clips)):
                if len(sampled_clip_annotations) != 0:
                    sampled_clips[i], sampled_clip_annotations[i] = video_transformation(sampled_clips[i], sampled_clip_annotations[i])
                else:
                    sampled_clips[i] = video_transformation(sampled_clips[i])
        sampled_data = torch.stack(sampled_clips)
        if len(sampled_clip_annotations) != 0:
            sampled_annotations = torch.stack(sampled_clip_annotations)
            if sampled_annotations.size(0) != 0:
                sampled_annotations = (255 * sampled_annotations).type(torch.uint8) 
                if sampled_annotations.shape[2] == 1:
                    sampled_annotations = sampled_annotations.squeeze(2)
        else:
            sampled_annotations = torch.empty(0)
        ## squeezing the annotations to be in the shape of (num_sample, num_clips, num_frames, height, width)
        return sampled_data, sampled_annotations
    
    

    ## Similar function to read_batch except the numbers are in the range 0-255 and uint8
    # def read_annotations(self, path, frame_transformation=None, video_transformation=None, indices=None):
    #     files = sorted(glob.glob(path + "/" + "*.jpg"), key=numericalSort)
    #     if len(files) == 0:
    #         files = sorted(glob.glob(path + "/" + "*.png"), key=numericalSort)
    #     if (frame_transformation is not None):
    #         images = []
    #         for file in files:
    #             frame = Image.open(file).squeeze()
    #             images.append(frame)
    #         images = frame_transformation(images)
    #         images = torch.stack(images)
    #     else:
    #         images = torch.stack([torch.Tensor(Image.open(file)) for file in files])
    #     size = images.shape[0]
    #     sampling_num = size if self.num_frames > size else self.num_frames
    #     if self.sampling_mode == SamplingMode.UNIFORM:
    #         data = []
    #         if video_transformation is not None:
    #             for i in range(self.num_clips):
    #                 if indices is not None:
    #                     idx = indices
    #                 else:
    #                     idx = random.sample(range(0, size), sampling_num)
    #                     idx.sort()
    #                 data += [images[idx]]
    #         else:
    #             for i in range(self.num_clips):
    #                 if indices is not None:
    #                     idx = indices
    #                 else:
    #                     idx = random.sample(range(0, size), sampling_num)
    #                     idx.sort()
    #                 data += [images[idx]]
    #         data = torch.stack(data)
    #     elif self.sampling_mode == SamplingMode.DENSE:
    #         data = []
    #         for i in range(self.num_clips):
    #             base = random.randint(0, size - sampling_num)
    #             if indices is not None:
    #                 idx = indices
    #             else:
    #                 idx = range(base, base + sampling_num)
    #             if video_transformation is not None:
    #                 data +=  [video_transformation(images[idx])]
    #             else:
    #                 data +=  [images[idx]]
    #         data = torch.stack(data)    
    #     elif self.sampling_mode == SamplingMode.Full:
    #         data = []
    #         for i in range(self.num_clips):
    #             if video_transformation is not None:
    #                 data +=  [video_transformation(images)]
    #             else:
    #                 data +=  [images]
    #         data = torch.stack(data) 
    #     data = (255 * data).type(torch.uint8)  
    #     return data        


    def __getitem__(self, idx): ### (B, num_clips, num_frames, C, H, W) returns None if the annotation flag is off. Be careful when loading the data.
        # idx = 0  ## This is a hack to make the code work with the dataloader.
        # idx = random.randint(0, 5)
        video_path = self.keys[idx]
        dir_name = video_path.split("/")[-1]
        video_label = np.tile(self.train_dict[video_path], (self.num_clips, ))
        annotations = None
        annotations_path = None
        if (self.use_annotations):
            annotations_path = self.annotation_keys[idx]
            # annotations = self.read_annotations(annotations_path, self.target_transform, indices=indices)
        data, annotations = self.read_batch(video_path, annotations_path, self.frame_transform, self.target_transform ,self.video_transform)   
        if self.meta_dict is not None:
            category_dict = make_categories_dict(self.meta_dict, "davis")
            meta_dict = self.meta_dict["videos"][dir_name]["objects"]
            annotations = map_instances(annotations, meta_dict, category_dict)
        # else:
            # annotations = convert_to_indexed_RGB(annotations)



        
        # print(data.shape)
        # print(annotations.shape)
        return data, annotations, torch.Tensor(video_label)
    





class YVOSDataset(VideoDataset):

    def __init__(self, classes_directory, annotations_directory, sampling_mode, num_clips, num_frames, num_labels, frame_transform=None, target_transform=None, video_transform=None, meta_file_directory=None, regular_step=1) -> None:
        super().__init__(classes_directory, annotations_directory, sampling_mode, num_clips, num_frames, num_labels, frame_transform, target_transform, video_transform, meta_file_directory, regular_step=regular_step)
        
        self.category_dict = make_categories_dict(self.meta_dict, "ytvos")

    def __getitem__(self, idx): ### (B, num_clips, num_frames, C, H, W) returns None if the annotation flag is off. Be careful when loading the data.
        video_path = self.keys[idx]
        dir_name = video_path.split("/")[-1]
        video_label = np.tile(self.train_dict[video_path], (self.num_clips, ))
        annotations = None
        annotations_path = None
        if (self.use_annotations):
            annotations_path = self.annotation_keys[idx]
            # annotations = self.read_annotations(annotations_path, self.target_transform, indices=indices)
        data, annotations = self.read_batch(video_path, annotations_path, self.frame_transform, self.target_transform ,self.video_transform)   
        if self.meta_dict is not None:
            annotations = map_instances(annotations, self.meta_dict["videos"][dir_name]["objects"], self.category_dict)

        # else:
            # annotations = convert_to_indexed_RGB(annotations)
        return data, annotations, torch.Tensor(video_label)



class Kinetics(VideoDataset):

    def __init__(self, classes_directory, sampling_mode, num_clips, num_frames, num_labels, frame_transform=None, target_transform=None, video_transform=None, meta_file_directory=None, regular_step=1) -> None:
        super().__init__(classes_directory, "", sampling_mode, num_clips, num_frames, num_labels, frame_transform, target_transform, video_transform, meta_file_directory, regular_step=regular_step)

    def __getitem__(self, idx): ### (B, num_clips, num_frames, C, H, W) returns None if the annotation flag is off. Be careful when loading the data.
        video_path = self.keys[idx]
        dir_name = video_path.split("/")[-1]
        video_label = np.tile(self.train_dict[video_path], (self.num_clips, ))
        annotations = None
        annotations_path = None
        data, annotations = self.read_batch(video_path, None, self.frame_transform, self.target_transform ,self.video_transform)   
        if self.meta_dict is not None:
            annotations = map_instances(annotations, self.meta_dict["videos"][dir_name]["objects"], self.category_dict)

        # else:
            # annotations = convert_to_indexed_RGB(annotations)
        return data, annotations, torch.Tensor(video_label)





class pascalVOCLoader(data.Dataset):
    """Data loader for the Pascal VOC semantic segmentation dataset.
    Annotations from both the original VOC data (which consist of RGB images
    in which colours map to specific classes) and the SBD (Berkely) dataset
    (where annotations are stored as .mat files) are converted into a common
    `label_mask` format.  Under this format, each mask is an (M,N) array of
    integer values from 0 to 21, where 0 represents the background class.
    The label masks are stored in a new folder, called `pre_encoded`, which
    is added as a subdirectory of the `SegmentationClass` folder in the
    original Pascal VOC data layout.
    A total of five data splits are provided for working with the VOC data:
        train: The original VOC 2012 training data - 1464 images
        val: The original VOC 2012 validation data - 1449 images
        trainval: The combination of `train` and `val` - 2913 images 
        train_aug: The unique images present in both the train split and
                   training images from SBD: - 8829 images (the unique members
                   of the result of combining lists of length 1464 and 8498)
        train_aug_val: The original VOC 2012 validation data minus the images
                   present in `train_aug` (This is done with the same logic as
                   the validation set used in FCN PAMI paper, but with VOC 2012
                   rather than VOC 2011) - 904 images
    """

    def __init__(
        self,
        root,
        sbd_path=None,
        split="train_aug",
        is_transform=False,
        img_size=512,
        augmentations=None,
        img_norm=True,
        test_mode=False,
        val=False,
    ):
        self.root = root
        self.sbd_path = sbd_path
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.test_mode = test_mode
        self.n_classes = 21
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = collections.defaultdict(list)
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)

        if not self.test_mode:
            for split in ["train", "val", "trainval"]:
                path = pjoin(self.root, "ImageSets/Segmentation", split + ".txt")
                file_list = tuple(open(path, "r"))
                file_list = [id_.rstrip() for id_ in file_list]
                self.files[split] = file_list
            self.setup_annotations()

        if val:
            self.tf = transforms.Compose(
                [
                    transforms.ToTensor(),
                    # transforms.RandomHorizontalFlip(),
                    # transforms.RandomResizedCrop(size=self.img_size),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.228, 0.224, 0.225])
                ]
            )
        else:
            self.tf = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomResizedCrop(size=self.img_size, scale=(0.8, 1.)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.228, 0.224, 0.225])
                ]
            )

    def __len__(self):
        return len(self.files[self.split])

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        im_name = self.files[self.split][index]
        im_path = pjoin(self.root, "JPEGImages", im_name + ".jpg")
        lbl_path = pjoin(self.root, "SegmentationClass/pre_encoded", im_name + ".png")
        im = Image.open(im_path)
        lbl = Image.open(lbl_path)
        if self.augmentations is not None:
            im, lbl = self.augmentations(im, lbl)
        if self.is_transform:
            im, lbl = self.transform(im, lbl)
        return im, lbl

    def transform(self, img, lbl):
        if self.img_size == ("same", "same"):
            pass
        else:
            img = img.resize((self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
            lbl = lbl.resize((self.img_size[0], self.img_size[1]), Image.NEAREST)
        img = self.tf(img)
        lbl = torch.from_numpy(np.array(lbl)).long()
        lbl[lbl == 255] = 0
        return img, lbl

    def get_pascal_labels(self):
        """Load the mapping that associates pascal classes with label colors
        Returns:
            np.ndarray with dimensions (21, 3)
        """
        return np.asarray(
            [
                [0, 0, 0],
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128],
            ]
        )

    def encode_segmap(self, mask):
        """Encode segmentation label images as pascal classes
        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.
        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_pascal_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask

    def decode_segmap(self, label_mask, plot=False):
        """Decode segmentation class labels into a color image
        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
              in a figure.
        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        label_colours = self.get_pascal_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

    def setup_annotations(self):
        """Sets up Berkley annotations by adding image indices to the
        `train_aug` split and pre-encode all segmentation labels into the
        common label_mask format (if this has not already been done). This
        function also defines the `train_aug` and `train_aug_val` data splits
        according to the description in the class docstring
        """
        sbd_path = self.sbd_path
        target_path = pjoin(self.root, "SegmentationClass/pre_encoded")
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        path = pjoin(sbd_path, "dataset/train.txt")
        sbd_train_list = tuple(open(path, "r"))
        sbd_train_list = [id_.rstrip() for id_ in sbd_train_list]
        train_aug = self.files["train"] + sbd_train_list

        # keep unique elements (stable)
        train_aug = [train_aug[i] for i in sorted(np.unique(train_aug, return_index=True)[1])]
        self.files["train_aug"] = train_aug
        set_diff = set(self.files["val"]) - set(train_aug)  # remove overlap
        self.files["train_aug_val"] = list(set_diff)

        pre_encoded = glob.glob(pjoin(target_path, "*.png"))
        expected = np.unique(self.files["train_aug"] + self.files["val"]).size

        if len(pre_encoded) != expected:
            print("Pre-encoding segmentation masks...")
            for ii in tqdm(sbd_train_list):
                lbl_path = pjoin(sbd_path, "dataset/cls", ii + ".mat")
                data = io.loadmat(lbl_path)
                lbl = data["GTcls"][0]["Segmentation"][0].astype(np.int32)
                lbl = toimage(lbl, high=lbl.max(), low=lbl.min())
                imsave(pjoin(target_path, ii + ".png"), lbl)

            for ii in tqdm(self.files["trainval"]):
                fname = ii + ".png"
                lbl_path = pjoin(self.root, "SegmentationClass", fname)
                lbl = self.encode_segmap(imread(lbl_path))
                lbl = toimage(lbl, high=lbl.max(), low=lbl.min())
                imsave(pjoin(target_path, fname), lbl)

        assert expected == 9733, "unexpected dataset sizes"




def make_loader(dataset_name, num_clip_frames, batch_size, regular_step=1, sampling_mode=SamplingMode.UNIFORM, frame_transform=None, target_transform=None, video_transform=None, shuffle=False, num_workers=6, pin_memory=True, world_size=1, rank=0):
    ## 
    # dataset = VideoDataset("../../../SOTA_Nips2021/dense-ulearn-vos/data/train_all_frames/", SamplingMode.UNIFORM, 1, 8, 1, trns)
    if "val" in dataset_name:
        ## get the part before underline
        name = dataset_name.split("_")[0]
    else:
        name = dataset_name
    prefix = dataset_path[platform][name]
    if dataset_name == "davis":
        data_path = os.path.join(prefix, "davis_2021/davis_data/JPEGImages/")
        annotation_path = os.path.join(prefix, "davis_2021/DAVIS/Annotations/")
        train_dataset = VideoDataset(data_path, annotation_path,  sampling_mode, 1, num_clip_frames, 1, frame_transform, target_transform, video_transform, regular_step=regular_step)
        # train_dataset = VideoDataset("../datasets/davis_2020/davis_data/JPEGImages/", "../datasets/davis_2020/DAVIS/Annotations/",  SamplingMode.UNIFORM, 1, num_frames, 1, trns, target_trns)
    elif dataset_name == "davis_val":
        data_path = os.path.join(prefix, "davis_2021/davis_data/val/")
        annotation_path = os.path.join(prefix, "davis_2021/DAVIS/val_annotation/")
        train_dataset = VideoDataset(data_path, annotation_path,  sampling_mode, 1, num_clip_frames, 1, frame_transform, target_transform, video_transform, regular_step=regular_step)
    elif dataset_name == "visor":
        data_path = os.path.join(prefix, "JPEGImages/")
        annotation_path = os.path.join(prefix, "Annotations/")
        train_dataset = VideoDataset(data_path, annotation_path,  sampling_mode, 1, num_clip_frames, 1, frame_transform, target_transform, video_transform, regular_step=regular_step)
    elif dataset_name == "visor_val":
        data_path = os.path.join(prefix, "davis_2021/davis_data/val/")
        annotation_path = os.path.join(prefix, "davis_2021/DAVIS/val_annotation/")
        train_dataset = VideoDataset(data_path, annotation_path,  sampling_mode, 1, num_clip_frames, 1, frame_transform, target_transform, video_transform, regular_step=regular_step)
    elif dataset_name == "ytvos":
        data_path = os.path.join(prefix, "train1/JPEGImages/")
        annotation_path = os.path.join(prefix, "train1/Annotations/")
        meta_file_path = os.path.join(prefix, "train1/meta.json")
        train_dataset = YVOSDataset(data_path, annotation_path,  sampling_mode, 1, num_clip_frames, 1, frame_transform, target_transform, video_transform, meta_file_directory=meta_file_path, regular_step=regular_step)
        # train_dataset = build_dataset('train', dataset_path, num_frames)

    elif dataset_name == "mose":
        data_path = os.path.join(prefix, "train/JPEGImages/")
        annotation_path = os.path.join(prefix, "train/Annotations/")
        train_dataset = VideoDataset(data_path, annotation_path,  sampling_mode, 1, num_clip_frames, 1, frame_transform, target_transform, video_transform, regular_step=regular_step)
        
    elif dataset_name == "ytvos_val":
        data_path = os.path.join(prefix, "val1/JPEGImages/")
        annotation_path = os.path.join(prefix, "val1/Annotations/")
        meta_file_path = os.path.join(prefix, "val1/meta.json")
        # train_dataset = build_dataset('train', dataset_path, num_frames)
        train_dataset = YVOSDataset(data_path, annotation_path,  sampling_mode, 1, num_clip_frames, 1, frame_transform, target_transform, video_transform, meta_file_directory=meta_file_path, regular_step=regular_step)
        # train_dataset = YVOSDataset("../../data/train/JPEGImages/", "../../data/train/Annotations/",  SamplingMode.UNIFORM, 1, num_frames, 1, trns, target_trns, meta_file_directory="../../data/train/meta.json")
    # elif dataset_name == "sand_box":
    #     train_dataset = VideoDataset("../../data/sand_box/JPEGImages/", "../../data/sand_box/Annotations/",  SamplingMode.DENSE, 1, num_clip_frames, 1, frame_transform, target_transform, video_transform)
    #     # train_dataset = VideoDataset("../../data/sand_box/JPEGImages/", "../../data/sand_box/Annotations/",  SamplingMode.DENSE, 1, num_frames, 1, trns)
    # elif dataset_name == "pascal":
    #     data_path = os.path.join(prefix, "pascal_voc/VOCdevkit/VOC2012/")
    #     data_path1 = os.path.join(prefix, "sbd/benchmark_RELEASE/")
    #     train_dataset =  pascalVOCLoader(data_path, is_transform=True, img_size=224, augmentations=None, sbd_path=data_path1, split='val', val=True)
    elif dataset_name == "kinetics":
        data_path = os.path.join(prefix, "kinetics/")
        train_dataset = Kinetics(data_path,  sampling_mode, 1, num_clip_frames, 1, frame_transform, target_transform, video_transform, regular_step=regular_step)
    elif dataset_name == "epic-kitchen":
        data_path = os.path.join(prefix, "train/480p/")
        train_dataset = VideoDataset(data_path, "",  sampling_mode, 1, num_clip_frames, 1, frame_transform, target_transform, video_transform, regular_step=regular_step)
    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        train_loader = DataLoader(train_dataset, batch_size, shuffle=False, num_workers=num_workers, sampler=train_sampler, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader


def pascal_loader(split, batch_size, frame_transform=None, target_transform=None, video_transform=None, shuffle=False, num_workers=10, pin_memory=True, world_size=1, rank=0):
    prefix = dataset_path[platform]["pascal"]
    data_path = os.path.join(prefix, "pascal_voc/VOCdevkit/VOC2012/")
    data_path1 = os.path.join(prefix, "sbd/benchmark_RELEASE/")
    if split == "train_aug":
        train_dataset =  pascalVOCLoader(data_path, is_transform=True, img_size=448, augmentations=None, sbd_path=data_path1, split=split, val=False)
    else:
        train_dataset =  pascalVOCLoader(data_path, is_transform=True, img_size=448, augmentations=None, sbd_path=data_path1, split=split, val=True)
    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, sampler=train_sampler, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader

    



def split_directory_to_train_val(train_dir, middle_path, annotaion_path):

    source_dir = os.path.join(train_dir, middle_path)
    annotation_source_dir = os.path.join(train_dir, annotaion_path)
    ## read train directory folders 
    train_folders = os.listdir(source_dir)
    ## shuffle the train and annotation folders
    random.shuffle(train_folders)
    ## split the folders to train and val
    train_folders, val_folders = train_folders[:int(len(train_folders)*0.8)], train_folders[int(len(train_folders)*0.8):]
    ## create train and val directories
    os.makedirs(os.path.join("train1", middle_path), exist_ok=True)
    os.makedirs(os.path.join("val1", middle_path), exist_ok=True)
    os.makedirs(os.path.join("train1", annotaion_path), exist_ok=True)
    os.makedirs(os.path.join("val1", annotaion_path), exist_ok=True)
    ## move the folders to train and val directories
    for folder in train_folders:
        ## copy folder to train directory by making a new folder
        shutil.copytree(os.path.join(source_dir, folder), os.path.join("train1", middle_path, folder))
        ## copy annotation folder to train directory by making a new folder
        shutil.copytree(os.path.join(annotation_source_dir, folder), os.path.join("train1", annotaion_path, folder))
    for folder in val_folders:
        ## copy folder to val directory
        shutil.copytree(os.path.join(source_dir, folder), os.path.join("val1", middle_path, folder))
        ## copy annotation folder to val directory
        shutil.copytree(os.path.join(annotation_source_dir, folder), os.path.join("val1", annotaion_path, folder))
    ## remove the original directory
    # os.rmdir(train_dir)
    ## rename the train and val directories

def make_all_the_images_zero_index(train_dir, middle_path):
    source_dir = os.path.join(train_dir, middle_path)
    ## read train directory folders 
    train_folders = os.listdir(source_dir)
    for folder in train_folders:
    ## read all the images in the folder in rename them to zero index
        images = sorted(os.listdir(os.path.join(source_dir, folder)))
        for i, image in enumerate(images):
            os.rename(os.path.join(source_dir, folder, image), os.path.join(source_dir, folder, str(i).zfill(5)+".jpg"))




if __name__ == "__main__":
    rand_color_jitter = video_transformations.RandomApply([video_transformations.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)], p=0.8)
    data_transform_list = [rand_color_jitter, video_transformations.RandomGrayscale(), video_transformations.RandomGaussianBlur()]
    data_transform = video_transformations.Compose(data_transform_list)
    video_transform_list = [video_transformations.Resize(224, 'bilinear'), video_transformations.RandomResizedCrop((224, 224)), video_transformations.ClipToTensor(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225])]
    video_transform = video_transformations.Compose(video_transform_list)
    num_frames = 8
    batch_size = 8
    num_workers = 4
    data_loader = make_loader("visor", num_frames, batch_size, SamplingMode.UNIFORM, frame_transform=data_transform, target_transform=None, video_transform=video_transform, shuffle=False, num_workers=num_workers, pin_memory=True)
    print(len(data_loader))

    logging_directory = "data_loader_log/"

    if os.path.exists(logging_directory):
        os.system(f'rm -r {logging_directory}')
    os.makedirs(logging_directory)

    for i, train_data in enumerate(data_loader):
        datum, annotations, label = train_data
        print("===========================")
        print("")
        annotations = annotations.squeeze(1)
        datum = datum.squeeze(1)
        print((torch.unique(annotations)))
        print(datum.shape)
        print(annotations.shape)
        # visualize_sampled_videos(datum, "data_loader_log/", f"test_{i}.avi")
        # visualize_sampled_videos(annotations, "data_loader_log/", f"test_anotations_{i}.avi")
        make_seg_maps(datum, annotations, logging_directory, f"test_seg_maps_{i}.avi")






