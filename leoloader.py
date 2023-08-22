"""

MIT License

Copyright (c) 2022 Adrian Ziegler

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import os
import pytorch_lightning as pl
from torchvision.datasets.vision import StandardTransform
from typing import Optional, Callable
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from typing import Tuple, Any
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms.functional as F
import random

class VOCDataModule(pl.LightningDataModule):

    CLASS_IDX_TO_NAME = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                         'train', 'tvmonitor']

    def __init__(self,
                 data_dir: str,
                 train_split: str,
                 val_split: str,
                 train_image_transform: Optional[Callable],
                 val_image_transform: Optional[Callable],
                 val_target_transform: Optional[Callable],
                 batch_size: int,
                 num_workers: int,
                 shuffle: bool = True,
                 return_masks: bool = False,
                 drop_last: bool = True):
        """
        Data module for PVOC data. "trainaug" and "train" are valid train_splits.
        If return_masks is set train_image_transform should be callable with imgs and masks or None.
        """
        super().__init__()
        self.root = os.path.join(data_dir, "VOCSegmentation")
        self.train_split = train_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_image_transform = train_image_transform
        self.val_image_transform = val_image_transform
        self.val_target_transform = val_target_transform
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.return_masks = return_masks

        # Set up datasets in __init__ as we need to know the number of samples to init cosine lr schedules
        assert train_split == "trainaug" or train_split == "train"
        self.voc_train = VOCDataset(root=self.root, image_set=train_split, transforms=self.train_image_transform,
                                    return_masks=self.return_masks)
        self.voc_val = VOCDataset(root=self.root, image_set=val_split, transform=self.val_image_transform,
                                  target_transform=self.val_target_transform)

    def __len__(self):
        return len(self.voc_train)

    def class_id_to_name(self, i: int):
        return self.CLASS_IDX_TO_NAME[i]

    def setup(self, stage: Optional[str] = None):
        print(f"Train size {len(self.voc_train)}")
        print(f"Val size {len(self.voc_val)}")

    def train_dataloader(self):
        return DataLoader(self.voc_train, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.num_workers,
                          drop_last=self.drop_last, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.voc_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          drop_last=self.drop_last, pin_memory=True)


# class TrainXVOCValDataModule(pl.LightningDataModule):
    # # wrapper class to allow for training on a different data set

    # def __init__(self, train_datamodule: pl.LightningDataModule, val_datamodule: VOCDataModule):
    #     super().__init__()
    #     self.train_datamodule = train_datamodule
    #     self.val_datamodule = val_datamodule

    # def setup(self, stage: str = None):
    #     self.train_datamodule.setup(stage)
    #     self.val_datamodule.setup(stage)

    # def class_id_to_name(self, i: int):
    #     return self.val_datamodule.class_id_to_name(i)

    # def __len__(self):
    #     return len(self.train_datamodule)

    # def train_dataloader(self):
    #     return self.train_datamodule.train_dataloader()

    # def val_dataloader(self):
    #     return self.val_datamodule.val_dataloader()



class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            img = F.hflip(img)
            target = F.hflip(target)
        return img, target


class RandomResizedCrop(object):
    def __init__(self, size, scale, ratio=(3. / 4., 4. / 3.)):
        self.rrc_transform = T.RandomResizedCrop(size=size, scale=scale, ratio=ratio)

    def __call__(self, img, target=None):
        y1, x1, h, w = self.rrc_transform.get_params(img, self.rrc_transform.scale, self.rrc_transform.ratio)
        img = F.resized_crop(img, y1, x1, h, w, self.rrc_transform.size, F.InterpolationMode.BILINEAR)
        target = F.resized_crop(target, y1, x1, h, w, self.rrc_transform.size, F.InterpolationMode.NEAREST)
        return img, target


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), F.to_tensor(target)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

class VOCDataset(VisionDataset):

    def __init__(
            self,
            root: str,
            image_set: str = "trainaug",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
            return_masks: bool = False
    ):
        super(VOCDataset, self).__init__(root, transforms, transform, target_transform)
        self.image_set = image_set
        if self.image_set == "trainaug" or self.image_set == "train":
            seg_folder = "SegmentationClassAug"
        elif self.image_set == "val":
            seg_folder = "SegmentationClass"
        else:
            raise ValueError(f"No support for image set {self.image_set}")
        seg_dir = os.path.join(root, seg_folder)
        image_dir = os.path.join(root, 'images')
        if not os.path.isdir(seg_dir) or not os.path.isdir(image_dir) or not os.path.isdir(root):
            raise RuntimeError('Dataset not found or corrupted.')
        splits_dir = os.path.join(root, 'sets')
        split_f = os.path.join(splits_dir, self.image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(seg_dir, x + ".png") for x in file_names]
        self.return_masks = return_masks

        assert all([Path(f).is_file() for f in self.masks]) and all([Path(f).is_file() for f in self.images])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = Image.open(self.images[index]).convert('RGB')
        if self.image_set == "val":
            mask = Image.open(self.masks[index])
            if self.transforms:
                img, mask = self.transforms(img, mask)
            return img, mask
        elif "train" in self.image_set:
            if self.transforms:
                if self.return_masks:
                    mask = Image.open(self.masks[index])
                    res = self.transforms(img, mask)
                else:
                    res = self.transforms(img)
                return res
            return img

    def __len__(self) -> int:
        return len(self.images)


def pascal_loader(batch_szie, root, split, val_size, train_size=448):
    train_transforms = Compose([
        RandomResizedCrop(size=train_size, scale=(0.8, 1.)),
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_image_transforms = T.Compose([T.Resize((train_size, train_size)),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_target_transforms = T.Compose([T.Resize((val_size, val_size), interpolation=InterpolationMode.NEAREST),
                                       T.ToTensor()])
    
    if split == 'trainaug':
        dataset = VOCDataset(root=root, image_set=split, transforms=StandardTransform(val_image_transforms, val_target_transforms),
                                return_masks=True)
    else:
        dataset = VOCDataset(root=root, image_set=split, transform=val_image_transforms,
                                target_transform=val_target_transforms)
    
    shuffle = True if split == "trainaug" else False
    shuffle = False
    return DataLoader(dataset, batch_size=batch_szie, shuffle=shuffle, num_workers=3, pin_memory=True)
