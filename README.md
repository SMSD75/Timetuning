# Time Does Tell a Lot: Self-Supervised Learning of Dense Image Representations


# Dataset Structure
The structure of your dataset should follow the structure of DAVIS (Densely Annotated VIdeo Segmentation) 2017 Unsupervised dataset.

# DAVIS Structure Example
The DAVIS dataset is organized as follows:

```
DAVIS
├── JPEGImages
│   └── 480p
│       └── object1
│           ├── 00000.jpg
│           ├── 00001.jpg
│           ├── 00002.jpg
│           └── ...
│       └── object2
│           ├── 00000.jpg
│           ├── 00001.jpg
│           ├── 00002.jpg
│           └── ...
├── Annotations
│   └── 480p
│       └── object1
│           ├── 00000.png
│           ├── 00001.png
│           ├── 00002.png
│           └── ...
│       └── object2
│           ├── 00000.png
│           ├── 00001.png
│           ├── 00002.png
│           └── ...
└── ImageSets
    └── 2017
        ├── train.txt
        ├── val.txt
        └── test-dev.txt
```

Also, the same for YTVOS:

```

YouTubeVOS
├── train
│ ├── Annotations
│ ├── JPEGImages
│ └── meta.json
└── valid
├── Annotations
├── JPEGImages
└── meta.json

```


Please ensure your dataset adheres to this structure for compatibility.

# Adjusting Paths in the Data Loader

You can adjust the paths of your dataset as has been set in the function `make_loader` in `data_loader.py`. To ensure accurate loading, this function should be set to point directly to the root of your dataset, specifically to the `JPEGImages` and `Annotations` for images and labels, respectively. If this path is incorrect, the dataloader will not be able to find and load your dataset correctly.

The correct configuration of this path is crucial. If the directory `JPEGImages` contains videos rather than images, the dataloader will automatically react. It generates distinct subdirectories, each named after a video, and fills these with the corresponding video frames. Moreover, the dataloader determines whether the given path is for an unconverted video dataset or a correctly converted image dataset. It does so by counting the depth of the directories it needs to traverse to reach video or image files. This counting helps it discern the type of data to handle.

Please ensure that the paths you provide are correct and accurately point to your dataset's location.

# Evaluation

1 - To replicate the results presented in the paper, you'll need to use `evaluation.py`, `linear_finetune.py`, and `cluster_based_foreground_extraction.py`.

2 - For per-dataset evaluations on Pascal VOC, certain lines should be uncommented in `evaluation.py`. These lines have been clearly indicated within the code.

3 - For video datasets, pass the `dataset_name + _val` argument to the `make_loader` function in `data_loader.py`. This will allow you to load the validation set.

Please follow these instructions to accurately reproduce our findings.


