# Dataset Structure
The structure of your dataset should follow the structure of [DAVIS](https://davischallenge.org/) (Densely Annotated VIdeo Segmentation) 2017 Unsupervised dataset.

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

Also, the same for [YTVOS](https://youtube-vos.org/dataset/vos/):

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

For [Pascal VOC](https://www.dropbox.com/s/6gd4x0i9ewasymb/voc_data.zip?dl=0) in the evaluation time : 

```
dataset root.
└───SegmentationClass
│   │   *.png
│   │   ...
└───SegmentationClassAug # contains segmentation masks from trainaug extension 
│   │   *.png
│   │   ...
└───images
│   │   *.jpg
│   │   ...
└───sets
│   │   train.txt
│   │   trainaug.txt
│   │   val.txt

```

Please ensure your dataset adheres to this structure for compatibility. For datasets that deviate from the standard structure, such as [VISOR](https://epic-kitchens.github.io/VISOR/), we've included a snippet of code to manage the necessary conversion.