# Time Does Tell a Lot: Self-Supervised Learning of Dense Image Representations


Official PyTorch implementation and pretrained models for ***TimeT***. For details, see [arXiv](). Our pretrained model can be found [here](https://www.dropbox.com/scl/fi/nnx2mm8ian9w49vstpgz0/TimeT.pth?rlkey=w9q3hvxd51nb63ammy33qhry0&dl=0). Optimizing with our model, ***TimeT***, does not necessitate a significant GPU budget. Our training process is conducted on a single NVIDIA GeForce RTX 3090, making ***TimeT*** an effortless solution to enhance your Vision Transformer's (ViT) spatial awareness. The only prerequisite is to create videos featuring the objects that you aim to segment in your subsequent tasks. After that, simply fine-tune a pre-trained ViT using our proprietary temporal loss. 


![Logo](Images/Fig1.jpg)

# Datasets

In the following sections, we provide a comprehensive guide that outlines the specific structures that your datasets should emulate, complete with illustrative examples. Adhering to these guidelines will ensure your dataset is appropriately formatted, thereby preventing potential complications during the training phase of our model.

For datasets that don't naturally conform to this structure, such as VISOR, we've accommodated this by providing a useful code snippet to aid the conversion process. More detailed information can be found by referring to the link below:

[Dataset Structures](dataset_README.md)

# Adjusting Paths in the Data Loader

You can adjust the paths of your dataset as has been set in the function `make_loader` in `data_loader.py`. To ensure accurate loading, this function should be set to point directly to the root of your dataset, specifically to the `JPEGImages` and `Annotations` for images and labels, respectively. If this path is incorrect, the dataloader will not be able to find and load your dataset correctly.

The correct configuration of this path is crucial. If the directory `JPEGImages` contains videos rather than images, the dataloader will automatically react. It generates distinct subdirectories, each named after a video, and fills these with the corresponding video frames. Moreover, the dataloader determines whether the given path is for an unconverted video dataset or a correctly converted image dataset. It does so by counting the depth of the directories it needs to traverse to reach video or image files. This counting helps it discern the type of data to handle.

Please ensure that the paths you provide are correct and accurately point to your dataset's location.


# Evaluation

1 - To replicate the results presented in the paper, you'll need to use `evaluation.py`, `linear_finetune.py`, and `cluster_based_foreground_extraction.py`.

2 - For per-dataset evaluations on Pascal VOC, certain lines should be uncommented in `evaluation.py`. These lines have been clearly indicated within the code.

3 - For video datasets, pass the `dataset_name + _val` argument to the `make_loader` function in `data_loader.py`. This will allow you to load the validation set.

For accurate replication of our results, please adhere to the following instructions. We've set all arguments to their default values for your convenience. However, if you wish to alter the number of inference clusters, such as in clustering or overclustering experiments, you may utilize the command detailed below : 

```python
python evaluation.py --num_clusters 21
```

For overclustering experiment ```many_to_one``` and ```precision_based``` should be set to **True**. 

# Training

To start training from scratch, execute `time_tuning.py`. By default, the argument values are set for single GPU training without the utilization of an Exponential Moving Average (EMA) teacher, and no queue is used. However, activating these features has been observed to yield a slight performance enhancement on certain datasets, like [MOSE](https://henghuiding.github.io/MOSE/). The validation performance is logged every four epochs, while the loss is recorded with each iteration.

The training starts by running the following command :

```python
python time_tuning.py
```

To modify various training parameters such as the number of training prototypes, whether to add or remove the queue or EMA teacher, the presence of a projection head, or the number of clip frames, you can directly add the relevant arguments to your execution command :

```python
python time_tuning.py --num_clusters 200 --use_queue False --use_teacher True --use_projection_head True --num_frames 4
```

# Visualizations

For more visualizations please download the visualizations folder.


|    |    |    |
|:--:|:--:|:--:|
| ![](Images/0_0.gif) | ![](Images/0_2.gif)  | ![](Images/0_7.gif) |
| ![](Images/0_9.gif) | ![](Images/0_20.gif) | ![](Images/0_48.gif) |
| ![](Images/1_23.gif)  | ![](Images/2_1.gif) | ![](Images/2_3.gif) |


# Citation

If you find this repository useful, please consider giving a star ⭐ and citation 📣:
