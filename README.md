# Ensemble One-Shot NAS 

## Introduction
While large amounts of data are available everywhere, most groups lack experts to guide the development of machine learning systems. Automated machine learning (AutoML) provides methods and processes to make advanced machine learning techniques available for non-machine learning experts. In this hackathon, we present an introduction to AutoML using the recently developed AutoGluon toolkit. AutoGluon is an AutoML tookit built on top of MXNet Gluon that enables easy-to-use and easy-to-extend AutoML, with a focus on deep learning and deploying models in real-world applications. AutoGluon supports core AutoML functionality such as: hyperparameter optimization algorithms and early stopping training mechanisms. With just a single call to AutoGluon’s fit() function, AutoGluon will automatically train many models under thousands of different hyperparameter configurations that affect the training process, and then return the best model within a reasonable runtime. In addition, you can easily exert greater control over the training process, for example to provide hard time limits for training, and what computational resources each training run should leverage. 

In this repository, we will focus on the task of image classification to demonstrate the usage of AutoGluon’s main APIs. We will show you how to easily train models on your own image dataset, and how to control the hyperparameter-tuning process to obtain competitive results on a Kaggle image classification competition as well as achieve state-of-the-art performance on CIFAR10 (one of the most popular benchmark image classification datasets).

This repository provides the implementation of 
+ [AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data](https://arxiv.org/abs/2003.06505).
+ [Single Path One-Shot Neural Architecture Search with Uniform Sampling](https://arxiv.org/abs/1904.00420).
+ [On Network Design Spaces for Visual Recognition](https://arxiv.org/abs/1905.13214).

##  Install
```bash
pip install mxnet autogluon
```

## Our Trained Model / Checkpoint

+ OneDrive: [Link](https://1drv.ms/u/s!Am_mmG2-KsrnajesvSdfsq_cN48?e=aH)

### Supernet

Our trained Supernet weight is in `$Link/Supernet/checkpoint-XX.pth.tar`, which can be used by Search.

### Search

Our search result is in `$Link/Search/checkpoint.pth.tar`, which can be used by Evaluation.

### Evaluation

Out searched models have been trained from scratch, is can be found in `$Link/Evaluation/$ARCHITECTURE`.

Here is a summary:

+ Cifar10:

|    Architecture         |  FLOPs    |   #Params |   Top-1 Acc   |   Top-5 Acc   |
|:------------------------|:---------:|:---------:|:---------:|:---------:|
(2, 1, 0, 1, 2)        |   9M     |	1.4M    |      96.4    |       99.8   |

+ Imagenet:

|    Architecture         |  FLOPs    |   #Params |   Top-1   |   Top-5   |
|:------------------------|:---------:|:---------:|:---------:|:---------:|
(2, 1, 0, 1, 2, 0, 2, 0, 2, 0, 2, 3, 0, 0, 0, 0, 3, 2, 3, 3)        |   323M     |	3.5M    |      25.6    |       8.0   |

## Usage

### 1. Setup Dataset and Flops Table

Download the ImageNet Dataset and move validation images to labeled subfolders. To do this, you can use the following script: [https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

Download the flops table to accelerate Flops calculation which is required in Uniform Sampling. It can be found in `$Link/op_flops_dict.pkl`.

We recommend to create a folder `data` and use it in both Supernet training and Evaluation training.

Here is a example structure of `data`:

```
data
|--- train                 ImageNet Training Dataset
|--- val                   ImageNet Validation Dataset
|--- op_flops_dict.pkl     Flops Table
```

### 2. Train Supernet

Train supernet with the following command:

```bash
cd src/Supernet
python3 train.py --train-dir $YOUR_TRAINDATASET_PATH --val-dir $YOUR_VALDATASET_PATH
```

### 3. Search in Supernet with Evolutionary Algorithm

Search in supernet with the following command:

```bash
cd src/Search
python3 search.py
```

It will use ```../Supernet/checkpoint-latest.pth.tar``` as Supernet's weight, please make sure it exists or modify the path manually.

### 4. Get Searched Architecture

Get searched architecture with the following command:

```bash
cd src/Evaluation
python3 eval.py
```

It will generate folder in ``data/$YOUR_ARCHITECTURE``. You can train the searched architecture from scratch in the folder.

### 5. Train from Scratch

Finally, train and evaluate the searched architecture with the following command.

Train:

```bash
cd src/Evaluation/data/$YOUR_ARCHITECTURE
python3 train.py --train-dir $YOUR_TRAINDATASET_PATH --val-dir $YOUR_VALDATASET_PATH
```

Evaluate:

```bash
cd src/Evaluation/data/$YOUR_ARCHITECTURE
python3 train.py --eval --eval-resume $YOUR_WEIGHT_PATH --train-dir $YOUR_TRAINDATASET_PATH --val-dir $YOUR_VALDATASET_PATH
```