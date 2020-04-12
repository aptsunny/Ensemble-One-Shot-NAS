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
autogluon 账号的安装有问题

pytorch_p36
pip install mxnet autogluon
pip._vendor.pkg_resources.ContextualVersionConflict: (scikit-learn 0.20.3 (/home/ubuntu/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages), Requirement.parse('scikit-learn==0.21.2'), {'autogluon'})


amazonei_python_36
pip3 install torch torchvision

No module named 'torch'


另一台机器pytorch_p36
pip install mxnet autogluon
pip install git+https://github.com/ildoonet/pytorch-randaugment

```

## Our Trained Model / Checkpoint

+ OneDrive: [Link](https://1drv.ms/u/s!Am_mmG2-KsrnajesvSdfsq_cN48?e=aH)

### Supernet

Our trained Supernet weight is in `$Link/Supernet/checkpoint-XX.pth.tar`, which can be used by Search.

```python
# Select A Best Supernet
    @ag.args(
        signal='Experiments',

        # Training mode
        eval=False,
        auto_continue=False, # False if num_trials > 1
        flops_restriction=False, # spos
        activations_count_restriction=False, # pycls
        RandA=False, # RA(RandAugment)

        # Supernet HPO, Some tricks work for specific hyperparameters

        # network_shuffle: 4 choice for 5 blocks (Shuffle3x3\Shuffle5x5\Shuffle7x7\Xception)
        # network_mobile: 1/2/3 choice for 12 layers {'conv': [1, 2], 'rate': 1},
        choice=4,
        block=12, # shuffle block 5, mobile 12 layers
        sample_choice=2, # shuffle ,1/2/3/4,
        # sample_choice=ag.space.Int(1, 2, 3), # mobile
        total_iters=39000,  # 39000,1950,19500, cifar10: one epoch 195 * 256 = 49920
        batch_size= 288,# shuffle 256, mobile 288

        fake=ag.space.Real(0.4, 0.8, log=True),
        # learning_rate=0.3, # test_lr_range

        # blockwisely lr_scheduler
        # lr & wd
        # different_hpo=False,
        different_hpo=True,
        # learning_rate=ag.space.Real(0.2, 0.3, log=True), # glboal hpo, mobile
        # learning_rate=ag.space.Real(0.4, 0.8, log=True), # glboal hpo, shuffle
        learning_rate=0.2, # without hpo
        # wd=ag.space.Real(1e-4, 5e-4, log=True),

        # randaug
        # randaug_n=ag.space.Int(3, 4, 5),
        # randaug_m=ag.space.Int(5, 10, 15),
        # choice=ag.space.Int(0, 1),

        # default parameters
        randaug_n=3,
        randaug_m=5,
        weight_decay=4e-5,
        momentum=0.9,
        save='./models',
        label_smooth=0.1,
        save_interval=5,
    )
    def ag_train_cifar(args, reporter):
        return pipeline(args, reporter)

    myscheduler = ag.scheduler.FIFOScheduler(ag_train_cifar,
                                             # resource={'num_cpus': 4, 'num_gpus': 1},
                                             num_trials=4,
                                             time_attr='epoch',
                                             reward_attr="val_acc")
    # print(myscheduler)
    myscheduler.run()
    myscheduler.join_jobs()
    myscheduler.get_training_curves(filename='Supernet_curves', plot=True, use_legend=False)
    print('The Best Supernet Configuration and Accuracy are: {}, {}'.format(myscheduler.get_best_config(),
                                                                   myscheduler.get_best_reward()))
```


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