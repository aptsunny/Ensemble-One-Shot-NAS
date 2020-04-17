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
pip install git+https://github.com/ildoonet/pytorch-randaugment
pip install --upgrade git+https://github.com/sovrasov/flops-counter.pytorch.git
```

## Our Trained Model / Checkpoint

+ OneDrive: [Link](https://1drv.ms/u/s!Am_mmG2-KsrnajesvSdfsq_cN48?e=aH)

### Supernet

Our trained Supernet weight is in `$Link/Supernet/checkpoint-XX.pth.tar`, which can be used by Search.

```python
def get_args():
    parser = argparse.ArgumentParser("OneShot_cifar_Experiments_Configuration")
    parser.add_argument('--signal', type=str, default='different_hpo', help='describe:glboal_hpo/')
    parser.add_argument('--different-hpo', action='store_true', help='blockwisely lr_scheduler')
    parser.add_argument('--auto-continue', type=bool, default=False, help=' # False if num_trials > 1, after hpo')
    parser.add_argument('--eval', default=False, action='store_true', help='Training mode')
    parser.add_argument('--eval-resume', type=str, default='./snet_detnas.pkl', help='path for eval model')
    # default
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--label-smooth', type=float, default=0.1, help='label smoothing')
    parser.add_argument('--save-interval', type=int, default=5, help='report frequency for saving trained models')
    parser.add_argument('--train-dir', type=str, default='data/train', help='path to training dataset')
    parser.add_argument('--val-dir', type=str, default='data/val', help='path to validation dataset')

    parser.add_argument('--num-trials', type=int, default=2, help='num_trials')
    # total_iters=7800,  # 1560, 3120, 6240, 390, 780, 39000, 1950, 19500, 195*36=7020, 195*40=7800
    parser.add_argument('--total-iters', type=int, default=1560, help='total iters')
    # shuffle 256, mobile 288, cifar-fast 64
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    # network_shuffle: 4 choice for 5 blocks (Shuffle3x3\Shuffle5x5\Shuffle7x7\Xception)
    # network_mobile: 1/2/3 choice for 12 layers {'conv': [1, 2], 'rate': 1},
    parser.add_argument('--choice', type=int, default=4, help='choice')
    # shuffle block 5, mobile 12 layers, cifar_fast 3 layers
    parser.add_argument('--block', type=int, default=3, help='block')
    # shuffle ,1/2/3/4, mobiel, sample_path=ag.space.Int(1, 2, 3),
    parser.add_argument('--sample_path', type=int, default=2, help='sample_path')

    # learning_rate=ag.space.Real(0.1, 0.3, log=True), # glboal hpo, mobile
    # learning_rate=ag.space.Real(0.4, 0.8, log=True), # glboal hpo, shuffle
    # learning_rate=ag.space.Real(0.01, 0.2, log=True),  # glboal hpo, fast
    # parser.add_argument('--learning-rate', type=float, default=0.5, help='init learning rate')
    parser.add_argument('--lr-range', type=str, default='0.01,0.2', help='learning-rate range. default is 0.01,0.2.')
    # parser.add_argument('--weight-decay', type=float, default=4e-5, help='weight decay')
    parser.add_argument('--wd-range', type=str, default='4e-5,5e-3', help='weight-decay range. default is 4e-5,5e-3.')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    arg = get_args()
    arg.local = os.path.split(os.path.realpath(__file__))[0]
    # Select A Best Supernet
    @ag.args(
        # Configuration
        signal=arg.signal,
        different_hpo=arg.different_hpo,
        eval=arg.eval,
        auto_continue=arg.auto_continue,

        # update
        flops_restriction=False, # spos
        activations_count_restriction=False, # pycls
        RandA=False, # RA(RandAugment)
        choice=arg.choice,
        block=arg.block,
        sample_path=arg.sample_path,
        total_iters=arg.total_iters,
        batch_size=arg.batch_size,

        # Supernet HPO, Some tricks work for specific hyperparameters
        # fake=ag.space.Real(0.4, 0.8, log=True),
        # lr & wd & randaug
        # learning_rate=0.4, # test_lr_range # without hpo
        learning_rate=ag.space.Real(arg.lr_range.split(',')[0], arg.lr_range.split(',')[1], log=True),  # glboal hpo, fast
        lr_range = arg.lr_range, # different hpo
        # weight_decay=ag.space.Real(4e-5, 5e-3, log=True),
        weight_decay=ag.space.Real(arg.wd_range.split(',')[0], arg.wd_range.split(',')[1], log=True), # glboal hpo, fast
        # randaug_n=ag.space.Int(3, 4, 5),
        # randaug_m=ag.space.Int(5, 10, 15),

        # default hyperparameters
        randaug_n=3,
        randaug_m=5,
        momentum=arg.momentum,
        label_smooth=arg.label_smooth,
        save_interval=1,
        # save='./models',
        # weight_decay=4e-5,
    )
    def ag_train_cifar(args, reporter):
        return pipeline(args, reporter)

    myscheduler = ag.scheduler.FIFOScheduler(ag_train_cifar,
                                             # resource={'num_cpus': 4, 'num_gpus': 1},
                                             num_trials=arg.num_trials,
                                             time_attr='epoch',
                                             reward_attr="val_acc")
    myscheduler.run()
    myscheduler.join_jobs()
    myscheduler.get_training_curves(filename='{}/save/{}/Supernet_curves'.format(arg.local, arg.signal), plot=True, use_legend=True)
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

+ Cifar10: train shuffle/mobile/resnet block with different_hpo/global_hpo

In my test, only cost 2min in a trial, reached at least 94% test set accuracy, Runtime for 1 epochs is roughly 15s by 64 batch-size.

```bash
cd src/Supernet_cifar
python3 train.py --num-classes 10 --signal different_hpo --different-hpo --num-trials 16 --total-iters 7800 --batch-size 64 --block 4 --lr-range "0.01,0.2" --wd-range "4e-5,5e-3"

python3 train.py --num-classes 10 --signal glboal_hpo --num-trials 16 --total-iters 7800 --batch-size 64 --block 4 --lr-range '0.01,0.2' --wd-range '4e-5,5e-3'

python3 train.py --num-classes 10 --signal without_hpo --num-trials 1 --total-iters 7800 --batch-size 64 --block 4 --lr-range "0.2,0.201" --wd-range "4e-5,5e-3"

---
python3 train.py --num-classes 10 --signal without_hpo --num-trials 2 --total-iters 7800 --batch-size 64 --block 3 --lr-range "0.2,0.201" --wd-range "4e-5,5e-3"


# cifar100
python3 train.py --num-classes 100 --signal cifar100_without_hpo --num-trials 1 --total-iters 7800 --batch-size 64 --block 4 --lr-range "0.2,0.201" --wd-range "4e-5,5e-3"

```


+ Imagenet:
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