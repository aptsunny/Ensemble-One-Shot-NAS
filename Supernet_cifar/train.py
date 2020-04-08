import os
import sys
import torch
import argparse
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import cv2
import numpy as np
import PIL
from PIL import Image
import time
import logging
import argparse
from network_cifar import ShuffleNetV2_OneShot_cifar
from utils import accuracy, AvgrageMeter, CrossEntropyLabelSmooth, save_checkpoint, get_lastest_model, get_parameters
# from utils import accuracy, AvgrageMeter, CrossEntropyLabelSmooth, save_checkpoint, get_parameters
from flops import get_cand_flops

import autogluon as ag

class OpencvResize(object):

    def __init__(self, size=256):
        self.size = size

    def __call__(self, img):
        assert isinstance(img, PIL.Image.Image)
        img = np.asarray(img) # (H,W,3) RGB
        img = img[:,:, ::-1] # 2 BGR
        img = np.ascontiguousarray(img)
        H, W, _ = img.shape
        target_size = (int(self.size/H * W + 0.5), self.size) if H < W else (self.size, int(self.size/W * H + 0.5))
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        img = img[:,:, ::-1] # 2 RGB
        img = np.ascontiguousarray(img)
        img = Image.fromarray(img)
        return img

class ToBGRTensor(object):

    def __call__(self, img):
        assert isinstance(img, (np.ndarray, PIL.Image.Image))
        if isinstance(img, PIL.Image.Image):
            img = np.asarray(img)
        img = img[:,:, ::-1] # 2 BGR
        img = np.transpose(img, [2, 0, 1]) # 2 (3, H, W)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float()
        return img

class DataIterator(object):

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = enumerate(self.dataloader)

    def next(self):
        try:
            _, data = next(self.iterator)
        except Exception:
            self.iterator = enumerate(self.dataloader)
            _, data = next(self.iterator)
        return data[0], data[1]

def adjust_bn_momentum(model, iters):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 1 / iters

def train(model, device, args, *, val_interval, bn_process=False, all_iters=None, task_id=0, reporter=None):

    optimizer = args.optimizer
    loss_function = args.loss_function
    scheduler = args.scheduler
    train_dataprovider = args.train_dataprovider

    t1 = time.time()
    Top1_err, Top5_err = 0.0, 0.0
    model.train()
    for iters in range(1, val_interval + 1):
        scheduler.step()
        if bn_process:
            adjust_bn_momentum(model, iters)

        all_iters += 1
        d_st = time.time()

        data, target = train_dataprovider.next()

        target = target.type(torch.LongTensor)
        data, target = data.to(device), target.to(device)
        data_time = time.time() - d_st


        # get_random_cand = lambda:tuple(np.random.randint(4) for i in range(20)) # imagenet
        get_random_cand = lambda:tuple(np.random.randint(2) for i in range(5)) # cifar
        # get_random_cand = lambda:tuple(np.random.randint(4) for i in range(5)) # cifar
        # get_random_cand = lambda:tuple(np.random.randint(args.choice) for i in range(5)) # cifar

        flops_l, flops_r, flops_step = 290, 360, 10
        bins = [[i, i+flops_step] for i in range(flops_l, flops_r, flops_step)]

        def get_uniform_sample_cand(*,timeout=500):
            idx = np.random.randint(len(bins))
            l, r = bins[idx]
            for i in range(timeout):
                cand = get_random_cand()
                if l*1e6 <= get_cand_flops(cand) <= r*1e6:
                    return cand
            return get_random_cand()

        # uniform
        output = model(data, get_uniform_sample_cand())
        loss = loss_function(output, target)
        optimizer.zero_grad()
        loss.backward()

        # reporter(task_id=task_id, total_iters=all_iters, val_acc=1 - Top1_err / args.display_interval)

        # reporter(task_id=task_id, total_iters=all_iters, loss=loss)

        for p in model.parameters():
            if p.grad is not None and p.grad.sum() == 0:
                p.grad = None

        optimizer.step()
        prec1, prec5 = accuracy(output, target, topk=(1, 5))

        Top1_err += 1 - prec1.item() / 100
        Top5_err += 1 - prec5.item() / 100



        if all_iters % args.display_interval == 0: #20
            printInfo = 'TRAIN Iter {}: lr = {:.6f},\tloss = {:.6f},\t'.format(all_iters, scheduler.get_lr()[0], loss.item()) + \
                        'Top-1 err = {:.6f},\t'.format(Top1_err / args.display_interval) + \
                        'Top-5 err = {:.6f},\t'.format(Top5_err / args.display_interval) + \
                        'data_time = {:.6f},\ttrain_time = {:.6f}'.format(data_time, (time.time() - t1) / args.display_interval)


            logging.info(printInfo)
            t1 = time.time()

            # if iters==val_interval:
            report_top1, report_top5 = 1 - Top1_err/ args.display_interval, 1 - Top5_err / args.display_interval

            reporter(task_id=task_id, total_iters=all_iters, val_acc=report_top1)

            Top1_err, Top5_err = 0.0, 0.0

        if all_iters % args.save_interval == 0:
            save_checkpoint({
                'state_dict': model.state_dict(),
                }, all_iters, tag='Supernet:{}_'.format(task_id))



    return all_iters, report_top1, report_top5

def validate(model, device, args, *, all_iters=None):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()

    loss_function = args.loss_function
    val_dataprovider = args.val_dataprovider

    model.eval()
    max_val_iters = 5 # 250
    t1  = time.time()
    with torch.no_grad():
        for _ in range(1, max_val_iters + 1):
            data, target = val_dataprovider.next()
            target = target.type(torch.LongTensor)
            data, target = data.to(device), target.to(device)

            # add
            cifar_architecture = [0, 0, 0, 0, 0]
            output = model(data, cifar_architecture)
            loss = loss_function(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            n = data.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)


    logInfo = 'TEST Iter {}: loss = {:.6f},\t'.format(all_iters, objs.avg) + \
              'Top-1 err = {:.6f},\t'.format(1 - top1.avg / 100) + \
              'Top-5 err = {:.6f},\t'.format(1 - top5.avg / 100) + \
              'val_time = {:.6f}'.format(time.time() - t1)
    logging.info(logInfo)


# def main():
def main(args, reporter):
    # print(args)
    task_id = args['task_id']

    # Log
    log_format = '[%(asctime)s] %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%d %I:%M:%S')
    t = time.time()
    local_time = time.localtime(t)
    if not os.path.exists('./log'):
        os.mkdir('./log')
    fh = logging.FileHandler(
        os.path.join('log/task_id{}-train-{}{:02}{}'.format(task_id, local_time.tm_year % 2000, local_time.tm_mon, t)))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    use_gpu = False
    if torch.cuda.is_available():
        use_gpu = True

    """
    # imagenet data
    assert os.path.exists(args.train_dir)
    train_dataset = datasets.ImageFolder(
        args.train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(0.5),
            ToBGRTensor(),
        ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=1, pin_memory=use_gpu)

    train_dataprovider = DataIterator(train_loader)

    assert os.path.exists(args.val_dir)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.val_dir, transforms.Compose([
            OpencvResize(256),
            transforms.CenterCrop(224),
            ToBGRTensor(),
        ])),
        batch_size=200, shuffle=False,
        num_workers=1, pin_memory=use_gpu
    )
    val_dataprovider = DataIterator(val_loader)

    """

    # load cifar
    import dataset_cifar #50000，10000
    dataset_train, dataset_valid = dataset_cifar.get_dataset("cifar10")
    train_loader = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=args.batch_size,
                                               num_workers=1)
    valid_loader = torch.utils.data.DataLoader(dataset_valid,
                                               batch_size=args.batch_size,
                                               num_workers=1)
    train_dataprovider = DataIterator(train_loader)
    val_dataprovider = DataIterator(valid_loader)

    print('load data successfully')

    # model = ShuffleNetV2_OneShot()
    model = ShuffleNetV2_OneShot_cifar(n_class=10)
    # get_parameters
    optimizer = torch.optim.SGD(get_parameters(model),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # loss func, ls=0.1
    # criterion_smooth = CrossEntropyLabelSmooth(1000, 0.1)
    criterion_smooth = CrossEntropyLabelSmooth(10, 0.1)

    if use_gpu:
        model = nn.DataParallel(model)
        loss_function = criterion_smooth.cuda()
        device = torch.device("cuda")
    else:
        loss_function = criterion_smooth
        device = torch.device("cpu")

    # lr_scheduler is related to total_iters
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lambda step: (
                                                              1.0 - step / args.total_iters) if step <= args.total_iters else 0,
                                                  last_epoch=-1)

    model = model.to(device)

    all_iters = 0
    if args.auto_continue: # 载入model
        lastest_model, iters = get_lastest_model()
        if lastest_model is not None:
            all_iters = iters
            checkpoint = torch.load(lastest_model, map_location=None if use_gpu else 'cpu')
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            print('load from checkpoint')
            for i in range(iters):
                scheduler.step() # lr 对齐

    args.optimizer = optimizer
    args.loss_function = loss_function
    args.scheduler = scheduler

    args.train_dataprovider = train_dataprovider
    args.val_dataprovider = val_dataprovider

    if args.eval:
        if args.eval_resume is not None:
            checkpoint = torch.load(args.eval_resume, map_location=None if use_gpu else 'cpu')
            # model.load_state_dict(checkpoint, strict=True)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            validate(model, device, args, all_iters=all_iters)
        exit(0)

    while all_iters < args.total_iters:
        all_iters, Top1_acc, Top5_acc = \
            train(model, device, args, val_interval=args.val_interval, bn_process=False, all_iters=all_iters, task_id=task_id, reporter=reporter)
        # print(all_iters, Top1_acc)

    # all_iters = train(model, device, args, val_interval=int(1280000/args.batch_size), bn_process=True, all_iters=all_iters)
    # save_checkpoint({'state_dict': model.state_dict(),}, args.total_iters, tag='bnps-')

    # reporter(task_id=task_id, val_acc=Top1_acc)

# useless
def get_args():
    parser = argparse.ArgumentParser("ShuffleNetV2_OneShot_cifar")
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--eval-resume', type=str, default='./snet_detnas.pkl', help='path for eval model')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--total-iters', type=int, default=150000, help='total iters') # 与lr直接相关
    parser.add_argument('--learning-rate', type=float, default=0.5, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float, default=4e-5, help='weight decay')
    parser.add_argument('--save', type=str, default='./models', help='path for saving trained models')
    parser.add_argument('--label-smooth', type=float, default=0.1, help='label smoothing')

    parser.add_argument('--auto-continue', type=bool, default=True, help='report frequency') # load
    parser.add_argument('--display-interval', type=int, default=20, help='report frequency')
    parser.add_argument('--val-interval', type=int, default=10000, help='report frequency') # iter 跑了多少轮
    parser.add_argument('--save-interval', type=int, default=10000, help='report frequency')

    parser.add_argument('--train-dir', type=str, default='data/train', help='path to training dataset')
    parser.add_argument('--val-dir', type=str, default='data/val', help='path to validation dataset')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    os.chdir(os.path.split(os.path.realpath(__file__))[0])
    print(os.getcwd())
    # 0
    # main()

    args = get_args()
    print(args)
    @ag.args(
        # training
        eval=False,
        auto_continue=False, # False if num_trials > 1

        # hpo
        # choice=ag.space.Int(0, 1),
        learning_rate=ag.space.Real(0.4, 0.5, log=True),
        # wd=ag.space.Real(1e-4, 5e-4, log=True),

        # learning_rate=0.5,
        weight_decay=4e-5,
        total_iters=2000,#200
        val_interval=1000,#100
        batch_size=256,
        momentum=0.9,
        save='./models',
        label_smooth=0.1,
        display_interval=20,
        save_interval=100
    )
    def ag_train_cifar(args, reporter):
        return main(args, reporter)

    myscheduler = ag.scheduler.FIFOScheduler(ag_train_cifar,
                                             # resource={'num_cpus': 4, 'num_gpus': 1},
                                             num_trials=2,
                                             time_attr='all_iters',
                                             reward_attr="val_acc")
    print(myscheduler)
    myscheduler.run()
    myscheduler.join_jobs()
    myscheduler.get_training_curves(filename='Supernet', plot=True, use_legend=False)
    print('The Best Configuration and Accuracy are: {}, {}'.format(myscheduler.get_best_config(),
                                                                   myscheduler.get_best_reward()))