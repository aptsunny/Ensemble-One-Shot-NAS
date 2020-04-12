import os
import sys
import torch
import torch.nn as nn
import numpy as np
import time
import logging
import argparse
import autogluon as ag
import dataset_cifar
from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit
from dataset_cifar import DataIterator, SubsetSampler
# import loader
# from utils.meters import TestMeter, TrainMeter

from network_cifar import ShuffleNetV2_OneShot_cifar
from utils import accuracy, AvgrageMeter, CrossEntropyLabelSmooth, save_checkpoint, get_lastest_model, get_parameters, get_dif_lr_parameters
from flops import get_cand_flops
from opt import Lookahead, BlocklyOptimizer

def adjust_bn_momentum(model, iters):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 1 / iters

def train(model, optimizer, device, args, *, bn_process=False, all_iters=None, reporter=None):
    # optimizer = args.optimizer
    loss_function = args.loss_function
    scheduler = args.scheduler
    train_dataprovider = args.train_dataprovider
    task_id = args.task_id
    val_interval = args.val_interval
    display_interval = args.val_interval

    t1 = time.time()
    Top1_err, Top5_err = 0.0, 0.0
    model.train()

    #
    for iters in range(1, val_interval + 1):
        if bn_process:
            adjust_bn_momentum(model, iters)

        all_iters += 1
        d_st = time.time()

        data, target = train_dataprovider.next()
        target = target.type(torch.LongTensor)
        data, target = data.to(device), target.to(device)
        data_time = time.time() - d_st

        # 4 choice in one block, 20 choices
        # get_random_cand = lambda:tuple(np.random.randint(4) for i in range(20)) # imagenet
        # get_random_cand = lambda:tuple(np.random.randint(2) for i in range(5)) # cifar

        get_random_cand = lambda:tuple(np.random.randint(args.sample_choice) for i in range(args.block)) # cifar

        # uniform
        # flops restriction
        if args.flops_restriction:
            flops_l, flops_r, flops_step = 290, 360, 10
            bins = [[i, i+flops_step] for i in range(flops_l, flops_r, flops_step)]
            # 300 * 1000 000
            def get_uniform_sample_cand(*,timeout=500):
                idx = np.random.randint(len(bins))
                l, r = bins[idx]
                for i in range(timeout):
                    cand = get_random_cand()
                    # if l*1e6 <= get_cand_flops(cand) <= r*1e6:
                    #     return cand
                    return cand
                return get_random_cand()
            output = model(data, get_uniform_sample_cand())
        else:
            output = model(data, get_random_cand())

        loss = loss_function(output, target)
        optimizer.zero_grad()
        loss.backward()

        for p in model.parameters():
            if p.grad is not None and p.grad.sum() == 0:
                p.grad = None

        optimizer.step()
        scheduler.step()

        prec1, prec5 = accuracy(output, target, topk=(1, 5))

        Top1_err += 1 - prec1.item() / 100
        Top5_err += 1 - prec5.item() / 100

        if all_iters % display_interval == 0: #20
            printInfo = 'TRAIN Epoch {}: lr = {:.6f},\tloss = {:.6f},\t'.format(all_iters / display_interval, scheduler.get_lr()[0], loss.item()) + \
                        'Top-1 err = {:.6f},\t'.format(Top1_err / display_interval) + \
                        'Top-5 err = {:.6f},\t'.format(Top5_err / display_interval) + \
                        'iter_load_data_time = {:.6f},\tepoch_train_time = {:.6f}'.format(data_time, time.time() - t1)
            logging.info(printInfo)
            t1 = time.time()
            report_top1, report_top5 = 1 - Top1_err/ display_interval, 1 - Top5_err / display_interval
            reporter(task_id=task_id, epoch=all_iters / display_interval, val_acc=report_top1)
            Top1_err, Top5_err = 0.0, 0.0

    if all_iters % (args.save_interval * val_interval) == 0:
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

def pipeline(args, reporter):
    # num_trials
    print('{}-task_id{}'.format(args['signal'], args['task_id']))

    # Log for one Supernet
    log_format = '[%(asctime)s] %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%d %I:%M:%S')
    t = time.time()
    local_time = time.localtime(t)
    if not os.path.exists('./log'):
        os.mkdir('./log')
    fh = logging.FileHandler(
        os.path.join('log/{}-task_id{}-train-{}{:02}{}'.format(args['signal'], args['task_id'], local_time.tm_year % 2000, local_time.tm_mon, t)))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    use_gpu = False
    if torch.cuda.is_available():
        use_gpu = True

    # load cifar
    dataset_train, dataset_valid = dataset_cifar.get_dataset("cifar10", N=args.randaug_n, M=args.randaug_m, RandA=args.RandA)
    split = 0.0
    split_idx = 0
    train_sampler = None
    if split > 0.0:
        sss = StratifiedShuffleSplit(n_splits=5, test_size=split, random_state=0)
        sss = sss.split(list(range(len(dataset_train))), dataset_train.targets)
        for _ in range(split_idx + 1):
            train_idx, valid_idx = next(sss)

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetSampler(valid_idx)
    else:
        valid_sampler = SubsetSampler([])

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True if train_sampler is None else False, num_workers=32,
        pin_memory=True,
        sampler=train_sampler, drop_last=True)

    valid_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True,
        sampler=valid_sampler, drop_last=False)

    train_dataprovider = DataIterator(train_loader)
    val_dataprovider = DataIterator(valid_loader)
    args.val_interval = int(len(dataset_train) / args.batch_size)
    print('load data successfully')

    model = ShuffleNetV2_OneShot_cifar(block=args['block'], n_class=10)

    # original
    # optimizer = torch.optim.SGD(get_parameters(model),
    #                             lr=args.learning_rate,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

    # parameters divided into groups
    # test lr_group (4 stage * 5choice + 1base_lr == 21)
    # lr_group = [i/100 for i in list(range(4,25,1))]
    # arch_search = list(np.random.randint(2) for i in range(5*2))
    # optimizer = torch.optim.SGD(get_dif_lr_parameters(model, lr_group, arch_search),

    # test lr_range
    args.learning_rate = args.learning_rate * (args['task_id']+ 1)


    # lr and parameters
    if args.different_hpo:
        nums_lr_group = args['block'] * args['choice'] + 1
        lr_group = list(np.random.uniform(0.4, 0.8) for i in range(nums_lr_group))
        optimizer = torch.optim.SGD(get_dif_lr_parameters(model, lr_group),
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.learning_rate, # without hpo / glboal hpo
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    # lookahead optimizer
    # base_opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    # optimizer = Lookahead(base_opt, k=5, alpha=0.5)

    # blockly optimizer
    # base_opt_2 = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
    # base_opt_3 = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.999))
    # base_opt_group = [base_opt, base_opt_2, base_opt_3]
    # optimizer = BlocklyOptimizer(base_opt_group, k=5, alpha=0.5)

    # loss func, ls=0.1
    criterion_smooth = CrossEntropyLabelSmooth(10, args['label_smooth'])

    if use_gpu:
        model = nn.DataParallel(model)
        loss_function = criterion_smooth.cuda()
        device = torch.device("cuda")
    else:
        loss_function = criterion_smooth
        device = torch.device("cpu")

    # lr_scheduler is related to total_iters
    scheduler = torch.optim.lr_scheduler.LambdaLR \
        (optimizer, lambda step: (1.0 - step / args.total_iters) if step <= args.total_iters else 0, last_epoch=-1)

    model = model.to(device)

    all_iters = 0
    if args.auto_continue: # load model
        lastest_model, iters = get_lastest_model()
        if lastest_model is not None:
            all_iters = iters
            checkpoint = torch.load(lastest_model, map_location=None if use_gpu else 'cpu')
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            print('load from checkpoint')
            for i in range(iters):
                scheduler.step() # lr Align

    # args.optimizer = optimizer
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

    # according to total_iters
    while all_iters < args.total_iters:
        all_iters, Top1_acc, Top5_acc = \
            train(model, optimizer, device, args, bn_process=False, all_iters=all_iters, reporter=reporter)

        # all_iters = train(model, device, args, val_interval=int(1280000/args.batch_size), bn_process=True, all_iters=all_iters)

        # save_checkpoint({'state_dict': model.state_dict(),}, args.total_iters, tag='bnps-')

        # print(all_iters, Top1_acc)

    # reporter(task_id=task_id, val_acc=Top1_acc)

# useless
def get_args():
    parser = argparse.ArgumentParser("ShuffleNetV2_OneShot_cifar")
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--eval-resume', type=str, default='./snet_detnas.pkl', help='path for eval model')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--learning-rate', type=float, default=0.5, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float, default=4e-5, help='weight decay')
    parser.add_argument('--save', type=str, default='./models', help='path for saving trained models')
    parser.add_argument('--label-smooth', type=float, default=0.1, help='label smoothing')

    parser.add_argument('--auto-continue', type=bool, default=True, help='report frequency') # load
    parser.add_argument('--display-interval', type=int, default=20, help='report frequency')

    parser.add_argument('--total-iters', type=int, default=150000, help='total iters')
    parser.add_argument('--val-interval', type=int, default=10000, help='report frequency')
    parser.add_argument('--save-interval', type=int, default=10000, help='report frequency')

    parser.add_argument('--train-dir', type=str, default='data/train', help='path to training dataset')
    parser.add_argument('--val-dir', type=str, default='data/val', help='path to validation dataset')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    os.chdir(os.path.split(os.path.realpath(__file__))[0])
    print(os.getcwd())
    # args = get_args()
    # print(args)

    # Select a best supernet
    @ag.args(
        # Training mode
        eval=False,
        auto_continue=False, # False if num_trials > 1
        flops_restriction=False,
        activations_count_restriction=False, # pycls
        RandA=False,
        # Supernet HPO, Some tricks work for specific hyperparameters
        # 4 choice for a block (Shuffle3x3\Shuffle5x5\Shuffle7x7\Xception)
        signal='Experiments',
        choice=4,
        block=5,
        sample_choice=1,
        total_iters=19500,  # 1950,19500, one epoch 195 * 256 = 49920
        batch_size=256,

        # blockly lr_scheduler
        # learning_rate_group=ag.space.Real(0.4, 0.5, log=True),

        fake=ag.space.Real(0.4, 0.8, log=True),
        # lr & wd
        different_hpo=False,
        # learning_rate=ag.space.Real(0.4, 0.8, log=True), # glboal hpo
        # learning_rate=0.4, # without hpo
        learning_rate=0.1, # test_lr_range
        # wd=ag.space.Real(1e-4, 5e-4, log=True),

        # randaug
        randaug_n=3,
        randaug_m=5,
        # randaug_n=ag.space.Int(3, 4, 5),
        # randaug_m=ag.space.Int(5, 10, 15),
        # choice=ag.space.Int(0, 1),

        # default parameters
        # randaug_n=3,
        # randaug_m=5,
        # learning_rate=0.5,
        weight_decay=4e-5,
        momentum=0.9,
        save='./models',
        label_smooth=0.1,
        # display_interval=195,#20
        save_interval=5
    )
    def ag_train_cifar(args, reporter):
        return pipeline(args, reporter)

    # FIFOScheduler
    myscheduler = ag.scheduler.FIFOScheduler(ag_train_cifar,
                                             # resource={'num_cpus': 4, 'num_gpus': 1},
                                             num_trials=8,
                                             time_attr='epoch',
                                             reward_attr="val_acc")
    # print(myscheduler)
    myscheduler.run()
    myscheduler.join_jobs()
    myscheduler.get_training_curves(filename='Supernet_curves', plot=True, use_legend=False)
    print('The Best Supernet Configuration and Accuracy are: {}, {}'.format(myscheduler.get_best_config(),
                                                                   myscheduler.get_best_reward()))