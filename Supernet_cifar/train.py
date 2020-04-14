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
from utils import accuracy, AvgrageMeter, CrossEntropyLabelSmooth, save_checkpoint, get_lastest_model, get_parameters, shuffle_dif_lr_parameters, fast_dif_lr_parameters, mobile_dif_lr_parameters, count_parameters_in_MB, random_choice, adjust_bn_momentum

# network
# from network_mobile import SuperNetwork
# from network_shuffle import ShuffleNetV2_OneShot_cifar
from network_cifar_fast import Network

# optimizer
# from flops import get_cand_flops
# from opt import Lookahead, BlocklyOptimizer

def train(model, device, args, *, bn_process=True, all_iters=None, reporter=None):
    optimizer = args.optimizer
    loss_function = args.loss_function
    scheduler = args.scheduler
    train_dataprovider = args.train_dataprovider
    task_id = args.task_id
    val_interval = args.val_interval
    display_interval = args.val_interval

    t1 = time.time()
    Top1_err, Top5_err = 0.0, 0.0
    model.train()

    for iters in range(1, val_interval + 1):
        if bn_process:
            adjust_bn_momentum(model, iters)

        all_iters += 1
        d_st = time.time()

        data, target = train_dataprovider.next()
        target = target.type(torch.LongTensor)
        data, target = data.to(device), target.to(device)
        data_time = time.time() - d_st

        # search space
        if args.block==5:
            # shuffle, 4 choice in one block, 20 choices
            # get_random_cand = lambda:tuple(np.random.randint(4) for i in range(20)) # imagenet
            # get_random_cand = lambda:tuple(np.random.randint(2) for i in range(5)) # cifar
            get_random_cand = lambda:tuple(np.random.randint(args.sample_path) for i in range(args.block)) # cifar
            # uniform
            # flops restriction
            if args.flops_restriction:
                flops_l, flops_r, flops_step = 290, 360, 10
                bins = [[i, i + flops_step] for i in range(flops_l, flops_r, flops_step)]

                # 300 * 1000 000
                def get_uniform_sample_cand(*, timeout=500):
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

        elif args.block==12:
            # s1, sample_choice=1,45s/epoch, sample_choice=3,65s/epoch
            choice = random_choice(path_num=args.choice, m=args.sample_path, layers=args.block)
            output = model(data, choice)

        elif args.block==3:
            # cifar_fast
            batch = {'input': data, 'target': target}
            states = model(batch)
            output = states['logits']

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
            # print('{}-task_id: {}, lr: {}'.format(args.signal, args.task_id, args.learning_rate))
            printInfo = '{}-Task_id: {}, Base_lr: {:.2f},\t'.format(args.signal, args.task_id, args.learning_rate) + \
                        'TRAIN Epoch {}: lr = {:.4f},\tloss = {:.4f},\t'.format(all_iters / display_interval, scheduler.get_lr()[0], loss.item()) + \
                        'Top-1 err = {:.4f},\t'.format(Top1_err / display_interval) + \
                        'Top-5 err = {:.4f},\t'.format(Top5_err / display_interval) + \
                        'epoch_train_time = {:.2f}'.format(time.time() - t1)
                        # 'iter_load_data_time = {:.6f},\tepoch_train_time = {:.6f}'.format(data_time, time.time() - t1)
            logging.info(printInfo)
            t1 = time.time()
            report_top1, report_top5 = 1 - Top1_err/ display_interval, 1 - Top5_err / display_interval
            # print(all_iters / display_interval, report_top1)
            reporter(task_id=task_id, epoch=all_iters / display_interval, val_acc=report_top1)
            Top1_err, Top5_err = 0.0, 0.0

    if all_iters % (args.save_interval * val_interval) == 0:
        save_checkpoint(args.path, {
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
    # max_val_iters = 250 # 250
    max_val_iters = len(val_dataprovider) # 250
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
    # print('{}-task_id: {}, base_lr: {}'.format(args.signal, args.task_id, args.learning_rate))
    print('{}-task_id: {}'.format(args.signal, args.task_id))

    # Log for one Supernet
    floder = '{}/task_id_{}'.format(args.signal, args.task_id)
    path = os.path.join(arg.local, 'save', floder)
    if not os.path.isdir(path):
        os.makedirs(path)
    args.path = path


    log_format = '[%(asctime)s] %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%d %I:%M:%S')
    t = time.time()
    local_time = time.localtime(t)

    if not os.path.exists('{}/log'.format(path)):
        os.mkdir('{}/log'.format(path))
    fh = logging.FileHandler(
        os.path.join('{}/log/{}-task_id{}-train-{}{:02}{}'.format(path, args['signal'], args['task_id'], local_time.tm_year % 2000, local_time.tm_mon, t)))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    # resource
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
    args.val_interval = int(len(dataset_train) / args.batch_size) # step
    print('load data successfully')

    # network
    # model = ShuffleNetV2_OneShot_cifar(block=args['block'], n_class=10)
    # model = SuperNetwork(shadow_bn=True, layers=args['block'], classes=10)
    # print("param size = %fMB" % count_parameters_in_MB(model))
    # model = Network(net()).to(device).half()
    model = Network()

    # lr and parameters
    # original optimizer lr & wd

    # test lr_range
    # args.learning_rate = args.learning_rate * (args['task_id']+ 1)

    # parameters divided into groups
    # test shuffle lr_group (4 stage * 5choice + 1base_lr == 21)
    # test mobile lr_group (12 stage * 12 choice + 1base_lr == 145)
    # test fast lr_group (3 stage * 1 choice + 1base_lr == 4)

    # lr_group = [i/100 for i in list(range(4,25,1))]
    # arch_search = list(np.random.randint(2) for i in range(5*2))
    # optimizer = torch.optim.SGD(get_dif_lr_parameters(model, lr_group, arch_search),

    if args.different_hpo:
        if args['block']==5:
            nums_lr_group = args['block'] * args['choice'] + 1
            lr_group = list(np.random.uniform(0.4, 0.8) for i in range(nums_lr_group))
            optimizer = torch.optim.SGD(shuffle_dif_lr_parameters(model, lr_group),
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
        elif args['block']==12:
            nums_lr_group=145
            lr_group = list(np.random.uniform(0.1, 0.3) for i in range(nums_lr_group))
            optimizer = torch.optim.SGD(mobile_dif_lr_parameters(model, lr_group),
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)

        elif args['block']==3:
            nums_lr_group=4
            lr_l, lr_r = float(arg.lr_range.split(',')[0]), float(arg.lr_range.split(',')[1])
            lr_group = list(np.random.uniform(lr_l, lr_r) for i in range(nums_lr_group))
            optimizer = torch.optim.SGD(fast_dif_lr_parameters(model, lr_group),
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.learning_rate, # without hpo / glboal hpo
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    # optimizer = torch.optim.SGD(get_parameters(model),
    #                             lr=args.learning_rate,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

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

    # lr_scheduler is related to total_iters
    scheduler = torch.optim.lr_scheduler.LambdaLR \
        (optimizer, lambda step: (1.0 - step / args.total_iters) if step <= args.total_iters else 0, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, float(args.total_iters / args.val_interval), eta_min=1e-8, last_epoch=-1)

    if use_gpu:
        model = nn.DataParallel(model)
        loss_function = criterion_smooth.cuda()
        device = torch.device("cuda")
    else:
        loss_function = criterion_smooth
        device = torch.device("cpu")

    model = model.to(device)

    all_iters = 0
    if args.auto_continue: # load model
        lastest_model, iters = get_lastest_model(args.path)
        if lastest_model is not None:
            all_iters = iters
            checkpoint = torch.load(lastest_model, map_location=None if use_gpu else 'cpu')
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            print('load from checkpoint')
            for i in range(iters):
                scheduler.step() # lr Align

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

    # according to total_iters
    while all_iters < args.total_iters:
        all_iters, Top1_acc, Top5_acc = \
            train(model, device, args, bn_process=True, all_iters=all_iters, reporter=reporter)
        # save_checkpoint({'state_dict': model.state_dict(),}, args.total_iters, tag='bnps-')
    scheduler.step()
    # reporter(task_id=task_id, val_acc=Top1_acc)

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