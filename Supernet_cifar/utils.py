import os
import re
import random
import collections
import numpy as np
import torch
import torch.nn as nn

class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(
            1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * \
            targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss

class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_checkpoint(state, iters, tag=''):
    if not os.path.exists("./models"):
        os.makedirs("./models")
    filename = os.path.join(
        "./models/{}checkpoint-{:06}.pth.tar".format(tag, iters))
    torch.save(state, filename)
    latestfilename = os.path.join(
        "./models/{}checkpoint-latest.pth.tar".format(tag))
    torch.save(state, latestfilename)

def get_lastest_model():
    if not os.path.exists('./models'):
        os.mkdir('./models')
    model_list = os.listdir('./models/')
    if model_list == []:
        return None, 0
    # path = os.path.abspath(model_list[0])
    # print(path)
    model_list.sort()
    lastest_model = model_list[-1]
    iters = re.findall(r'\d+', lastest_model)
    if iters == []:
        lastest_model = model_list[-2]
        iters = re.findall(r'\d+', lastest_model)
    return './models/' + lastest_model, int(iters[0])

def count_parameters_in_MB(model):
    # return np.sum(np.fromiter(np.prod(v.size())) for v in model.parameters())/1e6
    # return np.sum(np.prod(v.size()) for v in model.parameters())/1e6
    return sum(np.prod(v.size()) for v in model.parameters())/1e6

def random_choice(path_num, m, layers):
    # choice = {}
    choice = collections.OrderedDict()
    for i in range(layers):
        # expansion rate
        rate = np.random.randint(low=0, high=2, size=1)[0]
        # conv
        m_ = np.random.randint(low=1, high=(m+1), size=1)[0]
        rand_conv = random.sample(range(path_num), m_)
        choice[i] = {'conv': rand_conv, 'rate': rate}
    return choice

def adjust_bn_momentum(model, iters):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 1 / iters

# weight_decay
def get_parameters(model):
    group_no_weight_decay = []
    group_weight_decay = []
    for pname, p in model.named_parameters():
        if pname.find('weight') >= 0 and len(p.size()) > 1:
            # print('include ', pname, p.size())
            group_weight_decay.append(p)
        else:
            # print('not include ', pname, p.size())
            group_no_weight_decay.append(p)
    assert len(list(model.parameters())) == len(group_weight_decay) + len(group_no_weight_decay)
    groups = [dict(params=group_weight_decay), dict(params=group_no_weight_decay, weight_decay=0.)]
    return groups

def shuffle_dif_lr_parameters(model, lr_group, arch_search=None):
    features0 = []
    features1 = []
    features2 = []
    features3 = []
    features4 = []
    base_conv = []

    # stage
    for name, param in model.named_parameters():
        if name.find('features.0') >= 0 and len(param.size()) > 1:
            features0.append(param)
            # features0.append(name) # check
        elif name.find('features.1') >= 0 and len(param.size()) > 1:
            features1.append(param)
            # features1.append(name)
        elif name.find('features.2') >= 0 and len(param.size()) > 1:
            features2.append(param)
            # features2.append(name)
        elif name.find('features.3') >= 0 and len(param.size()) > 1:
            features3.append(param)
            # features3.append(name)
        elif name.find('features.4') >= 0 and len(param.size()) > 1:
            features4.append(param)
            # features4.append(name)
        else:
            base_conv.append(param)

    # five stages
    op_name_list = [features0, features1, features2, features3, features4]

    # 20 choices
    choice = []
    for stage in op_name_list:
        # (5,5,5,8=23 3,3,3,6=15(feature3))
        if len(stage)==23:
            for i in range(0, len(stage), 5):
                if i ==15:
                    a = stage[i:i + 8]
                    choice.append(a)
                    break
                else:
                    a = stage[i:i + 5]
                    choice.append(a)

        elif len(stage)==15:
            for i in range(0, len(stage), 3):
                if i ==9:
                    b = stage[i:i + 6]
                    choice.append(b)
                    break
                else:
                    b = stage[i:i + 3]
                    choice.append(b)
    choice.append(base_conv)
    groups = [dict(params=choice[x], lr=lr_group[x]) for x in range(len(choice))]
    return groups

def mobile_dif_lr_parameters(model, lr_group, arch_search=None):
    features0 = []
    features1 = []
    features2 = []
    features3 = []
    features4 = []
    features5 = []
    features6 = []
    features7 = []
    features8 = []
    features9 = []
    features10 = []
    features11 = []
    base_conv = []

    # stage
    for name, param in model.named_parameters():
        if name.find('Inverted_Block.0') >= 0 and len(param.size()) > 1:
            features0.append(param)
        elif name.find('Inverted_Block.10') >= 0 and len(param.size()) > 1:
            features10.append(param)
        elif name.find('Inverted_Block.11') >= 0 and len(param.size()) > 1:
            features11.append(param)
        elif name.find('Inverted_Block.1') >= 0 and len(param.size()) > 1:
            features1.append(param)
        elif name.find('Inverted_Block.2') >= 0 and len(param.size()) > 1:
            features2.append(param)
        elif name.find('Inverted_Block.3') >= 0 and len(param.size()) > 1:
            features3.append(param)
        elif name.find('Inverted_Block.4') >= 0 and len(param.size()) > 1:
            features4.append(param)
        elif name.find('Inverted_Block.5') >= 0 and len(param.size()) > 1:
            features5.append(param)
        elif name.find('Inverted_Block.6') >= 0 and len(param.size()) > 1:
            features6.append(param)
        elif name.find('Inverted_Block.7') >= 0 and len(param.size()) > 1:
            features7.append(param)
        elif name.find('Inverted_Block.8') >= 0 and len(param.size()) > 1:
            features8.append(param)
        elif name.find('Inverted_Block.9') >= 0 and len(param.size()) > 1:
            features9.append(param)
        else:
            base_conv.append(param)

    # 12 stages
    op_name_list = [features0, features1, features2, features3, features4,
                    features5, features6, features7, features8, features9,
                    features10, features11]

    # 144 + 1 choices
    choice = []
    for stage in op_name_list:
        # ()=12
        if len(stage)==12:
            for i in range(0, len(stage), 1):
                a = stage[i:i+1]
                choice.append(a)

    choice.append(base_conv)
    groups = [dict(params=choice[x], lr=lr_group[x]) for x in range(len(choice))]
    return groups