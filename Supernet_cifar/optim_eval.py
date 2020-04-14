import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models

# from network_cifar import ShuffleNetV2_OneShot_cifar
from network_mobile import SuperNetwork

import torch.nn.functional as F
import numpy as np
import os
"""
print([i for i in range(5)])

def continuous_str(lst):
    j = 0
    str1 = ''
    for i, item in enumerate(lst):
        if i > 0:
            if lst[i] != lst[i - 1] + 1:
                tmp = lst[j:i]
                if len(tmp) == 1:
                    str1 += str(tmp[0]) + ','
                else:
                    str1 += str(tmp[0]) + "~" + str(tmp[-1]) + ','
                j = i
    tmp2 = lst[j:]
    if len(tmp2) == 1:
        str1 += str(tmp2[0]) + ','
    else:
        str1 += str(tmp2[0]) + "~" + str(tmp2[-1]) + ','

    return str1[:-1]
"""
# fearure.0.0 fearure.0.1 fearure.0.2 fearure.0.3
# fearure.1.0 fearure.1.1 fearure.1.2 fearure.1.3
# fearure.2.0 fearure.2.1 fearure.2.2 fearure.2.3
# fearure.3.0 fearure.3.1 fearure.3.2 fearure.3.3
# fearure.4.0 fearure.4.1 fearure.4.2 fearure.4.3

# lst = [1, 2, 3, 4, 5, 7, 8, 15, 20, 21, 22, 23, 24, 28]
# print(continuous_str(lst))


# get_random_cand = list(np.random.randint(2) for i in range(5*2))
#
# get_random_cand = lambda:tuple(np.random.randint(2) for i in range(5))




# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.conv1 = nn.Conv2d(1, 20, 5)
#         self.conv2 = nn.Conv2d(20, 20, 5)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         return F.relu(self.conv2(x))

# class net(nn.Module):
#     def __init__(self):
#         super(net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, 1)
#         self.conv2 = nn.Conv2d(64, 64, 1)
#         self.conv3 = nn.Conv2d(64, 64, 1)
#         self.conv4 = nn.Conv2d(64, 64, 1)
#         self.conv5 = nn.Conv2d(64, 64, 1)
#     def forward(self, x):
#         out = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))
#         return out

# net = models.__dict__['resnet18'](pretrained=True)

# net = net()

# net = Model()

# net = ShuffleNetV2_OneShot_cifar(n_class=10)

choice = {
    0: {'conv': [0, 0], 'rate': 1},
    1: {'conv': [0, 0], 'rate': 1},
    2: {'conv': [0, 0], 'rate': 1},
    3: {'conv': [0, 0], 'rate': 1},
    4: {'conv': [0, 0], 'rate': 1},
    5: {'conv': [0, 0], 'rate': 1},
    6: {'conv': [0, 0], 'rate': 1},
    7: {'conv': [0, 0], 'rate': 1},
    8: {'conv': [0, 0], 'rate': 1},
    9: {'conv': [0, 0], 'rate': 1},
    10: {'conv': [1, 2], 'rate': 1},
    11: {'conv': [1, 2], 'rate': 0}}

net = SuperNetwork(shadow_bn=False, layers=12, classes=10)

lr = 0.001

op_name_list = []

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


for name, param in net.named_parameters():
    if param.requires_grad:
        print("requires_grad: True ", name)
    else:
        print("requires_grad: False ", name)

    if name.find('Inverted_Block.0') >= 0 and len(param.size()) > 1:
        features0.append(name)
    elif name.find('Inverted_Block.10') >= 0 and len(param.size()) > 1:
        features10.append(name)
    elif name.find('Inverted_Block.11') >= 0 and len(param.size()) > 1:
        features11.append(name)
    elif name.find('Inverted_Block.1') >= 0 and len(param.size()) > 1:
        features1.append(name)
    elif name.find('Inverted_Block.2') >= 0 and len(param.size()) > 1:
        features2.append(name)
    elif name.find('Inverted_Block.3') >= 0 and len(param.size()) > 1:
        features3.append(name)
    elif name.find('Inverted_Block.4') >= 0 and len(param.size()) > 1:
        features4.append(name)
    elif name.find('Inverted_Block.5') >= 0 and len(param.size()) > 1:
        features5.append(name)
    elif name.find('Inverted_Block.6') >= 0 and len(param.size()) > 1:
        features6.append(name)
    elif name.find('Inverted_Block.7') >= 0 and len(param.size()) > 1:
        features7.append(name)
    elif name.find('Inverted_Block.8') >= 0 and len(param.size()) > 1:
        features8.append(name)
    elif name.find('Inverted_Block.9') >= 0 and len(param.size()) > 1:
        features9.append(name)
    else:
        base_conv.append(param)


    # if name.find('features.0') >= 0 and len(param.size()) > 1:
    #     features0.append(name)
    # elif name.find('features.1') >= 0 and len(param.size()) > 1:
    #     features1.append(name)
    # elif name.find('features.2') >= 0 and len(param.size()) > 1:
    #     features2.append(name)
    # elif name.find('features.3') >= 0 and len(param.size()) > 1:
    #     features3.append(name)
    # elif name.find('features.4') >= 0 and len(param.size()) > 1:
    #     features4.append(name)





# 12 stages
op_name_list = [features0, features1, features2, features3, features4,
                features5, features6, features7, features8, features9,
                features10, features11]

# 144 choices
choice = []
for stage in op_name_list:
    # ()=12
    if len(stage) == 12:
        for i in range(0, len(stage), 1):
            a = stage[i:i + 1]
            choice.append(a)

choice.append(base_conv)
# groups = [dict(params=choice[x], lr=lr_group[x]) for x in range(len(choice))]


sub_block = net.children()

# self.conv1 = nn.Sequential(*list(net.features.children())[0:3])

# children = list(net.children())
#
# for child in net.children():
#     # for param in child.conv2d[0].parameters():
#     for param in child.parameters():
#         params = param.view(-1, )
#         print(params)
#
# for name, param in sub_block.named_parameters():
#     print(name, param)

#
# finst_conv_params = list(map(id, net.first_conv.parameters()))
#
# group_weight_decay = []
# group_no_weight_decay = []
# for pname, p in net.named_parameters():
#     # if pname in ['bias']:
#     #     print(p.size())
#     if pname.find('weight') >= 0 and len(p.size()) > 1:
#         # print('include ', pname, p.size())
#         group_weight_decay.append(p)
#     else:
#         # print('not include ', pname, p.size())
#         group_no_weight_decay.append(p)


# for key, param in net.features.parameters.items():
#     print(key, param)

# feature_params = list(map(id, net.features.parameters()))
# feature_params = list(map(id, net.features.0.0.branch_main.parameters()))


# conv5_params = list(map(id, net.conv5.parameters()))
# base_params = filter(lambda p: id(p) not in conv5_params, net.parameters())

# 多层分离
# conv4_params = list(map(id, net.conv4.parameters()))
# conv5_params = list(map(id, net.conv5.parameters()))
# base_params = filter(lambda p: id(p) not in conv5_params + conv4_params, net.parameters())


# conv5学习率是其他层的100倍
# optimizer = torch.optim.SGD([{'params': base_params},
#                              {'params': net.conv5.parameters(), 'lr': lr * 100}], lr=lr, momentum=0.9)

# optimizer = torch.optim.SGD([{'params': base_params},
#                              {'params': net.conv4.parameters(), 'lr': lr * 100},
#                              {'params': net.conv5.parameters(), 'lr': lr * 150}], lr=lr, momentum=0.9)
#



# fixed
"""
w1 = torch.randn(3, 3)
w1.requires_grad = True

w2 = torch.randn(3, 3)
w2.requires_grad = True

o = optim.Adam([w1])
print(o.param_groups)

o.add_param_group({'params': w2})
print(o.param_groups)

"""


if __name__ == "__main__":
    # path
    a = os.path.realpath(__file__)
    b = os.path.split(a)[0]
    c = os.chdir(b)
    os.chdir(os.path.split(os.path.realpath(__file__))[0])
    os.chdir(os.path.split(os.path.realpath(__file__))[0])
    # print(os.getcwd())
