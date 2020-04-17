import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, defaultdict

#####################
## dict utils
#####################

union = lambda *dicts: {k: v for d in dicts for (k, v) in d.items()}

def path_iter(nested_dict, pfx=()):
    for name, val in nested_dict.items():
        if isinstance(val, dict): yield from path_iter(val, (*pfx, name))
        else: yield ((*pfx, name), val)

def map_nested(func, nested_dict):
    return {k: map_nested(func, v) if isinstance(v, dict) else func(v) for k,v in nested_dict.items()}

def group_by_key(items):
    res = defaultdict(list)
    for k, v in items:
        res[k].append(v)
    return res

#####################
## graph building
#####################
sep = '/'

def split(path):
    i = path.rfind(sep) + 1
    return path[:i].rstrip(sep), path[i:]

def normpath(path):
    #simplified os.path.normpath
    parts = []
    for p in path.split(sep):
        if p == '..': parts.pop()
        elif p.startswith(sep): parts = [p]
        else: parts.append(p)
    return sep.join(parts)

has_inputs = lambda node: type(node) is tuple

def pipeline(net):
    return [(sep.join(path), (node if has_inputs(node) else (node, [-1]))) for (path, node) in path_iter(net)]

def build_graph(net):
    flattened = pipeline(net)
    resolve_input = lambda rel_path, path, idx: normpath(sep.join((path, '..', rel_path))) if isinstance(rel_path, str) else flattened[idx+rel_path][0]
    return {path: (node[0], [resolve_input(rel_path, path, idx) for rel_path in node[1]]) for idx, (path, node) in enumerate(flattened)}


# Network

def conv_bn_3(c_in, c_out):
    return {
        'conv3': nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False),
        'bn': BatchNorm(c_out),
        'relu': nn.ReLU(True)
    }

def conv_bn_5(c_in, c_out, pad=1):
    return {
        'conv5': nn.Conv2d(c_in, c_out, kernel_size=5, stride=1, padding=pad, bias=False),
        'bn': BatchNorm(c_out),
        'relu': nn.ReLU(True)
    }

def conv_bn_7(c_in, c_out, pad=1):
    return {
        'conv7': nn.Conv2d(c_in, c_out, kernel_size=7, stride=1, padding=pad, bias=False),
        'bn': BatchNorm(c_out),
        'relu': nn.ReLU(True)
    }

def residual(c):
    return {
        'in': Identity(),
        'res1': conv_bn_3(c, c),
        'res2': conv_bn_3(c, c),
        'add': (Add(), ['in', 'res2/relu']),
    }

def residual_5(c):
    return {
        'in': Identity(),
        'res1_5': conv_bn_5(c, c, pad=2),
        'res2_5': conv_bn_5(c, c, pad=2),
        'add': (Add(), ['in', 'res2_5/relu']),
        # 'res_5': (conv_bn_5(c, c), ['in']),
        # 'res2_5': conv_bn_5(c, c),
        # 'res_7': (conv_bn_7(c, c), ['in']),
        # 'res2_7': conv_bn_7(c, c),
        # 'add': (AddWeighted(1, 1, 0, 0), ['in', 'res2/relu', 'res2_5/relu', 'res2_7/relu']),
        # 'add_5': (AddWeighted(1, 0, 1, 0), ['in', 'res2/relu', 'res2_5/relu', 'res2_7/relu']),
        # 'add_7': (AddWeighted(1, 0, 0, 1), ['in', 'res2/relu', 'res2_5/relu', 'res2_7/relu']),
    }

def residual_7(c):
    return {
        'in': Identity(),
        'res1_7': conv_bn_7(c, c, pad=3),
        'res2_7': conv_bn_7(c, c, pad=3),
        'add': (Add(), ['in', 'res2_7/relu']),
        # 'add_7': (AddWeighted(1, 0, 0, 1), ['in', 'res2/relu', 'res2_5/relu', 'res2_7/relu']),
    }


class Identity(namedtuple('Identity', [])):
    def __call__(self, x): return x

class Add(namedtuple('Add', [])):
    def __call__(self, x, y):
        return x + y

class AddWeighted_bi(namedtuple('AddWeighted', ['wx', 'wy'])):
    def __call__(self, x, y): return self.wx * x + self.wy * y

class AddWeighted(namedtuple('AddWeighted', ['wx', 'wy', 'wu', 'wv'])):
    def __call__(self, x, y, u, v): return self.wx * x + self.wy * y + self.wu * u + self.wv * v

class Mul(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
    def __call__(self, x):
        return x * self.weight

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), x.size(1))

class Concat(nn.Module):
    def forward(self, *xs): return torch.cat(xs, 1)

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight_freeze=False, bias_freeze=False, weight_init=1.0,
                 bias_init=0.0):
        super().__init__(num_features, eps=eps, momentum=momentum)
        if weight_init is not None: self.weight.data.fill_(weight_init)
        if bias_init is not None: self.bias.data.fill_(bias_init)
        self.weight.requires_grad = not weight_freeze
        self.bias.requires_grad = not bias_freeze

class GhostBatchNorm(BatchNorm):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        self.register_buffer('running_mean', torch.zeros(num_features * self.num_splits))
        self.register_buffer('running_var', torch.ones(num_features * self.num_splits))

    def train(self, mode=True):
        if (self.training is True) and (mode is False):  # lazily collate stats when we are going to use them
            self.running_mean = torch.mean(self.running_mean.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
            self.running_var = torch.mean(self.running_var.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
        return super().train(mode)

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            return nn.functional.batch_norm(
                input.view(-1, C * self.num_splits, H, W), self.running_mean, self.running_var,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
        else:
            return nn.functional.batch_norm(
                input, self.running_mean[:self.num_features], self.running_var[:self.num_features],
                self.weight, self.bias, False, self.momentum, self.eps)

class Network(nn.Module):
    def __init__(self, channels=None, weight=0.125, pool=nn.MaxPool2d(2), extra_layers=(), res_layers=('layer1', 'layer3'), num_classes=10):
        # res_layers=('layer3'), shuffle_layers=('layer1')):
        super().__init__()
        channels = channels or {'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 512}
        n = {
            'input': (None, []),
            'prep': conv_bn_3(3, channels['prep']),

            'layer1': dict(conv_bn_3(channels['prep'], channels['layer1']), pool=pool),

            'layer2': dict(conv_bn_3(channels['layer1'], channels['layer2']), pool=pool),

            'layer3': dict(conv_bn_3(channels['layer2'], channels['layer3']), pool=pool),

            'pool': nn.MaxPool2d(4),
            'flatten': Flatten(),
            'linear': nn.Linear(channels['layer3'], num_classes, bias=False),
            'logits': Mul(weight),
        }
        for layer in res_layers:
            n[layer]['residual'] = residual(channels[layer])
            # n[layer]['residual_5'] = residual_5(channels[layer])
            # n[layer]['residual_7'] = residual_7(channels[layer])

        for layer in extra_layers:
            n[layer]['extra'] = conv_bn_3(channels[layer], channels[layer])

        # for layer in shuffle_layers:
        #     n[layer]['residual'] = residual(channels[layer])

        self.graph = build_graph(n)
        for path, (val, _) in self.graph.items():
            setattr(self, path.replace('/', '_'), val)

    def nodes(self):
        return (node for node, _ in self.graph.values())

    def forward(self, inputs):
        outputs = dict(inputs)
        for k, (node, ins) in self.graph.items():
            if k not in outputs:
                outputs[k] = node(*[outputs[x] for x in ins])
        return outputs

    def half(self):
        for node in self.nodes():
            if isinstance(node, nn.Module) and not isinstance(node, nn.BatchNorm2d):
                node.half()
        return self

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, choice=None):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.kernel_size = [3, 5, 7]
        self.kernel_padding = [1, 2, 3]
        self.conbine = nn.ModuleList([])
        result = [int(choice / 3), choice % 3]
        self.conbine= nn.Sequential(
                    self.conv_b(inplanes, planes, kernel_size=self.kernel_size[result[0]], padding=self.kernel_padding[result[0]]),
                    nn.BatchNorm2d(planes),
                    nn.ReLU(inplace=True),
                    self.conv_b(inplanes, planes, kernel_size=self.kernel_size[result[1]],padding=self.kernel_padding[result[1]]),
                    nn.BatchNorm2d(planes),
                    nn.ReLU(inplace=True)
                )

    def conv_b(self, in_planes, out_planes, kernel_size, padding, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                         padding=padding, bias=False)
    # def conv3x3(self, in_planes, out_planes, stride=1):
    #     "3x3 convolution with padding"
    #     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #                      padding=1, bias=False)
    # def conv5(self, in_planes, out_planes, stride=1):
    #     return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
    #                      padding=2, bias=False)
    # def conv7(self, in_planes, out_planes, stride=1):
    #     return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
    #                      padding=3, bias=False)

    def forward(self, x):
        residual = x
        # out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)
        # out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.relu(out)
        # out += residual
        out = self.conbine(x)
        out = out + residual
        return out

class Network_cifar(nn.Module):
    def __init__(self, num_classes=10):
        super(Network_cifar, self).__init__()
        channels = {'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 512}
        # prep
        self.conv1 = self.conv3x3(3, channels['prep'])
        self.bn1 = nn.BatchNorm2d(channels['prep'])
        self.relu = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True) #

        # layer1
        self.conv2 = self.conv3x3(channels['prep'], channels['layer1'])
        self.bn2 = nn.BatchNorm2d(channels['layer1'])
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2)
        self.layer1_features = torch.nn.ModuleList() # 9 choices
        for blockIndex in range(9):
            self.layer1_features.append(
                BasicBlock(channels['layer1'], channels['layer1'], stride=1, choice=blockIndex))

        # layer2
        self.conv3 = self.conv3x3(channels['layer1'], channels['layer2'], stride=1)
        self.bn3 = nn.BatchNorm2d(channels['layer2'])
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2)

        # layer3
        self.conv4 = self.conv3x3(channels['layer2'], channels['layer3'], stride=1)
        self.bn4 = nn.BatchNorm2d(channels['layer3'])
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2)
        self.layer3_features = torch.nn.ModuleList()
        for blockIndex in range(9):
            self.layer3_features.append(
                BasicBlock(channels['layer3'], channels['layer3'], stride=1, choice=blockIndex)) #

        self.avgpool = nn.MaxPool2d(4)
        self.fc = nn.Linear(channels['layer3'], num_classes, bias=False)


    def conv3x3(self, in_planes, out_planes, stride=1):
        "3x3 convolution with padding"
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)

    def swish(self, x, beta=1):
        return x * F.sigmoid(beta * x)

    def mish(self, x):
        return x * torch.tanh(F.softplus(x))

    def forward(self, x, architecture):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.layer1_features[architecture[0]](x)
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.relu3(x)
        # x = self.pool3(x)
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        # x = self.conv4(x)
        # x = self.bn4(x)
        # x = self.relu4(x)
        # x = self.pool4(x)
        x = self.layer3_features[architecture[1]](x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x) * 0.125 # weight=0.125
        # x = self.fc(x)  # weight=0.125
        return x
