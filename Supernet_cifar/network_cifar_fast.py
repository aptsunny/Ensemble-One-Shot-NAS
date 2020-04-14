import math
import torch
import torch.nn as nn
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

def conv_bn(c_in, c_out):
    return {
        'conv': nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False),
        'bn': BatchNorm(c_out),
        'relu': nn.ReLU(True)
    }

def residual(c):
    return {
        'in': Identity(),
        'res1': conv_bn(c, c),
        'res2': conv_bn(c, c),
        'add': (Add(), ['in', 'res2/relu']),
    }

class Identity(namedtuple('Identity', [])):
    def __call__(self, x): return x

class Add(namedtuple('Add', [])):
    def __call__(self, x, y): return x + y

class AddWeighted(namedtuple('AddWeighted', ['wx', 'wy'])):
    def __call__(self, x, y): return self.wx * x + self.wy * y

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
    def __init__(self, channels=None, weight=0.125, pool=nn.MaxPool2d(2), extra_layers=(), res_layers=('layer1', 'layer3')):
        # res_layers=('layer3'), shuffle_layers=('layer1')):
        super().__init__()
        channels = channels or {'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 512}
        n = {
            'input': (None, []),
            'prep': conv_bn(3, channels['prep']),

            'layer1': dict(conv_bn(channels['prep'], channels['layer1']), pool=pool),

            'layer2': dict(conv_bn(channels['layer1'], channels['layer2']), pool=pool),

            'layer3': dict(conv_bn(channels['layer2'], channels['layer3']), pool=pool),

            'pool': nn.MaxPool2d(4),
            'flatten': Flatten(),
            'linear': nn.Linear(channels['layer3'], 10, bias=False),
            'logits': Mul(weight),
        }
        for layer in res_layers:
            n[layer]['residual'] = residual(channels[layer])

        for layer in extra_layers:
            n[layer]['extra'] = conv_bn(channels[layer], channels[layer])

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
            # only compute nodes that are not supplied as inputs.
            if k not in outputs:
                outputs[k] = node(*[outputs[x] for x in ins])
        return outputs

    def half(self):
        for node in self.nodes():
            if isinstance(node, nn.Module) and not isinstance(node, nn.BatchNorm2d):
                node.half()
        return self




