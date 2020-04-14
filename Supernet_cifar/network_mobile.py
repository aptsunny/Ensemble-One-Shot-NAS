import math
import torch
import torch.nn as nn

#
class Inverted_Bottleneck(nn.Module):
    def __init__(self, inplanes, outplanes, shadow_bn, stride, activation=nn.ReLU6):
        super(Inverted_Bottleneck, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.shadow_bn = shadow_bn
        self.stride = stride
        self.kernel_list = [3, 5, 7, 9]
        self.expansion_rate = [3, 6]
        self.activation = activation(inplace=True)

        self.pw = nn.ModuleList([])
        self.mix_conv = nn.ModuleList([])
        self.mix_bn = nn.ModuleList([])
        self.pw_linear = nn.ModuleList([])

        for t in self.expansion_rate:
            # pw
            self.pw.append(nn.Sequential(
                nn.Conv2d(inplanes, inplanes * t, kernel_size=1, bias=False),
                nn.BatchNorm2d(inplanes * t),
                activation(inplace=True)
            ))
            # dw
            conv_list = nn.ModuleList([])
            for j in self.kernel_list:
                conv_list.append(nn.Sequential(
                    nn.Conv2d(inplanes * t, inplanes * t, kernel_size=j, stride=stride, padding=j // 2,
                              bias=False, groups=inplanes * t),
                    nn.BatchNorm2d(inplanes * t),
                    activation(inplace=True)
                ))

            self.mix_conv.append(conv_list)
            del conv_list
            # pw
            self.pw_linear.append(nn.Conv2d(inplanes * t, outplanes, kernel_size=1, bias=False))

            bn_list = nn.ModuleList([])
            if self.shadow_bn:
                for j in range(len(self.kernel_list)):
                    bn_list.append(nn.BatchNorm2d(outplanes))
                self.mix_bn.append(bn_list)
            else:
                self.mix_bn.append(nn.BatchNorm2d(outplanes))
            del bn_list

    def forward(self, x, choice):
        # choice: {'conv', 'rate'} sample path
        conv_ids = choice['conv']  # conv_ids, e.g. [0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]
        m_ = len(conv_ids)  # num of selected paths
        rate_id = choice['rate']  # rate_ids, e.g. 0, 1
        assert m_ in [1, 2, 3, 4]
        assert rate_id in [0, 1]
        residual = x
        # pw
        out = self.pw[rate_id](x)
        # dw
        if m_ == 1: # single path
            out = self.mix_conv[rate_id][conv_ids[0]](out)
        else: # multi path
            temp = []
            for id in conv_ids:
                temp.append(self.mix_conv[rate_id][id](out))
            out = sum(temp) # sum
        # pw
        out = self.pw_linear[rate_id](out)
        if self.shadow_bn:
            out = self.mix_bn[rate_id][m_ - 1](out)
        else:
            out = self.mix_bn[rate_id](out)

        # residual
        if self.stride == 1 and self.inplanes == self.outplanes:
            out = out + residual
        return out


channel = [32, 48, 48, 96, 96, 96, 192, 192, 192, 256, 256, 320, 320]
last_channel = 1280


class SuperNetwork(nn.Module):
    def __init__(self, shadow_bn, layers=12, classes=10):
        super(SuperNetwork, self).__init__()
        self.layers = layers

        self.stem = nn.Sequential(
            nn.Conv2d(3, channel[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel[0]),
            nn.ReLU6(inplace=True)
        )

        self.Inverted_Block = nn.ModuleList([])
        for i in range(self.layers):
            if i in [2, 5]: # layer3, layer6
                self.Inverted_Block.append(Inverted_Bottleneck(channel[i], channel[i + 1], shadow_bn, stride=2))
            else:
                self.Inverted_Block.append(Inverted_Bottleneck(channel[i], channel[i + 1], shadow_bn, stride=1))

        self.last_conv = nn.Sequential(
            nn.Conv2d(channel[-1], last_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(inplace=True)
        )

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(last_channel, classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(0)  # fan-out
                init_range = 1.0 / math.sqrt(n)
                m.weight.data.uniform_(-init_range, init_range)
                m.bias.data.zero_()

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


    def forward(self, x, choice=None):

        # choice = {
        #     0: {'conv': [0, 0], 'rate': 1},
        #     1: {'conv': [0, 0], 'rate': 1},
        #     2: {'conv': [0, 0], 'rate': 1},
        #     3: {'conv': [0, 0], 'rate': 1},
        #     4: {'conv': [0, 0], 'rate': 1},
        #     5: {'conv': [0, 0], 'rate': 1},
        #     6: {'conv': [0, 0], 'rate': 1},
        #     7: {'conv': [0, 0], 'rate': 1},
        #     8: {'conv': [0, 0], 'rate': 1},
        #     9: {'conv': [0, 0], 'rate': 1},
        #     10: {'conv': [1, 2], 'rate': 1},
        #     11: {'conv': [1, 2], 'rate': 0}}

        x = self.stem(x)
        for i in range(self.layers):
            x = self.Inverted_Block[i](x, choice[i])
        x = self.last_conv(x)
        x = self.global_pooling(x)
        x = x.view(-1, last_channel) #
        x = self.classifier(x)
        return x


if __name__ == '__main__':
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

    model = SuperNetwork(shadow_bn=False, layers=12, classes=10)
    print(model)
    input = torch.randn(3, 32, 32).unsqueeze(0)
    print(model(input, choice)) # (1, 10)

    #
    # params = list(model.parameters())
    # p_s = params[1].size()
    # model.conv1.zero_grad()
    # model.conv1.weight.grad()

    import torch
    from ptflops import get_model_complexity_info
    with  torch.cuda.device(0):
        # choice is added
        flops, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True, print_per_layer_stat=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

