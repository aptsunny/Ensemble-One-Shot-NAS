from collections import defaultdict
from itertools import chain
from torch.optim import Optimizer
import torch
import warnings


class BlocklyOptimizer(Optimizer):
    def __init__(self, optimizer_group, k=5, alpha=0.5):
        self.nums_of_block = len(optimizer_group) # first block , actually (block * choice)
        self.optimizer_group = optimizer_group # a group of different opt

        self.state = defaultdict(dict)
        # self.fast_state = self.optimizer.state

        # for group in self.param_groups:
        #     group["counter"] = 0  # insert counter
        self.param_groups = []
        # for optimizer_single in self.optimizer_group:
        for i, optimizer_single in enumerate(self.optimizer_group):
            self.param_groups.append(optimizer_single.param_groups)
            for group in self.param_groups[i]:
                group["location"] = i  # insert location
            # optimizer_single["location"] = i
        # 每个location信息写到各自parameter group里边了
        self.k = k
        self.alpha = alpha

    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)

        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)

            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)

class Lookahead(Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k # fast update 5 times, slow update once
        self.alpha = alpha # alpha
        self.param_groups = self.optimizer.param_groups # parameters copy
        self.state = defaultdict(dict) # one state copy
        self.fast_state = self.optimizer.state # the other state
        for group in self.param_groups:
            group["counter"] = 0 # insert counter

    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha # slow = slow + (fast-slow)*0.5
            fast.data.copy_(slow)

    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, closure=None):
        loss = self.optimizer.step(closure) # 执行子优化器的更新，fast weights的更新，常规更新
        # 相当于 save在self.param_groups 的哨兵 每k次step 才update once
        for group in self.param_groups: #fast weight 更新k次，则slow weight进行一次更新
            if group["counter"] == 0:
                self.update(group) # update once

            group["counter"] += 1
            if group["counter"] >= self.k: # update k times
                group["counter"] = 0
        return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        param_group["counter"] = 0 #更新counter
        self.optimizer.add_param_group(param_group)
