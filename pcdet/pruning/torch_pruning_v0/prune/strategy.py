# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Strategy of pruning."""
import torch
from abc import abstractclassmethod, ABC
from typing import Sequence
import random

# 定义一个函数，用来确保剪枝后的参数数量是某个整数的倍数
def round_pruning_amount(total_parameters, n_to_prune, round_to):
    """round the parameter amount after pruning to an integer multiple of `round_to`.
    """
    round_to = int(round_to)
    if round_to <= 1:
        return n_to_prune
    after_pruning = total_parameters - n_to_prune
    compensation = after_pruning % round_to
    # round to the nearest (round_to * N)
    # avoid negative n_to_prune
    if (compensation < round_to // 2 and after_pruning > round_to) or round_to > n_to_prune:
        n_to_prune = n_to_prune + compensation  # floor
    else:
        n_to_prune = n_to_prune - round_to + compensation  # ceiling
    return n_to_prune

# 声明一个抽象基类，定义剪枝策略的基础结构
class BaseStrategy(ABC):
    """Base Strategy class."""

    def __call__(self, *args, **kwargs):
        """Call method."""
        return self.apply(*args, **kwargs)

    @abstractclassmethod
    def apply(cls, weights, amount=0.0, round_to=1) -> Sequence[int]:  # return index
        """ Apply the strategy on weights with user specified pruning percentage.

        Parameters:
            weights (torch.Parameter): weights to be pruned.
            amount (Callable): the percentage of weights to be pruned (amount<1.0) or the amount of weights to be pruned (amount>=1.0)
            round_to (int): the number to which the number of pruned channels is rounded.
        """
        raise NotImplementedError

# 定义一个随机剪枝策略
# andomStrategy 类的关键部分是其对 random.sample 函数的使用，该函数从权重列表中随机选择出 n_to_prune 个索引。这些索引代表将要被剪枝的权重。这种策略并不关注权重的大小或其它特性，纯粹是随机选择，这可以在不考虑权重重要性的情况下提供一个基准。
class RandomStrategy(BaseStrategy):
    """Random Strategy class."""

    def apply(self, weights, amount=0.0, round_to=1) -> Sequence[int]:  # return index
        """Apply the strategy."""
        if amount <= 0:
            return []
        n = len(weights)
        # 计算出需要剪除的权重数量
        n_to_prune = int(amount * n) if amount < 1.0 else amount
        # 调用 round_pruning_amount 函数将要剪除的权重数量调整为 round_to 参数的整数倍
        n_to_prune = round_pruning_amount(n, n_to_prune, round_to)
        if n_to_prune == 0:
            return []
        # 这个函数返回的是一个从0到n-1的整数列表，其中包含 n_to_prune 个随机、唯一的元素
        indices = random.sample(list(range(n)), k=n_to_prune)
        # 返回被选中要剪枝的权重索引列表
        return indices


class LNStrategy(BaseStrategy):
    """LN magnitude based pruning strategy.

    Two mode of LN-magnitude-based (L1 or L2) pruning startegy are provided through this class:
    - "amount": The pruning algorithm in original Torch-pruning. "amount" means the ratio of
    number of filters to be pruned to the total number of filters. Suppose the total number of
    filters is N, then the number of filters to be pruned is N * amount. The filters are sorted
    along the LN-magnitude of each filter and the smallest N* amount filters will be pruned.
    - "thresh": The pruning algorithm in tao-keras. The filter with smaller LN-magnitude than
    a threshold will be pruned.

    Common tricks:
    - granularity. The pruned number of filters will be divisible by the granularity number.
    """

    def __init__(self, p, mode="amount"):
        """Constructor for LNS strategy."""
        self.p = p
        self.mode = mode # mode 参数确定剪枝的方式，可以是 "amount" 或 "thresh"
        if self.mode not in ["amount", "thresh"]:
            raise ValueError("Only support \"amount\" and \"thresh\" mode")

    def apply(self, weights, amount=0.0, round_to=1, scores=None) -> Sequence[int]:  # return index
        """Apply the pruning."""
        # 如果剪枝比例小于等于 0，则不进行剪枝，返回空列表
        if amount <= 0:
            return []
        # 获取权重的数量（例如，卷积层的滤波器数量）
        n = len(weights)
        # 如果没有给出 scores 参数，则计算给定 p 范数
        if scores is None:
            l1_norm = torch.norm(weights.view(n, -1), p=self.p, dim=1)
        else:
            l1_norm = scores

        if self.mode == "amount":
            n_to_prune = int(amount * n) if amount < 1.0 else amount
            n_to_prune = round_pruning_amount(n, n_to_prune, round_to)
            if n_to_prune == 0:
                return []
            threshold = torch.kthvalue(l1_norm, k=n_to_prune).values
            indices = torch.nonzero(l1_norm <= threshold).view(-1).tolist()
        elif self.mode == "thresh":
            # Thresh is the strategy in tao-tf
            # 如果模式是 "thresh"，首先规范化范数值
            l1_norm /= torch.max(l1_norm)
            # 找到所有大于 amount 的范数值的索引
            remained_idx = torch.nonzero(l1_norm > amount).view(-1).tolist()
            # 计算剩余的权重数量
            num_remained = len(remained_idx)
            # Granularity
            # 如果剩余的数量不是 round_to 的倍数，向上取到最近的倍数
            if num_remained % round_to > 0:
                num_remained += round_to - (num_remained % round_to)
            num_remained = min(num_remained, n) # 限制剩余数量不超过总权重数量
            if num_remained == n:
                return []
            sorted_idx = torch.argsort(-l1_norm)  # 对范数值进行排序，保留 num_remained 个最大的权重
            indices = torch.sort(sorted_idx[num_remained:])[0].view(-1).tolist()

        return indices


class CustomScoreStrategy(BaseStrategy):
    """Custom Score Strategy.

    A helper class to execute sorting and filtering with any pruning score.

    common trick:
    - granularity. The pruned number of filters will be divisible by the granularity number.
    """

    def apply(self, scores, thresh=0.0, round_to=1) -> Sequence[int]:
        """Apply the pruning."""
        if thresh <= 0:
            return []
        n = len(scores)
        remained_idx = torch.nonzero(scores > thresh).view(-1).tolist()
        num_remained = len(remained_idx)
        # Granularity
        if num_remained % round_to > 0:
            num_remained += round_to - (num_remained % round_to)
        # keep the min idxs
        num_remained = max(num_remained, round_to)
        num_remained = min(num_remained, n)
        if num_remained == n:
            return []
        sorted_idx = torch.argsort(-scores)
        indices = torch.sort(sorted_idx[num_remained:])[0].view(-1).tolist()

        return indices

# 实现了特定于L1范数的剪枝策略
class L1Strategy(LNStrategy):
    """L1 Strategy class."""

    def __init__(self):
        """Initialize."""
        super(L1Strategy, self).__init__(p=1)

# 实现了特定于L2范数的剪枝策略
class L2Strategy(LNStrategy):
    """L2 Strategy class."""

    def __init__(self):
        """Initialize."""
        super(L2Strategy, self).__init__(p=2)
