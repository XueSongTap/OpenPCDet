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


# 脚本中定义了几种剪枝策略类，每个策略类都继承自BaseStrategy。这些策略用于确定在模型剪枝过程中哪些参数应该被移除。下面是每个策略的详细说明：

# BaseStrategy
# 这是一个抽象基类（ABC），定义了所有剪枝策略应遵循的接口。它要求派生类实现apply方法，该方法将根据权重和用户指定的剪枝比例来应用剪枝策略。

# RandomStrategy
# 这个类实现了随机剪枝。它随机选取一定比例或数量的参数进行移除。apply方法首先确定需要剪掉的参数数量，然后从所有参数中随机选取相应数量的参数。

# LNStrategy
# 这个类是一个基于L1范数或L2范数的剪枝策略。它支持两种模式：“amount”和“thresh”：

# "amount": 按照用户指定的剪枝比例，对参数的L1或L2范数进行排序，然后移除范数值最小的一定比例的参数。
# "thresh": 设置一个阈值，移除范数值小于这个阈值的参数。
# CustomScoreStrategy
# 这个策略允许用户用自定义的分数来决定剪枝。给定一个分数向量和一个阈值，它会保留那些分数高于阈值的参数，移除其余的参数。这个策略与LNStrategy中的“thresh”模式类似，但它可以应用于任何类型的分数，不仅限于L1或L2范数。

# L1Strategy 和 L2Strategy
# 这两个类是LNStrategy的具体实例，分别用于L1范数（绝对值之和）和L2范数（平方和的平方根）剪枝。它们默认使用“amount”模式。

# 在使用这些策略类时，你通常会创建一个策略实例，并调用它的apply方法，传入模型的权重和指定的剪枝参数。策略会返回需要剪掉的参数索引列表。然后这些索引可以用来实际修改模型的参数，例如，通过将对应的权重设置为零或从参数列表中移除它们。

# 选择哪个策略取决于你的具体需求和剪枝目标。例如，如果你想随机剪枝，你会选择RandomStrategy。如果你想基于参数的重要性剪枝，你可能会选择L1Strategy或L2Strategy。如果你有自定义的剪枝标准或分数，你可以使用CustomScoreStrategy。

"""Strategy of pruning."""
import torch
from abc import abstractclassmethod, ABC
from typing import Sequence
import random


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


class RandomStrategy(BaseStrategy):
    """Random Strategy class."""

    def apply(self, weights, amount=0.0, round_to=1) -> Sequence[int]:  # return index
        """Apply the strategy."""
        if amount <= 0:
            return []
        n = len(weights)
        n_to_prune = int(amount * n) if amount < 1.0 else amount
        n_to_prune = round_pruning_amount(n, n_to_prune, round_to)
        if n_to_prune == 0:
            return []
        indices = random.sample(list(range(n)), k=n_to_prune)
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
        self.mode = mode
        if self.mode not in ["amount", "thresh"]:
            raise ValueError("Only support \"amount\" and \"thresh\" mode")

    def apply(self, weights, amount=0.0, round_to=1, scores=None) -> Sequence[int]:  # return index
        """Apply the pruning."""
        if amount <= 0:
            return []
        n = len(weights)
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
            l1_norm /= torch.max(l1_norm)
            remained_idx = torch.nonzero(l1_norm > amount).view(-1).tolist()
            num_remained = len(remained_idx)
            # Granularity
            if num_remained % round_to > 0:
                num_remained += round_to - (num_remained % round_to)
            num_remained = min(num_remained, n)
            if num_remained == n:
                return []
            sorted_idx = torch.argsort(-l1_norm)
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


class L1Strategy(LNStrategy):
    """L1 Strategy class."""

    def __init__(self):
        """Initialize."""
        super(L1Strategy, self).__init__(p=1)


class L2Strategy(LNStrategy):
    """L2 Strategy class."""

    def __init__(self):
        """Initialize."""
        super(L2Strategy, self).__init__(p=2)
