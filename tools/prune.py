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
# 这个脚本是用来对PointPillars模型进行剪枝的，PointPillars是一种用于LiDAR点云数据的3D目标检测算法。剪枝是一种优化技术，旨在通过删除模型中的一些参数（通常是权重）来减小模型大小并可能减少推理时的计算资源需求。通常剪枝后的模型推理更快，对于嵌入式和低资源环境尤其有用。

# 下面是该脚本的主要步骤和功能：

# 设置脚本参数：通过argparse库解析命令行参数，设置配置文件、输出目录、工作线程数量、剪枝阈值和加密密钥等。

# 加载配置和创建日志：通过配置文件初始化模型配置并设置日志器，日志信息将被写入文件。

# 数据加载器：构建数据加载器用于加载训练数据，并且从中获取一个输入示例用于构建依赖图。

# 加载模型：从指定路径加载PointPillars模型，并将其转移到CUDA设备上。

# 构建依赖图：使用torch_pruning库中的DependencyGraph来构建模型的依赖图，这是为了确保剪枝过程中不会破坏模型的结构。

# 剪枝执行：定义剪枝策略（例如L1策略），选择要剪枝的层（排除最后三个卷积层），并执行剪枝计划。

# 计算剪枝比例：在剪枝前后分别计算模型的参数总数，并打印剪枝比例。

# 保存剪枝模型：将剪枝后的模型保存到临时文件，然后使用提供的密钥加密该模型，并保存到输出目录。

# 状态日志：记录剪枝过程和结果的状态日志，这些信息将被写入状态文件中。

# 异常处理：如果在剪枝过程中发生异常或中断，会记录相应的状态和消息，并将异常抛出。

# 使用这个脚本时，需要指定剪枝的配置文件（包含模型结构和数据加载配置的YAML文件）、输出目录和加密密钥。脚本的主要目的是自动化剪枝过程，确保模型在剪枝后依然能保持其检测性能，并减小模型体积，方便部署。

# 剪枝通常会在模型训练完成后进行，因为剪枝有可能会影响模型的性能。因此，剪枝后的模型通常需要进行重新评估甚至微调，以确保性能满足要求。
"""Pruning script for PointPillars."""
import argparse
import datetime
import os
from pathlib import Path
import tempfile
import torch
# import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
# from nvidia_tao_pytorch.core.path_utils import expand_path
from pcdet.pruning import torch_pruning_v0 as tp
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file

from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader
from pcdet.models import (
    load_checkpoint,
    load_data_to_gpu
)
# from train_utils.train_utils import (
#     encrypt_pytorch
# )


def parse_args(args=None):
    """Argument Parser."""
    parser = argparse.ArgumentParser(description="model pruning")
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--output_dir', type=str, default=None, help='output directory.')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--pruning_thresh', "-pth", type=float, default=0.1, help='Pruning threshold')
    # parser.add_argument("--key", "-k", type=str, required=True, help="Encryption key")
    args = parser.parse_args()
    cfg_from_yaml_file(expand_path(args.cfg_file), cfg)
    return args, cfg


def prune_model():
    """Prune the PointPillars model."""
    args, cfg = parse_args()
    dist_train = False
    args.batch_size = 1
    args.epochs = cfg.train.num_epochs
    threshold = args.pruning_thresh
    if args.output_dir is None:
        if cfg.results_dir is None:
            raise OSError("Either provide results_dir in config file or provide output_dir as a CLI argument")
        else:
            args.output_dir = cfg.results_dir
    args.output_dir = expand_path(args.output_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
    # Set status logging
    status_file = os.path.join(str(output_dir), "status.json")
    # status_logging.set_status_logger(status_logging.StatusLogger(filename=status_file, append=True))
    # status_logging.get_status_logger().write(status_level=status_logging.Status.STARTED, message="Starting PointPillars Pruning")
    # -----------------------create dataloader & network & optimizer---------------------------
    train_loader = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=False,
        total_epochs=args.epochs
    )[1]
    input_dict = next(iter(train_loader))  # 获取数据加载器的一个输入样本
    load_data_to_gpu(input_dict)
    if cfg.prune.model is None:
        raise OSError("Please provide prune.model in config file")
    if not os.path.exists(expand_path(cfg.prune.model)):
        raise OSError(f"Model not found: {cfg.prune.model}")
    model = load_checkpoint(cfg.prune.model)[0]
    model = model.cuda()
    model = model.eval()
    unpruned_total_params = sum(p.numel() for p in model.parameters())
    strategy = tp.strategy.L1Strategy()  # or tp.strategy.RandomStrategy() # 使用L1策略，也可以选择其他策略如随机策略
    # strategy = tp.strategy.RandomStrategy() 
    # strategy = tp.strategy.L2Strategy()
    DG = tp.DependencyGraph()  # 创建依赖图对象
    DG.build_dependency(model, example_inputs=input_dict) # 根据模型和输入样本构建依赖图
    # conv layers
    layers = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
    # Exclude heads
    black_list = layers[-3:] # 排除最后三个卷积层
    count = 0
    for layer in layers:
        if layer in black_list:
            continue  # 跳过黑名单中的层
        # can run some algo here to generate threshold for every node
        threshold_run = threshold # 使用设定的剪枝阈值
        pruning_idxs = strategy(layer.weight, amount=threshold_run) # 获取剪枝索引
        pruning_plan = DG.get_pruning_plan(layer, tp.prune_conv, idxs=pruning_idxs) # 获取剪枝计划
        if pruning_plan is not None:
            pruning_plan.exec()  # 执行剪枝计划
        else:
            continue
        count += 1
    pruned_total_params = sum(p.numel() for p in model.parameters())
    print("Pruning ratio: {}".format(
        pruned_total_params / unpruned_total_params)
    )
    # status_logging.get_status_logger().write(
    #     status_level=status_logging.Status.RUNNING,
    #     message="Pruning ratio: {}".format(pruned_total_params / unpruned_total_params)
    # )
    # save_path = expand_path(f"{args.output_dir}/pruned_{threshold}.tlt")
    save_path = f"{output_dir}/pruned_{threshold}.pth"
    # handle, temp_file = tempfile.mkstemp()
    # os.close(handle)
    torch.save(model.state_dict(), save_path)
    # encrypt_pytorch(temp_file, save_path, args.key)
    print(f"Pruned model saved to {save_path}")
    # status_logging.get_status_logger().write(
    #     status_level=status_logging.Status.RUNNING,
    #     message=f"Pruned model saved to {save_path}"
    # )
    return model


if __name__ == "__main__":
    try:
        prune_model()
        # status_logging.get_status_logger().write(
        #     status_level=status_logging.Status.SUCCESS,
        #     message="Pruning finished successfully."
        # )
    except (KeyboardInterrupt, SystemExit):
        pass  
        # status_logging.get_status_logger().write(
        #     message="Pruning was interrupted",
        #     verbosity_level=status_logging.Verbosity.INFO,
        #     status_level=status_logging.Status.FAILURE
        # )
    except Exception as e:
        print(str(e))
        # status_logging.get_status_logger().write(
        #     message=str(e),
        #     status_level=status_logging.Status.FAILURE
        # )
        raise e
