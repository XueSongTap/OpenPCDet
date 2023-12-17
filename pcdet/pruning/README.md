# Usage of new pruning API:
导入torch_pruning库并设置剪枝比例为20%（这意味着将移除总权重的20%）。

模型被设置为评估模式，这通常在进行推理之前需要设置。

计算剪枝之前模型中的参数数量。

构建依赖图（DependencyGraph），这是为了保证剪枝过程不会破坏模型的结构完整性。

通过在一个虚拟输入上运行模型来构建这个依赖图。

确定不应该被剪枝的层（在这个例子中是model.model[-1]中的所有层）。

计算全局阈值，这是决定哪些权重应该被剪掉的一个标准。

对于使用shortcut连接的YOLOv5模型中的C3模块，手动确定哪些层应该被合并在一起进行剪枝，这是为了保持模型结构的一致性。

执行execute_custom_score_prune函数来按照全局阈值进行剪枝，使用包括依赖图、合并集合、排除的层以及剪枝的粒度。

计算剪枝后的模型参数数量。

打印剪枝前后的参数数量以展示剪枝效果。

将模型移回到设备（CPU或GPU）以进行进一步的操作或推理。
```python
    #################################################
    # Torch Pruning (Begin)
    #################################################
    import torch_pruning as tp
    prune_ratio = 0.2
    granularity = 8
    model.eval()
    num_params_before_pruning = tp.utils.count_params( model )
    # 1. build dependency graph
    DG = tp.DependencyGraph()
    out = model(torch.randn([1,3, input_size, input_size]).to(device))
    DG.build_dependency(model, example_inputs=torch.randn([1,3, input_size, input_size]))
    excluded_layers = list(model.model[-1].modules())
    print(excluded_layers)

    # 2. get global threshold
    global_thresh, module2scores = tp.utils.get_global_thresh(model, prune_ratio=prune_ratio)
    # Hard code the way to find the shortcut connection in YOLOV5 module
    from models.common import C3
    merged_sets = {}
    for name, m in model.named_modules():
        if isinstance(m, C3):
            if m.shortcut:
                merged_sets[m.cv1.conv] = set()
                for btnk in m.m:
                    merged_sets[m.cv1.conv].add(btnk.cv2.conv)

    # 3. Execute pruning
    tp.utils.execute_custom_score_prune(model,
                                        global_thresh=global_thresh,
                                        module2scores=module2scores,
                                        dep_graph=DG,
                                        granularity=granularity,
                                        excluded_layers=excluded_layers,
                                        merged_sets=merged_sets)
    num_params_after_pruning = tp.utils.count_params( model )
    print( "  Params: %s => %s"%( num_params_before_pruning, num_params_after_pruning))
    # exit(0)
    #################################################
    # Torch Pruning (End)
    #################################################
    model = model.to(device)
```
