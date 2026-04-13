# SegFormer 重训需求 —— 6 类版本

## 背景

当前仓库里 `segmentation/model/segformer_baseline.pth` 是 3 类模型（obstacle/water/sky），
下游正在开发 risk-aware warning 和 SORT tracking 模块，发现 3 类方案有两个问题：

1. **所有 non-water/non-sky 像素都归成一个 obstacle 类** → 岸线和独木舟/浮标在训练阶段就
   被合并，模型输出一个横跨 horizon 的大连通域，下游无法拿到独立的 instance。
2. **无法区分静态障碍（岸线）和动态物体（船/浮标）** → SORT tracking 没法只跟踪动态物体。

经讨论，决定重训一个 6 类版本，显式保留 static obstacle 并把 LaRS 最主要的两种动态类
（boat、buoy）单独拎出来。其余动态类合并到 other，因为样本量太少（单类 < 5%），
单独建类泛化风险高。

---

## 类别定义

| 新 class id | 新类别名 | LaRS 原始来源 | LaRS 占比 | 备注 |
|:-:|---|---|:-:|---|
| 0 | `water` | water (stuff) | — | 不变 |
| 1 | `sky` | sky (stuff) | — | 不变 |
| 2 | `static_obstacle` | static obstacle (stuff) | — | 岸边、树林、建筑等固定障碍 |
| 3 | `boat` | boat + row_boat | **73.6 %** | 两者视觉相近，合并为"船类"超类 |
| 4 | `buoy` | buoy | 12.2 % | 浮标 |
| 5 | `other_dynamic` | swimmer + animal + paddle_board + float + other | ~14.2 % | 其余所有 dynamic thing 类 |

**总计 6 类**。

### 标签映射表（dataloader 里写死）

假设 LaRS panoptic 原始 stuff id 和 thing 类别名可以读到，大致是：

```python
LARS_TO_NEW = {
    # stuff
    "water":           0,
    "sky":             1,
    "static_obstacle": 2,
    # things
    "boat":            3,
    "row_boat":        3,
    "buoy":            4,
    "swimmer":         5,
    "animal":          5,
    "paddle_board":    5,
    "float":           5,
    "other":           5,
}
```

> 具体键名以 LaRS 官方 meta 为准，上面只是示意。请照着 LaRS 的 dataset meta 做完整映射。

---

## 训练要求

### 1. Backbone 保持不变
**`nvidia/mit-b0`**，和当前 3 类模型一致。这样下游 `segmentation/_pipeline.py` 只需要改
`num_labels=3` → `num_labels=6` 一个地方，加载代码完全复用。如果你改成更大的
backbone（mit-b1/b2 等），请在交付时注明，我这边会相应调整。

### 2. Loss：必须加权
类别严重不均衡（boat 占 73.6%，buoy 12.2%，other 14.2%）。建议两种之一：
- **Weighted Cross-Entropy**：权重用 `1 / sqrt(class_freq)` 或 inverse-frequency 都行
- **Focal Loss**（`gamma=2`）：更鲁棒

否则模型很容易退化成"所有动态物全预测为 boat"。

### 3. Eval split 保持和之前 3 类实验一致
Week 2 的 Table I（DeepLab vs SegFormer）用的是 LaRS validation split，
**请继续用同一个 split**，方便我们在报告里做横向对比。

### 4. 需要在 eval 时同时汇报两组指标

**Per-class 6 类指标**：
- 每类 IoU
- 每类 recall / precision
- 6-class mean IoU

**合并回 3 类后的 super-obstacle 指标（和报告 Table I 对齐）**：
- 把 `{static_obstacle, boat, buoy, other_dynamic}` 合并成一个 super-obstacle 类
- 和 `water`、`sky` 一起算 mIoU、Obstacle Recall、Obstacle Precision
- 目标：Obstacle Recall 不要比 3 类 baseline 的 96.51% 掉太多（<1 个点都算可以接受）

这组对比是报告里证明"细分没有牺牲 recall"的关键数据，别忘了跑。

### 5. Inference latency 也要测
同样在 `infer_max_side=552` 的分辨率下测，方便和 Table I 的 29.74 ms 横向对比。

---

## 输出格式要求

**必须能被当前 `segmentation/_pipeline.py` 直接加载**，具体约束：

1. **文件名**：`segformer_6class.pth`（或其他名字都行，交付时告诉我就改配置）
2. **保存内容**：`state_dict`，以下三种 torch.save 格式都支持：
   ```python
   torch.save(model.state_dict(), "segformer_6class.pth")
   # 或
   torch.save({"state_dict": model.state_dict()}, ...)
   # 或
   torch.save({"model_state_dict": model.state_dict()}, ...)
   ```
3. **模型结构**：必须是 `SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=6)` 能加载的权重。
4. **不要 pickle 整个 model 对象**，只保存 state_dict。

### 给下游的 class id 约定

输出的 seg mask 里，像素值就是 0-5，对应上面表格的 class id。这个顺序是**硬约定**，
我会在 `fusion/obstacle.py` 里按这个 id 来区分 static obstacle / boat / buoy / other_dynamic。
**一旦确定不要再改**，否则下游的 class weight 查表会错位。

---

## 交付物清单

1. ✅ `segformer_6class.pth`（权重文件）
2. ✅ 训练脚本（方便后续复现）—— 至少给 hyperparams（lr、batch、epoch、loss 类型、weight decay）
3. ✅ Eval 结果两份表：
   - 6 类 per-class 指标
   - 合并为 3 类 super-obstacle 后的 Table I 对齐版
4. ✅ 平均 inference latency（ms/frame，`infer_max_side=552`）
5. ⚠️ 如果你改了 backbone，或者训练时发现需要改 `SegformerImageProcessor` 的预处理参数，
   请在交付时说明，我需要同步修改 `segmentation/_pipeline.py`。

---

## 下游如何消费（给你参考，不需要你做）

我这边会在新权重交付后做两件小事：
1. `segmentation/_pipeline.py` 里把 `num_labels=3` 改成 `num_labels=6`
2. `fusion/obstacle.py` 里把 `OBSTACLE_CLASS = 0` 改成一组动态类 id，并加 `CLASS_WEIGHTS`
   查表。Week 7 的 SORT tracking 只对 `{3, 4, 5}` 这三类跑，`static_obstacle (2)` 走
   "shoreline risk" 独立路径。

---

## 时间建议

能在 **Week 6 周末之前**拿到新模型最好，这样 Week 7 tracking 模块可以直接在新模型上开发。
如果训练慢，可以先交付一个初步能用的版本（不用调到最优），后续再迭代。
