# Credit Model 在 PPO 中的使用指南

## 概述

这个指南将帮助你将训练好的 Credit Model 集成到 PPO 训练中，实现细粒度的信用分配和奖励塑形。

## 前提条件

1. 你已经完成了 Credit Model 的训练（使用 `train_cm.py`）
2. 你有一个训练好的 SFT 模型（Actor）
3. 你有一个训练好的 Reward Model
4. 你的数据集已经准备好

## 集成方法

我们提供了两种集成方法：

### 方法一：完整的 Ray Actor 集成（推荐）

这是最完整的集成方式，提供最佳的性能和可扩展性。

#### 1. 使用修改后的 PPO 训练脚本

```bash
# 运行集成 Credit Model 的 PPO 训练
bash scripts/ppo_with_credit.sh
```

#### 2. 自定义参数

你可以修改 `scripts/ppo_with_credit.sh` 中的以下参数：

```bash
# Credit Model 相关参数
--credit_pretrain /path/to/your/credit/model     # Credit Model 路径
--credit_num_nodes 1                             # Credit Model 节点数
--credit_num_gpus_per_node 2                     # 每个节点的 GPU 数
--credit_num_outputs 8                           # Credit Model 输出维度
--credit_weight 0.1                              # Credit 奖励权重
--enable_credit_shaping                          # 启用信用塑形
--credit_aggregation "weighted_sum"              # 信用聚合方法
```

#### 3. 信用聚合方法

- `"weighted_sum"`: 根据位置权重加权求和（推荐）
- `"mean"`: 取平均值
- `"max"`: 取最大值

### 方法二：简化集成（快速开始）

如果你想要更简单的集成方式，可以直接修改奖励函数：

#### 1. 在你的 PPO 训练脚本中添加环境变量

```bash
export CREDIT_MODEL_PATH="/path/to/your/credit/model"
export CREDIT_WEIGHT="0.1"
export CREDIT_AGGREGATION="mean"
```

#### 2. 修改 reward 函数调用

```python
# 在你的训练代码中
reward_data = reward_func(
    queries, prompts, labels,
    credit_model_path=os.getenv("CREDIT_MODEL_PATH"),
    credit_weight=float(os.getenv("CREDIT_WEIGHT", "0.1")),
    credit_aggregation=os.getenv("CREDIT_AGGREGATION", "mean")
)
```

## 配置说明

### Credit Weight（信用权重）

- 范围：0.0 - 1.0
- 推荐值：0.1 - 0.3
- 作用：控制 Credit Model 奖励在总奖励中的比重

```python
# 最终奖励 = 原始奖励 + credit_weight * credit_奖励
enhanced_reward = original_reward + 0.1 * credit_reward
```

### Credit Aggregation（信用聚合）

不同的聚合方法适用于不同的场景：

1. **weighted_sum**: 适合强调序列中不同位置的重要性
2. **mean**: 适合平均考虑所有位置的贡献
3. **max**: 适合关注最重要的单个位置

### 资源配置

根据你的硬件情况调整以下参数：

```bash
# 单机多卡配置
--credit_num_nodes 1
--credit_num_gpus_per_node 2

# 多机配置
--credit_num_nodes 2
--credit_num_gpus_per_node 4
```

## 监控和调试

### 1. Wandb 日志

训练过程中会记录以下 Credit 相关指标：

- `mean_credit_reward`: 平均信用奖励
- `max_credit_reward`: 最大信用奖励
- `min_credit_reward`: 最小信用奖励
- `credit_weight`: 当前信用权重

### 2. 调试信息

启用详细日志来监控 Credit Model 的工作状态：

```python
# 在训练脚本中添加
import logging
logging.basicConfig(level=logging.INFO)
```

### 3. 性能监控

监控以下指标来确保 Credit Model 正常工作：

- Credit Model 推理时间
- 内存使用情况
- GPU 利用率

## 超参数调优建议

### 1. Credit Weight 调优

从小的权重开始，逐步增加：

```python
# 阶段1: 基线测试
credit_weight = 0.0

# 阶段2: 小权重测试
credit_weight = 0.05

# 阶段3: 标准权重
credit_weight = 0.1

# 阶段4: 更大权重（如果效果好）
credit_weight = 0.2
```

### 2. 聚合方法选择

根据你的任务特点选择：

- **SQL 生成任务**: 推荐 `weighted_sum`，因为不同子句的重要性不同
- **代码生成任务**: 推荐 `mean`，平等对待每个代码片段
- **摘要任务**: 推荐 `max`，关注最重要的信息

## 故障排除

### 1. Credit Model 加载失败

```bash
# 检查模型文件
ls -la /path/to/credit/model/
# 应该包含：
# - pytorch_model.bin 或 model.safetensors
# - config.json
# - regression_head.bin
# - credit_model_meta.json
```

### 2. GPU 内存不足

```bash
# 减少批次大小
--micro_train_batch_size 4
--micro_rollout_batch_size 8

# 启用梯度检查点
--gradient_checkpointing

# 减少 Credit Model GPU 数量
--credit_num_gpus_per_node 1
```

### 3. 信用奖励异常

如果信用奖励值异常，检查：

- Credit Model 是否正确训练
- 输入序列格式是否正确
- 聚合方法是否合适

## 高级用法

### 1. 动态调整 Credit Weight

```python
# 在训练过程中动态调整权重
def dynamic_credit_weight(epoch, performance):
    if performance > threshold:
        return min(0.3, initial_weight * 1.5)
    else:
        return max(0.05, initial_weight * 0.5)
```

### 2. 多个 Credit Model 集成

```python
# 使用多个专门的 Credit Model
credit_models = [
    "syntax_credit_model",    # 语法信用
    "semantic_credit_model",  # 语义信用
    "execution_credit_model"  # 执行信用
]
```

### 3. Credit-aware 采样

```python
# 基于信用分数进行采样
def credit_aware_sampling(sequences, credit_scores, temperature=0.8):
    # 根据信用分数调整采样概率
    pass
```

## 性能优化

### 1. 批量处理

```python
# 启用批量信用计算
credit_scores = credit_integrator.compute_credit_scores_batch(
    sequences, batch_size=32
)
```

### 2. 异步处理

```python
# 异步计算信用分数
import asyncio

async def async_credit_computation(sequences):
    # 异步计算逻辑
    pass
```

### 3. 缓存机制

```python
# 缓存常见序列的信用分数
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_credit_computation(sequence_hash):
    # 缓存计算结果
    pass
```

## 实验建议

### 1. 对照实验

进行以下对照实验来验证 Credit Model 的效果：

1. **Baseline**: 不使用 Credit Model 的 PPO 训练
2. **Credit-Enhanced**: 使用 Credit Model 的 PPO 训练
3. **Ablation**: 不同 credit_weight 值的对比

### 2. 评估指标

关注以下指标：

- **任务性能**: SQL 执行准确率、语法正确性等
- **训练效率**: 收敛速度、样本效率
- **奖励质量**: 奖励分布、方差等

### 3. 分析工具

使用以下工具分析 Credit Model 的影响：

```python
# 信用分数分布分析
import matplotlib.pyplot as plt

def analyze_credit_distribution(credit_scores):
    plt.hist(credit_scores, bins=50)
    plt.xlabel('Credit Score')
    plt.ylabel('Frequency')
    plt.title('Credit Score Distribution')
    plt.show()

# 奖励变化分析
def analyze_reward_change(original_rewards, enhanced_rewards):
    improvement = [e - o for e, o in zip(enhanced_rewards, original_rewards)]
    print(f"Mean improvement: {np.mean(improvement):.4f}")
    print(f"Std improvement: {np.std(improvement):.4f}")
```

## 常见问题

**Q: Credit Model 会显著增加训练时间吗？**
A: 增加约 10-20% 的训练时间，但通常能带来更好的样本效率。

**Q: 如何选择合适的 credit_weight？**
A: 从 0.1 开始，根据验证集性能调整。一般不超过 0.3。

**Q: Credit Model 和 Reward Model 有什么区别？**
A: Reward Model 给整个序列打分，Credit Model 给序列中的每个部分分配信用。

**Q: 可以在推理时使用 Credit Model 吗？**
A: 可以，但主要用于训练时的奖励塑形。推理时可以用于beam search的重排序。

希望这个指南能帮助你成功地将 Credit Model 集成到 PPO 训练中！
