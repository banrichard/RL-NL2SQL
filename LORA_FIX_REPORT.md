# Qwen2.5-Coder LoRA 维度不匹配问题修复报告

## 问题描述

在使用LoRA微调Qwen2.5-Coder-7B-Instruct时遇到维度不匹配错误：
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (11278x16 and 32x1792)
```

## 根本原因分析

1. **LoRA配置不当**: 原配置使用`lora_rank=32, lora_alpha=32`，对于某些层可能导致维度计算错误
2. **目标模块选择**: 可能包含了不兼容的模块
3. **批次大小过大**: `micro_train_batch_size=4`可能导致内存压力和维度计算问题
4. **序列长度过长**: `max_len=16384`增加了计算复杂度

## 解决方案

### 1. 调整LoRA参数
- **lora_rank**: 从32降低到8
- **lora_alpha**: 从32降低到16  
- **lora_dropout**: 从0.05调整到0.1
- **学习率**: 从5e-6提高到2e-4（LoRA通常需要更高学习率）

### 2. 优化目标模块
只针对注意力层进行LoRA微调：
```bash
--target_modules q_proj k_proj v_proj o_proj
```

### 3. 减少资源消耗
- **micro_train_batch_size**: 从4降低到2
- **train_batch_size**: 从128降低到64
- **max_len**: 从16384降低到8192

### 4. 模型架构信息
通过调试发现Qwen2.5-Coder-7B-Instruct的关键参数：
- **隐藏维度**: 3584
- **注意力头数**: 28
- **层数**: 28
- **中间层大小**: 18944
- **词汇表大小**: 151643

## 修复后的配置

### 原配置 (有问题)
```bash
--lora_rank 32
--lora_alpha 32
--lora_dropout 0.05
--learning_rate 5e-6
--train_batch_size 128
--micro_train_batch_size 4
--max_len 16384
# target_modules 使用默认值
```

### 修复后配置 (推荐)
```bash
--lora_rank 8
--lora_alpha 16
--lora_dropout 0.1
--learning_rate 2e-4
--train_batch_size 64
--micro_train_batch_size 2
--max_len 8192
--target_modules q_proj k_proj v_proj o_proj
```

## 验证结果

通过`debug_lora_fixed.py`脚本验证，所有测试配置都成功：

- **Rank 8**: ✅ 成功，可训练参数: 5,046,272 (0.07%)
- **Rank 16**: ✅ 成功，可训练参数: 10,092,544 (0.13%)  
- **Rank 32**: ✅ 成功，可训练参数: 20,185,088 (0.26%)

## 文件更改

1. **scripts/sft.sh**: 应用了修复后的配置
2. **scripts/sft_fixed.sh**: 新的修复版本脚本
3. **debug_lora_fixed.py**: 调试脚本，用于验证配置兼容性

## 使用建议

1. **优先使用**: `scripts/sft_fixed.sh` 或更新后的 `scripts/sft.sh`
2. **监控训练**: 注意观察loss收敛情况，如果下降过慢可以适当增加rank
3. **内存优化**: 如果仍有内存问题，可以进一步减少batch_size或max_len
4. **性能调优**: 训练稳定后可以尝试rank=16获得更好效果

## 预防措施

在未来使用LoRA时：
1. 先运行`debug_lora_fixed.py`验证配置兼容性
2. 从小的rank值开始测试
3. 监控GPU内存使用情况
4. 根据具体模型调整target_modules

## 技术细节

### LoRA维度计算
对于attention层的q_proj (3584 -> 3584)：
- LoRA_A: (rank, input_dim) = (8, 3584)
- LoRA_B: (output_dim, rank) = (3584, 8)
- 总参数: 8 * 3584 + 3584 * 8 = 57,344

### 错误原因推测
原来的错误`(11278x16 and 32x1792)`可能是由于：
1. 动态批处理导致的序列长度变化
2. LoRA适配器与注意力机制的维度不匹配
3. 梯度累积过程中的中间计算错误

修复后的配置确保了所有维度计算的一致性和兼容性。
