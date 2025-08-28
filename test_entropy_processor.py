"""
测试AdaptiveEntropyProcessor的功能
"""

import torch
import sys
import os
sys.path.append('/home/banlujie/RL-NL2SQL')

from algorithm.entropy_processor import AdaptiveEntropyLogitsProcessor, create_adaptive_entropy_processor
from transformers import AutoTokenizer
import numpy as np


def test_entropy_calculation():
    """测试熵计算功能"""
    print("=== 测试熵计算 ===")
    
    # 创建处理器
    processor = AdaptiveEntropyLogitsProcessor(
        low_entropy_threshold=2.0,
        high_entropy_threshold=4.0,
        low_k=5,
        high_k=50
    )
    
    # 测试不同分布的熵
    vocab_size = 1000
    
    # 1. 高置信度分布（低熵）
    sharp_logits = torch.zeros(vocab_size)
    sharp_logits[0] = 10.0  # 非常高的得分
    sharp_logits[1:10] = 1.0  # 其他一些token有小得分
    
    sharp_entropy = processor._calculate_entropy(sharp_logits)
    print(f"尖锐分布熵值: {sharp_entropy:.3f}")
    
    # 2. 低置信度分布（高熵）
    flat_logits = torch.randn(vocab_size) * 0.1  # 非常平坦的分布
    flat_entropy = processor._calculate_entropy(flat_logits)
    print(f"平坦分布熵值: {flat_entropy:.3f}")
    
    # 3. 中等分布
    medium_logits = torch.zeros(vocab_size)
    medium_logits[:100] = torch.randn(100)  # 前100个token有随机得分
    medium_entropy = processor._calculate_entropy(medium_logits)
    print(f"中等分布熵值: {medium_entropy:.3f}")
    
    return sharp_entropy, flat_entropy, medium_entropy


def test_adaptive_k_selection():
    """测试自适应k值选择"""
    print("\n=== 测试自适应k值选择 ===")
    
    processor = AdaptiveEntropyLogitsProcessor(
        low_entropy_threshold=2.0,
        high_entropy_threshold=4.0,
        low_k=5,
        high_k=50
    )
    
    # 测试不同熵值对应的k值
    test_entropies = [1.0, 2.5, 3.0, 4.5, 6.0]
    
    for entropy in test_entropies:
        k = processor._adaptive_k_selection(entropy)
        print(f"熵值: {entropy:.1f} -> k值: {k}")
    
    return True


def test_topk_filtering():
    """测试Top-K过滤功能"""
    print("\n=== 测试Top-K过滤 ===")
    
    processor = AdaptiveEntropyLogitsProcessor()
    
    # 创建测试logits
    vocab_size = 100
    logits = torch.randn(vocab_size)
    
    # 测试不同k值的过滤效果
    for k in [5, 10, 20]:
        filtered_logits = processor._apply_topk_filter(logits, k)
        
        # 统计非-inf的位置数量
        valid_positions = torch.sum(~torch.isinf(filtered_logits)).item()
        print(f"k={k}: 保留了 {valid_positions} 个位置")
        
        # 检查top-k是否正确
        original_topk_values, original_topk_indices = torch.topk(logits, k)
        filtered_topk_values = filtered_logits[original_topk_indices]
        
        # 验证top-k值是否保持不变
        assert torch.allclose(original_topk_values, filtered_topk_values), f"Top-{k}过滤失败"
    
    print("Top-K过滤测试通过")
    return True


def test_content_filtering():
    """测试内容过滤功能"""
    print("\n=== 测试内容过滤 ===")
    
    # 加载tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except:
        print("无法加载tokenizer，跳过内容过滤测试")
        return True
    
    processor = AdaptiveEntropyLogitsProcessor(
        enable_content_filtering=True,
        tokenizer=tokenizer
    )
    
    # 测试不同的文本内容
    test_texts = [
        "让我想想这个问题",  # thinking过程，应该被过滤
        "<thinking>我需要分析一下</thinking>",  # thinking标签，应该被过滤
        "<answer>```sql SELECT name FROM",  # SQL生成，应该被处理
        "SELECT * FROM users WHERE",  # 纯SQL，应该被处理
        "首先分析表结构",  # 分析过程，应该被过滤
    ]
    
    for text in test_texts:
        # 编码文本
        input_ids = tokenizer.encode(text, return_tensors="pt")
        
        # 测试是否应该处理
        should_process = processor._should_process_current_step(input_ids)
        print(f"文本: '{text[:30]}...' -> 是否处理: {should_process}")
    
    return True


def test_full_pipeline():
    """测试完整的处理流程"""
    print("\n=== 测试完整处理流程 ===")
    
    # 创建tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except:
        print("无法加载tokenizer，使用简化测试")
        tokenizer = None
    
    processor = create_adaptive_entropy_processor(
        low_entropy_threshold=2.0,
        high_entropy_threshold=4.0,
        low_k=5,
        high_k=50,
        enable_content_filtering=True,
        tokenizer=tokenizer
    )
    
    # 模拟生成过程
    batch_size = 2
    vocab_size = 1000
    seq_len = 50
    
    # 创建输入
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 创建不同置信度的logits
    # 第一个样本：高置信度（低熵）
    sharp_scores = torch.zeros(batch_size, vocab_size)
    sharp_scores[0, 0] = 10.0
    sharp_scores[0, 1:10] = 1.0
    
    # 第二个样本：低置信度（高熵）  
    sharp_scores[1] = torch.randn(vocab_size) * 0.1
    
    # 处理logits
    processed_scores = processor(input_ids, sharp_scores)
    
    # 检查处理结果
    print(f"原始scores形状: {sharp_scores.shape}")
    print(f"处理后scores形状: {processed_scores.shape}")
    
    # 统计每个样本的有效位置
    for i in range(batch_size):
        valid_positions = torch.sum(~torch.isinf(processed_scores[i])).item()
        print(f"样本{i}: 有效位置数 = {valid_positions}")
    
    # 显示统计信息
    stats = processor.get_stats()
    print(f"\n处理器统计信息:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    return True


def test_integration_with_ppo():
    """测试与PPO训练的集成"""
    print("\n=== 测试PPO集成 ===")
    
    # 创建处理器
    processor = create_adaptive_entropy_processor(
        low_entropy_threshold=1.5,
        high_entropy_threshold=3.5,
        low_k=3,
        high_k=30,
        enable_content_filtering=False  # PPO训练时可能不需要内容过滤
    )
    
    # 模拟PPO训练中的批次
    batch_size = 4
    vocab_size = 32000  # 更真实的词汇表大小
    seq_len = 128
    
    print(f"模拟PPO批次: batch_size={batch_size}, vocab_size={vocab_size}")
    
    # 创建多样化的logits分布
    scores = torch.zeros(batch_size, vocab_size)
    
    # 样本0: 极低熵（非常确定）
    scores[0, 0] = 15.0
    scores[0, 1] = 2.0
    
    # 样本1: 低熵（比较确定）
    scores[1, :20] = torch.randn(20) + 3.0
    
    # 样本2: 中等熵
    scores[2, :100] = torch.randn(100)
    
    # 样本3: 高熵（很不确定）
    scores[3] = torch.randn(vocab_size) * 0.2
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 处理多个批次
    total_processed = 0
    for batch_idx in range(5):
        processed_scores = processor(input_ids, scores)
        total_processed += 1
        
        if batch_idx % 2 == 0:
            # 打印中间统计
            stats = processor.get_stats()
            print(f"批次 {batch_idx}: 总调用 {stats['total_calls']}, 平均k值 {stats['avg_k_used']:.1f}")
    
    # 最终统计
    final_stats = processor.get_stats()
    print(f"\n最终统计信息:")
    print(f"  总调用次数: {final_stats['total_calls']}")
    print(f"  低熵调用: {final_stats['low_entropy_calls']} ({final_stats.get('low_entropy_ratio', 0):.2%})")
    print(f"  高熵调用: {final_stats['high_entropy_calls']} ({final_stats.get('high_entropy_ratio', 0):.2%})")
    print(f"  中等熵调用: {final_stats['medium_entropy_calls']} ({final_stats.get('medium_entropy_ratio', 0):.2%})")
    print(f"  平均熵值: {final_stats['avg_entropy']:.3f}")
    print(f"  平均k值: {final_stats['avg_k_used']:.1f}")
    
    return True


def main():
    """运行所有测试"""
    print("开始测试AdaptiveEntropyProcessor...")
    
    try:
        # 基础功能测试
        test_entropy_calculation()
        test_adaptive_k_selection() 
        test_topk_filtering()
        test_content_filtering()
        test_full_pipeline()
        test_integration_with_ppo()
        
        print("\n✅ 所有测试通过！")
        print("\n🔥 AdaptiveEntropyProcessor已准备好用于PPO训练")
        
        # 使用建议
        print("\n📋 使用建议:")
        print("1. 在PPO训练中添加logits_processor参数")
        print("2. 调整low_k和high_k值以平衡探索与利用")
        print("3. 根据任务特点调整熵阈值")
        print("4. 监控统计信息以优化参数")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()
