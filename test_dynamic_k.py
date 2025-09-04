#!/usr/bin/env python3
"""
测试动态k函数的工作情况
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from algorithm.entropy_processor import AdaptiveEntropyLogitsProcessor

def test_dynamic_k_function():
    """测试动态k函数"""
    print("=== 动态k函数测试 ===")
    
    # 创建处理器实例
    processor = AdaptiveEntropyLogitsProcessor(
        k_min=5,
        k_max=50,
        sigmoid_weight=2.0,
        entropy_center=3.0,
        use_dynamic_k=True,
        smooth_threshold=0.1
    )
    
    # 测试不同熵值下的k值
    test_entropies = np.linspace(0.5, 6.0, 50)
    k_values = []
    sigmoid_values = []
    
    print("熵值 -> k值 映射测试:")
    print("熵值\t\tk值\tSigmoid输出")
    print("-" * 35)
    
    for entropy in test_entropies:
        # 直接调用动态k计算函数
        k = processor._calculate_dynamic_k(entropy)
        k_values.append(k)
        
        # 计算对应的sigmoid值
        sigmoid_input = processor.sigmoid_weight * (entropy - processor.entropy_center)
        sigmoid_output = torch.sigmoid(torch.tensor(sigmoid_input)).item()
        sigmoid_values.append(sigmoid_output)
        
        if len(k_values) % 10 == 1:  # 每10个打印一次
            print(f"{entropy:.2f}\t\t{k}\t{sigmoid_output:.3f}")
    
    # 分析统计
    analysis = processor.get_dynamic_k_analysis()
    print("\n=== 动态k函数分析 ===")
    for key, value in analysis.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    return test_entropies, k_values, sigmoid_values

def plot_dynamic_k_function(entropies, k_values, sigmoid_values):
    """绘制动态k函数图像"""
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # 绘制k值随熵值的变化
        ax1.plot(entropies, k_values, 'b-', linewidth=2, label='k_dynamic')
        ax1.axhline(y=5, color='r', linestyle='--', alpha=0.7, label='k_min=5')
        ax1.axhline(y=50, color='r', linestyle='--', alpha=0.7, label='k_max=50')
        ax1.axvline(x=3.0, color='g', linestyle='--', alpha=0.7, label='H_center=3.0')
        ax1.set_xlabel('熵值 (H)')
        ax1.set_ylabel('k值')
        ax1.set_title('动态k函数: k = k_min + (k_max - k_min) * σ(w * (H - H_center))')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 绘制sigmoid函数
        ax2.plot(entropies, sigmoid_values, 'g-', linewidth=2, label='σ(w * (H - H_center))')
        ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='sigmoid=0.5')
        ax2.axvline(x=3.0, color='g', linestyle='--', alpha=0.7, label='H_center=3.0')
        ax2.set_xlabel('熵值 (H)')
        ax2.set_ylabel('Sigmoid输出')
        ax2.set_title('Sigmoid函数输出 (w=2.0)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/banlujie/RL-NL2SQL/dynamic_k_function_test.png', dpi=150, bbox_inches='tight')
        print("\n图像已保存到: /home/banlujie/RL-NL2SQL/dynamic_k_function_test.png")
        
    except ImportError:
        print("\n注意: matplotlib未安装，跳过图像绘制")

def test_real_logits():
    """使用真实的logits测试"""
    print("\n=== 真实Logits测试 ===")
    
    processor = AdaptiveEntropyLogitsProcessor(
        k_min=5,
        k_max=50,
        sigmoid_weight=2.0,
        entropy_center=3.0,
        use_dynamic_k=True,
        smooth_threshold=0.1
    )
    
    # 创建几个不同分布特征的测试logits
    vocab_size = 1000
    test_cases = [
        ("高置信度分布", torch.randn(vocab_size) * 0.1),  # 低熵
        ("中等置信度分布", torch.randn(vocab_size) * 1.0),  # 中等熵
        ("低置信度分布", torch.randn(vocab_size) * 3.0),  # 高熵
        ("均匀分布", torch.zeros(vocab_size)),  # 最高熵
    ]
    
    for case_name, logits in test_cases:
        # 添加batch维度
        input_ids = torch.randint(0, vocab_size, (1, 10))
        scores = logits.unsqueeze(0)
        
        # 处理logits
        processed_scores = processor(input_ids, scores)
        
        # 计算原始熵
        original_entropy = processor._calculate_entropy(logits)
        
        # 计算处理后的logits中非-inf的数量（即k值）
        valid_tokens = (processed_scores[0] != float('-inf')).sum().item()
        
        print(f"{case_name}:")
        print(f"  原始熵: {original_entropy:.3f}")
        print(f"  选择的k值: {valid_tokens}")
        print()

if __name__ == "__main__":
    # 运行测试
    entropies, k_values, sigmoid_values = test_dynamic_k_function()
    
    # 绘制图像
    plot_dynamic_k_function(entropies, k_values, sigmoid_values)
    
    # 测试真实logits
    test_real_logits()
    
    print("\n动态k函数测试完成!")
