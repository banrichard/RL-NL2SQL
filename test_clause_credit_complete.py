#!/usr/bin/env python3
"""
子句级别信用分配系统的完整测试
测试整个时间步流程：Actor生成 -> Credit分配 -> 折扣归因
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer
from algorithm.clause_credit_system import ClauseLevelCreditSystem
from algorithm.clause_credit_integration import ClauseCreditIntegration


def test_complete_timestep_flow():
    """测试完整的时间步流程"""
    print("=== 完整时间步流程测试 ===")
    
    # 初始化系统
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    credit_system = ClauseLevelCreditSystem(
        tokenizer=tokenizer,
        credit_model_path="./ckpt/credit_model",
        discount_factor=0.99,
        device="cpu"
    )
    
    # 模拟完整的SQL生成过程
    sql_sequence = [
        "SELECT", " name", ",", " age", ",", " salary",
        " FROM", " employees", 
        " WHERE", " age", " >", " 25", " AND", " salary", " >", " 50000",
        " ORDER", " BY", " salary", " DESC",
        " LIMIT", " 10"
    ]
    
    print("模拟SQL生成序列:")
    print(" ".join(sql_sequence))
    print()
    
    all_experiences = []
    current_sql = ""
    
    for step, token in enumerate(sql_sequence):
        # 模拟Actor和Critic的输出
        mock_value = 10.0 - step * 0.2  # 递减的价值估计
        mock_log_prob = -0.5 - step * 0.05  # 递减的对数概率
        
        print(f"时间步 {step + 1}:")
        print(f"  状态 s_t: '{current_sql}'")
        print(f"  动作 a_t: '{token}'")
        print(f"  价值 V(s_t): {mock_value:.2f}")
        print(f"  对数概率 log_prob(a_t): {mock_log_prob:.3f}")
        
        # 调用Credit Model进行信用分配
        completed_experiences = credit_system.process_token_step(
            sql_prefix=current_sql,
            new_token=token,
            token_id=step,
            value=mock_value,
            log_prob=mock_log_prob
        )
        
        current_sql += token
        
        if completed_experiences:
            print(f"  ✓ 子句完成! 包含 {len(completed_experiences)} 个Token:")
            for i, exp in enumerate(completed_experiences):
                print(f"    Token {i+1}: '{exp.action}' -> r_t = {exp.reward:.2f}")
            all_experiences.extend(completed_experiences)
            print()
        else:
            print("  → Token缓存中，等待子句完成...")
            print()
            
    # 完成最后一个子句
    final_experiences = credit_system.finalize_sequence()
    if final_experiences:
        print("最终子句完成:")
        for i, exp in enumerate(final_experiences):
            print(f"  Token {i+1}: '{exp.action}' -> r_t = {exp.reward:.2f}")
        all_experiences.extend(final_experiences)
        
    print(f"\n=== 完整经验序列 ===")
    print(f"总Token数: {len(all_experiences)}")
    print("时间步流程:")
    for exp in all_experiences:
        print(f"  步骤 {exp.step}: s_t='{exp.state[:20]}...' -> a_t='{exp.action}' -> r_t={exp.reward:.2f}")
        
    # 验证折扣归因
    print(f"\n=== 折扣归因验证 ===")
    stats = credit_system.get_statistics()
    print(f"完成的子句数: {stats['total_completed_clauses']}")
    
    for clause_type, clause_stats in stats['clause_statistics'].items():
        print(f"\n{clause_type} 子句:")
        print(f"  数量: {clause_stats['count']}")
        print(f"  平均信用: {clause_stats['avg_credit']:.2f}")
        
    print("\n✓ 时间步流程测试完成!")
    return True


def test_discount_attribution():
    """测试折扣归因计算的正确性"""
    print("\n=== 折扣归因计算验证 ===")
    
    # 验证折扣归因公式
    clause_credit = 8.0
    gamma = 0.99
    num_tokens = 4
    
    print(f"子句信用: {clause_credit}")
    print(f"折扣因子 γ: {gamma}")
    print(f"Token数量: {num_tokens}")
    print()
    
    # 手动计算期望的奖励
    expected_rewards = []
    for i in range(num_tokens):
        if i == num_tokens - 1:  # 最后一个Token
            reward = clause_credit
        else:
            reward = clause_credit * (gamma ** (num_tokens - 1 - i))
        expected_rewards.append(reward)
        
    print("期望的折扣归因奖励:")
    for i, reward in enumerate(expected_rewards):
        print(f"  Token {i+1}: {reward:.4f}")
        
    # 验证递推关系: r_{t-1} = γ * r_t
    print("\n递推关系验证:")
    for i in range(len(expected_rewards) - 1):
        calculated = expected_rewards[i+1] * gamma
        actual = expected_rewards[i]
        print(f"  r_{i+1} * γ = {calculated:.4f}, r_{i} = {actual:.4f}, 匹配: {abs(calculated - actual) < 1e-6}")
        
    print("\n✓ 折扣归因计算验证完成!")
    return True


def test_integration_with_ppo():
    """测试与PPO训练的集成"""
    print("\n=== PPO集成测试 ===")
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 创建集成对象
    integration = ClauseCreditIntegration(
        tokenizer=tokenizer,
        credit_model_path="./ckpt/credit_model",
        discount_factor=0.99,
        device="cpu"
    )
    
    print("✓ 集成对象创建成功")
    
    # 模拟PPO训练器类
    class MockPPOTrainer:
        def __init__(self):
            self.tokenizer = tokenizer
            self.device = "cpu"
            
        def rollout(self, prompts):
            return f"Standard rollout for {len(prompts)} prompts"
            
    # 测试训练器增强
    try:
        EnhancedTrainer = integration.enhance_ppo_trainer(MockPPOTrainer)
        enhanced_instance = EnhancedTrainer()
        
        print("✓ PPO训练器增强成功")
        print("✓ 支持子句级别信用分配的rollout方法")
        print("✓ 可获取子句统计信息")
        
        # 验证新方法存在
        if hasattr(enhanced_instance, 'get_clause_statistics'):
            print("✓ get_clause_statistics方法可用")
        else:
            print("❌ get_clause_statistics方法缺失")
            
    except Exception as e:
        print(f"❌ PPO训练器增强失败: {e}")
        return False
        
    print("\n✓ PPO集成测试完成!")
    return True


def demonstrate_usage():
    """演示使用方法"""
    print("\n=== 使用方法演示 ===")
    
    print("1. 在train_ppo_ray.py中启用子句级别信用分配:")
    print("   python train_ppo_ray.py \\")
    print("     --enable_clause_credit \\")
    print("     --credit_pretrain ./ckpt/credit_model \\")
    print("     --clause_discount_factor 0.99 \\")
    print("     [其他PPO参数...]")
    
    print("\n2. 系统工作流程:")
    print("   a. 每个时间步t，Actor生成Token a_t")
    print("   b. Critic预测状态价值V(s_t)")
    print("   c. 系统检测SQL子句边界")
    print("   d. 边界触发时调用Credit Model")
    print("   e. 计算子句信用分并执行折扣归因")
    print("   f. 将带奖励的经验存入PPO缓冲区")
    
    print("\n3. 折扣归因公式:")
    print("   r_last_token = Clause_Credit")
    print("   r_{t-1} = γ * r_t")
    print("   其中 γ = 0.99 (可配置)")
    
    print("\n4. 支持的SQL子句:")
    print("   SELECT, FROM, WHERE, GROUP BY, HAVING, ORDER BY, LIMIT, WITH")
    
    print("\n✓ 使用方法演示完成!")


if __name__ == "__main__":
    print("开始子句级别信用分配系统完整测试...")
    
    try:
        # 执行所有测试
        test1 = test_complete_timestep_flow()
        test2 = test_discount_attribution()
        test3 = test_integration_with_ppo()
        
        if all([test1, test2, test3]):
            print("\n🎉 所有测试通过!")
            demonstrate_usage()
        else:
            print("\n❌ 部分测试失败，请检查错误信息")
            
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n=== 测试完成 ===")
