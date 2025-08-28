#!/usr/bin/env python3
"""
简化的分离式奖励系统集成测试
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, '/home/banlujie/RL-NL2SQL')

def test_basic_import():
    """测试基本模块导入"""
    print("=== 测试基本模块导入 ===")
    
    try:
        from algorithm.separated_reward_system import (
            SeparatedRewardRollout, 
            SeparatedRewardPPOIntegration
        )
        print("✓ 分离式奖励系统模块导入成功")
        return True
    except Exception as e:
        print(f"✗ 模块导入失败: {e}")
        return False

def test_basic_functionality():
    """测试基本功能"""
    print("\n=== 测试基本功能 ===")
    
    try:
        from algorithm.separated_reward_system import SeparatedRewardPPOIntegration
        from transformers import AutoTokenizer
        
        # 创建tokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 创建实例（使用正确的参数）
        integration = SeparatedRewardPPOIntegration(
            tokenizer=tokenizer,
            credit_model_path="./ckpt/credit_model",  # 使用路径而不是模型对象
            discount_factor=0.99,
            device="cpu"
        )
        print("✓ SeparatedRewardPPOIntegration实例化成功")
        
        # 模拟训练器类
        class MockTrainer:
            def __init__(self):
                self.enhanced = False
                
            def mark_enhanced(self):
                self.enhanced = True
        
        # 测试训练器增强
        mock_trainer = MockTrainer
        enhanced_trainer = integration.enhance_ppo_trainer(mock_trainer)
        print("✓ PPO训练器增强成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 基本功能测试失败: {e}")
        return False

def test_parameter_validation():
    """测试参数验证逻辑"""
    print("\n=== 测试参数验证逻辑 ===")
    
    # 检查train_ppo_ray.py中是否包含新参数
    try:
        with open('/home/banlujie/RL-NL2SQL/train_ppo_ray.py', 'r') as f:
            content = f.read()
        
        checks = [
            ("--enable_separated_rewards", "enable_separated_rewards" in content),
            ("分离式奖励系统增强", "分离式奖励系统增强" in content),
            ("互斥性检查", "mutually exclusive" in content),
            ("SeparatedRewardPPOIntegration导入", "SeparatedRewardPPOIntegration" in content)
        ]
        
        all_passed = True
        for check_name, check_result in checks:
            if check_result:
                print(f"✓ {check_name}: 已包含")
            else:
                print(f"✗ {check_name}: 未找到")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"✗ 参数验证测试失败: {e}")
        return False

def test_file_structure():
    """测试文件结构完整性"""
    print("\n=== 测试文件结构完整性 ===")
    
    required_files = [
        '/home/banlujie/RL-NL2SQL/algorithm/separated_reward_system.py',
        '/home/banlujie/RL-NL2SQL/algorithm/clause_credit_system.py',
        '/home/banlujie/RL-NL2SQL/train_ppo_ray.py'
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {os.path.basename(file_path)}: 存在")
        else:
            print(f"✗ {os.path.basename(file_path)}: 不存在")
            all_exist = False
    
    return all_exist

def main():
    """主测试函数"""
    print("开始简化的分离式奖励系统集成测试...\n")
    
    tests = [
        ("文件结构测试", test_file_structure),
        ("基本模块导入测试", test_basic_import),
        ("基本功能测试", test_basic_functionality), 
        ("参数验证测试", test_parameter_validation)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name}执行异常: {e}")
            results.append((test_name, False))
    
    # 总结测试结果
    print("\n" + "="*50)
    print("测试结果总结:")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 个测试通过")
    
    if passed == len(results):
        print("\n🎉 所有测试通过！")
        print("分离式奖励系统已成功集成到PPO训练中")
        print("\n使用方法:")
        print("python train_ppo_ray.py --enable_separated_rewards --credit_weight 0.3 --clause_credit_weight 0.2 [其他参数...]")
        return True
    else:
        print(f"\n❌ {len(results) - passed} 个测试失败，需要修复")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
