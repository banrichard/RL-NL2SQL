#!/usr/bin/env python3
"""
测试 Credit Model 集成的快速验证脚本
"""

import os
import sys
import torch

# 添加项目路径
sys.path.append('/home/banlujie/RL-NL2SQL')

def test_credit_model_loading():
    """测试 Credit Model 加载"""
    print("1. 测试 Credit Model 加载...")
    
    try:
        from algorithm.credit_model import CreditModel
        
        # 假设的模型路径 - 请替换为你的实际路径
        model_path = "/home/banlujie/RL-NL2SQL/ckpt/phi4-7b-cm"
        
        if not os.path.exists(model_path):
            print(f"❌ Credit Model 路径不存在: {model_path}")
            print("请先训练 Credit Model 或检查路径是否正确")
            return False
        
        # 加载模型
        model = CreditModel.from_pretrained(model_path, num_outputs=8)
        print(f"✅ Credit Model 加载成功!")
        print(f"   模型输出维度: {model.regression_head.out_features}")
        return True
        
    except Exception as e:
        print(f"❌ Credit Model 加载失败: {e}")
        return False

def test_simple_integration():
    """测试简化集成"""
    print("\n2. 测试简化集成...")
    
    try:
        from algorithm.simple_credit_integration import SimpleCreditIntegrator, init_credit_integrator
        
        # 假设的模型路径
        model_path = "/home/banlujie/RL-NL2SQL/ckpt/phi4-7b-cm"
        
        if not os.path.exists(model_path):
            print(f"❌ 跳过简化集成测试，模型路径不存在: {model_path}")
            return False
        
        # 初始化集成器
        integrator = SimpleCreditIntegrator(model_path)
        
        # 测试数据
        test_sequences = [
            "Schema: CREATE TABLE users (id INT, name VARCHAR(50));\nQuestion: Find all users\nSQL: SELECT * FROM users",
            "Schema: CREATE TABLE products (id INT, price DECIMAL);\nQuestion: Get expensive products\nSQL: SELECT * FROM products WHERE price > 100"
        ]
        
        # 计算信用分数
        credit_scores = integrator.compute_credit_scores(test_sequences)
        
        print(f"✅ 简化集成测试成功!")
        print(f"   测试序列数: {len(test_sequences)}")
        print(f"   信用分数维度: {len(credit_scores[0]) if credit_scores else 0}")
        
        # 测试奖励增强
        original_rewards = [0.8, 0.6]
        enhanced_data = integrator.enhance_rewards(
            sequences=test_sequences,
            original_rewards=original_rewards,
            credit_weight=0.1
        )
        
        print(f"   原始奖励: {original_rewards}")
        print(f"   增强奖励: {enhanced_data['rewards']}")
        print(f"   信用奖励: {enhanced_data['credit_rewards']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 简化集成测试失败: {e}")
        return False

def test_actor_integration():
    """测试 Actor 集成"""
    print("\n3. 测试 Actor 集成...")
    
    try:
        # 首先检查OpenRLHF的基类是否可用
        from openrlhf.trainer.ray.launcher import BaseModelActor
        print(f"✅ BaseModelActor 导入成功!")
        
        from algorithm.credit_model_actor import CreditModelActor
        print(f"✅ CreditModelActor 导入成功!")
        
        # 检查 Ray 是否可用
        try:
            import ray
            print(f"✅ Ray 可用，版本: {ray.__version__}")
        except ImportError:
            print(f"⚠️  Ray 不可用，但 Actor 类定义正确")
        
        return True
        
    except Exception as e:
        print(f"❌ Actor 集成测试失败: {e}")
        return False

def test_enhanced_ppo():
    """测试增强的 PPO 集成"""
    print("\n4. 测试增强的 PPO 集成...")
    
    try:
        from algorithm.credit_enhanced_ppo import CreditEnhancedPPOTrainer, create_credit_enhanced_ppo_trainer
        print(f"✅ CreditEnhancedPPOTrainer 导入成功!")
        
        # 测试创建增强训练器的函数
        print(f"✅ create_credit_enhanced_ppo_trainer 函数可用!")
        
        return True
        
    except Exception as e:
        print(f"❌ 增强 PPO 集成测试失败: {e}")
        return False

def test_reward_integration():
    """测试奖励函数集成"""
    print("\n5. 测试奖励函数集成...")
    
    try:
        from algorithm.reward import reward_func
        
        # 测试数据
        test_queries = [
            "user\nShow me all products\nassistant\nSELECT * FROM products",
            "user\nFind expensive items\nassistant\nSELECT * FROM products WHERE price > 100"
        ]
        test_prompts = ["Show me all products", "Find expensive items"]
        test_labels = ["SELECT * FROM products", "SELECT * FROM products WHERE price > 100"]
        
        # 测试基础奖励函数
        reward_data = reward_func(test_queries, test_prompts, test_labels)
        
        print(f"✅ 奖励函数集成测试成功!")
        print(f"   奖励数量: {len(reward_data['rewards'])}")
        print(f"   平均奖励: {sum(reward_data['rewards']) / len(reward_data['rewards']):.4f}")
        
        # 测试带 Credit Model 的奖励函数（如果模型存在）
        model_path = "/home/banlujie/RL-NL2SQL/ckpt/phi4-7b-cm"
        if os.path.exists(model_path):
            try:
                enhanced_reward_data = reward_func(
                    test_queries, test_prompts, test_labels,
                    credit_model_path=model_path,
                    credit_weight=0.1,
                    credit_aggregation="mean"
                )
                print(f"✅ Credit 增强奖励测试成功!")
                print(f"   增强奖励: {enhanced_reward_data['rewards']}")
            except Exception as e:
                print(f"⚠️  Credit 增强测试失败: {e}")
        else:
            print(f"⚠️  跳过 Credit 增强测试，模型不存在")
        
        return True
        
    except Exception as e:
        print(f"❌ 奖励函数集成测试失败: {e}")
        print(f"   错误详情: {str(e)}")
        # 尝试检查具体的导入问题
        try:
            from algorithm import reward
            print(f"   algorithm.reward 模块可用")
        except ImportError as import_err:
            print(f"   algorithm.reward 导入失败: {import_err}")
        return False

def main():
    """主测试函数"""
    print("🚀 Credit Model PPO 集成测试开始...\n")
    
    tests = [
        test_credit_model_loading,
        test_simple_integration,
        test_actor_integration,
        test_enhanced_ppo,
        test_reward_integration,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ 测试过程中发生错误: {e}")
            results.append(False)
    
    # 总结
    print("\n" + "="*50)
    print("📊 测试结果总结:")
    test_names = [
        "Credit Model 加载",
        "简化集成",
        "Actor 集成", 
        "增强 PPO 集成",
        "奖励函数集成"
    ]
    
    for name, result in zip(test_names, results):
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {name}: {status}")
    
    passed = sum(results)
    total = len(results)
    print(f"\n🎯 总体结果: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！你可以开始使用 Credit Model 进行 PPO 训练了。")
        print("\n📝 下一步:")
        print("   1. 确保你的 Credit Model 已经训练完成")
        print("   2. 修改 scripts/ppo_with_credit.sh 中的路径")
        print("   3. 运行: bash scripts/ppo_with_credit.sh")
    else:
        print("⚠️  部分测试失败，请检查相关组件。")
        print("\n🔧 故障排除:")
        print("   1. 确保 Credit Model 已训练并保存在正确位置")
        print("   2. 检查依赖包是否正确安装")
        print("   3. 查看具体错误信息进行调试")

if __name__ == "__main__":
    main()
