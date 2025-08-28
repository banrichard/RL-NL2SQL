#!/usr/bin/env python3
"""
测试PPO阶段的奖励函数（不执行SQL）
"""

import sys
import os
sys.path.append('/home/banlujie/RL-NL2SQL')

def test_ppo_reward_function():
    """测试PPO奖励函数"""
    print("Testing PPO reward function without SQL execution...")
    
    # 模拟数据
    queries = [
        "SELECT name FROM students WHERE age > 18",
        "SELECT COUNT(*) FROM courses",
        "SELECT * FROM invalid_table"  # 这个SQL可能是错误的
    ]
    
    prompts = [
        "Find all students older than 18",
        "Count the number of courses", 
        "Show all data from invalid table"
    ]
    
    labels = [
        "SELECT name FROM students WHERE age > 18",
        "SELECT COUNT(*) FROM courses",
        "SELECT * FROM valid_table"
    ]
    
    # 模拟reward model的输出分数
    reward_model_scores = [0.8, 0.9, 0.1]  # 第三个SQL质量较差
    
    try:
        # 测试不带Credit Model的版本
        from algorithm.ppo_reward import ppo_reward_func_without_sql
        
        print("\n1. Testing without Credit Model:")
        result1 = ppo_reward_func_without_sql(
            queries=queries,
            prompts=prompts,
            labels=labels,
            reward_model_scores=reward_model_scores,
            enable_credit_shaping=False
        )
        
        print(f"Rewards: {result1['rewards']}")
        print(f"Extra logs: {result1['extra_logs']}")
        
        # 测试带Credit Model的版本（需要先初始化）
        print("\n2. Testing with Credit Model:")
        
        # 初始化Credit集成器
        credit_model_path = "/home/banlujie/RL-NL2SQL/ckpt/phi4-7b-cm"
        if os.path.exists(credit_model_path):
            from algorithm.simple_credit_integration import init_credit_integrator
            
            try:
                credit_integrator = init_credit_integrator(credit_model_path, "cuda")
                print("Credit integrator initialized successfully")
                
                result2 = ppo_reward_func_without_sql(
                    queries=queries,
                    prompts=prompts,
                    labels=labels,
                    reward_model_scores=reward_model_scores,
                    enable_credit_shaping=True,
                    credit_weight=0.1,
                    credit_aggregation="mean"
                )
                
                print(f"Enhanced rewards: {result2['rewards']}")
                print(f"Credit rewards: {result2['credit_rewards']}")
                print(f"Original rewards: {result2['original_rewards']}")
                print(f"Extra logs: {result2['extra_logs']}")
                
            except Exception as e:
                print(f"Credit Model loading failed: {e}")
                print("Continuing with basic test...")
        else:
            print(f"Credit Model not found at {credit_model_path}")
        
        print("\n3. Testing hybrid reward function:")
        from algorithm.ppo_reward import ppo_reward_func_hybrid
        
        result3 = ppo_reward_func_hybrid(
            queries=queries,
            prompts=prompts,
            labels=labels,
            reward_model_scores=reward_model_scores,
            use_sql_execution=False,  # PPO模式
            enable_credit_shaping=True,
            credit_weight=0.2
        )
        
        print(f"Hybrid rewards: {result3['rewards']}")
        print(f"Extra logs: {result3['extra_logs']}")
        
        print("\n✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reward_wrapper():
    """测试奖励函数包装器"""
    print("\nTesting reward function wrapper...")
    
    try:
        from algorithm.ppo_reward import create_ppo_reward_wrapper
        from algorithm.reward import reward_func as original_reward_func
        
        # 创建包装器
        wrapped_func = create_ppo_reward_wrapper(
            original_reward_func=original_reward_func,
            enable_credit=True,
            credit_weight=0.1,
            credit_aggregation="mean"
        )
        
        # 测试数据
        queries = ["SELECT COUNT(*) FROM students"]
        prompts = ["Count all students"]
        labels = ["SELECT COUNT(*) FROM students"]
        
        # 测试带预计算分数的情况
        result = wrapped_func(
            queries, prompts, labels,
            reward_model_scores=[0.8]
        )
        
        print(f"Wrapper result: {result}")
        print("✅ Reward wrapper test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Wrapper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=== PPO Reward Function Test ===")
    
    success1 = test_ppo_reward_function()
    success2 = test_reward_wrapper()
    
    if success1 and success2:
        print("\n🎉 All tests passed! PPO reward function is ready.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
