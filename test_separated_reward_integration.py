#!/usr/bin/env python3
"""
测试分离式奖励系统与PPO训练的完整集成
验证所有参数解析和配置是否正确工作
"""

import sys
import subprocess
import tempfile
import os
from unittest.mock import patch, MagicMock


def test_argument_parsing():
    """测试参数解析"""
    print("=== 测试参数解析 ===")

    # 模拟命令行参数
    test_args = [
        "train_ppo_ray.py",
        "--pretrain", "microsoft/DialoGPT-medium",
        "--reward_pretrain", "microsoft/DialoGPT-medium",
        "--credit_pretrain", "microsoft/DialoGPT-medium",
        "--enable_separated_rewards",
        "--credit_weight", "0.3",
        "--clause_credit_weight", "0.2"
    ]

    # 临时修改sys.argv来测试参数解析
    original_argv = sys.argv.copy()
    try:
        sys.argv = test_args

        # 导入train_ppo_ray来测试参数解析
        # 使用exec来避免导入问题
        with open('/home/banlujie/RL-NL2SQL/train_ppo_ray.py', 'r') as f:
            content = f.read()

        # 提取参数解析部分
        parser_start = content.find("parser = argparse.ArgumentParser")
        parser_end = content.find(
            "args = parser.parse_args()") + len("args = parser.parse_args()")

        if parser_start != -1 and parser_end != -1:
            import argparse
            exec_globals = {"argparse": argparse, "__name__": "__main__"}
            parser_code = content[parser_start:parser_end]

            try:
                exec(parser_code, exec_globals)
                args = exec_globals.get('args')

                if args:
                    print("✓ 参数解析成功")
                    print(
                        f"  - enable_separated_rewards: {getattr(args, 'enable_separated_rewards', 'Not found')}")
                    print(
                        f"  - credit_weight: {getattr(args, 'credit_weight', 'Not found')}")
                    print(
                        f"  - clause_credit_weight: {getattr(args, 'clause_credit_weight', 'Not found')}")
                    return True
                else:
                    print("✗ 参数解析失败：未找到args对象")
                    return False

            except Exception as e:
                print(f"✗ 参数解析执行失败: {e}")
                return False
        else:
            print("✗ 未找到参数解析代码段")
            return False

    finally:
        sys.argv = original_argv


def test_mutual_exclusion():
    """测试互斥性检查"""
    print("\n=== 测试互斥性检查 ===")

    test_args = [
        "train_ppo_ray.py",
        "--pretrain", "microsoft/DialoGPT-medium",
        "--reward_pretrain", "microsoft/DialoGPT-medium",
        "--credit_pretrain", "microsoft/DialoGPT-medium",
        "--enable_clause_credit",
        "--enable_separated_rewards"  # 这应该触发错误
    ]

    original_argv = sys.argv.copy()
    try:
        sys.argv = test_args

        with open('/home/banlujie/RL-NL2SQL/train_ppo_ray.py', 'r') as f:
            content = f.read()

        # 提取从参数解析到验证的代码
        parser_start = content.find("parser = argparse.ArgumentParser")
        validation_end = content.find("if args.input_template and")

        if parser_start != -1 and validation_end != -1:
            import argparse
            exec_globals = {"argparse": argparse, "__name__": "__main__"}
            code_section = content[parser_start:validation_end]

            try:
                exec(code_section, exec_globals)
                print("✗ 互斥性检查失败：应该抛出ValueError但没有")
                return False
            except ValueError as e:
                if "mutually exclusive" in str(e):
                    print("✓ 互斥性检查成功：正确检测到冲突配置")
                    print(f"  错误信息: {str(e)[:100]}...")
                    return True
                else:
                    print(f"✗ 互斥性检查失败：抛出了错误的ValueError: {e}")
                    return False
            except Exception as e:
                print(f"✗ 互斥性检查执行失败: {e}")
                return False
        else:
            print("✗ 未找到验证代码段")
            return False

    finally:
        sys.argv = original_argv


def test_separated_reward_system_import():
    """测试分离式奖励系统模块导入"""
    print("\n=== 测试模块导入 ===")

    try:
        from algorithm.separated_reward_system import (
            SeparatedRewardRollout,
            SeparatedRewardPPOIntegration
        )
        print("✓ 分离式奖励系统模块导入成功")

        # 测试类实例化
        integration = SeparatedRewardPPOIntegration(
            credit_model=None,  # 测试时使用None
            credit_weight=0.3,
            clause_credit_weight=0.2
        )
        print("✓ SeparatedRewardPPOIntegration实例化成功")
        return True

    except ImportError as e:
        print(f"✗ 模块导入失败: {e}")
        return False
    except Exception as e:
        print(f"✗ 实例化失败: {e}")
        return False


def test_integration_logic():
    """测试集成逻辑"""
    print("\n=== 测试集成逻辑 ===")

    try:
        # 模拟PPO训练器类
        class MockPPOTrainer:
            def __init__(self):
                self.name = "MockPPOTrainer"

        # 测试分离式奖励系统增强
        from algorithm.separated_reward_system import SeparatedRewardPPOIntegration

        integration = SeparatedRewardPPOIntegration(
            credit_model=None,
            credit_weight=0.3,
            clause_credit_weight=0.2
        )

        enhanced_trainer = integration.enhance_ppo_trainer(MockPPOTrainer)
        print("✓ PPO训练器增强成功")

        # 验证增强后的训练器
        if hasattr(enhanced_trainer, '__call__'):
            print("✓ 增强后的训练器是可调用的")
            return True
        else:
            print("✗ 增强后的训练器不可调用")
            return False

    except Exception as e:
        print(f"✗ 集成逻辑测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("开始测试分离式奖励系统与PPO训练的完整集成...")

    tests = [
        ("参数解析测试", test_argument_parsing),
        ("互斥性检查测试", test_mutual_exclusion),
        ("模块导入测试", test_separated_reward_system_import),
        ("集成逻辑测试", test_integration_logic)
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
        print("🎉 所有测试通过！分离式奖励系统集成完成")
        return True
    else:
        print("❌ 部分测试失败，需要修复")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
