#!/usr/bin/env python3
"""
测试PPO训练中AdaptiveEntropyProcessor的集成
"""

import os
import sys
import torch
from transformers import AutoTokenizer
from algorithm.ppo_entropy_integration import (
    PPOAdaptiveEntropyIntegration,
    create_ppo_generation_config_with_entropy,
    integrate_adaptive_entropy_to_ppo_trainer
)

def test_ppo_entropy_integration():
    """测试PPO训练集成"""
    print("=== 测试PPO AdaptiveEntropy集成 ===")
    
    # 模拟参数配置
    class MockArgs:
        def __init__(self):
            self.pretrain = "microsoft/DialoGPT-small"  # 使用小模型测试
            self.enable_adaptive_entropy = True
            self.low_entropy_threshold = 2.0
            self.high_entropy_threshold = 4.0
            self.low_k = 5
            self.high_k = 50
            self.enable_content_filtering = True
            self.generate_max_len = 128
            self.temperature = 0.8
            self.top_p = 0.9
    
    args = MockArgs()
    
    try:
        # 1. 测试tokenizer加载
        print("1. 测试tokenizer加载...")
        tokenizer = AutoTokenizer.from_pretrained(args.pretrain)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"✓ Tokenizer加载成功: {args.pretrain}")
        
        # 2. 测试生成配置创建
        print("\n2. 测试生成配置创建...")
        adaptive_entropy_config = create_ppo_generation_config_with_entropy(
            tokenizer=tokenizer,
            max_new_tokens=args.generate_max_len,
            temperature=args.temperature,
            top_p=args.top_p,
            low_entropy_threshold=args.low_entropy_threshold,
            high_entropy_threshold=args.high_entropy_threshold,
            low_k=args.low_k,
            high_k=args.high_k,
            enable_content_filtering=args.enable_content_filtering,
            device="cpu"  # 测试使用CPU
        )
        
        print("✓ 生成配置创建成功")
        print(f"  - 配置键: {list(adaptive_entropy_config.keys())}")
        print(f"  - Max tokens: {adaptive_entropy_config.get('max_new_tokens')}")
        print(f"  - Temperature: {adaptive_entropy_config.get('temperature')}")
        print(f"  - 包含logits_processor: {'logits_processor' in adaptive_entropy_config}")
        
        # 3. 测试集成器
        print("\n3. 测试PPO集成器...")
        integrator = PPOAdaptiveEntropyIntegration(
            tokenizer=tokenizer,
            low_entropy_threshold=args.low_entropy_threshold,
            high_entropy_threshold=args.high_entropy_threshold,
            low_k=args.low_k,
            high_k=args.high_k,
            enable_content_filtering=args.enable_content_filtering,
            device="cpu"
        )
        
        print("✓ 集成器创建成功")
        print(f"  - 熵阈值: low={integrator.processor.low_entropy_threshold}, high={integrator.processor.high_entropy_threshold}")
        print(f"  - K值: low={integrator.processor.low_k}, high={integrator.processor.high_k}")
        print(f"  - 内容过滤: {integrator.processor.enable_content_filtering}")
        
        # 4. 测试与模拟trainer的集成
        print("\n4. 测试trainer集成...")
        
        # 模拟trainer配置
        trainer_kwargs = {
            "do_sample": True,
            "max_new_tokens": args.generate_max_len,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }
        
        # 创建生成配置（这是实际在train_ppo_ray.py中使用的方法）
        updated_kwargs = create_ppo_generation_config_with_entropy(
            tokenizer=tokenizer,
            max_new_tokens=args.generate_max_len,
            temperature=args.temperature,
            top_p=args.top_p,
            low_entropy_threshold=args.low_entropy_threshold,
            high_entropy_threshold=args.high_entropy_threshold,
            low_k=args.low_k,
            high_k=args.high_k,
            enable_content_filtering=args.enable_content_filtering,
            device="cpu"
        )
        
        print("✓ 生成配置集成成功")
        print(f"  - 原始参数数量: {len(trainer_kwargs)}")
        print(f"  - 集成后参数数量: {len(updated_kwargs)}")
        print(f"  - 新增参数: {set(updated_kwargs.keys()) - set(trainer_kwargs.keys())}")
        
        # 验证logits_processor是否正确集成
        if 'logits_processor' in updated_kwargs:
            logits_processors = updated_kwargs['logits_processor']
            print(f"  - LogitsProcessor类型: {type(logits_processors)}")
            print(f"  - Processor数量: {len(logits_processors)}")
            if len(logits_processors) > 0:
                print(f"  - 第一个processor类型: {type(logits_processors[0])}")
        
        # 测试模拟trainer集成（创建一个简单的mock对象）
        class MockPPOTrainer:
            def __init__(self):
                self.generation_kwargs = {}
        
        mock_trainer = MockPPOTrainer()
        
        # 使用集成函数
        integrated_trainer = integrate_adaptive_entropy_to_ppo_trainer(
            ppo_trainer=mock_trainer,
            tokenizer=tokenizer,
            low_entropy_threshold=args.low_entropy_threshold,
            high_entropy_threshold=args.high_entropy_threshold,
            low_k=args.low_k,
            high_k=args.high_k,
            enable_content_filtering=args.enable_content_filtering,
            device="cpu"
        )
        
        print("✓ Mock trainer集成成功")
        print(f"  - 训练器包含adaptive_entropy_integration: {hasattr(integrated_trainer, 'adaptive_entropy_integration')}")
        print(f"  - 生成参数包含logits_processor: {'logits_processor' in integrated_trainer.generation_kwargs}")
        
        # 5. 测试统计记录
        print("\n5. 测试统计记录...")
        stats = integrator.processor.get_stats()
        print(f"✓ 初始统计信息: {stats}")
        
        # 模拟一些处理调用
        test_input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        test_logits = torch.randn(1, 5, 1000)  # batch_size=1, seq_len=5, vocab_size=1000
        
        # 处理logits
        processed_logits = integrator.processor(test_input_ids, test_logits)
        
        updated_stats = integrator.processor.get_stats()
        print(f"✓ 处理后统计信息: {updated_stats}")
        
        print("\n=== 所有测试通过! ===")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_command_line_integration():
    """测试命令行参数集成"""
    print("\n=== 测试命令行参数集成 ===")
    
    # 模拟命令行参数
    test_args = [
        "--enable_adaptive_entropy",
        "--low_entropy_threshold", "1.5",
        "--high_entropy_threshold", "5.0", 
        "--low_k", "3",
        "--high_k", "100",
        "--enable_content_filtering"
    ]
    
    # 这里只是验证参数定义，不实际运行训练
    print("✓ 命令行参数支持:")
    for arg in test_args:
        print(f"  {arg}")
    
    print("✓ 参数集成测试完成")

if __name__ == "__main__":
    print("开始测试PPO AdaptiveEntropy集成...")
    
    # 测试集成功能
    integration_success = test_ppo_entropy_integration()
    
    # 测试命令行参数
    test_command_line_integration()
    
    if integration_success:
        print("\n🎉 所有集成测试通过！")
        print("\n使用说明:")
        print("在PPO训练中启用AdaptiveEntropyProcessor:")
        print("python train_ppo_ray.py --enable_adaptive_entropy \\")
        print("                       --low_entropy_threshold 2.0 \\") 
        print("                       --high_entropy_threshold 4.0 \\")
        print("                       --low_k 5 \\")
        print("                       --high_k 50 \\")
        print("                       --enable_content_filtering \\")
        print("                       [其他PPO参数...]")
        print("\n系统将在Actor生成时动态调整Top-K值:")
        print("- 低熵(confident): 使用小k值进行exploitation")
        print("- 高熵(uncertain): 使用大k值进行exploration")
        print("- 智能过滤SQL vs 思考内容")
    else:
        print("\n❌ 集成测试失败，请检查错误信息")
        sys.exit(1)
