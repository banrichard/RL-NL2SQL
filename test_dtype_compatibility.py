#!/usr/bin/env python3
"""
Credit Model 数据类型兼容性测试脚本
"""

import sys
import torch
sys.path.append('/home/banlujie/RL-NL2SQL')

def test_credit_model_dtype():
    """测试Credit Model的数据类型兼容性"""
    print("🔧 测试 Credit Model 数据类型兼容性...")
    
    try:
        from algorithm.credit_model import CreditModel
        
        # 创建一个测试模型实例（使用小的参数避免加载大模型）
        print("   创建测试模型...")
        
        # 使用一个简单的配置进行测试
        test_config = {
            'num_outputs': 8,
            'bf16': True,
            'load_in_4bit': False,
            'lora_rank': 0,
        }
        
        # 测试路径
        model_path = "/home/banlujie/RL-NL2SQL/ckpt/phi4-7b-cm"
        
        if torch.cuda.is_available():
            print("   CUDA 可用，使用 GPU 测试")
            device = "cuda"
        else:
            print("   CUDA 不可用，使用 CPU 测试")
            device = "cpu"
        
        # 测试模型加载
        try:
            model = CreditModel.from_pretrained(model_path, **test_config)
            model = model.to(device)
            print(f"   ✅ 模型加载成功")
            print(f"   - 主模型数据类型: {next(model.model.parameters()).dtype}")
            print(f"   - 回归头数据类型: {model.regression_head.weight.dtype}")
            
            # 测试前向传播
            print("   测试前向传播...")
            test_input_ids = torch.randint(0, 1000, (2, 50)).to(device)
            test_attention_mask = torch.ones(2, 50).to(device)
            
            with torch.no_grad():
                outputs = model(input_ids=test_input_ids, attention_mask=test_attention_mask)
                print(f"   ✅ 前向传播成功")
                print(f"   - 输出形状: {outputs.shape}")
                print(f"   - 输出数据类型: {outputs.dtype}")
                print(f"   - 输出范围: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
            
            return True
            
        except FileNotFoundError:
            print(f"   ⚠️  模型文件不存在: {model_path}")
            print("   请先训练 Credit Model")
            return False
            
    except Exception as e:
        print(f"   ❌ 测试失败: {e}")
        return False

def test_simple_integration_dtype():
    """测试简化集成的数据类型"""
    print("\n🔧 测试简化集成数据类型兼容性...")
    
    try:
        from algorithm.simple_credit_integration import SimpleCreditIntegrator
        
        model_path = "/home/banlujie/RL-NL2SQL/ckpt/phi4-7b-cm"
        
        try:
            integrator = SimpleCreditIntegrator(model_path)
            
            # 测试序列
            test_sequences = [
                "Schema: CREATE TABLE users (id INT, name VARCHAR(50));\nQuestion: Find all users\nSQL: SELECT * FROM users"
            ]
            
            # 计算信用分数
            credit_scores = integrator.compute_credit_scores(test_sequences)
            
            print(f"   ✅ 简化集成数据类型测试成功")
            print(f"   - 信用分数: {credit_scores[0]}")
            print(f"   - 分数类型: {type(credit_scores[0][0])}")
            
            return True
            
        except FileNotFoundError:
            print(f"   ⚠️  模型文件不存在，跳过测试")
            return False
            
    except Exception as e:
        print(f"   ❌ 简化集成测试失败: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Credit Model 数据类型兼容性测试\n")
    
    # 检查PyTorch版本和CUDA
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 数量: {torch.cuda.device_count()}")
    print()
    
    # 运行测试
    tests = [
        test_credit_model_dtype,
        test_simple_integration_dtype,
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
    print("📊 数据类型兼容性测试结果:")
    test_names = [
        "Credit Model 数据类型",
        "简化集成数据类型",
    ]
    
    for name, result in zip(test_names, results):
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {name}: {status}")
    
    passed = sum(results)
    total = len(results)
    print(f"\n🎯 总体结果: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有数据类型兼容性测试通过！")
    else:
        print("⚠️  部分测试失败，请检查数据类型配置。")
        print("\n💡 建议:")
        print("   1. 确保回归头和主模型使用相同的数据类型")
        print("   2. 在前向传播中添加数据类型转换")
        print("   3. 检查bf16配置是否正确")
