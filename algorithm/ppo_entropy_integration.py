"""
将AdaptiveEntropyProcessor集成到PPO训练中
"""

import torch
from typing import Optional, Dict, Any, List
from transformers import LogitsProcessorList
from algorithm.entropy_processor import AdaptiveEntropyLogitsProcessor, create_adaptive_entropy_processor


class PPOAdaptiveEntropyIntegration:
    """
    PPO训练中的自适应熵处理器集成类
    """
    
    def __init__(
        self,
        tokenizer=None,
        low_entropy_threshold: float = 2.0,
        high_entropy_threshold: float = 4.0,
        low_k: int = 5,
        high_k: int = 50,
        enable_content_filtering: bool = True,
        device: str = "cuda"
    ):
        """
        初始化PPO自适应熵集成
        
        Args:
            tokenizer: 用于内容过滤的tokenizer
            low_entropy_threshold: 低熵阈值
            high_entropy_threshold: 高熵阈值
            low_k: 利用策略的k值
            high_k: 探索策略的k值
            enable_content_filtering: 是否启用内容过滤
            device: 计算设备
        """
        self.tokenizer = tokenizer
        self.low_entropy_threshold = low_entropy_threshold
        self.high_entropy_threshold = high_entropy_threshold
        self.low_k = low_k
        self.high_k = high_k
        self.enable_content_filtering = enable_content_filtering
        self.device = device
        
        # 创建处理器
        self.processor = create_adaptive_entropy_processor(
            low_entropy_threshold=low_entropy_threshold,
            high_entropy_threshold=high_entropy_threshold,
            low_k=low_k,
            high_k=high_k,
            enable_content_filtering=enable_content_filtering,
            tokenizer=tokenizer,
            device=device
        )
        
        print(f"PPO AdaptiveEntropy Integration initialized:")
        print(f"  Entropy thresholds: {low_entropy_threshold} - {high_entropy_threshold}")
        print(f"  K range: {low_k} - {high_k}")
        print(f"  Content filtering: {enable_content_filtering}")
    
    def create_logits_processor_list(self) -> LogitsProcessorList:
        """
        创建包含自适应熵处理器的LogitsProcessorList
        
        Returns:
            配置好的LogitsProcessorList
        """
        processor_list = LogitsProcessorList([self.processor])
        return processor_list
    
    def get_generation_kwargs(self) -> Dict[str, Any]:
        """
        获取生成时需要的参数字典
        
        Returns:
            包含logits_processor的参数字典
        """
        return {
            "logits_processor": self.create_logits_processor_list(),
            "do_sample": True,  # 必须启用采样才能使用logits processor
            "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer else None,
        }
    
    def update_ppo_trainer_config(self, trainer_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新PPO训练器配置以集成自适应熵处理器
        
        Args:
            trainer_config: 原始的trainer配置字典
            
        Returns:
            更新后的配置字典
        """
        # 获取生成参数
        generation_kwargs = self.get_generation_kwargs()
        
        # 更新配置
        updated_config = trainer_config.copy()
        
        # 添加logits processor到生成参数中
        if "generation_kwargs" not in updated_config:
            updated_config["generation_kwargs"] = {}
        
        updated_config["generation_kwargs"].update(generation_kwargs)
        
        # 确保启用采样
        updated_config["do_sample"] = True
        
        print("PPO trainer config updated with adaptive entropy processor")
        return updated_config
    
    def get_stats(self) -> Dict[str, Any]:
        """获取处理器统计信息"""
        return self.processor.get_stats()
    
    def reset_stats(self):
        """重置统计信息"""
        self.processor.reset_stats()
    
    def log_stats(self, step: int = None):
        """记录统计信息"""
        stats = self.get_stats()
        prefix = f"Step {step}: " if step is not None else ""
        
        print(f"{prefix}AdaptiveEntropy Stats:")
        print(f"  Total calls: {stats['total_calls']}")
        print(f"  Processing ratio: {stats.get('processing_ratio', 0):.2%}")
        print(f"  Low entropy: {stats.get('low_entropy_ratio', 0):.2%}")
        print(f"  High entropy: {stats.get('high_entropy_ratio', 0):.2%}")
        print(f"  Avg entropy: {stats.get('avg_entropy', 0):.3f}")
        print(f"  Avg k: {stats.get('avg_k_used', 0):.1f}")


def integrate_adaptive_entropy_to_ppo_trainer(
    ppo_trainer,
    tokenizer=None,
    low_entropy_threshold: float = 2.0,
    high_entropy_threshold: float = 4.0,
    low_k: int = 5,
    high_k: int = 50,
    enable_content_filtering: bool = True,
    device: str = "cuda"
):
    """
    将自适应熵处理器集成到现有的PPO训练器中
    
    Args:
        ppo_trainer: PPO训练器实例
        tokenizer: tokenizer对象
        low_entropy_threshold: 低熵阈值
        high_entropy_threshold: 高熵阈值
        low_k: 利用策略的k值
        high_k: 探索策略的k值
        enable_content_filtering: 是否启用内容过滤
        device: 计算设备
        
    Returns:
        集成了自适应熵处理器的训练器
    """
    # 创建集成器
    integration = PPOAdaptiveEntropyIntegration(
        tokenizer=tokenizer,
        low_entropy_threshold=low_entropy_threshold,
        high_entropy_threshold=high_entropy_threshold,
        low_k=low_k,
        high_k=high_k,
        enable_content_filtering=enable_content_filtering,
        device=device
    )
    
    # 获取logits processor
    logits_processor = integration.create_logits_processor_list()
    
    # 将处理器添加到训练器中
    # 这取决于具体的PPO训练器实现
    if hasattr(ppo_trainer, 'generation_kwargs'):
        ppo_trainer.generation_kwargs['logits_processor'] = logits_processor
        ppo_trainer.generation_kwargs['do_sample'] = True
    elif hasattr(ppo_trainer, 'actor'):
        # 如果训练器有actor属性，尝试设置其生成参数
        if hasattr(ppo_trainer.actor, 'generation_kwargs'):
            ppo_trainer.actor.generation_kwargs['logits_processor'] = logits_processor
            ppo_trainer.actor.generation_kwargs['do_sample'] = True
    
    # 将集成器附加到训练器以便后续访问统计信息
    ppo_trainer.adaptive_entropy_integration = integration
    
    print("AdaptiveEntropyProcessor successfully integrated into PPO trainer")
    return ppo_trainer


def create_ppo_generation_config_with_entropy(
    tokenizer=None,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
    low_entropy_threshold: float = 2.0,
    high_entropy_threshold: float = 4.0,
    low_k: int = 5,
    high_k: int = 50,
    enable_content_filtering: bool = True,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    创建包含自适应熵处理器的PPO生成配置
    
    Args:
        tokenizer: tokenizer对象
        max_new_tokens: 最大生成token数
        temperature: 温度参数
        top_p: nucleus采样参数
        low_entropy_threshold: 低熵阈值
        high_entropy_threshold: 高熵阈值
        low_k: 利用策略的k值
        high_k: 探索策略的k值
        enable_content_filtering: 是否启用内容过滤
        device: 计算设备
        
    Returns:
        完整的生成配置字典
    """
    # 创建自适应熵处理器
    entropy_processor = create_adaptive_entropy_processor(
        low_entropy_threshold=low_entropy_threshold,
        high_entropy_threshold=high_entropy_threshold,
        low_k=low_k,
        high_k=high_k,
        enable_content_filtering=enable_content_filtering,
        tokenizer=tokenizer,
        device=device
    )
    
    # 创建logits processor列表
    logits_processor_list = LogitsProcessorList([entropy_processor])
    
    # 构建完整的生成配置
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": True,  # 必须启用采样
        "logits_processor": logits_processor_list,
        "pad_token_id": tokenizer.pad_token_id if tokenizer else None,
        "eos_token_id": tokenizer.eos_token_id if tokenizer else None,
    }
    
    print(f"PPO generation config created with AdaptiveEntropyProcessor")
    print(f"  Max tokens: {max_new_tokens}")
    print(f"  Temperature: {temperature}")
    print(f"  Top-p: {top_p}")
    print(f"  Adaptive k range: {low_k}-{high_k}")
    
    return generation_config


# 使用示例函数
def setup_ppo_with_adaptive_entropy(
    ppo_trainer_class,
    model_args,
    tokenizer,
    entropy_config: Optional[Dict[str, Any]] = None
):
    """
    设置带有自适应熵处理器的PPO训练器
    
    Args:
        ppo_trainer_class: PPO训练器类
        model_args: 模型参数
        tokenizer: tokenizer对象
        entropy_config: 熵处理器配置
        
    Returns:
        配置好的PPO训练器
    """
    # 默认熵配置
    default_entropy_config = {
        "low_entropy_threshold": 2.0,
        "high_entropy_threshold": 4.0,
        "low_k": 5,
        "high_k": 50,
        "enable_content_filtering": True,
    }
    
    if entropy_config:
        default_entropy_config.update(entropy_config)
    
    # 创建生成配置
    generation_config = create_ppo_generation_config_with_entropy(
        tokenizer=tokenizer,
        max_new_tokens=model_args.get("max_new_tokens", 512),
        temperature=model_args.get("temperature", 1.0),
        top_p=model_args.get("top_p", 1.0),
        **default_entropy_config
    )
    
    # 更新模型参数
    updated_model_args = model_args.copy()
    updated_model_args.update(generation_config)
    
    # 创建PPO训练器
    ppo_trainer = ppo_trainer_class(**updated_model_args)
    
    print("PPO trainer with AdaptiveEntropyProcessor setup complete")
    return ppo_trainer
