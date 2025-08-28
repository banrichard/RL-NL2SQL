"""
AdaptiveEntropyProcessor: 自适应熵处理器
根据模型输出的置信度动态调整Top-K采样策略
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Any
import re
from transformers.generation.logits_process import LogitsProcessor


class AdaptiveEntropyLogitsProcessor(LogitsProcessor):
    """
    自适应熵Logits处理器
    
    核心机制：
    1. 计算当前输出分布的熵值（不确定性度量）
    2. 根据熵值自适应选择探索策略：
       - 低熵（高置信度）-> 利用策略（小k值）
       - 高熵（低置信度）-> 探索策略（大k值）
    3. 应用Top-K过滤，只保留前k个最可能的token
    """
    
    def __init__(
        self,
        low_entropy_threshold: float = 2.0,
        high_entropy_threshold: float = 4.0,
        low_k: int = 5,
        high_k: int = 50,
        enable_content_filtering: bool = True,
        tokenizer=None,
        device: str = "cuda"
    ):
        """
        初始化自适应熵处理器
        
        Args:
            low_entropy_threshold: 低熵阈值，低于此值采用利用策略
            high_entropy_threshold: 高熵阈值，高于此值采用探索策略
            low_k: 利用策略的Top-K值（小值，更专注）
            high_k: 探索策略的Top-K值（大值，更多样）
            enable_content_filtering: 是否启用内容过滤（过滤thinking等无效内容）
            tokenizer: tokenizer对象（用于内容过滤）
            device: 计算设备
        """
        self.low_entropy_threshold = low_entropy_threshold
        self.high_entropy_threshold = high_entropy_threshold
        self.low_k = low_k
        self.high_k = high_k
        self.enable_content_filtering = enable_content_filtering
        self.tokenizer = tokenizer
        self.device = device
        
        # SQL生成触发词
        self.sql_generation_trigger = "<answer>```sql"
        
        # 内容过滤模式
        self.thinking_patterns = [
            r"<thinking>",
            r"let me think",
            r"我需要思考",
            r"让我想想",
            r"首先分析",
            r"thinking:",
            r"解释:",
            r"explanation:",
        ]
        
        # SQL关键词模式
        self.sql_patterns = [
            r"SELECT\s",
            r"FROM\s", 
            r"WHERE\s",
            r"JOIN\s",
            r"GROUP BY",
            r"ORDER BY",
            r"INSERT\s",
            r"UPDATE\s",
            r"DELETE\s",
        ]
        
        # 统计信息
        self.stats = {
            "total_calls": 0,
            "low_entropy_calls": 0,
            "high_entropy_calls": 0,
            "medium_entropy_calls": 0,
            "filtered_calls": 0,
            "avg_entropy": 0.0,
            "avg_k_used": 0.0
        }
        
        print(f"AdaptiveEntropyProcessor initialized:")
        print(f"  Low entropy threshold: {low_entropy_threshold} (k={low_k})")
        print(f"  High entropy threshold: {high_entropy_threshold} (k={high_k})")
        print(f"  Content filtering: {enable_content_filtering}")
    
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        处理logits的核心函数
        
        Args:
            input_ids: 当前输入的token序列 [batch_size, seq_len]
            scores: 模型输出的logits [batch_size, vocab_size]
            
        Returns:
            处理后的logits [batch_size, vocab_size]
        """
        self.stats["total_calls"] += 1
        
        # 检查是否应该处理当前步骤
        if not self._should_process_current_step(input_ids):
            self.stats["filtered_calls"] += 1
            return scores
        
        # 批处理：为每个样本独立计算
        processed_scores = torch.zeros_like(scores)
        
        for i in range(scores.shape[0]):
            current_logits = scores[i]  # [vocab_size]
            processed_logits = self._process_single_sample(current_logits)
            processed_scores[i] = processed_logits
        
        return processed_scores
    
    def _process_single_sample(self, logits: torch.Tensor) -> torch.Tensor:
        """
        处理单个样本的logits
        
        Args:
            logits: 单个样本的logits [vocab_size]
            
        Returns:
            处理后的logits [vocab_size]
        """
        # 1. 计算熵值
        entropy = self._calculate_entropy(logits)
        
        # 2. 自适应决策：根据熵值选择k
        k = self._adaptive_k_selection(entropy)
        
        # 3. 应用Top-K过滤
        filtered_logits = self._apply_topk_filter(logits, k)
        
        # 4. 更新统计信息
        self._update_stats(entropy, k)
        
        return filtered_logits
    
    def _calculate_entropy(self, logits: torch.Tensor) -> float:
        """
        计算logits分布的信息熵
        
        Args:
            logits: [vocab_size] 的logits张量
            
        Returns:
            熵值（float）
        """
        # 将logits转换为概率分布
        probs = F.softmax(logits, dim=-1)
        
        # 避免log(0)的数值问题
        probs = torch.clamp(probs, min=1e-12)
        
        # 计算信息熵: H = -Σ(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs)).item()
        
        return entropy
    
    def _adaptive_k_selection(self, entropy: float) -> int:
        """
        根据熵值自适应选择k值
        
        Args:
            entropy: 当前分布的熵值
            
        Returns:
            选择的k值
        """
        if entropy < self.low_entropy_threshold:
            # 低熵：模型很自信，采用利用策略（小k）
            self.stats["low_entropy_calls"] += 1
            return self.low_k
        elif entropy > self.high_entropy_threshold:
            # 高熵：模型不确定，采用探索策略（大k）
            self.stats["high_entropy_calls"] += 1
            return self.high_k
        else:
            # 中等熵：线性插值
            self.stats["medium_entropy_calls"] += 1
            ratio = (entropy - self.low_entropy_threshold) / (
                self.high_entropy_threshold - self.low_entropy_threshold
            )
            k = int(self.low_k + ratio * (self.high_k - self.low_k))
            return max(self.low_k, min(k, self.high_k))
    
    def _apply_topk_filter(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """
        应用Top-K过滤
        
        Args:
            logits: 原始logits [vocab_size]
            k: 保留的top-k数量
            
        Returns:
            过滤后的logits [vocab_size]
        """
        if k >= logits.size(-1):
            # 如果k大于等于词汇表大小，不需要过滤
            return logits
        
        # 获取top-k的值和索引
        top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)
        
        # 创建新的logits张量，非top-k的位置设为-inf
        filtered_logits = torch.full_like(logits, float('-inf'))
        filtered_logits.scatter_(-1, top_k_indices, top_k_values)
        
        return filtered_logits
    
    def _should_process_current_step(self, input_ids: torch.Tensor) -> bool:
        """
        判断当前生成步骤是否应该进行熵处理
        过滤掉thinking过程和解释内容
        
        Args:
            input_ids: 当前输入序列 [batch_size, seq_len]
            
        Returns:
            是否应该处理当前步骤
        """
        if not self.enable_content_filtering or self.tokenizer is None:
            return True
        
        # 只检查第一个样本
        if input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            return True
        
        try:
            # 检查最近的tokens
            recent_length = min(100, input_ids.shape[1])
            last_tokens = input_ids[0, -recent_length:]
            decoded_text = self.tokenizer.decode(last_tokens, skip_special_tokens=True)
            decoded_text_lower = decoded_text.lower()
            
            # 检查是否包含SQL生成触发词
            sql_triggered = self.sql_generation_trigger in decoded_text
            
            # 检查是否在thinking过程中
            in_thinking = any(
                re.search(pattern, decoded_text_lower, re.IGNORECASE) 
                for pattern in self.thinking_patterns
            )
            
            # 检查是否在生成SQL内容
            in_sql_generation = any(
                re.search(pattern, decoded_text_lower, re.IGNORECASE)
                for pattern in self.sql_patterns
            )
            
            # 只在已触发SQL生成且不在thinking过程中时进行处理
            if sql_triggered and not in_thinking:
                return True
            
            # 如果明确在生成SQL内容，也进行处理
            if in_sql_generation and not in_thinking:
                return True
            
            return False
            
        except Exception as e:
            print(f"Warning: Content filtering failed: {e}")
            return True
    
    def _update_stats(self, entropy: float, k: int):
        """更新统计信息"""
        # 只统计实际处理的调用
        processed_calls = (self.stats["low_entropy_calls"] + 
                          self.stats["high_entropy_calls"] + 
                          self.stats["medium_entropy_calls"])
        
        if processed_calls == 0:
            self.stats["avg_entropy"] = entropy
            self.stats["avg_k_used"] = k
        else:
            # 更新平均熵值
            self.stats["avg_entropy"] = (
                (self.stats["avg_entropy"] * (processed_calls - 1) + entropy) / processed_calls
            )
            
            # 更新平均k值
            self.stats["avg_k_used"] = (
                (self.stats["avg_k_used"] * (processed_calls - 1) + k) / processed_calls
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = self.stats["total_calls"]
        processed = (self.stats["low_entropy_calls"] + 
                    self.stats["high_entropy_calls"] + 
                    self.stats["medium_entropy_calls"])
        
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            "processed_calls": processed,
            "processing_ratio": processed / total if total > 0 else 0,
            "filtering_ratio": self.stats["filtered_calls"] / total if total > 0 else 0,
            "low_entropy_ratio": self.stats["low_entropy_calls"] / processed if processed > 0 else 0,
            "high_entropy_ratio": self.stats["high_entropy_calls"] / processed if processed > 0 else 0,
            "medium_entropy_ratio": self.stats["medium_entropy_calls"] / processed if processed > 0 else 0,
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            "total_calls": 0,
            "low_entropy_calls": 0,
            "high_entropy_calls": 0,
            "medium_entropy_calls": 0,
            "filtered_calls": 0,
            "avg_entropy": 0.0,
            "avg_k_used": 0.0
        }
        print("AdaptiveEntropyProcessor stats reset")


def create_adaptive_entropy_processor(
    low_entropy_threshold: float = 2.0,
    high_entropy_threshold: float = 4.0,
    low_k: int = 5,
    high_k: int = 50,
    enable_content_filtering: bool = True,
    tokenizer=None,
    device: str = "cuda"
) -> AdaptiveEntropyLogitsProcessor:
    """
    创建自适应熵处理器的工厂函数
    
    Args:
        low_entropy_threshold: 低熵阈值
        high_entropy_threshold: 高熵阈值  
        low_k: 利用策略的k值
        high_k: 探索策略的k值
        enable_content_filtering: 是否启用内容过滤
        tokenizer: tokenizer对象（用于内容过滤）
        device: 计算设备
        
    Returns:
        配置好的AdaptiveEntropyLogitsProcessor实例
    """
    processor = AdaptiveEntropyLogitsProcessor(
        low_entropy_threshold=low_entropy_threshold,
        high_entropy_threshold=high_entropy_threshold,
        low_k=low_k,
        high_k=high_k,
        enable_content_filtering=enable_content_filtering,
        tokenizer=tokenizer,
        device=device
    )
    
    return processor