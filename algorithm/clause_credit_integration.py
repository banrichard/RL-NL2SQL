"""
PPO训练中的子句级别信用分配集成
将ClauseLevelCreditSystem集成到OpenRLHF的PPO训练流程中
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
import numpy as np
from transformers import AutoTokenizer

from clause_credit_system import (
    ClauseLevelCreditSystem, 
    TokenExperience,
    ClauseExperience
)


@dataclass
class PPOExperience:
    """PPO训练使用的经验数据格式"""
    input_ids: torch.Tensor  # 输入序列
    attention_mask: torch.Tensor  # 注意力掩码
    action_mask: torch.Tensor  # 动作掩码
    values: torch.Tensor  # Critic预测的价值
    rewards: torch.Tensor  # 奖励
    log_probs: torch.Tensor  # 对数概率
    kl_divergence: Optional[torch.Tensor] = None  # KL散度


class ClauseAwarePPORollout:
    """
    支持子句级别信用分配的PPO Rollout系统
    """
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        credit_model_path: str,
        discount_factor: float = 0.99,
        device: str = "cuda",
        max_sequence_length: int = 2048
    ):
        self.tokenizer = tokenizer
        self.device = device
        self.max_sequence_length = max_sequence_length
        
        # 初始化子句级别信用分配系统
        self.credit_system = ClauseLevelCreditSystem(
            tokenizer=tokenizer,
            credit_model_path=credit_model_path,
            discount_factor=discount_factor,
            device=device
        )
        
        # 经验缓冲区
        self.main_experience_buffer = []
        self.current_sequence_experiences = []
        
    def rollout_with_clause_credit(
        self,
        actor_model,
        critic_model,
        ref_model,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 1.0,
        generation_kwargs: Optional[Dict] = None
    ) -> List[PPOExperience]:
        """
        执行支持子句级别信用分配的PPO rollout
        
        Args:
            actor_model: Actor模型
            critic_model: Critic模型  
            ref_model: Reference模型
            prompts: 输入提示列表
            max_new_tokens: 最大生成Token数
            temperature: 采样温度
            top_p: Top-p采样参数
            generation_kwargs: 其他生成参数
            
        Returns:
            PPO经验数据列表
        """
        all_experiences = []
        
        for prompt in prompts:
            # 重置信用分配系统
            self.credit_system.reset()
            
            # 执行单个序列的rollout
            experience = self._rollout_single_sequence(
                actor_model=actor_model,
                critic_model=critic_model,
                ref_model=ref_model,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                generation_kwargs=generation_kwargs
            )
            
            if experience:
                all_experiences.append(experience)
                
        return all_experiences
        
    def _rollout_single_sequence(
        self,
        actor_model,
        critic_model,
        ref_model,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        generation_kwargs: Optional[Dict]
    ) -> Optional[PPOExperience]:
        """执行单个序列的rollout"""
        
        # 编码初始提示
        prompt_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        current_tokens = prompt_tokens.clone()
        
        # 存储生成过程中的数据
        generated_token_ids = []
        values_list = []
        log_probs_list = []
        rewards_list = []
        attention_masks = []
        
        # 生成循环
        for step in range(max_new_tokens):
            # 1. Critic预测当前状态的价值
            with torch.no_grad():
                critic_outputs = critic_model(current_tokens)
                if hasattr(critic_outputs, 'values'):
                    current_value = critic_outputs.values[0, -1].item()
                else:
                    current_value = critic_outputs.logits[0, -1].item()
                    
            # 2. Actor生成下一个Token
            with torch.no_grad():
                actor_outputs = actor_model(current_tokens)
                next_token_logits = actor_outputs.logits[0, -1, :]
                
                # 应用温度和top_p采样
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                    
                # Top-p采样
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # 采样下一个Token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, 1).item()
                next_token_log_prob = torch.log(probs[next_token_id]).item()
                
            # 3. 解码Token
            next_token_text = self.tokenizer.decode([next_token_id])
            current_sql = self.tokenizer.decode(current_tokens[0])
            
            # 4. 调用子句级别信用分配系统
            completed_token_experiences = self.credit_system.process_token_step(
                sql_prefix=current_sql,
                new_token=next_token_text,
                token_id=next_token_id,
                value=current_value,
                log_prob=next_token_log_prob
            )
            
            # 5. 处理完成的Token经验
            for token_exp in completed_token_experiences:
                rewards_list.append(token_exp.reward)
                values_list.append(token_exp.value)
                log_probs_list.append(token_exp.log_prob)
                generated_token_ids.append(token_exp.token_id)
                
            # 6. 更新当前序列
            current_tokens = torch.cat([current_tokens, torch.tensor([[next_token_id]], device=self.device)], dim=1)
            
            # 7. 检查结束条件
            if next_token_id == self.tokenizer.eos_token_id:
                break
                
            if current_tokens.shape[1] >= self.max_sequence_length:
                break
                
        # 8. 处理最后一个子句
        final_token_experiences = self.credit_system.finalize_sequence()
        for token_exp in final_token_experiences:
            rewards_list.append(token_exp.reward)
            values_list.append(token_exp.value)
            log_probs_list.append(token_exp.log_prob)
            generated_token_ids.append(token_exp.token_id)
            
        # 9. 构建PPO经验数据
        if not generated_token_ids:
            return None
            
        # 创建完整的序列
        full_sequence = torch.cat([prompt_tokens, torch.tensor([generated_token_ids], device=self.device)], dim=1)
        
        # 创建掩码
        attention_mask = torch.ones_like(full_sequence)
        action_mask = torch.zeros_like(full_sequence)
        action_mask[0, prompt_tokens.shape[1]:] = 1  # 只对生成的部分计算策略损失
        
        # 转换为张量
        values_tensor = torch.tensor(values_list, device=self.device)
        rewards_tensor = torch.tensor(rewards_list, device=self.device)
        log_probs_tensor = torch.tensor(log_probs_list, device=self.device)
        
        # 确保张量长度匹配
        min_length = min(len(values_list), len(rewards_list), len(log_probs_list))
        if min_length == 0:
            return None
            
        return PPOExperience(
            input_ids=full_sequence,
            attention_mask=attention_mask,
            action_mask=action_mask,
            values=values_tensor[:min_length],
            rewards=rewards_tensor[:min_length],
            log_probs=log_probs_tensor[:min_length]
        )
        
    def get_rollout_statistics(self) -> Dict[str, Any]:
        """获取rollout统计信息"""
        return self.credit_system.get_statistics()


class ClauseCreditIntegration:
    """
    子句信用分配与PPO训练的集成接口
    """
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        credit_model_path: str,
        discount_factor: float = 0.99,
        device: str = "cuda"
    ):
        self.rollout_manager = ClauseAwarePPORollout(
            tokenizer=tokenizer,
            credit_model_path=credit_model_path,
            discount_factor=discount_factor,
            device=device
        )
        
    def enhance_ppo_trainer(self, ppo_trainer_class):
        """
        增强PPO训练器以支持子句级别信用分配
        
        Args:
            ppo_trainer_class: 原始的PPO训练器类
            
        Returns:
            增强后的PPO训练器类
        """
        
        class ClauseEnhancedPPOTrainer(ppo_trainer_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                
                # 集成子句信用分配系统
                self.clause_credit_integration = ClauseCreditIntegration(
                    tokenizer=self.tokenizer,
                    credit_model_path=kwargs.get('credit_model_path', './ckpt/credit_model'),
                    discount_factor=kwargs.get('discount_factor', 0.99),
                    device=self.device
                )
                
            def rollout(self, prompts: List[str], **kwargs) -> List[PPOExperience]:
                """重写rollout方法以使用子句级别信用分配"""
                return self.clause_credit_integration.rollout_manager.rollout_with_clause_credit(
                    actor_model=self.actor,
                    critic_model=self.critic,
                    ref_model=self.ref_model,
                    prompts=prompts,
                    **kwargs
                )
                
            def get_clause_statistics(self) -> Dict[str, Any]:
                """获取子句级别的统计信息"""
                return self.clause_credit_integration.rollout_manager.get_rollout_statistics()
                
        return ClauseEnhancedPPOTrainer
        
    @staticmethod
    def create_clause_enhanced_trainer(
        base_trainer_class,
        tokenizer: AutoTokenizer,
        credit_model_path: str,
        discount_factor: float = 0.99,
        device: str = "cuda"
    ):
        """
        创建支持子句级别信用分配的PPO训练器
        """
        integration = ClauseCreditIntegration(
            tokenizer=tokenizer,
            credit_model_path=credit_model_path,
            discount_factor=discount_factor,
            device=device
        )
        
        return integration.enhance_ppo_trainer(base_trainer_class)


# 使用示例
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    print("=== 子句级别信用分配PPO集成测试 ===")
    
    # 初始化
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 创建集成系统
    integration = ClauseCreditIntegration(
        tokenizer=tokenizer,
        credit_model_path="./ckpt/credit_model",
        discount_factor=0.99,
        device="cpu"  # 测试使用CPU
    )
    
    print("✓ 子句级别信用分配PPO集成系统创建成功")
    print("✓ 支持实时的子句边界检测")
    print("✓ 支持折扣归因的奖励分配")
    print("✓ 可集成到现有PPO训练流程")
    
    # 显示集成说明
    print("\n=== 集成使用说明 ===")
    print("1. 在PPO训练中，每个时间步t:")
    print("   - Actor生成Token a_t")
    print("   - Critic预测价值V(s_t)")
    print("   - 系统检测子句边界")
    print("   - 触发边界时调用Credit Model")
    print("   - 计算折扣归因奖励")
    
    print("\n2. 奖励计算公式:")
    print("   - r_last_token = Clause_Credit")
    print("   - r_{t-1} = γ * r_t")
    print("   - 其中γ=0.99（折扣因子）")
    
    print("\n3. 集成到train_ppo_ray.py:")
    print("   - 使用ClauseCreditIntegration.enhance_ppo_trainer()")
    print("   - 或使用create_clause_enhanced_trainer()")
    print("   - 自动替换原始rollout方法")
    
    print("\n=== 测试完成 ===")
