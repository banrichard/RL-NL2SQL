"""
分离式奖励计算系统
Credit Model处理中间步骤的即时奖励
Reward Model处理最终的终端奖励
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
import numpy as np
from transformers import AutoTokenizer

from .clause_credit_system import (
    ClauseLevelCreditSystem,
    TokenExperience,
    ClauseExperience
)


@dataclass
class SeparatedRewardExperience:
    """分离式奖励经验数据"""
    input_ids: torch.Tensor  # 输入序列
    attention_mask: torch.Tensor  # 注意力掩码
    action_mask: torch.Tensor  # 动作掩码
    values: torch.Tensor  # Critic预测的价值
    intermediate_rewards: torch.Tensor  # 中间步骤奖励（Credit Model）
    terminal_reward: Optional[float] = None  # 终端奖励（Reward Model）
    log_probs: torch.Tensor = None  # 对数概率
    kl_divergence: Optional[torch.Tensor] = None  # KL散度


class SeparatedRewardRollout:
    """
    分离式奖励PPO Rollout系统
    Credit Model处理中间奖励，Reward Model处理终端奖励
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

        # 初始化子句级别信用分配系统（中间奖励）
        self.credit_system = ClauseLevelCreditSystem(
            tokenizer=tokenizer,
            credit_model_path=credit_model_path,
            discount_factor=discount_factor,
            device=device
        )

        # 经验缓冲区
        self.experience_buffer = []

    def rollout_with_separated_rewards(
        self,
        actor_model,
        critic_model,
        ref_model,
        reward_model,  # 只在最后调用
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 1.0,
        generation_kwargs: Optional[Dict] = None
    ) -> List[SeparatedRewardExperience]:
        """
        执行分离式奖励PPO rollout

        时间步流程：
        t=0,1,...,T-1: 只调用Credit Model获取即时奖励r_t
        t=T: 只调用Reward Model获取终端奖励R_T
        """
        all_experiences = []

        for prompt in prompts:
            # 重置信用分配系统
            self.credit_system.reset()

            # 执行单个序列的rollout
            experience = self._rollout_single_sequence_separated(
                actor_model=actor_model,
                critic_model=critic_model,
                ref_model=ref_model,
                reward_model=reward_model,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                generation_kwargs=generation_kwargs
            )

            if experience:
                all_experiences.append(experience)

        return all_experiences

    def _rollout_single_sequence_separated(
        self,
        actor_model,
        critic_model,
        ref_model,
        reward_model,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        generation_kwargs: Optional[Dict]
    ) -> Optional[SeparatedRewardExperience]:
        """执行单个序列的分离式奖励rollout"""

        # 编码初始提示
        prompt_tokens = self.tokenizer.encode(
            prompt, return_tensors="pt").to(self.device)
        current_tokens = prompt_tokens.clone()

        # 存储生成过程中的数据
        generated_token_ids = []
        values_list = []
        log_probs_list = []
        intermediate_rewards_list = []  # Credit Model的即时奖励

        print(f"开始生成序列，提示: {prompt[:50]}...")

        # === 中间步骤：Credit Model主导 (t = 0, 1, ..., T-1) ===
        for step in range(max_new_tokens):
            print(f"时间步 {step + 1}: Credit Model工作中...")

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
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1)
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

            # 4. 【关键】调用Credit Model进行即时奖励计算
            # 注意：此时Reward Model处于"休眠"状态，不被调用
            completed_token_experiences = self.credit_system.process_token_step(
                sql_prefix=current_sql,
                new_token=next_token_text,
                token_id=next_token_id,
                value=current_value,
                log_prob=next_token_log_prob
            )

            # 5. 处理Credit Model返回的即时奖励
            for token_exp in completed_token_experiences:
                intermediate_rewards_list.append(
                    token_exp.reward)  # Credit Model的r_t
                values_list.append(token_exp.value)
                log_probs_list.append(token_exp.log_prob)
                generated_token_ids.append(token_exp.token_id)
                print(
                    f"  Token '{token_exp.action}' -> Credit奖励: {token_exp.reward:.3f}")

            # 6. 更新当前序列
            current_tokens = torch.cat([current_tokens, torch.tensor(
                [[next_token_id]], device=self.device)], dim=1)

            # 7. 检查结束条件
            if next_token_id == self.tokenizer.eos_token_id:
                print(f"遇到结束符，准备调用Reward Model...")
                break

            if current_tokens.shape[1] >= self.max_sequence_length:
                print(f"达到最大长度，准备调用Reward Model...")
                break

        # 8. 处理最后一个子句的Credit Model奖励
        final_token_experiences = self.credit_system.finalize_sequence()
        for token_exp in final_token_experiences:
            intermediate_rewards_list.append(token_exp.reward)
            values_list.append(token_exp.value)
            log_probs_list.append(token_exp.log_prob)
            generated_token_ids.append(token_exp.token_id)
            print(
                f"  最终Token '{token_exp.action}' -> Credit奖励: {token_exp.reward:.3f}")

        # === 终端步骤：Reward Model主导 (t = T) ===
        print(f"时间步 T: Reward Model工作中...")
        print("注意：此时Credit Model进入'休眠'状态")

        # 生成完整的SQL查询
        full_sequence = self.tokenizer.decode(current_tokens[0])
        generated_sql = full_sequence[len(prompt):].strip()

        # 【关键】调用Reward Model进行终端奖励计算
        # 注意：此时Credit Model处于"休眠"状态，不被调用
        terminal_reward = None
        if reward_model is not None:
            try:
                with torch.no_grad():
                    # 准备Reward Model的输入
                    reward_inputs = self.tokenizer(
                        full_sequence,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(self.device)

                    # 调用Reward Model获取终端奖励R_T
                    reward_outputs = reward_model(**reward_inputs)
                    if hasattr(reward_outputs, 'rewards'):
                        terminal_reward = reward_outputs.rewards[0].item()
                    else:
                        terminal_reward = reward_outputs.logits[0, -1].item()

                print(f"  完整SQL -> Reward奖励: {terminal_reward:.3f}")
                print(f"  生成的SQL: {generated_sql[:100]}...")

            except Exception as e:
                print(f"Reward Model调用失败: {e}")
                terminal_reward = 0.0
        else:
            print("未提供Reward Model，使用默认终端奖励")
            terminal_reward = 0.0

        # 9. 构建分离式奖励经验数据
        if not generated_token_ids:
            return None

        # 创建完整的序列
        full_sequence_tensor = torch.cat([prompt_tokens, torch.tensor(
            [generated_token_ids], device=self.device)], dim=1)

        # 创建掩码
        attention_mask = torch.ones_like(full_sequence_tensor)
        action_mask = torch.zeros_like(full_sequence_tensor)
        action_mask[0, prompt_tokens.shape[1]:] = 1  # 只对生成的部分计算策略损失

        # 转换为张量
        min_length = min(len(values_list), len(
            intermediate_rewards_list), len(log_probs_list))
        if min_length == 0:
            return None

        values_tensor = torch.tensor(
            values_list[:min_length], device=self.device)
        intermediate_rewards_tensor = torch.tensor(
            intermediate_rewards_list[:min_length], device=self.device)
        log_probs_tensor = torch.tensor(
            log_probs_list[:min_length], device=self.device)

        return SeparatedRewardExperience(
            input_ids=full_sequence_tensor,
            attention_mask=attention_mask,
            action_mask=action_mask,
            values=values_tensor,
            intermediate_rewards=intermediate_rewards_tensor,  # Credit Model的r_t序列
            terminal_reward=terminal_reward,  # Reward Model的R_T
            log_probs=log_probs_tensor
        )

    def get_rollout_statistics(self) -> Dict[str, Any]:
        """获取rollout统计信息"""
        stats = self.credit_system.get_statistics()
        stats['rollout_type'] = 'separated_rewards'
        stats['description'] = {
            'intermediate_phase': 'Credit Model provides r_t for each token/clause',
            'terminal_phase': 'Reward Model provides R_T for complete SQL',
            'separation': 'Credit Model and Reward Model never work simultaneously'
        }
        return stats


class SeparatedRewardPPOIntegration:
    """
    分离式奖励PPO训练集成
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        credit_model_path: str,
        discount_factor: float = 0.99,
        device: str = "cuda"
    ):
        self.rollout_manager = SeparatedRewardRollout(
            tokenizer=tokenizer,
            credit_model_path=credit_model_path,
            discount_factor=discount_factor,
            device=device
        )

    def enhance_ppo_trainer(self, ppo_trainer_class):
        """
        增强PPO训练器以支持分离式奖励计算
        """

        class SeparatedRewardPPOTrainer(ppo_trainer_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

                # 集成分离式奖励系统
                self.separated_reward_integration = SeparatedRewardPPOIntegration(
                    tokenizer=self.tokenizer,
                    credit_model_path=kwargs.get(
                        'credit_model_path', './ckpt/credit_model'),
                    discount_factor=kwargs.get('discount_factor', 0.99),
                    device=self.device
                )

            def rollout(self, prompts: List[str], **kwargs) -> List[SeparatedRewardExperience]:
                """重写rollout方法以使用分离式奖励计算"""
                print("=== 分离式奖励Rollout开始 ===")
                print("中间步骤: Credit Model主导")
                print("终端步骤: Reward Model主导")

                return self.separated_reward_integration.rollout_manager.rollout_with_separated_rewards(
                    actor_model=self.actor,
                    critic_model=self.critic,
                    ref_model=self.ref_model,
                    reward_model=self.reward_model,
                    prompts=prompts,
                    **kwargs
                )

            def get_separated_reward_statistics(self) -> Dict[str, Any]:
                """获取分离式奖励统计信息"""
                return self.separated_reward_integration.rollout_manager.get_rollout_statistics()

        return SeparatedRewardPPOTrainer


# 使用示例和测试代码
if __name__ == "__main__":
    from transformers import AutoTokenizer

    print("=== 分离式奖励系统测试 ===")

    # 初始化
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 创建分离式奖励系统
    separated_reward_system = SeparatedRewardRollout(
        tokenizer=tokenizer,
        credit_model_path="./ckpt/credit_model",
        discount_factor=0.99,
        device="cpu"
    )

    print("✓ 分离式奖励系统创建成功")

    # 显示工作流程
    print("\n=== 分离式奖励工作流程 ===")
    print("1. 中间步骤 (t = 0, 1, ..., T-1):")
    print("   - 主导者: Credit Model")
    print("   - 工作: 每生成一个Token/子句，输出即时奖励r_t")
    print("   - 状态: Reward Model休眠")

    print("\n2. 终端步骤 (t = T):")
    print("   - 主导者: Reward Model")
    print("   - 工作: 对完整SQL查询进行评估，输出终端奖励R_T")
    print("   - 状态: Credit Model休眠")

    print("\n3. 奖励分离特性:")
    print("   - Credit Model和Reward Model永不同时工作")
    print("   - 中间奖励序列: [r_0, r_1, ..., r_{T-1}] (Credit Model)")
    print("   - 终端奖励: R_T (Reward Model)")
    print("   - 总奖励: 中间奖励 + 终端奖励的组合")

    stats = separated_reward_system.get_rollout_statistics()
    print(f"\n=== 系统状态 ===")
    print(f"系统类型: {stats['rollout_type']}")
    print(f"描述: {stats['description']}")

    print("\n✓ 分离式奖励系统测试完成!")
    print("系统已准备好集成到PPO训练中")
