"""
子句级别的Credit Model系统
实现PPO训练中的实时信用分配和折扣归因
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Any
import re
from collections import deque
from dataclasses import dataclass
import numpy as np
from transformers import AutoTokenizer


@dataclass
class TokenExperience:
    """单个Token的经验数据"""
    state: str  # 状态s_t (SQL前缀)
    action: str  # 动作a_t (生成的Token)
    token_id: int  # Token ID
    value: float  # V(s_t) Critic预测的价值
    log_prob: float  # log_prob(a_t) 动作的对数概率
    reward: Optional[float] = None  # r_t 即时奖励（延迟分配）
    step: int = 0  # 时间步


@dataclass
class ClauseExperience:
    """子句级别的经验数据"""
    clause_type: str  # 子句类型 (SELECT, FROM, WHERE等)
    tokens: List[TokenExperience]  # 子句内的所有Token经验
    clause_text: str  # 完整的子句文本
    clause_credit: Optional[float] = None  # 子句的总信用分


class ClauseParser:
    """SQL子句解析器"""
    
    # SQL关键字到子句类型的映射
    CLAUSE_KEYWORDS = {
        "SELECT": "SELECT",
        "FROM": "FROM", 
        "WHERE": "WHERE",
        "GROUP": "GROUP_BY",  # GROUP BY
        "HAVING": "HAVING",
        "ORDER": "ORDER_BY",  # ORDER BY
        "LIMIT": "LIMIT",
        "WITH": "WITH",
        "UNION": "UNION",
        "JOIN": "JOIN",
        "INNER": "JOIN",
        "LEFT": "JOIN", 
        "RIGHT": "JOIN",
        "FULL": "JOIN"
    }
    
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        
    def identify_clause_boundary(self, current_sql: str, new_token: str) -> Tuple[bool, Optional[str]]:
        """
        识别是否到达子句边界
        
        Args:
            current_sql: 当前的SQL前缀
            new_token: 新生成的Token
            
        Returns:
            (is_boundary, clause_type): 是否是边界以及当前子句类型
        """
        # 去除空格和特殊符号的Token
        clean_token = new_token.strip().upper()
        
        # 检查是否是SQL关键字
        if clean_token in self.CLAUSE_KEYWORDS:
            return True, self.CLAUSE_KEYWORDS[clean_token]
            
        # 检查结束符
        if new_token in ['</s>', '<|endoftext|>', self.tokenizer.eos_token]:
            return True, "END"
            
        # 检查多词关键字 (GROUP BY, ORDER BY)
        sql_tokens = current_sql.split()
        if len(sql_tokens) >= 2:
            last_two = " ".join(sql_tokens[-2:]).upper()
            if last_two in ["GROUP BY", "ORDER BY"]:
                return True, self.CLAUSE_KEYWORDS.get(sql_tokens[-2].upper(), "UNKNOWN")
                
        return False, None
        
    def get_current_clause_type(self, sql_prefix: str) -> str:
        """获取当前所在的子句类型"""
        sql_upper = sql_prefix.upper()
        
        # 按照优先级检查关键字
        keywords = ["WITH", "SELECT", "FROM", "WHERE", "GROUP BY", "HAVING", "ORDER BY", "LIMIT"]
        
        current_clause = "UNKNOWN"
        for keyword in keywords:
            if keyword in sql_upper:
                current_clause = keyword.replace(" ", "_")
                
        return current_clause


class ClauseBufferManager:
    """子句缓存管理器"""
    
    def __init__(self, discount_factor: float = 0.99):
        self.discount_factor = discount_factor
        self.current_clause_buffer = []  # 当前子句的Token缓存
        self.completed_clauses = []  # 已完成的子句
        self.current_clause_type = "UNKNOWN"
        
    def add_token_experience(self, experience: TokenExperience):
        """添加Token经验到当前子句缓存"""
        self.current_clause_buffer.append(experience)
        
    def complete_current_clause(self, clause_credit: float) -> List[TokenExperience]:
        """
        完成当前子句，计算折扣归因奖励
        
        Args:
            clause_credit: 子句的总信用分
            
        Returns:
            带有奖励的Token经验列表
        """
        if not self.current_clause_buffer:
            return []
            
        # 创建子句经验
        clause_text = " ".join([exp.action for exp in self.current_clause_buffer])
        clause_exp = ClauseExperience(
            clause_type=self.current_clause_type,
            tokens=self.current_clause_buffer.copy(),
            clause_text=clause_text,
            clause_credit=clause_credit
        )
        
        # 计算折扣归因奖励（从后往前）
        num_tokens = len(self.current_clause_buffer)
        for i in reversed(range(num_tokens)):
            if i == num_tokens - 1:
                # 最后一个Token获得完整的子句信用
                self.current_clause_buffer[i].reward = clause_credit
            else:
                # 前面的Token获得折扣后的奖励
                self.current_clause_buffer[i].reward = (
                    self.current_clause_buffer[i + 1].reward * self.discount_factor
                )
                
        # 保存完成的子句
        self.completed_clauses.append(clause_exp)
        
        # 返回带奖励的经验并清空缓存
        completed_tokens = self.current_clause_buffer.copy()
        self.current_clause_buffer = []
        
        return completed_tokens
        
    def start_new_clause(self, clause_type: str):
        """开始新的子句"""
        self.current_clause_type = clause_type
        
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "total_completed_clauses": len(self.completed_clauses),
            "current_clause_tokens": len(self.current_clause_buffer),
            "current_clause_type": self.current_clause_type
        }
        
        # 按子句类型统计
        clause_types = {}
        for clause in self.completed_clauses:
            clause_type = clause.clause_type
            if clause_type not in clause_types:
                clause_types[clause_type] = {"count": 0, "avg_credit": 0.0, "total_credit": 0.0}
            clause_types[clause_type]["count"] += 1
            clause_types[clause_type]["total_credit"] += clause.clause_credit or 0.0
            clause_types[clause_type]["avg_credit"] = (
                clause_types[clause_type]["total_credit"] / clause_types[clause_type]["count"]
            )
            
        stats["clause_statistics"] = clause_types
        return stats


class CreditAssignmentModel:
    """信用分配模型接口"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        # 这里应该加载您训练好的Credit Model
        # self.model = load_credit_model(model_path)
        
    def predict_clause_credit(self, sql_prefix: str, clause_text: str) -> float:
        """
        预测子句的信用分
        
        Args:
            sql_prefix: SQL前缀（上下文）
            clause_text: 子句文本
            
        Returns:
            子句的信用分数
        """
        # 模拟信用分配（实际应该调用训练好的模型）
        # 这里根据子句类型返回不同的基准分数
        clause_upper = clause_text.upper()
        
        if "SELECT" in clause_upper:
            base_score = 8.0
        elif "FROM" in clause_upper:
            base_score = 6.0
        elif "WHERE" in clause_upper:
            base_score = 7.0
        elif "GROUP BY" in clause_upper or "ORDER BY" in clause_upper:
            base_score = 5.0
        else:
            base_score = 3.0
            
        # 添加一些随机性来模拟真实的模型预测
        import random
        noise = random.uniform(-1.0, 1.0)
        return base_score + noise


class ClauseLevelCreditSystem:
    """子句级别的信用分配系统"""
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        credit_model_path: str,
        discount_factor: float = 0.99,
        device: str = "cuda"
    ):
        self.tokenizer = tokenizer
        self.parser = ClauseParser(tokenizer)
        self.buffer_manager = ClauseBufferManager(discount_factor)
        self.credit_model = CreditAssignmentModel(credit_model_path, device)
        self.current_sql_prefix = ""
        self.step_counter = 0
        
    def process_token_step(
        self,
        sql_prefix: str,
        new_token: str,
        token_id: int,
        value: float,
        log_prob: float
    ) -> List[TokenExperience]:
        """
        处理单个Token步骤
        
        Args:
            sql_prefix: 当前的SQL前缀
            new_token: 新生成的Token
            token_id: Token ID
            value: Critic预测的价值
            log_prob: 动作的对数概率
            
        Returns:
            完成的Token经验列表（如果触发了子句边界）
        """
        self.step_counter += 1
        
        # 创建Token经验
        token_exp = TokenExperience(
            state=sql_prefix,
            action=new_token,
            token_id=token_id,
            value=value,
            log_prob=log_prob,
            step=self.step_counter
        )
        
        # 检查是否到达子句边界
        is_boundary, new_clause_type = self.parser.identify_clause_boundary(
            self.current_sql_prefix, new_token
        )
        
        completed_tokens = []
        
        if is_boundary and self.buffer_manager.current_clause_buffer:
            # 完成当前子句
            current_clause_text = " ".join([
                exp.action for exp in self.buffer_manager.current_clause_buffer
            ])
            
            # 调用Credit Model计算子句信用
            clause_credit = self.credit_model.predict_clause_credit(
                sql_prefix, current_clause_text
            )
            
            # 完成当前子句并获得带奖励的Token经验
            completed_tokens = self.buffer_manager.complete_current_clause(clause_credit)
            
            # 开始新子句
            if new_clause_type and new_clause_type != "END":
                self.buffer_manager.start_new_clause(new_clause_type)
                
        # 添加当前Token到缓存
        if new_clause_type != "END":
            self.buffer_manager.add_token_experience(token_exp)
            
        # 更新SQL前缀
        self.current_sql_prefix = sql_prefix + new_token
        
        return completed_tokens
        
    def finalize_sequence(self) -> List[TokenExperience]:
        """结束序列，处理最后一个子句"""
        if self.buffer_manager.current_clause_buffer:
            current_clause_text = " ".join([
                exp.action for exp in self.buffer_manager.current_clause_buffer
            ])
            
            clause_credit = self.credit_model.predict_clause_credit(
                self.current_sql_prefix, current_clause_text
            )
            
            return self.buffer_manager.complete_current_clause(clause_credit)
        return []
        
    def reset(self):
        """重置系统状态"""
        self.current_sql_prefix = ""
        self.step_counter = 0
        self.buffer_manager = ClauseBufferManager(self.buffer_manager.discount_factor)
        
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.buffer_manager.get_statistics()
        stats["total_steps"] = self.step_counter
        stats["current_sql_length"] = len(self.current_sql_prefix.split())
        return stats


# 使用示例和测试代码
if __name__ == "__main__":
    # 创建测试系统
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    
    credit_system = ClauseLevelCreditSystem(
        tokenizer=tokenizer,
        credit_model_path="./ckpt/credit_model",
        discount_factor=0.99
    )
    
    # 模拟Token生成过程
    test_tokens = ["SELECT", " name", ",", " age", " FROM", " users", " WHERE", " age", " >", " 18"]
    sql_prefix = ""
    
    print("=== 子句级别信用分配系统测试 ===")
    
    all_completed_experiences = []
    
    for i, token in enumerate(test_tokens):
        # 模拟Critic和Actor的输出
        mock_value = float(len(test_tokens) - i)  # 递减的价值
        mock_log_prob = -0.1 * (i + 1)  # 模拟对数概率
        
        print(f"\n步骤 {i+1}: 生成Token '{token}'")
        print(f"  当前SQL前缀: '{sql_prefix}'")
        
        # 处理Token
        completed_tokens = credit_system.process_token_step(
            sql_prefix=sql_prefix,
            new_token=token,
            token_id=i,
            value=mock_value,
            log_prob=mock_log_prob
        )
        
        sql_prefix += token
        
        if completed_tokens:
            print(f"  ✓ 完成子句，包含 {len(completed_tokens)} 个Token:")
            for j, exp in enumerate(completed_tokens):
                print(f"    Token {j+1}: '{exp.action}' -> 奖励: {exp.reward:.2f}")
            all_completed_experiences.extend(completed_tokens)
            
    # 结束序列
    final_tokens = credit_system.finalize_sequence()
    if final_tokens:
        print(f"\n最终子句包含 {len(final_tokens)} 个Token:")
        for j, exp in enumerate(final_tokens):
            print(f"  Token {j+1}: '{exp.action}' -> 奖励: {exp.reward:.2f}")
        all_completed_experiences.extend(final_tokens)
        
    # 显示统计信息
    print(f"\n=== 统计信息 ===")
    stats = credit_system.get_statistics()
    print(f"总步骤数: {stats['total_steps']}")
    print(f"完成的子句数: {stats['total_completed_clauses']}")
    print(f"子句统计: {stats['clause_statistics']}")
    
    print(f"\n=== 所有Token经验 ===")
    print(f"总共处理了 {len(all_completed_experiences)} 个Token")
    for exp in all_completed_experiences:
        print(f"步骤 {exp.step}: '{exp.action}' -> 奖励: {exp.reward:.2f}")