from typing import Optional

import deepspeed
import torch

import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.integrations.deepspeed import HfDeepSpeedConfig
from transformers import BitsAndBytesConfig
from openrlhf.utils.logging_utils import init_logger
from peft import get_peft_model, LoraConfig,TaskType
from peft.tuners.lora import LoraLayer
from openrlhf.models.ring_attn_utils import gather_and_pad_tensor, unpad_and_slice_tensor


class CreditModel(nn.Module):
    """
    用於信用分配的迴歸模型，其結構仿照 OpenRLHF 的 Actor 模型。

    Args:
        pretrain_or_model (Union[str, nn.Module]): 預訓練模型的路徑或已載入的模型實例。
        num_outputs (int): 迴歸頭的輸出維度（例如，7個子句的信用分數）。
        ... (其他參數與 Actor 類別相同)
    """

    def __init__(
            self,
            pretrain_or_model,
            num_outputs: int = 8,
            use_flash_attention_2=False,
            bf16=True,
            load_in_4bit=False,
            lora_rank=0,
            lora_alpha=16,
            lora_dropout=0,
            target_modules=None,
            ds_config=None,
            device_map=None,
            packing_samples=False,
            temperature=1.0,
            use_liger_kernel=False,
            **kwargs,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        # --- 1. 載入主幹模型 (Backbone) ---
        # 這部分程式碼幾乎完全複製自 Actor 的 __init__ 方法
        if isinstance(pretrain_or_model, str):
            attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

            if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
                dschf = HfDeepSpeedConfig(ds_config)
            else:
                dschf = None

            if load_in_4bit:
                assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                nf4_config = None
            if use_liger_kernel:
                from liger_kernel.transformers import AutoLigerKernelForCausalLM

                model_class = AutoLigerKernelForCausalLM
            else:
                model_class = AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrain_or_model,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                quantization_config=nf4_config,
                torch_dtype=torch.bfloat16 if bf16 else "auto",
                device_map=device_map,
            )

            # LoRA (與 Actor 相同)
            if lora_rank > 0:
                self.model.enable_input_require_grads()
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                )
                self.model = get_peft_model(self.model, lora_config)

                if load_in_4bit:
                    # 處理混合精度 (與 Actor 相同)
                    for name, module in self.model.named_modules():
                        if isinstance(module, LoraLayer):
                            module = module.to(torch.bfloat16)
                        if "norm" in name:
                            module = module.to(torch.float32)
                        if "lm_head" in name or "embed_tokens" in name:
                            if hasattr(module, "weight"):
                                module = module.to(torch.bfloat16)

            self.model.config.use_cache = False
        else:
            self.model = pretrain_or_model

        # --- 2. 建立迴歸頭 (Regression Head) ---
        # 這是與 Actor 類別的主要不同之處
        hidden_size = self.model.config.hidden_size
        self.regression_head = nn.Linear(hidden_size, num_outputs, bias=False)

        # 初始化新頭部的權重
        self.regression_head.weight.data.normal_(mean=0.0, std=1 / (hidden_size + 1))

    @property
    def config(self):
        """提供对底层模型config的访问"""
        return self.model.config

    def save_pretrained(self, save_directory, state_dict=None, **kwargs):
        """保存预训练模型 - 兼容 DeepSpeed"""
        import os
        import json
        
        # 创建保存目录
        os.makedirs(save_directory, exist_ok=True)
        
        if state_dict is not None:
            # DeepSpeed 模式：从 state_dict 中分离权重
            transformer_state_dict = {}
            regression_head_state_dict = {}
            
            for key, value in state_dict.items():
                if key.startswith('regression_head.'):
                    # 回归头的权重，去掉前缀
                    new_key = key.replace('regression_head.', '')
                    regression_head_state_dict[new_key] = value
                else:
                    # transformer 模型的权重
                    transformer_state_dict[key] = value
            
            # 保存 transformer 模型
            self.model.save_pretrained(save_directory, state_dict=transformer_state_dict, **kwargs)
            
            # 保存回归头权重
            if regression_head_state_dict:
                regression_head_path = os.path.join(save_directory, "regression_head.bin")
                torch.save(regression_head_state_dict, regression_head_path)
        else:
            # 常规模式：直接保存当前状态
            self.model.save_pretrained(save_directory, **kwargs)
            
            # 保存回归头的权重
            regression_head_path = os.path.join(save_directory, "regression_head.bin")
            torch.save(self.regression_head.state_dict(), regression_head_path)
        
        # 保存模型的元信息
        meta_info = {
            "model_type": "CreditModel",
            "num_outputs": self.regression_head.out_features,
            "hidden_size": self.regression_head.in_features,
        }
        meta_path = os.path.join(save_directory, "credit_model_meta.json")
        with open(meta_path, 'w') as f:
            json.dump(meta_info, f, indent=2)

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        """从预训练模型加载"""
        import os
        import json
        
        # 确定num_outputs
        if "num_outputs" in kwargs:
            num_outputs = kwargs.pop("num_outputs")
        else:
            meta_path = os.path.join(model_path, "credit_model_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta_info = json.load(f)
                num_outputs = meta_info.get("num_outputs", 8)
            else:
                num_outputs = 8
        
        # 创建模型实例
        model = cls(model_path, num_outputs=num_outputs, **kwargs)
        
        # 尝试加载回归头的权重
        regression_head_path = os.path.join(model_path, "regression_head.bin")
        if os.path.exists(regression_head_path):
            regression_head_state = torch.load(regression_head_path, map_location="cpu")
            model.regression_head.load_state_dict(regression_head_state)
            print(f"Loaded regression head weights from {regression_head_path}")
        else:
            print("No regression head weights found, using randomly initialized weights")
        
        return model

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Credit Model 的前向傳播方法。
        """
        # 獲取主幹模型的輸出
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,  # 確保能獲取 hidden states
        )

        # 使用最後一層的 hidden states
        # outputs[0] 通常是 last_hidden_state
        last_hidden_state = outputs.hidden_states[-1]

        # 使用最後一個有效 token 的 hidden state 作為整個序列的表徵
        # 我們需要找到每個序列中最後一個 non-padding token 的位置
        if attention_mask is not None:
            last_token_indices = attention_mask.sum(dim=1) - 1
            sequence_representation = last_hidden_state[torch.arange(last_hidden_state.size(0)), last_token_indices]
        else:
            # 如果沒有 attention_mask，則假設沒有 padding，取最後一個
            sequence_representation = last_hidden_state[:, -1, :]

        # 將序列的表徵傳入迴歸頭，得到預測的信用分數
        predicted_credits = self.regression_head(sequence_representation)

        return predicted_credits

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()