from typing import Callable

import torch
from torch.utils.data import Dataset
from openrlhf.utils.utils import zero_pad_sequences


def preprocess_data(
        data, input_template=None, input_key="input_text", output_key=None, apply_chat_template=None, multiturn=False
):
    if apply_chat_template:
        if output_key:
            prompt_message = data[input_key]
            response_message = data[output_key]

            if isinstance(prompt_message, str) and isinstance(response_message, str):
                prompt_message = [{"role": "user", "content": prompt_message}]
                response_message = [{"role": "assistant", "content": response_message}]

            prompt = apply_chat_template(prompt_message, tokenize=False, add_generation_prompt=True)
            response = apply_chat_template(prompt_message + response_message, tokenize=False)[len(prompt):]
        else:
            prompt = apply_chat_template(data[input_key][:-1], tokenize=False, add_generation_prompt=True)
            response = apply_chat_template(data[input_key], tokenize=False)[len(prompt):]
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)
        # output_key is None for continue pretrain
        response = data[output_key] if output_key else ""
    return prompt, response


class CreditDataset(Dataset):
    """
    專門用於訓練 Credit Model 的 Dataset。
    它仿照 SFTDataset 的結構，但適應了迴歸任務的需求。

    Args:
        dataset: Hugging Face Dataset 物件，已載入 credit_model_training_data.jsonl。
        tokenizer: 用於處理輸入文本的 tokenizer。
        max_length: 輸入序列的最大長度。
        strategy: 訓練策略物件，用於獲取 args。
    """

    def __init__(
            self,
            dataset,
            tokenizer: Callable,
            max_length: int,
            strategy,
            input_template=None,
            pretrain_mode=False,
            num_processors=8,
            multiturn=False,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.pretrain_mode = pretrain_mode
        self.max_length = max_length
        self.multiturn = multiturn
        # 從 strategy.args 中獲取我們需要的 key name
        self.input_key = getattr(self.strategy.args, "input_key", "input_text")
        self.output_key = getattr(self.strategy.args, "output_key", "target_credits")
        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)
        self.input_template = input_template
        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

        processed_dataset = dataset.map(
            self.process_data,
            num_proc=num_processors,
            remove_columns=dataset.column_names,
        )
        # 過濾掉處理後可能為 None 的無效樣本
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)

        # 將處理好的資料儲存為列表，方便快速索引
        self.input_ids = processed_dataset["prompt"]
        self.labels = processed_dataset["response"]

    def process_data(self, data):
        if self.multiturn and self.output_key:
            data[self.input_key].append(data[self.output_key])
            data[self.output_key] = None

        if self.multiturn:
            assert (
                    not self.output_key or not data[self.output_key]
            ), "You should put the whole trajactory into data[input_key] and do not set output_key"
            input_key = self.input_key
            apply_chat_template = self.apply_chat_template
            response_ranges = []
            for idx, message in enumerate(data[input_key]):
                if message["role"] == "assistant":
                    prompt = apply_chat_template(data[input_key][:idx], tokenize=False, add_generation_prompt=True)
                    response = apply_chat_template(data[input_key][: idx + 1], tokenize=False)[len(prompt):]

                    start_idx = (
                        self.tokenizer(
                            prompt,
                            max_length=self.max_length,
                            padding=False,
                            truncation=True,
                            return_tensors="pt",
                            add_special_tokens=False,
                        )["attention_mask"]
                        .int()
                        .sum()
                        .item()
                    )

                    end_idx = (
                            start_idx
                            + self.tokenizer(
                        response,
                        max_length=self.max_length,
                        padding=False,
                        truncation=True,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )["attention_mask"]
                            .int()
                            .sum()
                            .item()
                            - 1
                    )
                    response_ranges.append((start_idx, end_idx))  # left close right close

        prompt, response = preprocess_data(
            data,
            None if self.pretrain_mode else self.input_template,
            self.input_key,
            self.output_key,
            apply_chat_template=None if self.pretrain_mode else self.apply_chat_template,
            multiturn=self.multiturn,
        )

        if not self.pretrain_mode:
            prompt_token = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_ids_len = prompt_token["attention_mask"].int().sum().item()
            # filter the sample whose length is greater than max_length (2 for answer length)
            if not prompt or not response or prompt_ids_len >= self.max_length - 2:
                prompt = None
        else:
            prompt_ids_len = 0

        return {
            "prompt": prompt,
            "response": response,
            "prompt_ids_len": prompt_ids_len,
        }

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        # 獲取單筆資料
        input_text = self.input_ids[idx]
        labels = self.labels[idx]

        # 對輸入文本進行 Tokenization
        encoded_input = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",  # 直接 padding 到最大長度，方便後續 collate
            truncation=True,
            return_tensors="pt",
        )

        # .squeeze(0) 移除由 return_tensors="pt" 產生的多餘批次維度
        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]

        # 將標籤轉換為 PyTorch Tensor
        labels_tensor = torch.tensor(labels,dtype=torch.bfloat16)
        if not self.pretrain_mode:
            # to avoid EOS_token truncation
            input_ids[0][-1] = self.tokenizer.eos_token_id
            attention_mask[0][-1] = True
        return input_ids, attention_mask, labels_tensor

    def collate_fn(self, item_list):
        """
        將從 __getitem__ 獲得的多個樣本組合成一個批次。
        """
        input_ids = []
        attention_masks = []
        labels = []

        for input_id, attention_mask, label in item_list:
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            labels.append(label)

        # 因為 __getitem__ 中已經 padding 到了 max_length，
        # 所以這裡的 zero_pad_sequences 實際上只是一個 torch.stack 操作。
        # 這種寫法與 SFTDataset 保持了最大的一致性。
        input_ids = zero_pad_sequences(input_ids, "right", self.tokenizer.pad_token_id)
        attention_masks = zero_pad_sequences(attention_masks, "right")

        # 將標籤列表堆疊成一個批次張量
        labels = torch.stack(labels)

        return input_ids, attention_masks, labels
