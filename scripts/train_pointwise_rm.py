import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from transformers.trainer_pt_utils import nested_detach

from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.utils import get_tokenizer


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_hf_dataset(path: str) -> Dataset:
    rows = load_jsonl(path)
    texts = []
    scores = []
    metas = []
    for row in rows:
        prompt = str(row.get("prompt", "")).rstrip()
        response = str(row.get("response", "")).strip()
        if not prompt or not response:
            continue
        text = f"{prompt}{response}"
        texts.append(text)
        scores.append(float(row.get("score", 0.0)))
        metas.append(
            {
                "question_id": row.get("question_id"),
                "db_id": row.get("db_id"),
                "label_type": row.get("label_type"),
                "source_type": row.get("source_type"),
            }
        )
    return Dataset.from_dict({"text": texts, "score": scores, "meta": metas})


@dataclass
class PointwiseCollator:
    tokenizer: Any
    max_len: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [f["text"] for f in features]
        scores = [float(f["score"]) for f in features]
        enc = self.tokenizer(
            texts,
            max_length=self.max_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        labels = torch.tensor(scores, dtype=torch.float32)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }


class PointwiseRMTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_names = ["labels"]

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").to(model.score.weight.device)
        rewards = model(inputs["input_ids"], inputs["attention_mask"])
        rewards = rewards.float().reshape(-1)
        loss = torch.nn.functional.mse_loss(rewards, labels)
        if return_outputs:
            return loss, {"logits": rewards}
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        labels = inputs.get("labels")

        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        loss = loss.detach().mean()
        logits = outputs["logits"]

        if prediction_loss_only:
            return loss, None, None

        logits = nested_detach(logits)
        labels = nested_detach(labels) if labels is not None else None
        return loss, logits, labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a pointwise reward model on prompt-response-score data.")
    parser.add_argument("--train-dataset", type=str, required=True)
    parser.add_argument("--eval-dataset", type=str, required=True)
    parser.add_argument("--pretrain", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--attn-implementation", type=str, default="flash_attention_2")
    parser.add_argument("--param-dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--value-head-prefix", type=str, default="score")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--deepspeed-config", type=str, default="")
    parser.add_argument("--resume-from-checkpoint", type=str, default="")
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--report-to", type=str, default="none")
    parser.add_argument("--run-name", type=str, default="pointwise_rm")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_dataset = build_hf_dataset(args.train_dataset)
    eval_dataset = build_hf_dataset(args.eval_dataset)

    model = get_llm_for_sequence_regression(
        args.pretrain,
        "reward",
        attn_implementation=args.attn_implementation,
        param_dtype=args.param_dtype,
        value_head_prefix=args.value_head_prefix,
    )
    tokenizer = get_tokenizer(args.pretrain, model, "left", strategy=None, use_fast=True)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir=args.save_path,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        save_strategy="steps",
        bf16=args.bf16 or args.param_dtype == "bf16",
        fp16=args.fp16 or args.param_dtype == "fp16",
        weight_decay=args.weight_decay,
        report_to=[] if args.report_to == "none" else [args.report_to],
        run_name=args.run_name,
        seed=args.seed,
        remove_unused_columns=False,
        deepspeed=args.deepspeed_config or None,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
    )

    trainer = PointwiseRMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=PointwiseCollator(tokenizer=tokenizer, max_len=args.max_len),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)

    os.makedirs(args.save_path, exist_ok=True)
    model.config.value_head_prefix = args.value_head_prefix
    trainer.save_model(args.save_path)
    tokenizer.save_pretrained(args.save_path)


if __name__ == "__main__":
    main()
