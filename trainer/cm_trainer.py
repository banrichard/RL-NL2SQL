# credit_trainer.py
import os

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
import torch
import torch.nn as nn
from openrlhf.utils.distributed_sampler import DistributedSampler
from tqdm import tqdm

# 我們可以從 SFTTrainer 繼承，以重用大部分非核心的程式碼
from openrlhf.trainer.sft_trainer import SFTTrainer


class CreditTrainer(SFTTrainer):
    """
    專門用於訓練信用分配模型 (Credit Model) 的 Trainer。
    它繼承自 SFTTrainer，但重寫了核心的訓練和評估迴圈。
    """

    def __init__(self, *args, **kwargs):
        # 直接呼叫父類別的 __init__ 方法來初始化所有基礎元件
        super().__init__(*args, **kwargs)

        # --- 關鍵修改 (1/3): 將損失函數從 SFTLoss 更換為 MSELoss ---
        # Credit Model 是一個迴歸任務，需要使用均方誤差損失
        self.loss_fn = nn.MSELoss()

    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        # --- 我們可以重用大部分 fit 方法的框架，但需要重寫核心的訓練迴圈 ---
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt
        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)

        epoch_bar = tqdm(
            range(start_epoch, self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )
        loss_sum = 0

        for epoch in range(self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )

            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            # --- 關鍵修改 (2/3): 重寫訓練步驟 ---
            self.model.train()
            for inputs, attention_masks, labels in self.train_dataloader:
                # 假設 DataLoader 產生的 batch 是一個字典
                inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
                attention_mask = attention_masks.to(torch.cuda.current_device()).squeeze(1)
                labels = labels.to(torch.cuda.current_device())  # 標籤是 target_credits 向量

                # 前向傳播
                predicted_credits = self.model(inputs, attention_mask)
                # mixtral
                if self.aux_loss:
                    aux_loss = predicted_credits.aux_loss
                else:
                    aux_loss = 0
                # 計算損失 (MSE Loss)
                mse_loss = self.loss_fn(predicted_credits, labels)
                loss = mse_loss + aux_loss * self.args.aux_loss_coef
                # 反向傳播和更新
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                loss_sum += mse_loss.item()
                logs_dict = {
                    "credit_loss": loss.item(),
                    "lr": self.scheduler.get_last_lr()[0],
                }
                if self.aux_loss:
                    logs_dict["aux_loss"] = aux_loss.item()
                logs_dict = self.strategy.all_reduce(logs_dict)
                step_bar.set_postfix(logs_dict)
                step_bar.update()

                if step % self.strategy.accumulated_gradient == 0:
                    logs_dict["loss_mean"] = loss_sum / self.strategy.accumulated_gradient
                    loss_sum = 0
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step * args.train_batch_size}
                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)
                step += 1

            epoch_bar.update()
            
        # 训练完成后才关闭 wandb 和 tensorboard
        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    def evaluate(self, eval_dataloader, steps=0):
        # --- 關鍵修改 (3/3): 重寫評估步驟 ---
        times = 0
        self.model.eval()
        with torch.no_grad():
            loss_sum = 0
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc=f"Eval stage of steps {steps}",
                disable=not self.strategy.is_rank_0(),
            )

            for inputs, attention_masks, labels in eval_dataloader:
                inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
                attention_mask = attention_masks.to(torch.cuda.current_device()).squeeze(1)
                labels = labels.to(torch.cuda.current_device())

                predicted_credits = self.model(inputs, attention_mask)

                loss = self.loss_fn(predicted_credits, labels)

                loss_sum += loss.item()
                times += 1

                bar_dict = {"eval_credit_loss": loss_sum / times}
                logs = self.strategy.all_reduce(bar_dict)
                step_bar.update()
                step_bar.set_postfix(logs)

            # 記錄評估結果 (可以重用父類別的日誌部分)
            if self.strategy.is_rank_0():
                if (self._wandb is not None and 
                    hasattr(self._wandb, 'run') and self._wandb.run is not None):
                    try:
                        logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                        self._wandb.log(logs)
                    except Exception as e:
                        print(f"Warning: Failed to log evaluation to wandb: {e}")
                elif self._tensorboard is not None:
                    for k, v in logs.items():
                        self._tensorboard.add_scalar(f"eval/{k}", v, steps)

        self.model.train()  # 恢復訓練模式

    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        # 手动处理日志记录，避免 wandb 初始化问题
        if global_step % args.logging_steps == 0:
            # wandb 日志记录 - 检查 wandb 是否存在且未被 finish
            if (self._wandb is not None and self.strategy.is_rank_0() and 
                hasattr(self._wandb, 'run') and self._wandb.run is not None):
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                try:
                    self._wandb.log(logs)
                except Exception as e:
                    # 如果 wandb 出现问题，打印警告但不中断训练
                    print(f"Warning: Failed to log to wandb: {e}")
            # TensorBoard 日志记录
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # 评估
        if global_step % args.eval_steps == 0:
            # do eval when eval_dataloader is not None and len(dataloader) > 0, avoid zero division in eval.
            if self.eval_dataloader is not None and len(self.eval_dataloader) > 0:
                self.evaluate(self.eval_dataloader, global_step)

        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            if not self.disable_ds_ckpt:
                self.strategy.save_ckpt(
                    self.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states
                )
            if self.save_hf_ckpt:
                save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
                self.strategy.save_model(self.model, self.tokenizer, save_path)
