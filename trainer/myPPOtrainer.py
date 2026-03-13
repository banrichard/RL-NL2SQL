import os
import time
import re
import json
from collections import defaultdict
from datetime import timedelta
from typing import Optional, List, Any, Dict, Tuple

import ray
import torch
import transformers
from tqdm import tqdm

_TRANSFORMERS_V5 = int(transformers.__version__.split(".")[0]) >= 5

from openrlhf.trainer.ppo_trainer import BasePPOTrainer, prepare_datasets
from openrlhf.trainer.ppo_utils.experience_maker import RemoteExperienceMaker, SamplesGenerator
from openrlhf.trainer.ppo_utils.kl_controller import AdaptiveKLController, FixedKLController
from openrlhf.trainer.ppo_utils.replay_buffer import balance_experiences
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.utils import get_tokenizer

# Set proxies if needed
os.environ['http_proxy'] = "127.0.0.1:7890"
os.environ['https_proxy'] = "127.0.0.1:7890"


try:
    # Ensure these are correct
    from algorithm.scm_actor import SCM_ModelActor, parse_schema
    # Ensure these are correct
    from algorithm.scm_processor import SCM_Config, SCM_Unified_Model

    SCM_AVAILABLE = True
    print("✅ SCM components imported successfully.")
except ImportError as e:
    print(
        f"⚠️ Warning: Could not import SCM components: {e}. SCM integration will be disabled.")
    SCM_AVAILABLE = False
    SCM_ModelActor = None  # Define as None to avoid errors later
### SCM INTEGRATION END ###
logger = init_logger(__name__)

# Patch NaiveReplayBuffer.setup_dynamic_batch in this process so that
# filtered (smaller) replay buffers don't crash with IndexError.
from trainer.safe_dynamic_batch import apply_patch as _apply_safe_dynamic_batch_patch
_apply_safe_dynamic_batch_patch()


@ray.remote
class PPOTrainer(BasePPOTrainer):
    """
    Trainer for PPO/REINFORCE++/GRPO/RLOO with SCM Integration.
    Based on latest OpenRLHF PPOTrainer with added SCM process reward and entropy filtering.
    """

    def __init__(
            self,
            pretrain: str,
            strategy: DeepspeedStrategy,
            actor_model_group: RayActorGroup,
            critic_model_group: RayActorGroup,
            reward_model_group: RayActorGroup,
            reference_model_group: RayActorGroup,
            vllm_engines,
            ### SCM parameters ###
            scm_model_group: Optional[RayActorGroup] = None,
            scm_beta: float = 0.1,  # Weight for entropy reward
            scm_gamma: float = 0.1,  # Weight for quality reward
            scm_lambda_entropy: float = 0.01,  # Weight for entropy regularization
            enable_entropy_filtering: bool = True,
            threshold_final_entropy: float = 5.0,
            threshold_slow_convergence_steps: int = 3,
            threshold_drop_rate: float = 0.1,
            threshold_entropy_increase: float = 0.5,
            schema_file_path: str = "schemas.json",
            ### SCM parameters END ###
            **generate_kwargs,
    ) -> None:
        # Get eval and save steps
        if strategy.args.eval_steps == -1:
            strategy.args.eval_steps = float("inf")  # do not evaluate
        if strategy.args.save_steps == -1:
            strategy.args.save_steps = float("inf")  # do not save ckpt

        # Tokenizer is shared across the sample generator and trainer to avoid duplicated loads.
        tokenizer = get_tokenizer(
            pretrain, None, "left", strategy,
            use_fast=not strategy.args.disable_fast_tokenizer
        )
        self.prompts_dataloader, self.eval_dataloader, self.max_steps = prepare_datasets(strategy, tokenizer)
        self.generate_kwargs = generate_kwargs

        # Sample generation
        self.samples_generator = SamplesGenerator(
            strategy=strategy,
            prompts_dataloader=self.prompts_dataloader,
            eval_dataloader=self.eval_dataloader,
            tokenizer=tokenizer,
            vllm_engines=vllm_engines,
        )

        # Initialize base trainer
        super().__init__(
            strategy,
            actor_model_group,
            critic_model_group,
            reward_model_group,
            reference_model_group,
            vllm_engines,
            tokenizer,
        )

        ### SCM INTEGRATION START ###
        self.scm_model_group = scm_model_group
        self.scm_beta = scm_beta
        self.scm_gamma = scm_gamma
        self.scm_lambda_entropy = scm_lambda_entropy
        self.enable_entropy_filtering = enable_entropy_filtering
        self.threshold_final_entropy = threshold_final_entropy
        self.threshold_slow_convergence_steps = threshold_slow_convergence_steps
        self.threshold_drop_rate = threshold_drop_rate
        self.threshold_entropy_increase = threshold_entropy_increase
        self.schema_file_path = schema_file_path
        self._schema_cache = {}
        self._scm_results_cache = {}  # 缓存 SCM 分析结果，避免 filtering + reward 重复调用

        # Validate SCM configuration
        if self.enable_entropy_filtering and not self.scm_model_group:
            logger.warning(
                "Entropy filtering enabled, but SCM model group is not provided. Filtering will be skipped.")
            self.enable_entropy_filtering = False

        if not SCM_AVAILABLE and self.scm_model_group:
            logger.error(
                "SCM components not found, but SCM features requested. Disabling SCM.")
            self.scm_model_group = None
            self.enable_entropy_filtering = False

        if self.scm_model_group:
            self._load_schema_cache()
            if not self._schema_cache:
                logger.error(
                    f"Schema cache failed to load from {self.schema_file_path}. SCM analysis will likely fail.")

        if self.enable_entropy_filtering:
            n_samples = self.generate_kwargs.get("n_samples_per_prompt", 1)
            if n_samples <= 1:
                logger.warning(
                    f"Entropy filtering requires n_samples_per_prompt > 1 (got {n_samples}). Filtering will be skipped.")
                self.enable_entropy_filtering = False
            else:
                logger.info("✅ Entropy filtering enabled.")
                logger.info(
                    f"    - Final Entropy Threshold: {self.threshold_final_entropy}")
                logger.info(
                    f"    - Slow Convergence Steps: {self.threshold_slow_convergence_steps}")
                logger.info(
                    f"    - Min Drop Rate (%): {self.threshold_drop_rate * 100:.1f}")
                logger.info(
                    f"    - Max Entropy Increase Allowed: {self.threshold_entropy_increase}")

        self.empty_step_warn_threshold = max(int(getattr(self.args, "empty_step_warn_threshold", 3)), 1)
        self._empty_step_streak = 0
        self._empty_step_total = 0
        self._empty_step_reasons = defaultdict(int)
        ### SCM INTEGRATION END ###

    def _record_empty_step(self, reason: str, global_step: int) -> None:
        self._empty_step_streak += 1
        self._empty_step_total += 1
        self._empty_step_reasons[reason] += 1
        logger.info(
            f"Step {global_step}: Empty-step stats => streak={self._empty_step_streak}, "
            f"total={self._empty_step_total}, reason={reason}, reason_count={self._empty_step_reasons[reason]}"
        )

        if self._empty_step_streak >= self.empty_step_warn_threshold and (
            self._empty_step_streak == self.empty_step_warn_threshold
            or self._empty_step_streak % self.empty_step_warn_threshold == 0
        ):
            reason_summary = ", ".join(
                f"{k}:{v}" for k, v in sorted(self._empty_step_reasons.items(), key=lambda item: item[1], reverse=True)
            )
            logger.warning(
                f"Step {global_step}: Consecutive empty steps reached {self._empty_step_streak} "
                f"(threshold={self.empty_step_warn_threshold}). "
                f"Possible over-filtering or upstream sampling issue. reason_breakdown=[{reason_summary}]"
            )

    def _record_non_empty_step(self, global_step: int) -> None:
        if self._empty_step_streak > 0:
            logger.info(
                f"Step {global_step}: Empty-step streak recovered from {self._empty_step_streak} to 0."
            )
        self._empty_step_streak = 0

    def get_max_steps(self):
        return self.max_steps

    def train_step(self, rollout_samples, global_step: int) -> Tuple[Dict, int]:
        """Override to add SCM process rewards after experience making."""
        # Turn raw rollouts into PPO-ready trajectories with rewards.
        experiences = self.experience_maker.make_experience_batch(rollout_samples)

        ### SCM INTEGRATION: Calculate and add SCM process rewards ###
        if self.scm_model_group:
            try:
                experiences = self._calculate_and_add_scm_rewards(experiences)
                logger.info(
                    f"Step {global_step}: SCM process rewards calculated and added for {len(experiences)} experiences.")
            except Exception as e:
                logger.error(
                    f"Step {global_step}: Failed to calculate SCM rewards: {e}", exc_info=True)
            finally:
                # 清空缓存，避免内存泄漏
                self._scm_results_cache.clear()
        ### SCM INTEGRATION END ###

        # Peek at the first decoded sample for quick sanity check.
        _decode = self.tokenizer.decode if _TRANSFORMERS_V5 else self.tokenizer.batch_decode
        sample0 = [
            _decode(experiences[0].sequences[0].unsqueeze(0), skip_special_tokens=True)[0],
            experiences[0].info["reward"][0].item(),
        ]
        print(sample0)

        # Balance experiences across DP ranks if needed.
        if self.args.use_dynamic_batch:
            experiences = balance_experiences(experiences, self.args)

        # Push experiences to actor (and critic) shards before PPO.
        refs = self.actor_model_group.async_run_method_batch(method_name="append", experience=experiences)
        if self.critic_model_group is not None:
            refs.extend(self.critic_model_group.async_run_method_batch(method_name="append", experience=experiences))
        ray.get(refs)

        # Perform PPO optimization for actor/critic and gather metrics.
        status = self.ppo_train(global_step)

        # Sync weights to vLLM.
        if self.vllm_engines is not None:
            self.broadcast_to_vllm()

        # Refresh KL controller with the latest measurement.
        if "kl" in status:
            self.kl_ctl.update(status["kl"], self.args.rollout_batch_size * self.args.n_samples_per_prompt)

        status["generated_samples"] = sample0
        return status, global_step + 1

    def _extract_info_from_prompt(self, prompt_text: str) -> Optional[Dict[str, Any]]:
        """
        從完整的 prompt 字符串中提取 question, schema_str 和 schema_columns。
        """
        # Debug: Print the actual prompt structure (first 1000 chars)
        # logger.warning(f"SCM: Prompt preview (first 1000 chars):\n{prompt_text[:1000]}\n---")
        # logger.warning(f"SCM: Checking for '<|im_start|>' in text: {'<|im_start|>' in prompt_text}")
        # logger.warning(f"SCM: Checking for 'user' in text: {'user' in prompt_text}")
        # logger.warning(f"SCM: Checking for '<|im_end|>' in text: {'<|im_end|>' in prompt_text}")
        
        # 1. 提取 Schema
        schema_match = re.search(
            r"Database Schema:\s*(.*?)(?:\n<\|im_start\|>|External Knowledge:|SQL Complexity Level:|\Z)",
            prompt_text,
            re.DOTALL | re.IGNORECASE,
        )
        if not schema_match:
            logger.warning(f"SCM: 無法從 Prompt 中提取 'Database Schema'。")
            return None
        schema_str = schema_match.group(1).strip()

        # 2. 解析 Schema
        parsed_schema = parse_schema(schema_str)
        if not parsed_schema["columns"]:
            logger.warning(f"SCM: 從 Schema 字符串中解析欄位失敗。")
            return None

        # 3. 提取 Question - 嘗試多種方法
        question = None
        
        # 方法1: 從 <|im_start|>user ... <|im_end|> 提取（ChatML 格式）
        user_turns = re.findall(
            r"<\|im_start\|>\s*user\s*(.*?)\s*<\|im_end\|>", prompt_text, re.DOTALL | re.IGNORECASE)
        if user_turns:
            question = user_turns[-1].strip()
            logger.info(f"SCM: Extracted question using ChatML format.")
            return {
                "question": question,
                "schema_str": schema_str,
                "schema_columns": parsed_schema["columns"]
            }
        
        # 方法2: 從 'Question:' 之後提取
        question_match = re.search(
            r"Question:\s*(.*?)(?:\n\n|$)", prompt_text, re.DOTALL | re.IGNORECASE
        )
        if question_match:
            question = question_match.group(1).strip()
            # 清理可能的 schema 殘留
            question = re.sub(r"Database Schema:.*", "", question, flags=re.DOTALL).strip()
            logger.info(f"SCM: Extracted question using 'Question:' key.")
            return {
                "question": question,
                "schema_str": schema_str,
                "schema_columns": parsed_schema["columns"]
            }
        
        # 方法3: 從 'Instruction:' 或 'Instructions:' 之後提取
        instruction_match = re.search(
            r"Instructions?:\s*(.*?)(?:\n\n|Output Format:|$)", prompt_text, re.DOTALL | re.IGNORECASE
        )
        if instruction_match:
            text = instruction_match.group(1).strip()
            # 取最後一段（通常是實際問題）
            sentences = text.split('\n')
            question = '\n'.join(sentences[-3:]) if len(sentences) > 0 else text
            logger.info(f"SCM: Extracted question using 'Instructions:' key.")
            return {
                "question": question,
                "schema_str": schema_str,
                "schema_columns": parsed_schema["columns"]
            }
        
        logger.warning(f"SCM: 無法從 Prompt 中提取 'user' 輪次或 Question。")
        return None

    def fit(self) -> None:
        checkpoint_states = self.init_checkpoint_states()
        # Restore step and start_epoch
        start_episode = checkpoint_states["episode"]
        global_step = checkpoint_states["global_step"]
        total_consumed_prompts = checkpoint_states["total_consumed_prompts"]
        # Keep vLLM weights and dataloader states in sync when resuming.
        if global_step:
            self.broadcast_to_vllm()
            state_dict = checkpoint_states["data_loader_state_dict"]
            if state_dict:
                self.prompts_dataloader.load_state_dict(state_dict)

        for episode in range(start_episode, self.args.num_episodes):
            dataset_length = len(self.prompts_dataloader)
            pbar = tqdm(
                range(dataset_length),
                desc=f"Episode [{episode + 1}/{self.args.num_episodes}]",
                initial=total_consumed_prompts % max(dataset_length, 1),
            )
            while True:
                # Draw one mini-batch of prompts; stop when loader is exhausted.
                rollout_samples, filter_pass_rate, prompts_consumed, is_exhausted = (
                    self.samples_generator.generate_samples(**self.generate_kwargs)
                )
                total_consumed_prompts += prompts_consumed
                if is_exhausted:
                    break

                ### SCM INTEGRATION: Entropy filtering ###
                if self.enable_entropy_filtering:
                    try:
                        num_before = len(rollout_samples)
                        rollout_samples = self._filter_samples_by_entropy(rollout_samples)
                        num_after = len(rollout_samples)
                        entropy_pass_rate = (num_after / num_before * 100) if num_before > 0 else 0
                        logger.info(
                            f"Step {global_step}: Entropy filtering pass rate: "
                            f"{entropy_pass_rate:.2f}% ({num_after}/{num_before})")
                        if not rollout_samples:
                            logger.warning(
                                f"Step {global_step}: No samples passed entropy filtering. Skipping step.")
                            self._record_empty_step("entropy_filter_empty", global_step)
                            pbar.update(prompts_consumed)
                            continue
                    except Exception as e:
                        logger.error(
                            f"Step {global_step}: Error during entropy filtering: {e}", exc_info=True)
                        logger.warning("Continuing with unfiltered samples due to filtering error.")
                ### SCM INTEGRATION END ###

                # Safety guard: grouped-advantage estimators require divisibility by n_samples_per_prompt.
                n_samples = max(int(getattr(self.args, "n_samples_per_prompt", 1)), 1)
                if n_samples > 1 and len(rollout_samples) % n_samples != 0:
                    valid_len = (len(rollout_samples) // n_samples) * n_samples
                    logger.warning(
                        f"Step {global_step}: rollout sample size {len(rollout_samples)} is not divisible by "
                        f"n_samples_per_prompt={n_samples}. Truncating to {valid_len} to keep group integrity.")
                    if valid_len == 0:
                        logger.warning(
                            f"Step {global_step}: No valid grouped samples left after truncation. Skipping step.")
                        self._record_empty_step("group_truncation_empty", global_step)
                        pbar.update(prompts_consumed)
                        continue
                    rollout_samples = rollout_samples[:valid_len]

                if not rollout_samples:
                    logger.warning(
                        f"Step {global_step}: Empty rollout_samples after preprocessing. Skipping step.")
                    self._record_empty_step("preprocess_empty", global_step)
                    pbar.update(prompts_consumed)
                    continue

                self._record_non_empty_step(global_step)

                # Run PPO update on this batch and bump the global step counter.
                status, global_step = self.train_step(rollout_samples, global_step)

                # Add generated samples to status dictionary
                if self.args.dynamic_filtering:
                    status["dynamic_filtering_pass_rate"] = filter_pass_rate
                log_status = {k: v for k, v in status.items() if k not in ["generated_samples"]}
                logger.info(f"✨ Global step {global_step}: {log_status}")

                # logs/checkpoints
                client_states = {
                    "episode": episode,
                    "global_step": global_step,
                    "total_consumed_prompts": total_consumed_prompts,
                    "data_loader_state_dict": self.prompts_dataloader.state_dict(),
                }
                self.save_logs_and_checkpoints(global_step, status, client_states)

                # Evaluation
                if global_step % self.args.eval_steps == 0 and self.eval_dataloader:
                    eval_generate_kwargs = self.generate_kwargs.copy()
                    eval_generate_kwargs["temperature"] = self.args.eval_temperature
                    eval_generate_kwargs["n_samples_per_prompt"] = self.args.eval_n_samples_per_prompt
                    self.evaluate(global_step, **eval_generate_kwargs)

                pbar.update(prompts_consumed)

        # Close trackers
        if self.wandb_logger:
            self.wandb_logger.close()
        if self.tensorboard_logger:
            self.tensorboard_logger.close()

    @torch.no_grad()
    def evaluate(self, global_step, **generate_kwargs):
        """Evaluate model performance on eval dataset."""
        start_time = time.time()
        logger.info(f"⏰ Evaluation start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # First collect all prompts and labels
        prompt_to_datasource = {}
        for datasources, prompts, labels in self.eval_dataloader:
            for prompt, datasource in zip(prompts, datasources):
                prompt_to_datasource[prompt] = datasource

        # Generate samples and calculate rewards
        samples_list = self.samples_generator.generate_eval_samples(**generate_kwargs)

        # duplicate prompts and labels for each sample
        all_prompts = sum([s.prompts for s in samples_list], [])
        n_samples_per_prompt = generate_kwargs["n_samples_per_prompt"]

        # Get rewards from samples
        rewards_list = []
        for samples in samples_list:
            rewards_list.append(samples.rewards)
        # Reshape rewards to (num_prompts, n_samples_per_prompt)
        rewards = torch.tensor(rewards_list).reshape(-1, n_samples_per_prompt)

        # Collect local statistics for each data source
        global_metrics = {}
        num_prompts = len(all_prompts) // n_samples_per_prompt
        for i in range(num_prompts):
            original_prompt = all_prompts[i * n_samples_per_prompt]
            datasource = prompt_to_datasource[original_prompt]
            if datasource not in global_metrics:
                global_metrics[datasource] = {f"pass{n_samples_per_prompt}": 0, "pass1": 0, "count": 0}

            chunk_rewards = rewards[i]
            if n_samples_per_prompt > 1:
                global_metrics[datasource][f"pass{n_samples_per_prompt}"] += chunk_rewards.max().float().item()
            global_metrics[datasource]["pass1"] += chunk_rewards.mean().float().item()
            global_metrics[datasource]["count"] += 1

        # Calculate global averages
        logs = {}
        for datasource, metrics in global_metrics.items():
            logs[f"eval_{datasource}_pass{n_samples_per_prompt}"] = (
                metrics[f"pass{n_samples_per_prompt}"] / metrics["count"]
            )
            logs[f"eval_{datasource}_pass1"] = metrics["pass1"] / metrics["count"]

        # Log to wandb/tensorboard
        if self.wandb_logger:
            self.wandb_logger.log_eval(global_step, logs)
        if self.tensorboard_logger:
            self.tensorboard_logger.log_eval(global_step, logs)

        end_time = time.time()
        duration = end_time - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        logger.info(f"✨ Evaluation completed in {time_str}, global_step {global_step}, eval_metrics: {logs}")

    def _calculate_and_add_scm_rewards(self, experiences: List[Any]) -> List[Any]:
        """
        Calls the SCM Actor to analyze sequences and adds process rewards.
        Extracts prompt information from experience sequences directly.
        """
        if not self.scm_model_group:
            logger.warning(
                "SCM model group not available, skipping process reward calculation.")
            return experiences

        # Prepare data for SCM Actor
        # Each experience is a BATCH with shape (B, S). We iterate over all
        # samples across all micro-batches, keeping (batch_idx, sample_idx) so
        # we can write the reward back to the correct position.
        sequences_data = []
        valid_indices = []  # list of (batch_idx, sample_idx)

        for batch_idx, experience in enumerate(experiences):
            batch_size = experience.sequences.shape[0]
            for sample_idx in range(batch_size):
                seq_tensor = experience.sequences[sample_idx]
                full_text = self.tokenizer.decode(seq_tensor, skip_special_tokens=False)
                info = self._extract_info_from_prompt(full_text)
                if info is None:
                    logger.warning(f"SCM: Could not extract info from experience batch {batch_idx} sample {sample_idx}. Skipping.")
                    continue

                sequences_data.append({
                    "question": info.get('question', " "),
                    "schema": info.get("schema_str", " "),
                    "full_text": full_text,
                    "schema_columns": info.get("schema_columns", [])
                })
                valid_indices.append((batch_idx, sample_idx))

        if not sequences_data:
            logger.warning("No valid data prepared for SCM analysis.")
            return experiences

        # 尝试从缓存中获取结果（来自 _filter_samples_by_entropy 的预计算）
        cached_results = []
        uncached_indices = []
        uncached_data = []

        for i, (batch_idx, sample_idx) in enumerate(valid_indices):
            seq_tensor = experiences[batch_idx].sequences[sample_idx]
            cache_key = hash(seq_tensor.data_ptr()) if hasattr(seq_tensor, 'data_ptr') else None
            cached = self._scm_results_cache.get(cache_key) if cache_key is not None else None
            if cached is not None:
                cached_results.append((i, cached))
            else:
                uncached_indices.append(i)
                uncached_data.append(sequences_data[i])

        # 只对未缓存的数据调用 SCM Actor
        if uncached_data:
            logger.info(
                f"Calling SCM Actor with {len(uncached_data)} sequences "
                f"(skipped {len(cached_results)} cached)...")
            fresh_results = ray.get(
                self.scm_model_group.async_run_method(
                    method_name="forward", sequences_data=uncached_data)
            )[0]
        else:
            logger.info(
                f"All {len(cached_results)} SCM results found in cache, skipping SCM Actor call.")
            fresh_results = []

        # 合并结果
        analysis_results = [None] * len(sequences_data)
        for orig_i, result in cached_results:
            analysis_results[orig_i] = result
        for j, orig_i in enumerate(uncached_indices):
            analysis_results[orig_i] = fresh_results[j]

        logger.info(
            f"SCM analysis complete: {len(analysis_results)} results "
            f"({len(cached_results)} cached + {len(fresh_results)} fresh).")

        # Add process rewards back to experiences using valid_indices mapping
        if len(analysis_results) != len(valid_indices):
            logger.error(
                f"Mismatch between SCM results ({len(analysis_results)}) and valid indices ({len(valid_indices)}). Skipping reward update.")
            return experiences

        for result_idx, (batch_idx, sample_idx) in enumerate(valid_indices):
            experience = experiences[batch_idx]
            entropies = analysis_results[result_idx].get('entropies', [])
            quality_scores = analysis_results[result_idx].get('quality_scores', [])

            if len(entropies) < 2 or len(quality_scores) < 2:  # Need at least H0/Q0 and H1/Q1
                logger.warning(
                    f"Insufficient SCM analysis data for batch {batch_idx} sample {sample_idx}. Skipping reward update.")
                continue

            # Calculate total process reward from the sequence
            total_process_reward = 0
            for t in range(1, len(entropies)):
                # Ensure indices are valid
                if t - 1 < len(entropies) and t < len(entropies) and \
                        t - 1 < len(quality_scores) and t < len(quality_scores):

                    entropy_reward = self.scm_beta * \
                        (entropies[t - 1] - entropies[t])
                    quality_reward = self.scm_gamma * \
                        (quality_scores[t] - quality_scores[t - 1])
                    total_process_reward += (entropy_reward + quality_reward)
                else:
                    logger.warning(
                        f"Index out of bounds during process reward calculation for batch {batch_idx} sample {sample_idx}, step {t}.")
                    break  # Stop calculating for this sample if indices mismatch

            # experience.rewards has shape (B,) — one scalar reward per sample.
            # Add the SCM process reward to the per-sample scalar reward.
            # It will be placed at the last action token during
            # compute_advantages_and_returns.
            if experience.rewards.dtype != torch.float32:
                experience.rewards = experience.rewards.float()

            experience.rewards[sample_idx] += total_process_reward

            # (Optional) Store total entropy for loss regularization if needed later
            # info values must be 1D Tensor of shape (B,) for split_experience_batch
            batch_size = experience.sequences.shape[0]
            if 'total_entropy' not in experience.info:
                experience.info['total_entropy'] = torch.zeros(batch_size)
            experience.info['total_entropy'][sample_idx] += sum(entropies)

        return experiences

    def _get_schema_string(self, db_id: str) -> Optional[str]:
        """ Helper function to retrieve schema string from cache. """
        schema = self._schema_cache.get(db_id)
        if schema is None:
            logger.warning(f"Schema not found in cache for db_id: {db_id}")
        return schema

    def _filter_samples_by_entropy(self, rollout_samples: List[Any]) -> List[Any]:
        """
        Filters rollout samples based on SCM entropy analysis.
        Extracts prompt information from sample sequences directly.
        """
        if not self.scm_model_group:
            return rollout_samples  # SCM不可用，不過濾

        # 準備數據給 SCM Actor
        sequences_data = []
        sample_indices = []  # 記錄原始樣本的索引，以便過濾後重建

        for i, sample in enumerate(rollout_samples):
            # 解碼序列並提取信息
            seq_tensor = sample.sequences[0]
            full_text = self.tokenizer.decode(
                seq_tensor, skip_special_tokens=False)
            info = self._extract_info_from_prompt(full_text)
            if info is None:
                logger.warning(
                    f"SCM Filtering: Cannot parse sample {i}. Keeping sample.")
                continue

            sequences_data.append({
                "question": info.get('question', ''),
                "schema": info.get('schema_str', ''),
                "full_text": full_text,
                "schema_columns": info.get('schema_columns', [])
            })
            # Store index corresponding to this sequence data
            sample_indices.append(i)

        if not sequences_data:
            return rollout_samples

        # 呼叫 SCM Actor 進行分析
        logger.info(
            f"Calling SCM Actor for entropy filtering analysis ({len(sequences_data)} samples)...")
        # Filter out None placeholders before sending to Ray actor
        valid_indices = [idx for idx, data in zip(
            sample_indices, sequences_data) if data is not None]
        valid_sequences_data = [
            data for data in sequences_data if data is not None]

        if not valid_sequences_data:  # If all samples lacked schema
            logger.warning(
                "No valid samples with schema found for SCM analysis in this batch.")
            return rollout_samples  # Return original samples

        analysis_results_list = ray.get(
            self.scm_model_group.async_run_method(
                method_name="forward", sequences_data=valid_sequences_data)
        )[0]
        logger.info(f"Received {len(analysis_results_list)} analysis results.")

        # 缓存 SCM 结果，供后续 _calculate_and_add_scm_rewards 复用
        for idx, result in zip(valid_indices, analysis_results_list):
            seq_tensor = rollout_samples[idx].sequences[0]
            cache_key = hash(seq_tensor.data_ptr()) if hasattr(seq_tensor, 'data_ptr') else id(rollout_samples[idx])
            self._scm_results_cache[cache_key] = result

        # Create a map from original index to analysis result
        analysis_map = {idx: result for idx, result in zip(
            valid_indices, analysis_results_list)}

        # 應用過濾規則（按 prompt 分組，保證 group size 一致）
        sample_keep_flags: Dict[int, bool] = {}
        for i in range(len(rollout_samples)):
            # Sample was skipped (e.g., no schema) or analysis failed -> keep by default
            if i not in analysis_map:
                logger.debug(f"Keeping sample {i} as it was not analyzed by SCM.")
                sample_keep_flags[i] = True
                continue

            analysis_result = analysis_map[i]
            entropies = analysis_result.get('entropies', [])
            if len(entropies) < 2:  # 至少需要 H0 和 H1
                logger.warning(f"Insufficient entropy data for sample {i}. Keeping sample.")
                sample_keep_flags[i] = True
                continue

            passed, reason = self._check_entropy_rules(entropies)
            sample_keep_flags[i] = passed
            if not passed:
                logger.info(f"Sample {i} filtered out due to entropy rules: {reason}")

        n_samples = max(int(getattr(self.args, "n_samples_per_prompt", 1)), 1)
        if n_samples <= 1:
            return [s for idx, s in enumerate(rollout_samples) if sample_keep_flags.get(idx, True)]

        total_samples = len(rollout_samples)
        if total_samples % n_samples != 0:
            logger.warning(
                f"Rollout samples size {total_samples} is not divisible by n_samples_per_prompt={n_samples}. "
                f"Tail samples will be ignored during group filtering.")

        filtered_samples = []
        group_count = total_samples // n_samples
        for group_idx in range(group_count):
            start = group_idx * n_samples
            end = start + n_samples
            group_pass = all(sample_keep_flags.get(i, True) for i in range(start, end))

            if group_pass:
                filtered_samples.extend(rollout_samples[start:end])
            else:
                logger.info(
                    f"Prompt group {group_idx} filtered out because at least one sample failed entropy rules.")

        return filtered_samples

    def _check_entropy_rules(self, entropies: List[float]) -> Tuple[bool, str]:
        passed = True
        reason = ""

        # Rule 1: Final Entropy Check
        final_entropy = entropies[-1]
        if final_entropy > self.threshold_final_entropy:
            passed = False
            reason = f"Final entropy {final_entropy:.2f} > {self.threshold_final_entropy}"

        # Rule 2: Slow Convergence Check (requires enough steps)
        if passed and len(entropies) > self.threshold_slow_convergence_steps:
            h0 = entropies[0]
            h_target = entropies[self.threshold_slow_convergence_steps]
            drop_rate = (h0 - h_target) / (h0 + 1e-9)  # Avoid division by zero
            if drop_rate < self.threshold_drop_rate:
                passed = False
                reason = f"Slow convergence: drop rate {drop_rate:.2f} < {self.threshold_drop_rate} in first {self.threshold_slow_convergence_steps} steps"

        # Rule 3: Entropy Increase Check
        if passed:
            for t in range(1, len(entropies)):
                if entropies[t] > entropies[t - 1] + self.threshold_entropy_increase:
                    passed = False
                    reason = f"Entropy increased significantly at step {t}: {entropies[t - 1]:.2f} -> {entropies[t]:.2f}"
                    break

        # 記錄並決定是否保留
        return passed, reason

    def _load_schema_cache(self):
        """Loads schemas from JSON/JSONL into a dictionary keyed by db_id."""
        if not self.schema_file_path or not os.path.exists(self.schema_file_path):
            logger.warning(
                f"Schema file '{self.schema_file_path}' not found or path is empty.")
            return
        try:
            schema_cache = {}
            with open(self.schema_file_path, 'r', encoding='utf-8') as f:
                if self.schema_file_path.endswith('.jsonl'):
                    for line_no, line in enumerate(f, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        if line in {'{', '}'}:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError as e:
                            # Fallback: tolerate lines like
                            #   "db_id": "schema ...",
                            # which are JSON object members rather than strict JSONL rows.
                            try:
                                member_line = line[:-1] if line.endswith(',') else line
                                kv_obj = json.loads('{' + member_line + '}')
                                if isinstance(kv_obj, dict) and len(kv_obj) == 1:
                                    db_id, schema_value = next(iter(kv_obj.items()))
                                    if isinstance(schema_value, (dict, list)):
                                        schema_value = json.dumps(schema_value, ensure_ascii=False)
                                    schema_cache[str(db_id)] = schema_value
                                    continue
                            except Exception:
                                pass

                            logger.warning(
                                f"Invalid JSONL at line {line_no} in {self.schema_file_path}: {e}")
                            continue

                        db_id = obj.get('db_id') or obj.get('database_id') or obj.get('id')
                        if not db_id:
                            logger.warning(
                                f"Missing db_id at line {line_no} in {self.schema_file_path}; skipping.")
                            continue

                        schema_value = (
                            obj.get('schema')
                            or obj.get('schema_string')
                            or obj.get('db_details')
                        )
                        if schema_value is None:
                            logger.warning(
                                f"Missing schema content for db_id '{db_id}' (line {line_no}); skipping.")
                            continue

                        if isinstance(schema_value, (dict, list)):
                            schema_value = json.dumps(schema_value, ensure_ascii=False)
                        schema_cache[str(db_id)] = schema_value
                else:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        schema_cache = loaded
                    elif isinstance(loaded, list):
                        for idx, obj in enumerate(loaded):
                            if not isinstance(obj, dict):
                                continue
                            db_id = obj.get('db_id') or obj.get('database_id') or obj.get('id')
                            schema_value = (
                                obj.get('schema')
                                or obj.get('schema_string')
                                or obj.get('db_details')
                            )
                            if db_id and schema_value is not None:
                                if isinstance(schema_value, (dict, list)):
                                    schema_value = json.dumps(schema_value, ensure_ascii=False)
                                schema_cache[str(db_id)] = schema_value
                            elif idx < 3:
                                logger.warning("Unexpected schema item format in JSON list; item skipped.")

            self._schema_cache = schema_cache
            logger.info(
                f"Loaded {len(self._schema_cache)} schemas from {self.schema_file_path}")
        except Exception as e:
            logger.error(
                f"Failed to load schemas from {self.schema_file_path}: {e}")
            self._schema_cache = {}  # Ensure it's a dict on failure
