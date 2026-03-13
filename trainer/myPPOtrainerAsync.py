import asyncio
import os
import time
from collections import defaultdict

import ray
from tqdm import tqdm
from typing import Optional
from openrlhf.trainer.ppo_trainer import BasePPOTrainer
from openrlhf.trainer.ppo_utils import AdaptiveKLController, FixedKLController
from openrlhf.trainer.ppo_utils.experience_maker import RemoteExperienceMaker
from openrlhf.trainer.ppo_utils.replay_buffer import balance_experiences
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.logging_utils import init_logger

import torch
from transformers import AutoTokenizer

logger = init_logger(__name__)

# Patch NaiveReplayBuffer.setup_dynamic_batch in this process so that
# filtered (smaller) replay buffers don't crash with IndexError.
from trainer.safe_dynamic_batch import apply_patch as _apply_safe_dynamic_batch_patch
_apply_safe_dynamic_batch_patch()

try:
    from algorithm.scm_actor import SCM_ModelActor, parse_schema
    SCM_AVAILABLE = True
    print("✅ SCM components imported successfully.")
except ImportError as e:
    print(
        f"⚠️ Warning: Could not import SCM components: {e}. SCM integration will be disabled.")
    SCM_AVAILABLE = False
    SCM_ModelActor = None
    def parse_schema(s): return {"tables": [], "columns": []}  # Dummy


@ray.remote(num_cpus=0)
class SignalActor:
    def __init__(self):
        self.generating_event = asyncio.Event()
        self.update_weights_event = asyncio.Event()
        self.generating_event.set()  # Initially allow generation
        self.update_weights_event.set()  # Initially allow weight updates

    async def wait_generating(self):
        """Wait for generation to be allowed."""
        return await self.generating_event.wait()

    async def wait_update_weights(self):
        """Wait for weight update to be allowed."""
        return await self.update_weights_event.wait()

    def set_generating(self, allow_generating):
        """Set generation state.

        Args:
            is_generating: True to allow generation, False to block it
        """
        if allow_generating:
            self.generating_event.set()
        else:
            self.generating_event.clear()

    def set_update_weights(self, allow_updating):
        """Set weight update state.

        Args:
            is_updating: True to allow weight updates, False to block it
        """
        if allow_updating:
            self.update_weights_event.set()
        else:
            self.update_weights_event.clear()


@ray.remote
class GenerateSamplesActor(BasePPOTrainer):
    def __init__(self,
                 # Add SCM actor group
                 scm_model_group: Optional[RayActorGroup] = None,
                 scm_beta: float = 0.1,  # Weight for entropy reward
                 scm_gamma: float = 0.1,  # Weight for quality reward
                 # Weight for entropy regularization (harder to integrate here)
                 scm_lambda_entropy: float = 0.01,
                 # Whether to filter samples based on entropy
                 enable_entropy_filtering: bool = True,
                 threshold_final_entropy: float = 5.0,  # 示例值，需要調優
                 threshold_slow_convergence_steps: int = 3,  # 檢查前3步
                 threshold_drop_rate: float = 0.1,  # 至少下降 10%
                 threshold_entropy_increase: float = 0.5,  # 允許的微小增加量
                 schema_file_path: str = "schemas.json",
                 *args, **kwargs):
        self.signal_actor = kwargs.pop("signal_actor")
        super().__init__(*args, **kwargs)

        self.samples_generator = self.generator_cls(
            self.vllm_engines,
            self.strategy,
            self.tokenizer,
            self.prompt_max_len,
        )
        self.prepare_datasets()

        self.scm_model_group = scm_model_group
        self.enable_entropy_filtering = enable_entropy_filtering
        self.threshold_final_entropy = threshold_final_entropy
        self.threshold_slow_convergence_steps = threshold_slow_convergence_steps
        self.threshold_drop_rate = threshold_drop_rate
        self.threshold_entropy_increase = threshold_entropy_increase
        self.schema_file_path = schema_file_path
        self._schema_cache = {}
        self._load_schema_cache()  # Load schemas needed for SCM analysis

        if self.enable_entropy_filtering and not self.scm_model_group:
            logger.warning(
                "GenerateSamplesActor: Entropy filtering enabled, but SCM model group not provided. Filtering disabled.")
            self.enable_entropy_filtering = False
        elif self.enable_entropy_filtering:
            n_samples = self.generate_kwargs.get("n_samples_per_prompt", 1)
            if n_samples <= 1:
                logger.warning(
                    f"GenerateSamplesActor: Entropy filtering requires n_samples_per_prompt > 1 (got {n_samples}). Filtering disabled.")
                self.enable_entropy_filtering = False
            else:
                logger.info(
                    "✅ GenerateSamplesActor: SCM Entropy filtering enabled.")

    def generate_samples(self, prompts, labels, **generate_kwargs):
        return self.samples_generator.generate_samples(prompts, labels, **generate_kwargs)

    def fit(self, start_episode, queue, data_loader_state_dict, remote_reward_model=None):
        if data_loader_state_dict:
            self.prompts_dataloader.load_state_dict(data_loader_state_dict)

        for episode in range(start_episode, self.args.num_episodes):
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Generate Episode [{episode + 1}/{self.args.num_episodes}]",
                disable=False,
            )

            filtered_samples = []
            number_of_samples = 0
            queue_log_counter = 0  # Counter for log interval
            for prompt_batch_idx, rand_prompts, labels in self.prompts_dataloader:
                # Wait until queue is not full
                # To support 1-step off-policy training
                while queue.full():
                    if queue_log_counter % 10 == 0:  # Print log every 10 seconds
                        logger.info(
                            "Queue is full, waiting for training to consume samples...")
                    queue_log_counter += 1
                    time.sleep(1)  # Wait for 1 second before checking again

                # Wait for generation to be allowed
                ray.get(self.signal_actor.wait_generating.remote())
                # Block weight updates
                ray.get(self.signal_actor.set_update_weights.remote(False))
                generate_kwargs_for_rollout = self.generate_kwargs.copy()
                if self.enable_entropy_filtering:
                    generate_kwargs_for_rollout["n_samples_per_prompt"] = max(
                        2, generate_kwargs_for_rollout.get("n_samples_per_prompt", 1))
                # Generate samples
                # remote_reward_model is used to get rewards for dynamic sampling
                raw_rollout_samples = self.generate_samples(
                    rand_prompts, labels, remote_reward_model=remote_reward_model, **generate_kwargs_for_rollout
                )
                samples_to_queue = [] # Samples that will be put in the queue
                analysis_results_map = {} # Map sample index to analysis result
                
                if self.enable_entropy_filtering and self.scm_model_group:
                    logger.info(f"Episode {episode+1}, Batch {prompt_batch_idx}: Starting SCM analysis for filtering...")
                    sequences_data = []
                    sample_indices_map = {} # Map seq_data index to raw_rollout_samples index
                    n_samples_per_prompt = generate_kwargs_for_rollout.get("n_samples_per_prompt", 1)

                    for i, sample in enumerate(raw_rollout_samples):
                        prompt_idx = i // n_samples_per_prompt
                        if prompt_idx >= len(rand_prompts): continue

                        question = rand_prompts[prompt_idx]
                        # Need db_id associated with the prompt
                        # Assuming rand_prompts_texts are simple strings, need another source for db_id
                        # Let's assume labels[prompt_idx] or another part of batch_data contains db_id
                        db_id = labels[prompt_idx].get('db_id', None) if isinstance(labels, list) and len(labels) > prompt_idx and isinstance(labels[prompt_idx], dict) else None
                        if not db_id: # Fallback or error if db_id is missing
                             logger.warning(f"SCM Filtering: db_id missing for prompt index {prompt_idx}. Cannot get schema. Sample {i} will be kept.")
                             # Keep the sample if we can't analyze it
                             samples_to_queue.append(sample)
                             analysis_results_map[i] = None # Mark as not analyzed
                             continue

                        schema_str = self._get_schema_string(db_id)
                        if not schema_str: 
                             logger.warning(f"SCM Filtering: Schema missing for db_id {db_id}. Sample {i} will be kept.")
                             samples_to_queue.append(sample)
                             analysis_results_map[i] = None
                             continue
                             
                        full_text = self.tokenizer.decode(sample.sequences[0], skip_special_tokens=True)
                        parsed_schema = parse_schema(schema_str)
                        schema_columns = parsed_schema["columns"]
                        
                        current_data_idx = len(sequences_data)
                        sample_indices_map[current_data_idx] = i
                        sequences_data.append({
                            "question": question, "schema": schema_str, 
                            "full_text": full_text, "schema_columns": schema_columns 
                        })

                    if sequences_data: # Only call SCM if there's data to analyze
                         logger.debug(f"Calling SCM Actor for filtering analysis ({len(sequences_data)} samples)...")
                         analysis_results_list = ray.get(self.scm_model_group[0].forward.remote(sequences_data))
                         logger.debug(f"Received {len(analysis_results_list)} analysis results.")

                         analysis_map_for_batch = {idx: result for idx, result in zip(sample_indices_map.values(), analysis_results_list)}
                         
                         for i, sample in enumerate(raw_rollout_samples):
                              if i in analysis_map_for_batch: # Was it analyzed?
                                   entropies = analysis_map_for_batch[i].get('entropies', [])
                                   passed, reason = self._check_entropy_rules(entropies)
                                   if passed:
                                       samples_to_queue.append(sample)
                                       analysis_results_map[i] = analysis_map_for_batch[i] # Store result for TrainingActor
                                   else:
                                       logger.debug(f"Sample {i} FAILED entropy filtering: {reason}")
                                       analysis_results_map[i] = None # Mark as filtered out
                              elif i not in analysis_results_map: # Was skipped earlier (e.g., no db_id/schema)
                                   samples_to_queue.append(sample) # Keep skipped samples
                                   analysis_results_map[i] = None
                         
                         num_before = len(raw_rollout_samples)
                         num_after = len(samples_to_queue)
                         pass_rate = (num_after / num_before * 100) if num_before > 0 else 0
                         logger.info(f"Entropy filtering pass rate: {pass_rate:.1f}% ({num_after}/{num_before})")
                         
                         if not samples_to_queue:
                              logger.warning("No samples passed filtering in this batch. Skipping queue put.")
                              # Allow weight updates and continue to next batch
                              ray.get(self.signal_actor.set_update_weights.remote(True))
                              pbar.update(1)
                              continue # Skip putting anything in queue
                    else: # No valid data for SCM analysis
                         samples_to_queue = raw_rollout_samples # Use all raw samples
                         analysis_results_map = {i: None for i in range(len(raw_rollout_samples))} # No analysis results
                         pass_rate = 100.0

                else: # Filtering disabled
                    samples_to_queue = raw_rollout_samples
                    analysis_results_map = {i: None for i in range(len(raw_rollout_samples))} # No analysis results
                    pass_rate = 100.0
                # Allow weight updates after generation is done
                output_data = [(sample, analysis_results_map[i]) for i, sample in enumerate(raw_rollout_samples) if i in analysis_results_map and analysis_results_map[i] is not None or sample in samples_to_queue]
                
                # Check if output_data is empty after potential filtering inconsistencies
                if not output_data:
                    logger.warning("Output data is empty after constructing bundle. Skipping queue put.")
                else:
                    # Put tuple: (list_of_tuples(sample, analysis_result), episode, state_dict, pass_rate)
                    queue.put((output_data, episode, self.prompts_dataloader.state_dict(), pass_rate))
                    
                ray.get(self.signal_actor.set_update_weights.remote(True))

                pbar.update()

                # dynamic filtering
                pass_rate = None
                if self.args.dynamic_filtering:
                    number_of_samples += len(rollout_samples)
                    # Group individual samples into batches of n_samples size
                    for i in range(0, len(rollout_samples), self.args.n_samples_per_prompt):
                        batch_samples = rollout_samples[i: i +
                                                        self.args.n_samples_per_prompt]
                        if len(batch_samples) < self.args.n_samples_per_prompt:
                            continue

                        # Calculate average reward for this batch of samples
                        avg_reward = sum(sample.scores[0].item(
                        ) for sample in batch_samples) / len(batch_samples)

                        # Check if average reward is within the specified range
                        min_reward, max_reward = self.args.dynamic_filtering_reward_range
                        if min_reward + 1e-6 < avg_reward < max_reward - 1e-6:
                            filtered_samples.extend(batch_samples)

                    # Continue sampling if filtered samples are insufficient
                    if len(filtered_samples) / self.args.n_samples_per_prompt < self.args.rollout_batch_size:
                        logger.info(
                            f"filtered_samples {len(filtered_samples) / self.args.n_samples_per_prompt} < rollout_batch_size {self.args.rollout_batch_size}, continue sampling"
                        )
                        continue

                    pass_rate = len(filtered_samples) / number_of_samples * 100
                    logger.info(
                        f"Dynamic filtering pass rate: {pass_rate:.2f}% ({len(filtered_samples)}/{number_of_samples})"
                    )
                    rollout_samples = filtered_samples[: self.args.rollout_batch_size *
                                                       self.args.n_samples_per_prompt]
                    filtered_samples = []
                    number_of_samples = 0

                queue.put((rollout_samples, episode,
                          self.prompts_dataloader.state_dict(), pass_rate))
        queue.put("done")

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
                        for obj in loaded:
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

            self._schema_cache = schema_cache
            logger.info(
                f"Loaded {len(self._schema_cache)} schemas from {self.schema_file_path}")
        except Exception as e:
            logger.error(
                f"Failed to load schemas from {self.schema_file_path}: {e}")
            self._schema_cache = {}  # Ensure it's a dict on failure

    def _get_schema_string(self, db_id: str) -> Optional[str]:
        """ Helper function to retrieve schema string from cache. """
        schema = self._schema_cache.get(db_id)
        if schema is None:
            logger.warning(f"Schema not found in cache for db_id: {db_id}")
        return schema

    def _check_entropy_rules(self, entropies: List[float]) -> Tuple[bool, str]:
        """
        Applies the defined entropy filtering rules to a single entropy sequence.
        """
        if len(entropies) < 2:
            return True, "Insufficient entropy data"  # Keep if analysis failed partially

        passed = True
        reason = "Passed"

        # Rule 1: Final Entropy Check
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


@ray.remote
class TrainingActor(BasePPOTrainer):
    def __init__(self, *args, **kwargs):
        self.signal_actor = kwargs.pop("signal_actor")
        self.remote_reward_model = kwargs.pop("remote_reward_model")
        super().__init__(*args, **kwargs)

        if self.kl_target:
            self.kl_ctl = AdaptiveKLController(
                self.init_kl_coef, self.kl_target, self.kl_horizon)
        else:
            self.kl_ctl = FixedKLController(self.init_kl_coef)

        self.experience_maker = RemoteExperienceMaker(
            self.actor_model_group,
            self.critic_model_group,
            self.reward_model_group,
            self.reference_model_group,
            self.kl_ctl,
            self.strategy,
            self.tokenizer,
            remote_reward_model=self.remote_reward_model,
        )

        self._init_wandb()
        self.eval_dataloader = None
        self.empty_step_warn_threshold = max(int(getattr(self.args, "empty_step_warn_threshold", 3)), 1)
        self._empty_step_streak = 0
        self._empty_step_total = 0
        self._empty_step_reasons = defaultdict(int)

    def _record_empty_step(self, reason: str, step: int) -> None:
        self._empty_step_streak += 1
        self._empty_step_total += 1
        self._empty_step_reasons[reason] += 1
        logger.info(
            f"Global step {step}: Empty-step stats => streak={self._empty_step_streak}, "
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
                f"Global step {step}: Consecutive empty steps reached {self._empty_step_streak} "
                f"(threshold={self.empty_step_warn_threshold}). "
                f"Possible over-filtering or upstream sampling issue. reason_breakdown=[{reason_summary}]"
            )

    def _record_non_empty_step(self, step: int) -> None:
        if self._empty_step_streak > 0:
            logger.info(
                f"Global step {step}: Empty-step streak recovered from {self._empty_step_streak} to 0."
            )
        self._empty_step_streak = 0

    def _broadcast_to_vllm(self):
        if self.vllm_engines is not None:
            # Block generation
            ray.get(self.signal_actor.set_generating.remote(False))
            # Wait for weight updates to be allowed
            ray.get(self.signal_actor.wait_update_weights.remote())

            # Perform weight update
            ray.get(self.actor_model_group.async_run_method(
                method_name="broadcast_to_vllm"))

            # Allow generation
            ray.get(self.signal_actor.set_generating.remote(True))

    def fit(self, queue, steps):
        args = self.args

        while True:
            output = queue.get()
            if output == "done":
                break
            rollout_samples, episode, data_loader_state_dict, pass_rate = output

            if not rollout_samples:
                logger.warning(
                    f"Global step {steps}: Received empty rollout_samples from queue. Skipping step.")
                self._record_empty_step("queue_empty_rollout", steps)
                continue

            experiences = self.experience_maker.make_experience_batch(
                rollout_samples)
            if not experiences:
                logger.warning(
                    f"Global step {steps}: make_experience_batch returned empty experiences. Skipping step.")
                self._record_empty_step("empty_experiences", steps)
                continue

            self._record_non_empty_step(steps)

            sample0 = self.tokenizer.batch_decode(
                experiences[0].sequences[0].unsqueeze(0), skip_special_tokens=True)
            print(sample0)

            # balance experiences across dp
            if args.use_dynamic_batch:
                experiences = balance_experiences(experiences, args)

            refs = self.actor_model_group.async_run_method_batch(
                method_name="append", experience=experiences)
            if self.critic_model_group is not None:
                refs.extend(
                    self.critic_model_group.async_run_method_batch(
                        method_name="append", experience=experiences)
                )
            ray.get(refs)

            status = self.ppo_train(steps)

            if "kl" in status:
                self.kl_ctl.update(
                    status["kl"], args.rollout_batch_size * args.n_samples_per_prompt)

            # Add generated samples to status dictionary
            if self.args.dynamic_filtering:
                status["dynamic_filtering_pass_rate"] = pass_rate
            logger.info(f"✨ Global step {steps}: {status}")
            status["generated_samples"] = [
                sample0[0], experiences[0].info["reward"][0]]

            # logs/checkpoints
            client_states = {
                "global_step": steps,
                "episode": episode,
                "data_loader_state_dict": data_loader_state_dict,
            }
            self.save_logs_and_checkpoints(
                args, steps, None, status, client_states)

            steps = steps + 1

        if self._wandb is not None:
            self._wandb.finish()
        if self._tensorboard is not None:
            self._tensorboard.close()


@ray.remote
class PPOTrainerAsync:
    def __init__(
        self,
        pretrain: str,
        strategy: DeepspeedStrategy,
        actor_model_group: RayActorGroup,
        critic_model_group: RayActorGroup,
        reward_model_group: RayActorGroup,
        reference_model_group: RayActorGroup,
        vllm_engines=None,
        prompt_max_len: int = 120,
        dataloader_pin_memory: bool = True,
        prompt_split: str = "train",
        eval_split: str = "test",
        **generate_kwargs,
    ) -> None:
        super().__init__()

        self.args = strategy.args
        self.strategy = strategy
        self.actor_model_group = actor_model_group
        self.critic_model_group = critic_model_group
        self.reward_model_group = reward_model_group
        self.reference_model_group = reference_model_group
        self.vllm_engines = vllm_engines
        self.prompt_max_len = prompt_max_len

        # Create signal actor for synchronization
        self.signal_actor = SignalActor.remote()

        if self.args.remote_rm_url and not self.args.remote_rm_url[0] == "agent":
            from openrlhf.utils.remote_rm_utils import RemoteRewardModel

            self.remote_reward_model = RemoteRewardModel.remote(
                self.args, self.args.remote_rm_url)
        else:
            self.remote_reward_model = None

        self.generator_actor = GenerateSamplesActor.remote(
            pretrain,
            strategy,
            actor_model_group,
            critic_model_group,
            reward_model_group,
            reference_model_group,
            vllm_engines,
            prompt_max_len,
            dataloader_pin_memory,
            prompt_split,
            eval_split,
            signal_actor=self.signal_actor,
            **generate_kwargs,
        )

        # get eval and save steps
        if self.args.eval_steps == -1:
            self.args.eval_steps = float("inf")  # do not evaluate
        if self.args.save_steps == -1:
            self.args.save_steps = float("inf")  # do not save ckpt

        self.trainer_actor = TrainingActor.remote(
            pretrain,
            strategy,
            actor_model_group,
            critic_model_group,
            reward_model_group,
            reference_model_group,
            vllm_engines,
            prompt_max_len,
            dataloader_pin_memory,
            prompt_split,
            eval_split,
            signal_actor=self.signal_actor,
            remote_reward_model=self.remote_reward_model,
            **generate_kwargs,
        )

        from ray.util.queue import Queue

        # the max size is used to control the degree of off-policy
        self.queue = Queue(maxsize=os.environ.get(
            "OPENRLHF_ASYNC_QUEUE_SIZE", 1))

    def fit(self) -> None:
        args = self.args

        # Update initial weights to vLLM engines
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path):
            checkpoint_states = ray.get(self.actor_model_group.async_run_method(method_name="get_checkpoint_states"))[
                0
            ]
            logger.info(f"checkpoint_states: {checkpoint_states}")
            ray.get(self.trainer_actor._broadcast_to_vllm.remote())
        else:
            checkpoint_states = {"global_step": 0,
                                 "episode": 0, "data_loader_state_dict": {}}

        # Restore step and start_epoch
        steps = checkpoint_states["global_step"] + 1
        episode = checkpoint_states["episode"]
        data_loader_state_dict = checkpoint_states["data_loader_state_dict"]

        # Launch async training
        remote_reward_model = self.remote_reward_model if self.args.dynamic_filtering else None
        generator_actor_ref = self.generator_actor.fit.remote(
            episode, self.queue, data_loader_state_dict, remote_reward_model
        )
        trainer_actor_ref = self.trainer_actor.fit.remote(self.queue, steps)
        ray.get([generator_actor_ref, trainer_actor_ref])

    def get_max_steps(self):
        return ray.get(self.generator_actor.get_max_steps.remote())
