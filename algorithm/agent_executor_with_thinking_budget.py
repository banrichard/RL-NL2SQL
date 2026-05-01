import asyncio
import os
from copy import deepcopy

from vllm import SamplingParams

from openrlhf.utils.agent import AgentExecutorBase, SingleTurnAgentExecutor
from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _supports_native_thinking_budget() -> bool:
    try:
        SamplingParams(thinking_token_budget=1)
        return True
    except TypeError:
        return False
    except Exception:
        return False


def _parse_stop_sequences(raw: str):
    return [s.strip() for s in raw.split(",") if s.strip()]


def _extract_output_text(output):
    if isinstance(output, str):
        return output
    if not isinstance(output, dict):
        return ""
    for key in ("query", "response", "output", "text", "completion"):
        value = output.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, dict):
            nested = _extract_output_text(value)
            if nested:
                return nested
    return ""


def _detect_stop_trigger(text: str, stop_sequences):
    if not text:
        return ""
    for stop_seq in stop_sequences:
        if text.endswith(stop_seq):
            return stop_seq
    for stop_seq in stop_sequences:
        if stop_seq in text:
            return stop_seq
    return ""


class AgentExecutor(AgentExecutorBase):
    """
    OpenRLHF custom agent executor with thinking-budget control.

    Modes:
    1) Native vLLM mode (if SamplingParams supports thinking_token_budget)
    2) Fallback mode: cap max_tokens to thinking_budget + answer_reserve
    """

    def __init__(self):
        remote_rm_url = os.environ.get("REMOTE_RM_URL_FOR_AGENT", "").strip() or None
        self.delegate = SingleTurnAgentExecutor(remote_rm_url=remote_rm_url)

        self.thinking_budget = max(int(os.environ.get("VLLM_THINKING_TOKEN_BUDGET", "0")), 0)
        self.answer_token_reserve = max(int(os.environ.get("VLLM_ANSWER_TOKEN_RESERVE", "192")), 1)
        self.native_budget_supported = _supports_native_thinking_budget()
        self.stop_sequences = _parse_stop_sequences(
            os.environ.get("VLLM_STOP_SEQUENCES", "</answer>,<|im_end|>,</s>")
        )

        logger.info(
            f"[thinking-budget] initialized with budget={self.thinking_budget}, "
            f"reserve={self.answer_token_reserve}, native_supported={self.native_budget_supported}, "
            f"stop_sequences={self.stop_sequences}, "
            f"remote_rm_url={'set' if remote_rm_url else 'unset'}"
        )

    async def execute(self, prompt, label, sampling_params, max_length: int, hf_tokenizer, llm_engine):
        effective_params, budget_mode = self._prepare_effective_params(
            sampling_params=sampling_params,
            prompt_token_len=len(hf_tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0]),
            max_length=max_length,
        )

        output = await self.delegate.execute(
            prompt=prompt,
            label=label,
            sampling_params=effective_params,
            max_length=max_length,
            hf_tokenizer=hf_tokenizer,
            llm_engine=llm_engine,
        )

        self._append_budget_logs(output, effective_params, budget_mode)
        return output

    def _prepare_effective_params(self, sampling_params, prompt_token_len: int, max_length: int):
        effective_params = deepcopy(sampling_params)
        existing_stop = list(getattr(effective_params, "stop", None) or [])
        for stop_seq in self.stop_sequences:
            if stop_seq not in existing_stop:
                existing_stop.append(stop_seq)
        effective_params.stop = existing_stop
        available_budget = max(1, int(max_length) - int(prompt_token_len))

        if self.thinking_budget > 0:
            if self.native_budget_supported:
                setattr(effective_params, "thinking_token_budget", self.thinking_budget)
                budget_mode = "native_vllm"
            else:
                hard_cap = min(self.thinking_budget + self.answer_token_reserve, available_budget)
                current_max_tokens = getattr(effective_params, "max_tokens", None)
                if current_max_tokens is None:
                    effective_params.max_tokens = hard_cap
                else:
                    effective_params.max_tokens = max(1, min(int(current_max_tokens), hard_cap))
                budget_mode = "max_tokens_fallback"
        else:
            current_max_tokens = getattr(effective_params, "max_tokens", None)
            if current_max_tokens is None:
                effective_params.max_tokens = available_budget
            else:
                effective_params.max_tokens = max(1, min(int(current_max_tokens), available_budget))
            budget_mode = "disabled"
        return effective_params, budget_mode

    def _append_budget_logs(self, output: dict, effective_params, budget_mode: str):
        generated_text = _extract_output_text(output)
        stop_trigger_sequence = _detect_stop_trigger(generated_text, self.stop_sequences)
        extra_logs = output.get("extra_logs") or {}
        extra_logs["thinking_budget_target"] = float(self.thinking_budget)
        extra_logs["thinking_budget_mode"] = budget_mode
        extra_logs["stop_sequence_count"] = float(len(self.stop_sequences))
        extra_logs["stop_on_answer"] = 1.0 if "</answer>" in self.stop_sequences else 0.0
        extra_logs["stop_triggered_by_sequence"] = 1.0 if stop_trigger_sequence else 0.0
        extra_logs["stop_trigger_sequence"] = stop_trigger_sequence or "none"
        extra_logs["max_generation_budget"] = float(getattr(effective_params, "max_tokens", 0) or 0)
        output["extra_logs"] = extra_logs
        return output

    async def execute_batch(
        self,
        prompt,
        label,
        sampling_params,
        max_length: int,
        hf_tokenizer,
        llm_engine,
        num_samples: int = 1,
    ):
        prompt_token_ids = hf_tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
        effective_params_list = []
        budget_mode_list = []
        for _ in range(num_samples):
            p, mode = self._prepare_effective_params(
                sampling_params=sampling_params,
                prompt_token_len=len(prompt_token_ids),
                max_length=max_length,
            )
            effective_params_list.append(p)
            budget_mode_list.append(mode)

        generate_tasks = [
            llm_engine.generate(prompt_token_ids, deepcopy(effective_params_list[i]))
            for i in range(num_samples)
        ]
        request_outputs = await asyncio.gather(*generate_tasks)

        outputs = []
        queries = []
        for i, request_output in enumerate(request_outputs):
            generation_output = request_output.outputs[0]
            action_token_ids = generation_output.token_ids
            observation_token_ids = prompt_token_ids + action_token_ids
            rollout_log_probs = None
            if sampling_params.logprobs is not None and generation_output.logprobs is not None:
                rollout_log_probs = [0.0] * len(prompt_token_ids)
                for token_id, logprob_dict in zip(action_token_ids, generation_output.logprobs):
                    token_logprob = logprob_dict.get(token_id)
                    rollout_log_probs.append(token_logprob.logprob if token_logprob is not None else 0.0)

            output = {
                "prompt": prompt,
                "label": label,
                "observation_tokens": observation_token_ids,
                "action_ranges": [(len(prompt_token_ids), len(observation_token_ids))],
                "rollout_log_probs": rollout_log_probs,
                "truncated": generation_output.finish_reason == "length",
                "reward": None,
                "scores": None,
                "extra_logs": {},
                "query": hf_tokenizer.decode(observation_token_ids, skip_special_tokens=False),
            }
            self._append_budget_logs(output, effective_params_list[i], budget_mode_list[i])
            queries.append(output["query"])
            outputs.append(output)

        if self.delegate.reward_endpoints:
            if self.delegate.reward_func:
                prompts = [prompt] * len(queries)
                labels = [label] * len(queries)
                reward_result = await asyncio.to_thread(self.delegate.reward_func, queries, prompts, labels)
                rewards = reward_result.get("rewards")
                scores = reward_result.get("scores")
                rewards_list = rewards if isinstance(rewards, list) else [rewards]
                scores_list = scores if isinstance(scores, list) else [scores]
                batch_extra_logs = reward_result.get("extra_logs") or {}
                per_sample_extra_keys = {
                    "clause_scores",
                    "clause_scores_raw",
                    "clause_texts",
                    "clause_names",
                    "extracted_sql",
                }
                shared_extra_logs = {k: v for k, v in batch_extra_logs.items() if k not in per_sample_extra_keys}

                for i, output in enumerate(outputs):
                    if i < len(rewards_list):
                        output["reward"] = rewards_list[i]
                    if i < len(scores_list):
                        output["scores"] = scores_list[i] if scores_list[i] is not None else output.get("reward")
                    elif output.get("reward") is not None:
                        output["scores"] = output["reward"]
                    merged_logs = output.get("extra_logs") or {}
                    merged_logs.update(shared_extra_logs)
                    for key in per_sample_extra_keys:
                        value = batch_extra_logs.get(key)
                        if isinstance(value, list):
                            if i < len(value):
                                merged_logs[key] = value[i]
                        elif value is not None:
                            merged_logs[key] = value
                    output["extra_logs"] = merged_logs
            else:
                reward_tasks = [
                    self.delegate._fetch_rewards_via_http([q], [prompt], [label])
                    for q in queries
                ]
                reward_batches = await asyncio.gather(*reward_tasks)
                reward_results = [rb[0] if rb else None for rb in reward_batches]
                for output, reward_result in zip(outputs, reward_results):
                    if not reward_result:
                        continue
                    output["reward"] = reward_result.get("rewards")
                    output["scores"] = reward_result.get("scores") or reward_result.get("rewards")
                    merged_logs = output.get("extra_logs") or {}
                    merged_logs.update(reward_result.get("extra_logs") or {})
                    output["extra_logs"] = merged_logs

        return outputs
