"""
Monkey-patch for NaiveReplayBuffer.setup_dynamic_batch.

The upstream implementation computes ``num_steps`` from fixed args
(rollout_batch_size * n_samples_per_prompt / train_batch_size) which
crashes with IndexError when the replay buffer actually holds fewer
samples (e.g. after entropy / group filtering).

Call ``apply_patch()`` in any process that uses a NaiveReplayBuffer
with dynamic batching.  In train_ppo_ray.py this is done automatically
via Ray's ``worker_process_setup_hook``.

IMPORTANT: This module intentionally has NO module-level imports of
torch / openrlhf.  All heavy imports are deferred to ``apply_patch()``
so the module can be safely imported (and pickled) in Ray's early
worker-setup phase *before* CUDA device isolation is configured.
"""

import logging

logger = logging.getLogger(__name__)


def apply_patch():
    """Replace ``NaiveReplayBuffer.setup_dynamic_batch`` **in the current
    process**.  Safe to call multiple times (idempotent).

    All heavy imports (torch, openrlhf) happen inside this function to
    avoid premature CUDA context creation when used as a Ray
    ``worker_process_setup_hook``.
    """
    # ── lazy imports (must stay inside function body) ───────────────────
    import torch
    from torch import distributed as dist
    from openrlhf.trainer.ppo_utils.replay_buffer import NaiveReplayBuffer
    from openrlhf.utils.seqlen_balancing import (
        get_minimum_num_micro_batch_size,
        get_seqlen_balanced_partitions,
    )

    if getattr(NaiveReplayBuffer, "_safe_dynamic_batch_patched", False):
        return

    # ── patch implementation ───────────────────────────────────────────

    @torch.no_grad()
    def _safe_setup_dynamic_batch(self, strategy):
        """Drop-in replacement that derives *num_steps* from the **actual**
        buffer size instead of the fixed ``args.rollout_batch_size *
        n_samples_per_prompt``.  Prevents ``IndexError`` when entropy /
        group filtering reduces the number of samples below expectation.
        """
        if not self.items:
            logger.warning(
                "setup_dynamic_batch: replay buffer is empty, skipping.")
            self.dynamic_indices = []
            self.dynamic_loss_scale = []
            self.dynamic_optimizer_step = []
            self.sample_batch_size = 1
            return

        args = strategy.args
        sample_lengths = [
            sample.info["total_length"].item() for sample in self.items
        ]

        world_size = dist.get_world_size()
        dp_size = (
            world_size // args.ring_attn_size // args.ds_tensor_parallel_size
        )
        local_train_batch_size = max(args.train_batch_size // dp_size, 1)

        # ── FIX: use actual buffer length ──
        num_steps = max(len(sample_lengths) // local_train_batch_size, 1)

        num_microbatches = []
        for i in range(num_steps):
            start = i * local_train_batch_size
            end = (i + 1) * local_train_batch_size
            chunk = sample_lengths[start:end]
            if not chunk:
                continue
            num_microbatches.append(
                get_minimum_num_micro_batch_size(
                    chunk,
                    args.train_max_tokens_per_gpu,
                    args.ring_attn_size,
                    args.ds_tensor_parallel_size,
                )
            )

        if not num_microbatches:
            logger.warning(
                "setup_dynamic_batch: no micro-batches could be formed, "
                "skipping.")
            self.dynamic_indices = []
            self.dynamic_loss_scale = []
            self.dynamic_optimizer_step = []
            self.sample_batch_size = 1
            return

        num_microbatches = torch.tensor(
            num_microbatches,
            dtype=torch.int,
            device=torch.cuda.current_device(),
        )
        num_microbatches = strategy.all_reduce(num_microbatches, op="max")
        num_microbatches = num_microbatches.tolist()

        micro_batch_indices = []
        data_partitions = []
        for i, num_mbs in enumerate(num_microbatches):
            start = i * local_train_batch_size
            end = (i + 1) * local_train_batch_size
            samples = sample_lengths[start:end]
            if not samples or num_mbs <= 0:
                continue
            partitions = get_seqlen_balanced_partitions(
                samples, num_mbs, equal_size=False,
            )
            for j in range(num_mbs):
                for k in range(len(partitions[j])):
                    partitions[j][k] += start
            micro_batch_indices.extend(partitions)
            data_partitions.append(partitions)

        self.dynamic_indices = micro_batch_indices
        self.sample_batch_size = 1

        loss_scales = []
        optimizer_steps = []
        for partitions in data_partitions:
            sample_num = sum(len(p) for p in partitions)
            if sample_num == 0:
                continue
            loss_scale = [len(p) / sample_num for p in partitions]
            optimizer_step = [0] * (len(partitions) - 1) + [1]
            loss_scales.extend(loss_scale)
            optimizer_steps.extend(optimizer_step)
        self.dynamic_loss_scale = loss_scales
        self.dynamic_optimizer_step = optimizer_steps

    # ── apply ──────────────────────────────────────────────────────────
    NaiveReplayBuffer.setup_dynamic_batch = _safe_setup_dynamic_batch
    NaiveReplayBuffer._safe_dynamic_batch_patched = True
    logger.info(
        "✅ Monkey-patched NaiveReplayBuffer.setup_dynamic_batch "
        "(filtered-buffer safety)."
    )
