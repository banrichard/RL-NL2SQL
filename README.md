# RL-NL2SQL

Reinforcement learning framework for training LLMs on the Natural Language to SQL (NL2SQL) task. Built on top of [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), supporting PPO and GRPO with a clause-level Process Reward Model (PRM).

## Overview

The pipeline consists of three stages:

1. **SFT warm-up** — supervised fine-tuning on NL2SQL demonstrations
2. **Reward model training** — train a PRM that scores SQL clauses independently
3. **RL fine-tuning** — PPO or GRPO with the PRM as the reward signal

The reward function decomposes generated SQL into clauses (SELECT, FROM, WHERE, …), scores each clause with the PRM, and aggregates the scores. An execution-based fallback reward is also supported.

## Project Structure

```
algorithm/       # Reward functions, PPO/GRPO agent, credit assignment
dataset/         # Dataset preprocessing and prompt construction
database/        # SQL execution, schema parsing, DB utilities
trainer/         # Custom PPO/SFT/SCM trainers
scripts/         # Training and evaluation shell scripts
templates/       # Prompt templates
configs/         # Dataset and baseline configs
```

## Setup

```bash
pip install openrlhf vllm deepspeed transformers peft sqlglot func_timeout
```

Set environment variables (see `scripts/ppo.sh` for full list):

```bash
export REWARD_MODEL_PATH=/path/to/reward-model
export REWARD_BIRD_TABLES_FILES=/path/to/tables.json
export REWARD_BIRD_DB_ROOTS=/path/to/databases
```

## Training

### 1. SFT

```bash
bash scripts/sft.sh
```

Trains on SynSQL or BIRD data with DeepSpeed ZeRO-2.

### 2. Reward Model

```bash
bash scripts/rm_train.sh
```

### 3. PPO

```bash
bash scripts/ppo.sh
```

Key hyperparameters (override via env):

| Variable | Default | Description |
|---|---|---|
| `PPO_ACTOR_LR` | `1e-6` | Actor learning rate |
| `PPO_INIT_KL_COEF` | `0.01` | Initial KL penalty coefficient |
| `PPO_N_SAMPLES_PER_PROMPT` | `2` | Rollouts per prompt |
| `VLLM_THINKING_TOKEN_BUDGET` | `512` | Max thinking tokens |
| `PPO_ASYNC_TRAIN` | `1` | Enable async rollout/train overlap |

### 4. GRPO (critic-free alternative)

```bash
bash scripts/grpo.sh
```

## Evaluation

Evaluate on BIRD dev set:

```bash
python eval_bird.py --model_path ./ckpt/qwen3.5-9b-ppo \
    --db_root /path/to/bird_dev/dev_databases \
    --tables_file /path/to/dev_tables.json \
    --output_file results.json
```

Batch evaluation scripts are in `scripts/eval_*.sh`.

## Reward Design

`algorithm/reward_func.py` implements the default PRM-based reward:

1. Extract SQL from `<answer>...</answer>` in the model output
2. Parse SQL into ordered clauses with `sqlglot`
3. Score each clause prefix with the reward model
4. Return `mean(clause_scores)` as the final reward

`algorithm/reward_func_token_level.py` provides a token-level variant used by GRPO.

An execution-based reward (exact-match on query results) can be enabled via `REWARD_ENABLE_EXEC=1`.

## Models

Default base model: **Qwen3.5-9B** with Flash Attention 2 and bf16.

Reward model: **Qwen3.5-9B** fine-tuned as a pointwise RM.

## Benchmarks

Evaluated on [BIRD](https://bird-bench.github.io/) and [Spider](https://yale-seas.github.io/spider/).
