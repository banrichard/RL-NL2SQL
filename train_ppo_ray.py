import os
os.environ['http_proxy']="127.0.0.1:7890"
os.environ['https_proxy']="127.0.0.1:7890"

import argparse
from datetime import datetime

import ray
from ray.util.placement_group import placement_group
from openrlhf.trainer.ray import create_vllm_engines
from openrlhf.trainer.ray.launcher import (
    RayActorGroup,
    ReferenceModelActor,
    RewardModelActor,
)
from openrlhf.trainer.ray.ppo_actor import PolicyModelActor
from openrlhf.trainer.ray.ppo_critic import CriticModelActor
from algorithm.credit_model_actor import CreditModelActor
from openrlhf.utils import get_strategy


def train(args):
    # initialize ray if not initialized
    if not ray.is_initialized():
        ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}})

    # configure strategy
    strategy = get_strategy(args)
    strategy.print(args)

    # init vllm / actor /critic /ref /reward model
    # if colocated, create placement group for actor and ref model explicitly.
    pg = None
    if args.colocate_actor_ref or args.colocate_all_models:
        if args.init_kl_coef > 0:
            assert (
                args.actor_num_nodes == args.ref_num_nodes
                and args.actor_num_gpus_per_node == args.ref_num_gpus_per_node
            ), f"num_nodes and num_gpus_per_node must be the same when colocate actor and ref model."

        bundles = [{"GPU": 1, "CPU": 1} for _ in range(args.actor_num_nodes * args.actor_num_gpus_per_node)]
        pg = placement_group(bundles, strategy="STRICT_PACK")
        ray.get(pg.ready())

    # init vLLM engine for text generation
    vllm_engines = None
    if args.vllm_num_engines is not None and args.vllm_num_engines > 0:
        max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
        if args.colocate_all_models and not args.async_train:
            assert (
                args.actor_num_nodes * args.actor_num_gpus_per_node
                == args.vllm_num_engines * args.vllm_tensor_parallel_size
            ), (
                f"actor_num_nodes * actor_num_gpus_per_node must be equal to "
                f"vllm_num_engines * vllm_tensor_parallel_size, got {args.actor_num_nodes * args.actor_num_gpus_per_node} "
                f"and {args.vllm_num_engines * args.vllm_tensor_parallel_size}"
            )

        if args.agent_func_path:
            from openrlhf.trainer.ray.vllm_engine_async import LLMRayActorAsync as LLMRayActor
        else:
            from openrlhf.trainer.ray.vllm_engine import LLMRayActor

        vllm_engines = create_vllm_engines(
            args.vllm_num_engines,
            args.vllm_tensor_parallel_size,
            args.pretrain,
            args.seed,
            args.full_determinism,
            args.enable_prefix_caching,
            args.enforce_eager,
            max_len,
            pg if args.colocate_all_models and not args.async_train else None,
            args.vllm_gpu_memory_utilization,
            args.vllm_enable_sleep,
            LLMRayActor,
            args.agent_func_path,
        )

    actor_model = RayActorGroup(
        args.actor_num_nodes,
        args.actor_num_gpus_per_node,
        PolicyModelActor,
        pg=pg,
        num_gpus_per_actor=0.2 if pg else 1,
        duplicate_actors=args.ring_attn_size * args.ds_tensor_parallel_size,
    )

    if args.init_kl_coef <= 0:
        ref_model = None
    else:
        ref_model = RayActorGroup(
            args.ref_num_nodes,
            args.ref_num_gpus_per_node,
            ReferenceModelActor,
            pg=pg,
            num_gpus_per_actor=0.2 if pg else 1,
            duplicate_actors=args.ring_attn_size * args.ds_tensor_parallel_size,
        )

    if not args.colocate_all_models:
        pg = None

    # if colocated, create placement group for critic and reward model explicitly.
    if args.critic_pretrain and args.colocate_critic_reward:
        assert (
            args.critic_num_nodes == args.reward_num_nodes
            and args.critic_num_gpus_per_node == args.reward_num_gpus_per_node
        ), f"num_nodes and num_gpus_per_node must be the same when colocate critic and reward model."

        bundles = [{"GPU": 1, "CPU": 1} for _ in range(args.critic_num_nodes * args.critic_num_gpus_per_node)]
        pg = placement_group(bundles, strategy="PACK")
        ray.get(pg.ready())

    if args.critic_pretrain:
        critic_model = RayActorGroup(
            args.critic_num_nodes,
            args.critic_num_gpus_per_node,
            CriticModelActor,
            pg=pg,
            num_gpus_per_actor=0.2 if pg else 1,
            duplicate_actors=args.ring_attn_size * args.ds_tensor_parallel_size,
        )
    else:
        critic_model = None

    # multiple reward models
    if not args.remote_rm_url:
        reward_pretrain = args.reward_pretrain
        reward_model = RayActorGroup(
            args.reward_num_nodes,
            args.reward_num_gpus_per_node,
            RewardModelActor,
            pg=pg,
            num_gpus_per_actor=0.2 if pg else 1,
            duplicate_actors=args.ring_attn_size * args.ds_tensor_parallel_size,
        )
    else:
        reward_model = None

    # Credit Model for fine-grained credit assignment
    credit_model = None
    if args.credit_pretrain:
        credit_model = RayActorGroup(
            args.credit_num_nodes,
            args.credit_num_gpus_per_node,
            CreditModelActor,
            pg=pg,
            num_gpus_per_actor=0.2 if pg else 1,
            duplicate_actors=args.ring_attn_size * args.ds_tensor_parallel_size,
        )

    # Import Credit Model components for simple integration
    if args.credit_pretrain:
        try:
            from algorithm.simple_credit_integration import create_credit_enhanced_ppo_trainer
        except ImportError:
            print("Warning: Could not import credit integration, using standard PPO")
            credit_model = None

    # 子句级别信用分配集成
    clause_credit_enhanced_trainer = None
    if args.credit_pretrain and hasattr(args, 'enable_clause_credit') and args.enable_clause_credit:
        print("Setting up clause-level credit assignment...")
        try:
            from algorithm.clause_credit_integration import ClauseCreditIntegration
            from transformers import AutoTokenizer
            
            # 加载tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(args.pretrain)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                print(f"Loaded tokenizer for clause credit: {args.pretrain}")
            except Exception as e:
                print(f"Warning: Failed to load tokenizer for clause credit: {e}")
                tokenizer = None
            
            if tokenizer:
                # 创建子句级别信用分配集成
                clause_integration = ClauseCreditIntegration(
                    tokenizer=tokenizer,
                    credit_model_path=args.credit_pretrain,
                    discount_factor=getattr(args, 'clause_discount_factor', 0.99),
                    device="cuda"
                )
                
                print("Clause-level credit assignment integration created successfully")
                print(f"Credit model path: {args.credit_pretrain}")
                print(f"Discount factor: {getattr(args, 'clause_discount_factor', 0.99)}")
                
        except Exception as e:
            print(f"Warning: Failed to setup clause-level credit assignment: {e}")
            clause_integration = None
    else:
        clause_integration = None

    # 分离式奖励系统集成
    separated_reward_integration = None
    if args.credit_pretrain and hasattr(args, 'enable_separated_rewards') and args.enable_separated_rewards:
        print("Setting up separated reward system...")
        try:
            from algorithm.separated_reward_system import SeparatedRewardPPOIntegration
            from transformers import AutoTokenizer
            
            # 加载tokenizer（如果之前没有加载）
            if 'tokenizer' not in locals() or tokenizer is None:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(args.pretrain)
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    print(f"Loaded tokenizer for separated rewards: {args.pretrain}")
                except Exception as e:
                    print(f"Warning: Failed to load tokenizer for separated rewards: {e}")
                    tokenizer = None
            
            if tokenizer:
                # 创建分离式奖励系统集成
                separated_reward_integration = SeparatedRewardPPOIntegration(
                    tokenizer=tokenizer,
                    credit_model_path=args.credit_pretrain,
                    discount_factor=getattr(args, 'clause_discount_factor', 0.99),
                    device="cuda"
                )
                
                print("Separated reward system integration created successfully")
                print("Credit Model: 中间步骤即时奖励 (t=0,1,...,T-1)")
                print("Reward Model: 终端步骤完整奖励 (t=T)")
                print("奖励分离: Credit Model和Reward Model永不同时工作")
                
        except Exception as e:
            print(f"Warning: Failed to setup separated reward system: {e}")
            separated_reward_integration = None
    else:
        separated_reward_integration = None

    if args.async_train:
        from openrlhf.trainer.ppo_trainer_async import PPOTrainerAsync as PPOTrainer
    else:
        from openrlhf.trainer.ppo_trainer import PPOTrainer

    # Use simple credit integration if available
    if credit_model is not None and args.credit_pretrain:
        print("Using simple credit integration...")
        try:
            # 设置PPO奖励函数（不执行SQL）
            from algorithm.simple_credit_integration import setup_ppo_reward_function
            setup_ppo_reward_function(
                credit_model_path=args.credit_pretrain,
                use_sql_execution=False,  # PPO阶段不执行SQL
                enable_credit=args.enable_credit_shaping,
                credit_weight=args.credit_weight,
                credit_aggregation=args.credit_aggregation,
                device="cuda"
            )
            
            PPOTrainer = create_credit_enhanced_ppo_trainer(
                base_trainer_class=PPOTrainer,
                credit_model_actor_group=credit_model,
                credit_weight=args.credit_weight,
                enable_credit_shaping=args.enable_credit_shaping,
                credit_aggregation=args.credit_aggregation
            )
            print("Simple credit integration successful")
            print("PPO reward function configured: reward_model + credit_model (no SQL execution)")
        except Exception as e:
            print(f"Warning: Credit integration failed: {e}")
            print("Continuing with standard PPO trainer")
    else:
        print("Using standard PPO trainer")

    # 子句级别信用分配增强（如果启用）
    if clause_integration is not None:
        print("Enhancing PPO trainer with clause-level credit assignment...")
        try:
            # 使用子句级别信用分配增强训练器
            PPOTrainer = clause_integration.enhance_ppo_trainer(PPOTrainer)
            print("✓ PPO trainer enhanced with clause-level credit assignment")
            print("✓ Supports real-time clause boundary detection")
            print("✓ Supports discounted attribution for token rewards")
        except Exception as e:
            print(f"Warning: Failed to enhance PPO trainer with clause credit: {e}")
            print("Continuing with previously configured trainer")

    # 分离式奖励系统增强（如果启用）
    separated_reward_integration = None
    if hasattr(args, 'enable_separated_rewards') and args.enable_separated_rewards:
        print("Enhancing PPO trainer with separated reward system...")
        try:
            from algorithm.separated_reward_system import SeparatedRewardPPOIntegration
            from transformers import AutoTokenizer
            
            # 加载tokenizer（分离式奖励系统需要）
            try:
                tokenizer = AutoTokenizer.from_pretrained(args.pretrain)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
            except Exception as e:
                print(f"Warning: Failed to load tokenizer for separated rewards: {e}")
                tokenizer = None
            
            if tokenizer is not None:
                # 创建分离式奖励系统集成
                separated_reward_integration = SeparatedRewardPPOIntegration(
                    tokenizer=tokenizer,
                    credit_model_path=args.credit_pretrain if args.credit_pretrain else "./ckpt/credit_model",
                    discount_factor=args.clause_discount_factor if hasattr(args, 'clause_discount_factor') else 0.99,
                    device="cuda"
                )
                
                # 增强PPO训练器
                PPOTrainer = separated_reward_integration.enhance_ppo_trainer(PPOTrainer)
                print("✓ PPO trainer enhanced with separated reward system")
                print("✓ Credit Model handles intermediate rewards (t=0,1,...,T-1)")
                print("✓ Reward Model handles terminal rewards (t=T)")
                print("✓ No temporal overlap between Credit and Reward models")
            else:
                print("Warning: Cannot create separated reward integration without tokenizer")
            
        except Exception as e:
            print(f"Warning: Failed to enhance PPO trainer with separated rewards: {e}")
            print("Continuing with previously configured trainer")

    # 创建PPO专用的奖励函数
    ppo_reward_fn = None
    if credit_model is not None and args.credit_pretrain and args.enable_credit_shaping:
        print("Creating PPO reward function with credit model integration...")
        try:
            from algorithm.ppo_reward import create_ppo_reward_wrapper
            from algorithm.reward import reward_func as original_reward_func
            
            ppo_reward_fn = create_ppo_reward_wrapper(
                original_reward_func=original_reward_func,
                enable_credit=args.enable_credit_shaping,
                credit_weight=args.credit_weight,
                credit_aggregation=args.credit_aggregation
            )
            print("PPO reward function created successfully")
        except Exception as e:
            print(f"Warning: Failed to create PPO reward function: {e}")
            ppo_reward_fn = None

    # 集成AdaptiveEntropyProcessor
    adaptive_entropy_config = None
    if hasattr(args, 'enable_adaptive_entropy') and args.enable_adaptive_entropy:
        print("Setting up AdaptiveEntropyProcessor for PPO training...")
        try:
            from algorithm.ppo_entropy_integration import create_ppo_generation_config_with_entropy
            from transformers import AutoTokenizer
            
            # 加载tokenizer用于内容过滤
            try:
                tokenizer = AutoTokenizer.from_pretrained(args.pretrain)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                print(f"Loaded tokenizer for entropy processing: {args.pretrain}")
            except Exception as e:
                print(f"Warning: Failed to load tokenizer for entropy processing: {e}")
                tokenizer = None
            
            # 创建自适应熵生成配置
            adaptive_entropy_config = create_ppo_generation_config_with_entropy(
                tokenizer=tokenizer,
                max_new_tokens=args.generate_max_len,
                temperature=args.temperature,
                top_p=args.top_p,
                low_entropy_threshold=getattr(args, 'low_entropy_threshold', 2.0),
                high_entropy_threshold=getattr(args, 'high_entropy_threshold', 4.0),
                low_k=getattr(args, 'low_k', 5),
                high_k=getattr(args, 'high_k', 50),
                enable_content_filtering=getattr(args, 'enable_content_filtering', True),
                device="cuda",
                use_dynamic_k=getattr(args, 'use_dynamic_k', False),
                k_min=getattr(args, 'k_min', 5),
                k_max=getattr(args, 'k_max', 50),
                sigmoid_weight=getattr(args, 'sigmoid_weight', 1.0),
                entropy_center=getattr(args, 'entropy_center', 3.0)
            )
            print("AdaptiveEntropyProcessor configuration created successfully")
            
        except Exception as e:
            print(f"Warning: Failed to setup AdaptiveEntropyProcessor: {e}")
            adaptive_entropy_config = None

    # init PPO trainer (Single controller)
    trainer_kwargs = {
        "pretrain": args.pretrain,
        "strategy": strategy,
        "actor_model_group": actor_model,
        "critic_model_group": critic_model,
        "reward_model_group": reward_model,
        "reference_model_group": ref_model,
        "vllm_engines": vllm_engines,
        "prompt_split": args.prompt_split,
        "eval_split": args.eval_split,
        # generate kwargs
        "do_sample": True,
        "prompt_max_len": args.prompt_max_len,
        "max_new_tokens": args.generate_max_len,
        "max_length": args.max_len,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    
    # 如果有自定义奖励函数，添加到kwargs中
    if ppo_reward_fn is not None:
        trainer_kwargs["reward_fn"] = ppo_reward_fn
        print("Custom PPO reward function added to trainer")
    
    # 如果有AdaptiveEntropyProcessor配置，添加到trainer中
    if adaptive_entropy_config is not None:
        # 将自适应熵配置的生成参数覆盖到trainer_kwargs
        for key, value in adaptive_entropy_config.items():
            if key not in ['logits_processor']:  # logits_processor需要特殊处理
                trainer_kwargs[key] = value
        
        # 添加logits_processor到生成配置中
        if 'logits_processor' in adaptive_entropy_config:
            trainer_kwargs['logits_processor'] = adaptive_entropy_config['logits_processor']
        
        print("AdaptiveEntropyProcessor integrated into PPO trainer")
        print(f"Entropy thresholds: low={getattr(args, 'low_entropy_threshold', 2.0)}, high={getattr(args, 'high_entropy_threshold', 4.0)}")
        print(f"K values: low={getattr(args, 'low_k', 5)}, high={getattr(args, 'high_k', 50)}")
        print(f"Content filtering: {getattr(args, 'enable_content_filtering', True)}")
    
    # 创建PPO trainer实例
    ppo_trainer = PPOTrainer.remote(**trainer_kwargs)
    # training update steps
    max_steps = ray.get(ppo_trainer.get_max_steps.remote())

    # init reference/reward/actor model
    refs = []
    if ref_model is not None:
        refs.extend(ref_model.async_init_model_from_pretrained(strategy, args.pretrain))
    refs.extend(actor_model.async_init_model_from_pretrained(strategy, args.pretrain, max_steps, vllm_engines))
    if not args.remote_rm_url:
        refs.extend(reward_model.async_init_model_from_pretrained(strategy, reward_pretrain))
    
    # Initialize credit model if available
    if credit_model is not None:
        refs.extend(credit_model.async_init_model_from_pretrained(
            strategy, args.credit_pretrain, args.credit_num_outputs
        ))
    
    ray.get(refs)

    if args.critic_pretrain:
        # critic scheduler initialization depends on max_step, so we have to init critic after actor
        # TODO: use first reward model as critic model
        refs.extend(critic_model.async_init_model_from_pretrained(strategy, args.critic_pretrain, max_steps))
        ray.get(refs)

    # train actor and critic model
    ray.get(ppo_trainer.fit.remote())

    # save model
    ray.get(actor_model.async_save_model())

    if args.critic_pretrain and args.save_value_network:
        ray.get(critic_model.async_save_model())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Ray and vLLM
    parser.add_argument("--ref_num_nodes", type=int, default=1, help="number of nodes for reference")
    parser.add_argument("--ref_num_gpus_per_node", type=int, default=8, help="number of gpus per node for reference")
    parser.add_argument("--reward_num_nodes", type=int, default=1, help="number of nodes for reward model")
    parser.add_argument(
        "--reward_num_gpus_per_node", type=int, default=8, help="number of gpus per node for reward model"
    )
    parser.add_argument(
        "--colocate_actor_ref",
        action="store_true",
        default=False,
        help="whether to colocate reference and actor model, if true, they will share same gpus.",
    )

    parser.add_argument("--actor_num_nodes", type=int, default=1, help="number of nodes for actor")
    parser.add_argument("--actor_num_gpus_per_node", type=int, default=8, help="number of gpus per node for actor")
    parser.add_argument("--critic_num_nodes", type=int, default=1, help="number of nodes for critic")
    parser.add_argument("--critic_num_gpus_per_node", type=int, default=8, help="number of gpus per node for critic")
    parser.add_argument(
        "--colocate_critic_reward",
        action="store_true",
        default=False,
        help="whether to colocate critic and reward model, if true, they will share same gpus.",
    )
    parser.add_argument(
        "--colocate_all_models",
        action="store_true",
        default=False,
        help="whether to colocate all models (including vLLM engines), if true, they will share same gpus.",
    )

    # vLLM for text generation
    parser.add_argument(
        "--vllm_num_engines", type=int, default=None, help="number of vLLM Engines, set to 0 to disable vLLM"
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="tensor parallel size of vLLM Engine for multi-GPU inference",
    )
    parser.add_argument("--vllm_sync_backend", type=str, default="nccl", help="DeepSpeed -> vLLM weight sync backend")
    parser.add_argument("--vllm_sync_with_ray", action="store_true", default=False)
    parser.add_argument("--enable_prefix_caching", action="store_true", default=False)
    parser.add_argument("--enforce_eager", action="store_true", default=False, help="Disable CUDA graph in vLLM")
    parser.add_argument(
        "--vllm_enable_sleep",
        action="store_true",
        default=False,
        help="Enable sleep mode for vLLM when using --colocate_all_models",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.95,
        help="vLLM gpu_memory_utilization",
    )
    # Your Efficient RL Framework Secretly Brings You Off-Policy RL Training: https://fengyao.notion.site/off-policy-rl
    parser.add_argument("--enable_vllm_is_correction", action="store_true", default=False)
    parser.add_argument("--vllm_is_truncated_threshold", type=float, default=2)

    # Async training using ray
    parser.add_argument("--async_train", action="store_true", default=False, help="Enable async training")

    # Checkpoints
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_ppo_ray")
    parser.add_argument("--save_hf_ckpt", action="store_true", default=False)
    parser.add_argument("--disable_ds_ckpt", action="store_true", default=False)
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument(
        "--use_ds_universal_ckpt", action="store_true", help="Use deepspeed universal checkpoint", default=False
    )

    # DeepSpeed
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--deepcompile", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    ## Make EMA as an optional feature
    parser.add_argument("--enable_ema", action="store_true", help="Enable EMA checkpoint for the model.")
    parser.add_argument("--ema_beta", type=float, default=0.992, help="EMA beta coefficient")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--actor_init_on_gpu", action="store_true", default=False)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        help="Attention implementation (e.g., eager, flash_attention_2, flash_attention_3, kernels-community/vllm-flash-attn3)",
    )
    parser.add_argument("--use_liger_kernel", action="store_true", default=False, help="Enable Liger Kernel")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--overlap_comm", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument(
        "--deepspeed_enable_sleep",
        action="store_true",
        default=False,
        help="Enable sleep mode for deepspeed when using --colocate_all_models",
    )
    parser.add_argument("--ds_tensor_parallel_size", type=int, default=1, help="DeepSpeed tensor parallel size")

    # packing samples using Flash Attention2
    parser.add_argument("--packing_samples", action="store_true", default=False)

    # dynamic batch size
    parser.add_argument("--use_dynamic_batch", action="store_true", default=False)
    parser.add_argument("--rollout_max_tokens_per_gpu", type=int, default=None)
    parser.add_argument("--train_max_tokens_per_gpu", type=int, default=16192)

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)

    # PPO
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--rollout_batch_size", type=int, default=1024, help="Batch size for make experience")
    parser.add_argument(
        "--vllm_generate_batch_size", type=int, default=None, help="Batch size for vLLM generating samples"
    )
    parser.add_argument("--micro_rollout_batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--prompt_max_len", type=int, default=1024, help="Max tokens for each prompt")
    parser.add_argument("--generate_max_len", type=int, default=1024, help="Max tokens to generate in PPO")
    parser.add_argument("--max_len", type=int, default=None, help="deprecated max_len")
    parser.add_argument("--max_samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--l2", type=float, default=0.0, help="weight decay loss")
    parser.add_argument("--ptx_coef", type=float, default=0.05, help="PPO-ptx loss coef")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--eps_clip_low_high", type=float, nargs=2, default=None, help="PPO-clip low and high")
    parser.add_argument("--dual_clip", type=float, default=None, help="Dual-clip PPO")
    parser.add_argument("--value_clip", type=float, default=0.5, help="PPO value clip range")
    parser.add_argument("--lambd", type=float, default=1, help="PPO GAE lambd")
    parser.add_argument("--gamma", type=float, default=1, help="PPO GAE gamma")
    parser.add_argument("--micro_train_batch_size", type=int, default=4, help="batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--full_determinism",
        action="store_true",
        default=False,
        help="Enable reproducible behavior during distributed training",
    )
    parser.add_argument("--freezing_actor_steps", type=int, default=-1, help="Used for critic initialization")
    parser.add_argument(
        "--n_samples_per_prompt", type=int, default=1, help="number of responses for each prompt in generation"
    )
    parser.add_argument("--save_value_network", action="store_true", default=False, help="Save critic model")
    parser.add_argument("--actor_learning_rate", type=float, default=1e-6)
    parser.add_argument("--critic_learning_rate", type=float, default=9e-6)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_min_lr")
    parser.add_argument("--kl_target", type=float, default=None)
    parser.add_argument("--kl_horizon", type=int, default=10000)
    parser.add_argument("--init_kl_coef", type=float, default=0.01, help="KL penalty in PPO")
    parser.add_argument("--policy_loss_type", type=str, default="ppo", choices=["ppo", "gspo"])
    parser.add_argument(
        "--kl_estimator",
        type=str,
        default="k1",
        choices=["k1", "k2", "k3"],
        help=(
            "In GRPO, k3 is utilized as the loss function, while k2, when used as the loss, is nearly equivalent to k1."
        ),
    )
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument(
        "--entropy_loss_coef",
        type=float,
        default=None,
        help="Entropy loss coef, set to 0 means only enable entropy logs",
    )
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")
    parser.add_argument("--reward_clip_range", type=float, nargs=2, default=(-10, 10), help="Reward clip range")

    # Reinforce/GRPO, etc
    parser.add_argument(
        "--advantage_estimator",
        type=str,
        choices=["gae", "reinforce", "rloo", "reinforce_baseline", "group_norm", "dr_grpo"],
        default="gae",
        help="Choose advantage estimation method: gae, reinforce, rloo, reinforce_baseline, group_norm, dr_grpo",
    )
    parser.add_argument("--use_kl_loss", action="store_true", default=False, help="whether to use KL loss from GRPO")
    parser.add_argument(
        "--no_advantage_std_norm",
        action="store_true",
        default=False,
        help="disable dividing by std for advantages while keeping mean normalization",
    )
    parser.add_argument(
        "--overlong_buffer_len", type=float, default=None, help="reward with optional overlong penalty"
    )
    parser.add_argument("--overlong_penalty_factor", type=float, default=1, help="overlong penalty factor")

    # Context Parallel
    parser.add_argument("--ring_attn_size", type=int, default=1, help="Ring attention group size")
    parser.add_argument(
        "--ring_head_stride",
        type=int,
        default=1,
        help="the number of heads to do ring attention each time. "
        "It should be a divisor of the number of heads. "
        "A larger value may results in faster training but will consume more memory.",
    )
    # Credit Model parameters
    parser.add_argument("--credit_pretrain", type=str, default=None, help="Credit Model path")
    parser.add_argument("--credit_num_nodes", type=int, default=1, help="number of nodes for credit model")
    parser.add_argument("--credit_num_gpus_per_node", type=int, default=1, help="number of gpus per node for credit model")
    parser.add_argument("--credit_num_outputs", type=int, default=8, help="number of output dimensions for credit model")
    parser.add_argument("--credit_weight", type=float, default=0.1, help="weight for credit-based reward adjustment")
    parser.add_argument("--enable_credit_shaping", action="store_true", default=False, help="enable credit-based reward shaping")
    parser.add_argument("--credit_aggregation", type=str, default="weighted_sum", 
                        choices=["weighted_sum", "max", "mean"], help="method to aggregate credit scores")
    parser.add_argument("--credit_agent_func_path", type=str, default=None, help="Credit Agent script path")
    #  Models
    parser.add_argument("--pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--remote_rm_url", type=str, default=None, help="remote RM API (HTTP)")
    parser.add_argument("--critic_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--value_head_prefix", type=str, default="score")
    parser.add_argument("--ref_reward_offload", action="store_true", default=False)
    parser.add_argument("--agent_func_path", type=str, default=None, help="Agent script path")

    # Custom dataset
    parser.add_argument("--prompt_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument(
        "--prompt_data_probs",
        type=str,
        default=None,
        help="sampling probs for datasets",
    )
    parser.add_argument("--prompt_split", type=str, default="train")
    parser.add_argument("--eval_dataset", type=str, default=None, help="Path to the evaluation dataset")
    parser.add_argument("--eval_split", type=str, default="train")
    parser.add_argument("--eval_temperature", type=float, default=0.6, help="Temperature for evaluation")
    parser.add_argument(
        "--eval_n_samples_per_prompt", type=int, default=4, help="Number of samples per prompt for evaluation"
    )

    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--label_key", type=str, default=None, help="JSON dataset key")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_ppo")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="ppo_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # Dynamic filtering
    parser.add_argument("--dynamic_filtering", action="store_true", default=False, help="Enable dynamic filtering")
    parser.add_argument(
        "--dynamic_filtering_reward_range", nargs=2, default=(0, 1), type=float, help="Dynamic filtering rewards range"
    )

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")

    # performance tuning
    parser.add_argument("--perf", action="store_true", default=False)

    # ModelScope parameters
    parser.add_argument("--use_ms", action="store_true", default=False)

    # AdaptiveEntropyProcessor parameters
    parser.add_argument("--enable_adaptive_entropy", action="store_true", default=False, 
                       help="Enable AdaptiveEntropyProcessor for dynamic exploration/exploitation balance")
    parser.add_argument("--low_entropy_threshold", type=float, default=2.0,
                       help="Entropy threshold below which to use low k-value (confident predictions)")
    parser.add_argument("--high_entropy_threshold", type=float, default=4.0,
                       help="Entropy threshold above which to use high k-value (uncertain predictions)")
    parser.add_argument("--low_k", type=int, default=5,
                       help="Top-K value for low entropy (confident) predictions")
    parser.add_argument("--high_k", type=int, default=50,
                       help="Top-K value for high entropy (uncertain) predictions")
    parser.add_argument("--enable_content_filtering", action="store_true", default=True,
                       help="Enable content filtering to distinguish SQL generation from thinking processes")

    # Clause-level Credit Assignment parameters
    parser.add_argument("--enable_clause_credit", action="store_true", default=False,
                       help="Enable clause-level credit assignment for fine-grained token rewards")
    parser.add_argument("--clause_discount_factor", type=float, default=0.99,
                       help="Discount factor for clause-level credit attribution (gamma)")
    parser.add_argument("--clause_credit_weight", type=float, default=0.1,
                       help="Weight for clause-level credit in the overall reward calculation")

    # Separated Reward System parameters
    parser.add_argument("--enable_separated_rewards", action="store_true", default=False,
                       help="Enable separated reward system where Credit Model handles intermediate rewards and Reward Model handles terminal rewards")

    # Dynamic Entropy and Top-K parameters (for backward compatibility)
    parser.add_argument("--entropy_threshold", type=float, default=2.0, help="Entropy threshold for adaptive sampling")
    parser.add_argument("--use_dynamic_k", action="store_true", default=False, help="Enable dynamic k-value for Top-K sampling")
    parser.add_argument("--k_min", type=int, default=5, help="Minimum k value for dynamic Top-K sampling")
    parser.add_argument("--k_max", type=int, default=50, help="Maximum k value for dynamic Top-K sampling")
    parser.add_argument("--sigmoid_weight", type=float, default=1.0, help="Weight for sigmoid function in dynamic k calculation")
    parser.add_argument("--entropy_center", type=float, default=2.5, help="Center point for entropy-based sigmoid function")
    parser.add_argument("--smooth_threshold", type=float, default=0.1, help="Smooth threshold for entropy processor")

    args = parser.parse_args()

    # Validate arguments
    if args.eps_clip_low_high is None:
        args.eps_clip_low_high = (args.eps_clip, args.eps_clip)

    # Validate mutually exclusive reward systems
    if hasattr(args, 'enable_clause_credit') and hasattr(args, 'enable_separated_rewards'):
        if args.enable_clause_credit and args.enable_separated_rewards:
            raise ValueError(
                "Cannot enable both clause-level credit assignment (--enable_clause_credit) "
                "and separated reward system (--enable_separated_rewards) simultaneously. "
                "These are mutually exclusive approaches:\n"
                "- Clause-level credit: Credit Model and Reward Model work simultaneously\n"
                "- Separated rewards: Credit Model and Reward Model work at different time phases"
            )

    if args.agent_func_path:
        args.remote_rm_url = "agent"

    if args.advantage_estimator not in ["gae"]:
        args.critic_pretrain = None
    elif args.critic_pretrain is None:
        if not args.remote_rm_url:
            args.critic_pretrain = args.reward_pretrain.split(",")[0]
        else:
            args.critic_pretrain = args.pretrain

    if args.advantage_estimator in ["rloo", "reinforce_baseline", "group_norm"]:
        assert args.n_samples_per_prompt > 1, f"{args.advantage_estimator} requires n_samples_per_prompt > 1"

    if args.remote_rm_url:
        args.remote_rm_url = args.remote_rm_url.split(",")

    if args.input_template and "{}" not in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    if args.input_template and "\\n" in args.input_template:
        print(
            "[Warning] input_template contains \\n chracters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )

    if args.ring_attn_size > 1:
        if not args.packing_samples:
            print("[Warning] --ring_attn_size > 1 requires --packing_samples.")
            args.packing_samples = True

    if args.use_dynamic_batch:
        if not args.packing_samples:
            print("[Warning] Please --packing_samples to accelerate when --use_dynamic_batch is enabled.")
            args.packing_samples = True
        if args.rollout_max_tokens_per_gpu is None:
            print("[Warning] Set --rollout_max_tokens_per_gpu to --train_max_tokens_per_gpu.")
            args.rollout_max_tokens_per_gpu = args.train_max_tokens_per_gpu

    if args.packing_samples:
        if "flash_attention" not in args.attn_implementation:
            print(
                "[Warning] Please use --attn_implementation with flash_attention to accelerate when --packing_samples is enabled."
            )
            args.attn_implementation = "flash_attention_2"
        assert args.vllm_num_engines > 0, "Only support `--packing_samples` with vLLM."

    if args.vllm_enable_sleep and not args.colocate_all_models:
        print("Set args.vllm_enable_sleep to False when args.colocate_all_models is disabled.")
        args.vllm_enable_sleep = False

    if args.colocate_all_models and args.async_train:
        print("[Warning] Using --colocate_all_models in async RLHF only colocates DeepSpeed models.")

    if args.async_train:
        assert not args.vllm_enable_sleep, "Async RLHF is not supported with --vllm_enable_sleep."

    if args.eval_dataset:
        assert args.remote_rm_url, "`--eval_dataset` is only supported with `--remote_rm_url`."

    if args.use_kl_loss:
        if args.kl_estimator not in ["k2", "k3"]:
            print(f"Recommend setting {args.kl_estimator} to 'k2' or 'k3' when using KL as a loss")
    else:
        if args.kl_estimator not in ["k1"]:
            print(f"Recommend setting {args.kl_estimator} to 'k1' when not using KL as a loss.")

    # Set vLLM generate_batch_size to rollout_batch_size if not specified
    if not args.vllm_generate_batch_size:
        args.vllm_generate_batch_size = args.rollout_batch_size

    if args.dynamic_filtering:
        assert (
            args.dynamic_filtering_reward_range[0] < args.dynamic_filtering_reward_range[1]
        ), "reward_clip_range[0] must be less than reward_clip_range[1]"
        assert (
            args.remote_rm_url or args.agent_func_path
        ), "remote_rm_url or agent_func_path must be specified when using dynamic filtering"
        assert (
            args.n_samples_per_prompt > 1
        ), "n_samples_per_prompt must be greater than 1 when using dynamic filtering"

    assert (
        args.n_samples_per_prompt * args.rollout_batch_size // args.micro_rollout_batch_size
        >= args.actor_num_nodes * args.actor_num_gpus_per_node // args.ring_attn_size // args.ds_tensor_parallel_size
    ), "The number of sample batches must be greater than or equal to the effective number of actor processes."

    if args.use_ms:
        from modelscope.utils.hf_util import patch_hub

        # Patch hub to download models from modelscope to speed up.
        patch_hub()

    train(args)