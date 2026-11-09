#!/usr/bin/env bash
# verl GRPO — qwen3-4b-lora on xlam (sftgrpo)
# Default four GPUs; global prompt batch and actor mini-batch are unchanged.
set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen3-4B}
DATA=datasets/original/grpo/verl
EXP=${EXP:-grpo_qwen3-4b-lora_xlam_sftgrpo}

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$DATA/xlam_sftgrpo_grpo.parquet \
  data.val_files=$DATA/xlam_val.parquet \
  data.train_batch_size=256 \
  data.max_prompt_length=3072 \
  data.max_response_length=1024 \
  actor_rollout_ref.model.path=$MODEL \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  custom_reward_function.path=scripts/legacy_reward.py \
  custom_reward_function.name=compute_score \
  trainer.logger=console \
  trainer.n_gpus_per_node=${N_GPUS:-4} trainer.nnodes=1 \
  trainer.project_name=tool_calling_training \
  trainer.experiment_name=$EXP \
  trainer.save_freq=50 trainer.test_freq=25 \
  trainer.total_epochs=2 "$@"
