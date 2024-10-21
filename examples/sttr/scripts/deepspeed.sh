#!/bin/bash
# export PYTHONPATH=/root/whisper:$PYTHONPATH
#export PYTHONPATH=/root/fairseq:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

# debug setting for multiple gpus
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

run_dir=/public/home/zhxgong/hxdou/STTR/SLAM-LLM
cd $run_dir
code_dir=examples/sttr

speech_encoder_path=/public/home/zhxgong/.cache/whisper/large-v3.pt

llm_path=/public/home/zhxgong/hxdou/STTR/pretrained/gemma2b
# llm_path=/nfs/maziyang.mzy/models/vicuna-13b-v1.5
train_data_path=/public/home/zhxgong/hxdou/STTR/data/sttr-train.jsonl
val_data_path=/public/home/zhxgong/hxdou/STTR/data/sttr-valid.jsonl

output_dir=/public/home/zhxgong/hxdou/STTR/SLAM-LLM/examples/sttr/output/whisper-linear-llama3-$(date +"%Y%m%d")

audio_root=/public/home/zhxgong/hxdou/0-Inbox/Data/en-de/v0
ds_rate=5

hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=gemma2b \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=2048 \
++model_config.encoder_name=whisper \
++model_config.encoder_projector_ds_rate=$ds_rate \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_dim=1280 \
++model_config.encoder_projector=linear \
++dataset_config.dataset=speech_dataset \
++dataset_config.train_data_path=$train_data_path \
++dataset_config.val_data_path=$val_data_path \
++dataset_config.input_type=mel \
++dataset_config.mel_size=128 \
++train_config.model_name=asr \
++train_config.num_epochs=5 \
++train_config.enable_deepspeed=true \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=1000 \
++train_config.total_steps=100000 \
++train_config.lr=1e-4 \
++train_config.validation_interval=1000 \
++train_config.batch_size_training=4 \
++train_config.val_batch_size=4 \
++train_config.num_workers_dataloader=4 \
++train_config.output_dir=$output_dir \
++metric=acc \
++train_config.use_peft=true \
"
# ++train_config.use_peft=true \
# ++train_config.peft_config.r=32 \
# ++model_config.encoder_projector=linear \
# ++model_config.encoder_projector_ds_rate=5 \
# ++train_config.peft_config.peft_method=lora \
# --peft_ckpt "/nfs/maziyang.mzy/exps/llama-2-hf-finetune-asr-ds5-proj2048-lr1e-5-whisper-prompt-padding30-20231228/asr/4" \
# --ckpt_path "/nfs/maziyang.mzy/exps/llama-2-hf-finetune-asr-ds5-proj2048-lr1e-5-whisper-prompt-padding30-20231228/asr/4/model.pt" \
#++log_config.log_file=/$output_dir/train.log \
#++log_config.use_wandb=true \
#++log_config.wandb_dir=$output_dir \
#++log_config.wandb_entity_name=zym22 \
#++log_config.wandb_project_name=slam-llm \
#++log_config.wandb_exp_name=${0##*/%.*} \
#++log_config.log_interval 5 \

deepspeed \
    --include localhost:4,5 \
    --master_port=29502 \
    $code_dir/deepspeed_finetune_asr.py \
    $hydra_args \
     --num_gpus=2 \
     --num_nodes=1