#!/bin/bash
# export PYTHONPATH=/root/fairseq:$PYTHONPATH
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
# llm_path=/public/home/zhxgong/mzlv/llama3/8B-Instruct
llm_path=/public/home/zhxgong/hxdou/STTR/pretrained/gemma2b
train_data_path=/public/home/zhxgong/hxdou/STTR/data/sttr-train.jsonl
val_data_path=/public/home/zhxgong/hxdou/STTR/data/sttr-valid.jsonl

output_dir=/public/home/zhxgong/hxdou/STTR/SLAM-LLM/examples/sttr/output/whisper-linear-llama3-$(date +"%Y%m%d")
audio_root=/public/home/zhxgong/hxdou/0-Inbox/Data/en-de/v0

ds_rate=5

hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=gemma2 \
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
++dataset_config.ds_rate=$ds_rate \
++dataset_config.audio_root=$audio_root \
++train_config.model_name=asr \
++train_config.num_epochs=3 \
++train_config.gradient_accumulation_steps=8 \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
++train_config.use_peft=true \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=1000 \
++train_config.total_steps=900000 \
++train_config.lr=1e-4 \
++train_config.validation_interval=10000 \
++train_config.batch_size_training=1 \
++train_config.val_batch_size=2 \
++train_config.num_workers_dataloader=2 \
++train_config.output_dir=$output_dir \
++log_config.log_file=$output_dir/log.txt \
++metric=acc \
"

# -m debugpy --listen 5678 --wait-for-client
if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
    python -m debugpy --listen 5678 --wait-for-client $code_dir/finetune_sttr.py \
        --config-path "conf" \
        $hydra_args
else
    torchrun \
        --nnodes 1 \
        --nproc_per_node 2 \
        --master_port=29503 \
        $code_dir/finetune_sttr.py \
        --config-path "conf" \
        ++train_config.enable_fsdp=false \
        ++train_config.enable_ddp=true \
        ++train_config.use_fp16=true \
        $hydra_args
fi
