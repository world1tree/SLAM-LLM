#!/bin/bash
#export PYTHONPATH=/root/whisper:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1

run_dir=/public/home/zhxgong/hxdou/STTR/SLAM-LLM
cd $run_dir
code_dir=examples/sttr

speech_encoder_path=/public/home/zhxgong/.cache/whisper/large-v3.pt
llm_path=/public/home/zhxgong/hxdou/STTR/pretrained/gemma2b

# output_dir=/public/home/zhxgong/hxdou/STTR/SLAM-LLM/examples/sttr/output/whisper-linear-llama3-20241012/asr_epoch_3_step_111842
# output_dir=/public/home/zhxgong/hxdou/STTR/SLAM-LLM/examples/sttr/output/whisper-linear-llama3-20241012/asr_epoch_2_step_95921
# output_dir=/public/home/zhxgong/hxdou/STTR/SLAM-LLM/examples/sttr/output/whisper-linear-llama3-20241015/asr_epoch_2_step_45921
output_dir=/public/home/zhxgong/hxdou/STTR/SLAM-LLM/examples/sttr/output/whisper-linear-llama3-20241020/asr_epoch_2_step_45921
ckpt_path=$output_dir
split=test
val_data_path=/public/home/zhxgong/hxdou/STTR/data/sttr-${split}.jsonl
decode_log=$ckpt_path/decode_${split}_beam3

audio_root=/public/home/zhxgong/hxdou/0-Inbox/Data/en-de/v0

# -m debugpy --listen 5678 --wait-for-client
python $code_dir/inference_sttr_batch.py \
        --config-path "conf" \
        hydra.run.dir=$ckpt_path \
        ++model_config.llm_name="gemma2b" \
        ++model_config.llm_path=$llm_path \
        ++model_config.llm_dim=2048 \
        ++model_config.encoder_name=whisper \
        ++model_config.encoder_projector_ds_rate=5 \
        ++model_config.encoder_path=$speech_encoder_path \
        ++model_config.encoder_dim=1280 \
        ++model_config.encoder_projector=linear \
        ++dataset_config.dataset=speech_dataset \
        ++dataset_config.val_data_path=$val_data_path \
        ++dataset_config.input_type=mel \
        ++dataset_config.mel_size=128 \
        ++dataset_config.inference_mode=true \
        ++dataset_config.audio_root=$audio_root \
        ++train_config.model_name=asr \
        ++train_config.freeze_encoder=true \
        ++train_config.freeze_llm=true \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=4 \
        ++train_config.num_workers_dataloader=2 \
        ++train_config.output_dir=$output_dir \
        ++decode_log=$decode_log \
        ++ckpt_path=$ckpt_path/model.pt \
        ++log_config.log_file=$output_dir/log.txt \
        ++train_config.use_peft=true
        # ++peft_ckpt=$ckpt_path \
        # ++train_config.peft_config.r=32 \
        # ++dataset_config.normalize=true \
        # ++model_config.encoder_projector=q-former \
        # ++dataset_config.fix_length_audio=64 \
