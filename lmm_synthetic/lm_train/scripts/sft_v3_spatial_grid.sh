WD="/home/sjoshi/lmm/lmm_synthetic/lm_train"
RUN_ID=sft_v3_spatial_grid
export WANDB_PROJECT="lm_train"
deepspeed \
    --include localhost:0,1,2,3 \
    --master_port 29501 \
    $WD/sft_lm.py \
    --deepspeed $WD/ds_configs/zero3.json \
    --model_name_or_path /home/sjoshi/lmm/lmm_synthetic/lm_train/checkpoints/v3_spatial_grid_gptj/checkpoint-1953 \
    --dataset_name /home/sjoshi/lmm/lmm_synthetic/data/generated/v3_spatial_grid_multimodal \
    --load_from_disk True \
    --split train \
    --output_dir $WD/checkpoints/$RUN_ID \
    --per_device_train_batch_size 64 \
    --num_train_epochs 5 \
    --learning_rate 1e-5 \
    --weight_decay 0.05 \
    --gradient_accumulation_steps 1 \
    --warmup_steps 100 \
    --lr_scheduler_type cosine \
    --logging_dir $WD/logs \
    --logging_steps 10 \
    --save_strategy epoch \
    --save_total_limit 1 
