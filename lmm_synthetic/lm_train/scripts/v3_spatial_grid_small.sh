WD="/home/sjoshi/lmm/lmm_synthetic/lm_train"
RUN_ID=v3_spatial_grid_small
export WANDB_PROJECT="lm_train"
deepspeed \
    --include localhost:0,1,2,3 \
    --master_port 29501 \
    $WD/pretrain_lm.py \
    --deepspeed $WD/ds_configs/zero3.json \
    --vocab_size 50257 \
    --n_positions 256 \
    --n_embd 512 \
    --n_layer 8 \
    --n_head 8 \
    --intermediate_size 2048 \
    --dataset_name /home/sjoshi/lmm/lmm_synthetic/data/generated/v3_spatial_grid \
    --load_from_disk True \
    --split train \
    --max_length 256 \
    --output_dir $WD/checkpoints/$RUN_ID \
    --per_device_train_batch_size 64 \
    --num_train_epochs 10 \
    --learning_rate 2e-3 \
    --weight_decay 0.05 \
    --gradient_accumulation_steps 2 \
    --warmup_steps 100 \
    --lr_scheduler_type cosine \
    --logging_dir $WD/logs \
    --logging_steps 10 \
    --save_strategy epoch \
    --save_total_limit 1