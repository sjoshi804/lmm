WD="/home/sjoshi/lmm/lm-train"
RUN_ID=v3_spatial_grid_gptj
export WANDB_PROJECT="gptjpretrain"
deepspeed \
    --include localhost:0,1,2,3,4,5,6,7 \
    --master_port 29501 \
    $WD/pretrain_lm.py \
    --deepspeed $WD/ds_configs/zero3.json \
    --vocab_size 50257 \
    --n_positions 1024 \
    --n_embd 768 \
    --n_layer 12 \
    --n_head 12 \
    --intermediate_size 3072 \
    --dataset_name /home/sjoshi/lmm/data/generated/v3_spatial_grid \
    --load_from_disk True \
    --split train \
    --max_length 768 \
    --output_dir $WD/checkpoints/$RUN_ID \
    --per_device_train_batch_size 16 \
    --num_train_epochs 10 \
    --learning_rate 2e-3 \
    --weight_decay 0.05 \
    --gradient_accumulation_steps 4 \
    --warmup_steps 100 \
    --lr_scheduler_type cosine \
    --logging_dir $WD/logs \
    --logging_steps 100 \
    --save_strategy epoch \
    --save_total_limit 1
