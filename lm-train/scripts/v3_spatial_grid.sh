WD="/home/sjoshi/lmm/lm-train"
RUN_ID=v3_spatial_grid
export WANDB_PROJECT="gpt2-pretrain"
deepspeed \
    --include localhost:4,5,6,7 \
    --master_port 29501 \
    $WD/pretrain_gpt2.py \
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
    --max_length 1024 \
    --output_dir $WD/checkpoints/$RUN_ID \
    --per_device_train_batch_size 8 \
    --num_train_epochs 2 \
    --learning_rate 5e-5 \
    --logging_dir $WD/logs \f
    --logging_steps 100 \
    --save_strategy epoch \
    --save_total_limit 1
